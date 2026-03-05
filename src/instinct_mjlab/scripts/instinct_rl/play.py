"""Play Instinct-RL policies on mjlab tasks."""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import tyro
from instinct_rl.runners import OnPolicyRunner

import instinct_mjlab.tasks  # noqa: F401
import mjlab
from instinct_mjlab.rl import InstinctRlVecEnvWrapper
from instinct_mjlab.utils.motion_validation import (
  validate_tracking_motion_file,
)
from instinct_mjlab.tasks.registry import (
  list_tasks,
  load_env_cfg,
  load_instinct_rl_cfg,
  load_runner_cls,
)
from instinct_mjlab.envs import InstinctRlEnv
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer


@dataclass(frozen=True)
class PlayConfig:
  agent: Literal["zero", "random", "trained"] = "trained"
  motion_file: str | None = None
  registry_name: str | None = None
  checkpoint_file: str | None = None
  load_run: str | None = None
  checkpoint_pattern: str | None = None
  num_envs: int | None = None
  device: str | None = None
  viewer: Literal["auto", "native", "viser", "none"] = "auto"
  max_steps: int | None = None
  video: bool = False
  video_length: int = 400
  video_dir: str | None = None
  video_height: int | None = None
  video_width: int | None = None
  export_onnx: bool = False
  onnx_output_dir: str | None = None
  use_onnx: bool = False
  onnx_model_dir: str | None = None
  no_resume: bool = False
  no_terminations: bool = False


class _ViewerEnvAdapter:
  """Adapter so mjlab viewers can consume instinct_rl vec envs."""

  def __init__(self, vec_env: InstinctRlVecEnvWrapper):
    self._vec_env = vec_env
    self.num_envs = vec_env.num_envs

  @property
  def device(self):
    return self._vec_env.device

  @property
  def cfg(self):
    return self._vec_env.cfg

  @property
  def unwrapped(self):
    return self._vec_env.unwrapped

  def get_observations(self):
    obs, _ = self._vec_env.get_observations()
    return obs

  def step(self, actions):
    return self._vec_env.step(actions)

  def reset(self):
    return self._vec_env.reset()

  def close(self):
    return self._vec_env.close()


def _resolve_device(device: str | None) -> str:
  if device is not None:
    return device
  return "cuda:0" if torch.cuda.is_available() else "cpu"


def _resolve_viewer_backend(viewer: str) -> str:
  if viewer != "auto":
    return viewer
  has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
  return "native" if has_display else "viser"


def _disable_headless_debug_visualization(env_cfg) -> None:
  """Disable GUI-dependent debug visualizations for headless play runs."""
  scene = getattr(env_cfg, "scene", None)
  if scene is not None:
    sensors = getattr(scene, "sensors", ())
    for sensor_cfg in sensors:
      if getattr(sensor_cfg, "debug_vis", None) is not None:
        sensor_cfg.debug_vis = False

  commands = getattr(env_cfg, "commands", None)
  if isinstance(commands, dict):
    for command_cfg in commands.values():
      if getattr(command_cfg, "debug_vis", None) is not None:
        command_cfg.debug_vis = False

  observations = getattr(env_cfg, "observations", None)
  if isinstance(observations, dict):
    for group_cfg in observations.values():
      if group_cfg is None:
        continue
      terms = getattr(group_cfg, "terms", None)
      if not isinstance(terms, dict):
        continue
      for term_cfg in terms.values():
        params = getattr(term_cfg, "params", None)
        if isinstance(params, dict) and "debug_vis" in params:
          params["debug_vis"] = False


def _resolve_rollout_steps(cfg: PlayConfig) -> int | None:
  if cfg.max_steps is not None:
    return cfg.max_steps
  if cfg.viewer == "none":
    return cfg.video_length if cfg.video else 300
  return None


def _force_viewer_realtime_1x(viewer) -> None:
  """Set viewer playback speed to 1x at startup."""
  multipliers = list(getattr(viewer, "SPEED_MULTIPLIERS", ()))
  if len(multipliers) == 0:
    return
  if 1.0 in multipliers:
    speed_index = multipliers.index(1.0)
  else:
    speed_index = min(range(len(multipliers)), key=lambda i: abs(multipliers[i] - 1.0))
  viewer._speed_index = int(speed_index)
  viewer._time_multiplier = float(multipliers[speed_index])


def _resolve_tracking_motion(_task_id: str, cfg: PlayConfig, env_cfg) -> None:
  is_tracking_task = "motion" in env_cfg.commands and isinstance(
    env_cfg.commands["motion"], MotionCommandCfg
  )
  if not is_tracking_task:
    return

  motion_cmd = env_cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)

  if cfg.motion_file is not None:
    motion_path = Path(cfg.motion_file).expanduser().resolve()
    validate_tracking_motion_file(motion_path)
    motion_cmd.motion_file = str(motion_path)
    return

  if cfg.registry_name:
    registry_name = cfg.registry_name
    if ":" not in registry_name:
      registry_name = registry_name + ":latest"
    import wandb

    api = wandb.Api()
    artifact = api.artifact(registry_name)
    motion_path = (Path(artifact.download()) / "motion.npz").resolve()
    validate_tracking_motion_file(motion_path)
    motion_cmd.motion_file = str(motion_path)
    return

  configured_motion = str(getattr(motion_cmd, "motion_file", "")).strip()
  if configured_motion:
    configured_path = Path(configured_motion).expanduser().resolve()
    validate_tracking_motion_file(configured_path)
    motion_cmd.motion_file = str(configured_path)
    print(f"[INFO] Using motion file from env config: {configured_path}")
    return

  raise ValueError(
    "Tracking play requires a motion file.\n"
    "  --motion-file /path/to/motion.npz"
  )


def _build_dummy_policy(agent_mode: str, action_shape: tuple[int, ...], device: str):
  if agent_mode == "zero":

    def zero_policy(_obs: torch.Tensor) -> torch.Tensor:
      return torch.zeros(action_shape, device=device)

    return zero_policy

  def random_policy(_obs: torch.Tensor) -> torch.Tensor:
    return 2.0 * torch.rand(action_shape, device=device) - 1.0

  return random_policy


def _resolve_checkpoint(
  task_id: str,
  cfg: PlayConfig,
  agent_cfg,
) -> Path:
  if cfg.checkpoint_file is not None:
    checkpoint = Path(cfg.checkpoint_file).expanduser().resolve()
    if not checkpoint.exists():
      raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    return checkpoint

  log_root_path = Path("logs") / "instinct_rl" / agent_cfg.experiment_name
  run_regex = cfg.load_run if cfg.load_run not in (None, "") else agent_cfg.load_run
  if run_regex in (None, ""):
    run_regex = ".*"
  checkpoint_regex = (
    cfg.checkpoint_pattern
    if cfg.checkpoint_pattern is not None
    else agent_cfg.load_checkpoint
  )
  if checkpoint_regex in (None, ""):
    checkpoint_regex = "model_.*.pt"

  if not log_root_path.exists():
    raise ValueError(f"Log path does not exist: {log_root_path}")

  candidate_runs: list[Path] = []
  for run in log_root_path.iterdir():
    if not run.is_dir():
      continue
    if run.name == "wandb_checkpoints":
      continue
    # Keep play artifacts out of auto checkpoint lookup by default.
    if run_regex == ".*" and run.name == "_play":
      continue
    if re.match(run_regex, run.name):
      candidate_runs.append(run)

  if len(candidate_runs) == 0:
    raise ValueError(
      f"No run directories found in {log_root_path} matching '{run_regex}'"
    )
  candidate_runs.sort()

  for run_path in reversed(candidate_runs):
    checkpoints = [
      file.name
      for file in run_path.iterdir()
      if file.is_file() and re.match(checkpoint_regex, file.name)
    ]
    if len(checkpoints) == 0:
      continue
    checkpoints.sort(key=lambda name: f"{name:0>15}")
    checkpoint = run_path / checkpoints[-1]
    print(f"[INFO] Auto-selected checkpoint for {task_id}: {checkpoint}")
    return checkpoint

  raise ValueError(
    "No checkpoint file found in matching runs.\n"
    f"  log_path: {log_root_path}\n"
    f"  run_regex: {run_regex}\n"
    f"  checkpoint_regex: {checkpoint_regex}"
  )


def _resolve_video_dir(
  *,
  cfg: PlayConfig,
  task_id: str,
  agent_cfg,
  checkpoint: Path | None,
) -> Path:
  if cfg.video_dir is not None:
    return Path(cfg.video_dir).expanduser().resolve()
  if checkpoint is not None:
    return checkpoint.parent / "videos" / "play"
  return Path("logs") / "instinct_rl" / agent_cfg.experiment_name / task_id / "videos" / "play"


def _resolve_onnx_output_dir(
  *,
  cfg: PlayConfig,
  task_id: str,
  agent_cfg,
  checkpoint: Path | None,
) -> Path:
  if cfg.onnx_output_dir is not None:
    return Path(cfg.onnx_output_dir).expanduser().resolve()
  if checkpoint is not None:
    return checkpoint.parent / "exported"
  return Path("logs") / "instinct_rl" / agent_cfg.experiment_name / task_id / "exported"


def _resolve_onnx_model_dir(
  *,
  cfg: PlayConfig,
  agent_cfg,
  checkpoint: Path | None,
) -> Path:
  if cfg.onnx_model_dir is not None:
    return Path(cfg.onnx_model_dir).expanduser().resolve()
  if checkpoint is not None:
    return checkpoint.parent / "exported"
  if cfg.load_run:
    load_run = Path(cfg.load_run).expanduser()
    if load_run.is_absolute():
      return load_run.resolve() / "exported"
    return Path("logs") / "instinct_rl" / agent_cfg.experiment_name / cfg.load_run / "exported"
  run_name = str(getattr(agent_cfg, "run_name", "") or "")
  return Path("logs") / "instinct_rl" / agent_cfg.experiment_name / f"{run_name}_play" / "exported"


def _build_parkour_onnx_policy(
  *,
  vec_env: InstinctRlVecEnvWrapper,
  agent_cfg,
  model_dir: Path,
):
  from instinct_rl.utils.utils import get_subobs_by_components, get_subobs_size
  from instinct_mjlab.tasks.parkour.scripts.onnxer import load_parkour_onnx_model

  encoder_path = model_dir / "0-depth_encoder.onnx"
  actor_path = model_dir / "actor.onnx"
  missing_files = [str(path) for path in (encoder_path, actor_path) if not path.exists()]
  if missing_files:
    missing_lines = "\n".join(f"  - {path}" for path in missing_files)
    raise FileNotFoundError(f"Missing ONNX model files:\n{missing_lines}")

  obs_segments = vec_env.get_obs_format()["policy"]
  if "depth_image" not in obs_segments:
    raise ValueError("Parkour ONNX inference requires `depth_image` in policy observations.")

  depth_components = agent_cfg.policy.encoder_configs.depth_encoder.component_names

  proprio_components = [
    "base_lin_vel",
    "base_ang_vel",
    "projected_gravity",
    "velocity_commands",
    "joint_pos",
    "joint_vel",
    "actions",
  ]
  return load_parkour_onnx_model(
    model_dir=str(model_dir),
    get_subobs_func=lambda obs: get_subobs_by_components(
      obs,
      depth_components,
      obs_segments,
      temporal=True,
    ),
    depth_shape=obs_segments["depth_image"],
    proprio_slice=slice(
      0,
      get_subobs_size(obs_segments, proprio_components),
    ),
  )


def _export_policy_to_onnx(
  *,
  runner,
  vec_env: InstinctRlVecEnvWrapper,
  task_id: str,
  export_dir: Path,
) -> None:
  export_dir.mkdir(parents=True, exist_ok=True)
  observations, _ = vec_env.get_observations()
  runner.export_as_onnx(obs=observations, export_model_dir=str(export_dir))

  metadata = {
    "task_id": task_id,
    "obs_format": vec_env.get_obs_format(),
    "num_actions": vec_env.num_actions,
    "num_rewards": vec_env.num_rewards,
  }
  metadata_path = export_dir / "metadata.json"
  metadata_path.write_text(
    json.dumps(metadata, ensure_ascii=False, indent=2),
    encoding="utf-8",
  )
  print(f"[INFO] Exported ONNX artifacts to: {export_dir}")


def _run_headless_rollout(
  env: _ViewerEnvAdapter,
  policy,
  *,
  num_steps: int | None,
) -> None:
  steps = num_steps if num_steps is not None else 300
  observations = env.get_observations()
  for _ in range(steps):
    with torch.no_grad():
      actions = policy(observations)
      observations, _, _, _ = env.step(actions)


def run_play(task_id: str, cfg: PlayConfig) -> None:
  if InstinctRlVecEnvWrapper is None:
    raise ImportError(
      "InstinctRlVecEnvWrapper is unavailable. Please install runtime deps:\n"
      "  pip install -e ../mjlab\n"
      "  pip install -e ../instinct_rl"
    )
  configure_torch_backends()
  viewer_backend = _resolve_viewer_backend(cfg.viewer)
  # Native viewer should use glfw; headless/video paths should use egl.
  if viewer_backend == "native":
    os.environ["MUJOCO_GL"] = "glfw"
  else:
    os.environ.setdefault("MUJOCO_GL", "egl")

  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_instinct_rl_cfg(task_id)
  device = _resolve_device(cfg.device)
  checkpoint: Path | None = None

  has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
  if viewer_backend != "native" or not has_display:
    _disable_headless_debug_visualization(env_cfg)

  if cfg.num_envs is not None:
    env_cfg.scene.num_envs = cfg.num_envs
  if cfg.video_height is not None:
    env_cfg.viewer.height = cfg.video_height
  if cfg.video_width is not None:
    env_cfg.viewer.width = cfg.video_width
  if cfg.no_terminations:
    env_cfg.terminations = {}
    print("[INFO] All terminations are disabled for play.")

  _resolve_tracking_motion(task_id, cfg, env_cfg)
  if cfg.use_onnx and "Parkour" not in task_id:
    raise ValueError("`--use-onnx` currently only supports parkour tasks.")
  if cfg.use_onnx and cfg.agent != "trained":
    raise ValueError("`--use-onnx` only supports `--agent trained`.")

  if cfg.agent == "trained":
    if cfg.no_resume:
      if cfg.export_onnx:
        raise ValueError("`--no-resume` cannot be combined with `--export-onnx`.")
      if not cfg.use_onnx:
        raise ValueError(
          "`--no-resume` with `--agent trained` requires `--use-onnx True`."
        )
      print("[INFO] No-resume mode is enabled. Skip checkpoint loading and use ONNX policy.")
    else:
      checkpoint = _resolve_checkpoint(task_id, cfg, agent_cfg)
  elif cfg.export_onnx:
    raise ValueError("`--export-onnx` only supports `--agent trained`.")

  env = InstinctRlEnv(
    cfg=env_cfg,
    device=device,
    render_mode="rgb_array" if cfg.video else None,
  )
  if cfg.video:
    video_dir = _resolve_video_dir(
      cfg=cfg,
      task_id=task_id,
      agent_cfg=agent_cfg,
      checkpoint=checkpoint,
    )
    env = VideoRecorder(
      env,
      video_folder=video_dir,
      step_trigger=lambda step: step == 0,
      video_length=cfg.video_length,
      disable_logger=True,
      name_prefix=task_id.replace("/", "_"),
    )
    print(f"[INFO] Recording play video to: {video_dir}")

  vec_env = InstinctRlVecEnvWrapper(
    env,
    policy_group=agent_cfg.policy_observation_group,
    critic_group=agent_cfg.critic_observation_group,
  )

  viewer_env = _ViewerEnvAdapter(vec_env)

  if cfg.agent in {"zero", "random"}:
    policy = _build_dummy_policy(cfg.agent, (vec_env.num_envs, vec_env.num_actions), device)
  else:
    runner = None
    if checkpoint is not None or cfg.export_onnx:
      runner_cls = load_runner_cls(task_id) or OnPolicyRunner
      agent_cfg_dict = agent_cfg.to_dict()
      runner = runner_cls(
        vec_env,
        agent_cfg_dict,
        log_dir=None,
        device=device,
      )

    if checkpoint is not None:
      if runner is None:
        raise RuntimeError("Runner is not initialized.")
      runner.load(str(checkpoint))
      print(f"[INFO] Loaded checkpoint: {checkpoint}")

    if cfg.export_onnx:
      if checkpoint is None:
        raise RuntimeError("checkpoint must be provided when export_onnx is enabled.")
      if runner is None:
        raise RuntimeError("Runner is not initialized.")
      onnx_output_dir = _resolve_onnx_output_dir(
        cfg=cfg,
        task_id=task_id,
        agent_cfg=agent_cfg,
        checkpoint=checkpoint,
      )
      _export_policy_to_onnx(
        runner=runner,
        vec_env=vec_env,
        task_id=task_id,
        export_dir=onnx_output_dir,
      )

    if cfg.use_onnx:
      onnx_model_dir = _resolve_onnx_model_dir(
        cfg=cfg,
        agent_cfg=agent_cfg,
        checkpoint=checkpoint,
      )
      policy = _build_parkour_onnx_policy(
        vec_env=vec_env,
        agent_cfg=agent_cfg,
        model_dir=onnx_model_dir,
      )
      print(f"[INFO] Loaded ONNX policy from: {onnx_model_dir}")
    else:
      if runner is None:
        raise RuntimeError("Runner is not initialized.")
      policy = runner.get_inference_policy(device=device)

  rollout_steps = _resolve_rollout_steps(cfg)

  if viewer_backend == "native":
    viewer = NativeMujocoViewer(viewer_env, policy)
    _force_viewer_realtime_1x(viewer)
    viewer.run(num_steps=rollout_steps)
  elif viewer_backend == "viser":
    viewer = ViserPlayViewer(viewer_env, policy)
    _force_viewer_realtime_1x(viewer)
    viewer.run(num_steps=rollout_steps)
  elif viewer_backend == "none":
    _run_headless_rollout(
      viewer_env,
      policy,
      num_steps=rollout_steps,
    )
  else:
    raise RuntimeError(f"Unsupported viewer backend: {viewer_backend}")

  viewer_env.close()


def main() -> None:
  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )

  args = tyro.cli(
    PlayConfig,
    args=remaining_args,
    default=PlayConfig(),
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  run_play(chosen_task, args)


if __name__ == "__main__":
  main()
