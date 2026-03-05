"""Train Instinct-RL policies on top of mjlab environments."""

from __future__ import annotations

import logging
import os
import re
import signal
import sys
from dataclasses import dataclass, fields, is_dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import torch
import torch.distributed as dist
import tyro
from instinct_rl.runners import OnPolicyRunner

import instinct_mjlab.tasks  # noqa: F401
import mjlab
from instinct_mjlab.rl import InstinctRlOnPolicyRunnerCfg, InstinctRlVecEnvWrapper
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
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.gpu import select_gpus
from mjlab.utils.os import dump_yaml, get_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer


def _to_yaml_data(data: Any) -> Any:
  if is_dataclass(data):
    return {item.name: _to_yaml_data(getattr(data, item.name)) for item in fields(data)}
  if isinstance(data, dict):
    return {str(key): _to_yaml_data(value) for key, value in data.items()}
  if isinstance(data, tuple):
    return [_to_yaml_data(value) for value in data]
  if isinstance(data, list):
    return [_to_yaml_data(value) for value in data]
  if isinstance(data, (str, int, float, bool)) or data is None:
    return data
  return repr(data)


@dataclass(frozen=True)
class TrainConfig:
  env: ManagerBasedRlEnvCfg
  agent: InstinctRlOnPolicyRunnerCfg
  motion_file: str | None = None
  registry_name: str | None = None
  num_envs: int | None = None
  device: str | None = None
  video: bool = False
  video_length: int = 200
  video_interval: int = 2_000
  viewer: Literal["none", "native"] = "none"
  viewer_fps: float = 60.0
  gpu_ids: list[int] | Literal["all"] | None = None
  torchrunx_log_dir: str | None = None

  @staticmethod
  def from_task(task_id: str) -> "TrainConfig":
    use_play_cfg = task_id.endswith("-Play-v0")
    return TrainConfig(
      env=load_env_cfg(task_id, play=use_play_cfg),
      agent=load_instinct_rl_cfg(task_id),
    )


@dataclass
class TrainCliConfig:
  motion_file: str | None = None
  registry_name: str | None = None
  num_envs: int | None = None
  device: str | None = None
  video: bool = False
  video_length: int = 200
  video_interval: int = 2_000
  viewer: Literal["none", "native"] = "none"
  viewer_fps: float = 60.0
  gpu_ids: list[int] | Literal["all"] | None = None
  torchrunx_log_dir: str | None = None


def _parse_cli_literal(raw: str) -> Any:
  lower = raw.lower()
  if lower == "none":
    return None
  if lower == "true":
    return True
  if lower == "false":
    return False
  if re.fullmatch(r"[+-]?\d+", raw):
    return int(raw)
  if re.fullmatch(r"[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?", raw):
    return float(raw)
  if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in {'"', "'"}:
    return raw[1:-1]
  if raw.startswith("[") and raw.endswith("]"):
    inner = raw[1:-1].strip()
    if not inner:
      return []
    return [_parse_cli_literal(item.strip()) for item in inner.split(",") if item.strip()]
  if raw.startswith("(") and raw.endswith(")"):
    inner = raw[1:-1].strip()
    if not inner:
      return ()
    return tuple(_parse_cli_literal(item.strip()) for item in inner.split(",") if item.strip())
  return raw


def _iter_dot_overrides(tokens: list[str]) -> list[tuple[str, Any]]:
  overrides: list[tuple[str, Any]] = []
  i = 0
  while i < len(tokens):
    token = tokens[i]
    if not token.startswith("--"):
      raise ValueError(f"Unexpected argument '{token}'. Dot-overrides must start with '--'.")

    flag = token[2:]
    if "=" in flag:
      key, raw_value = flag.split("=", 1)
      overrides.append((key, _parse_cli_literal(raw_value)))
      i += 1
      continue

    if i + 1 >= len(tokens) or tokens[i + 1].startswith("--"):
      overrides.append((flag, True))
      i += 1
      continue

    overrides.append((flag, _parse_cli_literal(tokens[i + 1])))
    i += 2
  return overrides


def _coerce_override_value(value: Any, current: Any) -> Any:
  if isinstance(current, tuple) and isinstance(value, list):
    return tuple(value)
  if isinstance(current, list) and isinstance(value, tuple):
    return list(value)
  if isinstance(current, float) and isinstance(value, int):
    return float(value)
  return value


def _set_nested_attr(target: Any, path: str, value: Any) -> None:
  parts = path.split(".")
  current = target
  for part in parts[:-1]:
    if isinstance(current, dict):
      if part not in current:
        raise ValueError(f"Unknown dict key '{part}' while applying override '{path}'.")
      current = current[part]
    else:
      if is_dataclass(current):
        if part not in current.__dataclass_fields__:
          raise ValueError(f"Unknown attribute '{part}' while applying override '{path}'.")
      elif part not in vars(current):
        raise ValueError(f"Unknown attribute '{part}' while applying override '{path}'.")
      current = getattr(current, part)

  last = parts[-1]
  if isinstance(current, dict):
    if last not in current:
      raise ValueError(f"Unknown dict key '{last}' while applying override '{path}'.")
    current[last] = _coerce_override_value(value, current[last])
    return

  if is_dataclass(current):
    if last not in current.__dataclass_fields__:
      raise ValueError(f"Unknown attribute '{last}' while applying override '{path}'.")
  elif last not in vars(current):
    raise ValueError(f"Unknown attribute '{last}' while applying override '{path}'.")
  existing = getattr(current, last)
  setattr(current, last, _coerce_override_value(value, existing))


def _apply_dot_overrides(cfg: TrainConfig, raw_args: list[str]) -> None:
  for key, value in _iter_dot_overrides(raw_args):
    if key.startswith("agent."):
      _set_nested_attr(cfg.agent, key[len("agent."):], value)
    elif key.startswith("env."):
      _set_nested_attr(cfg.env, key[len("env."):], value)
    else:
      raise ValueError(
        f"Unsupported override '{key}'. Use top-level flags or '--agent.*' / '--env.*'."
      )


def _resolve_tracking_motion(_task_id: str, cfg: TrainConfig) -> str | None:
  is_tracking_task = "motion" in cfg.env.commands and isinstance(
    cfg.env.commands["motion"], MotionCommandCfg
  )
  if not is_tracking_task:
    return None

  motion_cmd = cfg.env.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)

  if cfg.motion_file is not None:
    motion_path = Path(cfg.motion_file).expanduser().resolve()
    validate_tracking_motion_file(motion_path)
    motion_cmd.motion_file = str(motion_path)
    return None

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
    return registry_name

  configured_motion = str(getattr(motion_cmd, "motion_file", "")).strip()
  if configured_motion:
    configured_path = Path(configured_motion).expanduser().resolve()
    validate_tracking_motion_file(configured_path)
    motion_cmd.motion_file = str(configured_path)
    print(f"[INFO] Using motion file from env config: {configured_path}")
    return None

  raise ValueError(
    "Tracking training requires a motion file.\n"
    "  --motion-file /path/to/motion.npz"
  )


def _resolve_device(cfg: TrainConfig) -> str:
  if cfg.device is not None:
    return cfg.device
  return "cuda:0" if torch.cuda.is_available() else "cpu"


def _parse_cuda_device_index(device: str) -> int:
  if not device.startswith("cuda"):
    return 0
  if ":" not in device:
    return 0
  try:
    return int(device.split(":", 1)[1])
  except ValueError:
    return 0


def _resolve_distributed_runtime(
  cfg: TrainConfig,
) -> tuple[str, int, int, int, bool]:
  local_rank = int(os.environ.get("LOCAL_RANK", "0"))
  rank = int(os.environ.get("RANK", "0"))
  world_size = int(os.environ.get("WORLD_SIZE", "1"))
  is_distributed = world_size > 1
  if is_distributed:
    device = f"cuda:{local_rank}"
    seed = cfg.agent.seed + local_rank
  else:
    if cfg.gpu_ids is not None and os.environ.get("CUDA_VISIBLE_DEVICES", "") != "":
      device = "cuda:0"
    else:
      device = _resolve_device(cfg)
    seed = cfg.agent.seed
  return device, seed, rank, world_size, is_distributed


def run_train(task_id: str, cfg: TrainConfig, log_dir: Path) -> None:
  log_dir = log_dir.expanduser().resolve()

  if InstinctRlVecEnvWrapper is None:
    raise ImportError(
      "InstinctRlVecEnvWrapper is unavailable. Please install runtime deps:\n"
      "  pip install -e ../mjlab\n"
      "  pip install -e ../instinct_rl"
    )
  device, seed, rank, world_size, is_distributed = _resolve_distributed_runtime(cfg)

  if device.startswith("cuda"):
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(_parse_cuda_device_index(device))

  viewer_enabled = cfg.viewer == "native" and rank == 0
  video_enabled = cfg.video and rank == 0
  if viewer_enabled:
    os.environ["MUJOCO_GL"] = "glfw"
  else:
    os.environ.setdefault("MUJOCO_GL", "egl")
  if viewer_enabled:
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    if not has_display:
      raise RuntimeError(
        "Native viewer requires DISPLAY/WAYLAND_DISPLAY. "
        "Unset --viewer or use an X/Wayland session."
      )
  configure_torch_backends()

  if is_distributed and not dist.is_initialized():
    dist.init_process_group(
      backend="nccl",
      rank=rank,
      world_size=world_size,
    )

  if device.startswith("cpu"):
    raise ValueError(
      "The current instinct_rl training pipeline requires CUDA runtime stats. "
      "Use a GPU device, e.g. `--device cuda:0`."
    )
  cfg.agent.device = device
  cfg.agent.seed = seed
  cfg.env.seed = seed
  if cfg.num_envs is not None:
    cfg.env.scene.num_envs = cfg.num_envs

  registry_name = _resolve_tracking_motion(task_id, cfg)

  print(
    f"[INFO] Task={task_id}, device={device}, seed={seed}, "
    f"num_envs={cfg.env.scene.num_envs}, rank={rank}/{world_size}"
  )
  if rank == 0:
    print(f"[INFO] Logging to: {log_dir}")

  env = InstinctRlEnv(
    cfg=cfg.env,
    device=device,
    render_mode="rgb_array" if video_enabled else None,
  )

  if video_enabled:
    env = VideoRecorder(
      env,
      video_folder=log_dir / "videos" / "train",
      step_trigger=lambda step: step % cfg.video_interval == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )
    print("[INFO] Recording videos during training.")

  vec_env = InstinctRlVecEnvWrapper(
    env,
    policy_group=cfg.agent.policy_observation_group,
    critic_group=cfg.agent.critic_observation_group,
  )

  train_viewer = None
  if viewer_enabled:
    def _viewer_policy(_obs: torch.Tensor) -> torch.Tensor:
      return torch.zeros(
        (vec_env.num_envs, vec_env.num_actions),
        device=vec_env.device,
      )

    train_viewer = NativeMujocoViewer(
      vec_env,
      policy=_viewer_policy,
      frame_rate=cfg.viewer_fps,
    )
    train_viewer.setup()
    train_viewer.sync_env_to_viewer()
    print("[INFO] Native viewer enabled during training.")

  runner_cls = load_runner_cls(task_id) or OnPolicyRunner
  agent_cfg_dict = cfg.agent.to_dict()

  runner = runner_cls(
    vec_env,
    agent_cfg_dict,
    log_dir=str(log_dir),
    device=cfg.agent.device,
  )
  runner.add_git_repo_to_log(__file__)

  if cfg.agent.resume:
    log_root_path = log_dir.parent
    load_run = Path(cfg.agent.load_run).expanduser() if cfg.agent.load_run else None
    if load_run is not None and load_run.is_absolute():
      resume_path = get_checkpoint_path(
        log_path=load_run.parent,
        run_dir=load_run.name,
        checkpoint=cfg.agent.load_checkpoint,
      )
    else:
      resume_path = get_checkpoint_path(
        log_path=log_root_path,
        run_dir=cfg.agent.load_run,
        checkpoint=cfg.agent.load_checkpoint,
      )
    print(f"[INFO] Resuming from checkpoint: {resume_path}")
    runner.load(str(resume_path))

  if rank == 0:
    dump_yaml(log_dir / "params" / "env.yaml", _to_yaml_data(cfg.env))
    dump_yaml(log_dir / "params" / "agent.yaml", _to_yaml_data(cfg.agent))
    if registry_name is not None:
      dump_yaml(
        log_dir / "params" / "registry.yaml",
        {"registry_name": registry_name},
      )
  if train_viewer is not None:
    runner_rollout_step = runner.rollout_step

    def _rollout_step_with_view(obs, critic_obs):
      result = runner_rollout_step(obs, critic_obs)
      if train_viewer.is_running():
        train_viewer.sync_viewer_to_env()
        train_viewer.sync_env_to_viewer()
      return result

    runner.rollout_step = _rollout_step_with_view

  handled_signal_name: str | None = None
  signal_handlers_to_restore: dict[int, Any] = {}

  def _interrupt_handler(signum, _frame):
    nonlocal handled_signal_name
    try:
      handled_signal_name = signal.Signals(signum).name
    except ValueError:
      handled_signal_name = str(signum)
    raise KeyboardInterrupt

  install_signal_numbers: list[int] = [signal.SIGINT, signal.SIGTERM]
  install_signal_numbers.append(signal.SIGQUIT)
  for signum in install_signal_numbers:
    signal_handlers_to_restore[signum] = signal.getsignal(signum)
    signal.signal(signum, _interrupt_handler)

  try:
    runner.learn(
      num_learning_iterations=cfg.agent.max_iterations,
      init_at_random_ep_len=True,
    )
  except KeyboardInterrupt:
    interrupt_name = handled_signal_name or "KeyboardInterrupt"
    print(f"[WARN] Training interrupted by {interrupt_name}.")
  finally:
    for signum, previous_handler in signal_handlers_to_restore.items():
      signal.signal(signum, previous_handler)
    if train_viewer is not None:
      train_viewer.close()
    vec_env.close()
    if dist.is_initialized():
      dist.destroy_process_group()


def launch_training(task_id: str, args: TrainConfig | None = None) -> None:
  args = args or TrainConfig.from_task(task_id)
  log_root_path = Path("logs") / "instinct_rl" / args.agent.experiment_name
  log_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  if args.agent.run_name:
    log_dir_name += f"_{args.agent.run_name}"
  log_dir = log_root_path / log_dir_name

  if args.gpu_ids is None:
    run_train(task_id=task_id, cfg=args, log_dir=log_dir)
    return

  selected_gpus, num_gpus = select_gpus(args.gpu_ids)
  if selected_gpus is None or num_gpus == 0:
    raise ValueError(
      "No CUDA GPUs were selected. "
      "Provide valid --gpu-ids (e.g. --gpu-ids all or --gpu-ids [0,1])."
    )

  os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu_id) for gpu_id in selected_gpus)

  if num_gpus <= 1:
    run_train(task_id=task_id, cfg=args, log_dir=log_dir)
    return

  import torchrunx

  logging.basicConfig(level=logging.INFO)
  if "TORCHRUNX_LOG_DIR" not in os.environ:
    if args.torchrunx_log_dir is not None:
      os.environ["TORCHRUNX_LOG_DIR"] = args.torchrunx_log_dir
    else:
      os.environ["TORCHRUNX_LOG_DIR"] = str(log_dir / "torchrunx")

  print(f"[INFO] Launching multi-GPU training with {num_gpus} GPUs.")
  torchrunx.Launcher(
    hostnames=["localhost"],
    workers_per_host=num_gpus,
    backend=None,
    copy_env_vars=torchrunx.DEFAULT_ENV_VARS_FOR_COPY + ("MUJOCO*",),
  ).run(run_train, task_id, args, log_dir)


def main() -> None:
  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )

  cli_cfg, dot_override_args = tyro.cli(
    TrainCliConfig,
    args=remaining_args,
    default=TrainCliConfig(),
    return_unknown_args=True,
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )

  args = replace(
    TrainConfig.from_task(chosen_task),
    motion_file=cli_cfg.motion_file,
    registry_name=cli_cfg.registry_name,
    num_envs=cli_cfg.num_envs,
    device=cli_cfg.device,
    video=cli_cfg.video,
    video_length=cli_cfg.video_length,
    video_interval=cli_cfg.video_interval,
    viewer=cli_cfg.viewer,
    viewer_fps=cli_cfg.viewer_fps,
    gpu_ids=cli_cfg.gpu_ids,
    torchrunx_log_dir=cli_cfg.torchrunx_log_dir,
  )
  _apply_dot_overrides(args, dot_override_args)
  launch_training(task_id=chosen_task, args=args)


if __name__ == "__main__":
  main()
