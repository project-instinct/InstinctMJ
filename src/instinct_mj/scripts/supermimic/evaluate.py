"""Evaluate ``OnPolicyRunner``-format checkpoints (``instinct-train`` runs of the task, ...) on the Play setting of the task.

    instinct-supermimic-eval --checkpoint <run>/model_5000.pt --motion-folder ~/Datasets/20251116_50cm_kneeClimbStep1
    instinct-supermimic-eval --run-dir <run> --every 5 --watch --motion-folder ...

Setting: the task's Play env cfg (no domain randomization, no pushes, every episode starts at motion t=0, ``independent``
sampling), the same ``--motion-folder`` override and camera noise (part of the task cfg) as in training. The policy is
loaded through the task's registered runner (``SuperMimicRunner.load`` + ``get_inference_policy``, like ``play.run_play``).

Differences to Play, on purpose:
  * The three tracking terminations (``base_pos_too_far``, ``base_pg_too_far``, ``link_pos_too_far``) that Play removes are
    put back from the training cfg (``make_terminations``). Otherwise "success" would be vacuous.
  * The episode time limit is raised (``--episode-length-s``, default 30 s) so ``dataset_exhausted`` (motion finished) is
    what ends a successful episode instead of the 10 s ``time_out``.
  * The terrain generator runs in curriculum mode with one row per terrain of ``metadata.yaml`` so that every motion has
    a terrain to be placed on (random mode would leave some terrains out).
  * ``robot_reference`` / debug visualization are removed (many envs).

Each env plays exactly ONE episode; its further steps are ignored. Motions are assigned explicitly: env ``e`` plays motion
``e % K`` (every motion gets >= 3 episodes, ``num_envs >= min_episodes_per_motion * K`` is enforced). The assignment uses the env's own reset path:
for each motion the TerrainMotion buffer's sampling weights are set to a one-hot and the envs of that motion are reset
with ``env.reset(env_ids)``; the buffer then samples motion, start time (0) and terrain origin itself. The only internal
access is the ``_motion_weights`` tensor of the buffer (there is no public API to pin a motion).

Success = ``dataset_exhausted`` fires and no failure termination fires on the same step.
"""

from __future__ import annotations

import copy
import csv
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import torch
import tyro
import yaml
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer.viewer_config import ViewerConfig

import instinct_mj.tasks  # noqa: F401
from instinct_mj.scripts.supermimic.common import apply_motion_folder
from instinct_mj.tasks.registry import (
    load_env_cfg,
    load_env_cls,
    load_instinct_rl_cfg,
    load_runner_cls,
    load_vecenv_cls,
)
from instinct_mj.tasks.shadowing.perceptive.perceptive_env_cfg import make_terminations

PLAY_TASK = "Instinct-Perceptive-Shadowing-G1-Play-v0"
FAILURE_TERMS = ("base_pos_too_far", "base_pg_too_far", "link_pos_too_far", "illegal_reset_contact", "out_of_border")
SUCCESS_TERM = "dataset_exhausted"


@dataclass
class EvalArgs:
    motion_folder: str
    checkpoint: str | None = None
    """A single ``OnPolicyRunner``-format checkpoint (``model_<it>.pt``)."""
    run_dir: str | None = None
    """Evaluate every ``every``-th ``model_<it>.pt`` of this run dir."""
    every: int = 1
    watch: bool = False
    """Keep polling ``run_dir`` for new checkpoints and append their results."""
    watch_poll_s: float = 120.0
    watch_max_idle_s: float = 3600.0
    num_envs: int = 64
    min_episodes_per_motion: int = 3
    episode_length_s: float = 30.0
    device: str = "cuda:0"
    seed: int = 7
    video: bool = False
    """Record one episode per motion (first evaluated checkpoint only) to ``<run_dir>/eval/videos/``."""
    video_size: tuple[int, int] = (480, 640)
    """Video height x width."""
    video_only: bool = False
    """Only record videos (no evaluation, no 64-env eval env)."""
    video_motions: tuple[str, ...] = ()
    """Record only motions whose file name contains one of these substrings (default: all)."""
    video_distance: float = 3.0
    video_elevation: float = -20.0
    """Camera distance [m] / elevation [deg] (negative = looking down) relative to torso_link."""
    video_azimuth: float | None = None
    """Manual camera azimuth [deg, world frame]. Default: chosen automatically per motion so that the line of sight to the
    robot along the reference path is not blocked by terrain (side view preferred)."""


def build_eval_cfg(args: EvalArgs, num_envs: int, render: bool = False):
    cfg = load_env_cfg("Instinct-Perceptive-Shadowing-G1-v0", play=True)
    cfg.seed = args.seed
    cfg.scene.num_envs = num_envs
    cfg.episode_length_s = args.episode_length_s
    apply_motion_folder(cfg, args.motion_folder, restore_baseline_sampling=False)
    # no reference robot / debug visualization / monitors
    cfg.scene.entities.pop("robot_reference", None)
    motion_reference_cfg = next(s for s in cfg.scene.sensors if s.name == "motion_reference")
    motion_reference_cfg.debug_vis = False
    motion_reference_cfg.reference_entity_name = None
    next(s for s in cfg.scene.sensors if s.name == "camera").debug_vis = False
    cfg.observations["policy"].terms["depth_image"].params["debug_vis"] = False
    cfg.monitors = {}
    # tracking terminations that Play removes
    train_terms = make_terminations()
    for name in ("base_pos_too_far", "base_pg_too_far", "link_pos_too_far"):
        cfg.terminations[name] = train_terms[name]
    # one terrain row per terrain of the dataset
    with open(os.path.join(os.path.expanduser(args.motion_folder), "metadata.yaml"), encoding="utf-8") as f:
        metadata = yaml.safe_load(f)
    generator = cfg.scene.terrain.terrain_generator
    generator.curriculum, generator.num_rows, generator.num_cols = True, len(metadata["terrains"]), 1
    if render:
        cfg.viewer = ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY, entity_name="robot", body_name="torso_link",
            distance=4.0, elevation=-15.0, azimuth=140.0, height=args.video_size[0], width=args.video_size[1],
        )  # fmt: skip
    return cfg, [m["motion_file"] for m in metadata["motion_files"]]


def load_agent_dict(args: EvalArgs) -> dict:
    """The task's agent cfg, with the ``policy`` network cfg of the run that produced the checkpoint (``<run>/params/agent.yaml``,
    written by ``instinct-train``) so that runs with a non-default network (e.g. another encoder nonlinearity) are rebuilt as
    trained; checkpoints outside a run dir keep the task's network."""
    agent_cfg = load_instinct_rl_cfg(PLAY_TASK)
    agent_cfg.device = args.device
    d = agent_cfg.to_dict()
    params = Path(args.run_dir or Path(args.checkpoint).parent).expanduser() / "params" / "agent.yaml"
    if params.is_file():
        with open(params, encoding="utf-8") as f:
            d["policy"] = yaml.safe_load(f)["policy"]
    return d


class Evaluator:
    def __init__(self, args: EvalArgs):
        self.args = args
        cfg, self.motion_files = build_eval_cfg(args, args.num_envs)
        self.K = len(self.motion_files)
        if args.num_envs < args.min_episodes_per_motion * self.K:
            raise ValueError(
                f"num_envs={args.num_envs} < {args.min_episodes_per_motion} x {self.K} motions: not every motion would get "
                f"{args.min_episodes_per_motion} episodes."
            )
        agent_dict = load_agent_dict(args)
        env = load_env_cls(PLAY_TASK)(cfg=cfg, device=args.device, render_mode=None)
        self.vec = load_vecenv_cls(PLAY_TASK)(env, policy_group="policy", critic_group="critic")
        self.env = self.vec.unwrapped
        self.runner = load_runner_cls(PLAY_TASK)(self.vec, agent_dict, log_dir=None, device=args.device)
        self.dt = self.env.step_dt
        self.mgr = self.env.scene["motion_reference"]
        self.buffer = next(iter(self.mgr.motion_buffers.values()))
        assert bool((self.buffer._all_motion_num_selectable_origins > 0).all()), "a motion has no terrain origin"
        self.robot = self.env.scene["robot"]
        self.link_ids = self.robot.find_bodies(self.mgr.cfg.link_of_interests, preserve_order=True)[0]
        rm = self.env.reward_manager
        self.max_step_reward = self.dt * sum(
            max(rm.get_term_cfg(n).weight, 0.0) for n in rm.active_terms["rewards"]
        )  # positive weights x dt, as supermimic's MAX_STEP_REWARD
        self.motion_of_env = torch.arange(args.num_envs, device=args.device) % self.K

    def assign_motions(self) -> None:
        """Reset every env onto its motion (t = 0) through the env's own reset path."""
        for k in range(self.K):
            ids = (self.motion_of_env == k).nonzero(as_tuple=False).squeeze(-1)
            self.buffer._motion_weights[:] = 0.0
            self.buffer._motion_weights[k] = 1.0
            self.env.reset(env_ids=ids)
        names = self.mgr.get_current_motion_identifiers()
        expected = [os.path.join(os.path.expanduser(self.args.motion_folder), self.motion_files[int(k)]) for k in self.motion_of_env]
        assert names == expected, "motion assignment mismatch"

    @torch.no_grad()
    def run(self, checkpoint: str) -> dict:
        t0 = time.time()
        self.runner.load(checkpoint)
        policy = self.runner.get_inference_policy(device=self.args.device)
        self.assign_motions()
        N, dev = self.args.num_envs, self.args.device
        lengths = self.mgr.complete_motion_lengths.clone()  # seconds, per env
        obs, _ = self.vec.get_observations()
        active = torch.ones(N, dtype=torch.bool, device=dev)
        steps = torch.zeros(N, device=dev)
        ret = torch.zeros(N, device=dev)
        err_sum = torch.zeros(3, N, device=dev)  # base pos, link pos (relative world frame), joint pos
        err_n = torch.zeros(N, device=dev)
        success = torch.zeros(N, dtype=torch.bool, device=dev)
        fired = {n: torch.zeros(N, dtype=torch.bool, device=dev) for n in (*FAILURE_TERMS, SUCCESS_TERM, "time_out")}
        setup_s, loop_start, loop_steps = time.time() - t0, time.time(), 0
        tm = self.env.termination_manager
        for _ in range(int(self.args.episode_length_s / self.dt) + 5):
            obs, rew, dones, _ = self.vec.step(policy(obs))
            done = dones.bool() & active
            steps += active
            ret += rew.sum(-1) * active
            for name in fired:
                fired[name] |= tm.get_term(name) & done
            ok_step = active & ~done  # the state of a finished env is already its reset state
            ref = self.mgr.reference_frame
            e_base = (self.robot.data.root_link_pos_w - ref.base_pos_w[:, 0]).norm(dim=-1)
            e_link = (self.robot.data.body_link_pos_w[:, self.link_ids] - self.mgr.reference_link_pos_relative_w).norm(dim=-1).mean(-1)
            e_joint = (self.robot.data.joint_pos - ref.joint_pos[:, 0]).abs().mean(-1)
            err_sum += torch.stack((e_base, e_link, e_joint)) * ok_step
            err_n += ok_step
            active &= ~done
            loop_steps += 1
            if not bool(active.any()):
                break
        loop_s = time.time() - loop_start
        failure = torch.stack([fired[n] for n in FAILURE_TERMS]).any(0)
        success = fired[SUCCESS_TERM] & ~failure
        completed = torch.where(success, torch.ones_like(steps), (steps * self.dt / lengths).clamp(max=1.0))
        norm_return = ret / steps.clamp(min=1) / self.max_step_reward
        err = err_sum / err_n.clamp(min=1)
        result = {"checkpoint": checkpoint, "seconds": time.time() - t0, "setup_seconds": setup_s, "loop_steps": loop_steps,
                  "seconds_per_step": loop_s / loop_steps, "num_envs": N, "max_step_reward": self.max_step_reward, "motions": {}}
        for k, name in enumerate(self.motion_files + ["__overall__"]):
            m = (self.motion_of_env == k) if k < self.K else torch.ones(N, dtype=torch.bool, device=dev)
            reasons = {n: int(f[m].sum()) for n, f in fired.items() if int(f[m].sum()) > 0}
            result["motions"][name] = {
                "episodes": int(m.sum()), "success_rate": float(success[m].float().mean()),
                "mean_fraction_completed": float(completed[m].mean()), "normalized_return": float(norm_return[m].mean()),
                "base_pos_err_m": float(err[0][m].mean()), "link_pos_err_m": float(err[1][m].mean()),
                "joint_pos_err_rad": float(err[2][m].mean()), "termination_counts": reasons,
            }  # fmt: skip
        return result


def choose_camera(env, buffer, motion_index: int, args: EvalArgs) -> tuple[float, float]:
    """Pick the camera azimuth (and distance) for the motion the single render env was just reset onto.

    The camera tracks ``torso_link`` at ``video_distance`` / ``video_elevation``. For every candidate azimuth (15 deg steps)
    the line of sight from sampled points of the REFERENCE path (every 0.2 s, local + terrain origin) to the camera is cast
    against the terrain geoms only; the azimuth with the fewest blocked sample points wins, ties prefer a side view
    (perpendicular to the overall motion direction)."""
    if args.video_azimuth is not None:
        return args.video_azimuth, args.video_distance
    seq = buffer._all_motion_sequences
    path = seq.base_pos_w[motion_index].cpu().numpy() + buffer._assigned_env_origins[0].cpu().numpy()
    fps = float(seq.framerate[motion_index])
    path = path[:: max(int(0.2 * fps), 1)] + np.array([0.0, 0.0, 0.3])
    heading = np.arctan2(*(path[-1, 1::-1] - path[0, 1::-1]))  # atan2(dy, dx)
    cpu = copy.deepcopy(env.sim.mj_model)
    is_terrain = cpu.geom_bodyid == cpu.body("terrain").id
    cpu.geom_group[~is_terrain] = 5
    data = mujoco.MjData(cpu)
    mujoco.mj_forward(cpu, data)
    geomgroup = np.zeros(6, dtype=np.uint8)
    geomgroup[int(cpu.geom_group[is_terrain][0])] = 1
    geomid = np.zeros(1, dtype=np.int32)
    el = np.radians(args.video_elevation)
    best = None
    for distance in (args.video_distance, args.video_distance + 1.0):
        for az_deg in range(0, 360, 15):
            az = np.radians(az_deg)
            offset = -distance * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])  # camera - lookat (MuJoCo: camera = lookat - d * forward)
            blocked = 0
            for p in path:
                dist = mujoco.mj_ray(cpu, data, p, offset / distance, geomgroup, 1, -1, geomid)
                blocked += int(0 <= dist < distance - 0.05)
            score = (blocked / len(path)) - 0.05 * abs(np.sin(az - heading))
            if best is None or score < best[0] - 1e-9:
                best = (score, az_deg, distance, blocked / len(path))
        if best[3] < 0.02:
            break
    print(f"[INFO] video camera for motion {motion_index}: azimuth {best[1]} deg, distance {best[2]:.1f} m, blocked path fraction {best[3]:.2f}")
    return float(best[1]), float(best[2])


def record_videos(args: EvalArgs, checkpoint: str, folder: Path) -> None:
    """One episode per motion with the existing mjlab ``VideoRecorder`` on a single-env render env."""
    cfg, motion_files = build_eval_cfg(args, 1, render=True)
    agent_dict = load_agent_dict(args)
    env = load_env_cls(PLAY_TASK)(cfg=cfg, device=args.device, render_mode="rgb_array")
    buffer = next(iter(env.scene["motion_reference"].motion_buffers.values()))
    vec0 = load_vecenv_cls(PLAY_TASK)(env, policy_group="policy", critic_group="critic")
    runner = load_runner_cls(PLAY_TASK)(vec0, agent_dict, log_dir=None, device=args.device)
    runner.load(checkpoint)
    policy = runner.get_inference_policy(device=args.device)
    for k, name in enumerate(motion_files):
        if args.video_motions and not any(sub in name for sub in args.video_motions):
            continue
        buffer._motion_weights[:] = 0.0
        buffer._motion_weights[k] = 1.0
        recorder = VideoRecorder(env, folder, step_trigger=lambda step: step == 0, video_length=None,
                                 name_prefix=re.sub(r"[^A-Za-z0-9]+", "_", os.path.splitext(name)[0]), disable_logger=False)  # fmt: skip
        vec = load_vecenv_cls(PLAY_TASK)(recorder, policy_group="policy", critic_group="critic")  # resets -> motion k
        azimuth, distance = choose_camera(env, buffer, k, args)
        camera = env._offline_renderer._cam  # the renderer's free-tracking camera (no public setter)
        camera.azimuth, camera.distance, camera.elevation = azimuth, distance, args.video_elevation
        obs, _ = vec.get_observations()
        with torch.no_grad():
            for _ in range(int(args.episode_length_s / env.step_dt) + 5):
                obs, _, dones, _ = vec.step(policy(obs))
                if not recorder.is_recording:
                    break
    env.close()


CSV_COLUMNS = ["iteration", "motion", "episodes", "success_rate", "mean_fraction_completed", "normalized_return",
               "base_pos_err_m", "link_pos_err_m", "joint_pos_err_rad", "termination_counts"]  # fmt: skip


def iteration_of(path: str) -> int:
    return int(re.match(r"model_(\d+)\.pt$", os.path.basename(path)).group(1))


def print_table(result: dict, iteration: int) -> None:
    print(f"\n=== iteration {iteration}: {result['checkpoint']}  ({result['seconds']:.1f} s = {result['setup_seconds']:.1f} s setup + {result['loop_steps']} steps x {result['seconds_per_step']:.3f} s, {result['num_envs']} envs) ===")
    print(f"{'motion':58s} {'n':>3s} {'succ':>5s} {'compl':>6s} {'ret':>6s} {'e_base':>7s} {'e_link':>7s} {'e_joint':>7s}  terminations")
    for name, m in result["motions"].items():
        print(f"{name[:58]:58s} {m['episodes']:3d} {m['success_rate']:5.2f} {m['mean_fraction_completed']:6.2f} {m['normalized_return']:6.3f} "
              f"{m['base_pos_err_m']:7.3f} {m['link_pos_err_m']:7.3f} {m['joint_pos_err_rad']:7.3f}  {m['termination_counts']}")  # fmt: skip


def write_results(out_dir: Path, iteration: int, result: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"eval_{iteration}.json").write_text(json.dumps(result, indent=2))
    csv_path = out_dir / "results.csv"
    new = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if new:
            writer.writeheader()
        for name, m in result["motions"].items():
            writer.writerow({"iteration": iteration, "motion": name, **{k: m[k] for k in CSV_COLUMNS[2:-1]},
                             "termination_counts": json.dumps(m["termination_counts"])})  # fmt: skip


def evaluated_iterations(out_dir: Path) -> set[int]:
    return {int(re.match(r"eval_(\d+)\.json", p.name).group(1)) for p in out_dir.glob("eval_*.json")}


def main() -> None:
    args = tyro.cli(EvalArgs)
    os.environ.setdefault("MUJOCO_GL", "egl")
    assert (args.checkpoint is None) != (args.run_dir is None), "give exactly one of --checkpoint / --run-dir"
    run_dir = Path(args.run_dir or Path(args.checkpoint).parent).expanduser().resolve()
    out_dir = run_dir / "eval"
    if args.video_only:
        assert args.checkpoint is not None, "--video-only needs --checkpoint"
        record_videos(args, str(Path(args.checkpoint).expanduser().resolve()), out_dir / "videos")
        return
    evaluator = Evaluator(args)
    video_done = not args.video
    idle_since = time.time()
    while True:
        if args.checkpoint is not None:
            todo = [str(Path(args.checkpoint).expanduser().resolve())]
        else:
            ckpts = sorted((p for p in run_dir.glob("model_*.pt") if re.match(r"model_\d+\.pt$", p.name)), key=lambda p: iteration_of(str(p)))
            todo = [str(p) for p in ckpts[:: args.every]]
            todo = [p for p in todo if iteration_of(p) not in evaluated_iterations(out_dir)]
        for path in todo:
            result = evaluator.run(path)
            write_results(out_dir, iteration_of(path), result)
            print_table(result, iteration_of(path))
            if not video_done:
                record_videos(args, path, out_dir / "videos")
                video_done = True
            idle_since = time.time()
        if args.checkpoint is not None or not args.watch or time.time() - idle_since > args.watch_max_idle_s:
            break
        time.sleep(args.watch_poll_s)


if __name__ == "__main__":
    main()
