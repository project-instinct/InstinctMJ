"""Wall-time breakdown of one ACPRunner iteration (collect, world-model update, policy update, planner), with CUDA syncs between the parts.

    ACP_DATASET=<dir> python -m instinct_mj.scripts.acp.profile_runner [--num-envs 256] [--iters 20] [--compile] [--exact-scan 1]
"""

from __future__ import annotations

import argparse
import os
import time

import torch

import instinct_mj.tasks.acp.config.g1.acp_tracking_cfg as acp_g1

dataset = os.path.expanduser(os.environ["ACP_DATASET"])
buffer = acp_g1.motion_reference_cfg.motion_buffers["TerrainMotion"]
buffer.path, buffer.metadata_yaml = dataset, os.path.join(dataset, "metadata.yaml")

import instinct_mj.tasks  # noqa: E402,F401
from instinct_mj.rl import InstinctRlVecEnvWrapper  # noqa: E402
from instinct_mj.rl.acp.runner import ACPRunner  # noqa: E402
from instinct_mj.tasks.registry import load_env_cfg, load_env_cls, load_instinct_rl_cfg  # noqa: E402

TASK = "Instinct-ACP-Track-G1-v0"


def timed(fn, *args, **kwargs):
    torch.cuda.synchronize()
    t = time.time()
    out = fn(*args, **kwargs)
    torch.cuda.synchronize()
    return out, time.time() - t


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warm", type=int, default=3)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--exact-scan", type=int, default=1)
    parser.add_argument("--wm-batches", type=int, default=None)
    parser.add_argument("--torch-profile", action="store_true")
    args = parser.parse_args()
    cfg = load_env_cfg(TASK)
    cfg.scene.num_envs = args.num_envs
    env = load_env_cls(TASK)(cfg=cfg, device="cuda:0")
    vec = InstinctRlVecEnvWrapper(env, policy_group="policy", critic_group="critic")
    agent = load_instinct_rl_cfg(TASK).to_dict()
    agent.update(compile=args.compile, compile_policy=args.compile, exact_scan=bool(args.exact_scan))
    if args.wm_batches is not None:
        agent["wm_batches"] = args.wm_batches
    runner = ACPRunner(vec, agent, log_dir=None, device="cuda:0")
    runner.setup_training()
    stats = {k: [] for k in ("r", "ep_len", "term", "done", "jl_soft", "rec_episodes", "rec_samples")}
    runner.collect(agent["warmup_steps"], stats)
    steps_per_iter = 1.0 / agent["utd"]
    acc = {"collect": 0.0, "wm_update": 0.0, "policy_update": 0.0}
    total = 0.0
    prof = None
    if args.torch_profile:
        from torch.profiler import ProfilerActivity, profile

    parts: dict[str, float] = {}

    def wrap(obj, name, key):
        orig = getattr(obj, name)

        def timed_fn(*a, **k):
            torch.cuda.synchronize()
            t = time.time()
            out = orig(*a, **k)
            torch.cuda.synchronize()
            parts[key] = parts.get(key, 0.0) + time.time() - t
            return out

        setattr(obj, name, timed_fn)

    wrap(runner.env, "step", "  env.step (simulator + managers + sensors)")
    wrap(runner.scanner, "height_scan", "  height scans (all call sites, includes the rays)")
    wrap(runner, "_imagine", "  imagination rollout (policy + WM + obs, includes scans)")
    wrap(runner, "wm_rollout", "  world-model k-step rollout (forward)")
    wrap(runner, "critic_input", "  critic features")
    wrap(runner, "reward_term", "  analytic reward + soft termination")
    wrap(runner, "observe", "  imagined observation pieces")
    wrap(runner.kin, "forward", "  collect+all: forward kinematics")
    wrap(runner, "sim_stats", "  collect: world-model normalizer statistics")
    wrap(runner.buffer, "add", "  collect: replay buffer add")
    wrap(runner.pol_norm, "update", "  collect: policy normalizer update")
    wrap(runner.ac, "act_inference", "  collect: actor")
    wrap(runner.unwrapped, "reset", "  collect: env reset of done envs")
    wrap(runner.reader, "state", "  collect: sim state read")
    wrap(runner.reader, "contact", "  collect: contact read")
    u = runner.unwrapped
    wrap(u, "_reset_idx", "    reset: _reset_idx (events, curriculum, managers, motion)")
    wrap(u.sim, "forward", "    reset/step: sim.forward")
    wrap(u.sim, "sense", "    reset/step: sim.sense (ray casts)")
    wrap(u.command_manager, "compute", "    reset/step: command_manager.compute")
    wrap(u.observation_manager, "compute", "    reset/step: observation_manager.compute")
    wrap(u.scene, "write_data_to_sim", "    reset: scene.write_data_to_sim")
    wrap(runner.env, "get_observations", "    collect: env.get_observations after the reset")
    for i in range(args.warm + args.iters):
        if i == args.warm:
            parts.clear()
        if i == args.warm and args.torch_profile:
            prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
            prof.__enter__()
        t0 = time.time()
        n = 2 if i % 2 == 0 else 3  # utd 0.4: 2.5 simulator steps per iteration
        _, tc = timed(runner.collect, n, stats)
        _, tw = timed(lambda: [runner.wm_update() for _ in range(agent["wm_batches"])])
        _, tp = timed(runner.policy_update)
        if i >= args.warm:
            acc["collect"] += tc
            acc["wm_update"] += tw
            acc["policy_update"] += tp
            total += time.time() - t0
    if prof is not None:
        prof.__exit__(None, None, None)
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=18, max_name_column_width=70))
    print(f"\nPROFILE envs {args.num_envs} compile {args.compile} exact_scan {args.exact_scan} wm_batches {agent['wm_batches']}: {total / args.iters * 1e3:.0f} ms / iteration = {args.iters / total:.2f} it/s")
    for k, v in acc.items():
        if v:
            print(f"  {k:16s} {v / args.iters * 1e3:7.1f} ms ({100 * v / total:4.1f} %)")
    for k, v in sorted(parts.items(), key=lambda kv: -kv[1]):
        print(f"{k:62s} {v / args.iters * 1e3:7.1f} ms / iteration")
    print(f"  collect per simulator step: {acc['collect'] / args.iters / 2.5 * 1e3:.1f} ms")


if __name__ == "__main__":
    main()
