"""Cost and convergence of the physics solver settings on the same checkpoint (256 envs on the clips matching ``--clip-regex``, e.g. the climbs).

    ACP_DATASET=<dir> python -m instinct_mj.scripts.acp.measure_solver <checkpoint> --iterations 10 --ls-iterations 20 [--clip-regex 'climb_76_up_00|climb_58_00'] [--steps 400]

Per policy step (4 physics substeps): wall time of ``env.step`` (synchronized), the solver iterations of the last substep (``solver_niter``: the mean, and the share of
steps that used the full ``iterations``), the number of non-finite states, the episodes that fell (failure terminations) and the mean horizontal pelvis error to the reference."""

from __future__ import annotations

import argparse
import os
import re
import time

import torch

import instinct_mj.tasks.acp.config.g1.acp_tracking_cfg as acp_g1

dataset = os.path.expanduser(os.environ["ACP_DATASET"])
buffer = acp_g1.motion_reference_cfg.motion_buffers["TerrainMotion"]
buffer.path, buffer.metadata_yaml = dataset, os.path.join(dataset, "metadata.yaml")

import instinct_mj.tasks  # noqa: E402,F401
from instinct_mj.rl import InstinctRlVecEnvWrapper  # noqa: E402
from instinct_mj.tasks.registry import load_env_cfg, load_env_cls, load_instinct_rl_cfg, load_runner_cls  # noqa: E402

TASK = "Instinct-ACP-Track-G1-v1"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--iterations", type=int, default=10)
    ap.add_argument("--ls-iterations", type=int, default=20)
    ap.add_argument("--clip-regex", default="climb_76_up_00|climb_58_00")
    ap.add_argument("--num-envs", type=int, default=256)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    device = "cuda:0"
    torch.manual_seed(args.seed)
    cfg = load_env_cfg(TASK)
    cfg.scene.num_envs = args.num_envs
    cfg.sim.mujoco.iterations, cfg.sim.mujoco.ls_iterations = args.iterations, args.ls_iterations
    env = load_env_cls(TASK)(cfg=cfg, device=device)
    vec = InstinctRlVecEnvWrapper(env, policy_group="policy", critic_group="critic")
    agent = load_instinct_rl_cfg(TASK).to_dict()
    agent.update(compile=False, compile_policy=False, buffer_steps=64, warmup_steps=32, wm_scan="planner")  # (the checkpoint was trained with the 17 x 11 world-model scan)
    runner = load_runner_cls(TASK)(vec, agent, None, device)
    runner.load(args.checkpoint)
    runner.setup_training()
    policy = runner.get_inference_policy(device=device)
    torch.set_grad_enabled(False)
    start = env.curriculum_manager.get_term_cfg("episode_start").func
    start.mode = "start"  # every episode from the clip start
    mb = start.motion_buffer
    clips = [k for k, n in enumerate(runner.motion_names) if re.search(args.clip_regex, n)]
    assert clips, f"no clip matches {args.clip_regex}"
    mb._motion_weights[:] = 0.0
    mb._motion_weights[clips] = 1.0
    print(f"clips: {[runner.motion_names[k] for k in clips]}; solver iterations {env.sim.mj_model.opt.iterations} ls {env.sim.mj_model.opt.ls_iterations}", flush=True)
    env.cfg.auto_reset = False
    env.reset(env_ids=torch.arange(args.num_envs, device=device))
    obs, _ = vec.get_observations()
    reader, motion, mdp = runner.reader, runner.motion, runner.mdp
    times, niter_mean, niter_cap, bad_count, falls, episodes, xy = [], [], [], 0, 0, 0, []
    tm = env.termination_manager
    for step in range(args.steps):
        a = torch.clamp(policy(obs) + 0.1 * torch.randn(args.num_envs, 23, device=device), -20, 20)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        obs, _, dones, _ = vec.step(a)
        torch.cuda.synchronize()
        if step >= 10:
            times.append(time.perf_counter() - t0)
        niter = env.sim.data.solver_niter.float()
        niter_mean.append(float(niter.mean())), niter_cap.append(float((niter >= args.iterations).float().mean()))
        S = reader.state()
        bad = ~torch.stack([torch.isfinite(x).flatten(1).all(-1) for x in S]).all(0)
        bad_count += int(bad.sum())
        f1, phase1, origin1 = motion.current_frame()
        xy.append(float((S[0] - mdp.reference_state(f1, origin1)[0])[~bad, :2].norm(dim=-1).mean()))
        done = dones.bool() | bad
        if done.any():
            ids = done.nonzero().squeeze(-1)
            episodes += len(ids)
            falls += int((env.reset_terminated[ids].bool() | bad[ids]).sum())
            env.reset(env_ids=ids)
            obs, _ = vec.get_observations()
    t = torch.tensor(times)
    print(f"RESULT iterations {args.iterations}/{args.ls_iterations}: step time mean {1e3 * t.mean():.2f} ms (median {1e3 * t.median():.2f}), solver_niter mean {sum(niter_mean) / len(niter_mean):.2f}, "
          f"share of env-steps at the iteration cap {100 * sum(niter_cap) / len(niter_cap):.1f} %, non-finite env-steps {bad_count}, episodes {episodes} (failed {falls}), mean pelvis xy error {sum(xy) / len(xy):.3f} m")


if __name__ == "__main__":
    main()
