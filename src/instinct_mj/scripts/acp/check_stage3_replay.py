"""Stage 3 planner replay on the live env: goal-reaching episodes -> hindsight labels -> goal partition of the replay.

The goal envs of the even terrain types are put 0.35 m from their goal (standing, at the goal height): their episodes are successful (within 0.5 m
horizontally at |dz| <= 0.1 m), not within the 0.2 m truncation distance, so they must enter the goal partition keyed by (terrain type, level).
The goal envs of the odd types start 0.1 m from the goal: the recorded segment is truncated at the first step within 0.2 m, i.e. it is empty and
nothing may be stored. The paper's success filter is on (``planner_success_only``).

    python -m instinct_mj.scripts.acp.check_stage3_replay [--num-envs 64] [--steps 150]
"""

from __future__ import annotations

import argparse

import torch

import instinct_mj.tasks  # noqa: F401
from instinct_mj.rl import InstinctRlVecEnvWrapper
from instinct_mj.rl.acp.stage3_runner import ACPStage3Runner
from instinct_mj.tasks.acp.config.g1.rl_stage3_cfg import G1AcpStage3RunnerCfg
from instinct_mj.tasks.registry import load_env_cfg, load_env_cls

TASK = "Instinct-ACP-Stage3-G1-v1"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=150)
    args = parser.parse_args()
    device = "cuda:0"
    cfg = load_env_cfg(TASK)
    cfg.scene.num_envs = args.num_envs
    env = load_env_cls(TASK)(cfg=cfg, device=device)
    vec = InstinctRlVecEnvWrapper(env, policy_group="policy", critic_group="critic")
    agent = G1AcpStage3RunnerCfg()
    agent.compile, agent.compile_policy = False, False
    agent.planner_success_only = True
    agent.warmup_steps, agent.buffer_steps, agent.pi_batch, agent.wm_batch, agent.fail_windows = 40, 200, 64, 128, 64
    runner = ACPStage3Runner(vec, agent.to_dict(), log_dir=None, device=device)
    runner.setup_training()
    gt, robot = runner.goal_term, runner.robot
    stats = {k: [] for k in ("r", "ep_len", "term", "done", "jl_soft", "rec_episodes", "rec_samples")}
    goal = gt.is_goal
    even = goal & (gt.terrain_type % 2 == 0)
    odd = goal & (gt.terrain_type % 2 == 1)
    placed = {"even": 0, "odd": 0}
    admitted_types = set()
    for step in range(args.steps):
        fresh = env.episode_length_buf == 0
        for name, sel, dist in (("even", even, 0.35), ("odd", odd, 0.1)):
            ids = (sel & fresh).nonzero().flatten()
            if len(ids) == 0:
                continue
            ang = torch.rand(len(ids), device=device) * 6.283
            xy = gt.target_pos_w[ids, :2] + dist * torch.stack((torch.cos(ang), torch.sin(ang)), -1)
            pos = torch.cat((xy, gt.target_pos_w[ids, 2:3]), -1)
            yaw = gt.target_yaw[ids]
            quat = torch.stack((torch.cos(yaw / 2), torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(yaw / 2)), -1)
            robot.write_root_link_pose_to_sim(torch.cat((pos, quat), -1), env_ids=ids)
            robot.write_root_link_velocity_to_sim(torch.zeros(len(ids), 6, device=device), env_ids=ids)
            robot.write_joint_state_to_sim(robot.data.default_joint_pos[ids], torch.zeros_like(robot.data.default_joint_pos[ids]), env_ids=ids)
            env.sim.forward()
            placed[name] += len(ids)
        runner.collect(1, stats)
    replay = runner.replay
    types = sorted(replay.tiles)
    print(f"placed episodes: even-type {placed['even']}, odd-type {placed['odd']}")
    print(f"goal partition: {replay.goal_size} samples, terrain types {types}, tiles per type {[sorted(replay.tiles[t]) for t in types]}")
    qualities = [q for t in replay.tiles.values() for tile in t.values() for q in tile.quality]
    print(f"goal episodes stored {len(qualities)}, quality (work / max(d0, 0.5)) min {min(qualities, default=float('nan')):.3f} max {max(qualities, default=float('nan')):.3f}")
    print("reference partition:", len(replay.ref), "samples")
    failures = []
    if replay.goal_size == 0:
        failures.append("no goal episode admitted")
    if any(t % 2 == 1 for t in types):
        failures.append(f"odd-type (truncated-to-empty) episodes were stored: {types}")
    admitted_types = set(types)
    if not admitted_types <= set(gt.terrain_type[even].unique().tolist()):
        failures.append("stored types outside the placed ones")
    # the batch sampler mixes both partitions proportionally and the trainer runs on it
    batch = replay.sample(512)
    print("sampled batch:", {k: tuple(v.shape) for k, v in batch.items()})
    losses = runner.planner_trainer.update(replay)
    print("planner update on the partitioned replay:", losses if losses is None else {k: round(v, 3) for k, v in losses.items()})
    # eviction by quality: shrink the tile capacity and add a worse episode
    t0 = types[0]
    level0 = sorted(replay.tiles[t0])[0]
    tile = replay.tiles[t0][level0]
    n_before = len(tile.quality)
    replay.tile_capacity = max(tile.sizes[0] * 1, 1)  # one episode fits
    best = min(tile.quality)
    sample = {k: v[: tile.sizes[0]].clone() for k, v in tile.data.items()}
    kept_bad = replay.add_goal(sample, t0, level0, best + 100.0)
    kept_good = replay.add_goal(sample, t0, level0, best - 1.0)
    print(f"eviction: capacity 1 episode, a worse one is {'kept (FAIL)' if kept_bad else 'rejected'}, a better one is {'kept' if kept_good else 'rejected (FAIL)'}; tile holds {len(tile.quality)} (was {n_before})")
    if kept_bad or not kept_good or len(tile.quality) != 1:
        failures.append("quality-based eviction")
    if failures:
        print("FAILURES:", failures)
        raise SystemExit(1)
    print("OK")


if __name__ == "__main__":
    main()
