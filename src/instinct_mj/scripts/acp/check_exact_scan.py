"""Parity and cost of the exact imagined height scan (``rl/acp/rays.py``) against the env's ray-cast scanner and the 0.05 m height map.

Robots are put at random poses on the terrain tiles of the dataset (random yaw, 0.6-1.2 m above the ground), the env senses, and the critic scan of
the env (true pelvis pose, ``pelvis_z - hit_z - 0.5``) is compared per ray with the exact scan and with the height map scan.

    python -m instinct_mj.scripts.acp.check_exact_scan [--dataset-dir /tmp/acp_geoms_ds] [--num-envs 64] [--poses 8]
"""

from __future__ import annotations

import argparse
import os
import time

import torch

import instinct_mj.tasks  # noqa: F401  (registers the tasks)
from instinct_mj.rl import InstinctRlVecEnvWrapper
from instinct_mj.rl.acp.rays import ExactHeightScanner
from instinct_mj.rl.acp.sim import ACPSimReader, build_mdp
from instinct_mj.tasks.registry import load_env_cfg, load_env_cls

TASK = "Instinct-ACP-Track-G1-v0"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="/tmp/acp_geoms_ds")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--poses", type=int, default=8)
    parser.add_argument("--heightmap-cache", default="/tmp/acp_geoms_heightmaps")
    args = parser.parse_args()
    import instinct_mj.tasks.acp.config.g1.acp_tracking_cfg as acp_g1

    buffer = acp_g1.motion_reference_cfg.motion_buffers["TerrainMotion"]
    buffer.path, buffer.metadata_yaml = args.dataset_dir, os.path.join(args.dataset_dir, "metadata.yaml")
    device = "cuda:0"
    cfg = load_env_cfg(TASK)
    cfg.scene.num_envs = args.num_envs
    cfg.disable_randomization()
    env = load_env_cls(TASK)(cfg=cfg, device=device)
    vec = InstinctRlVecEnvWrapper(env, policy_group="policy", critic_group="critic")
    vec.reset()
    model = env.sim.mj_model
    import mujoco
    import numpy as np

    terrain_geoms = np.nonzero(model.geom_bodyid == model.body("terrain").id)[0]
    print("terrain geoms:", {mujoco.mjtGeom(t).name: int((model.geom_type[terrain_geoms] == t).sum()) for t in set(model.geom_type[terrain_geoms].tolist())})
    motion, kin, scanner, mdp = build_mdp(env, 0.05, args.heightmap_cache, device)
    exact = ExactHeightScanner(env, scanner)
    reader = ACPSimReader(env)
    robot = env.scene["robot"]
    free_q = robot.indexing.free_joint_q_adr.long()
    sensor = env.scene["critic_height_scanner"]
    buf = next(iter(env.scene["motion_reference"].motion_buffers.values()))
    g = torch.Generator(device=device).manual_seed(0)
    diffs_exact, diffs_map, miss_cells, total = [], [], 0, 0
    for _ in range(args.poses):
        origin = buf._assigned_env_origins
        n = args.num_envs
        qpos = env.sim.data.qpos.clone()
        xy = origin[:, :2] + (torch.rand(n, 2, device=device, generator=g) - 0.5) * torch.tensor([12.0, 5.0], device=device)
        yaw = (torch.rand(n, device=device, generator=g) - 0.5) * 2 * torch.pi
        ground = scanner.ground_height(xy[:, None]).squeeze(1)
        qpos[:, free_q[:2]] = xy
        qpos[:, free_q[2]] = ground + 0.6 + 0.6 * torch.rand(n, device=device, generator=g)
        qpos[:, free_q[3]], qpos[:, free_q[6]] = torch.cos(yaw / 2), torch.sin(yaw / 2)
        qpos[:, free_q[4]] = qpos[:, free_q[5]] = 0.0
        env.sim.data.qpos[:] = qpos
        env.sim.data.qvel[:] = 0
        env.sim.forward()
        env.sim.sense()
        S = reader.state()
        bp, bR, _, _ = mdp.base_of(kin.forward(*S))
        env_scan = sensor.data  # noqa: F841 (refreshed by sense)
        from instinct_mj.tasks.acp.mdp.observations import critic_height_scan

        ref = critic_height_scan(env).view(n, -1)  # pelvis_z - hit_z - 0.5, with the env's miss fill
        mine = exact.height_scan(bp, bR) - 0.5
        smooth = scanner.height_scan(bp, bR) - 0.5
        valid = (ref < 4.4) & (mine < 4.4)  # cells that are not saturated (a saturated cell is a miss / no terrain within range)
        miss_cells += int(((ref >= 4.4) != (mine >= 4.4)).sum())
        total += ref.numel()
        diffs_exact.append((mine - ref).abs()[valid])
        diffs_map.append((smooth - ref).abs()[valid])
    de, dm = torch.cat(diffs_exact), torch.cat(diffs_map)
    print(f"cells {total}: saturation disagreements {miss_cells}")
    print(f"exact scan vs env scanner: max |diff| {de.max():.2e} m, 99.9% {de.quantile(0.999):.2e}, cells > 1 cm: {int((de > 0.01).sum())}")
    print(f"height map scan vs env scanner: max |diff| {dm.max():.2e} m, 99.9% {dm.quantile(0.999):.2e}, cells > 1 cm: {int((dm > 0.01).sum())} ({100 * float((dm > 0.01).float().mean()):.2f} %)")

    # cost per call at the sizes of the runner: pi_batch 1024 states, and a world-model window batch
    for B in (1024, 4096, 16384):
        p = torch.randn(B, 3, device=device) + torch.tensor([float(origin[0, 0]), float(origin[0, 1]), 1.0], device=device)
        R = bR[:1].expand(B, 3, 3).contiguous()
        for fn, name in ((exact.height_scan, "exact rays + gradient map"), (scanner.height_scan, "height map")):
            fn(p, R)
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(20):
                fn(p, R)
            torch.cuda.synchronize()
            print(f"  B={B:6d} {name:26s} {(time.time() - t0) / 20 * 1e3:7.3f} ms ({B * 187 / 1e6:.2f} M rays)")


if __name__ == "__main__":
    main()
