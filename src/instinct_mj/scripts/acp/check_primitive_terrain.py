"""Check of primitive terrain geoms (``terrain_geoms`` in the dataset metadata): builds a synthetic dataset (floor, stairs, ladder rungs, a box,
a capsule as ``<name>.geoms.yaml``), loads the ACP env on it and compares the model's rays, the env's ray-cast scanner and the analytic SuperMimic
heightmap with the analytic heights of the primitives.

    python -m instinct_mj.scripts.acp.check_primitive_terrain [--dataset-dir /tmp/acp_geoms_ds] [--source-dataset ~/Datasets/acp/acp_motions_v1_50hz]
"""

from __future__ import annotations

import argparse
import math
import os
import shutil

import mujoco
import numpy as np
import torch
import yaml

import instinct_mj.tasks  # noqa: F401  (registers the tasks)
from instinct_mj.rl import InstinctRlVecEnvWrapper
from instinct_mj.rl.acp.sim import build_mdp
from instinct_mj.tasks.registry import load_env_cfg, load_env_cls

TASK = "Instinct-ACP-Track-G1-v1"
S45 = math.sqrt(0.5)


def synthetic_geoms() -> tuple[list[dict], dict[str, tuple[float, float, float]]]:
    """Primitives in the frame of the terrain STL (origin at the tile centre) and probe points (x, y, expected top z)."""
    geoms = [{"type": "box", "pos": [0.0, 0.0, -0.05], "quat": [1, 0, 0, 0], "size": [20.0, 20.0, 0.05]}]  # floor, larger than the tile
    probes = {"floor": (-3.0, 0.5, 0.0)}
    for i in range(4):  # stairs: 0.3 m deep, 0.1 m rise
        top = 0.1 * (i + 1)
        geoms.append({"type": "box", "pos": [2.0 + 0.3 * i, 0.0, top / 2], "quat": [1, 0, 0, 0], "size": [0.15, 1.0, top / 2]})
        probes[f"stair{i}"] = (2.0 + 0.3 * i, 0.3, top)
    for i in range(7):  # ladder rungs (cylinders along y, 1.5 m wide, 0.02 m radius), 50 degree climb
        x, z = 5.0 + 0.161 * i, 0.2 + 0.192 * i
        geoms.append({"type": "cylinder", "pos": [x, 0.0, z], "quat": [S45, S45, 0, 0], "size": [0.02, 0.75]})
        probes[f"rung{i}"] = (x, 0.2, z + 0.02)
    geoms.append({"type": "box", "pos": [9.0, 0.0, 0.25], "quat": [1, 0, 0, 0], "size": [1.0, 1.0, 0.25]})
    probes["box"] = (9.0, 0.4, 0.5)
    geoms.append({"type": "capsule", "pos": [-6.0, 0.0, 0.1], "quat": [S45, S45, 0, 0], "size": [0.1, 0.5]})  # capsule along y, top at 0.2
    probes["capsule"] = (-6.0, 0.2, 0.2)
    return geoms, probes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="/tmp/acp_geoms_ds")
    parser.add_argument("--source-dataset", default=os.path.expanduser("~/Datasets/acp/acp_motions_v1_50hz"))
    args = parser.parse_args()

    # a one-clip dataset whose terrain is given as primitive geoms (the STL stays as the viewing / labeling terrain)
    shutil.rmtree(args.dataset_dir, ignore_errors=True)
    os.makedirs(args.dataset_dir)
    meta = yaml.safe_load(open(os.path.join(args.source_dataset, "metadata.yaml")))
    entry = next(e for e in meta["motion_files"] if "walk" in e["motion_file"])
    terrain = next(t for t in meta["terrains"] if t["terrain_id"] == entry["terrain_id"])
    for rel in (entry["motion_file"], entry["motion_file"].replace(".npz", ".contacts.npz"), terrain["terrain_file"]):
        os.makedirs(os.path.dirname(os.path.join(args.dataset_dir, rel)), exist_ok=True)
        shutil.copy(os.path.join(args.source_dataset, rel), os.path.join(args.dataset_dir, rel))
    geoms, probes = synthetic_geoms()
    yaml.safe_dump(geoms, open(os.path.join(args.dataset_dir, "synthetic.geoms.yaml"), "w"))
    yaml.safe_dump(
        {"motion_files": [{"motion_file": entry["motion_file"], "terrain_id": 0}],
         "terrains": [{"terrain_id": 0, "terrain_file": terrain["terrain_file"], "terrain_geoms": "synthetic.geoms.yaml"}]},
        open(os.path.join(args.dataset_dir, "metadata.yaml"), "w"),
    )  # fmt: skip

    import instinct_mj.tasks.acp.config.g1.acp_tracking_cfg as acp_g1

    buffer = acp_g1.motion_reference_cfg.motion_buffers["TerrainMotion"]
    buffer.path, buffer.metadata_yaml = args.dataset_dir, os.path.join(args.dataset_dir, "metadata.yaml")
    device = "cuda:0"
    cfg = load_env_cfg(TASK)
    cfg.scene.num_envs = 4
    env = load_env_cls(TASK)(cfg=cfg, device=device)
    vec = InstinctRlVecEnvWrapper(env, policy_group="policy", critic_group="critic")
    vec.reset()
    model = env.sim.mj_model
    terrain_body = model.body("terrain").id
    geom_ids = np.nonzero(model.geom_bodyid == terrain_body)[0]
    types = [mujoco.mjtGeom(model.geom_type[g]).name for g in geom_ids]
    print(f"terrain body geoms: {len(geom_ids)} {sorted(set(types))} (expected {len(geoms)}); groups {sorted(set(model.geom_group[geom_ids].tolist()))}; "
          f"friction {sorted(set(model.geom_friction[geom_ids, 0].round(3).tolist()))}")
    assert len(geom_ids) == len(geoms) and all(t in ("mjGEOM_BOX", "mjGEOM_CYLINDER", "mjGEOM_CAPSULE") for t in types)
    assert set(model.geom_group[geom_ids].tolist()) == {0}
    floor = geom_ids[0]
    print("floor box half extents after clipping to the 30 x 16 m tile:", model.geom_size[floor].round(3).tolist())

    # tile placement: the origin of the tile (centre of the footprint = the STL frame origin) in the world
    tile_origin = env.scene.terrain.terrain_origins[0, 0].cpu().numpy()
    print("tile origin (spawn origin = STL frame origin):", tile_origin.round(3))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    group = np.zeros(6, np.uint8)
    group[0] = 1
    errs = {}
    for name, (x, y, top) in probes.items():
        origin = np.array([tile_origin[0] + x, tile_origin[1] + y, 3.0])
        geomid = np.zeros(1, np.int32)
        dist = mujoco.mj_ray(model, data, origin, np.array([0.0, 0.0, -1.0]), group, 1, -1, geomid)
        errs[name] = (origin[2] - dist) - (tile_origin[2] + top)
    worst = max(errs, key=lambda k: abs(errs[k]))
    print(f"model ray casts vs analytic tops: max |err| {abs(errs[worst]):.2e} at {worst}")
    assert abs(errs[worst]) < 1e-4

    # analytic heightmap of the ACP runner (mujoco ray casts on the terrain body): resolution 0.05 m
    motion, kin, scanner, mdp = build_mdp(env, 0.05, "/tmp/acp_geoms_heightmaps", device)
    print("heightmap (resolution 0.05 m):")
    for name, (x, y, top) in probes.items():
        xy = torch.tensor([[[tile_origin[0] + x, tile_origin[1] + y]]], device=device, dtype=torch.float32)
        h = float(scanner.ground_height(xy)[0, 0]) - tile_origin[2]
        print(f"  {name:8s} expected {top:6.3f} heightmap {h:6.3f}" + ("   <- thin rung between grid cells" if name.startswith("rung") and abs(h - top) > 0.01 else ""))

    # the env's ray-cast scanner (critic scan, true pelvis pose) vs the model's rays at its ray origins
    sensor = env.scene["critic_height_scanner"]
    dist_env = sensor.data.distances[0].cpu().numpy()
    origins = sensor._cached_world_origins[0].cpu().numpy()
    ref = np.zeros_like(dist_env)
    for k, o in enumerate(origins):
        geomid = np.zeros(1, np.int32)
        ref[k] = mujoco.mj_ray(model, data, o.astype(np.float64), np.array([0.0, 0.0, -1.0]), group, 1, -1, geomid)
    ok = (dist_env >= 0) & (ref >= 0)
    print(f"scanner rays vs model rays: {int(ok.sum())} / {len(ref)} hit by both, max |diff| {np.abs(dist_env[ok] - ref[ok]).max():.2e} m")
    print("OK")


if __name__ == "__main__":
    main()
