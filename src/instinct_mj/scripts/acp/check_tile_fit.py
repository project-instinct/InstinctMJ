"""Does the terrain tile size of the ACP task (``tasks/acp/terrain_layout.TILE_SIZE``) fit the dataset? Also the measurement it is derived from.

In the ``use_input_origin_frame`` coordinates of the terrains (tile centre = env origin = motion origin) it measures

1. the footprint of every terrain's primitive geoms (``<terrain>.geoms.yaml``): axis-aligned boxes are clipped to the tile, rotated boxes / cylinders /
   capsules must fit (``_add_primitive_terrain_geoms``),
2. the horizontal reach of every motion: root, body origins and the bounding spheres of all geoms of the robot over all frames, and with reset noise
   (U(-0.2, 0.2) rad on all joints, every 4th frame, ``--noise-draws`` draws),

then checks: (a) the motion incl. noise stays ``--margin`` (default 1 m) inside its own tile, (b) the geoms the terrain builder really emits for every
terrain lie inside the tile (+ its 2 cm tolerance), so the geometry of a tile never reaches the neighbour, (c) with the actual tile layout of the task the robot
of every motion stays in its own tile rectangle and at least ``--margin`` away from every other tile rectangle, (d) the env starts (reference frame 0 and
any frame, with noise) are inside the tile. Exit status 1 if a check fails.

    python -m instinct_mj.scripts.acp.check_tile_fit [--dataset ~/Datasets/acp/v3_train10] [--out report.json]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import yaml

from instinct_mj.assets.acp_g1 import write_acp_g1_mjcf
from instinct_mj.tasks.acp.terrain_layout import TILE_SIZE, grid_shape, tile_corner
from instinct_mj.terrains.trimesh.mesh_terrains import PRIMITIVE_TERRAIN_TOLERANCE, _add_primitive_terrain_geoms, _geom_aabb_half_extent


def terrain_footprints(dataset: Path, meta: dict) -> list[dict]:
    out = []
    for t in meta["terrains"]:
        lo = np.full(2, np.inf)
        hi = -lo.copy()
        nlo, nhi = lo.copy(), hi.copy()  # non-clippable (rotated box / cylinder / capsule)
        for e in yaml.safe_load((dataset / t["terrain_geoms"]).read_text()):
            pos = np.asarray(e["pos"], float)
            q = np.asarray(e.get("quat", [1, 0, 0, 0]), float)
            q /= np.linalg.norm(q)
            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, q)
            R = R.reshape(3, 3)
            h = _geom_aabb_half_extent(e["type"], np.asarray(e["size"], float), R)[:2]
            lo, hi = np.minimum(lo, pos[:2] - h), np.maximum(hi, pos[:2] + h)
            if not (e["type"] == "box" and np.allclose(R, np.eye(3), atol=1e-6)):
                nlo, nhi = np.minimum(nlo, pos[:2] - h), np.maximum(nhi, pos[:2] + h)
        out.append(dict(lo=lo, hi=hi, nlo=nlo, nhi=nhi))
    return out


def motion_reach(dataset: Path, meta: dict, noise_draws: int, noise: float, seed: int) -> list[dict]:
    model = mujoco.MjModel.from_xml_path(write_acp_g1_mjcf())
    data = mujoco.MjData(model)
    joint_adr = {model.joint(i).name: model.jnt_qposadr[i] for i in range(1, model.njnt)}
    rbound = model.geom_rbound
    rng = np.random.default_rng(seed)
    rows = []
    for mf in meta["motion_files"]:
        z = np.load(dataset / mf["motion_file"], allow_pickle=True)
        cols = [(i, joint_adr[str(n)]) for i, n in enumerate(z["joint_names"]) if str(n) in joint_adr]
        P, Q, J = z["base_pos_w"].astype(float), z["base_quat_w"].astype(float), z["joint_pos"].astype(float)

        def reach(joints, f):
            data.qpos[:3], data.qpos[3:7] = P[f], Q[f]
            for i, a in cols:
                data.qpos[a] = joints[i]
            mujoco.mj_kinematics(model, data)
            g = data.geom_xpos[:, :2]
            return (g - rbound[:, None]).min(0), (g + rbound[:, None]).max(0), data.xpos[1:, :2].min(0), data.xpos[1:, :2].max(0)

        inf = np.full(2, np.inf)
        lo_g, hi_g, lo_b, hi_b, lo_n, hi_n = inf.copy(), -inf, inf.copy(), -inf, inf.copy(), -inf
        for f in range(len(P)):
            a, b, c, d = reach(J[f], f)
            lo_g, hi_g, lo_b, hi_b = np.minimum(lo_g, a), np.maximum(hi_g, b), np.minimum(lo_b, c), np.maximum(hi_b, d)
            if f % 4 == 0:
                for _ in range(noise_draws):
                    a, b, _, _ = reach(J[f] + rng.uniform(-noise, noise, J.shape[1]), f)
                    lo_n, hi_n = np.minimum(lo_n, a), np.maximum(hi_n, b)
        rows.append(dict(file=mf["motion_file"], terrain=int(mf["terrain_id"]), root=(P[:, :2].min(0), P[:, :2].max(0)), body=(lo_b, hi_b),
                         geom=(lo_g, hi_g), noisy=(np.minimum(lo_g, lo_n), np.maximum(hi_g, hi_n)), start=P[0, :2]))  # fmt: skip
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="~/Datasets/acp/v3_train10")
    ap.add_argument("--tile", type=float, nargs=2, default=list(TILE_SIZE))
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--noise", type=float, default=0.2, help="reset joint noise [rad]")
    ap.add_argument("--noise-draws", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    dataset = Path(args.dataset).expanduser()
    meta = yaml.safe_load((dataset / "metadata.yaml").read_text())
    tile = np.array(args.tile)
    half = tile / 2
    fails: list[str] = []

    terr = terrain_footprints(dataset, meta)
    allx = np.abs(np.array([[t["lo"], t["hi"]] for t in terr])).max((0, 1))
    nc = np.array([np.maximum(-t["nlo"], t["nhi"]) for t in terr if np.isfinite(t["nlo"]).all()])
    print(f"{len(terr)} terrains; geom footprint max |x|, |y| = {allx[0]:.2f}, {allx[1]:.2f} m; non-clippable (rotated / round) = {nc.max(0)[0]:.2f}, {nc.max(0)[1]:.2f} m")
    print(f"tile {tile[0]:g} x {tile[1]:g} m (half {half[0]:g}, {half[1]:g})")
    clipped = sum(bool((np.maximum(-t["lo"], t["hi"]) > half).any()) for t in terr)
    print(f"{clipped} terrains have axis-aligned ground / boxes beyond the tile (clipped to it by the builder)")
    if (nc.max(0) > half - 0.0).any():
        fails.append("a rotated / round primitive exceeds the tile")

    rows = motion_reach(dataset, meta, args.noise_draws, args.noise, args.seed)
    worst = {k: np.max([np.maximum(-r[k][0], r[k][1]) for r in rows], 0) for k in ("root", "body", "geom", "noisy")}
    for k, v in worst.items():
        print(f"motion reach |x|, |y| ({k:5s}): {v[0]:6.2f}, {v[1]:6.2f} m")
    need = worst["noisy"] + args.margin
    print(f"=> tile half extents needed with the {args.margin:g} m margin: {need[0]:.2f}, {need[1]:.2f} m (tile >= {2 * need[0]:.2f} x {2 * need[1]:.2f}); rounded up to 0.5 m: {math.ceil(2 * need[0]) / 2:g}, {math.ceil(2 * need[1]) / 2:g}")
    slack = half - worst["noisy"]
    print(f"margin of the tile over the farthest reach: {slack[0]:.2f}, {slack[1]:.2f} m")
    if (slack < args.margin - 1e-9).any():
        fails.append(f"margin {slack} < {args.margin}")

    # (b) the geometry the builder really emits for every terrain lies in the tile (so that it never reaches the neighbouring tile)
    cfg = SimpleNamespace(size=tuple(tile))
    emitted_max = np.zeros(2)
    for t in meta["terrains"]:
        spec = mujoco.MjSpec()
        spec.worldbody.add_body(name="terrain")
        for g in _add_primitive_terrain_geoms(cfg, spec, str(dataset / t["terrain_geoms"])):
            pass
        for g in spec.body("terrain").geoms:
            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, np.asarray(g.quat))
            kind = {mujoco.mjtGeom.mjGEOM_BOX: "box", mujoco.mjtGeom.mjGEOM_CYLINDER: "cylinder", mujoco.mjtGeom.mjGEOM_CAPSULE: "capsule"}[g.type]
            n = {"box": 3, "cylinder": 2, "capsule": 2}[kind]
            h = _geom_aabb_half_extent(kind, np.asarray(g.size)[:n], R.reshape(3, 3))[:2]
            c = np.asarray(g.pos)[:2] - half  # tile-local (the builder adds the tile-centre offset)
            emitted_max = np.maximum(emitted_max, np.maximum(-(c - h), c + h) - half)
    print(f"emitted terrain geoms exceed the tile by at most {emitted_max[0]:.3f}, {emitted_max[1]:.3f} m (tolerance {PRIMITIVE_TERRAIN_TOLERANCE} m)")
    if (emitted_max > PRIMITIVE_TERRAIN_TOLERANCE + 1e-6).any():
        fails.append("emitted terrain geometry beyond the tile")

    # (c) / (d) the actual layout: one tile per terrain, tile r = terrain r, robot AABB (with noise) in the own tile and away from the others
    T = len(meta["terrains"])
    corners = np.array([tile_corner(i, T, tuple(tile)) for i in range(T)])
    nx, ny = grid_shape(T, tuple(tile))
    centres = corners + half
    print(f"layout: {T} tiles on a {nx} x {ny} grid; bbox x [{corners[:, 0].min():.1f}, {corners[:, 0].max() + tile[0]:.1f}] y [{corners[:, 1].min():.1f}, {corners[:, 1].max() + tile[1]:.1f}]; "
          f"farthest tile centre {np.linalg.norm(centres, axis=1).max():.1f} m from the origin")  # fmt: skip
    min_gap = np.inf
    for r in rows:
        lo, hi = r["noisy"][0] + centres[r["terrain"]], r["noisy"][1] + centres[r["terrain"]]
        own_lo, own_hi = corners[r["terrain"]], corners[r["terrain"]] + tile
        if (lo < own_lo).any() or (hi > own_hi).any():
            fails.append(f"{r['file']}: robot leaves its tile")
        if not (np.abs(r["start"]) < half).all():
            fails.append(f"{r['file']}: start outside the tile")
        other = np.arange(T) != r["terrain"]
        gap_x = np.maximum(corners[other, 0] - hi[0], lo[0] - (corners[other, 0] + tile[0]))
        gap_y = np.maximum(corners[other, 1] - hi[1], lo[1] - (corners[other, 1] + tile[1]))
        min_gap = min(min_gap, float(np.maximum(gap_x, gap_y).min()))
    print(f"all {len(rows)} motions (reset noise incl.) stay in their tile; smallest clearance to any other tile rectangle: {min_gap:.2f} m")
    if min_gap < args.margin - 1e-9:
        fails.append(f"clearance {min_gap:.2f} < {args.margin}")

    if args.out:
        Path(args.out).write_text(json.dumps(dict(tile=tile.tolist(), reach={k: v.tolist() for k, v in worst.items()}, min_clearance=min_gap, failures=fails)))
    print("FAILED: " + "; ".join(fails) if fails else "OK")
    raise SystemExit(1 if fails else 0)


if __name__ == "__main__":
    main()
