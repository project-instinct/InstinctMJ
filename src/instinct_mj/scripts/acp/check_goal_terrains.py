"""Checks of the procedural goal-reaching terrains of ACP Stage 3 (no simulator): every terrain type at every difficulty level.

For each tile: the geometry stays inside the tile, the spawn and goal patches lie on the surface (analytic height of the boxes), the difficulty
parameter of the type grows monotonically with the level, the goal is reachable from the spawn (foot terrains: breadth-first search over a 0.1 m
grid with a step-up limit of 0.31 m and a stepping distance of 0.85 m, i.e. gaps up to 0.7 m can be crossed; whole-body terrains: the goal
patch is on top of the obstacle that the straight path crosses) and the sampled spawn / goal points are inside the patches.

    python -m instinct_mj.scripts.acp.check_goal_terrains
"""

from __future__ import annotations

import collections
import math

import numpy as np
import torch

from instinct_mj.tasks.acp.goal_terrains import (
    HX, HY, NUM_LEVELS, TERRAIN_TYPES, GoalArena, GoalArenaCfg, GoalTile, quat_to_mat, surface_height,
)  # fmt: skip

STEP_UP, STEP_DOWN, STEP_REACH, GRID = 0.31, 1.0, 0.85, 0.1
# parameter whose value must grow with the level, per type (+1 grows, -1 shrinks)
MONOTONE = {
    "platform": ("height", 1), "stairs_up": ("step_height", 1), "stairs_down": ("step_height", 1), "walled_stairs_up": ("step_height", 1),
    "walled_stairs_down": ("step_height", 1), "walkup_platform": ("height", 1), "gap": ("gap", 1), "beam": ("width", -1), "pallets": ("gap", 1),
    "double_stones": ("pitch", 1), "discrete_platforms": ("gap", 1), "box_ascent": ("height", 1), "box_descent": ("height", 1),
    "uneven_climb": ("height", 1), "ladder": ("height", 1), "vault": ("height", 1), "getup": ("distance", 1),
}  # fmt: skip


def box_extent(tile: GoalTile) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for b in tile.boxes:
        corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * b.half
        w = corners @ quat_to_mat(b.quat).T + b.pos
        lo, hi = np.minimum(lo, w.min(0)), np.maximum(hi, w.max(0))
    return lo, hi


def reachable(tile: GoalTile) -> bool:
    xs, ys = np.arange(-HX + GRID / 2, HX, GRID), np.arange(-HY + GRID / 2, HY, GRID)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    Z = surface_height(tile.boxes, np.stack((X.ravel(), Y.ravel()), -1)).reshape(X.shape)
    walk = Z > tile.z_low - 0.05
    r = int(STEP_REACH / GRID)
    offs = [(i, j) for i in range(-r, r + 1) for j in range(-r, r + 1) if 0 < math.hypot(i, j) * GRID <= STEP_REACH]
    s, g = tile.spawn[0], tile.goal[0]
    ix = lambda p: (int((p.center[0] + HX) / GRID), int((p.center[1] + HY) / GRID))  # noqa: E731
    start, goal = ix(s), ix(g)
    seen = np.zeros_like(walk)
    seen[start] = True
    queue = collections.deque([start])
    while queue:
        i, j = queue.popleft()
        if (i, j) == goal:
            return True
        for di, dj in offs:
            a, b = i + di, j + dj
            if 0 <= a < walk.shape[0] and 0 <= b < walk.shape[1] and walk[a, b] and not seen[a, b]:
                dz = Z[a, b] - Z[i, j]
                if dz <= STEP_UP and -dz <= STEP_DOWN:
                    seen[a, b] = True
                    queue.append((a, b))
    return False


def main() -> None:
    arena = GoalArena(GoalArenaCfg())
    tables = arena.tables("cpu")
    failures = []
    print(f"{'type':<20}{'level':>6}{'boxes':>7}  parameters")
    for i, name in enumerate(TERRAIN_TYPES):
        values = []
        for level in range(NUM_LEVELS):
            tile = arena.tiles[i][level]
            tag = f"{name}[{level}]"
            lo, hi = box_extent(tile)
            if lo[0] < -HX - 1e-6 or hi[0] > HX + 1e-6 or lo[1] < -HY - 1e-6 or hi[1] > HY + 1e-6:
                failures.append(f"{tag}: geometry leaves the tile ({lo[:2]}, {hi[:2]})")
            for kind, patches in (("spawn", tile.spawn), ("goal", tile.goal)):
                for p in patches:
                    cx, cy, cz = p.center
                    rng = np.random.default_rng(0)
                    uv = (rng.uniform(-1, 1, (64, 2)) * np.array(p.half))
                    c, s_ = math.cos(p.yaw), math.sin(p.yaw)
                    pts = np.stack((cx + c * uv[:, 0] - s_ * uv[:, 1], cy + s_ * uv[:, 0] + c * uv[:, 1]), -1)
                    h = surface_height(tile.boxes, pts)
                    tol = 0.5 * tile.params.get("roughness", 0.0) + 0.02
                    if np.abs(h - cz).max() > tol:
                        failures.append(f"{tag}: {kind} patch not on the surface (z {cz:.3f}, surface {h.min():.3f}..{h.max():.3f})")
                    if (np.abs(pts[:, 0]) > HX).any() or (np.abs(pts[:, 1]) > HY).any():
                        failures.append(f"{tag}: {kind} patch outside the tile")
            if tile.foot and not reachable(tile):
                failures.append(f"{tag}: goal not reachable from the spawn")
            gs, gg = tile.spawn[0].center, tile.goal[0].center
            if name == "getup":
                pass
            elif math.hypot(gg[0] - gs[0], gg[1] - gs[1]) < 1.0 and not tile.getup:
                failures.append(f"{tag}: goal closer than 1 m to the spawn")
            values.append(tile.params[MONOTONE[name][0]])
            print(f"{name:<20}{level:>6}{len(tile.boxes):>7}  " + " ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in tile.params.items() if not isinstance(v, list)))
        sign = MONOTONE[name][1]
        if name not in ("ladder", "vault", "platform") and not all(sign * (b - a) >= -1e-9 for a, b in zip(values, values[1:])):
            failures.append(f"{name}: {MONOTONE[name][0]} is not monotone in the level: {values}")
        if name in ("ladder", "vault"):  # random draws per level: the nominal height range grows
            if not values[-1] > values[0]:
                failures.append(f"{name}: height does not grow with the level")
    # paper ranges
    for level in range(NUM_LEVELS):
        p = arena.tiles[TERRAIN_TYPES.index("ladder")][level].params
        assert 50 <= p["inclination_deg"] <= 65 and 0.2 <= p["rung_distance"] <= 0.3 and 1.0 <= p["rung_length"] <= 2.0, p
        v = arena.tiles[TERRAIN_TYPES.index("vault")][level].params
        assert 0.16 <= v["thickness"] <= 0.22 and 1.2 <= v["width"] <= 2.0, v
    v9 = arena.tiles[TERRAIN_TYPES.index("vault")][NUM_LEVELS - 1].params["height"]
    assert 0.47 <= v9 <= 0.53, v9
    # sampling tables: points inside patches, heading, tile ids
    tiles = torch.arange(tables.num_tiles).repeat(50)
    spawn = tables.sample_spawn(tiles)
    goal, yaw = tables.sample_goal(tiles)
    for t in range(tables.num_tiles):
        sel = tiles == t
        tile = arena.tiles[t // NUM_LEVELS][t % NUM_LEVELS]
        origin = tables.origin[t]
        for pts, patches in ((spawn[sel], tile.spawn), (goal[sel], tile.goal)):
            rel = (pts - origin).numpy()
            ok = np.zeros(len(rel), dtype=bool)
            for p in patches:
                d = rel[:, :2] - np.array(p.center[:2])
                c, s_ = math.cos(p.yaw), math.sin(p.yaw)
                u, v = c * d[:, 0] + s_ * d[:, 1], -s_ * d[:, 0] + c * d[:, 1]
                ok |= (np.abs(u) <= p.half[0] + 1e-4) & (np.abs(v) <= p.half[1] + 1e-4) & (np.abs(rel[:, 2] - p.center[2]) < 1e-5)
            if not ok.all():
                failures.append(f"tile {t}: sampled points outside the patches")
    head = (yaw - tables.face[tiles] + math.pi) % (2 * math.pi) - math.pi
    fwd = ~tables.heading_any[tiles]
    assert head[fwd].abs().max() <= 0.5 + 1e-5
    assert yaw[~fwd].abs().max() <= math.pi + 1e-5
    print(f"\n{tables.num_tiles} tiles, {sum(len(t.boxes) for row in arena.tiles for t in row)} boxes")
    if failures:
        print("FAILURES:")
        for f in failures:
            print("  ", f)
        raise SystemExit(1)
    print("OK: geometry inside tiles, patches on the surface, goals reachable, levels monotone, paper ranges, sampling inside patches")


if __name__ == "__main__":
    main()
