"""Procedural goal-reaching terrains of ACP Stage 3 (paper Supplementary "Goal-reaching terrains", "Terrains and Curriculum").

The arena is a grid of tiles, one tile per (terrain type, difficulty level): 10 levels per type (paper: "ten difficulty levels set by step
height, gap width, or stone spacing"). A tile is a list of collision boxes in the tile frame (x forward, z up, ground top at z = 0) plus the
spawn and goal patches the reset samples from. The tiles are mjlab ``SubTerrainCfg``-style builders of ``difficulty -> geometry`` (difficulty
d = level / 9 in [0, 1]); the geometry is added to the ``terrain`` body of the scene by :meth:`GoalArena.add_to_spec`, so the ray-cast sensors,
the terrain friction and the analytic heightmap see it as any other terrain.

Paper values used verbatim: ladders 50 - 65 deg, rung distance 0.20 - 0.30 m, rung length 1 - 2 m; vaults 1.2 - 2 m wide, 0.16 - 0.22 m thick,
0.47 - 0.53 m high at the top level with at most 5 deg roll, 3 deg pitch, 10 deg yaw; speed cap 0.8 m/s on the stair terrains, 2.0 m/s
elsewhere. Everything else (step heights, gap widths, platform sizes, which terrains a style term applies to, ...) is not given by the paper
and is marked ``# ACP-unspecified``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import mujoco
import numpy as np
import torch

NUM_LEVELS = 10
"""Difficulty levels per terrain type (paper Table S4)."""

TILE_SIZE = (14.0, 6.0)
"""ACP-unspecified: tile extent (x, y) [m]; the tile frame is centered, x is the nominal travel direction."""
HX, HY = TILE_SIZE[0] / 2, TILE_SIZE[1] / 2

BASE_STAND_HEIGHT = 0.78
"""ACP-unspecified: pelvis height above the surface of the standing G1 in its default pose (target base height of a goal)."""
PIT_DEPTH = 1.5
"""ACP-unspecified: depth of the pits under gaps, beams, pallets and stones (the robot falls and the episode terminates)."""
FALL_MARGIN = 0.25
"""ACP-unspecified: the episode terminates when the pelvis is lower than ``z_low - FALL_MARGIN`` (below the lowest walkable surface)."""
FORWARD_HEADING_RANGE = 0.5
"""ACP-unspecified: goal heading = travel heading +- this on the terrains that are crossed in one direction (else uniform)."""

TERRAIN_TYPES = (
    "platform", "stairs_up", "stairs_down", "walled_stairs_up", "walled_stairs_down", "walkup_platform", "gap", "beam", "pallets",
    "double_stones", "discrete_platforms", "box_ascent", "box_descent", "uneven_climb", "ladder", "vault", "getup",
)  # fmt: skip
NUM_TYPES = len(TERRAIN_TYPES)
STAIR_TYPES = ("stairs_up", "stairs_down", "walled_stairs_up", "walled_stairs_down")

ARENA_GEOM_PREFIX = "arena_"


def lerp(a: float, b: float, d: float) -> float:
    return a + (b - a) * d


# ----------------------------------------------------------------------------------------------------------------------------------
# geometry
# ----------------------------------------------------------------------------------------------------------------------------------
@dataclass
class Box:
    """Collision box in the tile frame (``half`` sizes, orientation as a quaternion (w, x, y, z))."""

    pos: np.ndarray
    half: np.ndarray
    quat: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))


@dataclass
class Patch:
    """Rectangle on a surface the reset samples a point from: center (x, y, surface z), half extents along the patch axes, yaw."""

    center: tuple[float, float, float]
    half: tuple[float, float]
    yaw: float = 0.0
    weight: float = 1.0


@dataclass
class GoalTile:
    boxes: list[Box]
    spawn: list[Patch]
    goal: list[Patch]
    face: float = 0.0
    """Nominal heading of the robot at the spawn (the direction towards the goal side of the tile)."""
    z_low: float = 0.0
    """Lowest walkable surface height of the tile."""
    foot: bool = True
    """Dense / sparse foot terrain: the ``flat_orientation`` and ``ang_vel_xy`` style terms apply and a tilt / low pelvis terminates."""
    stumble: bool = True
    slip: bool = True
    getup: bool = False
    lying: bool = False
    heading_any: bool = True
    v_max: float = 2.0
    params: dict = field(default_factory=dict)


def yaw_quat(a: float) -> np.ndarray:
    return np.array([math.cos(a / 2), 0.0, 0.0, math.sin(a / 2)])


def pitch_quat(a: float) -> np.ndarray:
    """Rotation about +y by ``a`` (takes +x to (cos a, 0, -sin a))."""
    return np.array([math.cos(a / 2), 0.0, math.sin(a / 2), 0.0])


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])  # fmt: skip


def euler_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    qx = np.array([math.cos(roll / 2), math.sin(roll / 2), 0.0, 0.0])
    return quat_mul(yaw_quat(yaw), quat_mul(pitch_quat(pitch), qx))


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(q, dtype=np.float64))
    return R.reshape(3, 3)


def block(x0: float, x1: float, y0: float, y1: float, z0: float, z1: float) -> Box:
    """Axis-aligned box spanning [x0, x1] x [y0, y1] x [z0, z1]."""
    return Box(np.array([(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2]), np.array([(x1 - x0) / 2, (y1 - y0) / 2, (z1 - z0) / 2]))


def slab(x0: float, x1: float, z_top: float = 0.0, y0: float = -HY, y1: float = HY, thickness: float = 1.0) -> Box:
    return block(x0, x1, y0, y1, z_top - thickness, z_top)


def pit_floor(x0: float = -HX, x1: float = HX) -> Box:
    return block(x0, x1, -HY, HY, -PIT_DEPTH - 0.5, -PIT_DEPTH)


# ----------------------------------------------------------------------------------------------------------------------------------
# tile builders: (difficulty d in [0, 1], rng) -> GoalTile
# ----------------------------------------------------------------------------------------------------------------------------------
def build_platform(d: float, rng: np.random.Generator) -> GoalTile:
    """Square platform of random orientation (paper: platforms with randomized orientation) to step up onto."""
    h = lerp(0.10, 0.30, d)  # ACP-unspecified
    half = rng.uniform(0.8, 1.2)
    cx, cy, yaw = rng.uniform(1.5, 3.0), rng.uniform(-0.6, 0.6), rng.uniform(-math.pi, math.pi)
    boxes = [slab(-HX, HX), Box(np.array([cx, cy, h - 0.5]), np.array([half, half, 0.5]), yaw_quat(yaw))]
    return GoalTile(boxes, [Patch((-5.0, 0.0, 0.0), (0.6, 1.0))], [Patch((cx, cy, h), (half - 0.35, half - 0.35), yaw)],
                    params=dict(height=h, size=2 * half, yaw=yaw))  # fmt: skip


def build_stairs(d: float, rng: np.random.Generator, walled: bool, down: bool) -> GoalTile:
    """Stairs up to a platform (``down``: from the platform to the ground), optionally bounded by side walls."""
    h = lerp(0.05, 0.20, d)  # ACP-unspecified step height range
    n, tread = 6, 0.30  # ACP-unspecified
    width = 1.4 if walled else 2.0
    x_start, x_top = -1.5, -1.5 + n * tread
    y0, y1 = -width / 2, width / 2
    boxes = [slab(-HX, HX)]
    for k in range(n):
        boxes.append(block(x_start + k * tread, x_start + (k + 1) * tread, y0, y1, 0.0, (k + 1) * h))
    top = n * h
    boxes.append(block(x_top, 4.5, y0 if walled else -1.2, y1 if walled else 1.2, 0.0, top))
    if walled:  # ACP-unspecified wall: 0.2 m thick, 0.9 m above the platform
        boxes.append(block(x_start, 4.5, width / 2, width / 2 + 0.2, -0.5, top + 0.9))
        boxes.append(block(x_start, 4.5, -width / 2 - 0.2, -width / 2, -0.5, top + 0.9))
    tile = GoalTile(boxes, [Patch((-5.0, 0.0, 0.0), (0.6, 1.0))], [Patch((2.4, 0.0, top), (1.5, width / 2 - 0.25))], v_max=0.8,
                    params=dict(step_height=h, num_steps=n, tread=tread, width=width))  # fmt: skip
    return reverse(tile) if down else tile


def build_walkup_platform(d: float, rng: np.random.Generator) -> GoalTile:
    """Platform reached by walking up a 15 deg ramp."""
    h = lerp(0.20, 0.80, d)  # ACP-unspecified
    alpha = math.radians(15.0)
    x_r0 = -1.0
    run = h / math.tan(alpha)
    x_r1 = x_r0 + run
    length, thick = h / math.sin(alpha), 0.3
    nrm = np.array([-math.sin(alpha), math.cos(alpha)])  # normal of the ramp top surface in the (x, z) plane
    mid = np.array([(x_r0 + x_r1) / 2, h / 2]) - 0.5 * thick * nrm
    ramp = Box(np.array([mid[0], 0.0, mid[1]]), np.array([length / 2, 1.0, thick / 2]), pitch_quat(-alpha))
    boxes = [slab(-HX, HX), ramp, block(x_r1, x_r1 + 3.0, -1.0, 1.0, 0.0, h)]
    return GoalTile(boxes, [Patch((-5.0, 0.0, 0.0), (0.6, 1.0))], [Patch((x_r1 + 1.5, 0.0, h), (1.0, 0.7))],
                    params=dict(height=h, ramp_deg=15.0))  # fmt: skip


def build_gap(d: float, rng: np.random.Generator) -> GoalTile:
    w = lerp(0.10, 0.70, d)  # ACP-unspecified gap width range
    boxes = [slab(-HX, -w / 2), slab(w / 2, HX), pit_floor()]
    return GoalTile(boxes, [Patch((-4.5, 0.0, 0.0), (0.8, 1.0))], [Patch((4.5, 0.0, 0.0), (0.8, 1.0))], heading_any=False,
                    params=dict(gap=w))  # fmt: skip


def build_beam(d: float, rng: np.random.Generator) -> GoalTile:
    b = lerp(0.60, 0.20, d)  # ACP-unspecified beam width range
    cy = rng.uniform(-0.3, 0.3)
    boxes = [slab(-HX, -2.0), slab(2.0, HX), pit_floor(), block(-2.0, 2.0, cy - b / 2, cy + b / 2, -0.15, 0.0)]
    return GoalTile(boxes, [Patch((-4.7, 0.0, 0.0), (0.8, 1.0))], [Patch((4.7, 0.0, 0.0), (0.8, 1.0))], heading_any=False,
                    params=dict(width=b, length=4.0))  # fmt: skip


def build_pallets(d: float, rng: np.random.Generator) -> GoalTile:
    """Separated narrow slats across the walking direction."""
    sx, gap = 0.14, lerp(0.05, 0.35, d)  # ACP-unspecified
    n = int(4.0 / (sx + gap))
    boxes = [pit_floor()]
    xs = -2.0 + sx / 2 + np.arange(n) * (sx + gap)
    for x in xs:
        boxes.append(block(x - sx / 2, x + sx / 2, -0.7, 0.7, -0.1, 0.0))
    x_end = float(xs[-1] + sx / 2 + gap)
    boxes += [slab(-HX, -2.0), slab(x_end, HX)]
    return GoalTile(boxes, [Patch((-4.7, 0.0, 0.0), (0.8, 1.0))], [Patch((x_end + 2.2, 0.0, 0.0), (0.8, 1.0))], heading_any=False,
                    params=dict(slat_width=sx, gap=gap, num_slats=n))  # fmt: skip


def build_double_stones(d: float, rng: np.random.Generator) -> GoalTile:
    """Pairs of stepping stones (one per foot) at growing pitch."""
    pitch = lerp(0.40, 0.65, d)  # ACP-unspecified
    n = int(4.0 / pitch) + 1
    boxes = [pit_floor()]
    for k in range(n):
        x = -2.0 + k * pitch + rng.uniform(-0.04, 0.04) * d
        for s in (-0.2, 0.2):
            y = s + rng.uniform(-0.03, 0.03)
            boxes.append(block(x - 0.15, x + 0.15, y - 0.15, y + 0.15, -0.2, 0.0))
    x_end = -2.0 + (n - 1) * pitch + 0.15 + 0.3
    boxes += [slab(-HX, -2.3), slab(x_end, HX)]
    return GoalTile(boxes, [Patch((-4.7, 0.0, 0.0), (0.8, 1.0))], [Patch((x_end + 1.6, 0.0, 0.0), (0.8, 1.0))], heading_any=False,
                    params=dict(pitch=pitch, num_pairs=n))  # fmt: skip


def build_discrete_platforms(d: float, rng: np.random.Generator) -> GoalTile:
    """Separated square platforms of varying height over a pit; the goal is on the last one."""
    k_num, size, gap = 5, 1.0, lerp(0.15, 0.65, d)  # ACP-unspecified
    dz = lerp(0.0, 0.30, d)
    x0 = -3.0
    z, boxes, tops, centers = 0.0, [pit_floor()], [], []
    for k in range(k_num):
        z = float(np.clip(z + rng.uniform(-1.0, 1.0) * dz, -0.3, 0.4)) if k > 0 else 0.0
        cx, cy = x0 + k * (size + gap), rng.uniform(-0.3, 0.3) * d
        sx = 1.4 if k == k_num - 1 else size
        boxes.append(block(cx - sx / 2, cx + sx / 2, cy - size / 2, cy + size / 2, z - 0.4, z))
        tops.append(z), centers.append((cx, cy, sx))
    boxes.append(slab(-HX, x0 - size / 2 - gap))
    cx, cy, sx = centers[-1]
    return GoalTile(boxes, [Patch((-5.6, 0.0, 0.0), (0.6, 1.0))], [Patch((cx, cy, tops[-1]), (sx / 2 - 0.25, size / 2 - 0.25))],
                    z_low=min(0.0, min(tops)), heading_any=False, params=dict(gap=gap, height_step=dz, tops=tops))  # fmt: skip


def build_box_ascent(d: float, rng: np.random.Generator) -> GoalTile:
    h = lerp(0.20, 0.70, d)  # ACP-unspecified box height range
    boxes = [slab(-HX, HX), block(0.0, 2.0, -1.2, 1.2, 0.0, h)]
    return GoalTile(boxes, [Patch((-4.0, 0.0, 0.0), (0.8, 1.0))], [Patch((1.0, 0.0, h), (0.6, 0.8))], foot=False, stumble=False,
                    slip=True, params=dict(height=h))  # fmt: skip


def build_uneven_climb(d: float, rng: np.random.Generator) -> GoalTile:
    """Climbing obstacle with an irregular top (6 x 6 columns of random height)."""
    hb, delta = lerp(0.15, 0.55, d), lerp(0.03, 0.12, d)  # ACP-unspecified
    boxes = [slab(-HX, HX), block(0.0, 2.4, -1.2, 1.2, 0.0, hb - delta)]
    for i in range(6):
        for j in range(6):
            x0, y0 = i * 0.4, -1.2 + j * 0.4
            boxes.append(block(x0, x0 + 0.4, y0, y0 + 0.4, hb - delta, hb - delta + rng.uniform(0.0, delta)))
    return GoalTile(boxes, [Patch((-4.0, 0.0, 0.0), (0.8, 1.0))], [Patch((1.2, 0.0, hb - delta / 2), (0.6, 0.8))], foot=False,
                    stumble=False, slip=True, params=dict(height=hb, roughness=delta))  # fmt: skip


def build_ladder(d: float, rng: np.random.Generator) -> GoalTile:
    """Ladder leaning against a platform: 50 - 65 deg, 0.20 - 0.30 m rung distance, 1 - 2 m rung length (paper)."""
    height = lerp(0.70, 1.50, d)  # ACP-unspecified
    theta = math.radians(rng.uniform(50.0, 65.0))
    pitch, length = rng.uniform(0.20, 0.30), rng.uniform(1.0, 2.0)
    x_t = 2.0
    x_b = x_t - height / math.tan(theta)
    rail_len = height / math.sin(theta)
    u = np.array([math.cos(theta), math.sin(theta)])
    boxes = [slab(-HX, HX), block(x_t, x_t + 3.5, -1.2, 1.2, 0.0, height)]
    for s in (-1, 1):  # side rails 4 cm square
        c = np.array([x_b, 0.0]) + u * rail_len / 2
        boxes.append(Box(np.array([c[0], s * length / 2, c[1]]), np.array([rail_len / 2, 0.02, 0.02]), pitch_quat(-theta)))
    t = pitch
    while t < rail_len - 0.05:  # rungs 4 cm square
        c = np.array([x_b, 0.0]) + u * t
        boxes.append(Box(np.array([c[0], 0.0, c[1]]), np.array([0.02, length / 2, 0.02])))
        t += pitch
    return GoalTile(boxes, [Patch((x_b - 2.2, 0.0, 0.0), (0.5, 0.8))], [Patch((x_t + 1.8, 0.0, height), (0.9, 0.8))], foot=False,
                    stumble=False, slip=False, heading_any=False,
                    params=dict(height=height, inclination_deg=math.degrees(theta), rung_distance=pitch, rung_length=length))  # fmt: skip


def build_vault(d: float, rng: np.random.Generator) -> GoalTile:
    """Low thin wall to vault over with hand support, flanked by high walls so that it cannot be walked around."""
    thick, width = rng.uniform(0.16, 0.22), rng.uniform(1.2, 2.0)
    height = lerp(0.25, 0.50, d) + rng.uniform(-0.03, 0.03) * d  # top level: 0.47 - 0.53 m (paper)
    roll, pitch, yaw = (math.radians(rng.uniform(-m, m)) * d for m in (5.0, 3.0, 10.0))
    boxes = [slab(-HX, HX),
             Box(np.array([0.0, 0.0, (height - 0.5) / 2]), np.array([thick / 2, width / 2, (height + 0.5) / 2]), euler_quat(roll, pitch, yaw))]  # fmt: skip
    boxes += [block(-thick / 2, thick / 2, width / 2, HY, 0.0, 1.2), block(-thick / 2, thick / 2, -HY, -width / 2, 0.0, 1.2)]
    return GoalTile(boxes, [Patch((-3.5, 0.0, 0.0), (0.8, 0.8))], [Patch((3.5, 0.0, 0.0), (0.8, 0.8))], foot=False, stumble=False,
                    slip=False, heading_any=False, params=dict(thickness=thick, width=width, height=height))  # fmt: skip


def build_getup(d: float, rng: np.random.Generator) -> GoalTile:
    """Flat ground, the robot starts lying (face up or face down) and has to stand up at a goal 0 - 2.5 m away."""
    dist = lerp(0.0, 2.5, d)  # ACP-unspecified
    return GoalTile([slab(-HX, HX)], [Patch((-1.5, 0.0, 0.0), (0.4, 0.4))], [Patch((-1.5 + dist, 0.0, 0.0), (0.3, 0.3))], foot=False,
                    stumble=False, slip=False, getup=True, lying=True, params=dict(distance=dist))  # fmt: skip


def reverse(tile: GoalTile) -> GoalTile:
    """Mirror the tile in x and swap spawn and goal: the same obstacle crossed in the opposite sense (stairs down, box descent)."""
    for b in tile.boxes:
        b.pos = b.pos * np.array([-1.0, 1.0, 1.0])
        b.quat = np.array([b.quat[0], b.quat[1], -b.quat[2], -b.quat[3]])  # mirror in x: (w, x, y, z) -> (w, x, -y, -z)
    flip = lambda p: Patch((-p.center[0], p.center[1], p.center[2]), p.half, math.pi - p.yaw, p.weight)  # noqa: E731
    tile.spawn, tile.goal = [flip(p) for p in tile.goal], [flip(p) for p in tile.spawn]
    return tile


def build_box_descent(d: float, rng: np.random.Generator) -> GoalTile:
    return reverse(build_box_ascent(d, rng))


BUILDERS = {
    "platform": build_platform,
    "stairs_up": lambda d, rng: build_stairs(d, rng, walled=False, down=False),
    "stairs_down": lambda d, rng: build_stairs(d, rng, walled=False, down=True),
    "walled_stairs_up": lambda d, rng: build_stairs(d, rng, walled=True, down=False),
    "walled_stairs_down": lambda d, rng: build_stairs(d, rng, walled=True, down=True),
    "walkup_platform": build_walkup_platform,
    "gap": build_gap,
    "beam": build_beam,
    "pallets": build_pallets,
    "double_stones": build_double_stones,
    "discrete_platforms": build_discrete_platforms,
    "box_ascent": build_box_ascent,
    "box_descent": build_box_descent,
    "uneven_climb": build_uneven_climb,
    "ladder": build_ladder,
    "vault": build_vault,
    "getup": build_getup,
}
assert tuple(BUILDERS) == TERRAIN_TYPES


# ----------------------------------------------------------------------------------------------------------------------------------
# analytic surface height of a tile (tests, heightmap cross-check)
# ----------------------------------------------------------------------------------------------------------------------------------
def surface_height(boxes: list[Box], xy: np.ndarray, floor: float = -100.0) -> np.ndarray:
    """Top surface height of the union of the boxes under the points ``xy`` (K, 2) in the box frame (vertical ray from above)."""
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    top = np.full(len(xy), floor)
    for b in boxes:
        R = quat_to_mat(b.quat)
        o = np.concatenate((xy - b.pos[None, :2], np.full((len(xy), 1), -b.pos[2])), -1)  # point at z = 0 relative to the box center
        p0 = o @ R  # R^T (x - c) for row vectors
        dz = R[2]  # R^T e_z: direction of the vertical ray in the box frame
        lo, hi = np.full(len(xy), -np.inf), np.full(len(xy), np.inf)
        ok = np.ones(len(xy), dtype=bool)
        for i in range(3):
            if abs(dz[i]) < 1e-9:
                ok &= np.abs(p0[:, i]) <= b.half[i]
            else:
                t0, t1 = (-b.half[i] - p0[:, i]) / dz[i], (b.half[i] - p0[:, i]) / dz[i]
                lo, hi = np.maximum(lo, np.minimum(t0, t1)), np.minimum(hi, np.maximum(t0, t1))
        ok &= lo <= hi
        top = np.where(ok, np.maximum(top, hi), top)
    return top


# ----------------------------------------------------------------------------------------------------------------------------------
# arena
# ----------------------------------------------------------------------------------------------------------------------------------
@dataclass(kw_only=True)
class GoalArenaCfg:
    seed: int = 0
    """Seed of the tile randomization (the geometry is a deterministic function of the cfg)."""
    origin: tuple[float, float] | None = None
    """World position of the center of the tile (type 0, level 0). ``None`` (default): beside the reference tiles, from their actual grid: the arena
    lies on the +y side of the reference grid rectangle, ``reference_gap`` away from it and centred on x = 0 (levels along x, types along y)."""
    reference_gap: float = 10.0
    """ACP-unspecified: clear distance [m] between the reference tile grid and the arena (the scans reach 3 m, the robot may fall off a tile edge)."""
    types: tuple[str, ...] = TERRAIN_TYPES


def _spec_geom_aabb(geom) -> tuple[np.ndarray, np.ndarray] | None:
    """World xy (lo, hi) of a box / cylinder / capsule geom of a (not yet compiled) spec, ``None`` for any other type."""
    names = {mujoco.mjtGeom.mjGEOM_BOX: "box", mujoco.mjtGeom.mjGEOM_CYLINDER: "cylinder", mujoco.mjtGeom.mjGEOM_CAPSULE: "capsule"}
    kind = names.get(geom.type)
    if kind is None:
        return None
    quat = np.asarray(geom.quat, dtype=np.float64)
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, quat / np.linalg.norm(quat))
    R = R.reshape(3, 3)
    size = np.asarray(geom.size, dtype=np.float64)
    if kind == "box":
        half = np.abs(R) @ size
    else:
        axis = np.abs(R[:, 2])
        half = size[1] * axis + size[0] * (1.0 if kind == "capsule" else np.sqrt(np.clip(1.0 - axis**2, 0.0, 1.0)))
    pos = np.asarray(geom.pos, dtype=np.float64)
    return pos[:2] - half[:2], pos[:2] + half[:2]


class GoalArena:
    """All tiles of the arena: ``tiles[type][level]``; tile id = ``type * NUM_LEVELS + level``.

    ``reference_bounds`` (a callable returning the (lo, hi) xy rectangle of the reference tile grid) places the arena when ``cfg.origin`` is ``None``; it is
    evaluated when the layout is used (spec build, tables), i.e. after the task cfg and the command line have fixed the tile size and count."""

    def __init__(self, cfg: GoalArenaCfg, reference_bounds: Callable[[], tuple[np.ndarray, np.ndarray]] | None = None):
        self.cfg = cfg
        self.types = cfg.types
        self._reference_bounds = reference_bounds
        rng = np.random.default_rng(cfg.seed)
        self.tiles: list[list[GoalTile]] = []
        for name in self.types:
            self.tiles.append([BUILDERS[name](level / (NUM_LEVELS - 1), rng) for level in range(NUM_LEVELS)])

    @property
    def num_types(self) -> int:
        return len(self.types)

    @property
    def origin(self) -> np.ndarray:
        """World xy of the center of the tile (type 0, level 0)."""
        if self.cfg.origin is not None:
            return np.asarray(self.cfg.origin, dtype=np.float64)
        if self._reference_bounds is None:
            raise ValueError("GoalArenaCfg.origin is None: the arena needs the reference tile grid (reference_bounds) to be placed.")
        _, ref_hi = self._reference_bounds()
        x_lo = -NUM_LEVELS * TILE_SIZE[0] / 2  # centred on the x = 0 line of the (origin-centred) reference grid
        y_lo = ref_hi[1] + self.cfg.reference_gap
        return np.array([x_lo + TILE_SIZE[0] / 2, y_lo + TILE_SIZE[1] / 2])

    def tile_center(self, type_idx: int, level: int) -> np.ndarray:
        o = self.origin
        return np.array([o[0] + level * TILE_SIZE[0], o[1] + type_idx * TILE_SIZE[1], 0.0])

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """(lo, hi) xy extent of the arena."""
        o = self.origin
        lo = o - np.array(TILE_SIZE) / 2
        hi = o + np.array([(NUM_LEVELS - 1) * TILE_SIZE[0], (self.num_types - 1) * TILE_SIZE[1]]) + np.array(TILE_SIZE) / 2
        return lo, hi

    def check_disjoint(self, spec: mujoco.MjSpec | None = None) -> None:
        """Assert that the arena rectangle does not intersect the reference tile grid and, with a ``spec``, none of the terrain geoms already in it
        (the reference tiles); AABB test, so it is conservative. Called with the spec before the arena geoms are added."""
        lo, hi = self.bounds
        if self._reference_bounds is not None:
            ref_lo, ref_hi = self._reference_bounds()
            if np.all(lo < ref_hi) and np.all(ref_lo < hi):
                raise AssertionError(f"Goal arena [{lo}, {hi}] overlaps the reference tile grid [{ref_lo}, {ref_hi}].")
        if spec is not None:
            for geom in spec.body("terrain").geoms:
                box = _spec_geom_aabb(geom)
                if box is not None and np.all(lo < box[1]) and np.all(box[0] < hi):
                    raise AssertionError(f"Terrain geom {geom.name!r} at [{box[0]}, {box[1]}] overlaps the goal arena [{lo}, {hi}].")

    def add_to_spec(self, spec: mujoco.MjSpec) -> None:
        """Add the boxes of all tiles to the ``terrain`` body (geoms named ``arena_<n>``)."""
        self.check_disjoint(spec)
        body = spec.body("terrain")
        n = 0
        for i in range(self.num_types):
            for level in range(NUM_LEVELS):
                c = self.tile_center(i, level)
                for b in self.tiles[i][level].boxes:
                    geom = body.add_geom(name=f"{ARENA_GEOM_PREFIX}{n}", type=mujoco.mjtGeom.mjGEOM_BOX, size=b.half, pos=b.pos + c, quat=b.quat)
                    geom.mass = 0
                    geom.rgba[:] = (0.62, 0.64, 0.68, 1.0)
                    n += 1

    def tables(self, device: str) -> "GoalArenaTables":
        return GoalArenaTables(self, device)


class GoalArenaTables:
    """Per-tile sampling data on the device: patches, headings and the style / termination masks."""

    def __init__(self, arena: GoalArena, device: str):
        self.device = device
        T = arena.num_types * NUM_LEVELS
        self.num_tiles = T
        flat = [(i, level, arena.tiles[i][level]) for i in range(arena.num_types) for level in range(NUM_LEVELS)]
        kmax_s = max(len(t.spawn) for _, _, t in flat)
        kmax_g = max(len(t.goal) for _, _, t in flat)

        def patches(attr: str, kmax: int) -> torch.Tensor:
            out = torch.zeros(T, kmax, 7)  # (cx, cy, z, hx, hy, yaw, weight), tile frame
            for n, (_, _, t) in enumerate(flat):
                for k, p in enumerate(getattr(t, attr)):
                    out[n, k] = torch.tensor([*p.center, *p.half, p.yaw, p.weight])
            return out

        to = lambda x: x.to(device)  # noqa: E731
        self.spawn, self.goal = to(patches("spawn", kmax_s)), to(patches("goal", kmax_g))
        self.origin = to(torch.tensor(np.stack([arena.tile_center(i, level) for i, level, _ in flat]), dtype=torch.float32))
        col = lambda f: to(torch.tensor([f(t) for _, _, t in flat]))  # noqa: E731
        self.face = col(lambda t: t.face).float()
        self.z_low = col(lambda t: t.z_low).float()
        self.v_max = col(lambda t: t.v_max).float()
        self.foot, self.stumble, self.slip = col(lambda t: t.foot), col(lambda t: t.stumble), col(lambda t: t.slip)
        self.getup, self.lying, self.heading_any = col(lambda t: t.getup), col(lambda t: t.lying), col(lambda t: t.heading_any)
        self.type_of_tile = to(torch.tensor([i for i, _, _ in flat]))
        self.level_of_tile = to(torch.tensor([level for _, level, _ in flat]))
        self.num_types = arena.num_types

    def tile_id(self, terrain_type: torch.Tensor, level: torch.Tensor) -> torch.Tensor:
        return terrain_type * NUM_LEVELS + level

    def _sample(self, table: torch.Tensor, tile: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Uniform point in a weight-chosen patch of each tile: world xyz (surface z) and the patch yaw."""
        n = len(tile)
        patch = table[tile]  # (n, K, 7)
        k = torch.multinomial(patch[..., 6], 1).squeeze(-1)
        p = patch[torch.arange(n, device=self.device), k]
        u = (2 * torch.rand(n, device=self.device) - 1) * p[:, 3]
        v = (2 * torch.rand(n, device=self.device) - 1) * p[:, 4]
        c, s = torch.cos(p[:, 5]), torch.sin(p[:, 5])
        xyz = self.origin[tile] + torch.stack((p[:, 0] + c * u - s * v, p[:, 1] + s * u + c * v, p[:, 2]), -1)
        return xyz, p[:, 5]

    def sample_spawn(self, tile: torch.Tensor) -> torch.Tensor:
        """World position (n, 3) of the spawn point (surface z) on each tile."""
        return self._sample(self.spawn, tile)[0]

    def sample_goal(self, tile: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """World goal position (n, 3) (x, y, surface z) and heading (n,) on each tile (``heading_any``: uniform, else travel heading +-
        ``FORWARD_HEADING_RANGE``)."""
        xyz, _ = self._sample(self.goal, tile)
        n = len(tile)
        uniform = (2 * torch.rand(n, device=self.device) - 1) * math.pi
        forward = self.face[tile] + (2 * torch.rand(n, device=self.device) - 1) * FORWARD_HEADING_RANGE
        return xyz, torch.where(self.heading_any[tile], uniform, forward)
