"""Tile size and compact tile layout of the ACP terrain.

Tile size
---------
``TILE_SIZE`` is derived from the ACP dataset itself (``scripts/acp/check_tile_fit.py`` re-measures it): in the ``use_input_origin_frame`` coordinates of the
terrains (tile centre = env origin = motion origin) the whole-body collision/visual geoms of all 345 motions of ``v3_train10`` / ``v3_train11`` plus the
reset joint noise (U(-0.2, 0.2) rad) reach |x| <= 13.45 m and |y| <= 7.87 m. The tile half extents are these values plus a margin of ~1 m, rounded up to
0.5 m: 14.5 x 9.0, i.e. 29 x 18 m (the former 30 x 16 m of the perceptive-shadowing scene was cut at y = +-8 m, below the reach of the widest turning clips).
The terrain primitives (``<terrain>.geoms.yaml``) reach at most 14.82 m (x) and 10.40 m (y), but only the axis-aligned ones (ground slabs, large boxes)
exceed the tile, which ``_add_primitive_terrain_geoms`` clips to the tile footprint like the STL crop; every rotated box / cylinder / capsule (non-clippable)
lies within 8.29 x 4.53 m, inside the tile. The margin keeps a robot that drifts from the reference inside its own tile, and the clipped geometry of a tile
never reaches into the neighbouring one.

Layout
------
The ACP task needs one tile per terrain (hundreds of them) and uses the curriculum layout, which is ONE column of ``num_rows`` tiles along x centred at
the origin: 334 tiles of 30 m put the outer terrains 5 km away from the world origin. The simulator is float32, and at 3-5 km (float32 spacing 0.25-0.5 mm) the
convex collision detection (GJK / EPA) of mujoco_warp returns contacts whose point is hundreds of metres off (EPA horizon warnings; the contact normal and distance
stay plausible). The lever arm of such a contact makes the constraint reference ``efc.aref`` ~1e5-1e6 and the solver returns NaN (the ``[collect] non-finite
simulator state`` of the ladder / vault clips). This generator lays the same tiles (same order, same origins array ``[row, 0]``) on a near-square 2-D grid
instead (as long in x as in y, in metres), so the farthest tile centre is about 300 m from the origin (the half diagonal of the grid). Only the world position
of the tiles changes: everything the task uses is relative to the env origin. ``grid_bounds`` is the one place that knows the grid rectangle (the Stage 3
arena is placed from it).
"""

from __future__ import annotations

import math

import numpy as np

from instinct_mj.terrains.terrain_generator import FiledTerrainGenerator


TILE_SIZE = (29.0, 18.0)
"""Terrain tile (x, y) [m], see the module docstring."""
TILE_BORDER_WIDTH = 0.0
"""No border around the tiles (the tiles are adjacent)."""


def grid_shape(num_tiles: int, size: tuple[float, float]) -> tuple[int, int]:
    """(nx, ny) tiles of the near-square grid of ``num_tiles`` tiles of ``size`` (about as long in x as in y, in metres)."""
    nx = max(1, math.ceil(math.sqrt(num_tiles * size[1] / size[0])))
    return nx, math.ceil(num_tiles / nx)


def tile_corner(index: int, num_tiles: int, size: tuple[float, float]) -> np.ndarray:
    """World (x, y) of the lower corner of tile ``index`` (row-major, grid centred at the origin)."""
    nx, ny = grid_shape(num_tiles, size)
    return np.array([(index % nx - nx / 2) * size[0], (index // nx - ny / 2) * size[1]])


def grid_bounds(num_tiles: int, size: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """(lo, hi) xy rectangle of the tile grid (the full ``nx x ny`` rectangle, also where the last row is not filled)."""
    nx, ny = grid_shape(num_tiles, size)
    half = np.array([nx * size[0], ny * size[1]]) / 2
    return -half, half


class CompactFiledTerrainGenerator(FiledTerrainGenerator):
    """``FiledTerrainGenerator`` whose tiles of the single curriculum column are wrapped into a near-square grid (row-major, centred at the origin)."""

    def _get_sub_terrain_position(self, row: int, col: int) -> np.ndarray:
        n = self.cfg.num_rows * self._num_cols
        index = row + col * self.cfg.num_rows
        return np.array([*tile_corner(index, n, tuple(self.cfg.size)), 0.0])
