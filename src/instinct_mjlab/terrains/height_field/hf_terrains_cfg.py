import copy
from dataclasses import MISSING, dataclass, field
from typing import List
import uuid

import mujoco
import numpy as np
import trimesh
from scipy import ndimage

from mjlab.terrains.terrain_generator import (
    FlatPatchSamplingCfg,
    SubTerrainCfg,
    TerrainGeometry,
    TerrainOutput,
)


def _unwrap_height_field_function(function_obj: object) -> object:
    """Unwrap decorator layers and return the raw height-field callable."""
    raw_function = function_obj
    wrapped_function = getattr(raw_function, "__wrapped__", None)
    while wrapped_function is not None:
        raw_function = wrapped_function
        wrapped_function = getattr(raw_function, "__wrapped__", None)
    return raw_function


def _add_wall_geometries(
    *,
    body: mujoco.MjsBody,
    cfg: "HfTerrainBaseCfg",
    rng: np.random.Generator,
) -> list[TerrainGeometry]:
    """Add optional side walls, matching legacy `generate_wall` semantics."""
    wall_prob = cfg.wall_prob
    if wall_prob is None:
        return []

    wall_height = float(cfg.wall_height)
    wall_thickness = float(cfg.wall_thickness)
    if wall_height <= 0.0 or wall_thickness <= 0.0:
        return []

    min_x, max_x = 0.0, float(cfg.size[0])
    min_y, max_y = 0.0, float(cfg.size[1])
    geoms: list[TerrainGeometry] = []

    # Left wall
    if rng.uniform() < wall_prob[0]:
        left_wall = body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(wall_thickness * 0.5, (max_y - min_y) * 0.5, wall_height * 0.5),
            pos=(min_x - wall_thickness * 0.5, (min_y + max_y) * 0.5, wall_height * 0.5),
        )
        geoms.append(TerrainGeometry(geom=left_wall))

    # Right wall
    if rng.uniform() < wall_prob[1]:
        right_wall = body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(wall_thickness * 0.5, (max_y - min_y) * 0.5, wall_height * 0.5),
            pos=(max_x + wall_thickness * 0.5, (min_y + max_y) * 0.5, wall_height * 0.5),
        )
        geoms.append(TerrainGeometry(geom=right_wall))

    # Front wall
    if rng.uniform() < wall_prob[2]:
        front_wall = body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=((max_x - min_x) * 0.5, wall_thickness * 0.5, wall_height * 0.5),
            pos=((min_x + max_x) * 0.5, min_y - wall_thickness * 0.5, wall_height * 0.5),
        )
        geoms.append(TerrainGeometry(geom=front_wall))

    # Back wall
    if rng.uniform() < wall_prob[3]:
        back_wall = body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=((max_x - min_x) * 0.5, wall_thickness * 0.5, wall_height * 0.5),
            pos=((min_x + max_x) * 0.5, max_y + wall_thickness * 0.5, wall_height * 0.5),
        )
        geoms.append(TerrainGeometry(geom=back_wall))

    return geoms


def _find_flat_patches_from_heightfield_nearest(
    *,
    heights: np.ndarray,
    terrain_size: tuple[float, float],
    horizontal_scale: float,
    z_offset: float,
    cfg: FlatPatchSamplingCfg,
    rng: np.random.Generator,
) -> np.ndarray:
    """Find flat patches on hfield; if range is empty, choose nearest valid patch."""
    heights_work = np.asarray(heights, dtype=np.float64)
    grid_scale = float(horizontal_scale)
    if cfg.grid_resolution is not None and float(cfg.grid_resolution) < grid_scale:
        zoom_factor = grid_scale / float(cfg.grid_resolution)
        heights_work = np.asarray(ndimage.zoom(heights_work, zoom_factor, order=1))
        grid_scale = float(cfg.grid_resolution)

    num_rows, num_cols = heights_work.shape

    if isinstance(cfg.patch_radius, (list, tuple)):
        patch_radius = max(float(radius) for radius in cfg.patch_radius)
    else:
        patch_radius = float(cfg.patch_radius)
    radius_pixels = int(np.ceil(patch_radius / grid_scale))

    y_grid, x_grid = np.ogrid[
        -radius_pixels : radius_pixels + 1,
        -radius_pixels : radius_pixels + 1,
    ]
    footprint = (x_grid**2 + y_grid**2) <= radius_pixels**2

    max_h = ndimage.maximum_filter(
        heights_work,
        footprint=footprint,
        mode="constant",
        cval=-np.inf,
    )
    min_h = ndimage.minimum_filter(
        heights_work,
        footprint=footprint,
        mode="constant",
        cval=np.inf,
    )
    valid_base_mask = (max_h - min_h) <= float(cfg.max_height_diff)
    valid_base_mask[:radius_pixels, :] = False
    valid_base_mask[-radius_pixels:, :] = False
    valid_base_mask[:, :radius_pixels] = False
    valid_base_mask[:, -radius_pixels:] = False

    z_values = heights_work + float(z_offset)
    z_valid = (z_values >= float(cfg.z_range[0])) & (z_values <= float(cfg.z_range[1]))
    valid_base_mask &= z_valid

    x_coords = np.arange(num_cols) * grid_scale
    y_coords = np.arange(num_rows) * grid_scale

    def _convert_range_to_grid(
        value_range: tuple[float, float],
        axis_size: float,
        axis_coords: np.ndarray,
    ) -> tuple[tuple[float, float], float]:
        lo = float(value_range[0])
        hi = float(value_range[1])
        if lo > hi:
            lo, hi = hi, lo

        half_axis = 0.5 * float(axis_size)
        # InstinctLab ranges are center-relative; keep absolute ranges unchanged.
        if lo >= -half_axis - 1.0e-6 and hi <= half_axis + 1.0e-6:
            lo += half_axis
            hi += half_axis

        min_coord = float(axis_coords[0]) if axis_coords.size > 0 else 0.0
        max_coord = float(axis_coords[-1]) if axis_coords.size > 0 else 0.0
        lo = float(np.clip(lo, min_coord, max_coord))
        hi = float(np.clip(hi, min_coord, max_coord))
        if lo > hi:
            lo, hi = hi, lo

        if (hi - lo) < grid_scale:
            target = 0.5 * (lo + hi)
            snapped = float(np.clip(np.round(target / grid_scale) * grid_scale, min_coord, max_coord))
            return (snapped, snapped), snapped

        lo = float(np.clip(np.floor(lo / grid_scale) * grid_scale, min_coord, max_coord))
        hi = float(np.clip(np.ceil(hi / grid_scale) * grid_scale, min_coord, max_coord))
        if lo > hi:
            hi = lo
        return (lo, hi), 0.5 * (lo + hi)

    x_range, target_x = _convert_range_to_grid(cfg.x_range, float(terrain_size[0]), x_coords)
    y_range, target_y = _convert_range_to_grid(cfg.y_range, float(terrain_size[1]), y_coords)

    x_valid = (x_coords >= float(x_range[0])) & (x_coords <= float(x_range[1]))
    y_valid = (y_coords >= float(y_range[0])) & (y_coords <= float(y_range[1]))
    valid_mask = valid_base_mask & y_valid[:, None] & x_valid[None, :]
    valid_indices = np.argwhere(valid_mask)

    if len(valid_indices) == 0:
        target_col = int(np.clip(np.round(target_x / grid_scale), 0, num_cols - 1))
        target_row = int(np.clip(np.round(target_y / grid_scale), 0, num_rows - 1))
        fallback_indices = np.argwhere(valid_base_mask)
        if len(fallback_indices) == 0:
            selected = np.tile(np.array([target_row, target_col], dtype=np.int64), (cfg.num_patches, 1))
        else:
            dist2 = (fallback_indices[:, 0] - target_row) ** 2 + (fallback_indices[:, 1] - target_col) ** 2
            sorted_idx = np.argsort(dist2)
            nearest = fallback_indices[sorted_idx]
            pool_size = max(1, min(len(nearest), cfg.num_patches * 4))
            pool = nearest[:pool_size]
            replace = pool_size < cfg.num_patches
            chosen = rng.choice(pool_size, size=cfg.num_patches, replace=replace)
            selected = pool[chosen]
    else:
        replace = len(valid_indices) < cfg.num_patches
        chosen = rng.choice(len(valid_indices), size=cfg.num_patches, replace=replace)
        selected = valid_indices[chosen]

    x = selected[:, 1] * grid_scale
    y = selected[:, 0] * grid_scale
    z = heights_work[selected[:, 0], selected[:, 1]] + float(z_offset)
    return np.stack([x, y, z], axis=-1)


def _height_field_to_output(
    *,
    heights: np.ndarray,
    cfg: "HfTerrainBaseCfg",
    spec: mujoco.MjSpec,
    rng: np.random.Generator,
) -> TerrainOutput:
    """Convert integer height field to MuJoCo hfield + TerrainOutput."""
    body = spec.body("terrain")
    heights_i16 = np.asarray(np.rint(heights), dtype=np.int16)

    elevation_min = int(np.min(heights_i16))
    elevation_max = int(np.max(heights_i16))
    elevation_range_i16 = elevation_max - elevation_min
    elevation_range_i16 = elevation_range_i16 if elevation_range_i16 > 0 else 1

    normalized_elevation = (heights_i16 - elevation_min) / elevation_range_i16
    max_physical_height = float(elevation_range_i16) * float(cfg.vertical_scale)
    base_thickness = max_physical_height * float(cfg.base_thickness_ratio)
    # MuJoCo hfield top height uses `geom_z + normalized * elevation_range`.
    # `base_thickness` only controls underside thickness and must NOT be
    # subtracted from geom z, otherwise each tile gets an extra depth-dependent
    # downward shift and seam steps appear between neighboring terrains.
    geom_z = float(elevation_min) * float(cfg.vertical_scale)

    unique_id = uuid.uuid4().hex
    hfield = spec.add_hfield(
        name=f"hfield_{unique_id}",
        size=[
            float(cfg.size[0]) / 2.0,
            float(cfg.size[1]) / 2.0,
            max_physical_height,
            base_thickness,
        ],
        nrow=heights_i16.shape[0],
        ncol=heights_i16.shape[1],
        userdata=normalized_elevation.astype(np.float32).flatten().tolist(),
    )

    hfield_geom = body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_HFIELD,
        hfieldname=hfield.name,
        pos=(float(cfg.size[0]) * 0.5, float(cfg.size[1]) * 0.5, geom_z),
    )

    x1 = int((float(cfg.size[0]) * 0.5 - 1.0) / float(cfg.horizontal_scale))
    x2 = int((float(cfg.size[0]) * 0.5 + 1.0) / float(cfg.horizontal_scale))
    y1 = int((float(cfg.size[1]) * 0.5 - 1.0) / float(cfg.horizontal_scale))
    y2 = int((float(cfg.size[1]) * 0.5 + 1.0) / float(cfg.horizontal_scale))
    x1 = max(0, min(x1, heights_i16.shape[0] - 1))
    x2 = max(x1 + 1, min(x2, heights_i16.shape[0]))
    y1 = max(0, min(y1, heights_i16.shape[1] - 1))
    y2 = max(y1 + 1, min(y2, heights_i16.shape[1]))
    origin_z = float(np.max(heights_i16[x1:x2, y1:y2])) * float(cfg.vertical_scale)
    origin = np.array([float(cfg.size[0]) * 0.5, float(cfg.size[1]) * 0.5, origin_z], dtype=np.float64)

    flat_patches: dict[str, np.ndarray] | None = None
    if cfg.flat_patch_sampling is not None:
        heights_phys = (heights_i16.astype(np.float64) - float(elevation_min)) * float(cfg.vertical_scale)
        z_offset = float(elevation_min) * float(cfg.vertical_scale)
        flat_patches = {}
        for patch_name, patch_cfg in cfg.flat_patch_sampling.items():
            flat_patches[patch_name] = _find_flat_patches_from_heightfield_nearest(
                heights=heights_phys,
                terrain_size=cfg.size,
                horizontal_scale=float(cfg.horizontal_scale),
                z_offset=z_offset,
                cfg=patch_cfg,
                rng=rng,
            )

    geometries = [TerrainGeometry(geom=hfield_geom, hfield=hfield)]
    geometries.extend(_add_wall_geometries(body=body, cfg=cfg, rng=rng))
    return TerrainOutput(origin=origin, geometries=geometries, flat_patches=flat_patches)


def _height_field_to_hfield_surface_mesh(
    heights: np.ndarray,
    cfg: "HfTerrainBaseCfg",
) -> trimesh.Trimesh:
    """Convert heights into a mesh sampled on the same XY grid as MuJoCo hfield."""
    heights_f64 = np.asarray(heights, dtype=np.float64)
    nrow, ncol = heights_f64.shape

    x = np.linspace(0.0, float(cfg.size[0]), nrow, dtype=np.float64)
    y = np.linspace(0.0, float(cfg.size[1]), ncol, dtype=np.float64)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    zz = heights_f64 * float(cfg.vertical_scale)

    vertices = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
    faces = np.empty((2 * (nrow - 1) * (ncol - 1), 3), dtype=np.int32)
    cursor = 0
    for row in range(nrow - 1):
        base = row * ncol
        for col in range(ncol - 1):
            ind0 = base + col
            ind1 = ind0 + 1
            ind2 = ind0 + ncol
            ind3 = ind2 + 1
            faces[cursor] = (ind0, ind3, ind1)
            faces[cursor + 1] = (ind0, ind2, ind3)
            cursor += 2
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


@dataclass(kw_only=True)
class HfTerrainBaseCfg(SubTerrainCfg):
    border_width: float = 0.0
    horizontal_scale: float = 0.1
    vertical_scale: float = 0.005
    slope_threshold: float | None = None
    flat_patch_sampling: dict[str, FlatPatchSamplingCfg] | None = None
    base_thickness_ratio: float = 1.0
    wall_prob: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    wall_height: float = 5.0
    wall_thickness: float = 0.05

    def _generate_height_field(self, difficulty: float, cfg_for_gen: "HfTerrainBaseCfg") -> np.ndarray:
        raise NotImplementedError(
            f"{type(self).__name__} must implement _generate_height_field(difficulty, cfg_for_gen)."
        )

    def function(
        self,
        difficulty: float,
        spec: mujoco.MjSpec,
        rng: np.random.Generator,
    ) -> TerrainOutput:
        if self.border_width > 0.0 and self.border_width < self.horizontal_scale:
            raise ValueError(
                f"The border width ({self.border_width}) must be greater than or equal to the"
                f" horizontal scale ({self.horizontal_scale})."
            )

        # NOTE:
        # MuJoCo hfield sampling in mjlab assumes grid counts use
        # `int(size / horizontal_scale)` (no `+1`), and border uses
        # `int(border_width / horizontal_scale)`.
        # Keeping legacy `+1` here introduces seam artifacts between
        # neighboring tiles (boundary samples fall into mismatched cells),
        # which amplifies with deeper terrains.
        width_pixels = int(self.size[0] / self.horizontal_scale)
        length_pixels = int(self.size[1] / self.horizontal_scale)
        border_pixels = int(self.border_width / self.horizontal_scale)
        heights = np.zeros((width_pixels, length_pixels), dtype=np.int16)

        sub_terrain_size = [width_pixels - 2 * border_pixels, length_pixels - 2 * border_pixels]
        sub_terrain_size = [dim * self.horizontal_scale for dim in sub_terrain_size]

        cfg_for_gen = copy.deepcopy(self)
        cfg_for_gen.size = tuple(sub_terrain_size)
        generated = self._generate_height_field(difficulty, cfg_for_gen)
        generated_i16 = np.asarray(np.rint(generated), dtype=np.int16)
        if border_pixels > 0:
            heights[border_pixels:-border_pixels, border_pixels:-border_pixels] = generated_i16
        else:
            heights[:, :] = generated_i16

        output = _height_field_to_output(heights=heights, cfg=self, spec=spec, rng=rng)
        # Keep a mesh representation for virtual-obstacle generation, sampled on
        # the same XY lattice as MuJoCo hfield to avoid edge position drift.
        output.instinct_surface_mesh = _height_field_to_hfield_surface_mesh(heights, self)
        return output


@dataclass(kw_only=True)
class HfPyramidSlopedTerrainCfg(HfTerrainBaseCfg):
    slope_range: tuple[float, float] = MISSING
    platform_width: float = 1.0
    inverted: bool = False


@dataclass(kw_only=True)
class HfInvertedPyramidSlopedTerrainCfg(HfPyramidSlopedTerrainCfg):
    inverted: bool = True


@dataclass(kw_only=True)
class HfPyramidStairsTerrainCfg(HfTerrainBaseCfg):
    step_height_range: tuple[float, float] = MISSING
    step_width: float = MISSING
    platform_width: float = 1.0
    inverted: bool = False


@dataclass(kw_only=True)
class HfInvertedPyramidStairsTerrainCfg(HfPyramidStairsTerrainCfg):
    inverted: bool = True


@dataclass(kw_only=True)
class HfDiscreteObstaclesTerrainCfg(HfTerrainBaseCfg):
    obstacle_height_mode: str = "choice"
    obstacle_width_range: tuple[float, float] = MISSING
    obstacle_height_range: tuple[float, float] = MISSING
    num_obstacles: int = MISSING
    platform_width: float = 1.0


@dataclass(kw_only=True)
class HfWaveTerrainCfg(HfTerrainBaseCfg):
    amplitude_range: tuple[float, float] = MISSING
    num_waves: int = 1


@dataclass(kw_only=True)
class HfSteppingStonesTerrainCfg(HfTerrainBaseCfg):
    stone_height_max: float = MISSING
    stone_width_range: tuple[float, float] = MISSING
    stone_distance_range: tuple[float, float] = MISSING
    holes_depth: float = -10.0
    platform_width: float = 1.0

from . import hf_terrains

_RAW_PERLIN_PLANE_TERRAIN_FN = _unwrap_height_field_function(hf_terrains.perlin_plane_terrain)
_RAW_PERLIN_PYRAMID_SLOPED_TERRAIN_FN = _unwrap_height_field_function(hf_terrains.perlin_pyramid_sloped_terrain)
_RAW_PERLIN_PYRAMID_STAIRS_TERRAIN_FN = _unwrap_height_field_function(hf_terrains.perlin_pyramid_stairs_terrain)
_RAW_PERLIN_DISCRETE_OBSTACLES_TERRAIN_FN = _unwrap_height_field_function(hf_terrains.perlin_discrete_obstacles_terrain)
_RAW_PERLIN_WAVE_TERRAIN_FN = _unwrap_height_field_function(hf_terrains.perlin_wave_terrain)
_RAW_PERLIN_STEPPING_STONES_TERRAIN_FN = _unwrap_height_field_function(hf_terrains.perlin_stepping_stones_terrain)
_RAW_PERLIN_PARAPET_TERRAIN_FN = _unwrap_height_field_function(hf_terrains.perlin_parapet_terrain)
_RAW_PERLIN_GUTTER_TERRAIN_FN = _unwrap_height_field_function(hf_terrains.perlin_gutter_terrain)
_RAW_PERLIN_STAIRS_UP_DOWN_TERRAIN_FN = _unwrap_height_field_function(hf_terrains.perlin_stairs_up_down_terrain)
_RAW_PERLIN_STAIRS_DOWN_UP_TERRAIN_FN = _unwrap_height_field_function(hf_terrains.perlin_stairs_down_up_terrain)
_RAW_PERLIN_TILT_TERRAIN_FN = _unwrap_height_field_function(hf_terrains.perlin_tilt_terrain)
_RAW_PERLIN_TILTED_RAMP_TERRAIN_FN = _unwrap_height_field_function(hf_terrains.perlin_tilted_ramp_terrain)
_RAW_PERLIN_SLOPE_TERRAIN_FN = _unwrap_height_field_function(hf_terrains.perlin_slope_terrain)
_RAW_PERLIN_CROSS_STONE_TERRAIN_FN = _unwrap_height_field_function(hf_terrains.perlin_cross_stone_terrain)
_RAW_PERLIN_SQUARE_GAP_TERRAIN_FN = _unwrap_height_field_function(hf_terrains.perlin_square_gap_terrain)

@dataclass
class WallTerrainCfgMixin:
    border_width: float = 0.0
    horizontal_scale: float = 0.1
    vertical_scale: float = 0.005
    slope_threshold: float | None = None
    flat_patch_sampling: dict[str, FlatPatchSamplingCfg] | None = None
    wall_prob: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])  # Probability of generating walls on [left, right, front, back] sides
    wall_height: float = 5.0  # Height of the walls
    wall_thickness: float = 0.05  # Thickness of the walls

@dataclass(kw_only=True)
class PerlinPlaneTerrainCfg(HfTerrainBaseCfg, WallTerrainCfgMixin):
    flat_patch_sampling: dict[str, FlatPatchSamplingCfg] | None = None

    noise_scale: float | List[float] = 0.05
    noise_frequency: int = 20

    fractal_octaves: int = 2

    fractal_lacunarity: float = 2.0

    fractal_gain: float = 0.25

    centering: bool = False  # If True, the noise will be centered around 0

    def _generate_height_field(self, difficulty: float, cfg_for_gen: HfTerrainBaseCfg) -> np.ndarray:
        return _RAW_PERLIN_PLANE_TERRAIN_FN(difficulty, cfg_for_gen)

@dataclass(kw_only=True)
class PerlinPyramidSlopedTerrainCfg(HfPyramidSlopedTerrainCfg, WallTerrainCfgMixin):
    flat_patch_sampling: dict[str, FlatPatchSamplingCfg] | None = None

    slope_range: tuple[float, float] = MISSING
    platform_width: float = 1.0
    inverted: bool = False
    perlin_cfg: PerlinPlaneTerrainCfg | None = None

    def _generate_height_field(self, difficulty: float, cfg_for_gen: HfTerrainBaseCfg) -> np.ndarray:
        return _RAW_PERLIN_PYRAMID_SLOPED_TERRAIN_FN(difficulty, cfg_for_gen)

@dataclass(kw_only=True)
class PerlinInvertedPyramidSlopedTerrainCfg(HfInvertedPyramidSlopedTerrainCfg, WallTerrainCfgMixin):
    flat_patch_sampling: dict[str, FlatPatchSamplingCfg] | None = None

    slope_range: tuple[float, float] = MISSING
    platform_width: float = 1.0
    inverted: bool = True
    perlin_cfg: PerlinPlaneTerrainCfg | None = None

    def _generate_height_field(self, difficulty: float, cfg_for_gen: HfTerrainBaseCfg) -> np.ndarray:
        return _RAW_PERLIN_PYRAMID_SLOPED_TERRAIN_FN(difficulty, cfg_for_gen)

@dataclass(kw_only=True)
class PerlinPyramidStairsTerrainCfg(HfPyramidStairsTerrainCfg, WallTerrainCfgMixin):
    flat_patch_sampling: dict[str, FlatPatchSamplingCfg] | None = None

    step_height_range: tuple[float, float] = MISSING
    step_width: float = MISSING
    platform_width: float = 1.0
    inverted: bool = False
    perlin_cfg: PerlinPlaneTerrainCfg | None = None

    def _generate_height_field(self, difficulty: float, cfg_for_gen: HfTerrainBaseCfg) -> np.ndarray:
        return _RAW_PERLIN_PYRAMID_STAIRS_TERRAIN_FN(difficulty, cfg_for_gen)

@dataclass(kw_only=True)
class PerlinInvertedPyramidStairsTerrainCfg(HfInvertedPyramidStairsTerrainCfg, WallTerrainCfgMixin):
    flat_patch_sampling: dict[str, FlatPatchSamplingCfg] | None = None

    step_height_range: tuple[float, float] = MISSING
    step_width: float = MISSING
    platform_width: float = 1.0
    inverted: bool = True
    perlin_cfg: PerlinPlaneTerrainCfg | None = None

    def _generate_height_field(self, difficulty: float, cfg_for_gen: HfTerrainBaseCfg) -> np.ndarray:
        return _RAW_PERLIN_PYRAMID_STAIRS_TERRAIN_FN(difficulty, cfg_for_gen)

@dataclass(kw_only=True)
class PerlinDiscreteObstaclesTerrainCfg(HfDiscreteObstaclesTerrainCfg, WallTerrainCfgMixin):
    flat_patch_sampling: dict[str, FlatPatchSamplingCfg] | None = None

    obstacle_height_mode: str = "choice"
    obstacle_width_range: tuple[float, float] = MISSING
    obstacle_height_range: tuple[float, float] = MISSING
    num_obstacles: int = MISSING
    platform_width: float = 1.0
    perlin_cfg: PerlinPlaneTerrainCfg | None = None

    def _generate_height_field(self, difficulty: float, cfg_for_gen: HfTerrainBaseCfg) -> np.ndarray:
        return _RAW_PERLIN_DISCRETE_OBSTACLES_TERRAIN_FN(difficulty, cfg_for_gen)

@dataclass(kw_only=True)
class PerlinWaveTerrainCfg(HfWaveTerrainCfg, WallTerrainCfgMixin):
    flat_patch_sampling: dict[str, FlatPatchSamplingCfg] | None = None

    amplitude_range: tuple[float, float] = MISSING
    num_waves: int = 1
    perlin_cfg: PerlinPlaneTerrainCfg | None = None

    def _generate_height_field(self, difficulty: float, cfg_for_gen: HfTerrainBaseCfg) -> np.ndarray:
        return _RAW_PERLIN_WAVE_TERRAIN_FN(difficulty, cfg_for_gen)

@dataclass(kw_only=True)
class PerlinSteppingStonesTerrainCfg(HfSteppingStonesTerrainCfg, WallTerrainCfgMixin):
    flat_patch_sampling: dict[str, FlatPatchSamplingCfg] | None = None

    stone_height_max: float = MISSING
    stone_width_range: tuple[float, float] = MISSING
    stone_distance_range: tuple[float, float] = MISSING
    holes_depth: float = -10.0
    platform_width: float = 1.0
    perlin_cfg: PerlinPlaneTerrainCfg | None = None

    def _generate_height_field(self, difficulty: float, cfg_for_gen: HfTerrainBaseCfg) -> np.ndarray:
        return _RAW_PERLIN_STEPPING_STONES_TERRAIN_FN(difficulty, cfg_for_gen)

# -- Newly added terrain configurations for parkour terrains-- #
@dataclass(kw_only=True)
class PerlinParapetTerrainCfg(HfTerrainBaseCfg, WallTerrainCfgMixin):
    """Configuration for a parapet terrain, can be used for jump and hurdle tasks."""

    flat_patch_sampling: dict[str, FlatPatchSamplingCfg] | None = None

    parapet_height: tuple[float, float] | float = (0.1, 0.3)
    parapet_length: tuple[float, float] | float = (0.1, 0.3)
    parapet_width: float | None = None
    curved_top_rate: float | None = None
    """The rate to generate curved top. If None, the top will be flat."""
    perlin_cfg: PerlinPlaneTerrainCfg | None = None

    def _generate_height_field(self, difficulty: float, cfg_for_gen: HfTerrainBaseCfg) -> np.ndarray:
        return _RAW_PERLIN_PARAPET_TERRAIN_FN(difficulty, cfg_for_gen)

@dataclass(kw_only=True)
class PerlinGutterTerrainCfg(HfTerrainBaseCfg, WallTerrainCfgMixin):
    """Configuration for a gutter parkour terrain."""

    flat_patch_sampling: dict[str, FlatPatchSamplingCfg] | None = None

    gutter_length: tuple[float, float] | float = (0.5, 1.5)  # the distance between gutters
    gutter_depth: tuple[float, float] | float = (0.1, 0.3)  # the depth of the gutter
    gutter_width: float | None = None  # the length of the gutter
    perlin_cfg: PerlinPlaneTerrainCfg | None = None

    def _generate_height_field(self, difficulty: float, cfg_for_gen: HfTerrainBaseCfg) -> np.ndarray:
        return _RAW_PERLIN_GUTTER_TERRAIN_FN(difficulty, cfg_for_gen)

@dataclass(kw_only=True)
class PerlinStairsUpDownTerrainCfg(HfTerrainBaseCfg, WallTerrainCfgMixin):
    """Configuration for a stairs up and down parkour terrain."""

    flat_patch_sampling: dict[str, FlatPatchSamplingCfg] | None = None

    per_step_height: tuple[float, float] | float = MISSING
    """The height of each step. Could be a fixed value or a range (min, max)."""
    per_step_width: float | None = None
    """The width of each step. If None, it will be equal to the width of the terrain."""
    per_step_length: tuple[float, float] | float = MISSING
    """The length of each step along the y-axis."""
    num_steps: tuple[int, int] | int = MISSING
    """The number of steps. Could be a fixed value or a range (min, max)."""

    platform_length: float = 1.0
    """The length of the platform at the bottom of the stairs."""

    perlin_cfg: PerlinPlaneTerrainCfg | None = None

    def _generate_height_field(self, difficulty: float, cfg_for_gen: HfTerrainBaseCfg) -> np.ndarray:
        return _RAW_PERLIN_STAIRS_UP_DOWN_TERRAIN_FN(difficulty, cfg_for_gen)

@dataclass(kw_only=True)
class PerlinStairsDownUpTerrainCfg(HfTerrainBaseCfg, WallTerrainCfgMixin):
    """Configuration for a stairs down and up parkour terrain."""

    flat_patch_sampling: dict[str, FlatPatchSamplingCfg] | None = None

    per_step_height: tuple[float, float] | float = MISSING
    """The height of each step. Could be a fixed value or a range (min, max)."""
    per_step_width: float | None = None
    """The width of each step. If None, it will be equal to the width of the terrain."""
    per_step_length: tuple[float, float] | float = MISSING
    """The length of each step along the y-axis."""
    num_steps: tuple[int, int] | int = MISSING
    """The number of steps. Could be a fixed value or a range (min, max)."""

    platform_length: float = 1.0
    """The length of the platform at the bottom of the stairs."""

    perlin_cfg: PerlinPlaneTerrainCfg | None = None

    def _generate_height_field(self, difficulty: float, cfg_for_gen: HfTerrainBaseCfg) -> np.ndarray:
        return _RAW_PERLIN_STAIRS_DOWN_UP_TERRAIN_FN(difficulty, cfg_for_gen)

@dataclass(kw_only=True)
class PerlinTiltTerrainCfg(HfTerrainBaseCfg, WallTerrainCfgMixin):
    """Configuration for a tilt terrain."""

    flat_patch_sampling: dict[str, FlatPatchSamplingCfg] | None = None

    wall_height: tuple[float, float] | float = MISSING
    wall_width: float | None = None
    wall_length: tuple[float, float] | float = MISSING
    wall_opening_angle: tuple[float, float] | float = MISSING  # in degrees
    wall_opening_width: tuple[float, float] | float = MISSING
    perlin_cfg: PerlinPlaneTerrainCfg | None = None

    def _generate_height_field(self, difficulty: float, cfg_for_gen: HfTerrainBaseCfg) -> np.ndarray:
        return _RAW_PERLIN_TILT_TERRAIN_FN(difficulty, cfg_for_gen)

@dataclass(kw_only=True)
class PerlinTiltedRampTerrainCfg(HfTerrainBaseCfg, WallTerrainCfgMixin):
    """Configuration for a tilted ramp terrain."""

    flat_patch_sampling: dict[str, FlatPatchSamplingCfg] | None = None

    tilt_angle: tuple[float, float] | float = MISSING  # in degrees
    tilt_height: tuple[float, float] | float = MISSING
    tilt_width: tuple[float, float] | float = MISSING
    tilt_length: tuple[float, float] | float = MISSING
    switch_spacing: tuple[float, float] | float = MISSING
    spacing_curriculum: bool | None = None
    overlap_size: float | None = None
    perlin_cfg: PerlinPlaneTerrainCfg | None = None

    def _generate_height_field(self, difficulty: float, cfg_for_gen: HfTerrainBaseCfg) -> np.ndarray:
        return _RAW_PERLIN_TILTED_RAMP_TERRAIN_FN(difficulty, cfg_for_gen)

@dataclass(kw_only=True)
class PerlinSlopeTerrainCfg(HfTerrainBaseCfg, WallTerrainCfgMixin):
    """Configuration for a slope up and down terrain with a flat ground in the middle."""

    flat_patch_sampling: dict[str, FlatPatchSamplingCfg] | None = None

    slope_angle: tuple[float, float] | float = MISSING  # in degrees
    per_slope_length: tuple[float, float] | float = MISSING
    platform_length: float = 1.0
    slope_width: float | None = None
    up_down: bool | None = None  # If True or None, the slope will be up and down, otherwise it will be down and up.
    perlin_cfg: PerlinPlaneTerrainCfg | None = None

    def _generate_height_field(self, difficulty: float, cfg_for_gen: HfTerrainBaseCfg) -> np.ndarray:
        return _RAW_PERLIN_SLOPE_TERRAIN_FN(difficulty, cfg_for_gen)

@dataclass(kw_only=True)
class PerlinCrossStoneTerrainCfg(HfTerrainBaseCfg, WallTerrainCfgMixin):
    """Configuration for a cross stone terrain."""

    flat_patch_sampling: dict[str, FlatPatchSamplingCfg] | None = None

    stone_size: tuple[float, float] = MISSING
    stone_height: tuple[float, float] | float = MISSING
    stone_spacing: tuple[float, float] | float = MISSING
    ground_depth: float = -0.5
    platform_width: float = 1.5
    xy_random_ratio: float = 0.2
    perlin_cfg: PerlinPlaneTerrainCfg | None = None

    def _generate_height_field(self, difficulty: float, cfg_for_gen: HfTerrainBaseCfg) -> np.ndarray:
        return _RAW_PERLIN_CROSS_STONE_TERRAIN_FN(difficulty, cfg_for_gen)

@dataclass(kw_only=True)
class PerlinSquareGapTerrainCfg(HfTerrainBaseCfg, WallTerrainCfgMixin):
    flat_patch_sampling: dict[str, FlatPatchSamplingCfg] | None = None

    gap_distance_range: tuple[float, float] = (0.1, 0.5)
    gap_depth: tuple[float, float] = (0.2, 0.5)
    platform_width: float = 1.5
    border_width: float = 0.0

    perlin_cfg: PerlinPlaneTerrainCfg | None = None

    def _generate_height_field(self, difficulty: float, cfg_for_gen: HfTerrainBaseCfg) -> np.ndarray:
        return _RAW_PERLIN_SQUARE_GAP_TERRAIN_FN(difficulty, cfg_for_gen)
