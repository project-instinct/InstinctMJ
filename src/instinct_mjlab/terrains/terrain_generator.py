from __future__ import annotations

import copy
from dataclasses import dataclass
import inspect
import mujoco
import numpy as np
import torch
import trimesh
from typing import TYPE_CHECKING, Literal

from mjlab.terrains import SubTerrainCfg as SubTerrainBaseCfg
from mjlab.terrains import TerrainGenerator

if TYPE_CHECKING:
    from .terrain_generator_cfg import FiledTerrainGeneratorCfg


def _resolve_patch_radii(
    patch_radius: float | list[float] | tuple[float, ...],
) -> list[float]:
    """Normalize patch radius config into a non-empty list."""
    if isinstance(patch_radius, (list, tuple)):
        patch_radii = [float(radius) for radius in patch_radius]
    else:
        patch_radii = [float(patch_radius)]
    if len(patch_radii) == 0:
        raise ValueError("patch_radius list cannot be empty.")
    if any(radius < 0.0 for radius in patch_radii):
        raise ValueError(f"patch_radius must be non-negative. Got: {patch_radii}")
    return patch_radii


def _find_flat_patches_on_surface_mesh(
    mesh: trimesh.Trimesh,
    *,
    device: str,
    num_patches: int,
    patch_radius: float | list[float] | tuple[float, ...],
    origin: np.ndarray | torch.Tensor | tuple[float, float, float],
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    z_range: tuple[float, float],
    max_height_diff: float,
) -> np.ndarray:
    """Find flat patches on mesh terrains (legacy-compatible behavior).

    This mirrors the legacy mesh-based `find_flat_patches()` flow used by the
    original InstinctLab terrain generator:
    1. Sample XY candidates in configured local ranges around `origin`.
    2. Ray-cast circular footprints onto the mesh.
    3. Reject candidates violating z-range or max height-difference.
    4. Return patch locations in the same frame as InstinctLab
       (mesh frame relative to `origin`).
    """
    from instinct_mjlab.utils.warp import convert_to_warp_mesh, raycast_mesh

    wp_mesh = convert_to_warp_mesh(mesh.vertices, mesh.faces, device=device)

    torch_device = torch.device(device)
    if isinstance(origin, np.ndarray):
        origin_t = torch.from_numpy(origin).to(dtype=torch.float, device=torch_device)
    elif isinstance(origin, torch.Tensor):
        origin_t = origin.to(dtype=torch.float, device=torch_device)
    else:
        origin_t = torch.tensor(origin, dtype=torch.float, device=torch_device)

    patch_radii = _resolve_patch_radii(patch_radius)

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    x_lo = max(x_range[0] + origin_t[0].item(), float(vertices[:, 0].min()))
    x_hi = min(x_range[1] + origin_t[0].item(), float(vertices[:, 0].max()))
    y_lo = max(y_range[0] + origin_t[1].item(), float(vertices[:, 1].min()))
    y_hi = min(y_range[1] + origin_t[1].item(), float(vertices[:, 1].max()))
    z_lo = z_range[0] + origin_t[2].item()
    z_hi = z_range[1] + origin_t[2].item()

    if x_lo > x_hi or y_lo > y_hi:
        raise RuntimeError(
            "Failed to find valid patches! Sampling range is outside mesh bounds."
            f"\n\tx_range after clipping: ({x_lo}, {x_hi})"
            f"\n\ty_range after clipping: ({y_lo}, {y_hi})"
        )

    angle = torch.linspace(0.0, 2.0 * np.pi, 10, device=torch_device)
    query_x = []
    query_y = []
    for radius in patch_radii:
        query_x.append(radius * torch.cos(angle))
        query_y.append(radius * torch.sin(angle))
    query_x = torch.cat(query_x).unsqueeze(1)
    query_y = torch.cat(query_y).unsqueeze(1)
    query_points = torch.cat([query_x, query_y, torch.zeros_like(query_x)], dim=-1)

    points_ids = torch.arange(num_patches, device=torch_device)
    flat_patches = torch.zeros(num_patches, 3, device=torch_device)

    iter_count = 0
    while len(points_ids) > 0 and iter_count < 10000:
        pos_x = torch.empty(len(points_ids), device=torch_device).uniform_(x_lo, x_hi)
        pos_y = torch.empty(len(points_ids), device=torch_device).uniform_(y_lo, y_hi)
        flat_patches[points_ids, :2] = torch.stack([pos_x, pos_y], dim=-1)

        points = flat_patches[points_ids].unsqueeze(1) + query_points
        points[..., 2] = 100.0
        dirs = torch.zeros_like(points)
        dirs[..., 2] = -1.0

        ray_hits = raycast_mesh(points.view(-1, 3), dirs.view(-1, 3), wp_mesh)[0]
        heights = ray_hits.view(points.shape)[..., 2]

        flat_patches[points_ids, 2] = heights[..., -1]

        not_valid = torch.any(torch.logical_or(heights < z_lo, heights > z_hi), dim=1)
        not_valid = torch.logical_or(
            not_valid,
            (heights.max(dim=1)[0] - heights.min(dim=1)[0]) > max_height_diff,
        )
        points_ids = points_ids[not_valid]
        iter_count += 1

    if len(points_ids) > 0:
        raise RuntimeError(
            "Failed to find valid patches! Please check the input parameters."
            f"\n\tMaximum number of iterations reached: {iter_count}"
            f"\n\tNumber of invalid patches: {len(points_ids)}"
            f"\n\tMaximum height difference: {max_height_diff}"
        )

    return (flat_patches - origin_t).cpu().numpy()


@dataclass
class _HfieldCollisionCfg:
    size: tuple[float, float]
    collision_hfield_resolution: float
    collision_hfield_base_thickness_ratio: float
    collision_hfield_num_workers: int
    collision_hfield_raycast_backend: Literal["cpu", "gpu"] = "cpu"
    collision_hfield_gpu_device: str = "cuda"
    collision_hfield_gpu_batch_size: int = 262144
    collision_hfield_sink_miss_cells: bool = False
    collision_hfield_use_disk_cache: bool = False
    collision_hfield_cache_dirname: str = ".hfield_cache"
    collision_hfield_stitch_edges: bool = False
    collision_hfield_stitch_border_pixels: int = 0
    collision_hfield_stitch_height: float | None = None


class FiledTerrainGenerator(TerrainGenerator):
    """A terrain generator that uses the filed generator."""

    def __init__(self, cfg: FiledTerrainGeneratorCfg, device: str = "cpu"):

        # Access the i-th row, j-th column subterrain config by
        # self._subterrain_specific_cfgs[i*num_cols + j]
        self._subterrain_specific_cfgs: list[SubTerrainBaseCfg] = []
        self._terrain_meshes: list[trimesh.Trimesh] = []
        self.terrain_mesh: trimesh.Trimesh | None = None
        # Keep original patch radii (including list-style radii) for
        # mesh flat-patch sampling after normalizing runtime cfg for mjlab core.
        self._original_patch_radii_by_cfg_id: dict[
            int, dict[str, float | list[float] | tuple[float, ...]]
        ] = {}
        # Keep InstinctLab semantics: generator-level scales apply to every heightfield-like subterrain.
        runtime_cfg = copy.deepcopy(cfg)
        self._sync_subterrain_scales(runtime_cfg)
        self._cache_original_patch_radii(runtime_cfg)
        self._normalize_patch_radii_for_mjlab_core(runtime_cfg)
        super().__init__(runtime_cfg, device)

    def _cache_original_patch_radii(self, cfg: FiledTerrainGeneratorCfg) -> None:
        """Cache original patch-radius config per subterrain for mesh sampling."""
        self._original_patch_radii_by_cfg_id.clear()
        for sub_cfg in cfg.sub_terrains.values():
            patch_sampling = sub_cfg.flat_patch_sampling
            if patch_sampling is None:
                continue
            cached_patch_radii: dict[str, float | list[float] | tuple[float, ...]] = {}
            for patch_name, patch_cfg in patch_sampling.items():
                cached_patch_radii[patch_name] = copy.deepcopy(patch_cfg.patch_radius)
            self._original_patch_radii_by_cfg_id[id(sub_cfg)] = cached_patch_radii

    @staticmethod
    def _normalize_patch_radii_for_mjlab_core(cfg: FiledTerrainGeneratorCfg) -> None:
        """Normalize list-style patch radii to scalar max for mjlab core init."""
        for sub_cfg in cfg.sub_terrains.values():
            patch_sampling = sub_cfg.flat_patch_sampling
            if patch_sampling is None:
                continue
            for patch_cfg in patch_sampling.values():
                patch_cfg.patch_radius = max(_resolve_patch_radii(patch_cfg.patch_radius))

    def _get_original_patch_radius(
        self,
        sub_terrain_cfg: SubTerrainBaseCfg,
        patch_name: str,
        fallback: float | list[float] | tuple[float, ...],
    ) -> float | list[float] | tuple[float, ...]:
        """Get original patch-radius config for this subterrain + patch name."""
        cfg_patch_radii = self._original_patch_radii_by_cfg_id.get(id(sub_terrain_cfg))
        if cfg_patch_radii is None:
            return fallback
        return cfg_patch_radii.get(patch_name, fallback)

    @staticmethod
    def _sync_subterrain_scales(cfg: FiledTerrainGeneratorCfg) -> None:
        """Apply generator-level scales to all compatible sub-terrains.

        Legacy InstinctLab terrain generation treats ``horizontal_scale``,
        ``vertical_scale`` and ``slope_threshold`` as generator-wide settings.
        Parkour configs rely on this behavior (for example stair step discretization).
        """
        from .height_field.hf_terrains_cfg import HfTerrainBaseCfg

        for sub_cfg in cfg.sub_terrains.values():
            if not isinstance(sub_cfg, HfTerrainBaseCfg):
                continue
            if cfg.horizontal_scale is not None:
                sub_cfg.horizontal_scale = cfg.horizontal_scale
            if cfg.vertical_scale is not None:
                sub_cfg.vertical_scale = cfg.vertical_scale
            if cfg.slope_threshold is not None:
                sub_cfg.slope_threshold = cfg.slope_threshold

    def _add_hfield_collision_from_surface_mesh(
        self,
        spec: mujoco.MjSpec,
        world_position: np.ndarray,
        surface_mesh_local: trimesh.Trimesh,
        sub_terrain_cfg: SubTerrainBaseCfg,
        sub_row: int,
        sub_col: int,
    ):
        """Add native hfield collision for terrain mesh surface."""
        from instinct_mjlab.terrains.trimesh.mesh_terrains import _add_collision_hfield_from_mesh

        resolution = self.cfg.hfield_resolution
        if resolution is None:
            resolution = sub_terrain_cfg.horizontal_scale
        if resolution is None:
            resolution = 0.05
        resolution = float(resolution)
        wall_thickness = float(sub_terrain_cfg.wall_thickness or 0.0)
        cfg_border_width = float(sub_terrain_cfg.border_width or 0.0)
        global_stitch_border_width = float(self.cfg.hfield_stitch_border_width or 0.0)
        # Keep terrain seams height-aligned over the intended flat border zone.
        # Use the widest available border hint so every sub-terrain has a
        # consistently flat outer ring before tile stitching.
        stitch_border_width = max(wall_thickness, cfg_border_width, global_stitch_border_width)
        if stitch_border_width > 0.0:
            # Match legacy InstinctLab border discretization semantics used by
            # `height_field_to_mesh`: `int(width / scale) + 1`.
            # This avoids a one-pixel under-coverage when width is an exact
            # multiple of the resolution (for example 1.5m / 0.05m).
            stitch_border_pixels = max(int(stitch_border_width / resolution) + 1, 1)
        else:
            stitch_border_pixels = 1
        # Keep InstinctLab-equivalent terrain tiling semantics:
        # outer tile borders are on a shared flat plane (z=0), so all terrain
        # seams stay level and gap-free independent of sub-terrain internals.
        stitch_height = 0.0

        collision_cfg = _HfieldCollisionCfg(
            size=(float(sub_terrain_cfg.size[0]), float(sub_terrain_cfg.size[1])),
            collision_hfield_resolution=resolution,
            collision_hfield_base_thickness_ratio=float(self.cfg.hfield_base_thickness_ratio),
            collision_hfield_num_workers=int(self.cfg.hfield_num_workers),
            collision_hfield_raycast_backend=str(self.cfg.hfield_raycast_backend),
            collision_hfield_gpu_device=str(self.cfg.hfield_gpu_device),
            collision_hfield_gpu_batch_size=int(self.cfg.hfield_gpu_batch_size),
            collision_hfield_stitch_edges=True,
            collision_hfield_stitch_border_pixels=stitch_border_pixels,
            collision_hfield_stitch_height=float(stitch_height),
        )
        terrain_linear_idx = sub_row * self.cfg.num_cols + sub_col
        hfield_geometry = _add_collision_hfield_from_mesh(
            collision_cfg,
            spec,
            surface_mesh_local,
            terrain_idx=terrain_linear_idx,
            terrain_abspath=None,
        )
        hfield_geometry.geom.pos = np.asarray(hfield_geometry.geom.pos, dtype=np.float64) + world_position
        hfield_geometry.geom.group = 0
        if self.cfg.color_scheme == "random":
            hfield_geometry.geom.rgba[:3] = self.np_rng.uniform(0.3, 0.8, 3)
            hfield_geometry.geom.rgba[3] = 1.0
        else:
            hfield_geometry.geom.rgba[:] = (0.5, 0.5, 0.5, 1.0)
        return hfield_geometry

    @staticmethod
    def _estimate_mesh_border_height(
        mesh: trimesh.Trimesh,
        *,
        size: tuple[float, float],
        edge_tolerance: float,
        probe_width: float,
    ) -> float:
        """Estimate border plane height from mesh vertices near tile boundaries."""
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        if vertices.size == 0:
            return 0.0

        x_max, y_max = float(size[0]), float(size[1])
        tol = max(float(edge_tolerance), 1.0e-9)
        strict_edge_mask = np.logical_or.reduce((
            np.abs(vertices[:, 0] - 0.0) <= tol,
            np.abs(vertices[:, 0] - x_max) <= tol,
            np.abs(vertices[:, 1] - 0.0) <= tol,
            np.abs(vertices[:, 1] - y_max) <= tol,
        ))
        border_vertices = vertices[strict_edge_mask]
        if border_vertices.size == 0:
            band = max(float(probe_width), tol)
            fallback_mask = np.logical_or.reduce((
                vertices[:, 0] <= band,
                vertices[:, 0] >= x_max - band,
                vertices[:, 1] <= band,
                vertices[:, 1] >= y_max - band,
            ))
            border_vertices = vertices[fallback_mask]
        if border_vertices.size == 0:
            return float(np.min(vertices[:, 2]))

        border_height = float(np.median(border_vertices[:, 2]))
        if np.isnan(border_height):
            return 0.0
        return border_height

    def compile(self, spec: mujoco.MjSpec) -> None:
        self._terrain_meshes = []
        self.terrain_mesh = None
        super().compile(spec)
        if len(self._terrain_meshes) == 1:
            self.terrain_mesh = self._terrain_meshes[0]
        elif len(self._terrain_meshes) > 1:
            self.terrain_mesh = trimesh.util.concatenate(self._terrain_meshes)

    def _get_subterrain_function(self, cfg: SubTerrainBaseCfg):
        terrain_function = inspect.getattr_static(type(cfg), "function")
        if isinstance(terrain_function, (staticmethod, classmethod)):
            terrain_function = terrain_function.__func__
        return terrain_function

    @staticmethod
    def _hfield_spec_to_world_mesh(
        hfield_spec,
        geom_pos: np.ndarray,
    ) -> trimesh.Trimesh | None:
        """Convert a MuJoCo hfield spec + geom pose into a world-frame trimesh."""
        nrow = int(hfield_spec.nrow)
        ncol = int(hfield_spec.ncol)
        if nrow <= 1 or ncol <= 1:
            return None

        userdata = np.asarray(hfield_spec.userdata, dtype=np.float64)
        if userdata.size != nrow * ncol:
            return None
        normalized_heights = userdata.reshape(nrow, ncol)

        size = np.asarray(hfield_spec.size, dtype=np.float64).reshape(-1)
        if size.size < 4:
            return None
        half_x, half_y, elevation_range, _base_thickness = size[:4]

        # MuJoCo hfield convention:
        #   columns -> x-axis, rows -> y-axis.
        # Keep axis mapping consistent with mjlab.terrains.utils.
        xs = np.linspace(-half_x, half_x, ncol, dtype=np.float64)
        ys = np.linspace(-half_y, half_y, nrow, dtype=np.float64)
        xx, yy = np.meshgrid(xs, ys, indexing="xy")
        # Hfield top surface follows `geom_z + normalized * elevation_range`.
        zz = normalized_heights * elevation_range

        vertices = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
        vertices += np.asarray(geom_pos, dtype=np.float64).reshape(1, 3)

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

    @staticmethod
    def _box_geom_to_world_mesh(geom) -> trimesh.Trimesh | None:
        """Convert a MuJoCo box geom into a world-frame trimesh."""
        if geom is None:
            return None
        if int(geom.type) != int(mujoco.mjtGeom.mjGEOM_BOX):
            return None

        size = np.asarray(geom.size, dtype=np.float64).reshape(-1)
        if size.size < 3:
            return None
        extents = 2.0 * size[:3]

        transform = np.eye(4, dtype=np.float64)
        quat = np.asarray(geom.quat, dtype=np.float64).reshape(-1)
        if quat.size >= 4:
            transform[:3, :3] = trimesh.transformations.quaternion_matrix(quat[:4])[:3, :3]
        transform[:3, 3] = np.asarray(geom.pos, dtype=np.float64).reshape(3)
        return trimesh.creation.box(extents=extents, transform=transform)

    def _create_terrain_geom(
        self,
        spec: mujoco.MjSpec,
        world_position: np.ndarray,
        difficulty: float,
        cfg: SubTerrainBaseCfg,
        sub_row: int,
        sub_col: int,
    ):
        """This function intercept the terrain mesh generation process and records the specific config
        for each subterrain.
        """
        terrain_function = self._get_subterrain_function(cfg)
        num_args = len(inspect.signature(terrain_function).parameters)
        if num_args != 4:
            raise TypeError(
                f"Unsupported terrain function signature for {type(cfg).__name__}: "
                f"expected mjlab-style 4 arguments `(self, difficulty, spec, rng)`, got {num_args}."
            )
        # Record mesh names before calling terrain function so we can identify newly-added geoms.
        mesh_names_before = {m.name for m in spec.meshes}
        hfield_names_before = {h.name for h in spec.hfields}

        # Inline mjlab.TerrainGenerator._create_terrain_geom so we can keep
        # direct access to sub-terrain output for mesh-based virtual obstacles.
        output = cfg.function(difficulty, spec, self.np_rng)
        for terrain_geom in output.geometries:
            if terrain_geom.geom is not None:
                terrain_geom.geom.pos = np.array(terrain_geom.geom.pos) + world_position
                if terrain_geom.geom.material is not None:
                    if self.cfg.color_scheme == "height" and terrain_geom.color:
                        terrain_geom.geom.rgba[:] = terrain_geom.color
                    elif self.cfg.color_scheme == "random":
                        terrain_geom.geom.rgba[:3] = self.np_rng.uniform(0.3, 0.8, 3)
                        terrain_geom.geom.rgba[3] = 1.0
                    elif self.cfg.color_scheme == "none":
                        terrain_geom.geom.rgba[:] = (0.5, 0.5, 0.5, 1.0)

        spawn_origin = output.origin + world_position
        for name, arr in self.flat_patches.items():
            if output.flat_patches is not None and name in output.flat_patches:
                patches = output.flat_patches[name]
                arr[sub_row, sub_col, : len(patches)] = patches + world_position
                arr[sub_row, sub_col, len(patches) :] = spawn_origin
            else:
                arr[sub_row, sub_col] = spawn_origin

        # Collect world-frame mesh for virtual obstacle generation.
        # Prefer meshes reconstructed from the exact MuJoCo hfield specs so
        # virtual obstacles stay aligned with runtime collision terrain.
        surface_mesh_parts: list[trimesh.Trimesh] = []
        has_hfield_surface = False
        for terrain_geom in output.geometries:
            geom = terrain_geom.geom
            if geom is None:
                continue
            hfield_name = geom.hfieldname
            if not isinstance(hfield_name, str) or hfield_name == "":
                continue
            mjs_hfield = spec.hfield(hfield_name)
            if mjs_hfield is None:
                continue
            hfield_world_mesh = self._hfield_spec_to_world_mesh(
                mjs_hfield,
                geom_pos=np.asarray(geom.pos, dtype=np.float64),
            )
            if hfield_world_mesh is not None:
                surface_mesh_parts.append(hfield_world_mesh)
                has_hfield_surface = True

        if not has_hfield_surface:
            surface_mesh_local = getattr(output, "instinct_surface_mesh", None)
            if isinstance(surface_mesh_local, trimesh.Trimesh):
                world_surface_mesh = surface_mesh_local.copy()
                world_surface_mesh.apply_translation(np.asarray(world_position, dtype=np.float64))
                surface_mesh_parts.append(world_surface_mesh)

        for terrain_geom in output.geometries:
            box_mesh = self._box_geom_to_world_mesh(terrain_geom.geom)
            if box_mesh is not None:
                surface_mesh_parts.append(box_mesh)

        if len(surface_mesh_parts) > 0:
            if len(surface_mesh_parts) == 1:
                self._terrain_meshes.append(surface_mesh_parts[0])
            else:
                self._terrain_meshes.append(trimesh.util.concatenate(surface_mesh_parts))
        else:
            # Fallback for non-heightfield sub-terrains that don't provide a
            # surface mesh in TerrainOutput.
            new_mesh_names = {m.name for m in spec.meshes} - mesh_names_before
            new_hfield_names = {h.name for h in spec.hfields} - hfield_names_before
            for terrain_geom in output.geometries:
                geom = terrain_geom.geom
                if geom is None:
                    continue
                mesh_name = geom.meshname
                if isinstance(mesh_name, str) and mesh_name in new_mesh_names:
                    mjs_mesh = spec.mesh(mesh_name)
                    if mjs_mesh is None:
                        continue
                    verts = np.array(mjs_mesh.uservert, dtype=np.float32).reshape(-1, 3)
                    faces = np.array(mjs_mesh.userface, dtype=np.int32).reshape(-1, 3)
                    geom_pos = np.array(geom.pos, dtype=np.float64)
                    world_mesh = trimesh.Trimesh(vertices=verts + geom_pos, faces=faces, process=False)
                    self._terrain_meshes.append(world_mesh)
                    continue

                hfield_name = geom.hfieldname
                if not isinstance(hfield_name, str) or hfield_name not in new_hfield_names:
                    continue
                mjs_hfield = spec.hfield(hfield_name)
                if mjs_hfield is None:
                    continue
                hfield_world_mesh = self._hfield_spec_to_world_mesh(
                    mjs_hfield,
                    geom_pos=np.asarray(geom.pos, dtype=np.float64),
                )
                if hfield_world_mesh is not None:
                    self._terrain_meshes.append(hfield_world_mesh)
        # >>> NOTE: This code snippet is copied from the super implementation because they copied the cfg
        # but we need to store the modified cfg for each subterrain.
        cfg = copy.deepcopy(cfg)
        # add other parameters to the sub-terrain configuration
        cfg.difficulty = float(difficulty)
        cfg.seed = self.cfg.seed
        # <<< NOTE
        self._subterrain_specific_cfgs.append(cfg)  # since in super function, cfg is a copy of the original config.

        return spawn_origin

    @property
    def subterrain_specific_cfgs(self) -> list[SubTerrainBaseCfg]:
        """Get the specific configurations for all subterrains."""
        return self._subterrain_specific_cfgs.copy()  # Return a copy to avoid external modification.

    def get_subterrain_cfg(
        self, row_ids: int | torch.Tensor, col_ids: int | torch.Tensor
    ) -> list[SubTerrainBaseCfg] | SubTerrainBaseCfg | None:
        """Get the specific configuration for a subterrain by its row and column index."""
        num_cols = self.cfg.num_cols
        idx = row_ids * num_cols + col_ids
        if isinstance(idx, torch.Tensor):
            idx = idx.cpu().numpy().tolist()  # Convert to list if it's a tensor.
            return [
                self._subterrain_specific_cfgs[i] if 0 <= i < len(self._subterrain_specific_cfgs) else None for i in idx
            ]
        if isinstance(idx, int):
            return self._subterrain_specific_cfgs[idx] if 0 <= idx < len(self._subterrain_specific_cfgs) else None
