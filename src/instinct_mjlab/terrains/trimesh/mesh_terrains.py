from __future__ import annotations

import atexit
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import numpy as np
import os
from scipy import ndimage
import scipy.spatial.transform as tf
import torch
import trimesh
import uuid
import yaml
from typing import TYPE_CHECKING

import coacd

# Cache CoACD decomposition results keyed by (abspath, params...) so the same
# STL is only decomposed once even when the terrain generator creates many tiles.
_COACD_PARTS_CACHE: dict[tuple, list[tuple[np.ndarray, np.ndarray]]] = {}
_COACD_PREWARM_DONE: set[tuple] = set()
_COACD_CACHE_VERSION = 3
_COACD_LOG_LEVEL_SET: str | None = None

# In-memory cache for hfield height arrays keyed by (abspath, st_size, st_mtime_ns, resolution, size_x, size_y).
# Avoids re-running ray-casting when the same STL is used across multiple terrain tiles.
_HFIELD_HEIGHT_CACHE: dict[tuple, np.ndarray] = {}
_HFIELD_CACHE_VERSION = 4
_HFIELD_RAYCAST_EXECUTORS: dict[int, ProcessPoolExecutor] = {}

import mujoco

from mjlab.terrains.terrain_generator import TerrainGeometry, TerrainOutput
from instinct_mjlab.terrains.height_field.utils import convert_height_field_to_mesh

from ..height_field.hf_terrains import generate_perlin_noise
from .utils import crop_terrain_mesh_aabb, generate_wall

if TYPE_CHECKING:
    from . import mesh_terrains_cfg


def _shutdown_hfield_raycast_executors() -> None:
    for executor in _HFIELD_RAYCAST_EXECUTORS.values():
        executor.shutdown(wait=False, cancel_futures=True)
    _HFIELD_RAYCAST_EXECUTORS.clear()


atexit.register(_shutdown_hfield_raycast_executors)


def _set_coacd_log_level(level: str) -> None:
    """Set CoACD logger level once per process."""
    global _COACD_LOG_LEVEL_SET
    level = str(level)
    if _COACD_LOG_LEVEL_SET == level:
        return
    coacd.set_log_level(level)
    _COACD_LOG_LEVEL_SET = level


def _top_surface_samples(
    terrain_mesh: trimesh.Trimesh,
    normal_z_threshold: float,
    resolution: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract top-surface samples from mesh for heightfield collision fitting."""
    face_mask = np.abs(terrain_mesh.face_normals[:, 2]) > normal_z_threshold
    if np.any(face_mask):
        vertex_indices = np.unique(terrain_mesh.faces[face_mask].reshape(-1))
        samples = terrain_mesh.vertices[vertex_indices]
    else:
        samples = terrain_mesh.vertices

    bucket = max(float(resolution) * 0.5, 1.0e-6)
    quantized_xy = np.round(samples[:, :2] / bucket).astype(np.int64)
    unique_xy, inverse_indices = np.unique(quantized_xy, axis=0, return_inverse=True)
    top_z = np.full(unique_xy.shape[0], -np.inf, dtype=np.float64)
    np.maximum.at(top_z, inverse_indices, samples[:, 2])

    sample_xy = unique_xy.astype(np.float64) * bucket
    return sample_xy, top_z


def _coacd_run_kwargs(cfg: mesh_terrains_cfg.MotionMatchedTerrainCfg) -> dict:
    """Collect CoACD kwargs from config."""
    return {
        "threshold": float(cfg.collision_coacd_threshold),
        "max_convex_hull": int(cfg.collision_coacd_max_convex_hull),
        "preprocess_mode": str(cfg.collision_coacd_preprocess_mode),
        "preprocess_resolution": int(cfg.collision_coacd_preprocess_resolution),
        "resolution": int(cfg.collision_coacd_resolution),
        "mcts_nodes": int(cfg.collision_coacd_mcts_nodes),
        "mcts_iterations": int(cfg.collision_coacd_mcts_iterations),
        "mcts_max_depth": int(cfg.collision_coacd_mcts_max_depth),
        "pca": bool(cfg.collision_coacd_pca),
        "merge": bool(cfg.collision_coacd_merge),
        "decimate": bool(cfg.collision_coacd_decimate),
        "max_ch_vertex": int(cfg.collision_coacd_max_ch_vertex),
        "extrude": bool(cfg.collision_coacd_extrude),
        "extrude_margin": float(cfg.collision_coacd_extrude_margin),
        "apx_mode": str(cfg.collision_coacd_apx_mode),
        "seed": int(cfg.collision_coacd_seed),
    }


def _coacd_cache_key(
    cfg: mesh_terrains_cfg.MotionMatchedTerrainCfg,
    terrain_abspath: str,
) -> tuple:
    """Build a stable cache key for CoACD decomposition."""
    terrain_abspath = os.path.abspath(terrain_abspath)
    stat_info = os.stat(terrain_abspath)
    return (
        _COACD_CACHE_VERSION,
        terrain_abspath,
        int(stat_info.st_size),
        int(stat_info.st_mtime_ns),
        float(cfg.size[0]),
        float(cfg.size[1]),
        float(cfg.collision_coacd_threshold),
        int(cfg.collision_coacd_max_convex_hull),
        str(cfg.collision_coacd_preprocess_mode),
        int(cfg.collision_coacd_preprocess_resolution),
        int(cfg.collision_coacd_resolution),
        int(cfg.collision_coacd_mcts_nodes),
        int(cfg.collision_coacd_mcts_iterations),
        int(cfg.collision_coacd_mcts_max_depth),
        bool(cfg.collision_coacd_pca),
        bool(cfg.collision_coacd_merge),
        bool(cfg.collision_coacd_decimate),
        int(cfg.collision_coacd_max_ch_vertex),
        bool(cfg.collision_coacd_extrude),
        float(cfg.collision_coacd_extrude_margin),
        str(cfg.collision_coacd_apx_mode),
        int(cfg.collision_coacd_seed),
    )


def _coacd_cache_path(
    cfg: mesh_terrains_cfg.MotionMatchedTerrainCfg,
    terrain_abspath: str,
    cache_key: tuple,
) -> str:
    """Get disk-cache file path for CoACD decomposition."""
    terrain_abspath = os.path.abspath(terrain_abspath)
    terrain_dir = os.path.dirname(terrain_abspath)
    cache_dir = os.path.join(terrain_dir, str(cfg.collision_coacd_cache_dirname))
    key_hash = hashlib.sha1(repr(cache_key).encode("utf-8")).hexdigest()[:16]
    terrain_stem = os.path.splitext(os.path.basename(terrain_abspath))[0]
    return os.path.join(cache_dir, f"{terrain_stem}.{key_hash}.npz")


def _load_coacd_parts_from_disk(cache_path: str) -> list[tuple[np.ndarray, np.ndarray]]:
    """Load CoACD parts from a compressed npz cache file."""
    with np.load(cache_path, allow_pickle=False) as cache:
        if "num_parts" not in cache:
            raise ValueError(f"Invalid CoACD cache file (missing num_parts): {cache_path}")
        num_parts = int(np.asarray(cache["num_parts"]).reshape(-1)[0])
        parts_arrays = []
        for part_idx in range(num_parts):
            verts_key = f"verts_{part_idx}"
            faces_key = f"faces_{part_idx}"
            if verts_key not in cache or faces_key not in cache:
                raise ValueError(
                    f"Invalid CoACD cache file (missing part arrays): {cache_path}, "
                    f"part_idx={part_idx}"
                )
            parts_arrays.append(
                (
                    np.asarray(cache[verts_key], dtype=np.float32),
                    np.asarray(cache[faces_key], dtype=np.int32),
                )
            )
    return parts_arrays


def _save_coacd_parts_to_disk(cache_path: str, parts_arrays: list[tuple[np.ndarray, np.ndarray]]) -> None:
    """Save CoACD parts into a compressed npz cache file."""
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    payload = {"num_parts": np.asarray([len(parts_arrays)], dtype=np.int32)}
    for part_idx, (verts, faces) in enumerate(parts_arrays):
        payload[f"verts_{part_idx}"] = np.asarray(verts, dtype=np.float32)
        payload[f"faces_{part_idx}"] = np.asarray(faces, dtype=np.int32)

    tmp_cache_path = f"{cache_path}.{uuid.uuid4().hex}.tmp.npz"
    np.savez_compressed(tmp_cache_path, **payload)
    os.replace(tmp_cache_path, cache_path)
    if os.path.exists(tmp_cache_path):
        os.remove(tmp_cache_path)


def _compute_motion_matched_border_height(
    terrain_mesh: trimesh.Trimesh,
    size: tuple[float, float],
    terrain_file: str,
) -> float:
    """Compute border height offset for motion-matched terrain alignment."""
    # Find border height offset w.r.t current center of this terrain mesh.
    # This is used to align the terrain mesh with the origin.
    # NOTE: Assuming the border is flat, we take the mean height of the vertices
    verts = terrain_mesh.vertices
    border_mask = np.logical_or(
        np.abs(verts[:, 0]) > (size[0] / 2 - 0.05),
        np.abs(verts[:, 1]) > (size[1] / 2 - 0.05),
    )
    border_verts_z = verts[border_mask][:, 2]

    if len(border_verts_z) == 0:
        # No vertices at cfg.size boundary — mesh is smaller than cfg.size.
        # Fall back to 0.0, matching original InstinctLab behavior for these terrains.
        # These STL meshes were recorded with the walkable floor at z≈0 by convention,
        # so no z-shift is needed.  Using AABB edges as fallback is unsafe because for
        # staircase/ramp terrains the AABB edges sample obstacle tops (z=0.6–0.7 m),
        # which would incorrectly sink the entire tile by that amount.
        border_height = 0.0
        print(
            f"[TerrainImporter] Terrain {terrain_file} has no border verts at cfg.size edges; "
            f"using border_height=0.0 (floor assumed at z≈0)."
        )
    else:
        border_height = float(np.mean(border_verts_z))
        if np.isnan(border_height):
            # Unexpected — empty slice mean; treat as 0.
            print(f"Warning: Terrain {terrain_file} does not have a valid border height. Using 0 as the border height.")
            border_height = 0.0
    return border_height


def _load_motion_matched_terrain_mesh(
    terrain_abspath: str,
    terrain_file: str,
    size: tuple[float, float],
) -> tuple[trimesh.Trimesh, float]:
    """Load, crop, and align one terrain mesh to generator convention."""
    terrain_mesh = trimesh.load(terrain_abspath, force="mesh")

    # crop terrain mesh by cfg.size
    # This does not change the terrain origin.
    terrain_mesh = crop_terrain_mesh_aabb(
        terrain_mesh,
        x_max=size[0] / 2,
        x_min=-size[0] / 2,
        y_max=size[1] / 2,
        y_min=-size[1] / 2,
    )
    # Normalize mesh winding/normals before CoACD to reduce manifold-orientation artifacts.
    terrain_mesh.remove_unreferenced_vertices()
    trimesh.repair.fix_winding(terrain_mesh)
    trimesh.repair.fix_normals(terrain_mesh, multibody=True)

    border_height = _compute_motion_matched_border_height(
        terrain_mesh=terrain_mesh,
        size=size,
        terrain_file=terrain_file,
    )

    # To follow the terrain_generator convention, we move the terrain mesh to (size[0]/2, size[1]/2, -border_height).
    move_terrain_transform = np.eye(4)
    move_terrain_transform[:2, 3] = np.asarray(size, dtype=np.float64) / 2.0
    move_terrain_transform[2, 3] = -border_height
    terrain_mesh.apply_transform(move_terrain_transform)
    return terrain_mesh, border_height


def _sanitize_coacd_parts(
    parts_arrays: list[tuple[np.ndarray, np.ndarray]],
    *,
    terrain_tag: str,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Drop degenerate CoACD parts that can trigger MuJoCo qhull failures."""
    sanitized: list[tuple[np.ndarray, np.ndarray]] = []
    skipped_parts = 0
    total_parts = len(parts_arrays)

    for part_verts, part_faces in parts_arrays:
        verts = np.asarray(part_verts, dtype=np.float64)
        faces = np.asarray(part_faces, dtype=np.int64)

        if verts.ndim != 2 or verts.shape[1] != 3:
            skipped_parts += 1
            continue
        if faces.ndim != 2 or faces.shape[1] != 3:
            skipped_parts += 1
            continue
        if verts.shape[0] < 4 or faces.shape[0] < 4:
            skipped_parts += 1
            continue

        valid_faces = np.logical_and(faces >= 0, faces < verts.shape[0]).all(axis=1)
        if not np.any(valid_faces):
            skipped_parts += 1
            continue
        faces = faces[valid_faces]

        used_vertex_ids = np.unique(faces.reshape(-1))
        if used_vertex_ids.size < 4:
            skipped_parts += 1
            continue

        remap = np.full(verts.shape[0], -1, dtype=np.int64)
        remap[used_vertex_ids] = np.arange(used_vertex_ids.size, dtype=np.int64)
        verts = verts[used_vertex_ids]
        faces = remap[faces]

        # Drop duplicated triangles (ignore winding for the duplicate check).
        sorted_faces = np.sort(faces, axis=1)
        _, unique_face_indices = np.unique(sorted_faces, axis=0, return_index=True)
        faces = faces[np.sort(unique_face_indices)]
        if faces.shape[0] < 4:
            skipped_parts += 1
            continue

        # Reject near-planar/near-line hulls that are not valid 3D collision meshes.
        centered = verts - np.mean(verts, axis=0, keepdims=True)
        _, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
        if singular_values.shape[0] < 3:
            skipped_parts += 1
            continue
        largest_sv = max(float(singular_values[0]), 1.0e-12)
        smallest_sv = float(singular_values[-1])
        if smallest_sv / largest_sv < 1.0e-8:
            skipped_parts += 1
            continue
        extents = np.ptp(verts, axis=0)
        if float(np.min(extents)) < 1.0e-7:
            skipped_parts += 1
            continue

        sanitized.append((verts.astype(np.float32), faces.astype(np.int32)))

    if len(sanitized) == 0:
        raise ValueError(
            "[TerrainImporter] CoACD produced no valid 3D hulls for "
            f"{terrain_tag}. Try increasing collision_coacd_threshold."
        )
    if skipped_parts > 0:
        print(
            f"[TerrainImporter] Skip {skipped_parts}/{total_parts} degenerate "
            f"CoACD hull(s) for {terrain_tag}."
        )
    return sanitized


def _adapt_coacd_kwargs_for_mesh(
    coacd_kwargs: dict,
    terrain_mesh: trimesh.Trimesh,
) -> dict:
    """Adapt CoACD quality parameters from mesh complexity."""
    adapted_kwargs = dict(coacd_kwargs)

    face_count = int(np.asarray(terrain_mesh.faces).shape[0])
    bounds = np.asarray(terrain_mesh.bounds, dtype=np.float64)
    extents = np.maximum(bounds[1] - bounds[0], 1.0e-9)
    xy_extent = float(max(extents[0], extents[1]))
    z_extent = float(extents[2])
    relief_ratio = z_extent / max(xy_extent, 1.0e-9)

    detail_gain = 1.0
    if face_count <= 120_000:
        detail_gain *= 1.35
    elif face_count <= 260_000:
        detail_gain *= 1.22
    elif face_count <= 520_000:
        detail_gain *= 1.12
    if relief_ratio >= 0.08:
        detail_gain *= 1.12
    elif relief_ratio <= 0.02:
        detail_gain *= 0.96

    threshold = float(adapted_kwargs["threshold"]) / detail_gain
    resolution = int(round(float(adapted_kwargs["resolution"]) * detail_gain))
    mcts_iterations = int(round(float(adapted_kwargs["mcts_iterations"]) * min(detail_gain, 1.2)))

    adapted_kwargs["threshold"] = float(np.clip(threshold, 0.015, 0.08))
    adapted_kwargs["resolution"] = int(np.clip(resolution, 1200, 5000))
    adapted_kwargs["mcts_iterations"] = int(np.clip(mcts_iterations, 80, 280))
    return adapted_kwargs


def _run_coacd_decomposition(
    terrain_mesh: trimesh.Trimesh,
    coacd_kwargs: dict,
    terrain_tag: str,
    coacd_log_level: str,
    verbose: bool = False,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Execute CoACD and return numpy-array parts."""
    _set_coacd_log_level(coacd_log_level)

    coacd_mesh = coacd.Mesh(
        vertices=np.asarray(terrain_mesh.vertices, dtype=np.float64),
        indices=np.asarray(terrain_mesh.faces, dtype=np.int32),
    )
    adaptive_coacd_kwargs = _adapt_coacd_kwargs_for_mesh(coacd_kwargs, terrain_mesh)

    if verbose:
        print(
            f"[TerrainImporter] CoACD start: {terrain_tag}, "
            f"th={adaptive_coacd_kwargs['threshold']}, "
            f"res={adaptive_coacd_kwargs['resolution']}, "
            f"iter={adaptive_coacd_kwargs['mcts_iterations']}, "
            f"max_hull={adaptive_coacd_kwargs['max_convex_hull']}, "
            f"merge={adaptive_coacd_kwargs['merge']}, "
            f"apx={adaptive_coacd_kwargs['apx_mode']}."
        )

    # Run CoACD convex decomposition.
    raw_parts = coacd.run_coacd(
        coacd_mesh,
        **adaptive_coacd_kwargs,
    )

    # Store numpy arrays in cache (not raw CoACD objects)
    parts_arrays = [
        (np.asarray(v, dtype=np.float32), np.asarray(f, dtype=np.int32))
        for v, f in raw_parts
    ]
    parts_arrays = _sanitize_coacd_parts(parts_arrays, terrain_tag=terrain_tag)
    if verbose:
        print(f"[TerrainImporter] CoACD done: {terrain_tag}, hulls={len(parts_arrays)}.")
    return parts_arrays


def _coacd_prewarm_worker(
    job: tuple[str, str, tuple[float, float], dict, tuple, str, str],
) -> tuple[str, str, int]:
    """Worker job for CoACD cache prewarm."""
    terrain_file, terrain_abspath, size, coacd_kwargs, cache_key, coacd_log_level, cache_path = job
    terrain_mesh, _ = _load_motion_matched_terrain_mesh(
        terrain_abspath=terrain_abspath,
        terrain_file=terrain_file,
        size=size,
    )
    parts_arrays = _run_coacd_decomposition(
        terrain_mesh=terrain_mesh,
        coacd_kwargs=coacd_kwargs,
        terrain_tag=f"terrain_file={terrain_file}",
        coacd_log_level=coacd_log_level,
        verbose=False,
    )
    _save_coacd_parts_to_disk(cache_path, parts_arrays)
    return terrain_abspath, cache_path, len(parts_arrays)


def _render_progress_bar(done: int, total: int, width: int = 24) -> str:
    """Render a one-line ASCII progress bar."""
    total = max(total, 1)
    done = int(np.clip(done, 0, total))
    filled = int(round(width * done / total))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def _print_prewarm_progress(done: int, total: int) -> None:
    """Print CoACD prewarm progress in-place."""
    progress_bar = _render_progress_bar(done, total)
    percent = (100.0 * done / max(total, 1))
    remaining = max(total - done, 0)
    print(
        f"\r[TerrainImporter] CoACD prewarm {progress_bar} "
        f"{done}/{total} ({percent:5.1f}%) remaining={remaining}",
        end="",
        flush=True,
    )


def _prewarm_coacd_disk_cache(
    cfg: mesh_terrains_cfg.MotionMatchedTerrainCfg,
    terrains: list[dict],
) -> None:
    """Precompute all required CoACD decompositions with compact progress output."""
    if not cfg.collision_coacd_prewarm_all:
        return

    prewarm_key = (
        os.path.abspath(cfg.path),
        os.path.abspath(cfg.metadata_yaml),
        float(cfg.size[0]),
        float(cfg.size[1]),
        float(cfg.collision_coacd_threshold),
        int(cfg.collision_coacd_max_convex_hull),
        str(cfg.collision_coacd_preprocess_mode),
        int(cfg.collision_coacd_preprocess_resolution),
        int(cfg.collision_coacd_resolution),
        int(cfg.collision_coacd_mcts_nodes),
        int(cfg.collision_coacd_mcts_iterations),
        int(cfg.collision_coacd_mcts_max_depth),
        bool(cfg.collision_coacd_pca),
        bool(cfg.collision_coacd_merge),
        bool(cfg.collision_coacd_decimate),
        int(cfg.collision_coacd_max_ch_vertex),
        bool(cfg.collision_coacd_extrude),
        float(cfg.collision_coacd_extrude_margin),
        str(cfg.collision_coacd_apx_mode),
        int(cfg.collision_coacd_seed),
        bool(cfg.collision_coacd_use_disk_cache),
        str(cfg.collision_coacd_cache_dirname),
    )
    if prewarm_key in _COACD_PREWARM_DONE:
        return

    size = (float(cfg.size[0]), float(cfg.size[1]))
    coacd_kwargs = _coacd_run_kwargs(cfg)
    coacd_log_level = str(cfg.collision_coacd_log_level)

    jobs_by_path: dict[str, tuple[str, str, tuple, str]] = {}
    for terrain_entry in terrains:
        terrain_file = terrain_entry["terrain_file"]
        terrain_abspath = os.path.join(cfg.path, terrain_file)
        cache_key = _coacd_cache_key(cfg, terrain_abspath)
        if _COACD_PARTS_CACHE.get(cache_key) is not None:
            continue
        cache_path = _coacd_cache_path(cfg, terrain_abspath, cache_key)
        if cfg.collision_coacd_use_disk_cache and os.path.exists(cache_path):
            continue
        # Deduplicate repeated terrain files in metadata.
        jobs_by_path[terrain_abspath] = (
            terrain_file,
            terrain_abspath,
            cache_key,
            cache_path,
        )

    jobs = list(jobs_by_path.values())
    if len(jobs) == 0:
        _COACD_PREWARM_DONE.add(prewarm_key)
        return

    worker_count = int(cfg.collision_coacd_prewarm_workers)
    if worker_count <= 0:
        worker_count = max(os.cpu_count() or 1, 1)
    worker_count = min(worker_count, len(jobs))

    total_jobs = len(jobs)
    _print_prewarm_progress(0, total_jobs)

    if worker_count == 1:
        for done_count, (terrain_file, terrain_abspath, cache_key, cache_path) in enumerate(jobs, start=1):
            terrain_mesh, _ = _load_motion_matched_terrain_mesh(
                terrain_abspath=terrain_abspath,
                terrain_file=terrain_file,
                size=size,
            )
            parts_arrays = _run_coacd_decomposition(
                terrain_mesh=terrain_mesh,
                coacd_kwargs=coacd_kwargs,
                terrain_tag=f"terrain_file={terrain_file}",
                coacd_log_level=coacd_log_level,
                verbose=False,
            )
            _COACD_PARTS_CACHE[cache_key] = parts_arrays
            if cfg.collision_coacd_use_disk_cache:
                _save_coacd_parts_to_disk(cache_path, parts_arrays)
            _print_prewarm_progress(done_count, total_jobs)
    else:
        worker_jobs = [
            (
                terrain_file,
                terrain_abspath,
                size,
                coacd_kwargs,
                cache_key,
                coacd_log_level,
                cache_path,
            )
            for terrain_file, terrain_abspath, cache_key, cache_path in jobs
        ]
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(_coacd_prewarm_worker, worker_job) for worker_job in worker_jobs]
            done_count = 0
            for future in as_completed(futures):
                future.result()
                done_count += 1
                _print_prewarm_progress(done_count, total_jobs)

    _COACD_PREWARM_DONE.add(prewarm_key)
    print()


def _add_collision_face_boxes(
    cfg: mesh_terrains_cfg.MotionMatchedTerrainCfg,
    spec: mujoco.MjSpec,
    terrain_mesh: trimesh.Trimesh,
) -> list[TerrainGeometry]:
    """Build surface-only collision for a closed mesh by placing a thin box geom on each face.

    Root cause: MuJoCo's polyhedral mesh collision model treats the interior of any
    closed mesh as solid.  Objects near the inner surface receive negative-dist contacts
    even when not physically penetrating, causing ``illegal_reset_contact`` to fire
    immediately after spawn.

    Fix: disable collision on the visual mesh geom and replace it with one thin
    ``mjGEOM_BOX`` per triangle face, placed on the outward side of the face.
    Each box only collides from one side, so the robot inside the room is free
    to move without spurious contacts.
    """
    thickness = float(cfg.face_box_thickness)
    geometries: list[TerrainGeometry] = []
    face_normals = terrain_mesh.face_normals  # [F, 3], unit normals

    for face, normal in zip(terrain_mesh.faces, face_normals):
        verts = terrain_mesh.vertices[face]  # [3, 3]
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-10:
            # Degenerate face — skip
            continue
        unit_normal = normal / norm_len

        # Face centroid
        center = verts.mean(axis=0)

        # Build a local frame: t1 along first edge, t2 = normal x t1
        edge = verts[1] - verts[0]
        edge_len = np.linalg.norm(edge)
        if edge_len < 1e-10:
            continue
        t1 = edge / edge_len
        t2 = np.cross(unit_normal, t1)

        # Project vertices onto the face plane to get half-extents
        proj = verts - center
        coords_t1 = proj @ t1
        coords_t2 = proj @ t2
        half_t1 = (coords_t1.max() - coords_t1.min()) / 2.0 + 0.01
        half_t2 = (coords_t2.max() - coords_t2.min()) / 2.0 + 0.01

        # Box center: offset outward by half-thickness so the inner face sits at the surface
        box_center = center + unit_normal * (thickness / 2.0)

        # Rotation matrix: columns are [t1, t2, normal] -> MuJoCo quaternion [w, x, y, z]
        rot = np.column_stack([t1, t2, unit_normal])
        mat4 = np.eye(4)
        mat4[:3, :3] = rot
        quat_wxyz = trimesh.transformations.quaternion_from_matrix(mat4)  # [w, x, y, z]

        geom = spec.body("terrain").add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[half_t1, half_t2, thickness / 2.0],
            pos=box_center.tolist(),
            quat=quat_wxyz.tolist(),
        )
        # Collision-only: invisible in default render group
        geom.group = 3
        geom.rgba[:] = (0.0, 0.0, 0.0, 0.0)
        geometries.append(TerrainGeometry(geom=geom))

    print(
        f"[TerrainImporter] Face-box collision: "
        f"{len(geometries)} box geoms generated from {len(terrain_mesh.faces)} faces."
    )
    return geometries


def _top_surface_height_map(
    sample_xy: np.ndarray,
    sample_z: np.ndarray,
    resolution: float,
) -> dict[tuple[int, int], float]:
    """Build max-z map indexed by quantized XY bins."""
    bucket = max(float(resolution), 1.0e-6)
    quantized_xy = np.round(sample_xy / bucket).astype(np.int64)
    height_map: dict[tuple[int, int], float] = {}
    for i in range(quantized_xy.shape[0]):
        key = (int(quantized_xy[i, 0]), int(quantized_xy[i, 1]))
        z = float(sample_z[i])
        previous = height_map.get(key)
        if previous is None or z > previous:
            height_map[key] = z
    return height_map


def _compute_coacd_auto_align_z(
    cfg: mesh_terrains_cfg.MotionMatchedTerrainCfg,
    terrain_mesh: trimesh.Trimesh,
    parts_arrays: list[tuple[np.ndarray, np.ndarray]],
) -> float:
    """Compute auto z-alignment from visual top surface to CoACD top surface."""
    if len(parts_arrays) == 0:
        return 0.0

    align_resolution = float(cfg.collision_coacd_auto_align_resolution)
    mesh_xy, mesh_z = _top_surface_samples(
        terrain_mesh=terrain_mesh,
        normal_z_threshold=cfg.collision_hfield_normal_z_threshold,
        resolution=align_resolution,
    )
    mesh_map = _top_surface_height_map(mesh_xy, mesh_z, align_resolution)

    verts_list: list[np.ndarray] = []
    faces_list: list[np.ndarray] = []
    vert_offset = 0
    for part_verts, part_faces in parts_arrays:
        verts = np.asarray(part_verts, dtype=np.float64)
        faces = np.asarray(part_faces, dtype=np.int64)
        verts_list.append(verts)
        faces_list.append(faces + vert_offset)
        vert_offset += verts.shape[0]
    coacd_mesh = trimesh.Trimesh(
        vertices=np.concatenate(verts_list, axis=0),
        faces=np.concatenate(faces_list, axis=0),
        process=False,
    )
    coacd_xy, coacd_z = _top_surface_samples(
        terrain_mesh=coacd_mesh,
        normal_z_threshold=cfg.collision_hfield_normal_z_threshold,
        resolution=align_resolution,
    )
    coacd_map = _top_surface_height_map(coacd_xy, coacd_z, align_resolution)

    common_keys = mesh_map.keys() & coacd_map.keys()
    if len(common_keys) == 0:
        return 0.0

    z_diff = np.asarray(
        [mesh_map[key] - coacd_map[key] for key in common_keys],
        dtype=np.float64,
    )
    if z_diff.size == 0:
        return 0.0
    return float(np.median(z_diff))


def _add_collision_coacd(
    cfg: mesh_terrains_cfg.MotionMatchedTerrainCfg,
    spec: mujoco.MjSpec,
    terrain_mesh: trimesh.Trimesh,
    terrain_idx: int,
    terrain_abspath: str,
) -> list[TerrainGeometry]:
    """Build collision geometry for a concave mesh using CoACD approximate convex decomposition.

    Root cause: MuJoCo's polyhedral mesh collision model treats the interior of any
    closed mesh as solid.  A robot standing inside a 3D-scanned room receives
    negative-dist contacts even when not physically penetrating.

    Fix: decompose the mesh into approximate convex hulls with CoACD.  Each hull
    is registered as a separate MuJoCo mesh and added as a collision-only geom
    (group 3, invisible).  The original mesh geom is kept for rendering / depth
    ray-casting (group 2, no collision).

    Results are cached in memory and on disk so the same STL is only decomposed
    once across terrain tiles and across process restarts.
    """
    cache_key = _coacd_cache_key(cfg, terrain_abspath)
    cache_path = _coacd_cache_path(cfg, terrain_abspath, cache_key)
    coacd_kwargs = _coacd_run_kwargs(cfg)
    loaded_from_disk_cache = False

    cached_parts = _COACD_PARTS_CACHE.get(cache_key)
    if cached_parts is not None:
        parts_arrays = cached_parts
    else:
        parts_arrays = None
        if cfg.collision_coacd_use_disk_cache and os.path.exists(cache_path):
            parts_arrays = _load_coacd_parts_from_disk(cache_path)
            loaded_from_disk_cache = True

        if parts_arrays is None:
            parts_arrays = _run_coacd_decomposition(
                terrain_mesh=terrain_mesh,
                coacd_kwargs=coacd_kwargs,
                terrain_tag=f"terrain_idx={terrain_idx}",
                coacd_log_level=str(cfg.collision_coacd_log_level),
                verbose=False,
            )
            if cfg.collision_coacd_use_disk_cache:
                _save_coacd_parts_to_disk(cache_path, parts_arrays)
        else:
            original_num_parts = len(parts_arrays)
            parts_arrays = _sanitize_coacd_parts(
                parts_arrays,
                terrain_tag=f"terrain_idx={terrain_idx} (disk-cache)",
            )
            if cfg.collision_coacd_use_disk_cache and loaded_from_disk_cache:
                if len(parts_arrays) != original_num_parts:
                    _save_coacd_parts_to_disk(cache_path, parts_arrays)
        _COACD_PARTS_CACHE[cache_key] = parts_arrays

    auto_align_z = 0.0
    if bool(cfg.collision_coacd_auto_align_top_surface):
        auto_align_z = _compute_coacd_auto_align_z(
            cfg=cfg,
            terrain_mesh=terrain_mesh,
            parts_arrays=parts_arrays,
        )

    collision_z = float(cfg.collision_coacd_z_offset) + auto_align_z

    use_default_hull_appearance = bool(cfg.collision_coacd_visualize_collision_hulls)
    if use_default_hull_appearance:
        hull_geom_group = 2
        hull_geom_rgba = None
    else:
        hull_geom_group = int(cfg.collision_coacd_collision_geom_group)
        hull_geom_rgba = tuple(float(v) for v in cfg.collision_coacd_collision_geom_rgba)
    if not 0 <= hull_geom_group <= 5:
        raise ValueError(
            "collision_coacd_collision_geom_group must be in [0, 5]. "
            f"Got: {hull_geom_group}"
        )
    if hull_geom_rgba is not None:
        if len(hull_geom_rgba) != 4:
            raise ValueError(
                "collision_coacd_collision_geom_rgba must have 4 elements "
                f"(r, g, b, a). Got: {hull_geom_rgba}"
            )

    geometries: list[TerrainGeometry] = []
    for part_idx, (part_verts, part_faces) in enumerate(parts_arrays):
        hull_mesh_name = f"motion_matched_coacd_t{terrain_idx}_h{part_idx}_{uuid.uuid4().hex}"
        spec.add_mesh(
            name=hull_mesh_name,
            uservert=part_verts.reshape(-1).tolist(),
            userface=part_faces.reshape(-1).tolist(),
        )
        hull_geom = spec.body("terrain").add_geom(
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=hull_mesh_name,
            pos=(0.0, 0.0, collision_z),
        )
        hull_geom.group = hull_geom_group
        if hull_geom_rgba is not None:
            hull_geom.rgba[:] = hull_geom_rgba
        hull_geom.margin = float(cfg.collision_coacd_geom_margin)
        hull_geom.gap = 0.0
        geometries.append(TerrainGeometry(geom=hull_geom))

    return geometries


def _hfield_cache_key(
    terrain_abspath: str,
    resolution: float,
    size: tuple[float, float],
) -> tuple:
    """Build a stable cache key for hfield ray-cast results."""
    terrain_abspath = os.path.abspath(terrain_abspath)
    stat_info = os.stat(terrain_abspath)
    return (
        _HFIELD_CACHE_VERSION,
        terrain_abspath,
        int(stat_info.st_size),
        int(stat_info.st_mtime_ns),
        float(resolution),
        float(size[0]),
        float(size[1]),
    )


def _hfield_disk_cache_path(
    terrain_abspath: str,
    cache_key: tuple,
    cache_dirname: str,
) -> str:
    """Get disk-cache file path for hfield height array."""
    terrain_abspath = os.path.abspath(terrain_abspath)
    terrain_dir = os.path.dirname(terrain_abspath)
    cache_dir = os.path.join(terrain_dir, cache_dirname)
    key_hash = hashlib.sha1(repr(cache_key).encode("utf-8")).hexdigest()[:16]
    terrain_stem = os.path.splitext(os.path.basename(terrain_abspath))[0]
    return os.path.join(cache_dir, f"{terrain_stem}.hfield.{key_hash}.npz")


def _raycast_rows_worker(
    job: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Worker: ray-cast a subset of rows and return (ray_indices_local, hit_z)."""
    import trimesh as _trimesh
    vertices, faces, ray_origins_chunk, ray_directions_chunk, _mesh_z_min, _unused = job
    mesh = _trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    ray_caster = _trimesh.ray.ray_triangle.RayMeshIntersector(mesh)
    hit_points, ray_indices, _ = ray_caster.intersects_location(
        ray_origins=ray_origins_chunk,
        ray_directions=ray_directions_chunk,
        multiple_hits=True,
    )
    if len(ray_indices) > 0:
        return ray_indices, hit_points[:, 2]
    return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)


def _raycast_hfield_gpu(
    terrain_mesh: trimesh.Trimesh,
    ray_origins: np.ndarray,
    n_rays: int,
    gpu_device: str,
    batch_size: int,
) -> np.ndarray:
    """Ray-cast all hfield sampling rays on GPU using Warp."""
    from instinct_mjlab.utils.warp import convert_to_warp_mesh, raycast_mesh

    if batch_size <= 0:
        raise ValueError(f"collision_hfield_gpu_batch_size must be > 0. Got: {batch_size}")

    device = torch.device(gpu_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "collision_hfield_raycast_backend='gpu' requires CUDA, but torch.cuda.is_available() is False."
        )

    wp_mesh = convert_to_warp_mesh(
        np.asarray(terrain_mesh.vertices, dtype=np.float64),
        np.asarray(terrain_mesh.faces, dtype=np.int32),
        device=str(device),
    )

    height = np.full(n_rays, np.nan, dtype=np.float64)
    for start in range(0, n_rays, batch_size):
        end = min(start + batch_size, n_rays)
        origins = torch.from_numpy(ray_origins[start:end]).to(device=device, dtype=torch.float32)
        directions = torch.zeros_like(origins)
        directions[:, 2] = -1.0
        ray_hits = raycast_mesh(origins, directions, wp_mesh)[0]
        ray_hits_z = ray_hits[:, 2]
        valid_hit_mask = torch.isfinite(ray_hits_z)
        if bool(valid_hit_mask.any()):
            local_indices = torch.nonzero(valid_hit_mask, as_tuple=False).flatten()
            global_indices = local_indices.cpu().numpy() + start
            height[global_indices] = ray_hits_z[valid_hit_mask].detach().cpu().numpy().astype(np.float64)
    return height


def _raycast_hfield_parallel(
    terrain_mesh: trimesh.Trimesh,
    ray_origins: np.ndarray,
    n_rays: int,
    num_workers: int,
) -> np.ndarray:
    """Ray-cast all rays in parallel across CPU cores, return per-ray max-z array.

    Grid points that miss the mesh entirely are returned as NaN so the caller
    can distinguish them from valid hits and fill them appropriately.
    """
    # Initialise with NaN — rays that miss the mesh stay NaN.
    height = np.full(n_rays, np.nan, dtype=np.float64)
    ray_directions = np.tile(np.array([0.0, 0.0, -1.0]), (n_rays, 1))

    vertices = np.asarray(terrain_mesh.vertices, dtype=np.float64)
    faces = np.asarray(terrain_mesh.faces, dtype=np.int32)

    # Split rays into chunks — one chunk per worker.
    chunk_size = max(1, (n_rays + num_workers - 1) // num_workers)
    chunks = []
    for start in range(0, n_rays, chunk_size):
        end = min(start + chunk_size, n_rays)
        chunks.append((
            vertices,
            faces,
            ray_origins[start:end],
            ray_directions[start:end],
            0.0,  # unused
            0.0,  # unused placeholder
        ))

    if num_workers == 1:
        # Single-process path: avoid ProcessPoolExecutor overhead.
        for chunk_start, chunk in zip(range(0, n_rays, chunk_size), chunks):
            local_indices, hit_z = _raycast_rows_worker(chunk)
            if len(local_indices) > 0:
                global_indices = local_indices + chunk_start
                # First hit: replace NaN with the z value.
                nan_mask = np.isnan(height[global_indices])
                height[global_indices[nan_mask]] = hit_z[nan_mask]
                # Subsequent hits: keep the maximum (top surface).
                np.maximum.at(height, global_indices[~nan_mask], hit_z[~nan_mask])
    else:
        executor = _HFIELD_RAYCAST_EXECUTORS.get(num_workers)
        if executor is None:
            executor = ProcessPoolExecutor(max_workers=num_workers)
            _HFIELD_RAYCAST_EXECUTORS[num_workers] = executor
        futures = []
        chunk_starts = list(range(0, n_rays, chunk_size))
        for chunk in chunks:
            futures.append(executor.submit(_raycast_rows_worker, chunk))
        for chunk_start, future in zip(chunk_starts, futures):
            local_indices, hit_z = future.result()
            if len(local_indices) > 0:
                global_indices = local_indices + chunk_start
                nan_mask = np.isnan(height[global_indices])
                height[global_indices[nan_mask]] = hit_z[nan_mask]
                np.maximum.at(height, global_indices[~nan_mask], hit_z[~nan_mask])

    return height


def _fill_missing_heights_with_nearest(
    raw_height: np.ndarray,
    hit_mask: np.ndarray,
    fallback_value: float,
) -> np.ndarray:
    """Fill NaN (miss) cells with nearest valid hit heights to avoid artificial seams."""
    if raw_height.shape != hit_mask.shape:
        raise ValueError("raw_height and hit_mask must have the same shape.")
    if np.all(hit_mask):
        return raw_height
    if not np.any(hit_mask):
        return np.full_like(raw_height, fallback_value, dtype=np.float64)

    filled = raw_height.copy()
    nearest_indices = ndimage.distance_transform_edt(
        ~hit_mask,
        return_distances=False,
        return_indices=True,
    )
    filled[~hit_mask] = filled[nearest_indices[0][~hit_mask], nearest_indices[1][~hit_mask]]
    return filled


def _add_collision_hfield_from_mesh(
    cfg: mesh_terrains_cfg.MotionMatchedTerrainCfg,
    spec: mujoco.MjSpec,
    terrain_mesh: trimesh.Trimesh,
    terrain_idx: int,
    terrain_abspath: str | None = None,
) -> TerrainGeometry:
    """Create a collision hfield from terrain mesh by ray-casting from above at each grid point.

    Each grid point fires a downward ray and records the highest intersection with the mesh.
    This gives accurate height values even between mesh vertices, unlike vertex-sampling
    + interpolation which can miss geometry between sparse vertices.
    Grid points that miss the mesh entirely fall back to the mesh AABB minimum z.

    Results are cached in memory and on disk (keyed by STL path + mtime + resolution + size)
    so the same STL tile is only ray-cast once across terrain tiles and process restarts.
    Ray-casting is parallelised across CPU cores via multiprocessing.
    """
    resolution = float(cfg.collision_hfield_resolution)
    if resolution <= 0.0:
        raise ValueError("collision_hfield_resolution must be greater than zero.")

    nrow = max(int(np.round(cfg.size[0] / resolution)) + 1, 2)
    ncol = max(int(np.round(cfg.size[1] / resolution)) + 1, 2)
    x = np.linspace(0.0, cfg.size[0], nrow, dtype=np.float64)
    y = np.linspace(0.0, cfg.size[1], ncol, dtype=np.float64)
    xx, yy = np.meshgrid(x, y, indexing="ij")

    # Build ray origins above the mesh AABB and shoot downward.
    mesh_z_max = float(terrain_mesh.bounds[1, 2])
    mesh_z_min = float(terrain_mesh.bounds[0, 2])
    ray_origin_z = mesh_z_max + 1.0

    n_rays = nrow * ncol
    ray_origins = np.column_stack([
        xx.reshape(-1),
        yy.reshape(-1),
        np.full(n_rays, ray_origin_z, dtype=np.float64),
    ])

    # --- Cache lookup ---
    cache_key: tuple | None = None
    cache_path: str | None = None
    if terrain_abspath is not None:
        terrain_abspath = os.path.abspath(terrain_abspath)
        cache_key = _hfield_cache_key(terrain_abspath, resolution, (cfg.size[0], cfg.size[1]))
        cache_path = _hfield_disk_cache_path(
            terrain_abspath, cache_key, str(cfg.collision_hfield_cache_dirname)
        )

    height: np.ndarray | None = None

    # 1. In-memory cache hit.
    if cache_key is not None:
        height = _HFIELD_HEIGHT_CACHE.get(cache_key)

    # 2. Disk cache hit.
    if height is None and cache_key is not None and cfg.collision_hfield_use_disk_cache and os.path.exists(cache_path):
        with np.load(cache_path, allow_pickle=False) as npz:
            loaded = np.asarray(npz["height"], dtype=np.float64)
            if loaded.shape == (nrow, ncol):
                height = loaded
                _HFIELD_HEIGHT_CACHE[cache_key] = height

    # 3. Compute via ray-casting.
    if height is None:
        raycast_backend = str(cfg.collision_hfield_raycast_backend)
        if raycast_backend not in ("cpu", "gpu"):
            raise ValueError(
                "collision_hfield_raycast_backend must be 'cpu' or 'gpu'. "
                f"Got: {raycast_backend!r}"
            )

        if raycast_backend == "gpu":
            gpu_device = str(cfg.collision_hfield_gpu_device)
            gpu_batch_size = int(cfg.collision_hfield_gpu_batch_size)
            raw = _raycast_hfield_gpu(
                terrain_mesh=terrain_mesh,
                ray_origins=ray_origins,
                n_rays=n_rays,
                gpu_device=gpu_device,
                batch_size=gpu_batch_size,
            ).reshape(nrow, ncol)
        else:
            num_workers = int(cfg.collision_hfield_num_workers)
            if num_workers <= 0:
                num_workers = max(os.cpu_count() or 1, 1)
            raw = _raycast_hfield_parallel(
                terrain_mesh=terrain_mesh,
                ray_origins=ray_origins,
                n_rays=n_rays,
                num_workers=num_workers,
            ).reshape(nrow, ncol)

        hit_mask = ~np.isnan(raw)
        if np.any(hit_mask):
            hit_z_min = float(np.min(raw[hit_mask]))
            hit_z_max = float(np.max(raw[hit_mask]))
        else:
            hit_z_min = mesh_z_min
            hit_z_max = mesh_z_min

        sink_miss_cells = bool(cfg.collision_hfield_sink_miss_cells)
        if sink_miss_cells:
            # Motion-matched terrain may intentionally contain out-of-mesh areas:
            # keep old behavior and sink misses deep below the surface.
            terrain_height_range = max(hit_z_max - hit_z_min, 1.0)
            sink_depth = terrain_height_range * 10.0
            height = np.where(hit_mask, raw, hit_z_min - sink_depth)
        else:
            # For tiled parkour terrains, filling misses with nearest valid heights avoids
            # seam cracks caused by sparse ray miss cells near tile boundaries.
            height = _fill_missing_heights_with_nearest(raw, hit_mask, fallback_value=hit_z_min)

        # Store in memory cache.
        if cache_key is not None:
            _HFIELD_HEIGHT_CACHE[cache_key] = height

        # Store on disk cache.
        if cache_key is not None and cfg.collision_hfield_use_disk_cache and cache_path is not None:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            tmp_path = f"{cache_path}.{uuid.uuid4().hex}.tmp.npz"
            np.savez_compressed(tmp_path, height=height.astype(np.float32))
            os.replace(tmp_path, cache_path)
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    stitch_edges = bool(cfg.collision_hfield_stitch_edges)
    stitch_border_pixels = int(cfg.collision_hfield_stitch_border_pixels)
    if stitch_edges and stitch_border_pixels > 0:
        max_border = min(nrow // 2, ncol // 2)
        stitch_border_pixels = max(1, min(stitch_border_pixels, max_border))
        stitch_height = cfg.collision_hfield_stitch_height
        if stitch_height is None:
            edge_values = np.concatenate([
                height[:stitch_border_pixels, :].reshape(-1),
                height[-stitch_border_pixels:, :].reshape(-1),
                height[:, :stitch_border_pixels].reshape(-1),
                height[:, -stitch_border_pixels:].reshape(-1),
            ])
            edge_values = edge_values[np.isfinite(edge_values)]
            if edge_values.size > 0:
                stitch_height = float(np.mean(edge_values))
            else:
                stitch_height = float(np.min(height))
        else:
            stitch_height = float(stitch_height)
        height[:stitch_border_pixels, :] = stitch_height
        height[-stitch_border_pixels:, :] = stitch_height
        height[:, :stitch_border_pixels] = stitch_height
        height[:, -stitch_border_pixels:] = stitch_height

    elevation_min = float(np.min(height))
    elevation_max = float(np.max(height))
    elevation_range = max(elevation_max - elevation_min, 1.0e-6)
    normalized_height = (height - elevation_min) / elevation_range
    base_thickness = max(
        elevation_range * cfg.collision_hfield_base_thickness_ratio,
        1.0e-6,
    )

    hfield_name = f"motion_matched_hfield_t{terrain_idx}_{uuid.uuid4().hex}"
    hfield = spec.add_hfield(
        name=hfield_name,
        size=[
            cfg.size[0] / 2,
            cfg.size[1] / 2,
            elevation_range,
            base_thickness,
        ],
        nrow=nrow,
        ncol=ncol,
        userdata=normalized_height.astype(np.float32).reshape(-1).tolist(),
    )
    # hfield pos is its bottom-centre in the terrain body frame.
    # MuJoCo hfield: the surface at normalized_height h sits at
    #   pos.z + base_thickness + h * elevation_range
    # We want the hit cells (normalized near 1) to align with the visual mesh.
    # hit_z_max corresponds to normalized_height = (hit_z_max - elevation_min) / elevation_range.
    # The actual world z of the top surface = pos.z + base_thickness + elevation_range.
    # We need: pos.z + base_thickness + elevation_range = elevation_max
    # => pos.z = elevation_max - base_thickness - elevation_range = elevation_min - base_thickness
    hfield_geom = spec.body("terrain").add_geom(
        type=mujoco.mjtGeom.mjGEOM_HFIELD,
        hfieldname=hfield_name,
        pos=(cfg.size[0] / 2, cfg.size[1] / 2, elevation_min - base_thickness),
    )
    # group=3: invisible in all default render groups; collision is still active.
    hfield_geom.group = 3
    hfield_geom.rgba[:] = (0.0, 0.0, 0.0, 0.0)
    return TerrainGeometry(geom=hfield_geom, hfield=hfield)


def _detect_floor_z(height_array: np.ndarray, percentile: float = 10.0) -> float:
    """Detect floor z-level from heightfield by finding the dominant low z-region.

    For scanned concave terrains (floor + walls), the floor is typically
    the large flat region at the lowest z-values. We use a low percentile
    of the height distribution to robustly estimate the floor level.

    Args:
        height_array: 2D array of z-heights from ray-casting.
        percentile: Percentile to use for floor detection (default 10%).

    Returns:
        Estimated floor z-level.
    """
    # Filter out NaN and very low outliers (sunk miss cells)
    valid_heights = height_array[~np.isnan(height_array)]
    if len(valid_heights) == 0:
        return 0.0

    # Use percentile to find floor level (robust to walls)
    floor_z = float(np.percentile(valid_heights, percentile))
    return floor_z


def _create_visual_hfield_from_mesh(
    cfg: mesh_terrains_cfg.STLHeightfieldTerrainCfg,
    spec: mujoco.MjSpec,
    terrain_mesh: trimesh.Trimesh,
    terrain_idx: int,
    terrain_abspath: str | None = None,
) -> tuple[TerrainGeometry, float, float, float]:
    """Create a visual+collision heightfield from terrain mesh by ray-casting.

    Returns:
        Tuple of (TerrainGeometry, elevation_min, elevation_max, floor_z)
    """
    resolution = float(cfg.hfield_resolution)
    if resolution <= 0.0:
        raise ValueError("hfield_resolution must be greater than zero.")

    nrow = max(int(np.round(cfg.size[0] / resolution)) + 1, 2)
    ncol = max(int(np.round(cfg.size[1] / resolution)) + 1, 2)
    x = np.linspace(0.0, cfg.size[0], nrow, dtype=np.float64)
    y = np.linspace(0.0, cfg.size[1], ncol, dtype=np.float64)
    xx, yy = np.meshgrid(x, y, indexing="ij")

    mesh_z_max = float(terrain_mesh.bounds[1, 2])
    mesh_z_min = float(terrain_mesh.bounds[0, 2])
    ray_origin_z = mesh_z_max + 1.0

    n_rays = nrow * ncol
    ray_origins = np.column_stack([
        xx.reshape(-1),
        yy.reshape(-1),
        np.full(n_rays, ray_origin_z, dtype=np.float64),
    ])

    # --- Cache lookup ---
    cache_key: tuple | None = None
    cache_path: str | None = None
    if terrain_abspath is not None:
        terrain_abspath = os.path.abspath(terrain_abspath)
        cache_key = _hfield_cache_key(terrain_abspath, resolution, (cfg.size[0], cfg.size[1]))
        cache_path = _hfield_disk_cache_path(
            terrain_abspath, cache_key, str(cfg.hfield_cache_dirname)
        )

    height: np.ndarray | None = None

    # 1. In-memory cache hit.
    if cache_key is not None:
        height = _HFIELD_HEIGHT_CACHE.get(cache_key)

    # 2. Disk cache hit.
    if height is None and cache_key is not None and cfg.hfield_use_disk_cache and cache_path and os.path.exists(cache_path):
        with np.load(cache_path, allow_pickle=False) as npz:
            loaded = np.asarray(npz["height"], dtype=np.float64)
            if loaded.shape == (nrow, ncol):
                height = loaded
                _HFIELD_HEIGHT_CACHE[cache_key] = height

    # 3. Compute via parallel ray-casting.
    if height is None:
        num_workers = int(cfg.hfield_num_workers)
        if num_workers <= 0:
            num_workers = max(os.cpu_count() or 1, 1)

        raw = _raycast_hfield_parallel(
            terrain_mesh=terrain_mesh,
            ray_origins=ray_origins,
            n_rays=n_rays,
            num_workers=num_workers,
        ).reshape(nrow, ncol)

        hit_mask = ~np.isnan(raw)
        if np.any(hit_mask):
            hit_z_min = float(np.min(raw[hit_mask]))
            hit_z_max = float(np.max(raw[hit_mask]))
        else:
            hit_z_min = mesh_z_min
            hit_z_max = mesh_z_min

        sink_miss_cells = bool(cfg.hfield_sink_miss_cells)
        if sink_miss_cells:
            terrain_height_range = max(hit_z_max - hit_z_min, 1.0)
            sink_depth = terrain_height_range * 10.0
            height = np.where(hit_mask, raw, hit_z_min - sink_depth)
        else:
            height = _fill_missing_heights_with_nearest(raw, hit_mask, fallback_value=hit_z_min)

        if cache_key is not None:
            _HFIELD_HEIGHT_CACHE[cache_key] = height

        if cache_key is not None and cfg.hfield_use_disk_cache and cache_path is not None:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            tmp_path = f"{cache_path}.{uuid.uuid4().hex}.tmp.npz"
            np.savez_compressed(tmp_path, height=height.astype(np.float32))
            os.replace(tmp_path, cache_path)
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    # Detect floor z-level from the heightfield data
    floor_z = _detect_floor_z(height, percentile=10.0)

    # Shift heights so floor is at z=0 (before applying user offset)
    height = height - floor_z

    # Apply user-specified floor z offset for final alignment
    height = height + cfg.hfield_floor_z_offset

    elevation_min = float(np.min(height))
    elevation_max = float(np.max(height))
    elevation_range = max(elevation_max - elevation_min, 1.0e-6)
    normalized_height = (height - elevation_min) / elevation_range
    # Use fixed base thickness (small value sufficient for collision stability)
    base_thickness = max(float(cfg.hfield_base_thickness), 1.0e-6)
    # Store original floor_z for return
    detected_floor_z = floor_z

    hfield_name = f"stl_hfield_t{terrain_idx}_{uuid.uuid4().hex}"
    hfield = spec.add_hfield(
        name=hfield_name,
        size=[
            cfg.size[0] / 2,
            cfg.size[1] / 2,
            elevation_range,
            base_thickness,
        ],
        nrow=nrow,
        ncol=ncol,
        userdata=normalized_height.astype(np.float32).reshape(-1).tolist(),
    )

    # Create height-based coloring for visualization
    unique_id = uuid.uuid4().hex
    texture_size = 128
    texture_name = f"hf_texture_{unique_id}"
    texture = spec.add_texture(
        name=texture_name,
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        width=texture_size,
        height=texture_size,
    )

    texture_elevation = ndimage.zoom(
        normalized_height,
        (texture_size / normalized_height.shape[0], texture_size / normalized_height.shape[1]),
        order=1,
    )
    texture_elevation = np.asarray(texture_elevation)

    # Height-based coloring: blue-green-yellow gradient
    hue = 0.5 - texture_elevation * 0.45
    saturation = 0.6 - texture_elevation * 0.2
    value = 0.4 + texture_elevation * 0.3

    c = value * saturation
    x_color = c * (1 - np.abs((hue * 6) % 2 - 1))
    m = value - c

    hue_sector = (hue * 6).astype(int) % 6

    r = np.zeros_like(hue)
    g = np.zeros_like(hue)
    b = np.zeros_like(hue)

    mask = hue_sector == 0
    r[mask] = c[mask]
    g[mask] = x_color[mask]

    mask = hue_sector == 1
    r[mask] = x_color[mask]
    g[mask] = c[mask]

    mask = hue_sector == 2
    g[mask] = c[mask]
    b[mask] = x_color[mask]

    mask = hue_sector == 3
    g[mask] = x_color[mask]
    b[mask] = c[mask]

    mask = hue_sector == 4
    r[mask] = x_color[mask]
    b[mask] = c[mask]

    mask = hue_sector == 5
    r[mask] = c[mask]
    b[mask] = x_color[mask]

    r += m
    g += m
    b += m

    rgb_data = np.stack([r, g, b], axis=-1)
    rgb_data = (rgb_data * 255).astype(np.uint8)
    rgb_data = np.flipud(rgb_data)
    texture.data = rgb_data.tobytes()

    material_name = f"hf_material_{unique_id}"
    material = spec.add_material(name=material_name)
    material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = texture_name

    # hfield geom: visible (group 0/2) and collision enabled
    # Position: center of terrain, with z at bottom of heightfield
    hfield_geom = spec.body("terrain").add_geom(
        type=mujoco.mjtGeom.mjGEOM_HFIELD,
        hfieldname=hfield_name,
        pos=(cfg.size[0] / 2, cfg.size[1] / 2, elevation_min - base_thickness),
        material=material_name,
    )
    # Group 0 for default visibility, collision enabled by default
    hfield_geom.group = 0

    return TerrainGeometry(geom=hfield_geom, hfield=hfield), elevation_min, elevation_max, detected_floor_z


def stl_heightfield_terrain(
    cfg: mesh_terrains_cfg.STLHeightfieldTerrainCfg,
    difficulty: float,
    spec: mujoco.MjSpec,
    rng: np.random.Generator,
) -> TerrainOutput:
    """Generate a terrain from STL using heightfield for both visualization and collision.

    This function loads an STL file and converts it to a MuJoCo heightfield,
    suitable for 3D-scanned concave terrains (e.g., floor with thin walls).

    Args:
        cfg: Configuration for STL heightfield terrain.
        difficulty: Difficulty level (0-1) used to select terrain from metadata.
        spec: MuJoCo spec to add terrain geometry to.
        rng: Random number generator (unused).

    Returns:
        TerrainOutput with spawn origin and heightfield geometry.
    """
    del rng

    # Load the YAML file containing terrain and motion data
    with open(cfg.metadata_yaml) as file:
        data = yaml.safe_load(file)

    terrains = data["terrains"]

    terrain_idx = int(np.clip(difficulty * len(terrains), 0, len(terrains) - 1))
    selected_terrain = terrains[terrain_idx]

    terrain_file = selected_terrain["terrain_file"]
    terrain_abspath = os.path.join(cfg.path, terrain_file)
    terrain_size = (float(cfg.size[0]), float(cfg.size[1]))

    # Load STL mesh (without border_height alignment - we'll handle floor alignment ourselves)
    terrain_mesh = trimesh.load(terrain_abspath, force="mesh")

    # Crop terrain mesh by cfg.size
    terrain_mesh = crop_terrain_mesh_aabb(
        terrain_mesh,
        x_max=terrain_size[0] / 2,
        x_min=-terrain_size[0] / 2,
        y_max=terrain_size[1] / 2,
        y_min=-terrain_size[1] / 2,
    )
    terrain_mesh.remove_unreferenced_vertices()
    trimesh.repair.fix_winding(terrain_mesh)
    trimesh.repair.fix_normals(terrain_mesh, multibody=True)

    # Move mesh to terrain generator convention: (size[0]/2, size[1]/2, 0)
    # Note: We do NOT apply z-offset here; floor alignment happens in heightfield generation
    move_terrain_transform = np.eye(4)
    move_terrain_transform[:2, 3] = np.asarray(terrain_size, dtype=np.float64) / 2.0
    terrain_mesh.apply_transform(move_terrain_transform)

    # Create heightfield for both visualization and collision
    hfield_geom, elevation_min, elevation_max, floor_z = _create_visual_hfield_from_mesh(
        cfg, spec, terrain_mesh, terrain_idx, terrain_abspath
    )

    # Origin at terrain center, floor at z=0 (after floor alignment in hfield generation)
    # The floor has been shifted to z=0 in _create_visual_hfield_from_mesh
    origin = np.array([cfg.size[0] / 2, cfg.size[1] / 2, 0.0])

    print(
        f"[STLHeightfieldTerrain] terrain_idx={terrain_idx}, file={terrain_file}, "
        f"floor_z_detected={floor_z:.3f}, elevation_range=[{elevation_min:.3f}, {elevation_max:.3f}]"
    )

    return TerrainOutput(origin=origin, geometries=[hfield_geom])


def motion_matched_terrain(
    cfg: mesh_terrains_cfg.MotionMatchedTerrainCfg,
    difficulty: float,
    spec: mujoco.MjSpec,
    rng: np.random.Generator,
) -> TerrainOutput:
    """Generate a motion-matched terrain based on the difficulty level and configuration.

    Args:
        difficulty (float): The difficulty level for the terrain.
        cfg (mesh_terrains_cfg.MotionMatchedTerrainCfg): Configuration for the motion-matched terrain.

    Returns:
        TerrainOutput: Spawn origin and generated MuJoCo terrain geometry.
    """
    del rng

    # Load the YAML file containing terrain and motion data
    with open(cfg.metadata_yaml) as file:
        data = yaml.safe_load(file)

    # Extract terrains and motions from the YAML data
    terrains = data["terrains"]  # order matters

    if cfg.collision_coacd:
        # Precompute missing CoACD caches once to better utilize multi-core CPU.
        _prewarm_coacd_disk_cache(cfg, terrains)

    terrain_idx = int(np.clip(difficulty * len(terrains), 0, len(terrains) - 1))
    selected_terrain = terrains[terrain_idx]

    terrain_file = selected_terrain["terrain_file"]
    terrain_abspath = os.path.join(cfg.path, terrain_file)
    terrain_size = (float(cfg.size[0]), float(cfg.size[1]))
    terrain_mesh, border_height = _load_motion_matched_terrain_mesh(
        terrain_abspath=terrain_abspath,
        terrain_file=terrain_file,
        size=terrain_size,
    )
    origin = np.array([cfg.size[0] / 2, cfg.size[1] / 2, -border_height])

    skip_source_mesh_import = bool(cfg.collision_coacd) and (
        bool(cfg.collision_coacd_hide_source_visual_mesh)
        or bool(cfg.collision_coacd_visualize_collision_hulls)
    )
    geometries: list[TerrainGeometry] = []
    geom = None
    if not skip_source_mesh_import:
        # Encode terrain index in mesh name so runtime can recover terrain-to-origin mapping
        # without relying on custom terrain importer compatibility fields.
        mesh_name = f"motion_matched_t{terrain_idx}_{uuid.uuid4().hex}"
        spec.add_mesh(
            name=mesh_name,
            uservert=np.asarray(terrain_mesh.vertices, dtype=np.float32).reshape(-1).tolist(),
            userface=np.asarray(terrain_mesh.faces, dtype=np.int32).reshape(-1).tolist(),
        )
        geom = spec.body("terrain").add_geom(
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=mesh_name,
            pos=(0.0, 0.0, 0.0),
        )
        geometries.append(TerrainGeometry(geom=geom))
    enabled_modes = sum([
        bool(cfg.face_box_collision),
        bool(cfg.collision_hfield),
        bool(cfg.collision_coacd),
    ])
    if enabled_modes > 1:
        raise ValueError(
            "face_box_collision, collision_hfield, and collision_coacd are mutually exclusive. "
            "Enable only one of them."
        )
    if cfg.collision_coacd:
        # Keep mesh for rendering/depth sensing but route physics contacts to CoACD convex hulls.
        # Root cause: MuJoCo's polyhedral mesh collision treats the interior of any closed
        # mesh as solid, reporting negative-dist contacts for objects near the inner surface.
        # CoACD decomposes the concave mesh into approximate convex hulls for correct collision.
        if geom is not None:
            geom.contype = 0
            geom.conaffinity = 0
            if bool(cfg.collision_coacd_visualize_collision_hulls):
                # Visualization mode: hide source mesh, render CoACD hulls.
                geom.group = 4
                geom.rgba[:] = (0.0, 0.0, 0.0, 0.0)
            else:
                source_geom_group = int(cfg.collision_coacd_source_visual_geom_group)
                if not 0 <= source_geom_group <= 5:
                    raise ValueError(
                        "collision_coacd_source_visual_geom_group must be in [0, 5]. "
                        f"Got: {source_geom_group}"
                    )
                geom.group = source_geom_group
                if bool(cfg.collision_coacd_hide_source_visual_mesh):
                    geom.rgba[:] = (0.0, 0.0, 0.0, 0.0)
        geometries.extend(
            _add_collision_coacd(cfg, spec, terrain_mesh, terrain_idx, terrain_abspath)
        )
    elif cfg.face_box_collision:
        # Keep mesh for rendering/depth sensing but route physics contacts to per-face boxes.
        # Root cause: MuJoCo's polyhedral mesh collision treats the interior of any closed
        # mesh as solid, reporting negative-dist contacts for objects near the inner surface.
        # Per-face box geoms placed on the outward side give correct surface-only collision.
        geom.contype = 0
        geom.conaffinity = 0
        geom.group = 2
        geometries.extend(
            _add_collision_face_boxes(cfg, spec, terrain_mesh)
        )
    elif cfg.collision_hfield:
        # Keep mesh for rendering/depth sensing but route physics contacts to hfield.
        geom.contype = 0
        geom.conaffinity = 0
        geom.group = 2
        geometries.append(
            _add_collision_hfield_from_mesh(cfg, spec, terrain_mesh, terrain_idx, terrain_abspath)
        )
    return TerrainOutput(origin=origin, geometries=geometries)


@generate_wall
def floating_box_terrain(
    difficulty: float, cfg: object
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generates a floating box terrain."""

    # resolve the terrain configuration
    # height of the floating box above the ground
    if isinstance(cfg.floating_height, (tuple, list)):
        floating_height = cfg.floating_height[1] - difficulty * (cfg.floating_height[1] - cfg.floating_height[0])
    else:
        floating_height = cfg.floating_height

    # length of the floating box
    if isinstance(cfg.box_length, (tuple, list)):
        box_length = cfg.box_length[1] - difficulty * (cfg.box_length[1] - cfg.box_length[0])
    else:
        box_length = cfg.box_length

    # height of the floating box
    if isinstance(cfg.box_height, (tuple, list)):
        box_height = np.random.uniform(*cfg.box_height)
    else:
        box_height = cfg.box_height

    # width of the floating box
    if cfg.box_width is None:
        box_width = cfg.size[0]
    else:
        box_width = cfg.box_width

    # initialize the list of meshes
    meshes_list = list()

    # extract quantities
    total_height = floating_height + box_height
    # constants for terrain generation
    terrain_height = 0.0

    # generate the box mesh
    dim = (box_width, box_length, box_height)
    pos = (0.5 * cfg.size[0], 0.5 * cfg.size[1], floating_height + box_height / 2)
    box_mesh = trimesh.creation.box(dim, trimesh.transformations.translation_matrix(pos))
    meshes_list.append(box_mesh)

    # generate the ground

    if cfg.perlin_cfg is None:
        dim = (cfg.size[0], cfg.size[1], terrain_height)
        pos = (0.5 * cfg.size[0], 0.5 * cfg.size[1], -terrain_height / 2)
        ground_mesh = trimesh.creation.box(dim, trimesh.transformations.translation_matrix(pos))
        meshes_list.append(ground_mesh)
    else:
        clean_ground_height_field = np.zeros(
            (int(cfg.size[0] / cfg.horizontal_scale) + 1, int(cfg.size[1] / cfg.horizontal_scale) + 1), dtype=np.int16
        )
        perlin_cfg = cfg.perlin_cfg
        perlin_cfg.size = cfg.size
        perlin_cfg.horizontal_scale = cfg.horizontal_scale
        perlin_cfg.vertical_scale = cfg.vertical_scale
        perlin_cfg.slope_threshold = cfg.slope_threshold
        perlin_noise = generate_perlin_noise(
            difficulty,
            perlin_cfg,  # type: ignore[arg-type]
        )
        h, w = perlin_noise.shape
        ground_h, ground_w = clean_ground_height_field.shape
        pad_h_left = max(0, (ground_h - h) // 2)
        pad_h_right = max(0, ground_h - h - pad_h_left)
        pad_w_left = max(0, (ground_w - w) // 2)
        pad_w_right = max(0, ground_w - w - pad_w_left)
        pad_width = ((pad_h_left, pad_h_right), (pad_w_left, pad_w_right))
        perlin_noise = np.pad(perlin_noise, pad_width, mode="constant", constant_values=0)
        if cfg.no_perlin_at_obstacle is True:
            box_width_px = int(box_width / cfg.horizontal_scale)
            box_length_px = int(box_length / cfg.horizontal_scale)
            box_width_start_px = int((cfg.size[0] - box_width) / 2 / cfg.horizontal_scale)
            box_length_start_px = int((cfg.size[1] - box_length) / 2 / cfg.horizontal_scale)
            perlin_noise[
                box_width_start_px : box_width_start_px + box_width_px,
                box_length_start_px : box_length_start_px + box_length_px,
            ] = 0
        ground_height_field = clean_ground_height_field + perlin_noise
        # convert to trimesh
        vertices, triangles = convert_height_field_to_mesh(
            ground_height_field, cfg.horizontal_scale, cfg.vertical_scale, cfg.slope_threshold
        )
        ground_mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
        meshes_list.append(ground_mesh)

    # specify the origin of the terrain
    origin = np.array([pos[0], pos[1], total_height])

    return meshes_list, origin


@generate_wall
def random_multi_box_terrain(
    difficulty: float, cfg: object
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generates a terrain containing multiple boxes with random size and orientation."""

    box_height_range = cfg.box_height_range
    box_length_range = cfg.box_length_range
    box_width_range = cfg.box_width_range

    if isinstance(cfg.box_height_mean, (tuple, list)):
        if cfg.box_height_mean[0] < box_height_range:
            raise RuntimeError("The minimum box height mean is smaller than the box height half range.")
        box_height_mean = cfg.box_height_mean[0] + difficulty * (cfg.box_height_mean[1] - cfg.box_height_mean[0])
    else:
        box_height_mean = cfg.box_height_mean
        if box_height_mean < box_height_range:
            raise RuntimeError("The minimum box height mean is smaller than the box height half range.")

    if isinstance(cfg.box_length_mean, (tuple, list)):
        if cfg.box_length_mean[0] < box_length_range:
            raise RuntimeError("The minimum box length mean is smaller than the box length half range.")
        box_length_mean = cfg.box_length_mean[0] + difficulty * (cfg.box_length_mean[1] - cfg.box_length_mean[0])
    else:
        box_length_mean = cfg.box_length_mean
        if box_length_mean < box_length_range:
            raise RuntimeError("The minimum box length mean is smaller than the box length half range.")

    if isinstance(cfg.box_width_mean, (tuple, list)):
        if cfg.box_width_mean[0] < box_width_range:
            raise RuntimeError("The minimum box width mean is smaller than the box width half range.")
        box_width_mean = cfg.box_width_mean[0] + difficulty * (cfg.box_width_mean[1] - cfg.box_width_mean[0])
    else:
        box_width_mean = cfg.box_width_mean
        if box_width_mean < box_width_range:
            raise RuntimeError("The minimum box width mean is smaller than the box width half range.")

    generation_ratio = cfg.generation_ratio

    width = cfg.size[0]
    length = cfg.size[1]

    mesh_list = []

    num_boxes = int(generation_ratio * (width * length) / (box_length_mean * box_width_mean))
    num_boxes = max(1, num_boxes)
    if cfg.perlin_cfg is None:
        dim = (cfg.size[0], cfg.size[1], 0.0)
        pos = (0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.0)
        ground_mesh = trimesh.creation.box(dim, trimesh.transformations.translation_matrix(pos))
        mesh_list.append(ground_mesh)
    else:
        clean_ground_height_field = np.zeros(
            (int(cfg.size[0] / cfg.horizontal_scale) + 1, int(cfg.size[1] / cfg.horizontal_scale) + 1), dtype=np.int16
        )
        perlin_cfg = cfg.perlin_cfg
        perlin_cfg.size = cfg.size
        perlin_cfg.horizontal_scale = cfg.horizontal_scale
        perlin_cfg.vertical_scale = cfg.vertical_scale
        perlin_cfg.slope_threshold = cfg.slope_threshold
        perlin_noise = generate_perlin_noise(
            difficulty,
            perlin_cfg,  # type: ignore[arg-type]
        )
        h, w = perlin_noise.shape
        ground_h, ground_w = clean_ground_height_field.shape
        pad_h_left = max(0, (ground_h - h) // 2)
        pad_h_right = max(0, ground_h - h - pad_h_left)
        pad_w_left = max(0, (ground_w - w) // 2)
        pad_w_right = max(0, ground_w - w - pad_w_left)
        pad_width = ((pad_h_left, pad_h_right), (pad_w_left, pad_w_right))
        perlin_noise = np.pad(perlin_noise, pad_width, mode="constant", constant_values=0)
        ground_height_field = clean_ground_height_field + perlin_noise
        # convert to trimesh
        vertices, triangles = convert_height_field_to_mesh(
            ground_height_field, cfg.horizontal_scale, cfg.vertical_scale, cfg.slope_threshold
        )
        ground_mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
        mesh_list.append(ground_mesh)

    if cfg.box_perlin_cfg is not None and cfg.no_perlin_at_obstacle is False:
        box_perlin_cfg = cfg.box_perlin_cfg
        box_perlin_cfg.horizontal_scale = (
            cfg.horizontal_scale if box_perlin_cfg.horizontal_scale is None else box_perlin_cfg.horizontal_scale
        )
        box_perlin_cfg.vertical_scale = (
            cfg.vertical_scale if box_perlin_cfg.vertical_scale is None else box_perlin_cfg.vertical_scale
        )
        box_perlin_cfg.slope_threshold = (
            cfg.slope_threshold if box_perlin_cfg.slope_threshold is None else box_perlin_cfg.slope_threshold
        )

    platform_width = cfg.platform_width

    for i in range(num_boxes):
        box_width = box_width_mean + np.random.uniform(-1, 1) * box_width_range
        box_length = box_length_mean + np.random.uniform(-1, 1) * box_length_range
        box_height = box_height_mean + np.random.uniform(-1, 1) * box_height_range
        dim = (box_width, box_length, box_height)
        x = np.random.uniform(box_width / 2, width - box_width / 2)
        y = np.random.uniform(box_length / 2, length - box_length / 2)
        if (
            x > width / 2 - platform_width / 2 - box_width / 2 and x < width / 2 + platform_width / 2 + box_width / 2
        ) and (
            y > length / 2 - platform_width / 2 - box_length / 2
            and y < length / 2 + platform_width / 2 + box_length / 2
        ):
            continue
        pos = (x, y, box_height / 2)
        theta = np.random.uniform(0, 2 * np.pi)
        translation_matrix = trimesh.transformations.translation_matrix(pos)
        rotation_matrix = trimesh.transformations.rotation_matrix(theta, (0, 0, 1))
        transform = translation_matrix @ rotation_matrix
        box_mesh = trimesh.creation.box(extents=dim)
        # top_z=box_mesh.vertices[:, 2].max()
        # top_face_mask=np.all(box_mesh.vertices[box_mesh.faces][:,:,2] == top_z, axis=1)
        # box_mesh.update_faces(~top_face_mask)
        # box_mesh.remove_unreferenced_vertices()
        box_mesh.apply_transform(transform)
        mesh_list.append(box_mesh)
        if cfg.box_perlin_cfg is not None and cfg.no_perlin_at_obstacle is False:
            box_perlin_cfg.size = (box_width, box_length)
            perlin_noise = generate_perlin_noise(
                difficulty,
                box_perlin_cfg,  # type: ignore[arg-type]
            )
            vertices, triangles = convert_height_field_to_mesh(
                perlin_noise,
                box_perlin_cfg.horizontal_scale,
                box_perlin_cfg.vertical_scale,
                box_perlin_cfg.slope_threshold,
            )
            box_noise = trimesh.Trimesh(vertices=vertices, faces=triangles)
            center_offset = (-box_width / 2, -box_length / 2, 0)
            center_translation = trimesh.transformations.translation_matrix(center_offset)
            box_noise.apply_transform(center_translation)
            noise_pos = (x, y, box_height)
            translation_matrix = trimesh.transformations.translation_matrix(noise_pos)
            transform = translation_matrix @ rotation_matrix
            box_noise.apply_transform(transform)
            mesh_list.append(box_noise)

    origin = np.array([0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.0])

    return mesh_list, origin
