"""Prepare OmniRetargeting outputs for InstinctMJ motion-matched terrain loading.

OmniRetargeting retargets on a scaled terrain mesh, but only writes the motion
``.npz`` to disk.  InstinctMJ expects the terrain mesh file and the motion
``base_pos_w`` values to share the same local, terrain-centered frame.

This script creates a processed dataset root where each case contains:
  - a scaled + XY recentered ``scene_tsdf.obj``
  - a motion ``.npz`` with the same XY recenter applied to ``base_pos_w``
  - a per-case ``metadata.yaml``

The output root also gets a top-level ``metadata.yaml`` that can be consumed by
``TerrainMotionCfg`` / ``MotionMatchedTerrainCfg`` directly.
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass

import numpy as np
import trimesh
import yaml


@dataclass(frozen=True)
class CaseScale:
    name: str
    terrain_scale: float


def _parse_case_scale(text: str) -> CaseScale:
    if "=" not in text:
        raise argparse.ArgumentTypeError(f"Invalid --case-scale '{text}'. Expected the form CASE=SCALE.")
    case_name, scale_text = text.split("=", 1)
    case_name = case_name.strip()
    if not case_name:
        raise argparse.ArgumentTypeError("Case name must not be empty.")
    try:
        terrain_scale = float(scale_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid terrain scale '{scale_text}' for case '{case_name}'.") from exc
    return CaseScale(name=case_name, terrain_scale=terrain_scale)


def _load_motion_payload(path: str) -> dict[str, np.ndarray | list[str] | float]:
    payload = np.load(path, allow_pickle=True)
    joint_names = payload["joint_names"].tolist()
    joint_pos = np.asarray(payload["joint_pos"])
    if joint_names and joint_names[0] == "floating_base_joint":
        joint_names = joint_names[1:]

    if len(joint_names) != joint_pos.shape[1]:
        raise ValueError(
            f"Joint name count mismatch in {path}: {len(joint_names)} names vs {joint_pos.shape[1]} joint columns."
        )

    return {
        "framerate": float(np.asarray(payload["framerate"]).item()),
        "joint_names": joint_names,
        "joint_pos": joint_pos,
        "base_pos_w": np.asarray(payload["base_pos_w"]).copy(),
        "base_quat_w": np.asarray(payload["base_quat_w"]),
    }


def _round_up(value: float, quantum: float = 0.5) -> float:
    return math.ceil(value / quantum) * quantum


def _compute_center_xy(
    mesh_scaled: trimesh.Trimesh,
    motion_base_pos_w: np.ndarray,
    center_mode: str,
) -> np.ndarray:
    if center_mode == "mesh_aabb":
        bounds = np.asarray(mesh_scaled.bounds)
        return 0.5 * (bounds[0, :2] + bounds[1, :2])
    if center_mode == "motion_aabb":
        motion_min = motion_base_pos_w[:, :2].min(axis=0)
        motion_max = motion_base_pos_w[:, :2].max(axis=0)
        return 0.5 * (motion_min + motion_max)
    raise ValueError(f"Unsupported center_mode: {center_mode}")


def _write_case_metadata(case_dir: str, case_name: str) -> None:
    metadata = {
        "terrains": [{"terrain_id": 0, "terrain_file": "scene_tsdf.obj"}],
        "motion_files": [{"terrain_id": 0, "motion_file": f"{case_name}-retargeted.npz"}],
    }
    with open(os.path.join(case_dir, "metadata.yaml"), "w") as file:
        yaml.safe_dump(metadata, file, sort_keys=False)


def _write_root_metadata(output_root: str, case_names: list[str]) -> None:
    metadata = {
        "terrains": [],
        "motion_files": [],
    }
    for terrain_id, case_name in enumerate(case_names):
        metadata["terrains"].append(
            {
                "terrain_id": terrain_id,
                "terrain_file": f"{case_name}/scene_tsdf.obj",
            }
        )
        metadata["motion_files"].append(
            {
                "terrain_id": terrain_id,
                "motion_file": f"{case_name}/{case_name}-retargeted.npz",
            }
        )
    with open(os.path.join(output_root, "metadata.yaml"), "w") as file:
        yaml.safe_dump(metadata, file, sort_keys=False)


def _write_preprocess_info(
    case_dir: str,
    case_name: str,
    terrain_scale: float,
    center_xy: np.ndarray,
    mesh_bounds: np.ndarray,
    motion_bounds: np.ndarray,
    terrain_margin: float,
) -> None:
    mesh_extent_xy = mesh_bounds[1, :2] - mesh_bounds[0, :2]
    motion_extent_xy = motion_bounds[1, :2] - motion_bounds[0, :2]
    suggested_size_xy = [
        _round_up(float(max(mesh_extent_xy[0], motion_extent_xy[0]) + terrain_margin)),
        _round_up(float(max(mesh_extent_xy[1], motion_extent_xy[1]) + terrain_margin)),
    ]
    info = {
        "case_name": case_name,
        "terrain_scale": float(terrain_scale),
        "xy_recenter_offset_applied": [float(center_xy[0]), float(center_xy[1])],
        "processed_mesh_bounds_min": mesh_bounds[0].tolist(),
        "processed_mesh_bounds_max": mesh_bounds[1].tolist(),
        "processed_motion_bounds_min": motion_bounds[0].tolist(),
        "processed_motion_bounds_max": motion_bounds[1].tolist(),
        "suggested_terrain_size_xy": suggested_size_xy,
        "notes": [
            "Motion base_pos_w is already terrain-scaled by OmniRetargeting.",
            "This preprocessing applies only the shared XY recenter to motion.",
            "Z is left unchanged so mesh and motion keep the same relative height as OmniRetargeting.",
        ],
    }
    with open(os.path.join(case_dir, "preprocess_info.yaml"), "w") as file:
        yaml.safe_dump(info, file, sort_keys=False)


def prepare_case(
    input_root: str,
    output_root: str,
    case_scale: CaseScale,
    center_mode: str,
    terrain_margin: float,
) -> None:
    case_name = case_scale.name
    input_case_dir = os.path.join(input_root, case_name)
    output_case_dir = os.path.join(output_root, case_name)
    os.makedirs(output_case_dir, exist_ok=True)

    input_mesh = os.path.join(input_case_dir, "scene_tsdf.obj")
    input_motion = os.path.join(input_case_dir, f"{case_name}-retargeted.npz")
    output_mesh = os.path.join(output_case_dir, "scene_tsdf.obj")
    output_motion = os.path.join(output_case_dir, f"{case_name}-retargeted.npz")

    mesh_scaled = trimesh.load(input_mesh, force="mesh")
    mesh_scaled.apply_scale(case_scale.terrain_scale)

    motion_payload = _load_motion_payload(input_motion)
    center_xy = _compute_center_xy(
        mesh_scaled=mesh_scaled,
        motion_base_pos_w=np.asarray(motion_payload["base_pos_w"]),
        center_mode=center_mode,
    )

    mesh_scaled.apply_translation(np.array([-center_xy[0], -center_xy[1], 0.0], dtype=np.float64))

    base_pos_w = np.asarray(motion_payload["base_pos_w"]).copy()
    base_pos_w[:, 0] -= center_xy[0]
    base_pos_w[:, 1] -= center_xy[1]

    mesh_scaled.export(output_mesh)
    np.savez(
        output_motion,
        framerate=motion_payload["framerate"],
        joint_names=np.asarray(motion_payload["joint_names"]),
        joint_pos=motion_payload["joint_pos"],
        base_pos_w=base_pos_w,
        base_quat_w=motion_payload["base_quat_w"],
    )

    _write_case_metadata(output_case_dir, case_name)
    _write_preprocess_info(
        case_dir=output_case_dir,
        case_name=case_name,
        terrain_scale=case_scale.terrain_scale,
        center_xy=center_xy,
        mesh_bounds=np.asarray(mesh_scaled.bounds),
        motion_bounds=np.vstack((base_pos_w.min(axis=0), base_pos_w.max(axis=0))),
        terrain_margin=terrain_margin,
    )

    print(
        f"[prepare_omniretargeting_dataset] {case_name}: "
        f"scale={case_scale.terrain_scale:.6f}, center_xy=({center_xy[0]:.3f}, {center_xy[1]:.3f})"
    )


def main(args: argparse.Namespace) -> None:
    if not args.case_scale:
        raise ValueError("At least one --case-scale must be provided.")

    output_root = os.path.abspath(os.path.expanduser(args.output_root))
    input_root = os.path.abspath(os.path.expanduser(args.input_root))
    os.makedirs(output_root, exist_ok=True)

    processed_case_names: list[str] = []
    for case_scale in args.case_scale:
        prepare_case(
            input_root=input_root,
            output_root=output_root,
            case_scale=case_scale,
            center_mode=args.center_mode,
            terrain_margin=args.terrain_margin,
        )
        processed_case_names.append(case_scale.name)

    _write_root_metadata(output_root, processed_case_names)
    print(f"[prepare_omniretargeting_dataset] Wrote root metadata: {output_root}/metadata.yaml")


def entry_point() -> None:
    parser = argparse.ArgumentParser(description="Prepare OmniRetargeting terrain-motion pairs for InstinctMJ.")
    parser.add_argument(
        "--input-root",
        type=str,
        required=True,
        help="Directory containing one subdirectory per case with raw scene_tsdf.obj and retargeted npz files.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        required=True,
        help="Directory to write the processed InstinctMJ-ready dataset.",
    )
    parser.add_argument(
        "--case-scale",
        type=_parse_case_scale,
        action="append",
        default=[],
        help="Per-case terrain scale in the form CASE=SCALE. Pass multiple times for multiple cases.",
    )
    parser.add_argument(
        "--center-mode",
        type=str,
        default="mesh_aabb",
        choices=["mesh_aabb", "motion_aabb"],
        help="How to choose the shared XY recenter offset.",
    )
    parser.add_argument(
        "--terrain-margin",
        type=float,
        default=1.0,
        help="Extra margin (meters) added when reporting suggested terrain size in preprocess_info.yaml.",
    )
    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    entry_point()
