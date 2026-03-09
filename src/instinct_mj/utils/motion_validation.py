"""Tracking motion file validation and discovery utilities."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

_TRACKING_MOTION_REQUIRED_KEYS = (
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)


_DEFAULT_DATASET_CANDIDATES = (
    # NOTE: Change these defaults if your local datasets root lives elsewhere, or
    # set `INSTINCT_DATASETS_ROOT` to override them without editing this file.
    "~/Xyk/Datasets",
    "~/Datasets",
)


def resolve_datasets_root() -> Path:
    """Resolve datasets root from override or known local workspace candidates."""
    override = os.environ.get("INSTINCT_DATASETS_ROOT", "").strip()
    if override:
        return Path(override).expanduser().resolve()

    for candidate in _DEFAULT_DATASET_CANDIDATES:
        candidate_path = Path(candidate).expanduser()
        if candidate_path.exists() and candidate_path.is_dir():
            return candidate_path.resolve()

    # Keep a deterministic fallback even when no candidate exists yet.
    return Path(_DEFAULT_DATASET_CANDIDATES[0]).expanduser().resolve()


def _tracking_motion_missing_keys(motion_path: Path) -> tuple[str, ...]:
    with np.load(motion_path) as data:
        keys = set(data.files)
    missing = tuple(key for key in _TRACKING_MOTION_REQUIRED_KEYS if key not in keys)
    return missing


def validate_tracking_motion_file(motion_path: Path) -> None:
    """Validate motion npz schema expected by `mjlab.tasks.tracking.mdp.MotionLoader`."""
    if not motion_path.exists() or not motion_path.is_file():
        raise FileNotFoundError(f"Motion file not found: {motion_path}")
    missing_keys = _tracking_motion_missing_keys(motion_path)
    if missing_keys:
        raise ValueError(
            "Motion file schema is incompatible with mjlab tracking MotionLoader: "
            f"{motion_path}\n"
            f"Missing keys: {list(missing_keys)}\n"
            f"Expected keys: {list(_TRACKING_MOTION_REQUIRED_KEYS)}"
        )
