"""Direct Viser playback for retargeted OMOMO HOI npz files.

This is a data visualization utility: it sets MuJoCo qpos directly from
``base_pos_w``, ``base_quat_w`` and ``joint_pos``.  No RL policy or physics
tracking is involved.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import mujoco
import numpy as np
import trimesh
import viser

from instinct_mj.assets.unitree_g1 import G1_MJCF_PATH
from instinct_mj.tasks.shadowing.perceptive_hoi.config.g1.perceptive_shadowing_cfg import (
    MESH_FILE_PATHS,
    MESH_FILE_SCALES,
)
from mjlab.viewer.viser import ViserMujocoScene


def _resolve_motion_file(path: Path) -> Path:
    if path.is_file():
        return path
    files = sorted(path.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found under {path}")
    return files[0]


def _object_name_from_file(path: Path) -> str | None:
    parts = path.name.split("_")
    if len(parts) < 2:
        return None
    return parts[1]


def _continuous_quat(quat: np.ndarray) -> np.ndarray:
    quat = np.array(quat, dtype=np.float64, copy=True)
    quat /= np.maximum(np.linalg.norm(quat, axis=-1, keepdims=True), 1e-12)
    for frame_idx in range(1, len(quat)):
        if np.dot(quat[frame_idx - 1], quat[frame_idx]) < 0.0:
            quat[frame_idx] *= -1.0
    return quat


def _load_qpos(model: mujoco.MjModel, motion_file: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    motion = np.load(motion_file, allow_pickle=True)
    framerate = float(motion["framerate"])
    base_pos = np.asarray(motion["base_pos_w"], dtype=np.float64)
    base_quat = _continuous_quat(np.asarray(motion["base_quat_w"], dtype=np.float64))
    joint_pos = np.asarray(motion["joint_pos"], dtype=np.float64)
    joint_names = motion["joint_names"] if isinstance(motion["joint_names"], list) else motion["joint_names"].tolist()

    model_joint_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        for joint_id in range(model.njnt)
        if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE
    ]
    joint_order = [joint_names.index(name) for name in model_joint_names]
    joint_pos = joint_pos[:, joint_order]

    qpos = np.zeros((len(base_pos), model.nq), dtype=np.float64)
    qpos[:, :3] = base_pos
    qpos[:, 3:7] = base_quat
    qpos[:, 7:] = joint_pos

    object_pos = np.asarray(motion["object_pos_w"], dtype=np.float64) if "object_pos_w" in motion else None
    object_quat = _continuous_quat(np.asarray(motion["object_quat_w"], dtype=np.float64)) if "object_quat_w" in motion else None
    return qpos, object_pos, object_quat, framerate


def _add_object(server: viser.ViserServer, motion_file: Path):
    object_name = _object_name_from_file(motion_file)
    if object_name is None or object_name not in MESH_FILE_PATHS:
        return None
    mesh_path = Path(os.path.expanduser(MESH_FILE_PATHS[object_name]))
    if not mesh_path.exists():
        print(f"[WARN] Object mesh missing: {mesh_path}")
        return None
    mesh = trimesh.load(mesh_path, force="mesh")
    scale = MESH_FILE_SCALES[object_name]
    return server.scene.add_mesh_trimesh(
        "/omomo_object",
        mesh,
        scale=scale,
        wxyz=(1.0, 0.0, 0.0, 0.0),
        position=(0.0, 0.0, 0.0),
    )


def run() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "motion_path",
        nargs="?",
        default="~/Datasets/OMOMO/retargeted_omniretarget_instinctmj_torso_v10_object_xy_align_foot_lock",
        help="Retargeted OMOMO .npz file or directory.",
    )
    parser.add_argument("--model", default=G1_MJCF_PATH)
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--loop", action="store_true", default=True)
    args = parser.parse_args()

    motion_file = _resolve_motion_file(Path(os.path.expanduser(args.motion_path)))
    model = mujoco.MjModel.from_xml_path(args.model)
    data = mujoco.MjData(model)
    qpos, object_pos, object_quat, framerate = _load_qpos(model, motion_file)

    print(f"[INFO] Direct playback motion: {motion_file}")
    print(f"[INFO] Frames: {len(qpos)}, framerate: {framerate}")

    server = viser.ViserServer(port=args.port, label="OMOMO direct playback")
    scene = ViserMujocoScene(server, model, num_envs=1)
    scene.create_scene_gui(show_debug_viz_control=False)
    object_handle = _add_object(server, motion_file)

    with server.gui.add_folder("Playback"):
        playing = server.gui.add_checkbox("Playing", initial_value=True)
        frame_slider = server.gui.add_slider("Frame", min=0, max=len(qpos) - 1, step=1, initial_value=0)
        speed = server.gui.add_slider("Speed", min=0.1, max=3.0, step=0.1, initial_value=1.0)

    frame_idx = 0
    last_time = time.time()
    try:
        while True:
            now = time.time()
            should_advance = playing.value and (now - last_time) >= (1.0 / (framerate * float(speed.value)))
            if should_advance:
                frame_idx = (frame_idx + 1) % len(qpos) if args.loop else min(frame_idx + 1, len(qpos) - 1)
                frame_slider.value = frame_idx
                last_time = now
            elif not playing.value:
                frame_idx = int(frame_slider.value)

            data.qpos[:] = qpos[frame_idx]
            mujoco.mj_forward(model, data)
            scene.update_from_mjdata(data)

            if object_handle is not None and object_pos is not None and object_quat is not None:
                object_handle.position = object_pos[frame_idx]
                object_handle.wxyz = object_quat[frame_idx]

            if scene.needs_update:
                scene.refresh_visualization()

            time.sleep(0.005)
    except KeyboardInterrupt:
        server.stop()


if __name__ == "__main__":
    run()
