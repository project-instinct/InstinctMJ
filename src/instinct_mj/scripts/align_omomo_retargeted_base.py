"""Align OMOMO retargeted motion base frames to InstinctMJ's G1 root body."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import mujoco
import numpy as np

from instinct_mj.assets.unitree_g1 import G1_MJCF_PATH


SUPPORTED_ENDINGS = ("retargeted.npz", "retargetted.npz")


def _quat_to_mat(quat: np.ndarray) -> np.ndarray:
  mat = np.empty(9, dtype=np.float64)
  mujoco.mju_quat2Mat(mat, quat.astype(np.float64, copy=False))
  return mat.reshape(3, 3)


def _mat_to_quat(mat: np.ndarray) -> np.ndarray:
  quat = np.empty(4, dtype=np.float64)
  mujoco.mju_mat2Quat(quat, mat.reshape(-1).astype(np.float64, copy=False))
  return quat


def _invert_pose(pos: np.ndarray, quat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  rot_t = _quat_to_mat(quat).T
  return -rot_t @ pos, _mat_to_quat(rot_t)


def _compose_pose(
  pos_a: np.ndarray,
  quat_a: np.ndarray,
  pos_b: np.ndarray,
  quat_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
  rot_a = _quat_to_mat(quat_a)
  rot_b = _quat_to_mat(quat_b)
  return pos_a + rot_a @ pos_b, _mat_to_quat(rot_a @ rot_b)


def _check_quat_norm(name: str, quat: np.ndarray, tolerance: float, path: Path) -> None:
  if tolerance <= 0.0:
    return
  norms = np.linalg.norm(quat.reshape(-1, 4), axis=-1)
  max_error = float(np.max(np.abs(norms - 1.0)))
  if max_error > tolerance:
    raise ValueError(
      f"{path}: {name} quaternion norm error {max_error:.6g} exceeds tolerance {tolerance:.6g}."
    )


def _motion_files(input_path: Path) -> list[Path]:
  if input_path.is_file():
    return [input_path]
  return sorted(path for path in input_path.rglob("*.npz") if path.name.endswith(SUPPORTED_ENDINGS))


def _output_path(input_file: Path, input_root: Path, output_root: Path) -> Path:
  if input_root.is_file():
    return output_root
  return output_root / input_file.relative_to(input_root)


class BaseFrameAligner:
  def __init__(self, model_path: str, src_frame: str, target_frame: str):
    self.model_path = os.path.expanduser(model_path)
    self.model = mujoco.MjModel.from_xml_path(self.model_path)
    self.data = mujoco.MjData(self.model)
    self.src_frame = src_frame
    self.target_frame = target_frame
    self.src_body_id = self._body_id(src_frame)
    self.target_body_id = self._body_id(target_frame)
    self.model_joint_names = [
      mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
      for joint_id in range(self.model.njnt)
    ]
    self.actuated_joint_names = [name for name in self.model_joint_names if name != "floating_base_joint"]

  def _body_id(self, name: str) -> int:
    body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body_id < 0:
      raise ValueError(f"Body '{name}' not found in {self.model_path}.")
    return body_id

  def _joint_pos_in_model_order(self, joint_names: list[str], joint_pos: np.ndarray) -> np.ndarray:
    missing = [name for name in self.actuated_joint_names if name not in joint_names]
    if missing:
      raise ValueError(f"Missing joints required by the G1 model: {missing}")
    return np.asarray(
      [joint_pos[:, joint_names.index(name)] for name in self.actuated_joint_names],
      dtype=np.float64,
    ).T

  def _relative_pose(self, joint_pos_frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    self.data.qpos[:] = 0.0
    self.data.qpos[3] = 1.0
    self.data.qpos[7:] = joint_pos_frame
    mujoco.mj_forward(self.model, self.data)
    src_pos = self.data.xpos[self.src_body_id].copy()
    src_quat = self.data.xquat[self.src_body_id].copy()
    target_pos = self.data.xpos[self.target_body_id].copy()
    target_quat = self.data.xquat[self.target_body_id].copy()
    return src_pos, src_quat, target_pos, target_quat

  def convert_base_pose(
    self,
    joint_names: list[str],
    joint_pos: np.ndarray,
    src_pos_w: np.ndarray,
    src_quat_w: np.ndarray,
  ) -> tuple[np.ndarray, np.ndarray]:
    if self.src_frame == self.target_frame:
      return src_pos_w.copy(), src_quat_w.copy()

    joint_pos_model = self._joint_pos_in_model_order(joint_names, joint_pos)
    target_pos_w = np.empty_like(src_pos_w, dtype=np.float64)
    target_quat_w = np.empty_like(src_quat_w, dtype=np.float64)
    for frame_idx in range(joint_pos_model.shape[0]):
      src_pos_b, src_quat_b, target_pos_b, target_quat_b = self._relative_pose(joint_pos_model[frame_idx])
      inv_src_pos_b, inv_src_quat_b = _invert_pose(src_pos_b, src_quat_b)
      root_pos_w, root_quat_w = _compose_pose(
        src_pos_w[frame_idx],
        src_quat_w[frame_idx],
        inv_src_pos_b,
        inv_src_quat_b,
      )
      target_pos_w[frame_idx], target_quat_w[frame_idx] = _compose_pose(
        root_pos_w,
        root_quat_w,
        target_pos_b,
        target_quat_b,
      )
    return target_pos_w.astype(src_pos_w.dtype), target_quat_w.astype(src_quat_w.dtype)


def _load_joint_names(payload: np.lib.npyio.NpzFile, joint_pos: np.ndarray, path: Path) -> list[str]:
  joint_names = payload["joint_names"].tolist()
  if joint_names and joint_names[0] == "floating_base_joint" and len(joint_names) == joint_pos.shape[1] + 1:
    joint_names = joint_names[1:]
  if len(joint_names) != joint_pos.shape[1]:
    raise ValueError(f"{path}: joint_names has {len(joint_names)} entries, joint_pos has {joint_pos.shape[1]} columns.")
  return [str(name) for name in joint_names]


def convert_file(
  input_file: Path,
  output_file: Path,
  aligner: BaseFrameAligner,
  quat_tolerance: float,
  overwrite: bool,
) -> None:
  if output_file.exists() and not overwrite:
    raise FileExistsError(f"{output_file} already exists. Pass --overwrite to replace it.")

  payload = np.load(input_file, allow_pickle=True)
  data = {key: payload[key] for key in payload.files}
  joint_pos = np.asarray(payload["joint_pos"])
  joint_names = _load_joint_names(payload, joint_pos, input_file)
  base_pos_w = np.asarray(payload["base_pos_w"])
  base_quat_w = np.asarray(payload["base_quat_w"])

  _check_quat_norm("base_quat_w", base_quat_w, quat_tolerance, input_file)
  if "object_quat_w" in payload:
    _check_quat_norm("object_quat_w", np.asarray(payload["object_quat_w"]), quat_tolerance, input_file)

  target_pos_w, target_quat_w = aligner.convert_base_pose(joint_names, joint_pos, base_pos_w, base_quat_w)
  data["joint_names"] = np.asarray(joint_names, dtype=np.str_)
  data["base_pos_w"] = target_pos_w
  data["base_quat_w"] = target_quat_w

  output_file.parent.mkdir(parents=True, exist_ok=True)
  np.savez(output_file, **data)


def main(args: argparse.Namespace | None = None) -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--input", required=True, help="Input retargeted .npz file or directory.")
  parser.add_argument("--output", required=True, help="Output .npz file or directory.")
  parser.add_argument("--model", default=G1_MJCF_PATH, help="InstinctMJ G1 MJCF path.")
  parser.add_argument("--src-frame", default="pelvis", help="Frame currently stored in base_pos_w/base_quat_w.")
  parser.add_argument("--target-frame", default="torso_link", help="Frame expected by InstinctMJ.")
  parser.add_argument("--quat-norm-tolerance", type=float, default=1e-2)
  parser.add_argument("--overwrite", action="store_true")
  parsed = parser.parse_args() if args is None else args

  input_path = Path(os.path.expanduser(parsed.input)).resolve()
  output_path = Path(os.path.expanduser(parsed.output)).resolve()
  files = _motion_files(input_path)
  if not files:
    raise ValueError(f"No supported retargeted npz files found under {input_path}.")
  if input_path.is_file() and output_path.is_dir():
    raise ValueError("--output must be a file path when --input is a file.")

  aligner = BaseFrameAligner(parsed.model, parsed.src_frame, parsed.target_frame)
  for input_file in files:
    convert_file(
      input_file=input_file,
      output_file=_output_path(input_file, input_path, output_path),
      aligner=aligner,
      quat_tolerance=parsed.quat_norm_tolerance,
      overwrite=parsed.overwrite,
    )
  print(
    f"[align_omomo_retargeted_base] converted {len(files)} file(s): "
    f"{parsed.src_frame} -> {parsed.target_frame}"
  )


def entry_point() -> None:
  main()


if __name__ == "__main__":
  entry_point()
