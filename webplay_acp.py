"""Realtime ACP reference browser and policy playback (no video recording).

  uv run --no-sync python webplay_acp.py refs --dataset /path/to/TerrainMotion
  uv run --no-sync python webplay_acp.py policy [--checkpoint /path/to/model_100.pt] [--dataset ~/Datasets/acp/v3_train6] --port 8192 [--host 127.0.0.1]
  uv run --no-sync python webplay_acp.py policy --checkpoint model_1500.pt --dataset /path/to/data --self-test
  uv run --no-sync python webplay_acp.py refs --dataset /path/to/TerrainMotion --self-test

Reference mode uses CPU MuJoCo kinematics and the same mjviser robot mesh renderer
as mjlab's ViserPlayViewer. Only the current clip is loaded; the terrain LRU is
bounded. Playback changes link transforms and overlay data, never robot meshes.

Policy mode draws the contact layers of the ACP Stage 2 policy on the live env: the
observed reference contact targets, the planner's predicted next contacts and the
measured physical contacts (``acp_contact_points``). A dropdown feeds the planner's
targets into the tracker's observation instead of the reference ones (eta_obs = 1).
"""

from __future__ import annotations

import argparse
import fcntl
import ipaddress
import json
import os
import queue
import signal
import socket
import subprocess
import tempfile
import time
import traceback
import zipfile
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import trimesh
import viser
import yaml
from mjviser import ViserMujocoScene

DATASET_ROOT = Path("/home/duanxin/Datasets/acp")
ONLY_DATASET = False  # --only: list just the given dataset in the dropdown
EXTRA_DATASETS: list[Path] = []  # --also: further datasets listed in the dropdown (with --only)
ROBOT_XML = Path(__file__).parent / "src/instinct_mj/assets/resources/acp_g1/g1_29dof.xml"
AUTHORS_XML = DATASET_ROOT / "playground/assets/g1/g1_29dof.xml"
CONTACT_LINKS = {
  "left_knee_link": "left_knee", "right_knee_link": "right_knee", "pelvis": "pelvis",
  "left_ankle_roll_link": "left_foot", "right_ankle_roll_link": "right_foot",
  "left_elbow_link": "left_elbow", "right_elbow_link": "right_elbow",
  "left_wrist_yaw_link": "left_hand", "right_wrist_yaw_link": "right_hand",
}
COLORS = {"foot": (55, 130, 255), "knee": (60, 220, 110), "hand": (255, 65, 65),
          "elbow": (255, 155, 35), "pelvis": (190, 85, 255)}
# Body order of the planner / contact command / `acp_contact_points` (CONTACT_BODIES, planner.py): the
# CONTACT_LINKS mapping is declared in that order, so its values keep the per-body colours aligned.
CONTACT_ORDER = tuple(CONTACT_LINKS.values())
CONTACT_COLORS = np.array([COLORS[name.split("_")[-1]] for name in CONTACT_ORDER], dtype=np.uint8)
CONTACT_RADIUS = 0.045  # measured contact: solid sphere ≈4.5 cm (as the reference browser)
TARGET_RADIUS = 0.028   # next-contact target: smaller translucent sphere of the same colour
COLLISION_OPACITY = 0.45  # collision geoms of groups 3 (meshes, hand half-spheres) and 4 (foot capsules) are drawn see-through
GRID_XY = np.stack(np.meshgrid(np.linspace(-0.8, 0.8, 17), np.linspace(-0.5, 0.5, 11)), -1).reshape(-1, 2)
REVIEW_MARKS = {"keep": "✓", "reject": "✗", "edited": "↗", "unreviewed": "○"}
BODY_LABELS = {name: "Pelvis" if name == "pelvis" else name.replace("left_", "L ").replace("right_", "R ")
               for name in CONTACT_LINKS.values()}
# Initial camera: three-quarter side view of the whole clip, per-client fov/aspect.
FRAME_SIDE_DEG = 60.0
FRAME_ELEVATION_DEG = 20.0
FRAME_MARGIN = 0.6      # keep the framed extent inside the central 60% of the image
FRAME_MIN_DISTANCE = 3.5
FRAME_LOOK_HEIGHT = 0.6  # mid-robot height above the local ground
FRAME_ROBOT_HEIGHT = 1.8
FRAME_REACH = 1.0        # +1 m beyond the pelvis trajectory extent


class CurationStore:
  """Read/modify/write under a stable lock file; readers see a complete YAML."""

  def __init__(self, dataset: Path):
    self.path = dataset / "curation.yaml"

  def read(self) -> dict[str, dict]:
    if not self.path.exists():
      return {}
    with self.path.open(encoding="utf-8") as stream:
      records = yaml.safe_load(stream) or {}
    if not isinstance(records, dict):
      raise ValueError(f"Expected clip → curation mapping in {self.path}")
    for name, record in records.items():
      if not isinstance(record, dict) or record.get("status") not in ("keep", "reject", "edited"):
        raise ValueError(f"Invalid curation entry {name!r} in {self.path}")
    return records

  def update(self, clip: str, status: str | None, note: str, *, note_only: bool = False) -> dict[str, dict]:
    # Serialize browser-tab writes AND independent viewer processes. Always merge
    # the latest file, rather than overwriting it with a viewer's cached mapping.
    with self.path.with_name(".curation.lock").open("a", encoding="utf-8") as lock:
      fcntl.flock(lock, fcntl.LOCK_EX)
      records = self.read()
      if note_only:
        if clip not in records:
          return records  # Unreviewed note drafts are saved with Keep / Reject.
        status = records[clip]["status"]
      if status is None:
        records.pop(clip, None)
      elif status in ("keep", "reject", "edited"):
        records[clip] = {**records.get(clip, {}), "status": status, "note": note,
                         "time": datetime.now(timezone.utc).isoformat(timespec="seconds")}
      else:
        raise ValueError(f"Unknown curation status: {status}")
      temporary: Path | None = None
      try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.path.parent,
                                         prefix=".curation-", suffix=".tmp", delete=False) as stream:
          temporary = Path(stream.name)
          yaml.safe_dump(records, stream, allow_unicode=True, sort_keys=False)
          stream.flush()
          os.fsync(stream.fileno())
        os.replace(temporary, self.path)
      finally:
        if temporary is not None:
          temporary.unlink(missing_ok=True)
      return records


def announce(port: int) -> None:
  addresses = subprocess.check_output(["hostname", "-I"], text=True).split()
  print("[acp-viewer] Listening on 0.0.0.0", flush=True)
  for address in addresses:
    if ipaddress.ip_address(address).version == 4:
      print(f"  http://{address}:{port}", flush=True)
  print(f"  http://localhost:{port}", flush=True)
  print(f"  SSH: ssh -L {port}:localhost:{port} <host>  (host: {socket.gethostname()})", flush=True)


def robot_model() -> mujoco.MjModel:
  # Match get_acp_g1_spec without importing task registration or torch in refs mode.
  spec = mujoco.MjSpec.from_file(str(ROBOT_XML if ROBOT_XML.exists() else AUTHORS_XML))
  spec.delete(spec.body("target_sphere"))
  for sensor in list(spec.sensors):
    spec.delete(sensor)
  for actuator in list(spec.actuators):
    spec.delete(actuator)
  for geom in spec.geoms:
    if geom.group == 0:
      geom.group = 3
  pelvis = spec.body("pelvis")
  pelvis.pos, pelvis.quat = [0, 0, 0.78], [1, 0, 0, 0]
  model = spec.compile()
  if model.nq != 30 or model.njnt != 24:
    raise ValueError(f"Expected pelvis freejoint + 23 joints; got nq={model.nq}, njnt={model.njnt}")
  return model


def yaw_rotation(wxyz: np.ndarray) -> np.ndarray:
  w, x, y, z = wxyz
  yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
  c, s = np.cos(yaw), np.sin(yaw)
  return np.array([[c, -s], [s, c]])


def heading_xy(wxyz: np.ndarray) -> np.ndarray:
  """Unit XY heading implied by a pelvis quaternion (local +x in world XY)."""
  return yaw_rotation(wxyz).T @ np.array([1.0, 0.0])


def frame_clip_camera(base_pos: np.ndarray, heading: np.ndarray, ground: float, fov: float,
                      aspect: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """3/4 side camera (60 deg off travel, 20 deg elevation) framing a whole clip.

  The look-at point is the XY centre of the pelvis trajectory at mid-robot height.
  Travel direction is the first PCA axis of the XY trajectory (oriented start to end),
  or the pelvis heading at frame 0 when the clip barely moves. The distance fits the
  trajectory plus 1 m and the robot height inside the central 60% of the image for the
  given vertical fov and aspect; clips far from the origin use their actual coordinates.
  """
  xy = np.asarray(base_pos[:, :2], dtype=np.float64)
  centre = 0.5 * (xy.min(axis=0) + xy.max(axis=0))
  offset = xy - centre
  radius = float(np.linalg.norm(offset, axis=1).max())
  travel = xy[-1] - xy[0]
  if radius >= 0.5 and np.linalg.norm(travel) >= 0.5:
    _, vectors = np.linalg.eigh(offset.T @ offset)
    forward = vectors[:, -1]
    if float(forward @ travel) < 0.0:
      forward = -forward
  else:
    forward = np.asarray(heading, dtype=np.float64)
  forward = forward / np.linalg.norm(forward)
  side_angle = np.deg2rad(FRAME_SIDE_DEG)
  cos_a, sin_a = np.cos(side_angle), np.sin(side_angle)
  side = np.array([cos_a * forward[0] - sin_a * forward[1], sin_a * forward[0] + cos_a * forward[1]])
  elevation = np.deg2rad(FRAME_ELEVATION_DEG)
  eye_dir = np.array([np.cos(elevation) * side[0], np.cos(elevation) * side[1], np.sin(elevation)])
  look_at = np.array([centre[0], centre[1], ground + FRAME_LOOK_HEIGHT])
  # Feet and head height over the trajectory disc grown by FRAME_REACH.
  extent = radius + FRAME_REACH
  angles = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
  samples = np.vstack((xy, centre + extent * np.column_stack((np.cos(angles), np.sin(angles)))))
  points = np.concatenate((
    np.column_stack((samples, np.full(len(samples), ground))),
    np.column_stack((samples, np.full(len(samples), ground + FRAME_ROBOT_HEIGHT)))))
  # Viser/three.js camera basis: local -z looks at the target, up = world +z.
  right = np.cross(np.array([0.0, 0.0, 1.0]), eye_dir)
  right /= np.linalg.norm(right)
  up = np.cross(eye_dir, right)
  q = points - look_at
  tan_v = np.tan(0.5 * fov)
  tan_h = tan_v * aspect
  distance = 0.0
  for lateral, tangent in ((q @ right, tan_h), (q @ up, tan_v)):
    distance = max(distance, float(np.max(q @ eye_dir + np.abs(lateral) / (FRAME_MARGIN * tangent))))
  distance = max(FRAME_MIN_DISTANCE, distance)
  return look_at + distance * eye_dir, look_at, np.array([0.0, 0.0, 1.0])


def next_onsets(flag: np.ndarray) -> np.ndarray:
  """First onset STRICTLY after each frame, as AcpReferenceContacts._build_tables.

  The initial contact segment is not an onset. No wrap at the clip boundary.
  A sentinel T represents no future onset; the 75-frame horizon is applied later.
  """
  length, bodies = flag.shape
  onset = flag & ~np.concatenate((np.ones_like(flag[:1]), flag[:-1]))
  indices = np.where(onset, np.arange(length)[:, None], length)
  after = np.concatenate((indices[1:], np.full((1, bodies), length)), axis=0)
  return np.minimum.accumulate(after[::-1], axis=0)[::-1]


@dataclass
class ContactTrack:
  names: list[str]
  flag: np.ndarray
  pos: np.ndarray
  times: np.ndarray
  next_frame: np.ndarray
  force: np.ndarray | None = None
  source: str = "kinematic labeler"
  provenance: str = ""

  @classmethod
  def load(cls, path: Path, physics: bool = False) -> ContactTrack:
    with np.load(path, allow_pickle=False) as arrays:
      names_key = "link_names" if physics else "body_names"
      names = [CONTACT_LINKS.get(str(n), str(n)) for n in arrays[names_key]]
      flag = arrays["contact_flag"].astype(bool)
      pos = arrays["contact_pos_w"].copy()
      if flag.ndim != 2 or pos.shape != (*flag.shape, 3) or flag.shape[1] != len(names):
        raise ValueError(f"Invalid contact array shapes in {path}")
      fps = float(arrays["framerate"])
      times = arrays["sample_time_s"].copy() if physics and "sample_time_s" in arrays else np.arange(len(flag)) / fps
      force = arrays["force_w"].copy() if physics and "force_w" in arrays else None
      params = json.loads(str(arrays["params"].item())) if "params" in arrays else {}
      hint = str(params.get("source", ""))
      if "motionfix" in hint.casefold():
        source = "motionfix"
      elif physics or "physics" in hint.casefold() or "force_w" in arrays or "normal_force" in arrays:
        source = "physics"
      else:
        source = "kinematic labeler"
      return cls(names, flag, pos, times, next_onsets(flag), force, source,
                 json.dumps(params, ensure_ascii=False) if params else ", ".join(arrays.files))

  def frame_at(self, seconds: float) -> int:
    return int(np.clip(np.searchsorted(self.times, seconds + 1e-8, side="right") - 1, 0, len(self.flag) - 1))

  @property
  def colors(self) -> np.ndarray:
    return np.array([COLORS[name.split("_")[-1]] for name in self.names], dtype=np.uint8)

  def targets(self, frame: int, base: np.ndarray, rotation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    future = self.next_frame[frame]
    valid = (future < len(self.flag)) & (self.times[np.minimum(future, len(self.flag) - 1)] - self.times[frame] <= 1.5 + 1e-8)
    pos = self.pos[np.minimum(future, len(self.flag) - 1), np.arange(len(self.names))]
    local_xy = (pos[:, :2] - base[:2]) @ rotation
    valid &= (np.abs(local_xy[:, 0]) <= 0.8) & (np.abs(local_xy[:, 1]) <= 0.5)
    valid &= np.isfinite(pos).all(axis=1)
    return pos, valid


class Terrain:
  """STL render geometry and exact downward triangle rays, including concavities.

  A 1 m XY bin index avoids tracing all triangles for every scan point. Vertical
  triangles have no area in XY; all other triangles are tested barycentrically.
  This uses the original STL surface, rather than MuJoCo's convex mesh hull.
  """

  def __init__(self, path: Path):
    mesh = trimesh.load_mesh(path, process=False)
    self.vertices = np.asarray(mesh.vertices, dtype=np.float32)
    self.faces = np.asarray(mesh.faces, dtype=np.uint32)
    triangles = self.vertices[self.faces].astype(np.float64)
    a = triangles[:, 0]
    e1, e2 = triangles[:, 1] - a, triangles[:, 2] - a
    det = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    keep = np.abs(det) > 1e-12
    self.a, self.e1, self.e2, self.det = a[keep], e1[keep], e2[keep], det[keep]
    triangles = triangles[keep]
    lower = np.floor(triangles[:, :, :2].min(axis=1)).astype(int)
    upper = np.floor(triangles[:, :, :2].max(axis=1)).astype(int)
    bins: dict[tuple[int, int], list[int]] = {}
    for i, (lo, hi) in enumerate(zip(lower, upper)):
      for x in range(lo[0], hi[0] + 1):
        for y in range(lo[1], hi[1] + 1):
          bins.setdefault((x, y), []).append(i)
    self.bins = {key: np.asarray(value, dtype=np.int32) for key, value in bins.items()}
    self.empty = np.empty(0, dtype=np.int32)
    self.nbytes = sum(a.nbytes for a in (self.vertices, self.faces, self.a, self.e1, self.e2, self.det))
    self.nbytes += sum(a.nbytes + 128 for a in self.bins.values())

  def heights(self, xy: np.ndarray, origin_z: float) -> np.ndarray:
    cells = np.floor(xy).astype(int)
    candidates = [self.bins.get((int(x), int(y)), self.empty) for x, y in cells]
    counts = np.array([len(indices) for indices in candidates])
    heights = np.full(len(xy), -np.inf)
    if counts.sum() == 0:
      return np.full(len(xy), np.nan)
    ray = np.repeat(np.arange(len(xy)), counts)
    tri = np.concatenate(candidates)
    a, e1, e2, det = self.a[tri], self.e1[tri], self.e2[tri], self.det[tri]
    delta = xy[ray] - a[:, :2]
    u = (delta[:, 0] * e2[:, 1] - delta[:, 1] * e2[:, 0]) / det
    v = (delta[:, 1] * e1[:, 0] - delta[:, 0] * e1[:, 1]) / det
    z = a[:, 2] + u * e1[:, 2] + v * e2[:, 2]
    hit = (u >= -1e-8) & (v >= -1e-8) & (u + v <= 1 + 1e-8) & (z <= origin_z)
    np.maximum.at(heights, ray[hit], z[hit])
    heights[~np.isfinite(heights)] = np.nan
    return heights


class TerrainCache:
  def __init__(self):
    self.items: OrderedDict[Path, Terrain] = OrderedDict()

  def get(self, path: Path) -> Terrain:
    path = path.resolve()
    if path not in self.items:
      terrain = Terrain(path)
      # Eight STLs / 128 MiB of terrain data; no cache of motion arrays or GPU models.
      while self.items and (len(self.items) >= 8 or sum(t.nbytes for t in self.items.values()) + terrain.nbytes > 128 * 2**20):
        self.items.popitem(last=False)
      self.items[path] = terrain
    self.items.move_to_end(path)
    return self.items[path]


class ReferenceViewer:
  def __init__(self, dataset: Path, port: int = 8188):
    self.model = robot_model()
    self.data = mujoco.MjData(self.model)
    self.free_adr = int(self.model.jnt_qposadr[np.flatnonzero(self.model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)[0]])
    self.joint_ids = np.flatnonzero(self.model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE)
    self.joint_names = [self.model.joint(int(j)).name for j in self.joint_ids]
    self.joint_adr = self.model.jnt_qposadr[self.joint_ids]
    self.cache = TerrainCache()
    self.events: queue.SimpleQueue[tuple[str, Any]] = queue.SimpleQueue()
    self.running = True
    self.frame = 0.0
    self.last_tick = time.perf_counter()
    self.last_displayed = -1
    self.last_status = 0.0
    self.camera_base: np.ndarray | None = None
    # Per-clip framing inputs for the initial camera (pelvis heading and local ground).
    self.frame_heading: np.ndarray | None = None
    self.frame_ground = 0.0
    self.curation: dict[str, dict] = {}
    self.note_drafts: dict[tuple[Path, str], str] = {}
    self.skipped_clips: list[str] = []
    self.curation_context: tuple[Path, str] | None = None
    self.last_curation_check = 0.0
    self.terrain_path: Path | None = None
    self.terrain_handle = None
    self.server = viser.ViserServer(host="0.0.0.0", port=port, label="ACP · Reference browser")
    self.server.scene.set_up_direction("+z")
    self.robot = ViserMujocoScene(self.server, self.model, num_envs=1)
    self.robot.camera_tracking_enabled = False
    self.server.gui.configure_theme(control_layout="collapsible", control_width="medium")
    self._build_gui(dataset.expanduser().resolve())
    self.reference = None
    self.physics = None
    self.load_dataset(dataset)
    self.server.on_client_connect(self._client_connected)

  def _build_gui(self, dataset: Path) -> None:
    datasets = [] if ONLY_DATASET else sorted(p.resolve() for p in DATASET_ROOT.iterdir() if p.is_dir() and (p / "metadata.yaml").is_file())
    for extra in [dataset, *EXTRA_DATASETS]:
      if extra not in datasets:
        datasets.append(extra)
    self.datasets = {p.name: p for p in datasets}
    listed = " · ".join(f"`{name}`" for name in self.datasets)
    self.server.gui.add_markdown(
      "**ACP · G1 (23 joints, welded wrists)**\n\n"
      "Live reference playback · CPU kinematics · 50 Hz\n\n"
      f"**Datasets ({len(self.datasets)}):** {listed}\n\n"
      "切换 **Dataset** 下拉即可浏览各数据集，Clip 列表会随之刷新（默认打开 `v3_clean`）。")
    with self.server.gui.add_folder("Dataset & clip"):
      self.dataset_gui = self.server.gui.add_dropdown(f"Dataset · {len(self.datasets)} available", tuple(self.datasets), initial_value=dataset.name)
      self.group_gui = self.server.gui.add_dropdown("Group", ("All",))
      self.filter_gui = self.server.gui.add_text("Filter name / group", "")
      self.review_filter_gui = self.server.gui.add_dropdown("Review filter", ("all", "unreviewed", "kept", "rejected"))
      self.clip_gui = self.server.gui.add_dropdown("Clip", ("Loading…",))
      self.matches_gui = self.server.gui.add_markdown("")
      self.load_error_gui = self.server.gui.add_markdown("")
    with self.server.gui.add_folder("Curation"):
      self.review_gui = self.server.gui.add_markdown("")
      self.note_gui = self.server.gui.add_text("Clip note", "", hint="Saved with Keep / Reject; reviewed notes autosave.")
      self.keep_gui = self.server.gui.add_button("✓ Keep", color="green")
      self.reject_gui = self.server.gui.add_button("✗ Reject", color="red")
      self.clear_gui = self.server.gui.add_button("Clear decision & note", color="gray")
      self.unreviewed_gui = self.server.gui.add_button("Next unreviewed", color="blue")
      self.review_feedback_gui = self.server.gui.add_markdown("Notes on unreviewed clips are saved with Keep / Reject.")
    with self.server.gui.add_folder("Camera"):
      self.follow_gui = self.server.gui.add_checkbox("Camera follow", False)
      self.reset_view_gui = self.server.gui.add_button("Reset view")
      self.snap_gui = self.server.gui.add_button("Snap to robot")
      self.server.gui.add_markdown("Drag to orbit / pan; scroll to zoom. Follow preserves your view.")
    with self.server.gui.add_folder("Playback"):
      self.play_gui = self.server.gui.add_checkbox("Play", True)
      self.speed_gui = self.server.gui.add_slider("Speed ×", 0.25, 2.0, 0.25, 1.0)
      self.timeline_gui = self.server.gui.add_slider("Frame (scrub)", 0, 1, 1, 0)
      self.loop_gui = self.server.gui.add_checkbox("Loop", True)
      self.previous_gui = self.server.gui.add_button("−1 frame")
      self.next_gui = self.server.gui.add_button("+1 frame")
      self.status_gui = self.server.gui.add_markdown("")
      self.active_contacts_gui = self.server.gui.add_markdown("**Contacts:** none")
    with self.server.gui.add_folder("Overlays"):
      self.contacts_gui = self.server.gui.add_checkbox("Reference contacts", True)
      self.physics_gui = self.server.gui.add_checkbox("Physics contacts", True)
      self.targets_gui = self.server.gui.add_checkbox("Next-contact targets (≤1.5 s)", True)
      self.labels_gui = self.server.gui.add_checkbox("Labels", True)
      self.contact_timeline_gui = self.server.gui.add_checkbox("Contact timeline", False)
      self.contact_source_gui = self.server.gui.add_markdown("")
      self.contact_timeline_image = self.server.gui.add_image(np.full((1, 1, 3), 255, np.uint8),
        label="Contact timeline", format="png", visible=False)
      self.scan_gui = self.server.gui.add_checkbox("Height scan · 17×11", True)
      self.trail_gui = self.server.gui.add_checkbox("Pelvis trail", True)
      self.server.gui.add_markdown("**Contact colour legend**\n\n"
        "🟦 Feet · 🟩 Knees · 🟥 Hands · 🟧 Elbows · 🟪 Pelvis\n\n"
        "**Current contact:** solid sphere, radius ≈4 cm.\n\n"
        "**Next target:** smaller translucent sphere of the same colour, with a thin line from the body. "
        "The onset must be strictly in the future, within 1.5 s and the pelvis-yaw footprint (±0.8 × ±0.5 m). "
        "Targets use reference labels when available, otherwise physics labels.")
      self.details_gui = self.server.gui.add_markdown("")
    # Programmatic GUI synchronization has client_id=None and must not enqueue scrubs.
    for handle, name in ((self.dataset_gui, "dataset"), (self.group_gui, "filter"),
                         (self.filter_gui, "filter"), (self.review_filter_gui, "filter"),
                         (self.clip_gui, "clip"), (self.timeline_gui, "scrub")):
      async def changed(event, name=name):
        if event.client_id is not None:
          value = self.clip_labels.get(event.target.value) if name == "clip" else event.target.value
          self.events.put((name, value))
      handle.on_update(changed)
    self.previous_gui.on_click(lambda _: self.events.put(("step", -1)))
    self.next_gui.on_click(lambda _: self.events.put(("step", 1)))
    self.reset_view_gui.on_click(lambda event: self.events.put(("reset_view", event.client_id)))
    self.snap_gui.on_click(lambda event: self.events.put(("snap", event.client_id)))
    for button, status in ((self.keep_gui, "keep"), (self.reject_gui, "reject"), (self.clear_gui, None)):
      async def review(event, status=status):
        # Bind the action to the displayed clip before it enters the main queue.
        self.events.put(("review", (self.curation_context, status, self.note_gui.value)))
      button.on_click(review)
    async def note_changed(event):
      if event.client_id is not None:
        self.events.put(("note", (self.curation_context, event.target.value)))
    self.note_gui.on_update(note_changed)
    self.unreviewed_gui.on_click(lambda _: self.events.put(("next_unreviewed", None)))
    self.marker_handles: dict[str, Any] = {}
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=1)
    for name, opacity, wireframe in (("reference", 1.0, False), ("physics", 1.0, False), ("targets", 0.38, False)):
      self.marker_handles[name] = self.server.scene.add_batched_meshes_simple(
        f"/contacts/{name}", sphere.vertices, sphere.faces,
        batched_positions=np.zeros((9, 3)), batched_wxyzs=np.tile([1, 0, 0, 0], (9, 1)),
        batched_scales=np.zeros(9), opacity=opacity, wireframe=wireframe,
        cast_shadow=False, receive_shadow=False)
    self.contact_label_handles = {name: self.server.scene.add_label(
      f"/contacts/labels/{name}", label, visible=False, anchor="bottom-center",
      font_screen_scale=0.85, depth_test=False) for name, label in BODY_LABELS.items()}
    self.contact_body_ids = {name: self.model.body(link).id for link, name in CONTACT_LINKS.items()}
    self.target_lines = self.server.scene.add_line_segments("/contacts/target_lines",
      np.zeros((0, 2, 3), np.float32), (180, 180, 180), line_width=1.25, visible=False)
    self.scan_handle = self.server.scene.add_point_cloud("/height_scan", np.zeros((0, 3)), (255, 225, 70), point_size=0.025, precision="float32")
    self.trail_handle = self.server.scene.add_line_segments("/pelvis_trail", np.zeros((0, 2, 3)), (205, 110, 255), line_width=2)

  @staticmethod
  def group(entry: dict) -> str:
    return str(entry.get("group", Path(entry["motion_file"]).parent.as_posix()))

  def load_dataset(self, path: Path) -> None:
    dataset = path.expanduser().resolve()
    with (dataset / "metadata.yaml").open() as stream:
      metadata = yaml.safe_load(stream)
    entries = {entry["motion_file"]: entry for entry in metadata["motion_files"]}
    terrains = {int(t["terrain_id"]): dataset / t["terrain_file"] for t in metadata["terrains"]}
    if not entries:
      raise ValueError(f"No motions in {dataset}")
    curation_store = CurationStore(dataset)
    curation = curation_store.read()
    # Commit only after the metadata parsed cleanly: a dataset being rewritten by
    # another process must not leave the viewer half-switched.
    self.dataset, self.metadata = dataset, metadata
    self.entries, self.terrains = entries, terrains
    self.curation_store, self.curation = curation_store, curation
    self.dataset_gui.value = self.dataset.name
    self.group_gui.options = ("All", *sorted({self.group(entry) for entry in self.entries.values()}))
    self.group_gui.value = "All"
    self.filter_gui.value = ""
    self.review_filter_gui.value = "all"
    self.clip_name = ""
    self.apply_filter()

  def clip_label(self, name: str) -> str:
    status = self.curation.get(name, {}).get("status", "unreviewed")
    return f"{REVIEW_MARKS[status]} {name}"

  def apply_filter(self, *, select: bool = True) -> None:
    query = self.filter_gui.value.casefold().split()
    group = self.group_gui.value
    review_filter = self.review_filter_gui.value
    wanted = {"unreviewed": "unreviewed", "kept": "keep", "rejected": "reject"}
    options = [name for name, entry in self.entries.items()
               if (group == "All" or self.group(entry) == group)
               and all(word in f"{name} {self.group(entry)}".casefold() for word in query)
               and (review_filter == "all" or self.curation.get(name, {}).get("status", "unreviewed") == wanted[review_filter])]
    self.clip_labels = {self.clip_label(name): name for name in options}
    self.clip_gui.options = tuple(self.clip_labels) if options else ("No matching clips",)
    self.clip_gui.disabled = not options
    self.matches_gui.content = f"{len(options)} / {len(self.entries)} clips"
    if self.clip_name in options:
      self.clip_gui.value = self.clip_label(self.clip_name)
    elif options and select:
      # A clip that is still being written is skipped rather than aborting the switch.
      for name in options:
        if self._safe_load_clip(name):
          break

  def _show_review(self) -> None:
    record = self.curation.get(self.clip_name, {})
    status = record.get("status", "unreviewed")
    self.review_gui.content = (f"**{self.clip_label(self.clip_name)}**\n\n"
      f"Status: **{status}**" + (f" · {record['time']}" if "time" in record else ""))
    self.note_gui.value = record.get("note", "") if record else self.note_drafts.get((self.dataset, self.clip_name), "")
    self.last_status = 0.0

  def save_review(self, context: tuple[Path, str], status: str | None, note: str, *, note_only: bool = False) -> None:
    dataset, clip = context
    store = CurationStore(dataset)
    records = store.update(clip, status, note, note_only=note_only)
    if note_only and clip not in records:
      self.note_drafts[context] = note
    else:
      self.note_drafts.pop(context, None)
    if dataset == self.dataset:
      self.curation = records
      self._show_review()
      self.apply_filter()
      self.review_feedback_gui.content = (f"Saved {self.clip_label(clip)} to `curation.yaml`."
        if clip in records or not note_only else "Draft note; choose Keep / Reject to save it.")

  def next_unreviewed(self) -> None:
    self.curation = self.curation_store.read()
    names = list(self.entries)
    index = names.index(self.clip_name)
    group = self.group_gui.value
    query = self.filter_gui.value.casefold().split()
    for name in names[index + 1:] + names[:index + 1]:
      entry = self.entries[name]
      if (name not in self.curation and (group == "All" or self.group(entry) == group)
          and all(word in f"{name} {self.group(entry)}".casefold() for word in query)):
        self.review_filter_gui.value = "unreviewed"
        self.apply_filter(select=False)
        self.load_clip(name)
        self.review_feedback_gui.content = "Next unreviewed clip."
        return
    self.review_feedback_gui.content = "No unreviewed clips in the current group / text filter."

  def load_clip(self, name: str) -> None:
    path = self.dataset / name
    with np.load(path, allow_pickle=False) as arrays:
      names = [str(n) for n in arrays["joint_names"]]
      columns = [names.index(n) for n in self.joint_names]
      self.joint_pos = arrays["joint_pos"][:, columns].copy()
      self.base_pos = arrays["base_pos_w"].copy()
      self.base_quat = arrays["base_quat_w"].copy()
      self.fps = float(arrays["framerate"])
    self.length = len(self.base_pos)
    if self.joint_pos.shape != (self.length, 23) or self.base_quat.shape != (self.length, 4) or self.length == 0:
      raise ValueError(f"Invalid motion shapes in {path}")
    if self.fps <= 0 or not np.isfinite(self.base_pos).all() or not np.isfinite(self.base_quat).all() or not np.isfinite(self.joint_pos).all():
      raise ValueError(f"Invalid motion values in {path}")
    self.base_quat /= np.linalg.norm(self.base_quat, axis=1, keepdims=True)
    reference_path = path.with_suffix(".contacts.npz")
    physics_path = path.with_suffix(".physics.npz")
    self.reference = ContactTrack.load(reference_path) if reference_path.exists() else None
    self.physics = ContactTrack.load(physics_path, physics=True) if physics_path.exists() else None
    self.physics_keys: list[str] = []
    if self.physics is not None:
      with np.load(physics_path, allow_pickle=False) as arrays:
        self.physics_keys = arrays.files
      print(f"[acp-viewer] physics {name}: {', '.join(self.physics_keys)}", flush=True)
    self.contacts_gui.disabled = self.reference is None
    self.physics_gui.disabled = self.physics is None
    source = self.reference if self.reference is not None else self.physics
    self.contact_source_gui.content = (f"**Contact source:** {source.source}\n\n`{source.provenance}`"
      if source is not None else "**Contact source:** no sidecar")
    if self.reference is not None and self.physics is not None:
      self.contact_source_gui.content += f"\n\nAdditional physics overlay: {self.physics.source}. Labels/targets use the primary source above."
    self.contact_timeline_gui.disabled = source is None
    self.contact_timeline_strip = None
    self.last_timeline_upload = 0.0
    self.last_timeline_seconds = None
    terrain_path = self.terrains[int(self.entries[name]["terrain_id"])].resolve()
    self.terrain = self.cache.get(terrain_path)
    if terrain_path != self.terrain_path:
      if self.terrain_handle is not None:
        self.terrain_handle.remove()
      self.terrain_handle = self.server.scene.add_mesh_simple(
        "/terrain", self.terrain.vertices, self.terrain.faces,
        color=(155, 170, 180), side="double", flat_shading=True)
      self.terrain_path = terrain_path
    self.clip_name = name
    self.curation_context = (self.dataset, name)
    if self.clip_label(name) in self.clip_gui.options:
      self.clip_gui.value = self.clip_label(name)
    self._show_review()
    self.frame, self.last_displayed, self.last_status = 0.0, -1, 0.0
    self.timeline_gui.max = max(1, self.length - 1)
    self.timeline_gui.value = 0
    self.trail_handle.points = np.stack((self.base_pos[:-1], self.base_pos[1:]), axis=1).astype(np.float32)
    target_source = "reference" if self.reference is not None else "physics" if self.physics is not None else "none"
    self.details_gui.content = (f"**Target source:** {target_source}\n\n"
      f"**Terrain:** `{terrain_path.name}`\n\n"
      f"**Physics keys:** {', '.join(self.physics_keys) if self.physics_keys else 'No physics sidecar'}")
    self.camera_base = None
    centre_xy = 0.5 * (self.base_pos[:, :2].min(axis=0) + self.base_pos[:, :2].max(axis=0))
    ground = float(self.terrain.heights(centre_xy[None], float(self.base_pos[:, 2].max()) + 5.0)[0])
    if not np.isfinite(ground):
      ground = float(self.base_pos[:, 2].min()) - 0.78
    self.frame_heading = heading_xy(self.base_quat[0])
    self.frame_ground = ground
    self.render(force=True)
    self._request_frame()
    self.last_tick = time.perf_counter()
    print(f"[acp-viewer] {self.dataset.name}/{name} · {self.length} frames @ {self.fps:g} Hz", flush=True)

  def _client_connected(self, client: viser.ClientHandle) -> None:
    # Viser registers a client only after its first camera message, so its camera
    # state is readable; framing still runs on the main tick that owns all state.
    self.events.put(("reset_view", client.client_id))

  def _request_frame(self, client_id: int | None = None) -> None:
    for cid, client in self.server.get_clients().items():
      if client_id is not None and cid != client_id:
        continue
      height = client.camera.image_height
      aspect = client.camera.image_width / height if height > 0 else 16.0 / 9.0
      position, look_at, up = frame_clip_camera(
        self.base_pos, self.frame_heading, self.frame_ground, float(client.camera.fov), aspect)
      with client.atomic():
        client.camera.position = position
        client.camera.look_at = look_at
        client.camera.up_direction = up

  def _snap_camera(self, client_id: int | None) -> None:
    base = self.data.body("pelvis").xpos.copy()
    for cid, client in self.server.get_clients().items():
      if client_id is not None and cid != client_id:
        continue
      with client.atomic():
        # A one-shot translation retains the user's zoom and orbit angle.
        client.camera.position = np.asarray(client.camera.position) + base - np.asarray(client.camera.look_at)

  def _follow_camera(self, base: np.ndarray) -> None:
    if self.follow_gui.value and self.camera_base is not None:
      delta = base - self.camera_base
      if np.any(delta):
        for client in self.server.get_clients().values():
          with client.atomic():
            # Viser's position setter translates look_at by the same offset.
            client.camera.position = np.asarray(client.camera.position) + delta
    # Baseline bookkeeping does not read or write a client camera. With follow
    # off, playback, scrub and clip changes never override the user's view.
    self.camera_base = base.copy()

  def _safe_load_clip(self, name: str) -> bool:
    try:
      self.load_clip(name)
    except (OSError, EOFError, ValueError, KeyError, yaml.YAMLError, zipfile.BadZipFile) as exc:
      message = f"⚠ 跳过 `{name}`（读取失败）: {type(exc).__name__}: {exc}"
      print(f"[acp-viewer] {self.dataset.name}: {message}", flush=True)
      self.skipped_clips.append(name)
      self.load_error_gui.content = message
      return False
    self.load_error_gui.content = (f"⚠ 已跳过 {len(self.skipped_clips)} 个无法读取的片段: "
      f"{', '.join(f'`{n}`' for n in self.skipped_clips)}" if self.skipped_clips else "")
    return True

  def _safe_load_dataset(self, name: str) -> bool:
    try:
      self.skipped_clips = []
      self.load_dataset(self.datasets[name])
    except (OSError, EOFError, ValueError, KeyError, yaml.YAMLError, zipfile.BadZipFile) as exc:
      message = f"⚠ 无法打开 `{name}`: {type(exc).__name__}: {exc}"
      print(f"[acp-viewer] {message}", flush=True)
      self.load_error_gui.content = message
      self.dataset_gui.value = self.dataset.name
      return False
    return True

  def process_events(self) -> None:
    while not self.events.empty():
      action, value = self.events.get_nowait()
      if action == "dataset":
        self._safe_load_dataset(value)
      elif action == "filter":
        self.apply_filter()
      elif action == "clip" and value in self.entries:
        self._safe_load_clip(value)
      elif action == "scrub":
        self.frame = float(np.clip(value, 0, self.length - 1))
        self.last_tick = time.perf_counter()
      elif action == "step":
        self.play_gui.value = False
        self.frame = float(np.clip(int(self.frame) + value, 0, self.length - 1))
      elif action == "reset_view":
        self._request_frame(value)
      elif action == "snap":
        self._snap_camera(value)
      elif action == "review":
        context, status, note = value
        if context is not None:
          self.save_review(context, status, note)
      elif action == "note":
        context, note = value
        if context is not None:
          self.save_review(context, None, note, note_only=True)
      elif action == "next_unreviewed":
        self.next_unreviewed()

  def _markers(self, key: str, track: ContactTrack | None, frame: int, enabled: bool, radius: float) -> None:
    handle = self.marker_handles[key]
    handle.visible = enabled and track is not None
    if not handle.visible:
      return
    assert track is not None
    valid = track.flag[frame] & np.isfinite(track.pos[frame]).all(axis=1)
    handle.batched_positions = np.nan_to_num(track.pos[frame]).astype(np.float32)
    handle.batched_scales = valid.astype(np.float32) * radius
    handle.batched_colors = track.colors

  def _contact_labels(self) -> None:
    visible = {}
    # If both overlays contain the same body, the primary source owns its label.
    for track, enabled in ((self.physics, self.physics_gui.value), (self.reference, self.contacts_gui.value)):
      if track is None or not enabled:
        continue
      frame = track.frame_at(int(self.frame) / self.fps)
      for i, name in enumerate(track.names):
        if track.flag[frame, i] and np.isfinite(track.pos[frame, i]).all():
          visible[name] = track.pos[frame, i]
    for name, handle in self.contact_label_handles.items():
      handle.visible = self.labels_gui.value and name in visible
      if handle.visible:
        handle.position = visible[name] + [0, 0, 0.09]

  def _contact_timeline(self, source: ContactTrack | None, seconds: float) -> None:
    self.contact_timeline_image.visible = self.contact_timeline_gui.value and source is not None
    if not self.contact_timeline_image.visible:
      return
    assert source is not None
    if self.contact_timeline_strip is None:
      from matplotlib.backends.backend_agg import FigureCanvasAgg
      from matplotlib.figure import Figure
      fig = Figure(figsize=(6, 2.35), dpi=100)
      canvas = FigureCanvasAgg(fig)
      ax = fig.add_axes((.14, .20, .84, .74))
      image = np.where(source.flag.T[..., None], source.colors[:, None, :], np.array([240, 240, 240], np.uint8))
      end = max(float(source.times[-1]), 1e-6)
      ax.imshow(image, aspect="auto", interpolation="nearest", extent=(0, end, len(source.names) - .5, -.5))
      ax.set_yticks(np.arange(len(source.names)), [BODY_LABELS[name] for name in source.names], fontsize=8)
      ax.set_xlabel("Time (s)", fontsize=8)
      ax.tick_params(axis="x", labelsize=8)
      canvas.draw()
      self.contact_timeline_strip = np.asarray(canvas.buffer_rgba())[..., :3].copy()
      self.contact_timeline_bounds = (int(ax.bbox.x0), int(ax.bbox.x1),
        self.contact_timeline_strip.shape[0] - int(ax.bbox.y1), self.contact_timeline_strip.shape[0] - int(ax.bbox.y0), end)
    now = time.perf_counter()
    # Only the small PNG cursor changes (at most 5 Hz); matplotlib runs once/clip.
    if seconds != self.last_timeline_seconds and now - self.last_timeline_upload >= .2:
      x0, x1, y0, y1, end = self.contact_timeline_bounds
      x = int(np.clip(round(x0 + seconds / end * (x1 - x0)), x0, x1 - 1))
      image = self.contact_timeline_strip.copy()
      image[y0:y1, max(x0, x - 1):min(x1, x + 2)] = (20, 20, 20)
      self.contact_timeline_image.image = image
      self.last_timeline_upload, self.last_timeline_seconds = now, seconds

  def render(self, force: bool = False) -> None:
    frame = int(np.clip(np.floor(self.frame + 1e-8), 0, self.length - 1))
    seconds = frame / self.fps
    # Continuous transforms at 50 Hz, including slow motion and non-50 Hz clips.
    # Contacts retain floor-indexed semantics rather than advancing labels early.
    following = min(frame + 1, self.length - 1)
    alpha = float(np.clip(self.frame - frame, 0, 1))
    base = (1 - alpha) * self.base_pos[frame] + alpha * self.base_pos[following]
    q0, q1 = self.base_quat[frame], self.base_quat[following]
    q1 = q1 if np.dot(q0, q1) >= 0 else -q1
    quat = (1 - alpha) * q0 + alpha * q1
    quat /= np.linalg.norm(quat)
    rotation = yaw_rotation(quat)
    ref_frame = self.reference.frame_at(seconds) if self.reference is not None else 0
    phys_frame = self.physics.frame_at(seconds) if self.physics is not None else 0
    with self.server.atomic():
      if force or self.frame != self.last_displayed:
        self.data.qpos[self.free_adr:self.free_adr + 3] = base
        self.data.qpos[self.free_adr + 3:self.free_adr + 7] = quat
        self.data.qpos[self.joint_adr] = (1 - alpha) * self.joint_pos[frame] + alpha * self.joint_pos[following]
        mujoco.mj_kinematics(self.model, self.data)
        self.robot.update_from_mjdata(self.data)
      self._markers("reference", self.reference, ref_frame, self.contacts_gui.value, 0.045)
      self._markers("physics", self.physics, phys_frame, self.physics_gui.value, 0.045)
      self._contact_labels()
      source = self.reference if self.reference is not None else self.physics
      target_handle = self.marker_handles["targets"]
      target_handle.visible = self.targets_gui.value and source is not None
      self.target_lines.visible = target_handle.visible
      if target_handle.visible:
        assert source is not None
        pos, valid = source.targets(source.frame_at(seconds), base, rotation)
        target_handle.batched_positions = np.nan_to_num(pos).astype(np.float32)
        target_handle.batched_scales = valid.astype(np.float32) * 0.028
        target_handle.batched_colors = source.colors
        bodies = np.array([self.contact_body_ids[name] for name in source.names], dtype=int)
        self.target_lines.points = np.stack((self.data.xpos[bodies[valid]], pos[valid]), axis=1).astype(np.float32)
        self.target_lines.colors = np.repeat(source.colors[valid, None, :], 2, axis=1)
      self.active_contacts_gui.content = f"**Contacts:** {self._active_names(source, source.frame_at(seconds)) if source is not None else 'none'}"
      self._contact_timeline(source, seconds)
      self.scan_handle.visible = self.scan_gui.value
      if self.scan_gui.value:
        xy = GRID_XY @ rotation.T + base[:2]
        z = self.terrain.heights(xy, float(base[2]) + 5.0)
        points = np.column_stack((xy, z))
        self.scan_handle.points = points[np.isfinite(z)].astype(np.float32)
      self.trail_handle.visible = self.trail_gui.value
      self.timeline_gui.value = frame
      self._follow_camera(base)
      now = time.perf_counter()
      if force or now - self.last_status >= 0.1 or not self.play_gui.value:
        self.status_gui.content = (f"**{self.clip_label(self.clip_name)}**\n\nGroup: `{self.group(self.entries[self.clip_name])}`\n\n"
          f"{seconds:.2f} / {(self.length - 1) / self.fps:.2f} s · frame {frame} / {self.length - 1} · {self.fps:g} Hz")
        self.last_status = now
    self.last_displayed = self.frame

  @staticmethod
  def _active_names(track: ContactTrack | None, frame: int) -> str:
    if track is None:
      return "no sidecar"
    active = [BODY_LABELS[name]
              for i, name in enumerate(track.names) if track.flag[frame, i]]
    return " · ".join(active) or "none"

  def tick(self) -> None:
    self.process_events()
    now = time.perf_counter()
    if now - self.last_curation_check >= 1.0:
      edited = DATASET_ROOT / "edited"
      if not ONLY_DATASET and "edited" not in self.datasets and (edited / "metadata.yaml").is_file():
        self.datasets["edited"] = edited.resolve()
        self.dataset_gui.options = tuple(self.datasets)
      records = self.curation_store.read()
      if records != self.curation:
        self.curation = records
        self._show_review()
        self.apply_filter()
      self.last_curation_check = now
    if self.play_gui.value:
      self.frame += (now - self.last_tick) * self.fps * self.speed_gui.value
      if self.frame >= self.length:
        if self.loop_gui.value:
          self.frame %= self.length
        else:
          self.frame = float(self.length - 1)
          self.play_gui.value = False
    self.last_tick = now
    self.render()

  def run(self) -> None:
    announce(self.server.get_port())
    deadline = time.perf_counter()
    while self.running:
      self.tick()
      deadline += 1 / 50
      remaining = deadline - time.perf_counter()
      if remaining > 0:
        time.sleep(remaining)
      else:
        deadline = time.perf_counter()

  def close(self) -> None:
    self.running = False
    self.server.stop()


def self_test(viewer: ReferenceViewer, seconds: float) -> None:
  """Headless scene + GUI operations, contact semantics, FK and update throughput."""
  import resource

  flag = np.array([[1, 0], [1, 1], [0, 1], [1, 0], [1, 1]], dtype=bool)
  np.testing.assert_array_equal(next_onsets(flag), [[3, 1], [3, 4], [3, 4], [5, 4], [5, 5]])
  np.testing.assert_allclose(GRID_XY[[0, 16, 17, -1]], [[-0.8, -0.5], [0.8, -0.5], [-0.8, -0.4], [0.8, 0.5]])
  viewer.play_gui.value = False
  names = list(viewer.entries)
  # Exercise a distinct clip from every group, then restore the first one (LRU hit).
  selected: dict[str, str] = {}
  for name, entry in viewer.entries.items():
    selected.setdefault(viewer.group(entry), name)
  for name in selected.values():
    viewer.events.put(("clip", name))
    viewer.process_events()
    viewer.frame = min(37, viewer.length - 1)
    viewer.render(force=True)
    np.testing.assert_allclose(viewer.data.body("pelvis").xpos, viewer.base_pos[int(viewer.frame)], atol=1e-6)
    np.testing.assert_allclose(viewer.data.qpos[viewer.joint_adr], viewer.joint_pos[int(viewer.frame)])
    np.testing.assert_allclose(viewer.data.body("pelvis").xquat, viewer.base_quat[int(viewer.frame)], atol=1e-6)
    for track in (viewer.reference, viewer.physics):
      if track is None:
        continue
      for b in range(len(track.names)):
        onsets = np.flatnonzero(track.flag[1:, b] & ~track.flag[:-1, b]) + 1
        for f in (0, min(37, len(track.flag) - 1), len(track.flag) - 1):
          future = onsets[onsets > f]
          assert track.next_frame[f, b] == (future[0] if len(future) else len(track.flag))
  viewer.events.put(("clip", names[0]))
  viewer.process_events()
  viewer.events.put(("scrub", 10))
  viewer.events.put(("step", 1))
  viewer.process_events()
  assert viewer.frame == 11 and not viewer.play_gui.value
  viewer.filter_gui.value = "__no_such_clip__"
  viewer.apply_filter()
  assert viewer.clip_gui.disabled
  viewer.filter_gui.value = ""
  viewer.apply_filter()
  assert not viewer.clip_gui.disabled
  assert viewer.contacts_gui.value and viewer.physics_gui.value and viewer.targets_gui.value
  assert viewer.labels_gui.value
  source = viewer.reference if viewer.reference is not None else viewer.physics
  if source is not None:
    assert source.source in ("physics", "kinematic labeler", "motionfix")
    active_frames = np.flatnonzero(source.flag.any(axis=1))
    if len(active_frames):
      viewer.frame = min(float(source.times[active_frames[0]]) * viewer.fps, viewer.length - 1)
      viewer.render(force=True)
      assert any(handle.visible for handle in viewer.contact_label_handles.values())
      assert "Contacts:" in viewer.active_contacts_gui.content
      viewer.labels_gui.value = False
      viewer.render(force=True)
      assert not any(handle.visible for handle in viewer.contact_label_handles.values())
      viewer.labels_gui.value = True
    viewer.targets_gui.value = False
    viewer.render(force=True)
    assert not viewer.target_lines.visible
    viewer.targets_gui.value = True
    viewer.contact_timeline_gui.value = True
    viewer.render(force=True)
    assert viewer.contact_timeline_strip is not None and viewer.contact_timeline_strip.shape[0] >= 9
    assert viewer.contact_timeline_image.visible
  if ONLY_DATASET:
    viewer.tick()
    assert len(viewer.datasets) == 1 + len(EXTRA_DATASETS) and next(iter(viewer.datasets)) == viewer.dataset.name
  # Benchmark with every overlay enabled and changing poses. No pacing sleep here:
  # this measures capacity, rather than mistaking a throttled 50 Hz loop for speed.
  samples = []
  start = time.perf_counter()
  count = 0
  while time.perf_counter() - start < seconds:
    viewer.frame = float(count % viewer.length)
    t0 = time.perf_counter()
    viewer.render(force=True)
    samples.append(time.perf_counter() - t0)
    count += 1
  elapsed = time.perf_counter() - start
  fps = count / elapsed
  rss_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
  assert fps >= 50, f"Scene update throughput {fps:.1f} fps < 50 Hz"
  assert rss_mib < 8 * 1024, f"Peak RSS exceeds 8 GiB: {rss_mib:.1f} MiB"
  print(f"[SELF-TEST PASS] {viewer.dataset.name}: {len(selected)} groups switched, "
        f"{count} updates / {elapsed:.2f}s = {fps:.1f} fps, "
        f"p95={np.percentile(samples, 95) * 1000:.2f}ms, peak RSS={rss_mib:.1f} MiB", flush=True)


def policy_playback(checkpoint: Path, port: int, dataset: Path, host: str = "127.0.0.1", self_test: bool = False,
                    clip: str = "", simplified_collision: bool = False) -> None:
  """Live policy playback (one env, simulated in real time): ``run_play`` of the registered ACP Play task on ``dataset``, with a clip dropdown
  and a reset button in the viewer (the dataset override of ``scripts/acp/train_on_dataset.py``).

  Contact overlays (ACP Stage 2): the contact targets the tracker observes (``contact_targets`` of the command), the planner's predicted
  next contacts (``planner_state_dict`` of the checkpoint, run as ``ACPRunner.plan_contacts`` does) and the measured physical contacts
  (``acp_contact_points``). The "Tracker contact commands" dropdown replaces the observed targets with the planner's (eta_obs = 1)."""
  import torch
  import instinct_mj.tasks.acp.config.g1.acp_tracking_cfg as acp_g1
  from instinct_mj.rl.acp.mdp import targets_to_world
  from instinct_mj.rl.acp.planner import AttentionContactPlanner
  from instinct_mj.rl.acp.runner import GOAL_SLICE, PLANNER_PROPRIO, SCAN_DIM, TARGETS_SLICE
  from instinct_mj.tasks.acp.acp_env_cfg import CONTACTS_COMMAND
  from instinct_mj.tasks.acp.mdp.commands import yaw_of_quat

  buffer = acp_g1.motion_reference_cfg.motion_buffers["TerrainMotion"]
  buffer.path, buffer.metadata_yaml = str(dataset), str(dataset / "metadata.yaml")

  import instinct_mj.scripts.instinct_rl.play as playmod
  from instinct_mj.scripts.instinct_rl.play import PlayConfig, run_play
  from mjlab.viewer import ViserPlayViewer
  from mjlab.viewer.base import VerbosityLevel, ViewerAction

  ALL_CLIPS = "(all clips, random)"
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  if simplified_collision:
    # ``run_play`` builds the env cfg itself, so the opt-in collision spec (the same switch as ``G1AcpTrackV1EnvCfg(simplified_collision=True)``)
    # is applied to the cfg it loads. Everything else -- contacts, rewards, observations, the checkpoint -- is unchanged.
    _load_env_cfg = playmod.load_env_cfg

    def _load_simplified_env_cfg(task_id: str, play: bool = False):
      cfg = _load_env_cfg(task_id, play=play)
      acp_g1.apply_simplified_collision(cfg)
      return cfg

    playmod.load_env_cfg = _load_simplified_env_cfg
    print("[acp-viewer] collision: analytic primitives (simplified_collision=True)", flush=True)
  # The play task loads the runner with the Stage-1 config (no planner), so the planner of a Stage-2 checkpoint is built here, exactly as
  # ACPRunner.__init__ builds it, from ``planner_state_dict`` (absent in a Stage-1 checkpoint → the planner overlay is hidden).
  planner = None
  checkpoint_state = torch.load(str(checkpoint.expanduser().resolve()), map_location=device, weights_only=True)
  planner_state = checkpoint_state.get("planner_state_dict") if isinstance(checkpoint_state, dict) else None
  if planner_state is not None:
    candidate = AttentionContactPlanner().to(device).eval()
    try:
      candidate.load_state_dict(planner_state)
      planner = candidate
      print(f"[acp-viewer] planner: loaded planner_state_dict of {Path(checkpoint).name}", flush=True)
    except RuntimeError as exc:
      print(f"[acp-viewer] planner: cannot load planner_state_dict ({exc}); planner overlay hidden", flush=True)
  else:
    print(f"[acp-viewer] planner: {Path(checkpoint).name} has no planner_state_dict (Stage 1); planner overlay hidden", flush=True)

  class ClipPlayViewer(ViserPlayViewer):
    """``ViserPlayViewer`` plus the clip selection (the dropdown sets the motion weights of the Curriculum State Initialization: the episode
    starts at phase 0 of the chosen clip and restarts on it after a fall / the clip end) and the ACP contact overlays."""

    def __init__(self, env, policy, **kwargs):
      super().__init__(env, lambda obs: policy(obs).clamp(-20, 20), **kwargs)  # (the action clip of training and record_rollouts)
      # The full observation dict (policy / planner / critic) of every step, as ``ACPRunner`` reads it: ``extras["observations"]``.
      self._vector_env = getattr(env, "_vec_env", None)
      self._draw_data: dict[str, Any] | None = None
      self._plan_targets: torch.Tensor | None = None  # base-yaw targets of the planner at this step (the eta_obs = 1 substitution)
      self._probe = {"steps": 0, "planner_steps": 0, "planner_empty": 0, "planner_occupied": 0, "planner_max_dist": 0.0,
                     "planner_finite": True, "observed_occupied": 0, "measured_contacts": 0,
                     "planner_used": 0, "subst_changed": 0, "subst_diff": 0.0, "layers": set(), "draw_max_origin": 0.0,
                     "ref_checks": 0, "ref_match": True, "planner_tracker_checks": 0, "planner_tracker_match": True}

    def setup(self) -> None:
      super().setup()
      env = self.env.unwrapped
      self._motion = env.scene["motion_reference"].motion_buffers["TerrainMotion"]
      self._weights0 = self._motion._motion_weights.clone()
      start = env.curriculum_manager.get_term_cfg("episode_start").func
      start.mode = "start"  # every episode starts at phase 0 of its clip
      self._names = [os.path.splitext(os.path.basename(f))[0] for f in self._motion._all_motion_files]
      groups = [e.get("group", "") for e in self._motion.yaml_data["motion_files"]]
      labels = sorted(f"{g} / {n}" for g, n in zip(groups, self._names))
      self._label_to_clip = {f"{g} / {n}": k for k, (g, n) in enumerate(zip(groups, self._names))}
      initial_label = next((label for label in labels if clip.lower() in label.lower()), None) if clip else None
      if clip:
        message = f"starting on {initial_label}" if initial_label else f"no clip matches {clip!r}; starting on {ALL_CLIPS}"
        print(f"[acp-viewer] clip: {message}", flush=True)
      with self._server.gui.add_folder("ACP clip"):
        self._clip_dropdown = self._server.gui.add_dropdown("Clip", options=[ALL_CLIPS, *labels],
                                                            initial_value=initial_label or ALL_CLIPS)
        self._clip_info = self._server.gui.add_html("")
        play_button = self._server.gui.add_button("Play clip from start")

        @play_button.on_click
        def _(_) -> None:
          self._actions.append((ViewerAction.CUSTOM, {"type": "acp_clip", "label": self._clip_dropdown.value}))
      if initial_label is not None:  # the same path as the button: motion weights + reset to phase 0, before the first step
        self._actions.append((ViewerAction.CUSTOM, {"type": "acp_clip", "label": initial_label}))
      self._setup_contacts(env)

    # -- contact overlays -------------------------------------------------------------------------

    def _setup_contacts(self, env) -> None:
      """Markers of the three contact layers, in the colour / sphere / line style of the ``refs`` mode."""
      self._command_term = env.command_manager.get_term(CONTACTS_COMMAND)
      self._scanner = env.scene["height_scanner"]
      self._contact_sensor = env.scene["acp_contact_points"]
      # the sensor lists its primaries in model body order: index of every contact link (CONTACT_ORDER order)
      self._sensor_ids = [self._contact_sensor.primary_names.index(name) for name in CONTACT_LINKS]
      self._link_ids = env.scene["robot"].find_bodies(list(CONTACT_LINKS), preserve_order=True)[0]
      with self._server.gui.add_folder("ACP contacts"):
        self._tracker_cb = self._server.gui.add_checkbox("Tracker targets · contact_targets (policy input)", True)
        self._planner_cb = self._server.gui.add_checkbox("Planner targets · top-1 + occupancy", True)
        self._measured_cb = self._server.gui.add_checkbox("Measured contacts · acp_contact_points", True)
        self._collision_cb = self._server.gui.add_checkbox(
          "Show collision geoms · G3 / G4 (semi-transparent)", True,
          hint="Body collision meshes + hand half-spheres (group 3) and foot capsules (group 4), default-hidden by mjviser")
        self._command_dd = self._server.gui.add_dropdown("Tracker contact commands", ("reference", "planner"), initial_value="planner")
        self._contact_info = self._server.gui.add_markdown("")
        self._server.gui.add_markdown(
          "🟦 Feet · 🟩 Knees · 🟥 Hands · 🟧 Elbows · 🟪 Pelvis\n\n"
          "**Tracker targets:** the 9 `contact_targets` of the policy observation this step (base-yaw → world, occupancy flag = the sphere "
          "is drawn), i.e. the reference contacts command or, when the source below is `planner`, the planner's targets.\n\n"
          "**Planner targets:** the Stage-2 planner's next contacts (top-1 affordance point, occupancy > 0.5), same style.\n\n"
          "**Measured contacts:** solid spheres (≈4.5 cm) at the strongest `acp_contact_points` contact of every link.\n\n"
          "**Collision geoms:** the robot's collision geometry (group 3: body meshes and the hand half-spheres on the wrist yaw links; "
          "group 4: foot capsules), semi-transparent; the `Groups` tab `G3` / `G4` checkboxes follow this switch.\n\n"
          "**Tracker contact commands:** `planner` (the default) feeds the planner's targets into the tracker's observation instead of the "
          "reference ones (`plan_contacts(obs, planner_obs, eta_obs=1)` of the runner); `reference` runs the reference command.")
      if planner is None:  # Stage-1 checkpoint: the planner layer has nothing to show
        self._planner_cb.visible = False
        self._command_dd.visible = False
      # Collision geoms: mjviser hides groups 3/4 by default (the visuals are group 1); show them see-through so the hands' half-spheres
      # and the foot capsules are visible under the contact overlays.
      self._collision_handles = [mg.handle for mg in self._scene._mesh_groups if mg.group_id in (3, 4)]

      @self._collision_cb.on_update
      def _(_) -> None:
        self._set_collision_geoms_visible(bool(self._collision_cb.value))

      self._set_collision_geoms_visible(True)
      sphere = trimesh.creation.icosphere(subdivisions=2, radius=1)
      self._marker_handles: dict[str, Any] = {}
      for name, opacity in (("tracker", 0.38), ("planner", 0.38), ("measured", 1.0)):
        self._marker_handles[name] = self._server.scene.add_batched_meshes_simple(
          f"/acp_contacts/{name}", sphere.vertices, sphere.faces,
          batched_positions=np.zeros((9, 3)), batched_wxyzs=np.tile([1, 0, 0, 0], (9, 1)),
          batched_scales=np.zeros(9), batched_colors=CONTACT_COLORS, opacity=opacity,
          cast_shadow=False, receive_shadow=False)
      self._line_handles = {name: self._server.scene.add_line_segments(
        f"/acp_contacts/{name}_lines", np.zeros((0, 2, 3), np.float32), (180, 180, 180), line_width=1.25, visible=False)
        for name in ("tracker", "planner")}

    def _group_tab_checkbox(self, group: int) -> viser.GuiCheckboxHandle | None:
      """The mjviser ``Groups`` tab checkbox of a geom group (same state as ``geom_groups_visible``), if the scene GUI created it."""
      for handle in getattr(self._server.gui, "_gui_input_handle_from_uuid", {}).values():
        if isinstance(handle, viser.GuiCheckboxHandle) and handle.label == f"G{group}":
          return handle
      return None

    def _set_collision_geoms_visible(self, visible: bool) -> None:
      """Show/hide the robot's collision geoms (groups 3 and 4), semi-transparent, and keep the ``Groups`` tab checkboxes in step."""
      scene = self._scene

      def mutate() -> None:
        scene.geom_groups_visible[3] = visible
        scene.geom_groups_visible[4] = visible
        scene._sync_visibilities()

      scene._apply_visualization_change(mutate)  # under _update_lock; also refreshes from the cached state
      for group in (3, 4):
        box = self._group_tab_checkbox(group)
        if box is not None and bool(box.value) != visible:
          box.value = visible
      for handle in self._collision_handles:
        handle.opacity = COLLISION_OPACITY

    def _tracker_obs(self, obs: torch.Tensor) -> torch.Tensor:
      """The policy observation with the planner's targets in the contact-target slice when the dropdown says ``planner`` (eta_obs = 1)."""
      if planner is None or self._command_dd.value != "planner" or self._plan_targets is None:
        return obs
      previous = obs[:, TARGETS_SLICE]
      obs = obs.clone()
      obs[:, TARGETS_SLICE] = self._plan_targets.flatten(1)
      if self_test:
        self._probe["planner_used"] += 1
        self._probe["subst_changed"] += int(bool((previous != obs[:, TARGETS_SLICE]).any()))
        self._probe["subst_diff"] = max(self._probe["subst_diff"], float((obs[:, TARGETS_SLICE] - self._plan_targets.flatten(1)).abs().max()))
      return obs

    def _capture_contacts(self, obs: torch.Tensor, extras: dict) -> torch.Tensor:
      """Contact layers (world frame) of the current observation, before the env steps (the robot pose of the frame about to be drawn), and
      the observation the policy runs on: with the ``planner`` command source the tracker's contact-target slice is replaced by the planner's
      targets (``plan_contacts(obs, planner_obs, eta_obs=1)`` of the runner)."""
      env = self.env.unwrapped
      scanner = self._scanner
      est_pos, est_yaw = scanner.estimated_pos_w, yaw_of_quat(scanner.estimated_quat_w)
      # 1. the planner's next contacts: the frozen planner on the env's planner group (as ACPRunner.plan_contacts)
      self._plan_targets = None
      plan_w = torch.zeros(1, 9, 3, device=device)
      plan_valid = torch.zeros(1, 9, dtype=torch.bool, device=device)
      tgt_b = None
      planner_obs = extras.get("observations", {}).get("planner")
      if planner is not None and planner_obs is not None:
        points = planner_obs.view(-1, SCAN_DIM, 3)
        out = planner(obs[:, :PLANNER_PROPRIO], obs[:, GOAL_SLICE], points)
        tgt_b, valid = AttentionContactPlanner.decode(out, points)
        self._plan_targets = tgt_b
        plan_w, plan_valid = targets_to_world(est_pos, est_yaw, tgt_b), valid
      # 2. the contact targets the tracker observes: the ``contact_targets`` term of the policy observation after the source substitution
      obs = self._tracker_obs(obs)
      ref_b = obs[:, TARGETS_SLICE].view(-1, 9, 3)
      ref_valid = ref_b.abs().sum(-1) > 1e-6
      ref_w = targets_to_world(est_pos, est_yaw, ref_b)
      # 3. the measured physical contact of every link (strongest contact, sensor primary order → CONTACT_ORDER)
      contact = self._contact_sensor.data
      found = contact.found[0][self._sensor_ids].bool() if contact.found is not None else torch.zeros(9, dtype=torch.bool, device=device)
      pos = contact.pos[0][self._sensor_ids] if contact.pos is not None else torch.zeros(9, 3, device=device)
      measured = torch.where(found.unsqueeze(-1), pos, torch.zeros_like(pos))
      links = env.scene["robot"].data.body_link_pos_w[0, self._link_ids]
      to_np = lambda t: np.nan_to_num(t.detach().float().cpu().numpy())  # fmt: skip
      self._draw_data = {
        "tracker": (to_np(ref_w[0]), ref_valid[0].cpu().numpy()),
        "planner": (to_np(plan_w[0]), plan_valid[0].cpu().numpy().astype(bool)),
        "measured": (to_np(measured), found.cpu().numpy()),
        "links": to_np(links),
      }
      if self_test:
        probe = self._probe
        probe["steps"] += 1
        occupied = int(ref_valid.sum())
        probe["observed_occupied"] += occupied
        probe["measured_contacts"] += int(found.sum())
        if self._command_dd.value == "planner" and tgt_b is not None:
          probe["planner_tracker_checks"] += 1
          probe["planner_tracker_match"] = probe["planner_tracker_match"] and bool(torch.equal(ref_b, tgt_b))
        elif self._command_dd.value == "reference":  # the drawn layer is the command's targets in the world frame
          _, command_valid = self._command_term.targets(estimated=True)
          probe["ref_checks"] += 1
          probe["ref_match"] = probe["ref_match"] and bool(np.array_equal(command_valid[0].cpu().numpy(), ref_valid[0].cpu().numpy()))
        if planner is not None and planner_obs is not None:
          planner_world, planner_valid = self._draw_data["planner"]
          probe["planner_steps"] += 1
          probe["planner_finite"] = probe["planner_finite"] and bool(np.isfinite(planner_world).all())
          probe["planner_occupied"] += int(planner_valid.sum())
          if planner_valid.any():
            distance = np.linalg.norm(planner_world[planner_valid] - to_np(est_pos[0]), axis=-1)
            probe["planner_max_dist"] = max(probe["planner_max_dist"], float(distance.max()))
          else:
            probe["planner_empty"] += 1
        if probe["steps"] == 30:  # exercise both command sources in the self-test (the default is planner)
          self._command_dd.value = "reference"
        elif probe["steps"] == 60:
          self._command_dd.value = "planner"
      return obs

    def _draw_contacts(self) -> None:
      data = self._draw_data
      # The mjlab viewer keeps the robot at the origin for the camera (``camera_tracking_enabled``): the scene geometry is shifted by
      # ``_scene_offset = -tracked body``. A custom overlay must use the same shift, otherwise it is drawn in world coordinates far
      # from the framed robot (invisible after the first few metres of walking).
      offset = np.asarray(self._scene._scene_offset, dtype=np.float32)
      for name, checkbox, radius in (("tracker", self._tracker_cb, TARGET_RADIUS),
                                     ("planner", self._planner_cb, TARGET_RADIUS),
                                     ("measured", self._measured_cb, CONTACT_RADIUS)):
        handle, line = self._marker_handles[name], self._line_handles.get(name)
        visible = bool(checkbox.value) and data is not None
        handle.visible = visible
        if line is not None:
          line.visible = visible
        if not visible:
          continue
        positions, valid = data[name]
        positions = positions + offset
        handle.batched_positions = positions.astype(np.float32)
        handle.batched_scales = valid.astype(np.float32) * radius
        if self_test and valid.any():
          self._probe["layers"].add(name)
          self._probe["draw_max_origin"] = max(self._probe["draw_max_origin"], float(np.linalg.norm(positions[valid], axis=-1).max()))
        if line is not None:
          line.points = (np.stack((data["links"][valid] + offset, positions[valid]), axis=1).astype(np.float32)
                         if valid.any() else np.zeros((0, 2, 3), np.float32))
          line.colors = np.repeat(CONTACT_COLORS[valid][:, None, :], 2, axis=1)

    def _contact_status(self) -> str:
      if self._draw_data is None:
        return "**Contacts:** waiting for the first step"
      counts = {name: int(self._draw_data[name][1].sum()) for name in ("tracker", "planner", "measured")}
      source = self._command_dd.value if planner is not None else "reference"
      return (f"**Occupied:** tracker {counts['tracker']}/9 · planner {counts['planner']}/9 · measured {counts['measured']}/9\n\n"
              f"**Tracker contact commands:** `{source}`")

    def _execute_step(self) -> bool:
      """One obs → policy → env.step cycle that also captures the contact layers (``BaseViewer._execute_step``)."""
      try:
        with torch.no_grad():
          if self._vector_env is not None:
            obs, extras = self._vector_env.get_observations()
          else:  # no access to the planner group: reference targets only
            obs, extras = self.env.get_observations(), {}
          policy_obs = self._capture_contacts(obs, extras)
          self.env.step(self.policy(policy_obs))
          self._step_count += 1
          self._stats_steps += 1
          return True
      except Exception:
        self._last_error = traceback.format_exc()
        self.log(f"[ERROR] Exception during step:\n{self._last_error}", VerbosityLevel.SILENT)
        self.pause()
        return False

    def _handle_custom_action(self, action, payload) -> bool:
      if isinstance(payload, dict) and payload.get("type") == "acp_clip":
        weights = self._weights0.clone() if payload["label"] == ALL_CLIPS else torch.zeros_like(self._weights0)
        if payload["label"] != ALL_CLIPS:
          weights[self._label_to_clip[payload["label"]]] = 1.0
        self._motion._motion_weights[:] = weights
        self.reset_environment()
        return True
      return super()._handle_custom_action(action, payload)

    def sync_env_to_viewer(self) -> None:
      super().sync_env_to_viewer()
      self._draw_contacts()
      if self._counter % 10 == 0:
        clip = int(self._motion._assigned_env_motion_selection[0])
        self._clip_info.content = f'<div style="font-size:0.85em;padding:0 1em 0.5em 1em;"><strong>Playing:</strong> {self._names[clip]}</div>'
        self._contact_info.content = self._contact_status()

  # Supply the host/port explicitly; run_play otherwise constructs its own server.
  server = viser.ViserServer(host=host, port=port, label="ACP · Policy")
  viewer_box: dict[str, ClipPlayViewer] = {}

  def make_viewer(env, policy):
    viewer_box["viewer"] = ClipPlayViewer(env, policy, viser_server=server)
    return viewer_box["viewer"]

  playmod.ViserPlayViewer = make_viewer
  announce(server.get_port())
  try:
    run_play("Instinct-ACP-Track-G1-Play-v1", PlayConfig(
      viewer="viser", num_envs=1, checkpoint_file=str(checkpoint.expanduser().resolve()), video=False,
      max_steps=200 if self_test else None))
  finally:
    server.stop()
  if self_test:
    policy_self_test(viewer_box["viewer"], planner)


def policy_self_test(viewer, planner, min_steps: int = 10) -> None:
  """Headless checks of the policy contact overlays: the planner runs on the env's planner group, its targets are finite and near the
  robot, the ``planner`` command source replaces the tracker's targets, and the measured-contacts sensor is read."""
  probe = viewer._probe
  data = viewer._draw_data
  assert data is not None and probe["steps"] >= min_steps, f"too few steps captured: {probe['steps']}"
  assert planner is not None, "no planner in the checkpoint: the self-test needs a Stage-2 model"
  assert probe["planner_steps"] >= min_steps, f"planner ran on {probe['planner_steps']} steps"
  assert probe["planner_finite"], "planner produced non-finite targets"
  assert probe["planner_occupied"] > 0, "planner never predicted an occupied contact"
  assert probe["planner_max_dist"] < 3.0, f"planner target {probe['planner_max_dist']:.2f} m from the estimated pelvis"
  assert probe["observed_occupied"] > 0, "the reference command had no occupied target"
  assert probe["measured_contacts"] > 0, "the acp_contact_points sensor reported no contact"
  assert probe["planner_used"] >= min_steps, f"the planner command source ran on {probe['planner_used']} steps"
  assert probe["subst_changed"] > 0, "the planner command source left the tracker's targets unchanged"
  assert probe["subst_diff"] == 0.0, "the tracker observation does not carry exactly the planner's targets"
  assert viewer._command_dd.value == "planner" and viewer._command_dd.visible
  for name in ("tracker", "planner", "measured"):
    assert name in probe["layers"] and viewer._marker_handles[name].visible, f"contact layer {name} was never drawn"
  assert viewer._line_handles["planner"].visible
  # the overlays are drawn with the scene's camera-tracking shift, so they stay around the framed robot instead of drifting away in world coords
  assert probe["draw_max_origin"] < 3.0, f"a contact marker was drawn {probe['draw_max_origin']:.2f} m from the scene origin"
  # collision geoms (groups 3/4): on by default, see-through, with the hand half-spheres and foot capsules among the handles
  handles = [(mg.group_id, mg.handle) for mg in viewer._scene._mesh_groups if mg.group_id in (3, 4)]
  assert viewer._collision_cb.value and handles, "no collision geoms were created for the robot"
  assert {group for group, _ in handles} == {3, 4}, "collision geoms of group 3 and/or 4 are missing"
  assert all(handle.visible for _, handle in handles), "collision geoms are not visible although the checkbox is on"
  assert all(abs(float(handle.opacity) - COLLISION_OPACITY) < 1e-6 for _, handle in handles), "collision geoms are not semi-transparent"
  assert any(group == 3 and "wrist_yaw_link" in handle.name for group, handle in handles), "the hand half-spheres are not drawn"
  assert any(group == 4 and "ankle_roll_link" in handle.name for group, handle in handles), "the foot capsules are not drawn"
  # the tracker layer follows the policy's command source: the reference command's targets, or the planner's (the default)
  assert probe["ref_checks"] > 0 and probe["ref_match"], "in reference mode the tracker layer is not the reference command's targets"
  assert probe["planner_tracker_checks"] > 0 and probe["planner_tracker_match"], "in planner mode the tracker layer is not the planner's targets"
  print(f"[SELF-TEST PASS] {probe['steps']} steps · planner occupied {probe['planner_occupied']} times "
        f"(max {probe['planner_max_dist']:.2f} m from the pelvis, {probe['planner_empty']} empty steps) · "
        f"planner commands {probe['planner_used']} steps · observed targets {probe['observed_occupied']} · "
        f"measured contacts {probe['measured_contacts']} · markers within {probe['draw_max_origin']:.2f} m of the framed robot", flush=True)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  subparsers = parser.add_subparsers(dest="mode", required=True)
  refs = subparsers.add_parser("refs", help="Browse TerrainMotion reference / rollout clips")
  refs.add_argument("--dataset", type=Path, required=True)
  refs.add_argument("--port", type=int, default=8188)
  refs.add_argument("--self-test", action="store_true", help="Build scene, switch clips, benchmark then exit")
  refs.add_argument("--test-seconds", type=float, default=3.0)
  refs.add_argument("--only", action="store_true", help="List only --dataset in the dataset dropdown")
  refs.add_argument("--also", type=Path, action="append", default=[], help="Also list this dataset in the dropdown (repeatable)")
  policy = subparsers.add_parser("policy", help="Stream a checkpoint using the registered ACP Play task")
  policy.add_argument("--checkpoint", type=Path, default=None, help="default: the newest logs/instinct_rl/g1_acp_track_final6/*/model_*.pt")
  policy.add_argument("--dataset", type=Path, default=Path(os.environ.get("ACP_DATASET", "~/Datasets/acp/v3_train6")).expanduser())
  policy.add_argument("--port", type=int, default=8192)
  policy.add_argument("--host", default="127.0.0.1", help="bind address of the viser server (0.0.0.0: reachable from other machines)")
  policy.add_argument("--clip", default=os.environ.get("ACP_CLIP", ""),
                      help="start on the first clip whose 'group / name' contains this text (e.g. ladder__clean); default: all clips")
  policy.add_argument("--self-test", action="store_true", help="Run 200 steps headless, check the contact overlays, then exit")
  policy.add_argument("--simplified-collision", action="store_true",
                      help="Collide with the analytic primitives of assets/acp_g1.py instead of the torso / arm / pelvis / leg meshes")
  args = parser.parse_args()
  if args.mode == "policy":
    checkpoint = args.checkpoint or max(Path(".").glob("logs/instinct_rl/g1_acp_track_final6/*/model_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    policy_playback(checkpoint, args.port, args.dataset.resolve(), args.host, self_test=args.self_test, clip=args.clip,
                    simplified_collision=args.simplified_collision)
    return
  global ONLY_DATASET
  ONLY_DATASET = bool(getattr(args, "only", False))
  EXTRA_DATASETS[:] = [p.expanduser().resolve() for p in getattr(args, "also", [])]
  viewer = ReferenceViewer(args.dataset.resolve(), args.port)
  signal.signal(signal.SIGTERM, lambda *_: setattr(viewer, "running", False))
  signal.signal(signal.SIGINT, lambda *_: setattr(viewer, "running", False))
  try:
    if args.self_test:
      self_test(viewer, args.test_seconds)
    else:
      viewer.run()
  finally:
    viewer.close()


if __name__ == "__main__":
  main()
