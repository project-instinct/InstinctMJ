"""Client-side ACP reference player (three.js page + binary clip payloads).

  uv run --no-sync python acp_player/server.py refs \
      --dataset /home/duanxin/Datasets/acp/v3_clean --only \
      --also /home/duanxin/Datasets/acp/v3_extra --port 8188

The old viser viewer pushed ~60 scene-node updates per display frame (195 KiB/s
while playing). This server instead bakes one clip into a single binary payload
(robot body poses at every frame, contact flags/positions, target onsets) and the
browser plays it locally at 50 Hz. The server only answers when the page loads,
switches dataset/clip or saves a review.

Geometry comes from the same MuJoCo model and sidecars as ``webplay_acp.py``:
``robot_model()`` for the mesh, ``ContactTrack`` for labels/targets, ``Terrain``
for the STL surface and ``CurationStore`` for review records.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import zipfile
import gzip
import json
import mimetypes
import os
import signal
import socket
import struct
import sys
import threading
import time
import traceback
import zipfile
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import webplay_acp as base  # noqa: E402  (shared model / sidecar / curation code)

STATIC = Path(__file__).resolve().parent / "static"
DATASET_ROOT = base.DATASET_ROOT
MAGIC = b"ACPC"
MAX_BAKED_CLIPS = 24

# Hand-fixed contact flags live outside the dataset so ``v3_final`` stays generated
# content. One npz per (dataset name, clip path); ``apply`` copies them into a
# dataset (e.g. a training copy) by rewriting that clip's ``.contacts.npz``.
CONTACT_EDIT_ROOT = Path(os.environ.get("ACP_CONTACT_EDIT_ROOT", str(DATASET_ROOT / "contact_edits")))


def contact_edit_path(dataset: str, clip: str) -> Path:
  return CONTACT_EDIT_ROOT / dataset / clip


def prune_contact_edit_dirs(path: Path) -> None:
  """Remove the now-empty dataset/clip directories after the last edit is gone."""
  root = CONTACT_EDIT_ROOT.resolve()
  current = path.parent.resolve()
  while current != root and root in current.parents:
    try:
      current.rmdir()
    except OSError:
      break
    current = current.parent


def load_contact_edit(path: Path, shape: tuple[int, int],
                      names: list[str]) -> tuple[np.ndarray | None, dict, np.ndarray | None]:
  """Read a saved edit, ignoring anything whose shape / body order drifted.

  ``contact_pos_w`` is optional: edits saved before positions were uploaded only
  carry flags, and the baker falls back to FK positions for new contacts.
  """
  if not path.exists():
    return None, {}, None
  try:
    with np.load(path, allow_pickle=False) as arrays:
      flag = arrays["contact_flag"].astype(bool)
      saved_names = [str(name) for name in arrays["body_names"]] if "body_names" in arrays else names
      saved_at = str(arrays["saved_at"]) if "saved_at" in arrays else ""
      changed = int(arrays["changed"]) if "changed" in arrays else -1
      positions = arrays["contact_pos_w"].astype(np.float32) if "contact_pos_w" in arrays else None
  except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile) as exc:
    print(f"[acp-player] ignoring contact edit {path}: {type(exc).__name__}: {exc}", flush=True)
    return None, {}, None
  if flag.shape != tuple(shape) or saved_names != list(names):
    print(f"[acp-player] ignoring contact edit {path}: shape/name mismatch", flush=True)
    return None, {}, None
  if positions is not None and positions.shape != tuple(shape) + (3,):
    print(f"[acp-player] ignoring contact positions in {path}: shape mismatch", flush=True)
    positions = None
  return flag, {"edited": True, "saved_at": saved_at, "changed": changed}, positions

# Kimodo interactive demo (viser) runs on a loopback-only port; this server exposes it
# under /kimodo/ so a single forwarded port (8188) serves both pages. HTTP and the
# viser websocket are tunnelled verbatim (path prefix stripped) to the demo process.
KIMODO_PREFIX = "/kimodo"
KIMODO_HOST = os.environ.get("KIMODO_DEMO_HOST", "127.0.0.1")
KIMODO_PORT = int(os.environ.get("KIMODO_DEMO_PORT", "8191"))
# Workhorse egocentric-render dataset viewer (plans/workhorse_viewer, port 8190), exposed under
# /workhorse/ the same way (HTTP only, path prefix stripped).
WORKHORSE_PREFIX = "/workhorse"
WORKHORSE_TARGET = ("127.0.0.1", int(os.environ.get("WORKHORSE_VIEWER_PORT", "8190")))
# Live Stage-1 policy playback (webplay_acp.py policy, viser, port 8192), exposed under /policy/ like /kimodo/ (HTTP + websocket).
POLICY_PREFIX = "/policy"
POLICY_TARGET = ("127.0.0.1", int(os.environ.get("ACP_POLICY_LIVE_PORT", "8192")))
# Live Workhorse tracker playback (webplay_workhorse.py, viser, port 8193), exposed under /tracker/ like /policy/ (HTTP + websocket).
TRACKER_PREFIX = "/tracker"
TRACKER_TARGET = ("127.0.0.1", int(os.environ.get("WORKHORSE_TRACKER_PORT", "8193")))


def quat_to_mat(wxyz: np.ndarray) -> np.ndarray:
  w, x, y, z = wxyz
  return np.array([
    [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
    [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
    [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
  ])


def pack_payload(header: dict, arrays: dict[str, np.ndarray]) -> bytes:
  """8-byte aligned framing + JSON header + raw typed arrays.

  Each array starts on an 8-byte boundary so the browser can wrap it in
  Float32Array/Int32Array views without copying.
  """
  blobs: list[bytes] = []
  meta: dict[str, dict] = {}
  offset = 0
  for name, array in arrays.items():
    data = np.ascontiguousarray(array)
    pad = (-offset) % 8
    if pad:
      blobs.append(b"\x00" * pad)
      offset += pad
    meta[name] = {"offset": offset, "dtype": data.dtype.str, "shape": list(data.shape)}
    raw = data.tobytes()
    blobs.append(raw)
    offset += len(raw)
  header = dict(header, arrays=meta)
  encoded = json.dumps(header, separators=(",", ":"), allow_nan=False).encode()
  pad = (-(len(MAGIC) + 4 + len(encoded))) % 8
  encoded += b" " * pad
  return MAGIC + struct.pack("<I", len(encoded)) + encoded + b"".join(blobs)


class Store:
  """Dataset metadata + curation, shared by all HTTP request threads."""

  def __init__(self, paths: list[Path]):
    self.lock = threading.Lock()
    self.paths = {path.name: path.resolve() for path in paths}
    self.meta: dict[str, tuple[float, dict]] = {}
    self.meta_lock = threading.Lock()
    self.note_drafts: dict[tuple[str, str], str] = {}

  def metadata(self, name: str) -> dict:
    path = self.paths[name]
    metadata_path = path / "metadata.yaml"
    mtime = metadata_path.stat().st_mtime_ns
    cached = self.meta.get(name)
    if cached is not None and cached[0] == mtime:
      return cached[1]
    with self.meta_lock:
      with metadata_path.open() as stream:
        metadata = yaml.safe_load(stream)
      entries = {entry["motion_file"]: entry for entry in metadata["motion_files"]}
      if not entries:
        raise ValueError(f"No motions in {path}")
      terrains = {int(t["terrain_id"]): path / t["terrain_file"] for t in metadata["terrains"]}
      data = {"entries": entries, "terrains": terrains, "path": path}
      self.meta[name] = (mtime, data)
    return data

  def curation(self, name: str) -> dict[str, dict]:
    return base.CurationStore(self.paths[name]).read()

  def review(self, name: str, clip: str, status: str | None, note: str, note_only: bool) -> dict:
    store = base.CurationStore(self.paths[name])
    with self.lock:
      records = store.update(clip, status, note, note_only=note_only)
    if note_only and clip not in records:
      self.note_drafts[(name, clip)] = note
    else:
      self.note_drafts.pop((name, clip), None)
    return records

  def config(self, default: str) -> dict:
    datasets = []
    for name in self.paths:
      try:
        data = self.metadata(name)
        curation = self.curation(name)
      except (OSError, yaml.YAMLError, ValueError, KeyError):
        # A dataset being rewritten by another process still shows its last
        # good clip list; a brand new one simply reports the read error.
        cached = self.meta.get(name)
        if cached is None:
          datasets.append({"name": name, "clips": [], "error": "metadata unreadable"})
          continue
        data = cached[1]
        curation = {}
      clips = [{
        "name": clip,
        "group": base.ReferenceViewer.group(entry),
        "status": curation.get(clip, {}).get("status", "unreviewed"),
        "note": curation.get(clip, {}).get("note", ""),
      } for clip, entry in data["entries"].items()]
      datasets.append({"name": name, "clips": clips})
    return {
      "default": default,
      "datasets": datasets,
      "marks": base.REVIEW_MARKS,
      "body_labels": base.BODY_LABELS,
      "contact_colors": {key: list(value) for key, value in base.COLORS.items()},
      "contact_links": base.CONTACT_LINKS,
      "robot_url": f"/api/robot?v={int(ROBOT_MTIME)}",
      "grid": base.GRID_XY.astype(np.float32).reshape(-1).tolist(),
    }


ROBOT_MTIME = 0


class Baker:
  """MuJoCo FK per frame + contact sidecars + terrain, cached."""

  def __init__(self, store: Store, model: mujoco.MjModel):
    self.store = store
    self.model = model
    self.data = mujoco.MjData(model)
    self.free_adr = int(model.jnt_qposadr[np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)[0]])
    self.joint_ids = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE)
    self.joint_names = [model.joint(int(j)).name for j in self.joint_ids]
    self.joint_adr = model.jnt_qposadr[self.joint_ids]
    self.terrain_cache = base.TerrainCache()
    self.body_names = [model.body(i).name for i in range(1, model.nbody)]
    self.contact_body_index = {
      link: int(model.body(link).id) - 1 for link in base.CONTACT_LINKS
    }
    self.lock = threading.Lock()
    self.bake_lock = threading.Lock()
    self.clips: OrderedDict[tuple[str, str, int], bytes] = OrderedDict()
    self.terrains: dict[tuple[str, int, int], tuple[bytes, str]] = {}
    self.robot = self._robot_geometry()

  def _robot_geometry(self) -> bytes:
    """Merge visible mesh geoms into indexed per-body buffers (vertices in body frame).

    Normals are averaged over the shared mesh vertices (the meshes are dense
    triangle soups), stored as int8. Colours are the geom rgba as uint8. Index
    values are local to each body's vertex range, so uint16 is enough.
    """
    global ROBOT_MTIME
    positions: list[np.ndarray] = []
    normals: list[np.ndarray] = []
    colors: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    bodies: list[dict] = []
    vertex_base = 0
    index_base = 0
    for body_id in range(1, self.model.nbody):
      body = self.model.body(body_id)
      names: list[str] = []
      body_vertex_start = vertex_base
      body_index_start = index_base
      for geom_id in range(self.model.ngeom):
        if int(self.model.geom_bodyid[geom_id]) != body_id:
          continue
        if int(self.model.geom_group[geom_id]) not in (1, 2):
          continue
        if int(self.model.geom_type[geom_id]) != mujoco.mjtGeom.mjGEOM_MESH:
          continue
        mesh_id = int(self.model.geom_dataid[geom_id])
        vert_adr = int(self.model.mesh_vertadr[mesh_id])
        vert_num = int(self.model.mesh_vertnum[mesh_id])
        face_adr = int(self.model.mesh_faceadr[mesh_id])
        face_num = int(self.model.mesh_facenum[mesh_id])
        verts = self.model.mesh_vert[vert_adr:vert_adr + vert_num].astype(np.float64)
        faces = self.model.mesh_face[face_adr:face_adr + face_num].astype(np.int64)
        rotation = quat_to_mat(self.model.geom_quat[geom_id])
        verts = verts @ rotation.T + self.model.geom_pos[geom_id]
        tri = verts[faces]
        face_normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        face_normal /= np.maximum(np.linalg.norm(face_normal, axis=1, keepdims=True), 1e-12)
        smooth = np.zeros_like(verts)
        for corner in range(3):
          np.add.at(smooth, faces[:, corner], face_normal)
        smooth /= np.maximum(np.linalg.norm(smooth, axis=1, keepdims=True), 1e-12)
        positions.append(verts)
        normals.append(np.clip(np.rint(smooth * 127.0), -127, 127))
        colors.append(np.tile(np.rint(np.clip(self.model.geom_rgba[geom_id][:3], 0.0, 1.0) * 255.0), (vert_num, 1)))
        indices.append((faces + (vertex_base - body_vertex_start)).astype(np.uint16))
        vertex_base += vert_num
        index_base += faces.size
        names.append(self.model.mesh(mesh_id).name)
      if names:
        bodies.append({"name": body.name,
                       "vertex_start": body_vertex_start, "vertex_count": vertex_base - body_vertex_start,
                       "index_start": body_index_start, "index_count": index_base - body_index_start,
                       "geoms": names})
    pos = np.concatenate(positions).astype(np.float32) if positions else np.zeros((0, 3), np.float32)
    nor = np.concatenate(normals).astype(np.int8) if normals else np.zeros((0, 3), np.int8)
    col = np.concatenate(colors).astype(np.uint8) if colors else np.zeros((0, 3), np.uint8)
    idx = np.concatenate(indices).astype(np.uint16) if indices else np.zeros(0, np.uint16)
    header = {"bodies": bodies, "body_names": self.body_names, "vertex_count": len(pos)}
    blob = pack_payload(header, {"positions": pos, "normals": nor, "colors": col, "indices": idx})
    ROBOT_MTIME = int(time.time())
    return blob

  def terrain(self, dataset: str, terrain_id: int) -> tuple[bytes, str]:
    path = self.store.metadata(dataset)["terrains"][terrain_id].resolve()
    key = (dataset, terrain_id, path.stat().st_mtime_ns)
    with self.lock:
      cached = self.terrains.get(key)
    if cached is not None:
      return cached
    surface = self.terrain_cache.get(path)
    blob = pack_payload({"name": path.name}, {
      "positions": surface.vertices.astype(np.float32),
      "indices": surface.faces.astype(np.uint32),
    })
    value = (blob, f'"{path.stat().st_mtime_ns}"')
    with self.lock:
      # Terrain cache holds a handful of STLs; keep the payloads in step.
      if len(self.terrains) > 16:
        self.terrains.pop(next(iter(self.terrains)))
      self.terrains[key] = value
    return value

  def clip(self, dataset: str, clip: str) -> bytes:
    entries = self.store.metadata(dataset)["entries"]
    if clip not in entries:
      raise KeyError(clip)
    path = self.store.paths[dataset] / clip
    edit = contact_edit_path(dataset, clip)
    # A save can land while we are baking; only cache the blob when the inputs
    # still match the key we computed, otherwise the next request would get a
    # stale bake under a fresh mtime.
    for _ in range(3):
      stat = path.stat()
      # The sidecars are relabeled in place (contact fixes, physics passes) while
      # the motion file stays untouched, so their mtimes belong to the key too.
      sidecars = tuple(p.stat().st_mtime_ns if p.exists() else 0 for p in
                       (path.with_suffix(".contacts.npz"), path.with_suffix(".physics.npz"),
                        path.with_suffix(".support.npz")))
      edit_mtime = edit.stat().st_mtime_ns if edit.exists() else 0
      key = (dataset, clip, stat.st_mtime_ns, sidecars, edit_mtime)
      with self.lock:
        cached = self.clips.get(key)
        if cached is not None:
          self.clips.move_to_end(key)
          return cached
      blob = self._bake(dataset, clip, path)
      same = path.stat().st_mtime_ns == stat.st_mtime_ns and sidecars == tuple(
        p.stat().st_mtime_ns if p.exists() else 0 for p in
        (path.with_suffix(".contacts.npz"), path.with_suffix(".physics.npz"),
         path.with_suffix(".support.npz"))) and (
        edit.stat().st_mtime_ns if edit.exists() else 0) == edit_mtime
      if not same:
        continue
      with self.lock:
        self.clips[key] = blob
        self.clips.move_to_end(key)
        while len(self.clips) > MAX_BAKED_CLIPS:
          self.clips.popitem(last=False)
      return blob
    return blob

  def _bake(self, dataset: str, clip: str, path: Path) -> bytes:
    started = time.perf_counter()
    with np.load(path, allow_pickle=False) as arrays:
      names = [str(n) for n in arrays["joint_names"]]
      columns = [names.index(n) for n in self.joint_names]
      joint_pos = arrays["joint_pos"][:, columns].copy()
      base_pos = arrays["base_pos_w"].copy()
      base_quat = arrays["base_quat_w"].copy()
      fps = float(arrays["framerate"])
    length = len(base_pos)
    if joint_pos.shape != (length, len(self.joint_names)) or base_quat.shape != (length, 4) or length == 0:
      raise ValueError(f"Invalid motion shapes in {path}")
    if fps <= 0 or not np.isfinite(base_pos).all() or not np.isfinite(base_quat).all() or not np.isfinite(joint_pos).all():
      raise ValueError(f"Invalid motion values in {path}")
    base_quat /= np.linalg.norm(base_quat, axis=1, keepdims=True)
    poses = np.empty((length, len(self.body_names), 7), np.float32)
    with self.bake_lock:
      for frame in range(length):
        self.data.qpos[self.free_adr:self.free_adr + 3] = base_pos[frame]
        self.data.qpos[self.free_adr + 3:self.free_adr + 7] = base_quat[frame]
        self.data.qpos[self.joint_adr] = joint_pos[frame]
        mujoco.mj_kinematics(self.model, self.data)
        poses[frame, :, :3] = self.data.xpos[1:]
        poses[frame, :, 3:] = self.data.xquat[1:]
    # Keep quaternion hemispheres continuous so the client can slerp.
    for frame in range(1, length):
      flip = np.einsum("bc,bc->b", poses[frame, :, 3:], poses[frame - 1, :, 3:]) < 0
      poses[frame, flip, 3:] *= -1

    arrays_out = {"poses": poses}
    tracks: dict[str, dict] = {}
    short_to_link = {short: link for link, short in base.CONTACT_LINKS.items()}
    for key, sidecar in (("reference", path.with_suffix(".contacts.npz")),
                         ("physics", path.with_suffix(".physics.npz"))):
      if not sidecar.exists():
        continue
      track = base.ContactTrack.load(sidecar, physics=key == "physics")
      edit_info: dict = {}
      pos = track.pos.copy()
      if key == "reference":
        edited, edit_info, edit_pos = load_contact_edit(
          contact_edit_path(dataset, clip), track.flag.shape, track.names)
        if edited is not None:
          track.flag = edited
          if edit_pos is not None:
            # Use the positions captured by the editor wherever it has one; a
            # finite zero means "not supplied" (the labeler stores zeros off
            # contact), so leave those to the FK fallback below.
            supplied = (edited & np.isfinite(edit_pos).all(axis=2)
                        & (np.abs(edit_pos).sum(axis=2) > 0))
            pos[supplied] = edit_pos[supplied]
          track.next_frame = base.next_onsets(edited)
      # A hand-added contact can sit on a frame whose sidecar position was never
      # defined. The sidecars encode that as a *finite* zero, so test both.
      zero = np.abs(np.nan_to_num(pos, nan=np.inf, posinf=np.inf, neginf=np.inf)).sum(axis=2) == 0
      missing = track.flag & (~np.isfinite(pos).all(axis=2) | zero)
      if missing.any():
        for i, short in enumerate(track.names):
          link = short_to_link.get(short)
          body = self.contact_body_index.get(link)
          if body is None:
            continue
          rows = missing[:, i]
          pos[rows, i] = poses[rows, body, :3]
      array = np.nan_to_num(pos, nan=0.0).astype(np.float32)
      arrays_out[f"flag_{key}"] = (track.flag & np.isfinite(pos).all(axis=2)).astype(np.uint8)
      arrays_out[f"pos_{key}"] = array
      arrays_out[f"times_{key}"] = track.times.astype(np.float32)
      arrays_out[f"next_{key}"] = np.minimum(track.next_frame, len(track.flag)).astype(np.int32)
      tracks[key] = {"names": track.names, "source": track.source,
                     "provenance": track.provenance, "frames": int(len(track.flag)),
                     "edit": edit_info}

    # Static-support analysis of the frames (``support_analysis.py``): the contact labels only cover
    # nine bodies, this sidecar also says when the pose is not held up by anything ("hovering").
    support_info = None
    support_path = path.with_suffix(".support.npz")
    if support_path.exists():
      sup = np.load(support_path, allow_pickle=True)
      cls = sup["cls"].astype(np.uint8)
      if len(cls) == length:
        classes = [str(x) for x in sup["class_names"]]
        arrays_out["support_cls"] = cls
        arrays_out["support_com_out"] = sup["com_out"].astype(np.float32)
        arrays_out["support_zmin"] = sup["zmin"].astype(np.float32)
        if "core_z" in sup:
          arrays_out["support_core_z"] = sup["core_z"].astype(np.float32)
        arrays_out["support_touch"] = sup["touch_mask"].astype(np.uint8)
        support_info = {
          "classes": classes,
          "counts": {name: int((cls == i).sum()) for i, name in enumerate(classes)},
          "bodies": [str(x) for x in sup["body_names"]],
          "params": json.loads(str(sup["params"])),
        }

    terrain_id = int(self.store.metadata(dataset)["entries"][clip]["terrain_id"])
    terrain_path = self.store.metadata(dataset)["terrains"][terrain_id]
    header = {
      "dataset": dataset,
      "clip": clip,
      "group": base.ReferenceViewer.group(self.store.metadata(dataset)["entries"][clip]),
      "fps": fps,
      "frames": length,
      "bodies": self.body_names,
      "contact_body_index": self.contact_body_index,
      "primary": "reference" if "reference" in tracks else ("physics" if "physics" in tracks else None),
      "tracks": tracks,
      "support": support_info,
      "terrain": {
        "dataset": dataset,
        "id": terrain_id,
        "name": terrain_path.name,
        "url": f"/api/terrain?dataset={dataset}&id={terrain_id}&v={terrain_path.stat().st_mtime_ns}",
      },
      "bake_ms": round((time.perf_counter() - started) * 1000, 1),
    }
    return pack_payload(header, arrays_out)


class Player:
  def __init__(self, dataset: Path, extras: list[Path], port: int,
               kimodo: tuple[str, int] = (KIMODO_HOST, KIMODO_PORT)):
    paths = [dataset.resolve(), *[path.resolve() for path in extras]]
    self.store = Store(paths)
    self.model = base.robot_model()
    self.baker = Baker(self.store, self.model)
    self.default = dataset.resolve().name
    self.port = port
    self.kimodo = kimodo
    self.server = ThreadingHTTPServer(("0.0.0.0", port), self._handler())
    self.server.daemon_threads = True

  def _handler(self):
    player = self

    class Handler(BaseHTTPRequestHandler):
      protocol_version = "HTTP/1.1"

      def log_message(self, fmt: str, *args) -> None:
        pass

      # ---------------------------------------------------------------- kimodo
      def _kimodo_upstream_path(self, prefix: str = KIMODO_PREFIX) -> str:
        url = urlparse(self.path)
        path = url.path[len(prefix):] or "/"
        return path + (f"?{url.query}" if url.query else "")

      def _kimodo_unreachable(self, exc: OSError) -> None:
        print(f"[acp-player] kimodo proxy: {type(exc).__name__}: {exc}", flush=True)
        self._json({"error": f"kimodo demo unreachable: {exc}"}, 502)

      def _proxy_kimodo_http(self, method: str, prefix: str = KIMODO_PREFIX,
                             target: tuple[str, int] | None = None) -> None:
        """Forward one request/response verbatim to the loopback demo server (or ``target`` under ``prefix``)."""
        body = b""
        if method in ("POST", "PUT", "PATCH"):
          body = self.rfile.read(int(self.headers.get("Content-Length", 0) or 0))
        host, port = player.kimodo if target is None else target
        try:
          conn = socket.create_connection((host, port), timeout=15)
        except OSError as exc:
          self._kimodo_unreachable(exc)
          return
        try:
          head = [f"{method} {self._kimodo_upstream_path(prefix)} HTTP/1.1"]
          for key, value in self.headers.items():
            if key.lower() in ("host", "connection", "keep-alive", "proxy-connection"):
              continue
            head.append(f"{key}: {value}")
          head.append(f"Host: {host}:{port}")
          head.append("Connection: close")
          conn.sendall(("\r\n".join(head) + "\r\n\r\n").encode("latin-1"))
          if body:
            conn.sendall(body)
          conn.settimeout(None)
          while True:
            chunk = conn.recv(65536)
            if not chunk:
              break
            self.wfile.write(chunk)
          self.wfile.flush()
        except OSError as exc:
          self._kimodo_unreachable(exc)
        finally:
          conn.close()
          self.close_connection = True

      def _proxy_kimodo_websocket(self, prefix: str = KIMODO_PREFIX, target: tuple[str, int] | None = None) -> None:
        """Tunnel the viser websocket (raw TCP after the upgrade handshake)."""
        host, port = player.kimodo if target is None else target
        try:
          conn = socket.create_connection((host, port), timeout=15)
        except OSError as exc:
          self._kimodo_unreachable(exc)
          return
        try:
          head = [f"GET {self._kimodo_upstream_path(prefix)} HTTP/1.1"]
          for key, value in self.headers.items():
            if key.lower() in ("host", "proxy-connection"):
              continue
            head.append(f"{key}: {value}")
          head.append(f"Host: {host}:{port}")
          conn.sendall(("\r\n".join(head) + "\r\n\r\n").encode("latin-1"))
          conn.settimeout(None)
          self.connection.settimeout(None)
        except OSError as exc:
          self._kimodo_unreachable(exc)
          conn.close()
          return

        closed = threading.Event()

        def pump_to_client() -> None:
          try:
            while not closed.is_set():
              data = conn.recv(65536)
              if not data:
                break
              self.wfile.write(data)
              self.wfile.flush()
          except OSError:
            pass
          finally:
            closed.set()
            try:
              self.connection.shutdown(socket.SHUT_RDWR)
            except OSError:
              pass

        thread = threading.Thread(target=pump_to_client, daemon=True)
        thread.start()
        try:
          while not closed.is_set():
            data = self.rfile.read1(65536)
            if not data:
              break
            conn.sendall(data)
        except OSError:
          pass
        finally:
          closed.set()
          thread.join(timeout=5)
          conn.close()
          self.close_connection = True

      def _send(self, body: bytes, content_type: str, *, status: int = 200,
                cache: str = "no-store", compress: bool = False) -> None:
        headers = {"Content-Type": content_type, "Cache-Control": cache}
        if compress and "gzip" in self.headers.get("Accept-Encoding", "") and len(body) > 1024:
          body = gzip.compress(body, 5)
          headers["Content-Encoding"] = "gzip"
        self.send_response(status)
        for key, value in headers.items():
          self.send_header(key, value)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.command != "HEAD":
          self.wfile.write(body)

      def _json(self, payload: dict, status: int = 200) -> None:
        self._send(json.dumps(payload, allow_nan=False).encode(), "application/json", status=status)

      def _contact_edit(self) -> None:
        """Save / clear the hand-fixed contact flags of one clip."""
        length = int(self.headers.get("Content-Length", 0) or 0)
        if length > 64 * 1024 * 1024:
          self._json({"error": "payload too large"}, 413)
          return
        try:
          payload = json.loads(self.rfile.read(length) or b"{}")
          dataset = str(payload["dataset"])
          clip = str(payload["clip"])
          if dataset not in player.store.paths or clip not in player.store.metadata(dataset)["entries"]:
            raise KeyError(f"{dataset}/{clip}")
          sidecar = (player.store.paths[dataset] / clip).with_suffix(".contacts.npz")
          if not sidecar.exists():
            raise ValueError(f"{clip} has no contact sidecar")
          with np.load(sidecar, allow_pickle=False) as arrays:
            names = [str(name) for name in arrays["body_names"]]
            base_flag = arrays["contact_flag"].astype(bool)
            framerate = float(arrays["framerate"])
          target = contact_edit_path(dataset, clip)
          if payload.get("revert"):
            existed = target.exists()
            target.unlink(missing_ok=True)
            prune_contact_edit_dirs(target)
            print(f"[acp-player] contact edit clear {dataset}/{clip} "
                  f"({'removed' if existed else 'nothing saved'})", flush=True)
            self._json({"edited": False, "changed": 0, "saved_at": "", "unchanged": not existed})
            return
          raw = base64.b64decode(str(payload["flags"]), validate=True)
          flag = np.frombuffer(raw, dtype=np.uint8)
          if flag.size != base_flag.size:
            raise ValueError(f"contact flags have {flag.size} cells, expected {base_flag.size}")
          flag = flag.reshape(base_flag.shape).astype(bool)
          positions = None
          if payload.get("positions") is not None:
            praw = base64.b64decode(str(payload["positions"]), validate=True)
            positions = np.frombuffer(praw, dtype="<f4")
            if positions.size != base_flag.size * 3:
              raise ValueError(f"contact positions have {positions.size} values, "
                               f"expected {base_flag.size * 3}")
            positions = positions.reshape(base_flag.shape + (3,)).copy()
          changed = int((flag != base_flag).any(axis=1).sum())
          if changed == 0:
            target.unlink(missing_ok=True)
            prune_contact_edit_dirs(target)
            self._json({"edited": False, "changed": 0, "saved_at": "", "unchanged": True})
            return
          target.parent.mkdir(parents=True, exist_ok=True)
          saved_at = time.strftime("%Y-%m-%dT%H:%M:%S")
          tmp = target.with_name(f".{target.name}.{os.getpid()}.{threading.get_ident()}.tmp")
          fields = dict(contact_flag=flag.astype(np.uint8),
                        body_names=np.array(names), framerate=np.float64(framerate),
                        saved_at=np.array(saved_at), changed=np.int64(changed))
          if positions is not None:
            fields["contact_pos_w"] = positions.astype(np.float32)
          with open(tmp, "wb") as handle:
            np.savez(handle, **fields)
            handle.flush()
            os.fsync(handle.fileno())
          os.replace(tmp, target)
          print(f"[acp-player] contact edit {dataset}/{clip}: {changed} frames -> {target}",
                flush=True)
          self._json({"edited": True, "changed": changed, "saved_at": saved_at})
        except (KeyError, ValueError, OSError, json.JSONDecodeError, binascii.Error) as exc:
          self._json({"error": f"{type(exc).__name__}: {exc}"}, 400)

      def _file(self, path: Path, content_type: str) -> None:
        stat = path.stat()
        self._send(path.read_bytes(), content_type, cache="no-cache",
                   compress=stat.st_size > 64 * 1024)

      def do_HEAD(self) -> None:
        self.do_GET()

      def do_GET(self) -> None:
        url = urlparse(self.path)
        query = {key: values[-1] for key, values in parse_qs(url.query).items()}
        try:
          if url.path == KIMODO_PREFIX and self.headers.get("Upgrade", "").lower() == "websocket":
            self._proxy_kimodo_websocket()
          elif url.path == KIMODO_PREFIX:
            self.send_response(301)
            self.send_header("Location", f"{KIMODO_PREFIX}/")
            self.send_header("Content-Length", "0")
            self.end_headers()
          elif url.path.startswith(f"{KIMODO_PREFIX}/"):
            if self.headers.get("Upgrade", "").lower() == "websocket":
              self._proxy_kimodo_websocket()
            else:
              self._proxy_kimodo_http("HEAD" if self.command == "HEAD" else "GET")
          elif url.path == POLICY_PREFIX and self.headers.get("Upgrade", "").lower() == "websocket":
            self._proxy_kimodo_websocket(POLICY_PREFIX, POLICY_TARGET)
          elif url.path == POLICY_PREFIX:
            self.send_response(301)
            self.send_header("Location", f"{POLICY_PREFIX}/")
            self.send_header("Content-Length", "0")
            self.end_headers()
          elif url.path.startswith(f"{POLICY_PREFIX}/"):
            if self.headers.get("Upgrade", "").lower() == "websocket":
              self._proxy_kimodo_websocket(POLICY_PREFIX, POLICY_TARGET)
            else:
              self._proxy_kimodo_http("HEAD" if self.command == "HEAD" else "GET", POLICY_PREFIX, POLICY_TARGET)
          elif url.path == TRACKER_PREFIX and self.headers.get("Upgrade", "").lower() == "websocket":
            self._proxy_kimodo_websocket(TRACKER_PREFIX, TRACKER_TARGET)
          elif url.path == TRACKER_PREFIX:
            self.send_response(301)
            self.send_header("Location", f"{TRACKER_PREFIX}/")
            self.send_header("Content-Length", "0")
            self.end_headers()
          elif url.path.startswith(f"{TRACKER_PREFIX}/"):
            if self.headers.get("Upgrade", "").lower() == "websocket":
              self._proxy_kimodo_websocket(TRACKER_PREFIX, TRACKER_TARGET)
            else:
              self._proxy_kimodo_http("HEAD" if self.command == "HEAD" else "GET", TRACKER_PREFIX, TRACKER_TARGET)
          elif url.path == WORKHORSE_PREFIX:
            self.send_response(301)
            self.send_header("Location", f"{WORKHORSE_PREFIX}/")
            self.send_header("Content-Length", "0")
            self.end_headers()
          elif url.path.startswith(f"{WORKHORSE_PREFIX}/"):
            self._proxy_kimodo_http("HEAD" if self.command == "HEAD" else "GET", WORKHORSE_PREFIX, WORKHORSE_TARGET)
          elif url.path in ("/", "/index.html"):
            self._file(STATIC / "index.html", "text/html; charset=utf-8")
          elif url.path.startswith("/static/"):
            relative = url.path[len("/static/"):]
            if ".." in relative:
              self._json({"error": "bad path"}, 400)
              return
            content_type = mimetypes.guess_type(relative)[0] or "application/octet-stream"
            if relative.endswith(".js"):
              content_type = "text/javascript; charset=utf-8"
            self._file(STATIC / relative, content_type)
          elif url.path == "/api/config":
            self._send(json.dumps(player.store.config(player.default)).encode(),
                       "application/json", compress=True)
          elif url.path == "/api/curation":
            dataset = query["dataset"]
            self._json({"curation": player.store.curation(dataset)})
          elif url.path == "/api/clip":
            started = time.perf_counter()
            blob = player.baker.clip(query["dataset"], query["clip"])
            print(f"[acp-player] clip {query['dataset']}/{query['clip']} "
                  f"{len(blob) / 1024:.0f} KiB {1000 * (time.perf_counter() - started):.0f} ms",
                  flush=True)
            self._send(blob, "application/octet-stream", compress=True)
          elif url.path == "/api/terrain":
            blob, etag = player.baker.terrain(query["dataset"], int(query["id"]))
            self._send(blob, "application/octet-stream", cache="no-cache", compress=True)
          elif url.path == "/api/robot":
            self._send(player.baker.robot, "application/octet-stream",
                       cache="private, max-age=3600", compress=True)
          else:
            self._json({"error": "not found"}, 404)
        except BrokenPipeError:
          pass
        except (KeyError, ValueError, OSError, yaml.YAMLError, zipfile.BadZipFile) as exc:
          print(f"[acp-player] {url.path}: {type(exc).__name__}: {exc}", flush=True)
          self._json({"error": f"{type(exc).__name__}: {exc}"}, 400)
        except Exception as exc:  # noqa: BLE001 - report, never kill the server
          traceback.print_exc()
          self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)

      def do_POST(self) -> None:
        url = urlparse(self.path)
        if url.path.startswith(f"{KIMODO_PREFIX}/"):
          self._proxy_kimodo_http("POST")
          return
        if url.path.startswith(f"{POLICY_PREFIX}/"):
          self._proxy_kimodo_http("POST", POLICY_PREFIX, POLICY_TARGET)
          return
        if url.path.startswith(f"{TRACKER_PREFIX}/"):
          self._proxy_kimodo_http("POST", TRACKER_PREFIX, TRACKER_TARGET)
          return
        if url.path == "/api/contact_edit":
          self._contact_edit()
          return
        if url.path != "/api/review":
          self._json({"error": "not found"}, 404)
          return
        length = int(self.headers.get("Content-Length", 0))
        try:
          payload = json.loads(self.rfile.read(length))
          dataset = str(payload["dataset"])
          clip = str(payload["clip"])
          if dataset not in player.store.paths or clip not in player.store.metadata(dataset)["entries"]:
            raise KeyError(f"{dataset}/{clip}")
          status = payload.get("status")
          if status not in (None, "keep", "reject", "edited"):
            raise ValueError(f"bad status {status!r}")
          note = str(payload.get("note", ""))
          records = player.store.review(dataset, clip, status, note, bool(payload.get("note_only")))
          print(f"[acp-player] review {dataset}/{clip} -> {status or 'clear'} "
                f"({len(note)} chars)", flush=True)
          self._json({"curation": records})
        except (KeyError, ValueError, OSError, yaml.YAMLError, json.JSONDecodeError) as exc:
          self._json({"error": f"{type(exc).__name__}: {exc}"}, 400)

    return Handler

  def run(self) -> None:
    print(f"[acp-player] {len(self.store.paths)} datasets: {', '.join(self.store.paths)}", flush=True)
    print(f"[acp-player] robot {len(self.baker.robot) / 1024:.0f} KiB, "
          f"{len(self.baker.body_names)} bodies", flush=True)
    base.announce(self.port)
    self.server.serve_forever()

  def close(self) -> None:
    self.server.shutdown()
    self.server.server_close()


def self_test(player: Player) -> None:
  """Bake every clip of every listed dataset and report payload sizes."""
  total = ok = 0
  failures: list[str] = []
  started = time.perf_counter()
  for dataset in player.store.paths:
    entries = player.store.metadata(dataset)["entries"]
    print(f"== {dataset}: {len(entries)} clips", flush=True)
    for clip in entries:
      total += 1
      try:
        begin = time.perf_counter()
        blob = player.baker.clip(dataset, clip)
        elapsed = 1000 * (time.perf_counter() - begin)
        header_len = struct.unpack("<I", blob[4:8])[0]
        header = json.loads(blob[8:8 + header_len])
        assert header["frames"] > 0 and len(header["bodies"]) == len(player.baker.body_names)
        assert "poses" in header["arrays"]
        print(f"  {clip:64s} {header['frames']:4d} fr {len(blob) / 1024:7.1f} KiB {elapsed:6.1f} ms",
              flush=True)
        ok += 1
      except Exception as exc:  # noqa: BLE001 - self-test reports every skip
        failures.append(f"{dataset}/{clip}: {type(exc).__name__}: {exc}")
        print(f"  FAIL {dataset}/{clip}: {type(exc).__name__}: {exc}", flush=True)
  elapsed = time.perf_counter() - started
  print(f"[SELF-TEST] {ok}/{total} clips baked in {elapsed:.1f}s "
        f"({elapsed / max(total, 1) * 1000:.0f} ms/clip), {len(failures)} failures", flush=True)
  if failures:
    for line in failures[:20]:
      print("  " + line, flush=True)
    raise SystemExit(1)


def apply_edits(dataset: Path, clip: str | None, dry_run: bool,
                edits_from: str | None = None) -> None:
  """Write saved contact edits into a dataset's ``.contacts.npz`` files.

  The viewer keeps fixes in ``CONTACT_EDIT_ROOT`` so generated datasets stay
  pristine. Point this at a training copy (e.g. ``v3_train3``) to make the
  hand-fixed flags part of that dataset; the edit layer itself is untouched.
  """
  dataset = dataset.resolve()
  if dataset == (DATASET_ROOT / "v3_final").resolve():
    raise SystemExit(
      f"{dataset} is generated by tools/build_final.sh and must stay pristine; "
      f"point --dataset at the training copy instead.")
  root = CONTACT_EDIT_ROOT / (edits_from or dataset.name)
  if not root.is_dir() and edits_from is None and (CONTACT_EDIT_ROOT / "v3_final").is_dir():
    # The viewer normally reviews the generated ``v3_final`` dataset, while the
    # fixes are applied to a training copy with a different directory name.
    root = CONTACT_EDIT_ROOT / "v3_final"
    print(f"[apply] no edits for {dataset.name}; using the {root.name} edit layer",
          flush=True)
  if not root.is_dir():
    raise SystemExit(f"no saved contact edits for {edits_from or dataset.name}: {root}")
  candidates = sorted(root.rglob("*.npz"))
  if clip is not None:
    candidates = [path for path in candidates if str(path.relative_to(root)) == clip]
  if not candidates:
    raise SystemExit(f"no saved contact edits matching {clip or '*'} under {root}")
  written = updated = 0
  for edit in candidates:
    relative = str(edit.relative_to(root))
    sidecar = (dataset / relative).with_suffix(".contacts.npz")
    if not sidecar.exists():
      print(f"  skip {relative}: no sidecar in {dataset}", flush=True)
      continue
    with np.load(edit, allow_pickle=False) as arrays:
      flag = arrays["contact_flag"].astype(bool)
      edit_names = [str(name) for name in arrays["body_names"]]
      saved_at = str(arrays["saved_at"]) if "saved_at" in arrays else ""
      edit_pos = arrays["contact_pos_w"].astype(np.float32) if "contact_pos_w" in arrays else None
    with np.load(sidecar, allow_pickle=False) as arrays:
      data = {key: arrays[key] for key in arrays.files}
    old = data["contact_flag"].astype(bool)
    names = [str(name) for name in data["body_names"]]
    if flag.shape != old.shape or edit_names != names:
      print(f"  skip {relative}: shape/body order differs", flush=True)
      continue
    pos = data["contact_pos_w"].copy() if "contact_pos_w" in data else None
    moved = None
    if edit_pos is not None and pos is not None and edit_pos.shape == pos.shape:
      supplied = flag & np.isfinite(edit_pos).all(axis=2) & (np.abs(edit_pos).sum(axis=2) > 0)
      # a contact point moved inside an existing segment changes no flag: treat it as a change too
      moved = supplied & (np.abs(pos - edit_pos).max(axis=2) > 1e-6)
      pos[supplied] = edit_pos[supplied]
    elif edit_pos is not None:
      print(f"  skip {relative}: position shape differs", flush=True)
      continue
    added = flag & ~old
    if added.any() and (pos is None or (np.abs(pos[added]).sum(axis=1) == 0).any()):
      # Old edits carry flags only; the trainer would see contacts at (0,0,0).
      print(f"  skip {relative}: added contacts without positions; re-save them "
            f"in the viewer before applying", flush=True)
      continue
    changed = int(
      ((flag != old) | moved).any(axis=1).sum() if moved is not None else (flag != old).any(axis=1).sum()
    )
    if changed == 0:
      print(f"  ok   {relative}: already up to date", flush=True)
      continue
    params: dict = {}
    if "params" in data:
      try:
        params = json.loads(str(data["params"].item()))
      except (ValueError, AttributeError):
        params = {}
    history = params.get("motionedit")
    if not isinstance(history, list):
      history = []
    history.append({"handfix": {"tool": "acp_player", "saved_at": saved_at,
                                "changed_frames": changed}})
    params["motionedit"] = history
    data["contact_flag"] = flag
    if pos is not None:
      data["contact_pos_w"] = pos
      if "onset_pos_w" in data:
        # derived field: the position at the first frame of every contact segment
        onset = np.zeros_like(pos)
        for b in range(flag.shape[1]):
          steps = np.diff(np.concatenate(([0], flag[:, b].astype(np.int8), [0])))
          for start, end in zip(np.nonzero(steps == 1)[0], np.nonzero(steps == -1)[0]):
            onset[start:end, b] = pos[start, b]
        data["onset_pos_w"] = onset
    data["params"] = np.array(json.dumps(params, ensure_ascii=False))
    if dry_run:
      print(f"  would update {relative}: {changed} frames", flush=True)
      continue
    tmp = sidecar.with_name(f".{sidecar.name}.{os.getpid()}.tmp")
    with open(tmp, "wb") as handle:
      np.savez(handle, **data)
    os.replace(tmp, sidecar)
    written += 1
    updated += changed
    print(f"  write {relative}: {changed} frames", flush=True)
  verb = "would write" if dry_run else "wrote"
  print(f"[apply] {verb} {written} sidecars, {updated} changed frames", flush=True)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  subparsers = parser.add_subparsers(dest="mode", required=True)
  refs = subparsers.add_parser("refs", help="Browse TerrainMotion reference clips")
  refs.add_argument("--dataset", type=Path, required=True)
  refs.add_argument("--port", type=int, default=8188)
  refs.add_argument("--only", action="store_true", help="List only --dataset in the dataset dropdown")
  refs.add_argument("--also", type=Path, action="append", default=[], help="Also list this dataset (repeatable)")
  refs.add_argument("--self-test", action="store_true", help="Bake every clip, report sizes, then exit")
  apply = subparsers.add_parser("apply", help="Write saved contact edits into a dataset copy")
  apply.add_argument("--dataset", type=Path, required=True)
  apply.add_argument("--clip", default=None, help="Only this clip path (e.g. beam/foo.npz)")
  apply.add_argument("--dry-run", action="store_true")
  apply.add_argument("--edits-from", default=None,
                     help="Edit-layer name to read (default: the target dataset name, "
                          "then v3_final when that has no edits)")
  args = parser.parse_args()
  if args.mode == "apply":
    apply_edits(args.dataset, args.clip, args.dry_run, args.edits_from)
    return
  if args.mode != "refs":
    raise SystemExit("unknown mode")
  player = Player(args.dataset, args.also, args.port)
  signal.signal(signal.SIGTERM, lambda *_: threading.Thread(target=player.close, daemon=True).start())
  signal.signal(signal.SIGINT, lambda *_: threading.Thread(target=player.close, daemon=True).start())
  try:
    if args.self_test:
      self_test(player)
    else:
      player.run()
  finally:
    player.server.server_close()


if __name__ == "__main__":
  main()
