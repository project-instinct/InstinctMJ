"""Remove terrain penetrations from retargeted ACP reference motions with a minimal whole-trajectory IK.

    python -m instinct_mj.scripts.acp.refine_motion --dataset ~/Datasets/acp/v3_train8 --out /tmp/refined \
        [--clips REGEX | --clips-file penetration_clips.csv] [--margin 0.002] [--joint-margin 0.026] [--iters 500] [--no-touch]
    python -m instinct_mj.scripts.acp.refine_motion --dataset ~/Datasets/acp/v3_train8 --check /tmp/refined   # acceptance report only

Port of ``supermimic/acp/refine_motion.py`` of the upstream reproduction (Quangvuisme/Acp_supermimic_vibed): per clip the root pose
(translation + rotation vector, composed onto the source) and the joint angles are optimized with Adam, the loss being
``track + w_clear * clearance + w_smooth * smoothness`` (+ ``w_keep`` and ``w_foot``, below). ``track`` is the squared displacement of
all body frames from the source (+ a small ridge on the deltas), ``smoothness`` the first and second differences of the deltas
(upstream: 1 + 10 weighting), the result is clamped to the joint ranges (``--joint-margin`` keeps a margin to them: the clamp then stops
short, so a pipeline that later pushes the joints a degree or two inside the limits cannot reintroduce a penetration by that push).
Clips that already clear the terrain are copied byte for byte;
in the others the frames that are away from any penetration move by (almost) nothing, because the gradient there is zero.

Differences to upstream (which has a 9-body sphere approximation and analytic boxes):
  * Terrain: any STL (boxes, ladder rungs, stairs, the floor slab). The signed distance of a point is a warp BVH query on the GPU
    (closest point; the sign is that of the outward normal of the closest triangle when the
    closest point is inside it, and warp's generalized winding number sign on edges / vertices (the pseudo-normal one is wrong on faceted rungs): the face normal of a tie flips the sign of points that
    touch the plane of a box top just beyond its edge (a 0.7 m false penetration). STL winding must be outward).
    The gradient w.r.t. the point is the first order one, ``sign * (p - closest) / |p - closest|`` (the face normal when p is on the
    surface), requeried at every iteration (``SignedDistance`` autograd function). Several clips with different terrains are optimized in
    one batch (one mesh id per clip).
  * Robot: the collision geometry of the task's MJCF (``get_acp_g1_spec()``, the authors' meshes, or ``--simplified-collision``) for
    ALL links, not only the nine contact bodies. Check points are the convex hull vertices of every mesh geom plus a ~``--spacing``
    grid on the hull faces (so a flat palm edge cannot slip between two vertices and over a thin rung), the sampled surfaces of
    capsules / spheres / boxes / cylinders. FK is a differentiable torch re-implementation of the MJCF kinematic tree (checked
    against ``mj_kinematics``). The six wrist joints of the files do not exist in the model (welded) and stay as in the source.
  * Constraints: the hard requirement is "no penetration": signed distance of every check point >= ``--margin`` (default 2 mm; squared
    hinge ``w_clear``; the optimization works with ``--slack`` more because it sees a thinned point set, the verification in the report
    uses every sampled point). Two modes, tried in this order for every clip that penetrates:
      1. ``touch`` (upstream's touch term, ``--touch``, default on): a body in contact in the source sidecar and within d_on of the terrain
         (``label_contacts`` D_ON: feet 3 cm, others 1.5 cm) is pulled to ``--touch-target`` (1 mm) from the surface (``w_touch``);
      2. ``notouch`` (``--no-touch``, also the per-clip fallback): no touch term, a hand 1 mm to a few mm off the object is fine; contact
         bodies must only not be pushed out of contact: their min distance (label_contacts' vertex points) is penalized above
         ``d_on - 4 mm`` (``w_keep``) and never below what the source had.
    In both, a foot in contact does not slide (horizontal position of its points held to the source, ``w_foot``).
    ``--touch-range`` widens the frames the touch term applies to (default: the labeler's d_on), ``--touch-labels core``
    restricts it to the core of each contact segment (real contact; the hysteresis tail is left at its source height),
    ``--track-scale`` selectively relaxes the body tracking of e.g. the arm bodies so the touch term may straighten them,
    ``--need-contact-gap`` selects clips for refinement by the labelled contact gap (not only by penetration) - all four
    default to the previous behaviour.
    Per clip the touch result is used if it passes the acceptance below, else the notouch one (also recorded when it fails too):
    ``<out>/refine_report.csv`` (+ ``refine_report.json``) has the mode and the reason per clip. Clips that do not penetrate are copied.
    Acceptance (``ACCEPT``, ``failures``): no point deeper than 1 mm inside the terrain; joint change max <= 30 deg / mean <= 3 deg; root
    <= 8 cm; any body frame <= 20 cm (mean <= 3 cm; upstream's PHP clips need up to 9 cm hand and 4 cm foot corrections); a contact foot
    drifts <= 3 cm from the source and its slip grows <= 3 cm; third difference of joints / root <= 1.5 x the source's + 0.01 rad /
    3 mm (no spikes, no oscillation); joint-limit frames grow <= 2 %; self-collision frames (> 5 mm, MuJoCo) grow <= 3 %; the share of
    labelled contact frames within d_on of the terrain drops <= 5 points.
  * Data: the TerrainMotion dataset (``metadata.yaml`` -> npz + ``.contacts.npz`` sidecars), output in the same layout and fields
    (``joint_pos``, ``base_pos_w``, ``base_quat_w`` replaced; framerate, joint names, everything else copied), never over the source.
    The source sidecars ``<stem>.contacts.npz`` are copied unchanged (the labels stay those of the unrefined data, no relabelling);
    ``--check`` reports whether the refined motion still agrees with them (distance of each labelled body to the terrain within
    0..1.5 cm / d_on, and its distance to the labelled contact point ``contact_pos_w``).
    Before training on the result, refresh those geometric fields against the refined poses while preserving the source sequence::

      python -m instinct_mj.scripts.acp.refresh_contacts --dataset REFINED --reference-dataset SOURCE \
        --out TRAINING_SNAPSHOT --simplified-collision --workers 3

    Use the collision-model option of the target training run. Contacts that cannot reach their original support are reported for
    further motion repair; refreshing contact geometry does not make those poses dynamically feasible.

Memory: batches are cut so that ``clips * frames * points <= --budget`` (default 6e6 points, ~3 GB of GPU memory with autograd).
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import re
import shutil
import time

import mujoco
import numpy as np
import torch
import trimesh
import warp as wp
import yaml
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from instinct_mj.assets.acp_g1 import get_acp_g1_spec
from instinct_mj.rl.acp.planner import CONTACT_BODIES, CONTACT_LINKS
from instinct_mj.rl.supermimic.rotations import axis_angle_to_mat, mat_to_quat, quat_to_mat, so3_exp
from instinct_mj.scripts.acp.label_contacts import D_ON, D_ON_BODY, FOOT_BODIES, _sample_geom

wp.config.quiet = True

FEET_B = np.array([b in FOOT_BODIES for b in CONTACT_BODIES])


def labeler_moving(speed: np.ndarray, params: dict) -> np.ndarray:
    """The labeler's own ``moving`` (label_contacts.contact_segments): speed >= v_max, runs shorter than ``move_frames``
    cleared, first / last frame cleared."""
    moving = np.asarray(speed) >= params["v_max"]
    for b in range(moving.shape[1]):
        col = moving[:, b]
        edges = np.diff(np.r_[0, col.astype(np.int8), 0])
        for s, e in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            if e - s < params["move_frames"]:
                col[s:e] = False
        col[[0, -1]] = False
    return moving


def core_tail_masks(flag: np.ndarray, min_dist: np.ndarray, speed: np.ndarray, params: dict) -> tuple[np.ndarray, np.ndarray]:
    """Split the labelled contact frames (T, B) into the ``core`` (real contact: entered below d_on and static, or a sliding
    touch - the labeler only keeps moving frames that touch below d_touch) and the ``tail`` (hysteresis: static with
    d_on <= d <= d_off). Distances / speeds are the sidecar's own, i.e. those of the ORIGINAL retarget the labels were made
    on; see /tmp/acp_touch11/core_tail.py (unit-tested: contact_segments reproduces the sidecar's flags exactly)."""
    d_on = np.where(FEET_B, params["d_on"], params["d_on_body"])
    moving = labeler_moving(speed, params)
    tail = flag & ~moving & (min_dist >= d_on)
    return flag & ~tail, tail

AXIS_STEP = 0.004  # m, spacing of the capsule axis samples of the optimization
MARGIN = 0.002  # m, required signed distance of every check point (ACP-unspecified: ours)
KEEP_SLACK = 0.004  # m, a body in contact is held within d_on - KEEP_SLACK (the labeler's d_on is 3 cm feet / 1.5 cm others)
SIGN_RANGE = 0.25  # m, farther from the terrain than this the closest feature is never decisive: points there count as outside
MAX_DIST = 1.0  # m, terrain queries beyond are "clear"
PEN_EPS = 0.001  # m, "penetrating" in the report
FEET = np.array([b in FOOT_BODIES for b in CONTACT_BODIES])
D_ON_BODIES = np.where(FEET, D_ON, D_ON_BODY)


# ---------------------------------------------------------------------------------------------------- terrain signed distance
class AnalyticTerrain:
    """Differentiable exact SDF of a primitive terrain (the geoms.yaml boxes / cylinders, the env collision truth for exterior
    points; validated against the geom-exact STLs to <= 0.04 mm). Used for the optimization loop of ``--analytic-terrain``:
    the warp BVH query on the same points is ~100x slower on long clips; the verification keeps the mesh path.

    ``sd`` is exact for points outside the union of geoms; inside, the min over geoms underestimates the depth to the union
    boundary near shared faces (only relaxes the interior of overlapping geoms, e.g. a slab under a step)."""

    def __init__(self, geoms: list[dict], device: str):
        self.dev = device
        quat = torch.tensor([g.get("quat", [1.0, 0.0, 0.0, 0.0]) for g in geoms], dtype=torch.float32, device=device)
        quat = quat / torch.linalg.vector_norm(quat, dim=-1, keepdim=True)
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        self.R = torch.stack([
            torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
            torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
            torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
        ], -2)  # (G, 3, 3) local->world; loc[n, g] = (p - pos_g) @ R_g
        self.pos = torch.tensor([g["pos"] for g in geoms], dtype=torch.float32, device=device)
        self.size = torch.tensor([list(g["size"]) + [0.0] * (3 - len(g["size"])) for g in geoms], dtype=torch.float32, device=device)
        self.box = torch.tensor([g["type"] == "box" for g in geoms], device=device)
        assert all(g["type"] in ("box", "cylinder") for g in geoms), "analytic terrain: only box / cylinder geoms"

    def sd_no_grad(self, pts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(N,3) -> (sd (N,), argmin geom idx (N,)), no graph: the exact min over the geoms, one geom at a time
        (no (N, G, *) intermediates - the batched form is memory-bound and much slower)."""
        with torch.no_grad():
            best_s = best_i = None
            for g in range(len(self.pos)):
                loc = torch.matmul(pts - self.pos[g], self.R[g])  # loc_k = sum_i v_i R[i,k] (row form, same as the batched path)
                q_abs = loc.abs()
                if bool(self.box[g]):
                    q = q_abs - self.size[g]
                    sd = torch.linalg.vector_norm(torch.clamp(q, min=0.0), dim=-1) + torch.clamp(q.max(-1).values, max=0.0)
                else:
                    d2 = torch.sqrt(loc[:, 0] ** 2 + loc[:, 1] ** 2) - self.size[g, 0]
                    dz = q_abs[:, 2] - self.size[g, 1]
                    dd = torch.stack([d2, dz], -1)
                    sd = torch.linalg.vector_norm(torch.clamp(dd, min=0.0), dim=-1) + torch.clamp(dd.max(-1).values, max=0.0)
                if best_s is None:
                    best_s, best_i = sd, torch.full_like(sd, g, dtype=torch.long)
                else:
                    upd = sd < best_s
                    best_s = torch.where(upd, sd, best_s)
                    best_i = torch.where(upd, torch.full_like(best_i, g), best_i)
            return best_s, best_i

    def grad_of(self, pts: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """(N,3) points with their winning geom indices -> (N,3) d(sd)/d(point), the exact SDF gradient."""
        pos, R, size, box = self.pos[idx], self.R[idx], self.size[idx], self.box[idx]  # (N,3),(N,3,3),(N,3),(N,)
        loc = torch.matmul((pts - pos).unsqueeze(1), R).squeeze(1)  # loc_k = sum_i v_i R[i,k] (same form as sd_no_grad)
        q_abs = loc.abs()
        d2 = torch.sqrt(loc[:, 0] ** 2 + loc[:, 1] ** 2) - size[:, 0]
        dz = q_abs[:, 2] - size[:, 1]
        q = torch.stack([torch.where(box, q_abs[:, 0] - size[:, 0], d2),
                         torch.where(box, q_abs[:, 1] - size[:, 1], dz),
                         torch.where(box, q_abs[:, 2] - size[:, 2], -1e9)], -1)  # (N,3)
        nrm = torch.linalg.vector_norm(torch.clamp(q, min=0.0), dim=-1).clamp(min=1e-12)
        outside = q.max(-1).values > 0
        w_out = torch.where(q > 0, q / nrm[:, None], torch.zeros_like(q))  # dsd/dq outside
        amax = q.argmax(-1)
        w_in = torch.zeros_like(q)
        w_in.scatter_(1, amax[:, None], 1.0)  # dsd/dq inside: the closest face
        dq = torch.where(outside[:, None], w_out, w_in)
        sign_box = torch.where(q_abs > 0, torch.sign(loc), torch.zeros_like(loc))  # dq/dloc (box form)
        xy = torch.sqrt(loc[:, 0] ** 2 + loc[:, 1] ** 2).clamp(min=1e-12)
        # cylinder: d(d2)/dloc = (loc_xy/|loc_xy|, 0), d(dz)/dloc = (0, 0, sign(loc_z)); dq = (d sd/d d2, d sd/d dz, 0)
        gl_cyl = torch.cat([dq[:, 0:1] * loc[:, 0:1] / xy[:, None],
                            dq[:, 0:1] * loc[:, 1:2] / xy[:, None],
                            dq[:, 1:2] * torch.sign(loc[:, 2:3])], dim=1)
        grad_loc = torch.where(box[:, None], dq * sign_box, gl_cyl)  # (N,3)
        return torch.matmul(grad_loc.unsqueeze(1), R.transpose(-1, -2)).squeeze(1)  # d loc_k / d v_i = R[i,k]


def _chunk_min_eager(p, pos, R, size, box):
    v = p[:, None, :] - pos[None]  # (N, G, 3)
    loc = torch.matmul(v.unsqueeze(2), R[None]).squeeze(2)
    q_abs = loc.abs()
    d2 = torch.sqrt(loc[..., 0] ** 2 + loc[..., 1] ** 2) - size[None, :, 0]
    dz = q_abs[..., 2] - size[None, :, 1]
    q = torch.stack([torch.where(box[None], q_abs[..., 0] - size[None, :, 0], d2),
                     torch.where(box[None], q_abs[..., 1] - size[None, :, 1], dz),
                     torch.where(box, q_abs[..., 2] - size[None, :, 2], -1e9 * d2)], -1)
    sd = torch.linalg.vector_norm(torch.clamp(q, min=0.0), dim=-1) + torch.clamp(q.max(-1).values, max=0.0)
    return sd.min(-1).values, sd.argmin(-1)


try:  # a fused kernel is ~3x faster on the memory-bound chain; fall back to eager when compile is unavailable
    _chunk_min = torch.compile(_chunk_min_eager, dynamic=False)
except Exception:  # pragma: no cover
    _chunk_min = _chunk_min_eager


class _AnalyticSD(torch.autograd.Function):
    """AnalyticTerrain SDF with the exact first-order gradient (argmin geom)."""

    @staticmethod
    def forward(ctx, pts: torch.Tensor, ter: AnalyticTerrain):
        sd, idx = ter.sd_no_grad(pts)
        ctx.save_for_backward(pts, idx)
        ctx.ter = ter
        return sd

    @staticmethod
    def backward(ctx, g):
        (pts, idx) = ctx.saved_tensors
        return g[:, None] * ctx.ter.grad_of(pts, idx), None


@wp.kernel
def _sdf_kernel(
    ids: wp.array(dtype=wp.uint64),
    pts: wp.array2d(dtype=wp.vec3),
    max_dist: float,
    sd: wp.array2d(dtype=wp.float32),
    grad: wp.array2d(dtype=wp.vec3),
):
    b, i = wp.tid()
    mid = ids[b]
    p = pts[b, i]
    face = int(0)
    u = float(0.0)
    v = float(0.0)
    if wp.mesh_query_point_no_sign(mid, p, max_dist, face, u, v):
        c = wp.mesh_eval_position(mid, face, u, v)
        m = wp.mesh_get(mid)
        a0 = m.points[m.indices[3 * face]]
        a1 = m.points[m.indices[3 * face + 1]]
        a2 = m.points[m.indices[3 * face + 2]]
        n = wp.normalize(wp.cross(a1 - a0, a2 - a0))
        d = p - c
        length = wp.length(d)
        s = float(1.0)
        if u > 1.0e-3 and v > 1.0e-3 and u + v < 1.0 - 1.0e-3:
            if wp.dot(d, n) < 0.0:  # the closest point is inside a triangle: its normal tells the side
                s = -1.0
        elif length < SIGN_RANGE:  # on an edge / vertex the face normal is ambiguous (a tie of two faces): winding number sign
            sgn = float(1.0)
            f2 = int(0)
            u2 = float(0.0)
            v2 = float(0.0)
            wp.mesh_query_point_sign_winding_number(mid, p, max_dist, sgn, f2, u2, v2, 2.0, 0.5)
            if sgn < 0.0:
                s = -1.0
        sd[b, i] = s * length
        if length > 1.0e-6:
            grad[b, i] = d * (s / length)
        else:
            grad[b, i] = n
    else:
        sd[b, i] = max_dist
        grad[b, i] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def _dist_kernel(ids: wp.array(dtype=wp.uint64), pts: wp.array2d(dtype=wp.vec3), max_dist: float, dist: wp.array2d(dtype=wp.float32)):
    b, i = wp.tid()
    face = int(0)
    u = float(0.0)
    v = float(0.0)
    p = pts[b, i]
    if wp.mesh_query_point_no_sign(ids[b], p, max_dist, face, u, v):
        dist[b, i] = wp.length(p - wp.mesh_eval_position(ids[b], face, u, v))
    else:
        dist[b, i] = max_dist


def unsigned_distance(pts: torch.Tensor, ids: wp.array) -> torch.Tensor:
    """(B, M, 3) -> (B, M) distance to the terrain of clip b (no sign, no gradient: the labeler's own metric)."""
    pts = pts.detach().contiguous().float()
    out = torch.empty(pts.shape[:2], device=pts.device)
    with wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(pts.device))):
        wp.launch(_dist_kernel, dim=pts.shape[:2], inputs=[ids, wp.from_torch(pts, dtype=wp.vec3), MAX_DIST, wp.from_torch(out, dtype=wp.float32)],
                  device=wp.device_from_torch(pts.device))  # fmt: skip
    return out


class SignedDistance(torch.autograd.Function):
    """(B, M, 3) points -> (B, M) signed distance to the terrain of clip b (positive outside); first order gradient."""

    @staticmethod
    def forward(ctx, pts: torch.Tensor, ids: wp.array):
        dev = wp.device_from_torch(pts.device)
        pts = pts.detach().contiguous().float()
        sd = torch.empty(pts.shape[:2], device=pts.device)
        grad = torch.empty_like(pts)
        with wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(pts.device))):  # same stream as the torch ops
            wp.launch(_sdf_kernel, dim=pts.shape[:2], inputs=[ids, wp.from_torch(pts, dtype=wp.vec3), MAX_DIST,
                      wp.from_torch(sd, dtype=wp.float32), wp.from_torch(grad, dtype=wp.vec3)], device=dev)  # fmt: skip
        ctx.save_for_backward(grad)
        return sd

    @staticmethod
    def backward(ctx, g):
        (grad,) = ctx.saved_tensors
        return g[..., None] * grad, None


class Terrains:
    """GPU warp meshes of the STL files (kept alive while in use)."""

    def __init__(self, device: str):
        self.dev, self.meshes = device, {}

    def mesh_id(self, path: str) -> int:
        if path not in self.meshes:
            m = trimesh.load(path, force="mesh")
            self.meshes[path] = wp.Mesh(
                points=wp.array(np.asarray(m.vertices, np.float32), dtype=wp.vec3, device=self.dev),
                indices=wp.array(np.asarray(m.faces, np.int32).flatten(), dtype=wp.int32, device=self.dev),
                support_winding_number=True,
            )
        return self.meshes[path].id

    def ids(self, paths: list[str]) -> wp.array:
        return wp.array(np.array([self.mesh_id(p) for p in paths], dtype=np.uint64), dtype=wp.uint64, device=self.dev)

    def evict(self, keep: set[str]):
        for k in [k for k in self.meshes if k not in keep]:
            del self.meshes[k]


# ---------------------------------------------------------------------------------------------------- robot geometry / FK
def _sample_cylinder(radius: float, half: float) -> np.ndarray:
    ang = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    ring = np.stack([np.cos(ang), np.sin(ang)], 1) * radius
    zs = np.linspace(-half, half, max(int(np.ceil(2 * half / 0.02)) + 1, 2))
    side = np.concatenate([np.concatenate([ring, np.full((len(ring), 1), z)], 1) for z in zs])
    rs = np.linspace(0, radius, 4)[1:]
    disk = np.concatenate([np.concatenate([np.stack([np.cos(ang), np.sin(ang)], 1) * r, np.full((len(ang), 1), z)], 1) for r in rs for z in (-half, half)])
    return np.concatenate([side, disk])


def _hull_faces(v: np.ndarray, spacing: float) -> np.ndarray:
    """Points on the faces of the convex hull of ``v``, ~``spacing`` apart (hull vertices included)."""
    hull = ConvexHull(v)
    out = [v[hull.vertices]]
    for tri in hull.simplices:
        a, b, c = v[tri]
        n = max(int(math.ceil(max(np.linalg.norm(b - a), np.linalg.norm(c - b), np.linalg.norm(a - c)) / spacing)), 1)
        for i in range(n + 1):
            for j in range(n + 1 - i):
                out.append((a + (b - a) * (i / n) + (c - a) * (j / n))[None])
    return np.unique(np.round(np.concatenate(out), 5), axis=0)


class Robot:
    """The collision geometry of the task's G1 and its differentiable forward kinematics (torch, any batch dims)."""

    def __init__(self, device: str, simplified_collision: bool = False, spacing: float = 0.03, voxel: float = 0.01, voxel_contact: float = 0.008,
                 joint_margin: float = 0.0):
        self.dev = device
        m = self.model = get_acp_g1_spec(simplified_collision=simplified_collision).compile()
        self.nb = m.nbody
        self.body_names = [m.body(i).name for i in range(m.nbody)]
        t = lambda x: torch.tensor(np.asarray(x), dtype=torch.float32, device=device)  # noqa: E731
        self.body_pos = t(m.body_pos)
        self.body_rot = quat_to_mat(t(m.body_quat))
        self.parent = [int(x) for x in m.body_parentid]
        self.joint_of = {}  # body -> (q column, axis (3,))
        self.joint_names = []
        for j in range(1, m.njnt):
            assert m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE and not np.any(m.jnt_pos[j]), "hinge joints at the body origin only"
            self.joint_of[int(m.jnt_bodyid[j])] = (j - 1, t(m.jnt_axis[j]))
            self.joint_names.append(m.joint(j).name)
        self.jrange = t(m.jnt_range[1:])
        assert joint_margin >= 0.0, "the joint margin cannot be negative"
        # the optimizer clamps inside the ranges by ``joint_margin``; the acceptance metrics keep using the true ranges
        self.clamp = self.jrange + t([joint_margin, -joint_margin])
        assert bool((self.clamp[:, 0] <= self.clamp[:, 1]).all()), "--joint-margin exceeds a joint range"
        assert all(m.jnt_qposadr[j] == 7 + j - 1 for j in range(1, m.njnt))
        # check points
        pts, body, is_vertex = [], [], []
        for g in np.nonzero((m.geom_contype != 0) | (m.geom_conaffinity != 0))[0]:
            kind = m.geom_type[g]
            if kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
                p, vert = _sample_cylinder(m.geom_size[g][0], m.geom_size[g][1]), False
            elif kind == mujoco.mjtGeom.mjGEOM_MESH:
                p = _hull_faces(_sample_geom(m, g), spacing)  # includes the hull vertices; marked below
                hv = _sample_geom(m, g)
                key = {tuple(np.round(x, 5)) for x in hv}
                vert = np.array([tuple(x) in key for x in p])
            else:
                p, vert = _sample_geom(m, g), True  # the labeler's own samples (capsules, spheres, boxes)
            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, m.geom_quat[g])
            pts.append(p @ R.reshape(3, 3).T + m.geom_pos[g])
            body.append(np.full(len(p), m.geom_bodyid[g]))
            is_vertex.append(np.broadcast_to(vert, (len(p),)))
        pts, body, is_vertex = np.concatenate(pts), np.concatenate(body), np.concatenate(is_vertex)
        # the labeler's points of the contact links (hull vertices): kept in full for the distance metrics (no gradient)
        link_ids = [m.body(n).id for n in CONTACT_LINKS]
        full = [(pts[(body == b) & is_vertex], b) for b in link_ids]
        self.mpts = [t(p) for p, _ in full]
        self.mbody = link_ids
        order = np.argsort(body, kind="stable")  # all samples, for the verification of the result
        self.full_pts, self.full_slices = t(pts[order]), {int(b): (int(np.searchsorted(body[order], b)), int(np.searchsorted(body[order], b, "right"))) for b in np.unique(body)}
        # the optimization uses one point per voxel (``voxel`` m) of each link: 100k hull samples are too many to query at every step
        # (the nine contact links finer: their min distance is compared with the labeler's, which sees all hull vertices)
        size = np.where(np.isin(body, link_ids), voxel_contact, voxel)[:, None]
        cell = np.floor(pts / size).astype(np.int64)
        _, first = np.unique(np.concatenate([body[:, None], cell], 1), axis=0, return_index=True)
        pts, body, is_vertex = pts[first], body[first], is_vertex[first]
        # continuous capsule / sphere geometry: the axis samples (every ``AXIS_STEP``) with the radius subtracted from the terrain SDF at
        # the sample are the exact signed distance of the capsule to a convex terrain primitive; the sparse surface samples above miss
        # the deepest point of a thick capsule between two of them (a 6 cm pelvis capsule: 2.4 mm)
        ax_pts, ax_body, ax_r = [], [], []
        for g in np.nonzero((m.geom_contype != 0) | (m.geom_conaffinity != 0))[0]:
            if m.geom_type[g] not in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_SPHERE):
                continue
            h = float(m.geom_size[g][1]) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE else 0.0
            zs = np.linspace(-h, h, max(int(np.ceil(2 * h / AXIS_STEP)) + 1, 1)) if h > 0 else np.zeros(1)
            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, m.geom_quat[g])
            ax_pts.append(np.stack([np.zeros_like(zs), np.zeros_like(zs), zs], 1) @ R.reshape(3, 3).T + m.geom_pos[g])
            ax_body.append(np.full(len(zs), m.geom_bodyid[g]))
            ax_r.append(np.full(len(zs), float(m.geom_size[g][0])))
        radius = np.zeros(len(pts))
        if ax_pts:
            pts, body = np.concatenate([pts] + ax_pts), np.concatenate([body] + ax_body)
            is_vertex, radius = np.concatenate([is_vertex, np.zeros(sum(map(len, ax_pts)), bool)]), np.concatenate([radius] + ax_r)
        order = np.argsort(body, kind="stable")
        pts, body, is_vertex, radius = pts[order], body[order], is_vertex[order], radius[order]
        self.pts, self.pbody, self.is_vertex, self.pr = t(pts), body, is_vertex, t(radius)
        self.slices = {int(b): (int(np.searchsorted(body, b)), int(np.searchsorted(body, b, "right"))) for b in np.unique(body)}
        self.contact_idx = [torch.tensor(np.nonzero(body == b)[0], device=device) for b in link_ids]
        self.foot_idx = [self.contact_idx[CONTACT_BODIES.index(n)] for n in FOOT_BODIES]
        self.collision_bodies = sorted(self.slices)
        self.csd = ContinuousSD(self, device)

    def columns(self, joint_names: list[str]) -> tuple[list[int], list[int]]:
        """(file columns of the model joints in model order, file columns that are not in the model (welded wrists))."""
        pos = {n: i for i, n in enumerate(joint_names)}
        cols = [pos[n] for n in self.joint_names]
        return cols, [i for i in range(len(joint_names)) if i not in cols]

    def fk(self, p: torch.Tensor, R: torch.Tensor, q: torch.Tensor):
        """Root position (..., 3) / rotation (..., 3, 3), joints (..., 23) -> body positions (..., nb, 3), rotations (..., nb, 3, 3)."""
        P, Rs = [p * 0, p], [R * 0, R]  # index 0 = world (unused), 1 = pelvis (free joint)
        for i in range(2, self.nb):
            par = self.parent[i]
            Pi = P[par] + torch.einsum("...ij,j->...i", Rs[par], self.body_pos[i])
            Ri = Rs[par] @ self.body_rot[i]
            if i in self.joint_of:
                col, axis = self.joint_of[i]
                Ri = Ri @ axis_angle_to_mat(axis.expand(q[..., col].shape + (3,)), q[..., col])
            P.append(Pi)
            Rs.append(Ri)
        return torch.stack(P, -2), torch.stack(Rs, -3)

    def world_points(self, bp: torch.Tensor, bR: torch.Tensor, full: bool = False) -> torch.Tensor:
        out = []
        pts, slices = (self.full_pts, self.full_slices) if full else (self.pts, self.slices)
        for b in self.collision_bodies:
            s, e = slices[b]
            out.append(torch.einsum("...ij,pj->...pi", bR[..., b, :, :], pts[s:e]) + bp[..., b, None, :])
        return torch.cat(out, -2)


# ---------------------------------------------------------------------------------------------------- continuous verification
CONT_TERRAINS: dict[str, "AnalyticTerrain"] = {}  # terrain STL path -> its geoms.yaml primitives (filled by main; empty = mesh verification)
CONT_STEP = 0.003  # m, coarse spacing of the capsule axis samples of the verification (the minimum is then refined to ~um)


class ContinuousSD:
    """Verification geometry: the exact signed distance of every capsule / sphere collision geom of the robot to the terrain primitives.

    sd(capsule) = min over the axis point a of ( SDF_terrain(a) - r ) (exact for a convex primitive); the axis is sampled every
    ``CONT_STEP`` and the best interval refined by ternary search, so unlike the surface samples of the optimization nothing can slip
    between two points. Also returns, per geom and frame, the terrain primitive that is closest (the contact surface)."""

    def __init__(self, robot: "Robot", device: str):
        m = robot.model
        gs = np.nonzero((m.geom_contype != 0) | (m.geom_conaffinity != 0))[0]
        self.ok = all(m.geom_type[g] in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_SPHERE) for g in gs)
        self.dev, self.body, self.pos, self.rot, self.h, self.r = device, [], [], [], [], []
        t = lambda x: torch.tensor(np.asarray(x), dtype=torch.float32, device=device)  # noqa: E731
        for g in gs if self.ok else []:
            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, m.geom_quat[g])
            self.body.append(int(m.geom_bodyid[g]))
            self.pos.append(t(m.geom_pos[g]))
            self.rot.append(t(R.reshape(3, 3)))
            self.h.append(float(m.geom_size[g][1]) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE else 0.0)
            self.r.append(float(m.geom_size[g][0]))

    @torch.no_grad()
    def eval(self, ter: "AnalyticTerrain", bp: torch.Tensor, bR: torch.Tensor, chunk: int = 256) -> tuple[torch.Tensor, torch.Tensor]:
        """bp (T,nb,3), bR (T,nb,3,3) -> (T, G) signed distance of each geom, (T, G) index of the closest terrain primitive."""
        T = bp.shape[0]
        sd_out = torch.empty(T, len(self.body), device=bp.device)
        id_out = torch.empty(T, len(self.body), dtype=torch.long, device=bp.device)
        for t0 in range(0, T, chunk):
            P, Rm = bp[t0 : t0 + chunk], bR[t0 : t0 + chunk]
            n = len(P)
            for gi, (b, gp, gr, h, r) in enumerate(zip(self.body, self.pos, self.rot, self.h, self.r)):
                c0 = P[:, b] + torch.einsum("tij,j->ti", Rm[:, b], gp)  # geom centre (n,3)
                ax = torch.einsum("tij,j->ti", Rm[:, b], gr[:, 2])  # unit axis (n,3)

                def at(z):  # z (n,) or (n,S) axis coordinate -> world points
                    return c0.unsqueeze(1) + ax.unsqueeze(1) * z.reshape(n, -1, 1)

                if h == 0.0:
                    d, i = ter.sd_no_grad(c0)
                    sd_out[t0 : t0 + n, gi], id_out[t0 : t0 + n, gi] = d - r, i
                    continue
                S = int(math.ceil(2 * h / CONT_STEP)) + 1
                zs = torch.linspace(-h, h, S, device=bp.device)
                d, i = ter.sd_no_grad(at(zs.expand(n, S)).reshape(-1, 3))
                d, i = d.view(n, S), i.view(n, S)
                best = d.argmin(-1)
                lo = zs[(best - 1).clamp(min=0)]
                hi = zs[(best + 1).clamp(max=S - 1)]
                for _ in range(14):  # ternary search on the bracket around the best sample
                    m1, m2 = lo + (hi - lo) / 3, hi - (hi - lo) / 3
                    f1 = ter.sd_no_grad(at(m1).reshape(-1, 3))[0]
                    f2 = ter.sd_no_grad(at(m2).reshape(-1, 3))[0]
                    left = f1 < f2
                    hi = torch.where(left, m2, hi)
                    lo = torch.where(left, lo, m1)
                zm = (lo + hi) / 2
                fm, im = ter.sd_no_grad(at(zm).reshape(-1, 3))
                coarse = d.gather(1, best[:, None])[:, 0]
                use = fm < coarse
                sd_out[t0 : t0 + n, gi] = torch.where(use, fm, coarse) - r
                id_out[t0 : t0 + n, gi] = torch.where(use, im, i.gather(1, best[:, None])[:, 0])
        return sd_out, id_out


# ---------------------------------------------------------------------------------------------------- clips
class Clip:
    """One motion of the dataset on the GPU (poses of the source, contact flags of its sidecar)."""

    def __init__(self, root: str, rel: str, terrain_path: str, robot: Robot, device: str):
        self.rel, self.terrain_path = rel, terrain_path
        self.src = dict(np.load(os.path.join(root, rel), allow_pickle=True))
        self.fps = float(self.src["framerate"])
        cols, self.other_cols = robot.columns([str(n) for n in self.src["joint_names"]])
        self.cols = cols
        t = lambda x: torch.tensor(np.asarray(x), dtype=torch.float32, device=device)  # noqa: E731
        self.p0, self.R0, self.q0 = t(self.src["base_pos_w"]), quat_to_mat(t(self.src["base_quat_w"])), t(self.src["joint_pos"][:, cols])
        self.T = len(self.p0)
        side = os.path.join(root, rel[: -len(".npz")] + ".contacts.npz")
        lab = np.load(side) if os.path.exists(side) else None
        self.flag = lab["contact_flag"] if lab is not None else np.zeros((self.T, len(CONTACT_BODIES)), bool)
        self.cpos = lab["contact_pos_w"] if lab is not None else np.zeros((self.T, len(CONTACT_BODIES), 3), np.float32)
        assert self.flag.shape[0] == self.T, f"{rel}: sidecar has {self.flag.shape[0]} frames, motion {self.T}"
        # core / tail of the labelled segments from the labeler's own stored features (original retarget); None when the
        # sidecar predates them (fall back to the flags)
        try:
            self.core, self.tail = core_tail_masks(self.flag, lab["min_dist"], lab["contact_speed"], json.loads(str(lab["params"])))
            self.moving = labeler_moving(lab["contact_speed"], json.loads(str(lab["params"])))  # sliding contacts: not pinned by the no-slide term
        except (KeyError, ValueError):
            self.core, self.tail = self.flag, np.zeros_like(self.flag)
            self.moving = np.zeros_like(self.flag)

    def window(self, s: int, e: int, anchor: int) -> "Clip":
        """Sub-clip of frames [s - anchor, e + anchor) (clipped to the clip): only [s, e) are free variables, the anchor
        frames on either side stay at the source (``free`` = 0) and tie the optimized part to the untouched rest."""
        a, b = max(s - anchor, 0), min(e + anchor, self.T)
        w = copy.copy(self)
        w.parent, w.win_range, w.win_free = self, (a, b), (s, e)
        w.p0, w.R0, w.q0 = self.p0[a:b], self.R0[a:b], self.q0[a:b]
        w.flag, w.cpos, w.core, w.tail, w.moving = self.flag[a:b], self.cpos[a:b], self.core[a:b], self.tail[a:b], self.moving[a:b]
        w.T = b - a
        w.free = torch.zeros(w.T, device=self.p0.device)
        w.free[s - a : e - a] = 1.0
        w.jit_pre = self.jit_pre[a:b] if getattr(self, "jit_pre", None) is not None else None
        return w


def pad(x: torch.Tensor, T: int) -> torch.Tensor:
    return torch.cat([x, x[-1:].expand(T - len(x), *x.shape[1:])]) if len(x) < T else x


def pad_ones(x: torch.Tensor, T: int) -> torch.Tensor:
    return torch.cat([x, x.new_ones(T - len(x))]) if len(x) < T else x


class Batch:
    """Clips padded to a common length; evaluates the terrain distance of a trajectory."""

    def __init__(self, clips: list[Clip], robot: Robot, terrains: Terrains, device: str, analytics: dict | None = None):
        self.clips, self.robot, self.dev = clips, robot, device
        self.analytic = {c.terrain_path: analytics[c.terrain_path] for c in clips} if analytics else None
        self.T = max(c.T for c in clips)
        self.p0 = torch.stack([pad(c.p0, self.T) for c in clips])
        self.R0 = torch.stack([pad(c.R0, self.T) for c in clips])
        self.q0 = torch.stack([pad(c.q0, self.T) for c in clips])
        self.valid = torch.stack([torch.arange(self.T, device=device) < c.T for c in clips]).float()  # (B, T)
        self.flag = torch.stack([pad(torch.tensor(c.flag, device=device), self.T) for c in clips]).float() * self.valid[..., None]
        self.core = torch.stack([pad(torch.tensor(getattr(c, "core", c.flag), device=device), self.T) for c in clips]).float() * self.valid[..., None]
        self.ids = terrains.ids([c.terrain_path for c in clips])
        # window mode: 0 on the fixed anchor frames (their deltas are forced to zero), 1 elsewhere (and on the padding)
        self.free = None
        if any(hasattr(c, "free") for c in clips):
            self.free = torch.stack([pad_ones(getattr(c, "free", torch.ones(c.T, device=device)), self.T) for c in clips])[..., None]  # (B, T, 1)

    def pose(self, dp=None, dr=None, dq=None):
        p, R, q = self.p0, self.R0, self.q0
        if dp is not None:
            if self.free is not None:
                dp, dr, dq = dp * self.free, dr * self.free, dq * self.free
            p, R = p + dp, R @ so3_exp(dr)
            q_new = torch.max(torch.min(q + dq, self.robot.clamp[:, 1]), self.robot.clamp[:, 0])
            q = q_new if self.free is None else torch.where(self.free > 0, q_new, q)  # fixed frames are exactly the source
        return p, R, q

    def evaluate(self, p, R, q):
        """(body positions (B,T,nb,3), signed distance of the check points (B,T,P), world points (B,T,P,3))."""
        bp, bR = self.robot.fk(p, R, q)
        pts = self.robot.world_points(bp, bR)
        B, T, P, _ = pts.shape
        if self.analytic:  # exact primitive SDF (fast path of --analytic-terrain); the mesh path stays for the verification
            sd = torch.stack([_AnalyticSD.apply(pts[k].reshape(T * P, 3), self.analytic[c.terrain_path]).view(T, P) for k, c in enumerate(self.clips)])
            return bp, sd - self.robot.pr, pts
        return bp, SignedDistance.apply(pts.reshape(B, T * P, 3), self.ids).view(B, T, P) - self.robot.pr, pts


def _contact_metrics(self, bp, bR, dist: bool = True):
    """(B,T,9) unsigned distance of each contact link's labeler points (full hull vertex sets) to the terrain (mesh; zeros without ``dist``),
    and (B,T,9) distance of the closest of those points to the labelled contact position ``contact_pos_w`` (numpy)."""
    d, tp = [], []
    cpos = torch.stack([pad(torch.tensor(c.cpos, device=self.dev), self.T) for c in self.clips])
    for i, (pts, bid) in enumerate(zip(self.robot.mpts, self.robot.mbody)):
        w = torch.einsum("...ij,pj->...pi", bR[..., bid, :, :], pts) + bp[..., bid, None, :]  # (B,T,M,3)
        B, T, M, _ = w.shape
        if dist:
            sd = unsigned_distance(w.reshape(B, T * M, 3), self.ids).view(B, T, M)
            d.append(sd.min(-1).values)
        else:
            d.append(torch.zeros(B, T, device=self.dev))
        tp.append((w - cpos[:, :, i, None]).norm(dim=-1).min(-1).values)
    return torch.stack(d, -1).cpu().numpy(), torch.stack(tp, -1).cpu().numpy()


Batch.contact_metrics = _contact_metrics


def _dense_sd(self, p, R, q, chunk: int = 24):
    """(B,T,Pfull) signed distance of all collision samples (hull faces + vertices, no thinning), per body min is taken by the caller."""
    bp, bR = self.robot.fk(p, R, q)
    out = []
    for t0 in range(0, self.T, chunk):
        pts = self.robot.world_points(bp[:, t0 : t0 + chunk], bR[:, t0 : t0 + chunk], full=True)
        B, T, P, _ = pts.shape
        out.append(SignedDistance.apply(pts.reshape(B, T * P, 3), self.ids).view(B, T, P))
    return torch.cat(out, 1)


Batch.dense_sd = _dense_sd


def body_track_weights(robot: Robot, spec: str | None, device: str) -> torch.Tensor:
    """Per body weights of the tracking term: 1.0, scaled by the first matching 'regex=scale' of --track-scale (bodies 1.. of the tree)."""
    w = torch.ones(robot.nb, device=device)
    if spec:
        for rule in spec.split(","):
            pat, _, val = rule.rpartition("=")
            scale = float(val)
            sel = [i for i, n in enumerate(robot.body_names) if re.search(pat, n)]
            assert sel, f"--track-scale pattern '{pat}' matches no body"
            w[torch.tensor(sel, device=device, dtype=torch.long)] = scale
    return w


def _alternating_mask(sign: torch.Tensor, win: int) -> torch.Tensor:
    """(N,) bool: frames covered by >= win consecutive alternating signs (numerical oscillation). ``sign`` has zeros below the floor."""
    N = sign.shape[0]
    if N < win + 1:
        return torch.zeros(N, dtype=torch.bool, device=sign.device)
    alt = (sign != 0) & (sign == -torch.roll(sign, 1, 0))
    alt[0] = False
    k = torch.ones(win - 1, dtype=sign.dtype, device=sign.device)
    c = torch.nn.functional.conv1d(alt.float().view(1, 1, -1), k.view(1, 1, -1)).view(-1) >= win - 1
    m = torch.zeros(N, dtype=torch.bool, device=sign.device)
    starts = torch.nonzero(c).flatten().tolist()
    for st in starts:
        m[st : st + win] = True
    return m


def jitter_frames(q0: torch.Tensor, p0: torch.Tensor, win: int, floor_q: float, floor_p: float) -> torch.Tensor:
    """(T,) bool: frames inside alternating-sign second-difference windows of ANY joint / root axis (numerical jitter:
    frame-to-frame oscillation without physical meaning; real fast motion - takeoff, landing - is a sustained or single-signed
    acceleration and never alternates for ``win`` frames). Per joint / axis, not on the max over joints (different joints
    peaking in turn must not read as oscillation)."""
    with torch.no_grad():
        T = q0.shape[0]
        out = torch.zeros(T, dtype=torch.bool, device=q0.device)
        for arr, floor in ((q0, floor_q), (p0, floor_p)):
            if arr.shape[0] < win + 3:
                continue
            a2 = torch.diff(arr, 2, dim=0)
            sign = torch.where(a2.abs() >= floor, torch.sign(a2), torch.zeros_like(a2))
            for j in range(arr.shape[1]):
                m = torch.zeros(T, dtype=torch.bool, device=arr.device)
                m[: a2.shape[0]] = _alternating_mask(sign[:, j], win)
                out |= m
        return out


def core_anchor_runs(sd0: torch.Tensor, pts0: torch.Tensor, core: torch.Tensor, clip_k: int, robot: Robot,
                     slice_m: float = 0.015, moving: np.ndarray | None = None) -> list[tuple[int, int, torch.Tensor, torch.Tensor]]:
    """Per core contact run of every contact body: (s, e, anchor point indices, anchor xy). The anchors are the run's first-frame
    touching points (within ``slice_m`` of the body's closest point); their horizontal world position is held through the run
    (material points stick - a rolling foot pivots over them, a sliding one is dragged back)."""
    T = core.shape[0]
    out = []
    for bi, idx in enumerate(robot.contact_idx):
        col = core[:, bi].cpu().numpy() if core.is_cuda else core[:, bi].numpy()
        edges = np.diff(np.r_[0, col.astype(np.int8), 0])
        for s, e in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            if e - s < 3 or (moving is not None and moving[s:e, bi].mean() > 0.5):  # a sliding contact legitimately moves
                continue
            d = sd0[clip_k, s, idx]
            sel = (d <= d.min() + slice_m).nonzero(as_tuple=True)[0]
            if len(sel) >= 3:
                out.append((int(s), int(e), idx[sel], pts0[clip_k, s, idx[sel], :2].detach().clone()))
    return out


def dejitter_loss(q: torch.Tensor, p: torch.Tensor, jit: torch.Tensor) -> torch.Tensor:
    """Third difference of the output joints (B,T,J) / root (B,T,3) on the jitter frames (B,T); squared per joint / axis first, then
    averaged, so opposite-sign jitter of two joints (or a joint and the root) cannot cancel."""
    m = jit.sum().clamp(min=1.0)
    d3q = torch.cat([(torch.diff(q, 3, dim=1) ** 2).mean(-1), q.new_zeros(q.shape[0], 3)], 1)
    d3p = torch.cat([(torch.diff(p, 3, dim=1) ** 2).sum(-1), p.new_zeros(p.shape[0], 3)], 1)
    return ((d3q + 0.01 * d3p) * jit).sum() / m


GROUND_SD = 0.003  # m, a contact point is "on the ground" within this distance of the terrain
GROUND_VZ = 0.15  # m/s, and (nearly) not moving vertically (a swinging foot just above the floor is not stance)
GROUND_MIN_FRAMES = 3


def grounded_segments(sd: torch.Tensor, pts: torch.Tensor, fps: float, sd_thr: float = GROUND_SD, vz_thr: float = GROUND_VZ,
                      min_len: int = GROUND_MIN_FRAMES):
    """Material-point stance segments. sd (T,P) signed distance of the points to the terrain, pts (T,P,3) world positions. A point is
    grounded in a frame when sd <= ``sd_thr`` and its vertical speed <= ``vz_thr``; a segment is a maximal run of grounded frames of ONE
    point (a rolling foot hands the contact over to new points, so the heel->toe roll or a swing is never one segment). Returns
    (valid (T,P): grounded in a segment of >= ``min_len`` frames, start (T,P): first frame of the point's segment, disp (T,P): horizontal
    distance of the point from where its segment started)."""
    T, P = sd.shape
    z = pts[..., 2]
    vz = torch.zeros_like(z)
    if T > 2:
        vz[1:-1] = (z[2:] - z[:-2]) * (fps / 2.0)
        vz[0], vz[-1] = vz[1], vz[-2]
    g = (sd <= sd_thr) & (vz.abs() <= vz_thr)
    tt = torch.arange(T, device=sd.device)[:, None].expand(T, P)
    start = torch.cummax(torch.where(g, torch.full_like(tt, -1), tt), 0).values + 1
    nxt = torch.flip(torch.cummin(torch.flip(torch.where(g, torch.full_like(tt, T), tt), [0]), 0).values, [0])
    valid = g & ((nxt - start) >= min_len)
    xy = pts[..., :2]
    x0 = torch.gather(xy, 0, start.clamp(0, T - 1)[..., None].expand(T, P, 2))
    return valid, start, (xy - x0).norm(dim=-1)


def build_noslide(b: "Batch", sd0: torch.Tensor, pts0: torch.Tensor, bodies_regex: str, max_points: int = 24):
    """No-slide anchors of the REAL stance: every contact point of the matching contact links, in every frame where it is grounded
    (within ``GROUND_SD`` of the terrain, vertical speed ~0) and the contact is labelled core and static; the anchor is the point's position
    at the start of its grounded segment (swing / lift-off / touch-down frames and sliding contacts are never pinned). Flattened to one
    gather per iteration: (frame, clip, point index, weight, anchor xy) or None. The weight of one point in one frame is 1 / (T_clip * 16)
    (a float tensor), a per-clip frame mean like the other terms; at most ``max_points`` points per contact link are kept."""
    rob, dev = b.robot, b.dev
    want = [bool(re.search(bodies_regex, n)) for n in CONTACT_BODIES]
    T_idx, K_idx, P_idx, W, A = [], [], [], [], []
    for k, c in enumerate(b.clips):
        T_k = c.T
        moving = getattr(c, "moving", None)
        for bi, idx in enumerate(rob.contact_idx):
            if not want[bi]:
                continue
            valid, start, _ = grounded_segments(sd0[k, :T_k][:, idx], pts0[k, :T_k][:, idx], c.fps)
            lab = torch.as_tensor(c.core[:, bi] & (~moving[:, bi] if moving is not None else True), device=dev)
            valid = valid & lab[:, None]
            if not bool(valid.any()):
                continue
            keep = valid.sum(0).topk(min(max_points, valid.shape[1])).indices
            valid, start = valid[:, keep], start[:, keep]
            tt, jj = valid.nonzero(as_tuple=True)
            pi = idx[keep][jj]
            T_idx.append(tt); K_idx.append(torch.full_like(tt, k)); P_idx.append(pi)
            W.append(torch.full(tt.shape, 1.0 / (T_k * 16), dtype=torch.float32, device=dev))
            A.append(pts0[k, start[tt, jj], pi, :2])
    if not T_idx:
        return None
    return torch.cat(T_idx), torch.cat(K_idx), torch.cat(P_idx), torch.cat(W), torch.cat(A)


def noslide_loss(pts: torch.Tensor, ns, tol: float = 0.0) -> torch.Tensor:
    """Squared hinge of the horizontal drift of the anchor points beyond ``tol`` (m): only an OBVIOUS slide is pulled back (to the
    edge of the ``tol`` band), a slight one is left alone."""
    ns_t, ns_k, ns_p, ns_w, ns_a = ns
    d = torch.sqrt(((pts[ns_k, ns_t, ns_p, :2] - ns_a) ** 2).sum(-1) + 1e-12)
    return (ns_w * torch.relu(d - tol) ** 2).sum()


def refine_batch(b: Batch, args, touch: bool = False, noslide: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Adam over root / joint deltas of the batch; returns the new (p, R, q)."""
    rob, dev = b.robot, b.dev
    with torch.no_grad():
        bp0, sd0, pts0 = b.evaluate(*b.pose())
        mind0 = torch.stack([sd0[..., idx].min(-1).values for idx in rob.contact_idx], -1)  # (B, T, 9) label-style contact distance
        foot0 = [pts0[..., idx, :2] for idx in rob.foot_idx]
    d_on = torch.tensor(D_ON_BODIES, dtype=torch.float32, device=dev)
    keep_thr = torch.maximum(d_on - args.keep_slack, mind0)  # never demand closer than the source was
    touch_rng = d_on if args.touch_range is None else torch.full_like(d_on, args.touch_range)
    labels = b.core if getattr(args, "touch_labels", "flag") == "core" else b.flag
    touch_mask = labels * (mind0.abs() <= touch_rng).float()  # frames the touch term pulls (core or flag, within the range)
    tw = body_track_weights(rob, args.track_scale, dev)  # bodies 1.. (0 = world stays 1, never used)
    tw[0] = tw[1:].mean() if len(tw) > 1 else 1.0
    # no-slide anchors of the core contact runs (the run's first touching slice: material points whose xy is held; a rolling
    # foot pivots over them, a skating one is pulled back) - flattened to one gather per iteration (a long ladder clip has
    # ~1000 runs; a per-run python loop in the loss would dominate the runtime)
    ns = build_noslide(b, sd0, pts0, args.noslide_bodies) if (noslide and args.w_noslide > 0) else None
    # numerical-jitter frames of the source (alternating-sign second differences): tracking relaxes there, a third-difference
    # penalty flattens them; real fast motion (single-signed impulses) is never selected
    jit = None
    if args.de_jitter > 0:
        jit = torch.stack([pad(c.jit_pre if getattr(c, "jit_pre", None) is not None else
                               jitter_frames(c.q0, b.p0[k, : c.T], args.de_jitter_win, args.de_jitter, args.de_jitter_floor_p), b.T)
                           for k, c in enumerate(b.clips)]) * b.valid
    w_tr = (1.0 - args.de_jitter_track * jit).clamp(min=0.05) if jit is not None else None
    margin = (args.touch_margin if touch else args.margin) + args.slack  # + slack: the optimization sees a thinned point set
    foot_flag = [b.flag[..., CONTACT_BODIES.index(n)] for n in FOOT_BODIES]
    nvalid = b.valid.sum(-1).clamp(min=1)
    mean_t = lambda x: ((x * b.valid).sum(-1) / nvalid).sum()  # noqa: E731  per clip mean over frames, summed over clips
    dp, dr, dq = (torch.zeros(*s, device=dev, requires_grad=True) for s in (b.p0.shape, b.p0.shape, b.q0.shape))
    opt = torch.optim.Adam((dp, dr, dq), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.iters)
    best, stale = float("inf"), 0
    for it in range(args.iters):
        p, R, q = b.pose(dp, dr, dq)
        bp, sd, pts = b.evaluate(p, R, q)
        if b.free is not None:  # anchors: deltas are zero in the pose; the same masked deltas enter the ridge / smoothness terms
            dpe, dre, dqe = dp * b.free, dr * b.free, dq * b.free
        else:
            dpe, dre, dqe = dp, dr, dq
        track_body = ((((bp[..., 1:, :] - bp0[..., 1:, :]) ** 2) * tw[1:, None]).sum(-1).sum(-1) / tw[1:].sum())
        if w_tr is not None:  # tracking relaxes on the numerical-jitter frames (the de-jitter term rebuilds them smooth)
            track = ((track_body * w_tr * b.valid).sum(-1).sum() / (w_tr * b.valid).sum(-1).sum().clamp(min=1.0)) \
                + mean_t(0.01 * (dqe**2).mean(-1) + 0.01 * (dre**2).mean(-1))
        else:
            track = mean_t(track_body) + mean_t(0.01 * (dqe**2).mean(-1) + 0.01 * (dre**2).mean(-1))
        clear = mean_t((torch.relu(margin - sd) ** 2).sum(-1))
        mind = torch.stack([sd[..., idx].min(-1).values for idx in rob.contact_idx], -1)
        keep = mean_t((b.flag * torch.relu(mind - keep_thr) ** 2).sum(-1))
        touching = mean_t((touch_mask * (mind - args.touch_target - args.slack) ** 2).sum(-1)) if touch else 0.0
        foot = sum(mean_t(f * ((pts[..., idx, :2] - f0) ** 2).sum(-1).mean(-1)) for f, idx, f0 in zip(foot_flag, rob.foot_idx, foot0))
        noslide_term = noslide_loss(pts, ns, args.noslide_tol) if ns is not None else 0.0
        smooth = sum(
            mean_t(torch.cat([((x[:, 1:] - x[:, :-1]) ** 2).mean(-1), x.new_zeros(len(x), 1)], 1))
            + 10 * mean_t(torch.cat([((x[:, 2:] - 2 * x[:, 1:-1] + x[:, :-2]) ** 2).mean(-1), x.new_zeros(len(x), 2)], 1))
            for x in (dpe, dre, dqe)
        )  # fmt: skip
        dejitter = 0.0
        if jit is not None:  # third difference of the OUTPUT joints / root on the jitter frames only
            dejitter = dejitter_loss(q, p, jit)
        loss = track + args.w_clear * clear + args.w_keep * keep + args.w_foot * foot + args.w_touch * touching + args.w_smooth * smooth \
            + args.w_noslide * noslide_term + args.w_jitter * dejitter
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if args.early_stop > 0 and (it + 1) % 25 == 0:  # converged = no 0.1% loss improvement for ``early_stop`` checks
            with torch.no_grad():
                if loss.item() < best * 0.999:
                    best, stale = loss.item(), 0
                else:
                    stale += 1
                    if stale >= args.early_stop:
                        break
    with torch.no_grad():
        return b.pose(dp, dr, dq)


# ---------------------------------------------------------------------------------------------------- metrics
def measure(b: Batch, p, R, q) -> list[dict]:
    """Per clip: per-body min signed distance, penetration, contact-distance and foot metrics of the trajectory (p, R, q).
    With the geoms.yaml primitives registered (``CONT_TERRAINS``) and a capsule / sphere robot the distances are the exact continuous
    ones (``ContinuousSD``, ``backend`` = "continuous"); otherwise the dense surface samples against the STL."""
    rob = b.robot
    cont = rob.csd.ok and all(c.terrain_path in CONT_TERRAINS for c in b.clips)
    with torch.no_grad():
        bp, sd, pts = b.evaluate(p, R, q)
        bp_f, bR_f = rob.fk(p, R, q)
        full_contact = b.contact_metrics(bp_f, bR_f, dist=not cont)
        if not cont:
            sd = b.dense_sd(p, R, q)  # verification on every sampled point (not the thinned optimization set)
        out = []
        for k, c in enumerate(b.clips):
            per_body, surf = {}, None
            if cont:
                sg, ig = rob.csd.eval(CONT_TERRAINS[c.terrain_path], bp_f[k, : c.T], bR_f[k, : c.T])
                for bid in rob.collision_bodies:
                    gi = [i for i, gb in enumerate(rob.csd.body) if gb == bid]
                    if gi:
                        per_body[rob.body_names[bid]] = float(sg[:, gi].min())
                frame_min = sg.min(-1).values
                cd, surf = [], []
                for bid in rob.mbody:
                    gi = [i for i, gb in enumerate(rob.csd.body) if gb == bid]
                    dmin, am = sg[:, gi].min(-1)
                    cd.append(dmin)
                    surf.append(ig[:, gi].gather(1, am[:, None])[:, 0])
                mind = torch.stack(cd, -1).cpu().numpy()
                surf = torch.stack(surf, -1).cpu().numpy()
                to_pos = full_contact[1][k, : c.T]
            else:
                s = sd[k, : c.T]
                for bid in rob.collision_bodies:
                    lo, hi = rob.full_slices[bid]
                    per_body[rob.body_names[bid]] = float(s[:, lo:hi].min())
                frame_min = s.min(-1).values
                mind, to_pos = full_contact[0][k, : c.T], full_contact[1][k, : c.T]
            feet = [pts[k, : c.T][:, idx, :2].mean(1).cpu().numpy() for idx in rob.foot_idx]
            fa = torch.cat(rob.foot_idx)
            gv, _, gd = grounded_segments(sd[k, : c.T][:, fa] if not cont else b.thin_sd(p, R, q, k)[:, fa], pts[k, : c.T][:, fa], c.fps)
            real_slip = float(gd[gv].max()) if bool(gv.any()) else 0.0
            out.append(dict(per_body=per_body, min_sd=float(frame_min.min()), pen_frac=float((frame_min < -PEN_EPS).float().mean()),
                            pen_max=float(max(-frame_min.min(), 0)), viol_frac=float((frame_min < args_margin_default()).float().mean()),
                            frame_min=frame_min.cpu().numpy(), contact_dist=mind, to_pos=to_pos, surf=surf, backend="continuous" if cont else "mesh",
                            feet_xy=feet, real_slip=real_slip, body_pos=bp[k, : c.T, 1:].cpu().numpy()))  # fmt: skip
    return out


def args_margin_default() -> float:
    return MARGIN


def runs(f: np.ndarray):
    e = np.diff(np.r_[0, f.astype(np.int8), 0])
    return list(zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)))


ACCEPT = dict(
    joint_max_deg=30.0, joint_mean_deg=3.0, root_max_cm=8.0, body_move_mean_cm=3.0, body_move_max_cm=20.0,
    foot_drift_cm=3.0, foot_slip_growth_cm=3.0, jerk_ratio=1.5, jerk_q_abs=0.01, jerk_p_abs=0.003, limit_hit_growth=0.02,
    selfcol_growth=0.03, contact_in_d_on_drop=0.05, pen_tol=PEN_EPS,
    # absolute goals (the relative ones above only forbid getting worse): the core contact gap, the contact surface, joint steps, stance slip
    # naturalness gates (tiers only): root rotation, speed / acceleration / jerk growth (p99 and max, ratio to the source), high-frequency energy,
    # near-limit frames, correlation of each moving joint's time series with the source
    root_rot_deg=5.0, dyn_ratio=1.2, hf_ratio=1.25, near_limit_growth=0.05, joint_corr_min=0.95,
    core_gap_m=0.0015, core_gap_ok_frac=0.9, touch_range_m=0.035, surface_change_frac=0.05, joint_step_rad=0.35, core_slip_cm=1e9,
)  # fmt: skip
PHYSICS_KEYS = ("pen_tol", "limit_hit_growth", "selfcol_growth", "jerk_ratio", "jerk_q_abs", "jerk_p_abs", "joint_step_rad")
"""Acceptance thresholds per clip (touch candidates and the fallback alike), see the file header."""


def apply_accept_overrides(spec: str | None) -> None:
    """--accept-override 'joint_max_deg=45,root_max_cm=30,...': raise individual acceptance thresholds (a slide fix moves a
    skating stance leg by design; the physics gates - penetration, joint limits, jerk - stay as hard as before)."""
    if not spec:
        return
    for rule in spec.split(","):
        k, _, v = rule.partition("=")
        assert k in ACCEPT, f"--accept-override: unknown key '{k}' (have {sorted(ACCEPT)})"
        assert k not in PHYSICS_KEYS or float(v) <= ACCEPT[k], f"--accept-override: the physics gate '{k}' may only be tightened ({ACCEPT[k]} -> {v})"
        ACCEPT[k] = float(v)


def self_collision_frames(robot: Robot, p: np.ndarray, R: np.ndarray, q: np.ndarray, depth: float = 0.005) -> float:
    """Fraction of frames with a robot-robot collision geom pair penetrating by more than ``depth`` (MuJoCo, kinematics + collision)."""
    m = robot.model
    d = mujoco.MjData(m)
    quat = Rotation.from_matrix(R.astype(np.float64)).as_quat()[:, [3, 0, 1, 2]]
    bad = 0
    for t in range(len(p)):
        d.qpos[:3], d.qpos[3:7], d.qpos[7:] = p[t], quat[t], q[t]
        mujoco.mj_kinematics(m, d)
        mujoco.mj_collision(m, d)
        bad += bool(d.ncon and (d.contact.dist[: d.ncon] < -depth).any())
    return bad / len(p)


DYN_FLOOR = dict(q=(0.004, 0.004, 0.008), p=(0.0004, 0.0004, 0.0008))  # per-frame |diff| floors (rad / m) of orders 1, 2, 3


def natural_metrics(q0: np.ndarray, qn: np.ndarray, p0: np.ndarray, pn: np.ndarray, robot: Robot) -> dict:
    """Smoothness / human-likeness numbers of a trajectory against its source: p99 and max of the |velocity| / |acceleration| / |jerk| (per frame
    differences of orders 1-3) of the joints and the root, the high-frequency (third difference) energy, the share of frames near a joint limit,
    and the lowest correlation of a moving joint's time series with the source."""
    out = {}
    for nm, (a, b) in (("q", (q0, qn)), ("p", (p0, pn))):
        for o, lab in ((1, "vel"), (2, "acc"), (3, "jerk")):
            for tag, x in (("before", a), ("after", b)):
                d = np.abs(np.diff(x, o, axis=0)) if len(x) > o else np.zeros((1, x.shape[1]))
                out[f"{lab}_{nm}_p99_{tag}"] = float(np.percentile(d, 99))
                out[f"{lab}_{nm}_max_{tag}"] = float(d.max())
    hf = lambda x: float((np.diff(x, 3, axis=0) ** 2).mean()) if len(x) > 3 else 0.0  # noqa: E731
    out["hf_q_before"], out["hf_q_after"] = hf(q0), hf(qn)
    lo, hi = robot.jrange[:, 0].cpu().numpy(), robot.jrange[:, 1].cpu().numpy()
    near = lambda x: float(((x <= lo + 0.05) | (x >= hi - 0.05)).any(-1).mean())  # noqa: E731
    out["near_limit_before"], out["near_limit_after"] = near(q0), near(qn)
    moving = q0.std(0) > 0.05
    cors = []
    for j in np.nonzero(moving)[0]:
        if qn[:, j].std() < 1e-9:
            cors.append(0.0)
        else:
            cors.append(float(np.corrcoef(q0[:, j], qn[:, j])[0, 1]))
    out["joint_corr_min"] = min(cors) if cors else 1.0
    return out


def compare(clip: Clip, robot: Robot, before: dict, after: dict, pn: np.ndarray, Rn: np.ndarray, qn: np.ndarray) -> dict:
    """Acceptance numbers of one clip: the source (``clip``, ``before`` from :func:`measure`) against the trajectory (pn, Rn, qn)."""
    p0, R0, q0 = clip.p0.cpu().numpy(), clip.R0.cpu().numpy(), clip.q0.cpu().numpy()
    joint_delta, root_delta, rot_delta = qn - q0, pn - p0, so3_angle(R0, Rn)
    flag = clip.flag
    on = flag
    cons = {}
    was_on = flag & (before["contact_dist"] <= D_ON_BODIES)
    for name, m in (("before", before), ("after", after)):
        d = m["contact_dist"]
        cons[f"contact_in_1p5cm_{name}"] = float((d[on] <= 0.015).mean()) if on.any() else 1.0
        cons[f"contact_in_d_on_{name}"] = float((d[on] <= D_ON_BODIES[np.nonzero(on)[1]]).mean()) if on.any() else 1.0
        cons[f"contact_to_labelled_pos_p95_cm_{name}"] = float(100 * np.percentile(m["to_pos"][on], 95)) if on.any() else 0.0
    cons["contact_dist_mean_mm_after"] = float(1000 * after["contact_dist"][on].mean()) if on.any() else 0.0
    cons["contact_dist_p95_mm_after"] = float(1000 * np.percentile(after["contact_dist"][on], 95)) if on.any() else 0.0
    slip = {}
    for name, m in (("before", before), ("after", after)):
        worst = 0.0
        for fi, foot_name in enumerate(FOOT_BODIES):
            for s, e in runs(flag[:, CONTACT_BODIES.index(foot_name)]):
                worst = max(worst, float(np.linalg.norm(m["feet_xy"][fi][s:e] - m["feet_xy"][fi][s], axis=1).max()))
        slip[name] = worst
    foot_move = 0.0
    for fi, foot_name in enumerate(FOOT_BODIES):
        for s, e in runs(flag[:, CONTACT_BODIES.index(foot_name)]):
            foot_move = max(foot_move, float(np.linalg.norm(after["feet_xy"][fi][s:e] - before["feet_xy"][fi][s:e], axis=1).max()))
    body_move = np.linalg.norm(after["body_pos"] - before["body_pos"], axis=-1)
    jerk = lambda x: float(np.abs(np.diff(x, 3, axis=0)).max()) if len(x) > 3 else 0.0  # noqa: E731
    lo, hi = robot.jrange[:, 0].cpu().numpy(), robot.jrange[:, 1].cpu().numpy()
    hit = lambda x: float(((x <= lo + 1e-3) | (x >= hi - 1e-3)).any(-1).mean())  # noqa: E731
    # no-slide / de-jitter acceptance numbers: absolute drift of each foot's mean xy over its CORE runs, and the
    # numerical-jitter oscillation fraction / second-difference p99 of the joint trajectory
    core_slip = {}
    for name, m in (("before", before), ("after", after)):
        worst = 0.0
        for fi, foot_name in enumerate(FOOT_BODIES):
            col = clip.core[:, CONTACT_BODIES.index(foot_name)]
            for s, e in runs(col):
                if clip.moving[s:e, CONTACT_BODIES.index(foot_name)].mean() > 0.5:  # a sliding contact legitimately moves
                    continue
                xy = m["feet_xy"][fi][s:e]
                worst = max(worst, float(np.linalg.norm(xy - xy[0], axis=1).max()))
        core_slip[name] = 100 * worst
    osc = {name: float(jit.float().mean()) for name, jit in
           (("before", jitter_frames(torch.tensor(q0), torch.tensor(p0), 4, 0.02, 0.002)),
            ("after", jitter_frames(torch.tensor(qn), torch.tensor(pn), 4, 0.02, 0.002)))}
    # absolute goals: share of the core frames (reachable by the touch term in the source) within the gap target, and the contact surface
    reach = clip.core & (before["contact_dist"] <= ACCEPT["touch_range_m"])
    gap_ok = {n: float((m["contact_dist"][reach] <= ACCEPT["core_gap_m"]).mean()) if reach.any() else 1.0 for n, m in (("before", before), ("after", after))}
    if reach.any():
        gap_p95 = float(1000 * np.percentile(after["contact_dist"][reach], 95))
        gap_med = float(1000 * np.median(after["contact_dist"][reach]))
    else:
        gap_p95 = gap_med = 0.0
    surf_change = float((after["surf"][reach] != before["surf"][reach]).mean()) if reach.any() and before.get("surf") is not None and after.get("surf") is not None else 0.0
    nat = natural_metrics(q0, qn, p0, pn, robot)
    step = lambda x: float(np.abs(np.diff(x, axis=0)).max()) if len(x) > 1 else 0.0  # noqa: E731
    d2p99 = {name: float(np.percentile(np.abs(np.diff(x, 2, axis=0)), 99)) for name, x in (("before", q0), ("after", qn))}
    return dict(
        min_sd_before=before["min_sd"], min_sd_after=after["min_sd"], pen_frac_before=before["pen_frac"], pen_frac_after=after["pen_frac"],
        pen_max_before=before["pen_max"], pen_max_after=after["pen_max"],
        viol_frac_before=before["viol_frac"], viol_frac_after=after["viol_frac"],
        per_body_min_before=before["per_body"], per_body_min_after=after["per_body"],
        joint_max_deg=float(np.degrees(np.abs(joint_delta).max())), joint_mean_deg=float(np.degrees(np.abs(joint_delta).mean())),
        root_max_cm=float(100 * np.abs(root_delta).max()), root_mean_cm=float(100 * np.linalg.norm(root_delta, axis=-1).mean()),
        root_rot_max_deg=float(np.degrees(rot_delta.max())),
        body_move_max_cm=float(100 * body_move.max()), body_move_mean_cm=float(100 * body_move.mean()), frames_moved_frac=float((body_move.max(-1) > 0.001).mean()),
        contact_frames=int(flag.sum()), **cons,
        foot_slip_before_cm=100 * slip["before"], foot_slip_after_cm=100 * slip["after"], foot_drift_cm=100 * foot_move,
        jerk_q_before=jerk(q0), jerk_q_after=jerk(qn), jerk_p_before=jerk(p0), jerk_p_after=jerk(pn),
        core_gap_ok_before=gap_ok["before"], core_gap_ok_after=gap_ok["after"], core_gap_p95_mm_after=gap_p95, core_gap_med_mm_after=gap_med,
        surface_change_frac=surf_change, **nat, joint_step_before=step(q0), joint_step_after=step(qn), backend=after.get("backend", "mesh"),
        core_slip_before_cm=core_slip["before"], core_slip_after_cm=core_slip["after"],
        osc_frac_before=osc["before"], osc_frac_after=osc["after"], d2q_p99_before=d2p99["before"], d2q_p99_after=d2p99["after"],
        limit_hit_before=hit(q0), limit_hit_after=hit(qn),
        selfcol_before=self_collision_frames(robot, p0, R0, q0), selfcol_after=self_collision_frames(robot, pn, Rn, qn),
    )  # fmt: skip


def failures(s: dict, tier: str | None = None) -> list[str]:
    """Acceptance test of one clip's numbers; the list of violated criteria (empty = accepted). ``tier``: None = everything;
    "A": everything, the stance slip must end <= max(half the source's, --accept-core-slip-cm); "B": the slip only not worse
    (+1 cm); "C": penetration and the physics gates only (the contact is best effort)."""
    a, out = ACCEPT, []
    if s["min_sd_after"] < -a["pen_tol"]:
        out.append(f"penetration {-1000 * s['min_sd_after']:.1f}mm")
    for k in ("joint_max_deg", "joint_mean_deg", "root_max_cm", "body_move_mean_cm", "body_move_max_cm", "foot_drift_cm"):
        if s[k] > a[k]:
            out.append(f"{k} {s[k]:.1f}>{a[k]}")
    if s["foot_slip_after_cm"] > s["foot_slip_before_cm"] + a["foot_slip_growth_cm"]:
        out.append(f"foot slip {s['foot_slip_before_cm']:.1f}->{s['foot_slip_after_cm']:.1f}cm")
    if s["jerk_q_after"] > a["jerk_ratio"] * s["jerk_q_before"] + a["jerk_q_abs"]:
        out.append(f"joint jerk {s['jerk_q_before']:.3f}->{s['jerk_q_after']:.3f}")
    if s["jerk_p_after"] > a["jerk_ratio"] * s["jerk_p_before"] + a["jerk_p_abs"]:
        out.append(f"root jerk {s['jerk_p_before']:.4f}->{s['jerk_p_after']:.4f}")
    if s["limit_hit_after"] > s["limit_hit_before"] + a["limit_hit_growth"]:
        out.append("joint limits")
    if s["selfcol_after"] > s["selfcol_before"] + a["selfcol_growth"]:
        out.append(f"self collision {s['selfcol_before']:.2f}->{s['selfcol_after']:.2f}")
    if tier != "C":
        if s["core_gap_ok_after"] < a["core_gap_ok_frac"]:
            out.append(f"core gap: {100 * s['core_gap_ok_after']:.1f}% of core frames within {1000 * a['core_gap_m']:.1f} mm < {100 * a['core_gap_ok_frac']:.0f}%")
        if s["surface_change_frac"] > a["surface_change_frac"]:
            out.append(f"contact surface changed on {100 * s['surface_change_frac']:.1f}% of the core frames")
    if s["joint_step_after"] > max(a["joint_step_rad"], s["joint_step_before"]):
        out.append(f"joint step {s['joint_step_after']:.2f} rad/frame")
    if tier == "B":
        if s["core_slip_after_cm"] > s["core_slip_before_cm"] + 1.0:
            out.append(f"core slip got worse {s['core_slip_before_cm']:.1f}->{s['core_slip_after_cm']:.1f}cm")
    elif tier != "C" and s["core_slip_after_cm"] > a["core_slip_cm"] and s["core_slip_after_cm"] > 0.5 * s["core_slip_before_cm"]:
        out.append(f"core slip {s['core_slip_before_cm']:.1f}->{s['core_slip_after_cm']:.1f}cm")
    if tier in ("A", "B", "C"):  # naturalness: small, smooth, human-like changes
        if s["root_rot_max_deg"] > a["root_rot_deg"]:
            out.append(f"root rotation {s['root_rot_max_deg']:.1f}deg")
        for nm in ("q", "p"):
            for o, lab in enumerate(("vel", "acc", "jerk")):
                for stat in ("p99", "max"):
                    bf, af = s[f"{lab}_{nm}_{stat}_before"], s[f"{lab}_{nm}_{stat}_after"]
                    if af > a["dyn_ratio"] * bf + DYN_FLOOR[nm][o]:
                        out.append(f"{lab}_{nm} {stat} {bf:.4f}->{af:.4f}")
        if s["hf_q_after"] > a["hf_ratio"] * s["hf_q_before"] + 1e-8:
            out.append(f"high-frequency energy {s['hf_q_before']:.2e}->{s['hf_q_after']:.2e}")
        if s["near_limit_after"] > s["near_limit_before"] + a["near_limit_growth"]:
            out.append(f"near-limit frames {100 * s['near_limit_before']:.1f}->{100 * s['near_limit_after']:.1f}%")
        if s["joint_corr_min"] < a["joint_corr_min"]:
            out.append(f"joint correlation with the source {s['joint_corr_min']:.3f}")
    if tier != "C" and s["contact_in_d_on_after"] < s["contact_in_d_on_before"] - a["contact_in_d_on_drop"]:
        out.append(f"contact in d_on {100 * s['contact_in_d_on_before']:.1f}->{100 * s['contact_in_d_on_after']:.1f}%")
    return out


def so3_angle(Ra: np.ndarray, Rb: np.ndarray) -> np.ndarray:
    return (Rotation.from_matrix(Ra).inv() * Rotation.from_matrix(Rb)).magnitude()


# ---------------------------------------------------------------------------------------------------- dataset io
def write_clip(clip: Clip, out_path: str, p, R, q, frames: np.ndarray | None = None):
    """Source fields with ``base_pos_w`` / ``base_quat_w`` / ``joint_pos`` replaced (same dtypes, quaternion sign as the source);
    with ``frames`` (bool (T,)) only those frames are replaced, the others stay byte-identical to the source."""
    out = dict(clip.src)
    quat = Rotation.from_matrix(R.astype(np.float64)).as_quat()[:, [3, 0, 1, 2]]
    quat *= np.where((quat * clip.src["base_quat_w"]).sum(-1, keepdims=True) < 0, -1.0, 1.0)
    jp = clip.src["joint_pos"].copy()
    jp[:, clip.cols] = q
    if frames is not None:
        quat = np.where(frames[:, None], quat, clip.src["base_quat_w"])
        p = np.where(frames[:, None], p, clip.src["base_pos_w"])
        jp = np.where(frames[:, None], jp, clip.src["joint_pos"])
    out.update(base_pos_w=p.astype(clip.src["base_pos_w"].dtype), base_quat_w=quat.astype(clip.src["base_quat_w"].dtype), joint_pos=jp.astype(clip.src["joint_pos"].dtype))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path, **out)


def select_clips(entries: list[dict], regex: str | None, clips_file: str | None) -> list[dict]:
    if regex:
        entries = [e for e in entries if re.search(regex, e["motion_file"])]
    if clips_file:
        tokens = set()
        with open(clips_file) as f:
            for row in csv.reader(f):
                tokens.update(c.strip() for c in row)
        stem = lambda s: os.path.basename(s)[: -len(".npz")] if s.endswith(".npz") else os.path.basename(s)  # noqa: E731
        entries = [e for e in entries if e["motion_file"] in tokens or stem(e["motion_file"]) in tokens or stem(e["motion_file"]).endswith(tuple(t for t in tokens if t and t.startswith("__")))]  # fmt: skip
    return entries


def batches(clips: list[Clip], dims: int, budget: float):
    clips = sorted(clips, key=lambda c: c.T)
    cur = []
    for c in clips:
        if cur and (len(cur) + 1) * c.T * dims > budget:
            yield cur
            cur = []
        cur.append(c)
    if cur:
        yield cur


def format_row(name: str, s: dict) -> str:
    return (
        f"{name[-52:]:52s} minSD {1000 * s['min_sd_before']:7.1f}->{1000 * s['min_sd_after']:6.1f} mm pen>1mm {100 * s['pen_frac_before']:4.1f}->{100 * s['pen_frac_after']:4.1f}% "
        f"| joint max/mean {s['joint_max_deg']:4.1f}/{s['joint_mean_deg']:4.2f} deg root {s['root_max_cm']:3.1f} cm body max/mean {s['body_move_max_cm']:4.1f}/{s['body_move_mean_cm']:3.2f} cm "
        f"| contact({s['contact_frames']}) d_on {100 * s['contact_in_d_on_before']:.1f}->{100 * s['contact_in_d_on_after']:.1f}% 1.5cm {100 * s['contact_in_1p5cm_before']:.1f}->{100 * s['contact_in_1p5cm_after']:.1f}% "
        f"| foot slip {s['foot_slip_before_cm']:.1f}->{s['foot_slip_after_cm']:.1f} drift {s['foot_drift_cm']:.1f} cm"
    )  # fmt: skip


# ---------------------------------------------------------------------------------------------------- window mode
def flag_frames(c: Clip, bf: dict, robot: Robot, args) -> tuple[np.ndarray, dict]:
    """(T,) bool frames of the source that need work, by the acceptance criteria, and the count per reason:
    penetration (< margin), core contact gap (> --need-contact-gap), a supporting foot sliding over its core run
    (> --window-slide-cm from the run's first frame), numerical jitter, joint limit violation, a joint jump (> --window-jump rad/frame)."""
    T = c.T
    why = {}
    m = bf["frame_min"] < args.margin
    why["pen"] = m
    sel = c.core if args.touch_labels == "core" else c.flag
    if args.need_contact_gap is not None:
        why["gap"] = (sel & (bf["contact_dist"] > args.need_contact_gap)).any(-1)
    slide = np.zeros(T, bool)
    for fi, name in enumerate(FOOT_BODIES):
        xy = bf["feet_xy"][fi]
        for s, e in runs(c.core[:, CONTACT_BODIES.index(name)]):
            if c.moving[s:e, CONTACT_BODIES.index(name)].mean() > 0.5:
                continue
            slide[s:e] |= np.linalg.norm(xy[s:e] - xy[s], axis=1) > args.window_slide_cm / 100
    why["slide"] = slide
    if args.de_jitter > 0:
        c.jit_pre = jitter_frames(c.q0, c.p0, args.de_jitter_win, args.de_jitter, args.de_jitter_floor_p)
        why["jitter"] = c.jit_pre.cpu().numpy()
    q0 = c.q0.cpu().numpy()
    lo, hi = robot.jrange[:, 0].cpu().numpy(), robot.jrange[:, 1].cpu().numpy()
    why["limit"] = ((q0 < lo - 1e-4) | (q0 > hi + 1e-4)).any(-1)
    jump = np.zeros(T, bool)
    if T > 1:
        j = np.abs(np.diff(q0, axis=0)).max(-1) > args.window_jump
        jump[:-1] |= j
        jump[1:] |= j
    why["jump"] = jump
    mask = np.zeros(T, bool)
    for v in why.values():
        mask |= v
    return mask, {k: int(v.sum()) for k, v in why.items() if v.any()}


def needs_work(c: Clip, bf: dict, robot: Robot, args) -> bool:
    """Whole-clip mode: does the clip need refinement? penetration, core contact gap, and (when the corresponding terms are on)
    a sliding stance foot or numerical jitter - a clip with only those problems used to be copied untouched."""
    if getattr(args, "pen_only", False):  # only a source that penetrates (continuous minSD < 0) is touched
        return bool(bf["min_sd"] < 0.0)
    _, why = flag_frames(c, bf, robot, args)
    keys = ["pen", "gap"] + (["slide"] if args.w_noslide > 0 else []) + (["jitter"] if args.de_jitter > 0 else [])
    return any(k in why for k in keys)


def make_windows(c: Clip, mask: np.ndarray, args) -> list[tuple[int, int]]:
    """Flagged frames -> windows: dilated by --window-pad s, extended over every core contact run they touch (the no-slide anchors
    are the run's first frame), neighbours closer than twice the anchor length merged."""
    pad_f = max(int(round(args.window_pad * c.fps)), 1)
    if not mask.any():
        return []
    m = np.convolve(mask.astype(np.float32), np.ones(2 * pad_f + 1, np.float32), "same") > 0
    if args.window_extend_runs:
        core_runs = [r for b in range(c.core.shape[1]) for r in runs(c.core[:, b])]
        while True:
            before = m.copy()
            for s, e in core_runs:
                if m[s:e].any():
                    m[s:e] = True
            if (m == before).all():
                break
    wins = [list(r) for r in runs(m)]
    merged = [wins[0]]
    for s, e in wins[1:]:
        if s - merged[-1][1] <= 2 * args.window_anchor:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return [(int(s), int(e)) for s, e in merged]


def window_mode(args, robot: Robot, terrains: Terrains, clips: list[Clip], root: str, ref_root: str, tgeoms: dict, t0: float) -> dict:
    """Refine only the windows around the frames that need it, all windows of all clips packed into shared batches."""
    dev = args.device

    def analytics_of(wclips):
        if not args.analytic_terrain:
            return None
        paths = sorted({c.terrain_path for c in wclips})
        an = {p: AnalyticTerrain(yaml.safe_load(open(tgeoms[p])), dev) for p in paths if p in tgeoms}
        assert len(an) == len(paths), "every terrain of the batch needs a geoms.yaml for --analytic-terrain"
        return an

    # 1. detection on the whole clips (one measurement, as the whole-clip mode does for its `before`)
    before, wins = {}, {}
    for bc in batches(clips, len(robot.pts), args.budget):
        b = Batch(bc, robot, terrains, dev)
        for c, m in zip(bc, measure(b, *b.pose())):
            before[c.rel] = m
            mask, why = flag_frames(c, m, robot, args)
            c.flag_why, c.win = why, make_windows(c, mask, args)
        terrains.evict(set())
    nw = sum(len(c.win) for c in clips)
    print(f"window mode: {sum(bool(c.win) for c in clips)} / {len(clips)} clips need work, {nw} windows, "
          f"{sum(e - s for c in clips for s, e in c.win)} / {sum(c.T for c in clips)} frames  [{time.time() - t0:.0f} s]", flush=True)

    # 2. optimization of all windows (touch first; the no-touch fallback only for the clips whose touch result fails the acceptance)
    def optimize(cs: list[Clip], touch: bool) -> dict:
        wl = [(c, c.window(s, e, args.window_anchor)) for c in cs for s, e in c.win]
        res = {c.rel: [] for c in cs}
        by_id = {id(w): c for c, w in wl}
        for wb in batches([w for _, w in wl], len(robot.pts), args.budget):
            an = analytics_of(wb)
            b = Batch(wb, robot, terrains, dev, analytics=an) if an else Batch(wb, robot, terrains, dev)
            p, R, q = refine_batch(b, args, touch)
            for k, w in enumerate(wb):
                res[by_id[id(w)].rel].append((w, p[k, : w.T], R[k, : w.T], q[k, : w.T]))
            print(f"  [{'touch' if touch else 'notouch'}: {sum(len(v) for v in res.values())}/{len(wl)} windows, {time.time() - t0:.0f} s, "
                  f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB]", flush=True)
        return res

    def stitch(c: Clip, parts) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        p, R, q = c.p0.clone(), c.R0.clone(), c.q0.clone()
        frames = np.zeros(c.T, bool)
        for w, pw, Rw, qw in parts:
            a, e = w.win_range
            p[a:e], R[a:e], q[a:e] = pw, Rw, qw
            frames[a:e] = True
        pn, Rn, qn = p.cpu().numpy(), R.cpu().numpy(), q.cpu().numpy()
        quat = Rotation.from_matrix(Rn[frames].astype(np.float64)).as_quat()[:, [3, 0, 1, 2]]  # judge what ends up on disk
        Rn[frames] = quat_to_mat(torch.tensor(quat, dtype=torch.float32)).numpy()
        Rn[~frames] = c.R0.cpu().numpy()[~frames]
        return pn, Rn, qn, frames

    def judge(c: Clip, parts):
        pn, Rn, qn, frames = stitch(c, parts)
        after = measure(Batch([c], robot, terrains, dev), *(torch.tensor(x, device=dev)[None] for x in (pn, Rn, qn)))[0]
        st = compare(c, robot, before[c.rel], after, pn, Rn, qn)
        return st, failures(st), (pn, Rn, qn, frames)

    todo = [c for c in clips if c.win]
    cands = {c.rel: {} for c in todo}
    for mode in (["touch"] if args.touch else []) + ["notouch"]:
        cs = todo if mode == "touch" or not args.touch else [c for c in todo if cands[c.rel]["touch"][1]]
        by_rel = {c.rel: c for c in cs}
        for rel, parts in (optimize(cs, mode == "touch") if cs else {}).items():
            cands[rel][mode] = judge(by_rel[rel], parts)

    # 3. outputs (same candidate rule as the whole-clip mode: touch accepted -> notouch accepted -> the source, marked failed)
    report = {}
    for c in clips:
        if not c.win:
            mode, accepted = "clean", True
            pn, Rn, qn = c.p0.cpu().numpy(), c.R0.cpu().numpy(), c.q0.cpu().numpy()
            st = compare(c, robot, before[c.rel], before[c.rel], pn, Rn, qn)
            fails_t, fails_n, frames = [], [], np.zeros(c.T, bool)
        else:
            cd = cands[c.rel]
            fails_t, fails_n = (cd["touch"][1] if "touch" in cd else []), (cd["notouch"][1] if "notouch" in cd else None)
            if "touch" in cd and not fails_t:
                mode, accepted, key = "touch", True, "touch"
            elif "notouch" in cd and not fails_n:
                mode, accepted, key = "notouch", True, "notouch"
            else:
                mode, accepted, key = "failed", False, ("notouch" if "notouch" in cd else "touch")
            st, _, (pn, Rn, qn, frames) = cd[key]
            st["touch_candidate"] = {k: v for k, v in cd["touch"][0].items() if k in ("joint_max_deg", "joint_mean_deg", "body_move_max_cm", "body_move_mean_cm", "foot_drift_cm", "min_sd_after")} if "touch" in cd else None  # fmt: skip
        st["refined"], st["mode"], st["accepted"] = bool(c.win), mode, accepted
        st["accepted_mode"] = mode if accepted else None
        st["failure_reasons"] = [f"touch: {r}" for r in fails_t] + [f"notouch: {r}" for r in (fails_n or [])]
        st["reason"] = ("; ".join(fails_t) if mode in ("notouch", "failed") else "") + (" | notouch also fails: " + "; ".join(fails_n) if mode in ("notouch", "failed") and fails_n else "")
        st["windows"], st["window_frames"], st["frames"], st["flag_reasons"] = [list(w) for w in c.win], int(frames.sum()), c.T, c.flag_why
        publish(args, args._prev, root, ref_root, c, st, pn, Rn, qn, frames)
        report[c.rel] = st
        print(format_row(c.rel, st) + f"  [{mode}{': ' + st['reason'] if st['reason'] else ''}; {len(c.win)} windows, {int(frames.sum())}/{c.T} frames]", flush=True)
    return report


def _sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def publish(args, prev: dict, root: str, ref_root: str, c: Clip, s: dict, pn, Rn, qn, frames: np.ndarray | None = None) -> None:
    """Write one clip's result: the accepted candidate, else the SOURCE motion (rejected candidate under ``_rejected/`` for diagnosis).
    A re-run never replaces an earlier accepted result of the same output folder by a non-accepted one (``prev`` = the folder's report)."""
    dst = os.path.join(ref_root, c.rel)
    side = c.rel[: -len(".npz")] + ".contacts.npz"
    old = prev.get(c.rel)
    if old and old.get("accepted") and not s["accepted"]:
        s.clear(); s.update(old); s["kept_previous"] = True  # noqa: E702  the earlier accepted entry stays (file untouched)
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(os.path.join(root, side)):  # labels stay those of the source (not relabelled)
        shutil.copyfile(os.path.join(root, side), os.path.join(ref_root, side))
    if s["mode"] in ("touch", "notouch", "tierA", "tierB", "tierC"):
        write_clip(c, dst, pn, Rn, qn, frames)
        s["output_is_source"] = False
    else:
        if s["mode"] == "failed" and pn is not None:
            write_clip(c, os.path.join(ref_root, "_rejected", c.rel), pn, Rn, qn, frames)
        if s["mode"] == "failed" and not s.get("source_pen_free", True):
            s["no_output"] = True  # no candidate passes and the source itself penetrates: nothing is published (report only)
            s["output_is_source"] = False
            return
        shutil.copyfile(os.path.join(root, c.rel), dst)
        s["output_is_source"] = True
    s["src_sha"], s["out_sha"] = _sha(os.path.join(root, c.rel)), _sha(dst)


def merge_manifest(args, meta: dict, root: str, ref_root: str, prev: dict, report: dict) -> dict:
    """Write metadata.yaml / terrains of ``ref_root`` for the UNION of the earlier report and this run (a subset re-run into the same
    folder keeps every earlier clip) and return the merged report."""
    merged = {**prev, **report}
    os.makedirs(ref_root, exist_ok=True)
    m2 = dict(meta)
    m2["motion_files"] = [e for e in meta["motion_files"] if e["motion_file"] in merged and not merged[e["motion_file"]].get("no_output")]
    used = {e["terrain_id"] for e in m2["motion_files"]}
    m2["terrains"] = [t for t in meta["terrains"] if t["terrain_id"] in used]
    yaml.safe_dump(m2, open(os.path.join(ref_root, "metadata.yaml"), "w"), sort_keys=False)
    for t in m2["terrains"]:
        for key in ("terrain_file", "terrain_geoms"):
            if key in t:
                os.makedirs(os.path.dirname(os.path.join(ref_root, t[key])), exist_ok=True)
                shutil.copyfile(os.path.join(root, t[key]), os.path.join(ref_root, t[key]))
    return merged


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", required=True, help="source folder with metadata.yaml (never modified)")
    ap.add_argument("--out", default=None, help="output folder (same layout; only the processed clips)")
    ap.add_argument("--check", default=None, help="acceptance report only: compare this refined folder with --dataset")
    ap.add_argument("--clips", default=None, help="regex on the motion path (relative to the dataset)")
    ap.add_argument("--clips-file", default=None, help="csv; any cell equal to a motion path / clip stem selects the clip")
    ap.add_argument("--margin", type=float, default=MARGIN, help="required signed distance of every check point [m]")
    ap.add_argument("--joint-margin", type=float, default=0.0,
                    help="rad; the optimized joints are clamped this far inside the MJCF ranges (0 = clamp to the ranges themselves)")
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--early-stop", type=int, default=0,
                    help="stop the optimization when the loss has improved by less than 0.1% for this many 25-iteration checks (0 = never)")
    ap.add_argument("--w-clear", type=float, default=3000.0)
    ap.add_argument("--w-smooth", type=float, default=30.0)
    ap.add_argument("--w-keep", type=float, default=1000.0)
    ap.add_argument("--w-foot", type=float, default=100.0)
    ap.add_argument("--touch", action=argparse.BooleanOptionalAction, default=True,
                    help="try the upstream touch term first (contact bodies pulled to --touch-target), per clip fall back to no-touch when the result fails the acceptance")
    ap.add_argument("--touch-labels", choices=("flag", "core"), default="flag",
                    help="which labelled frames the touch term pulls: all contact flags (upstream behaviour) or only the core frames "
                    "(real contact: entered below d_on and static, or a sliding touch; the hysteresis tail d_on..d_off is left alone - "
                    "core_tail_masks above)")
    ap.add_argument("--touch-target", type=float, default=0.001)
    ap.add_argument("--touch-range", type=float, default=None,
                    help="m; contact frames whose body is within this distance of the terrain in the source are pulled to the touch target "
                    "(default: the labeler's per-body d_on, feet 3 cm / others 1.5 cm)")
    ap.add_argument("--need-contact-gap", type=float, default=None,
                    help="m; refine clips that have a labelled contact frame with a body min distance above this (in addition to min_sd < --margin)")
    ap.add_argument("--track-scale", default=None,
                    help="comma list 'regex=scale': the body-frame tracking weight of bodies whose name matches is scaled (e.g. 'wrist|hand=0.3'); "
                    "lets the touch term move those bodies further from the source")
    ap.add_argument("--w-noslide", type=float, default=0.0,
                    help="weight of the no-slide term (0 = off): in every core contact run the material points of the first touching "
                    "slice keep their horizontal world position (a rolling foot pivots over them, a skating foot is dragged back)")
    ap.add_argument("--noslide-tol", type=float, default=0.02,
                    help="m; drift of a stance point from its anchor up to this is not penalized (only obvious slides are fixed)")
    ap.add_argument("--tiers", action=argparse.BooleanOptionalAction, default=True,
                    help="per clip, stepwise: tier A = touch + no-penetration + no-slide; B = touch + no-penetration (slide not worse); "
                    "C = no-touch, no-penetration only; a later tier runs only for the clips the previous one failed. Off: the --lazy-notouch "
                    "touch / notouch logic")
    ap.add_argument("--pen-only", action="store_true",
                    help="only repair penetration: a clip whose continuous minSD is >= 0 is copied untouched, one with minSD < 0 gets "
                    "tier C (no-touch, minSD >= 0, physics gates); other criteria (gap, slide, jitter) are ignored. Implies --tiers")
    ap.add_argument("--noslide-bodies", default="foot|hand|knee|pelvis|elbow", help="regex on the contact body names the no-slide term applies to")
    ap.add_argument("--de-jitter", type=float, default=0.0,
                    help="rad/frame^2 floor of the numerical-jitter detector (0 = off): alternating-sign second differences of the joints; "
                    "on those frames tracking relaxes (--de-jitter-track) and the third difference of the output is penalized (--w-jitter)")
    ap.add_argument("--de-jitter-floor-p", type=float, default=0.002, help="m/frame^2 floor of the jitter detector for the root position")
    ap.add_argument("--de-jitter-win", type=int, default=4, help="min frames of alternating second differences = one jitter window")
    ap.add_argument("--de-jitter-track", type=float, default=0.2, help="tracking weight reduction on the jitter frames (0.2 = 80% kept)")
    ap.add_argument("--w-jitter", type=float, default=300.0, help="weight of the third-difference penalty on the jitter frames")
    ap.add_argument("--accept-override", default=None,
                    help="comma list 'key=value' raising individual acceptance thresholds (e.g. 'joint_max_deg=45,root_max_cm=30' for "
                    "slide fixes); physics gates - penetration, joint limits, jerk - should not be relaxed")
    ap.add_argument("--touch-margin", type=float, default=0.0005, help="clearance margin in touch mode")
    ap.add_argument("--slack", type=float, default=0.0015, help="extra clearance of the optimization over --margin (thinned check points miss up to a few mm between samples)")
    ap.add_argument("--w-touch", type=float, default=1000.0)
    ap.add_argument("--keep-slack", type=float, default=KEEP_SLACK)
    ap.add_argument("--spacing", type=float, default=0.03, help="check point spacing on the faces of the collision hulls [m]")
    ap.add_argument("--voxel", type=float, default=0.01, help="check points are thinned to one per voxel of this size per link [m]")
    ap.add_argument("--voxel-contact", type=float, default=0.008, help="same for the nine contact links (their min distance enters the keep / touch terms)")
    ap.add_argument("--budget", type=float, default=6e6, help="max clips * frames * points per batch")
    ap.add_argument("--simplified-collision", action="store_true")
    ap.add_argument("--analytic-terrain", action="store_true",
                    help="optimize against the exact analytic SDF of the geoms.yaml primitives (env collision truth) instead of the "
                    "warp BVH query on the STL - ~100x faster on long clips, same distances; the verification keeps the mesh path")
    ap.add_argument("--window-mode", action="store_true",
                    help="optimize only windows around the frames that need work (penetration, core contact gap, foot sliding, jitter, joint "
                    "limits, jumps); the rest of every clip stays at the source and anchors the window ends; all windows share batches")
    ap.add_argument("--window-pad", type=float, default=0.4, help="s kept free before / after every flagged frame")
    ap.add_argument("--window-anchor", type=int, default=6, help="fixed source frames on either side of a window (smoothness / continuity anchors)")
    ap.add_argument("--window-slide-cm", type=float, default=1.0, help="a foot sliding more than this over its core run flags the run")
    ap.add_argument("--window-jump", type=float, default=0.35, help="rad/frame joint jump that flags the two frames")
    ap.add_argument("--window-extend-runs", action=argparse.BooleanOptionalAction, default=True,
                    help="extend windows over the whole core contact runs they touch (the no-slide anchors are the run's first frame)")
    ap.add_argument("--lazy-notouch", action=argparse.BooleanOptionalAction, default=True,
                    help="run the no-touch fallback only for the clips whose touch result fails the acceptance, on a batch of just those "
                    "(--no-lazy-notouch: the old behaviour, both modes on every batch)")
    ap.add_argument("--fresh", action="store_true",
                    help="do not merge with an existing refine_report.json / metadata.yaml of --out (default: a re-run of a subset into the same "
                    "folder KEEPS the clips of earlier runs and never replaces an accepted result by a non-accepted one)")
    ap.add_argument("--no-continuous", action="store_true",
                    help="verify against the dense surface samples / STL instead of the exact capsule / sphere - primitive distances")
    ap.add_argument("--accept-core-slip-cm", type=float, default=4.0,
                    help="with --w-noslide: a stance foot may slip at most this far over a static core run (or half the source's)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--report", default=None, help="json path of the per-clip numbers (default <out>/refine_report.json)")
    args = ap.parse_args()
    assert args.out or args.check
    if args.pen_only:
        args.tiers = True
    if args.need_contact_gap is not None:
        ACCEPT["core_gap_m"] = args.need_contact_gap + 1e-4  # the labelled target (+0.1 mm: the sampling of a rounded value)
    ACCEPT["touch_range_m"] = args.touch_range if args.touch_range is not None else ACCEPT["touch_range_m"]
    if args.w_noslide > 0:
        ACCEPT["core_slip_cm"] = args.accept_core_slip_cm
    apply_accept_overrides(args.accept_override)
    if args.check:
        args.out = None  # report only, nothing is written but the report next to the refined clips
    root = os.path.abspath(os.path.expanduser(args.dataset))
    ref_root = os.path.abspath(os.path.expanduser(args.out or args.check))
    assert ref_root != root, "--out must differ from --dataset"
    meta = yaml.safe_load(open(os.path.join(root, "metadata.yaml")))
    tfile = {t["terrain_id"]: os.path.join(root, t["terrain_file"]) for t in meta["terrains"]}
    entries = select_clips(meta["motion_files"], args.clips, args.clips_file)
    print(f"{len(entries)} / {len(meta['motion_files'])} clips selected")
    robot = Robot(args.device, args.simplified_collision, args.spacing, args.voxel, args.voxel_contact, args.joint_margin)
    terrains = Terrains(args.device)
    print(f"{len(robot.pts)} check points on {len(robot.collision_bodies)} links")
    clips = [Clip(root, e["motion_file"], tfile[e["terrain_id"]], robot, args.device) for e in entries]
    if args.check:  # sidecars come from the source; the refined poses from --check
        pass
    report, t0, done = {}, time.time(), []
    prev = {}
    if args.out and not args.fresh and os.path.exists(os.path.join(ref_root, "refine_report.json")):
        prev = json.load(open(os.path.join(ref_root, "refine_report.json")))
        print(f"merging into the existing {ref_root} ({len(prev)} clips in its report)")
    args._prev = prev
    if not args.no_continuous and robot.csd.ok:
        for t in meta["terrains"]:
            if "terrain_geoms" in t:
                CONT_TERRAINS[os.path.join(root, t["terrain_file"])] = AnalyticTerrain(yaml.safe_load(open(os.path.join(root, t["terrain_geoms"]))), args.device)
        print(f"verification: continuous capsule/sphere distances to {len(CONT_TERRAINS)} geoms.yaml terrains")
    if args.window_mode:
        assert args.out and not args.check, "--window-mode refines (--out)"
        tgeoms = {os.path.join(root, t["terrain_file"]): os.path.join(root, t["terrain_geoms"]) for t in meta["terrains"] if "terrain_geoms" in t}
        report = window_mode(args, robot, terrains, clips, root, ref_root, tgeoms, t0)
        clips = []
    modes = (["touch"] if args.touch else []) + ["notouch"]
    tgeoms = {os.path.join(root, t["terrain_file"]): os.path.join(root, t["terrain_geoms"]) for t in meta["terrains"] if "terrain_geoms" in t}
    for batch_clips in batches(clips, len(robot.pts), args.budget):
        b = Batch(batch_clips, robot, terrains, args.device)
        analytics = None
        if args.analytic_terrain:
            paths = sorted({c.terrain_path for c in batch_clips})
            analytics = {p: AnalyticTerrain(yaml.safe_load(open(tgeoms[p])), args.device) for p in paths if p in tgeoms}
            assert len(analytics) == len(paths), "every terrain of the batch needs a geoms.yaml for --analytic-terrain"
        b_opt = Batch(batch_clips, robot, terrains, args.device, analytics=analytics) if analytics else b
        before = measure(b, *b.pose())
        sel_labels = [c.core if args.touch_labels == "core" else c.flag for c in b.clips]
        need = [needs_work(c, m, robot, args) for c, m in zip(batch_clips, before)]
        res = {}  # (mode, clip index) -> (p, R, q) of that clip, (T, ...) device tensors

        def optimize(mode: str, idxs: list[int]):
            """Run ``mode`` on the batch of clips ``idxs`` (the whole batch = the unchanged batch composition of the old path)."""
            if len(idxs) == len(batch_clips):
                bb, bo = b, b_opt
            else:
                sub = [batch_clips[k] for k in idxs]
                bb = Batch(sub, robot, terrains, args.device)
                bo = Batch(sub, robot, terrains, args.device, analytics=analytics) if analytics else bb
            nd = [need[k] for k in idxs]
            if any(nd):
                p, R, q = refine_batch(bo, args, mode in ("touch", "A", "B"), noslide=mode not in ("B", "C"))
                keep = torch.tensor([not n for n in nd], device=args.device)
                p, R, q = (torch.where(keep.view(-1, *[1] * (x.dim() - 1)), x0, x) for x0, x in zip((bb.p0, bb.R0, bb.q0), (p, R, q)))
            else:
                p, R, q = bb.p0, bb.R0, bb.q0
            for j, k in enumerate(idxs):
                res[(mode, k)] = (p[j], R[j], q[j])

        def evaluate(mode: str, k: int):
            """(stats, failures, (pn, Rn, qn)) of one clip's candidate; the on-disk (float32 quaternion) trajectory is measured once."""
            c = batch_clips[k]
            p, R, q = res[(mode, k)]
            pn, Rn, qn = p[: c.T].cpu().numpy(), R[: c.T].cpu().numpy(), q[: c.T].cpu().numpy()
            if args.check:
                after = checked_after[k]
            elif need[k]:  # judge what ends up on disk (float32 round trip through the quaternion)
                quat = Rotation.from_matrix(Rn.astype(np.float64)).as_quat()[:, [3, 0, 1, 2]]
                Rn = quat_to_mat(torch.tensor(quat, dtype=torch.float32)).numpy()
                after = measure(Batch([c], robot, terrains, args.device), *(torch.tensor(x, device=args.device)[None] for x in (pn, Rn, qn)))[0]
            else:
                after = before[k]
            st = compare(c, robot, before[k], after, pn, Rn, qn)
            return st, (failures(st, mode if mode in ("A", "B", "C") else None) if need[k] or args.check else []), (pn, Rn, qn)

        sts_all = {}
        if args.check:
            refined = [Clip(ref_root, c.rel, c.terrain_path, robot, args.device) for c in batch_clips]
            for c, r in zip(batch_clips, refined):
                r.flag = c.flag
            nb = Batch(refined, robot, terrains, args.device)
            checked_after = measure(nb, nb.p0, nb.R0, nb.q0)
            for k in range(len(batch_clips)):
                res[("checked", k)] = (nb.p0[k], nb.R0[k], nb.q0[k])
            todo_modes = ["checked"]
        else:
            todo_modes = modes
            if args.tiers:  # A: touch + no-slide, B: touch, C: no-touch; each tier only for the clips the previous one failed
                tiers = ["C"] if args.pen_only else (["A"] if args.touch and args.w_noslide > 0 else []) + (["B"] if args.touch else []) + ["C"]
                todo = [k for k in range(len(batch_clips)) if need[k]]
                for tier in tiers:
                    if not todo:
                        break
                    optimize(tier, list(range(len(batch_clips))) if tier == tiers[0] else todo)
                    for k in todo:
                        sts_all[(tier, k)] = evaluate(tier, k)
                    todo = [k for k in todo if sts_all[(tier, k)][1]]
            elif args.lazy_notouch and args.touch:  # touch for the whole batch; notouch only for the clips whose touch fails the acceptance
                optimize("touch", list(range(len(batch_clips))))
                sts_all.update({("touch", k): evaluate("touch", k) for k in range(len(batch_clips))})
                redo = [k for k in range(len(batch_clips)) if need[k] and sts_all[("touch", k)][1]]
                if redo:
                    optimize("notouch", redo)
                    sts_all.update({("notouch", k): evaluate("notouch", k) for k in redo})
            else:
                for mode in modes:
                    optimize(mode, list(range(len(batch_clips))))
        for k, c in enumerate(batch_clips):
            Tn = c.T
            sts, fails = {}, {}
            for mode in todo_modes:
                if (mode, k) not in sts_all and (mode, k) in res:
                    sts_all[(mode, k)] = evaluate(mode, k)
            order = ("checked", "A", "B", "C", "touch", "notouch")
            for mode in order:
                if (mode, k) in sts_all:
                    st, fl, arrs = sts_all[(mode, k)]
                    sts[mode], fails[mode] = (st, arrs), fl
            tier_names = [m for m in ("A", "B", "C") if m in sts]
            if args.check:
                mode, accepted = "checked", not fails["checked"]
            elif not need[k]:
                mode, accepted = "clean", True
                if not sts:  # tiers: nothing was optimized, the source is the output
                    p0, R0, q0 = c.p0.cpu().numpy(), c.R0.cpu().numpy(), c.q0.cpu().numpy()
                    sts["clean"], fails["clean"] = (compare(c, robot, before[k], before[k], p0, R0, q0), (p0, R0, q0)), []
            else:
                mode, accepted = "failed", False
                for cand in tier_names + [m for m in ("touch", "notouch") if m in sts]:  # first passing tier / candidate wins
                    if not fails[cand]:
                        mode, accepted = (f"tier{cand}" if cand in ("A", "B", "C") else cand), True
                        break
            if mode == "clean":
                key = "clean"
            elif accepted:
                key = mode[-1] if mode.startswith("tier") else mode
            else:  # failed: the numbers of the last (loosest) candidate
                key = (tier_names or [m for m in ("touch", "notouch") if m in sts])[-1]
            s, (pn, Rn, qn) = sts[key]
            s["refined"], s["mode"], s["accepted"] = bool(need[k]), mode, accepted
            s["accepted_mode"] = mode if accepted else None
            s["tier"] = key if key in ("A", "B", "C") else None
            s["tier_failures"] = {m: fails[m] for m in tier_names}
            s["failure_reasons"] = [f"{m}: {r}" for m in ("A", "B", "C", "touch", "notouch", "checked") for r in fails.get(m, [])]
            s["reason"] = " | ".join(f"{m}: {'; '.join(fails[m])}" for m in ("A", "B", "C", "touch", "notouch") if fails.get(m))
            s["source_pen_free"] = bool(before[k]["min_sd"] >= 0.0)
            s["touch_candidate"] = None
            if not args.check:
                publish(args, prev, root, ref_root, c, s, pn, Rn, qn)
            report[c.rel] = s
            print(format_row(c.rel, s) + f"  [{mode}{': ' + s['reason'] if s['reason'] else ''}]", flush=True)
        terrains.evict({c.terrain_path for c in batch_clips} - {c.terrain_path for c in batch_clips})
        done.append(len(batch_clips))
        print(f"  [{sum(done)}/{len(clips)} clips, {time.time() - t0:.0f} s, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB]", flush=True)
    merged = merge_manifest(args, meta, root, ref_root, prev, report) if args.out else report
    path = args.report or os.path.join(ref_root, "refine_report.json" if args.out else "check_report.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(merged, open(path, "w"), indent=1)
    cols = ["mode", "tier", "accepted", "reason", "min_sd_before", "min_sd_after", "pen_frac_before", "pen_frac_after", "joint_max_deg", "joint_mean_deg", "root_max_cm", "body_move_max_cm",
            "body_move_mean_cm", "foot_slip_before_cm", "foot_slip_after_cm", "foot_drift_cm", "core_slip_before_cm", "core_slip_after_cm",
            "osc_frac_before", "osc_frac_after", "d2q_p99_before", "d2q_p99_after", "jerk_q_before", "jerk_q_after", "contact_frames",
            "contact_in_d_on_before", "contact_in_d_on_after", "contact_in_1p5cm_before", "contact_in_1p5cm_after", "selfcol_before", "selfcol_after",
            "core_gap_ok_before", "core_gap_ok_after", "core_gap_med_mm_after", "core_gap_p95_mm_after", "surface_change_frac", "joint_step_after", "backend"]
    with open(os.path.splitext(path)[0].replace("_report", "") + ".csv" if args.report else os.path.join(ref_root, "refine_report.csv" if args.out else "check_report.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["clip"] + cols)
        for name, st in merged.items():
            w.writerow([name] + [round(st.get(c, ""), 5) if isinstance(st.get(c), float) else st.get(c, "") for c in cols])
    r = list(report.values())
    ref = [s for s in r if s["refined"]]
    print(f"\n{len(r)} clips, {len(ref)} needed refinement ({time.time() - t0:.0f} s); report -> {path}")
    if ref:
        print(f"  min SD worst: {1000 * min(s['min_sd_before'] for s in ref):.1f} -> {1000 * min(s['min_sd_after'] for s in ref):.1f} mm; "
              f"clips with pen>1mm frames after: {sum(s['pen_frac_after'] > 0 for s in ref)}; "
              f"joint max {max(s['joint_max_deg'] for s in ref):.1f} deg, body max {max(s['body_move_max_cm'] for s in ref):.1f} cm; "
              f"contact in d_on min {100 * min(s['contact_in_d_on_after'] for s in ref):.1f}%; modes: "
              + ", ".join(f"{m} {sum(s['mode'] == m for s in r)}" for m in ("tierA", "tierB", "tierC", "touch", "notouch", "failed", "clean", "checked")))  # fmt: skip
    bad = [k for k, v in report.items() if not v.get("accepted", True)]
    if bad and args.out:
        print(f"  {len(bad)} clips NOT accepted (output = source motion): " + ", ".join(bad[:10]) + (" ..." if len(bad) > 10 else ""))


if __name__ == "__main__":
    main()
