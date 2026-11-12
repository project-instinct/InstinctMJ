"""Label per-frame body-environment contacts of the retargeted G1 reference motions for ACP.

    python -m instinct_mj.scripts.acp.label_contacts --dataset ~/Datasets/20251116_50cm_kneeClimbStep1 [--overwrite] \
        [--viz 50cm_kneeClimbStep/jumpsit2-ziwen-retargeted.npz]

ACP (arXiv 2610.31608) conditions the policy on the active contact sequence and contact locations (world frame) of nine bodies
(``CONTACT_BODIES`` / ``CONTACT_LINKS`` of ``instinct_mj.rl.acp.planner``). For every motion ``<stem>.npz`` of the TerrainMotion
dataset (``metadata.yaml``: ``motion_files`` -> ``terrain_id`` -> ``terrains[].terrain_file`` STL) a sidecar ``<stem>.contacts.npz``
is written.

Method (purely kinematic, no simulation): a body is in contact when it is close to the terrain (distance) AND the contact can
support the robot (physical reasoning, no force computation): the environment can only push through a contact that sticks, so the
touching material points of a body that is not in touch (within the hover tolerance of retargeted motions) must be at rest
relative to the (static) terrain - a body that hovers past the surface carries no load, a body that touches it does (sliding
contacts still carry a normal force).
  1. Replay: qpos = [base_pos_w, base_quat_w (wxyz), joint_pos (mapped by joint name)] on the MJCF used by the tasks
     (``instinct_mj.assets.acp_g1.get_acp_g1_spec()``: the ACP playground G1, free root = pelvis, 23 joints, the authors' collision
     meshes; the task simulates the same geometry, so labels and simulation measure one surface - use ``--simplified-collision`` only
     for the analytic capsule / sphere model of COLLISION_SIMPLIFY.md, with sidecars relabelled the same way), ``mj_kinematics`` per
     frame.
  2. Contact points of a body = the collision geoms of that link only (contype/conaffinity != 0): the playground G1's collision meshes
     are sampled by their convex hull vertices (half-sphere hand); with ``--simplified-collision`` the analytic capsule / sphere
     surfaces of the same links are sampled instead. Points are
     transformed to the world frame at each frame. The paper detects contacts on the links' collision geometry (the playground policy
     outputs one 3-D point per link, a 9 x 3 ``planner_contacts``), so a contact body is exactly one link: ``pelvis`` is the pelvis
     link, ``*_knee_link`` the knee link. Neighbour links are not folded in: the hip motors (``*_hip_pitch_link`` /
     ``*_hip_roll_link``, 5-9 cm below the pelvis mesh when the G1 sits) and the thigh (``*_hip_yaw_link``) are no contact body, so
     such supports only appear when that link itself comes within ``d_on`` or through the support completion.
  3. The terrain STL is NOT put into MuJoCo (MuJoCo convexifies meshes for collision). Its exact point-to-triangle distance
     (closest point, closest triangle) is computed with a warp BVH on the CPU (``TerrainDistance``); the distance is unsigned (the STL
     is not watertight), which is fine for contacts since a body that penetrates the terrain only slightly is within ``d_on`` anyway.
  4. Per body: ``min_dist`` = min over its points; contact position = mean over the points within ``d_on`` (else the closest point)
     projected onto the terrain surface (closest point); contact normal (``contact_normal``) = direction of separation when the body is
     clear of the terrain (> ``D_NORMAL``), else the outward normal of the closest triangle (also when the body sank into it).
     Contact speed = speed of the touching material points (points within ``d_off``, else the closest point), central finite
     difference of the replayed points; the 20th percentile over the touching points is used, so a foot rolling over its toe
     (heel moving, toe at rest) still sticks.
  5. ``contact_segments``: distance hysteresis (enter when min_dist < d_on, stay while <= d_off); frames where the contact moves
     (speed >= v_max for at least ``move_frames`` consecutive frames: shorter bursts are finite-difference noise) are dropped
     unless the body touches (min_dist < d_touch: a foot skidding into a take-off still pushes on the ground) within
     ``skid_frames`` of a static frame of the same segment (friction brings a sliding contact to rest; a foot dragged through
     the floor by a defective clip never sticks and is not a support);
     optionally contacts whose terrain normal has z < ``min_normal_z`` (cannot carry weight); then segments shorter than
     ``min_duration`` seconds are dropped (paper: shorter than ten 50 Hz steps; feet and hands: ``min_duration_foot`` /
     ``min_duration_hand``, a hand planted on a box in a vault touches for ~0.1 s). Feet use their own d_on / d_off
     (some retargeted stance feet hover 1-2 cm; steps and take-offs are short); the other bodies use tighter distances, otherwise a
     shank hovering a few cm above the ground next to a planted foot reads as a knee contact.
  6. Support completion (``support``; the labeled distances alone miss the supports of a prone / kneeling / crawling robot: retargeting
     leaves its knees, hands and elbows hovering 2-5 cm above the floor, so a 33 kg robot "rests" on two elbows). Qualitative statics,
     no forces: in the quasi-static frames (``quasi_static``: CoM speed <= ``QUASI_STATIC_V``, CoM acceleration <= ``QUASI_STATIC_A``;
     flight, push-offs, falls and walking at speed are no statics problem) whose labeled supports stand on (nearly) flat faces, the
     convex hull of the supports' footprints (the lowest ``SUPPORT_SLICE`` of each body, projected on the horizontal plane) must hold
     the vertical projection of the CoM (within ``SUPPORT_MARGIN``). If it does not, the bodies that are within ``SUPPORT_D`` of the
     terrain, at rest (contact speed <= ``SUPPORT_V``) on a face that can carry weight (normal z >= ``SUPPORT_NORMAL_Z``) are added,
     nearest first, those that bring the hull closer to the CoM, until it holds it; if it never does nothing is added (``complete_support``).
     The added bodies then go through the same hysteresis / moving-frames / minimum-duration filters as every contact. Frames whose
     labeled supports are on steep faces (a hand hooked on a rung, a knee braced on a wall: held by grip / friction, the horizontal
     projection says nothing) are not checked. Nothing is added to a stance that holds already, a foot that carries the CoM alone, a
     CoM within the margin of it (single-leg stance), or a motion that is not quasi-static (flight).

Sidecar keys: framerate, body_names (9), link_names (9), contact_flag (T,9) bool, contact_pos_w (T,9,3) float32 (zeros without contact),
onset_pos_w (T,9,3) float32 (contact position at the first frame of the segment the frame belongs to, zeros without contact),
contact_normal_w (T,9,3) float32 (terrain normal at the contact, zeros without contact), min_dist (T,9) float32 (exact), contact_speed (T,9)
float32 (m/s, as defined above), com_w (T,3) float32 (CoM of the robot), support_added (T,9) bool (contacts that the support
completion added), support_status (T,) int8 (``SUPPORT_*``: outcome of the support check of the frame), params (json string).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os

import mujoco
import numpy as np
import trimesh
import warp as wp
import yaml
from scipy.spatial import ConvexHull, QhullError

from instinct_mj.rl.acp.planner import CONTACT_BODIES, CONTACT_LINKS
from instinct_mj.assets.acp_g1 import get_acp_g1_spec

# disable warp module initialization messages (warp initializes lazily, in the worker processes: a CUDA context does not survive fork)
wp.config.quiet = True

FOOT_BODIES = ("left_foot", "right_foot")
HAND_BODIES = ("left_hand", "right_hand")
D_ON = 0.03  # feet: stance feet of retargeted motions hover ~1-2 cm, a static foot within d_on / d_off is a contact
D_OFF = 0.05
D_TOUCH = 0.015  # ACP-unspecified: closer than this a body touches (counts even while sliding)
D_ON_BODY = 0.015  # ACP-unspecified: knees, pelvis, elbows, hands
D_OFF_BODY = 0.05  # ACP-unspecified: stays on while a kneeling knee / propping hand hovers up to 5 cm (retargeting); entry stays at d_on_body
V_MAX = 0.5  # m/s, the paper's slip speed (contact-slip penalty)
MOVE_FRAMES = 3
SKID_FRAMES = 5  # ACP-unspecified: a touching body may slide this many frames before / after it sticks
MIN_DURATION = 0.2  # paper: ten 50 Hz steps
MIN_DURATION_FOOT = 0.08  # ACP-unspecified: take-offs / quick steps touch for ~0.1 s
MIN_DURATION_HAND = 0.08  # ACP-unspecified: a hand planted on a box during a vault touches for ~0.1 s
MIN_NORMAL_Z = None  # ACP-unspecified: None keeps contacts on vertical faces (bracing a knee / hand against a wall)
SPEED_QUANTILE = 0.2
STEEP_NORMAL_Z = 0.2  # statistics only: contacts on faces steeper than this (walls, undersides) are counted in the summary
D_NORMAL = 0.003  # ACP-unspecified: farther than this from the terrain the contact normal is the direction of separation
# support completion (reference motions are retargeted: knees / hands / elbows of a crawling or prone robot hover a few cm above the floor,
# the contacts that are labeled alone cannot carry the robot): ACP-unspecified, see ``complete_support``
SUPPORT_D = 0.05  # a body can be added as a support when it is at most this close to the terrain
SUPPORT_V = 0.2  # m/s, ... and at rest there (contact speed)
SUPPORT_NORMAL_Z = 0.5  # ... on a face that carries weight (the terrain pushes up: slope < 60 deg)
SUPPORT_FLAT_Z = 0.7  # the check is done only when the labeled supports are on (nearly) flat faces: a hand hooked on a rung / a knee braced against a
# wall are held by friction / grip, the horizontal projection of the CoM does not say whether they carry the robot
SUPPORT_MARGIN = 0.05  # m, the CoM may be this far outside the hull of the supports (the footprint is the lowest slice of a body, retargeting is noisy)
SUPPORT_SLICE = 0.015  # m, footprint of a body = its points at most this far above its closest point to the terrain
QUASI_STATIC_V = 0.4  # m/s, CoM speed: statics (CoM above the support polygon) is the criterion only for slow motions ...
QUASI_STATIC_A = 2.0  # m/s^2 (0.2 g), ... and small CoM accelerations (flight: |a| = g)
QUASI_STATIC_WINDOW = 0.1  # s, CoM speed / acceleration by central differences over +- this time
# per-frame outcome of the support check (sidecar ``support_status``)
SUPPORT_NOT_CHECKED, SUPPORT_HELD, SUPPORT_COMPLETED, SUPPORT_UNREACHABLE, SUPPORT_NO_CANDIDATE, SUPPORT_STEEP = range(6)
FRAME_CHUNK = 64  # frames labeled at once (bounds the memory of the replay points and the terrain query)


def _runs(flag: np.ndarray) -> list[tuple[int, int]]:
    """(start, end_exclusive) of the True runs of a 1D bool array."""
    padded = np.concatenate([[False], flag, [False]]).astype(np.int8)
    diff = np.diff(padded)
    return list(zip(np.nonzero(diff == 1)[0].tolist(), np.nonzero(diff == -1)[0].tolist()))


def contact_segments(
    min_dist: np.ndarray,
    speed: np.ndarray,
    framerate: float,
    d_on,
    d_off,
    v_max: float,
    min_duration,
    move_frames: int = MOVE_FRAMES,
    normal_z: np.ndarray | None = None,
    min_normal_z: float | None = None,
    d_touch=0.0,
    skid_frames: int = SKID_FRAMES,
) -> np.ndarray:
    """Contact flags from min distance and contact speed, over axis 0 (works for (T,) and (T, B) arrays).

    Hysteresis (on below ``d_on``, stay while ``<= d_off``; the stay-state is ended by a moving frame that does not touch, so a
    contact that was dropped while moving must enter below ``d_on`` again), then the moving frames are dropped (``speed >= v_max`` for at least
    ``move_frames`` consecutive frames; first/last frame exempt) unless the body touches (``min_dist < d_touch``: a sliding
    contact still carries a normal force, only a hovering one does not) - sliding frames only within ``skid_frames`` of the
    segment's static frames (skid into a stance / push-off; a segment that never sticks is dragged, not supported) -, then frames with ``normal_z < min_normal_z`` (if
    given), then segments shorter than ``min_duration`` seconds are discarded. For (T, B) input, ``d_on`` / ``d_off`` /
    ``min_duration`` / ``d_touch`` may be per-body sequences.
    """
    min_dist = np.asarray(min_dist, dtype=np.float64)
    if min_dist.ndim == 2:
        per = lambda x, i: x[i] if np.ndim(x) else x  # noqa: E731
        return np.stack(
            [
                contact_segments(
                    min_dist[:, i], np.asarray(speed)[:, i], framerate, per(d_on, i), per(d_off, i), v_max, per(min_duration, i),
                    move_frames, None if normal_z is None else np.asarray(normal_z)[:, i], min_normal_z, per(d_touch, i),
                    skid_frames,
                )
                for i in range(min_dist.shape[1])
            ],
            axis=1,
        )  # fmt: skip
    num_frames = min_dist.shape[0]
    moving = np.asarray(speed) >= v_max
    for start, end in _runs(moving):
        if end - start < move_frames:
            moving[start:end] = False
    sticks = ~moving  # the first/last-frame exemption below is no evidence of sticking
    moving[[0, -1]] = False
    flag = np.zeros(num_frames, dtype=bool)
    state = False
    for t in range(num_frames):
        # the "stay on until d_off" state does not survive a moving frame that does not touch: in a climb the pelvis brushes the
        # wall while the robot moves up, and without this reset the state lingers while the thigh passes 2-5 cm from the wall and
        # the body is labeled again above d_on once the motion stops (a contact that never entered)
        stay = state and min_dist[t] <= d_off and not (moving[t] and min_dist[t] >= d_touch)
        state = stay or min_dist[t] < d_on
        flag[t] = state
    flag &= ~moving | (min_dist < d_touch)
    # a sliding contact is a skid into / out of a sticking one (friction brings the body to rest): keep the moving frames of a
    # segment only within ``skid_frames`` of its static frames, a segment without a static frame (dragging) is no support
    for start, end in _runs(flag):
        static = np.nonzero(sticks[start:end])[0]
        if static.size == 0:
            flag[start:end] = False
            continue
        gap = np.abs(np.arange(end - start)[:, None] - static[None]).min(axis=1)
        flag[start:end] &= gap <= skid_frames
    if normal_z is not None and min_normal_z is not None:
        flag &= np.asarray(normal_z) >= min_normal_z
    min_frames = int(round(min_duration * framerate))
    for start, end in _runs(flag):
        if end - start < min_frames:
            flag[start:end] = False
    return flag


def _convex_hull_2d(points: np.ndarray) -> np.ndarray:
    """Vertices (K, 2) of the convex hull (counter-clockwise, K >= 1; collinear points leave their two end points)."""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) > 8:  # qhull (counter-clockwise in 2D) for the point clouds of a footprint; degenerate (collinear) ones fall through
        try:
            return pts[ConvexHull(pts).vertices]
        except QhullError:
            pass
    pts = np.unique(np.round(pts, 9), axis=0)
    if len(pts) <= 2:
        return pts
    order = np.lexsort((pts[:, 1], pts[:, 0]))
    pts = pts[order]

    def half(seq):
        chain = []
        for p in seq:
            while len(chain) >= 2 and (chain[-1] - chain[-2])[0] * (p - chain[-2])[1] - (chain[-1] - chain[-2])[1] * (p - chain[-2])[0] <= 0:
                chain.pop()
            chain.append(p)
        return chain[:-1]

    return np.array(half(pts) + half(pts[::-1]))


def _hull_distance(hull: np.ndarray, q: np.ndarray) -> float:
    """Distance of point ``q`` (2,) to the convex hull with vertices ``hull`` (K, 2, counter-clockwise; K = 1, 2: a point, a segment); 0 inside."""
    if len(hull) == 0:
        return np.inf
    nxt = np.roll(hull, -1, axis=0)
    edge = nxt - hull
    if len(hull) >= 3:
        cross = edge[:, 0] * (q[1] - hull[:, 1]) - edge[:, 1] * (q[0] - hull[:, 0])
        if (cross >= 0).all():
            return 0.0
    length2 = np.maximum((edge**2).sum(axis=1), 1e-18)
    t = np.clip(((q - hull) * edge).sum(axis=1) / length2, 0.0, 1.0)
    return float(np.linalg.norm(hull + t[:, None] * edge - q, axis=1).min())


def support_distance(support: list[np.ndarray], com_xy: np.ndarray) -> float:
    """Distance of the CoM (2,) to the convex hull of the ``support`` footprints (N, 2) (0 when it is above them, inf without any)."""
    points = np.concatenate(support) if support else np.zeros((0, 2))
    return _hull_distance(_convex_hull_2d(points), com_xy) if len(points) else np.inf


def complete_support(com_xy: np.ndarray, support: list[np.ndarray], candidates: list[np.ndarray], margin: float) -> list[int]:
    """Which ``candidates`` (footprints (N, 2), nearest to the terrain first) to add to the labeled ``support`` footprints so that the
    convex hull of all of them holds the CoM (within ``margin``). A candidate is taken only if it brings the hull closer to the CoM (a
    shank hovering next to a planted foot adds nothing); if the CoM is out of reach even with all useful candidates, none is added: the
    motion is no statics problem (a fall, a push) and the hovering bodies carry nothing. Returns [] when the supports hold already."""
    chosen, taken = [], list(support)
    distance = support_distance(taken, com_xy)
    for i, foot in enumerate(candidates):
        if distance <= margin:
            break
        new_distance = support_distance(taken + [foot], com_xy)
        if new_distance < distance - 1e-4:
            taken, distance = taken + [foot], new_distance
            chosen.append(i)
    return chosen if distance <= margin else []


def quasi_static(com: np.ndarray, framerate: float, v_max: float = QUASI_STATIC_V, a_max: float = QUASI_STATIC_A) -> np.ndarray:
    """Frames where the CoM (T, 3) moves slowly and does not accelerate (central differences over ``QUASI_STATIC_WINDOW``; the clip ends
    are padded with their first / last frame): the weight is carried by the supports alone, so the CoM must be above their polygon.
    Flight (a = g), a push-off, a fall and walking at speed are not."""
    k = max(int(round(QUASI_STATIC_WINDOW * framerate)), 1)
    padded = np.concatenate([np.repeat(com[:1], k, axis=0), com, np.repeat(com[-1:], k, axis=0)])
    speed = np.linalg.norm(padded[2 * k :] - padded[: -2 * k], axis=1) / (2 * k / framerate)
    acc = np.linalg.norm(padded[2 * k :] - 2 * com + padded[: -2 * k], axis=1) / (k / framerate) ** 2
    return (speed <= v_max) & (acc <= a_max)


@wp.kernel
def _closest_point_kernel(
    mesh: wp.uint64,
    points: wp.array(dtype=wp.vec3),
    max_dist: float,
    dist: wp.array(dtype=wp.float32),
    closest: wp.array(dtype=wp.vec3),
    tri: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    face = int(0)  # closest face index
    u = float(0.0)  # barycentric coordinates on the face
    v = float(0.0)
    if wp.mesh_query_point_no_sign(mesh, points[tid], max_dist, face, u, v):
        closest[tid] = wp.mesh_eval_position(mesh, face, u, v)
        dist[tid] = wp.length(closest[tid] - points[tid])
        tri[tid] = face
    else:
        closest[tid] = points[tid]
        dist[tid] = max_dist
        tri[tid] = -1


class TerrainDistance:
    """Exact distance / closest point of query points to a (non-watertight) triangle mesh (warp BVH on the CPU: the labeling
    runs next to the training jobs, which own the GPU)."""

    def __init__(self, mesh: trimesh.Trimesh):
        self.vertices = np.asarray(mesh.vertices, dtype=np.float64)
        self.faces = np.asarray(mesh.faces, dtype=np.int64)
        self.triangles = self.vertices[self.faces]
        self.mesh = wp.Mesh(
            points=wp.array(self.vertices.astype(np.float32), dtype=wp.vec3, device="cpu"),
            indices=wp.array(self.faces.astype(np.int32).flatten(), dtype=wp.int32, device="cpu"),
        )

    def query(self, points: np.ndarray, max_dist: float = 1e6, return_tri: bool = False):
        """Returns (distance (N,), closest point (N,3)[, closest triangle index (N,)]); points farther than ``max_dist`` get
        distance ``max_dist``, themselves as closest point and triangle -1. The BVH is float32."""
        n = len(points)
        dist = wp.zeros(n, dtype=wp.float32, device="cpu")
        closest = wp.zeros(n, dtype=wp.vec3, device="cpu")
        tri = wp.zeros(n, dtype=wp.int32, device="cpu")
        if n:
            query_points = wp.array(np.ascontiguousarray(points, dtype=np.float32), dtype=wp.vec3, device="cpu")
            wp.launch(_closest_point_kernel, dim=n, inputs=[self.mesh.id, query_points, max_dist, dist, closest, tri], device="cpu")
        out = dist.numpy().astype(np.float64), closest.numpy().astype(np.float64)
        return (*out, tri.numpy().astype(np.int64)) if return_tri else out

    def normal(self, tri: np.ndarray) -> np.ndarray:
        """Unit normals (N,3) of triangles ``tri`` (STL winding: outward for the terrain meshes)."""
        t = self.triangles[tri]
        n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
        return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


def contact_normal(terrain: TerrainDistance, point: np.ndarray, closest: np.ndarray, dist: np.ndarray, tri: np.ndarray) -> np.ndarray:
    """Terrain normal (N,3) of the contact between body points ``point`` and their closest terrain points ``closest`` (distance
    ``dist``, closest triangle ``tri``): the direction of separation when the body is clear of the terrain (exact on edges and rungs,
    where the closest triangle is one of several), else the closest triangle's outward normal (STL winding). The latter also holds
    for a body that sank into the terrain: the unsigned distance cannot tell sides, the separation direction then points into the
    solid (against the winding normal) and the body's centroid is no reference either (a hand buried below a box top)."""
    n = terrain.normal(tri)
    sep = (point - closest) / np.maximum(dist, 1e-12)[:, None]
    clear = (dist > D_NORMAL) & (np.einsum("ij,ij->i", sep, n) > 0.0)
    return np.where(clear[:, None], sep, n)


def _sample_geom(model: mujoco.MjModel, g: int) -> np.ndarray:
    """Surface points of a collision geom in the geom frame."""
    gtype = model.geom_type[g]
    size = model.geom_size[g]
    if gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
        n = 200
        i = np.arange(n) + 0.5
        phi = np.arccos(1 - 2 * i / n)
        theta = np.pi * (1 + 5**0.5) * i
        return size[0] * np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], axis=1)
    if gtype == mujoco.mjtGeom.mjGEOM_CAPSULE:
        r, h = size[0], size[1]
        ang = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        ring = np.stack([np.cos(ang), np.sin(ang)], axis=1) * r
        zs = np.linspace(-h, h, max(int(np.ceil(2 * h / 0.02)) + 1, 2))
        side = np.concatenate([np.concatenate([ring, np.full((len(ring), 1), z)], axis=1) for z in zs])
        n = 40
        i = np.arange(n) + 0.5
        phi = np.arccos(1 - i / n)  # upper hemisphere
        theta = np.pi * (1 + 5**0.5) * i
        cap = r * np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], axis=1)
        return np.concatenate([side, cap + [0, 0, h], cap * [1, 1, -1] - [0, 0, h]])
    if gtype == mujoco.mjtGeom.mjGEOM_BOX:
        s = size
        g1 = [np.linspace(-a, a, max(int(np.ceil(2 * a / 0.02)) + 1, 2)) for a in s]
        grid = np.stack(np.meshgrid(*g1, indexing="ij"), axis=-1).reshape(-1, 3)
        return grid[(np.abs(grid) >= s - 1e-9).any(axis=1)]
    if gtype == mujoco.mjtGeom.mjGEOM_MESH:
        m = model.geom_dataid[g]
        a, n = model.mesh_vertadr[m], model.mesh_vertnum[m]
        v = model.mesh_vert[a : a + n]
        if n >= 4:
            v = v[ConvexHull(v).vertices]
        return v
    raise ValueError(f"unsupported collision geom type {mujoco.mjtGeom(gtype).name}")


class ContactLabeler:
    def __init__(self, simplified_collision: bool = False):
        # the collision geometry the labels are measured on must be the one the task simulates (``G1AcpTrackV1EnvCfg.simplified_collision``,
        # default off): the authors' collision meshes, matching v3_final / v3_train6
        self.model = get_acp_g1_spec(simplified_collision=simplified_collision).compile()
        self.data = mujoco.MjData(self.model)
        self.geom_ids: list[np.ndarray] = []
        self.local_pts: list[np.ndarray] = []  # per body, (n, 3) in geom frames
        self.pt_geom: list[np.ndarray] = []  # per body, geom index (into geom_ids[b]) of each point
        for link in CONTACT_LINKS:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, link)
            geoms = np.nonzero(self.model.geom_bodyid == bid)[0]
            col = geoms[(self.model.geom_contype[geoms] != 0) | (self.model.geom_conaffinity[geoms] != 0)]
            if len(col) == 0:
                col = geoms[self.model.geom_type[geoms] == mujoco.mjtGeom.mjGEOM_MESH]
            assert len(col) > 0, f"{link} has no geoms"
            pts = [_sample_geom(self.model, g) for g in col]
            self.geom_ids.append(col)
            self.local_pts.append(np.concatenate(pts))
            self.pt_geom.append(np.concatenate([np.full(len(p), i) for i, p in enumerate(pts)]))

    def world_points(self, motion: dict, start: int, end: int) -> list[np.ndarray]:
        """Per body (end - start, n, 3) world points of the kinematic replay of frames [start, end)."""
        model, data = self.model, self.data
        # the motion files carry 29 joints; the 6 wrist joints of the file do not exist in the 23-joint model (welded) and are ignored
        names = [str(n) for n in motion["joint_names"]]
        columns = [i for i, n in enumerate(names) if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) >= 0]
        assert len(columns) == model.njnt - 1, "every joint of the model must be in the motion file"
        qadr = np.array([model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, names[i])] for i in columns])
        out = [np.zeros((end - start, len(p), 3)) for p in self.local_pts]
        for t in range(start, end):
            data.qpos[:3] = motion["base_pos_w"][t]
            data.qpos[3:7] = motion["base_quat_w"][t]  # wxyz, as MuJoCo
            data.qpos[qadr] = motion["joint_pos"][t][columns]
            mujoco.mj_kinematics(model, data)
            for b, geoms in enumerate(self.geom_ids):
                rot = data.geom_xmat[geoms].reshape(-1, 3, 3)[self.pt_geom[b]]
                out[b][t - start] = np.einsum("nij,nj->ni", rot, self.local_pts[b]) + data.geom_xpos[geoms][self.pt_geom[b]]
        return out

    @staticmethod
    def _support_candidates(flag, min_dist, speed, normal_z, com, static, footprint, params) -> tuple[np.ndarray, np.ndarray]:
        """((T, B) bodies to add as supports (``complete_support``), (T,) outcome codes) for the quasi-static frames whose labeled contacts do
        not hold the CoM."""
        added = np.zeros_like(flag)
        status = np.full(len(flag), SUPPORT_NOT_CHECKED, dtype=np.int8)
        for t in np.nonzero(static)[0]:
            held = np.array([b for b in np.nonzero(flag[t])[0] if (t, b) in footprint], dtype=int)
            if (normal_z[t, held] < SUPPORT_FLAT_Z).any():
                status[t] = SUPPORT_STEEP
                continue
            support = [footprint[t, b] for b in held]
            if support_distance(support, com[t, :2]) <= params["support_margin"]:
                status[t] = SUPPORT_HELD
                continue
            candidates = [
                b for b in np.argsort(min_dist[t])
                if not flag[t, b] and min_dist[t, b] <= params["support_d"] and speed[t, b] <= SUPPORT_V and normal_z[t, b] >= SUPPORT_NORMAL_Z
            ]  # fmt: skip
            chosen = complete_support(com[t, :2], support, [footprint[t, b] for b in candidates], params["support_margin"])
            added[t, [candidates[i] for i in chosen]] = True
            status[t] = SUPPORT_COMPLETED if chosen else SUPPORT_UNREACHABLE if candidates else SUPPORT_NO_CANDIDATE
        return added, status

    def center_of_mass(self, motion: dict) -> np.ndarray:
        """(T, 3) CoM of the whole robot of the kinematic replay."""
        model, data = self.model, self.data
        names = [str(n) for n in motion["joint_names"]]
        columns = [i for i, n in enumerate(names) if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) >= 0]
        qadr = np.array([model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, names[i])] for i in columns])
        mass = model.body_mass[1:]  # without the world body
        com = np.zeros((len(motion["joint_pos"]), 3))
        for t in range(len(com)):
            data.qpos[:3] = motion["base_pos_w"][t]
            data.qpos[3:7] = motion["base_quat_w"][t]
            data.qpos[qadr] = motion["joint_pos"][t][columns]
            mujoco.mj_kinematics(model, data)
            com[t] = mass @ data.xipos[1:] / mass.sum()
        return com

    def label(self, motion: dict, terrain: TerrainDistance, params: dict) -> dict:
        return self.finish(self.extract(motion, terrain, params), params)

    def extract(
        self, motion: dict, terrain: TerrainDistance, params: dict, static_limits: tuple[float, float] | None = None,
        *, reference_positions: np.ndarray | None = None, reference_flags: np.ndarray | None = None,
    ) -> dict:
        """Kinematic replay and terrain queries of the clip (the expensive part, independent of the thresholds of ``finish``): per frame and
        body the min distance, contact speed, contact position and normal, plus the CoM and the footprints (hull of the lowest slice of a
        body) of the bodies near the terrain in the quasi-static frames (``static_limits``: CoM speed / acceleration, default those of
        ``params``; ``finish`` may use tighter ones)."""
        framerate = float(motion["framerate"])
        num_frames, num_bodies = motion["joint_pos"].shape[0], len(CONTACT_LINKS)
        if (reference_positions is None) != (reference_flags is None):
            raise ValueError("Reference contact positions and flags must be supplied together")
        if reference_positions is not None:
            if reference_positions.shape != (num_frames, num_bodies, 3) or reference_flags.shape != (num_frames, num_bodies):
                raise ValueError("Reference contacts must match motion length and CONTACT_LINKS order")
            if not np.isfinite(reference_positions).all():
                raise ValueError("Non-finite reference contact position")
        foot = np.array([b in FOOT_BODIES for b in CONTACT_BODIES])
        d_on = np.where(foot, params["d_on"], params["d_on_body"])
        d_off = np.where(foot, params["d_off"], params["d_off_body"])
        min_dist = np.zeros((num_frames, num_bodies))
        speed = np.zeros((num_frames, num_bodies))
        pos = np.zeros((num_frames, num_bodies, 3))
        normal = np.zeros((num_frames, num_bodies, 3))
        patch_dist = np.zeros_like(min_dist) if reference_positions is not None else None
        surface_changed = np.zeros_like(min_dist, dtype=bool) if reference_positions is not None else None
        com = self.center_of_mass(motion)
        v_max, a_max = static_limits or (params["quasi_static_v"], params["quasi_static_a"])
        static = quasi_static(com, framerate, v_max, a_max) if params["support"] else np.zeros(num_frames, bool)
        footprint: dict[tuple[int, int], np.ndarray] = {}  # (frame, body) -> hull (K, 2) of the lowest slice of the body, where support matters
        for start in range(0, num_frames, FRAME_CHUNK):
            end = min(start + FRAME_CHUNK, num_frames)
            lo, hi = max(start - 1, 0), min(end + 1, num_frames)  # one frame of margin for the central difference
            points = self.world_points(motion, lo, hi)
            for b, all_pts in enumerate(points):
                vel = np.gradient(all_pts, axis=0) * framerate if hi - lo > 1 else np.zeros_like(all_pts)
                pts, vel = all_pts[start - lo : end - lo], vel[start - lo : end - lo]
                dist, closest, tri = terrain.query(pts.reshape(-1, 3), return_tri=True)
                dist = dist.reshape(end - start, -1)
                closest, tri = closest.reshape(end - start, -1, 3), tri.reshape(end - start, -1)
                min_dist[start:end, b] = dist.min(axis=1)
                is_min = dist == dist.min(axis=1, keepdims=True)
                near = dist < d_on[b]
                weight = np.where(near.any(axis=1, keepdims=True), near, is_min).astype(np.float64)
                mean_pt = (pts * weight[..., None]).sum(axis=1) / weight.sum(axis=1, keepdims=True)
                _, proj = terrain.query(mean_pt)
                pos[start:end, b] = proj
                # speed of the touching material points (low quantile: a foot rolling over its toe still sticks)
                touching = dist <= d_off[b]
                touching = np.where(touching.any(axis=1, keepdims=True), touching, is_min)
                pt_speed = np.where(touching, np.linalg.norm(vel, axis=2), np.inf)
                count = touching.sum(axis=1)
                speed[start:end, b] = np.sort(pt_speed, axis=1)[np.arange(end - start), (SPEED_QUANTILE * (count - 1)).astype(int)]
                k = dist.argmin(axis=1)
                rows = np.arange(end - start)
                normal[start:end, b] = contact_normal(terrain, pts[rows, k], closest[rows, k], dist[rows, k], tri[rows, k])
                if reference_positions is not None:
                    # Keep the source contact region: a force-weighted toe plant must
                    # not become the centre of the whole sole after a geometric edit.
                    anchor = reference_positions[start:end, b]
                    active = reference_flags[start:end, b].astype(bool)
                    near_anchor = np.linalg.norm(pts - anchor[:, None], axis=-1).argmin(axis=1)
                    pos[start:end, b][active] = closest[rows, near_anchor][active]
                    patch_normal = contact_normal(
                        terrain, pts[rows, near_anchor], closest[rows, near_anchor], dist[rows, near_anchor], tri[rows, near_anchor],
                    )
                    normal[start:end, b][active] = patch_normal[active]
                    patch_dist[start:end, b] = dist[rows, near_anchor]
                    if active.any():
                        _, _, original_tri = terrain.query(anchor[active], return_tri=True)
                        old_n = terrain.normal(original_tri)
                        new_n = terrain.normal(tri[rows, near_anchor][active])
                        surface_changed[start:end, b][active] = np.einsum("ij,ij->i", old_n, new_n) < .5
                for i in np.nonzero(static[start:end] & (min_dist[start:end, b] <= params["support_d"]))[0]:
                    footprint[start + i, b] = _convex_hull_2d(pts[i][dist[i] <= dist[i].min() + SUPPORT_SLICE][:, :2])
        result = dict(framerate=framerate, min_dist=min_dist, speed=speed, pos=pos, normal=normal, com=com, footprint=footprint)
        if reference_positions is not None:
            result.update(contact_patch_dist=patch_dist, surface_changed=surface_changed)
        return result

    def finish(self, features: dict, params: dict) -> dict:
        """Contact flags (``contact_segments``, then the support completion) and the sidecar arrays from the ``extract`` features."""
        framerate, min_dist, speed, pos, normal = (features[k] for k in ("framerate", "min_dist", "speed", "pos", "normal"))
        com, footprint = features["com"], features["footprint"]
        num_frames, num_bodies = min_dist.shape
        foot = np.array([b in FOOT_BODIES for b in CONTACT_BODIES])
        hand = np.array([b in HAND_BODIES for b in CONTACT_BODIES])
        d_on = np.where(foot, params["d_on"], params["d_on_body"])
        d_off = np.where(foot, params["d_off"], params["d_off_body"])
        min_duration = np.where(foot, params["min_duration_foot"], np.where(hand, params["min_duration_hand"], params["min_duration"]))
        d_touch = np.where(foot, params["d_touch"], params["d_on_body"])
        flag = contact_segments(
            min_dist, speed, framerate, d_on, d_off, params["v_max"], min_duration, params["move_frames"],
            normal[..., 2], params["min_normal_z"], d_touch, params["skid_frames"],
        )  # fmt: skip
        added = np.zeros_like(flag)
        status = np.full(num_frames, SUPPORT_NOT_CHECKED, dtype=np.int8)
        if params["support"]:
            static = quasi_static(com, framerate, params["quasi_static_v"], params["quasi_static_a"])
            added, status = self._support_candidates(flag, min_dist, speed, normal[..., 2], com, static, footprint, params)
            if added.any():  # the added bodies enter like touching ones: hysteresis, moving frames and minimum duration as for any contact
                flag = contact_segments(
                    np.where(added, 0.0, min_dist), speed, framerate, d_on, d_off, params["v_max"], min_duration, params["move_frames"],
                    normal[..., 2], params["min_normal_z"], d_touch, params["skid_frames"],
                )  # fmt: skip
        contact_pos = np.where(flag[..., None], pos, 0.0)
        onset = np.zeros_like(contact_pos)
        for b in range(num_bodies):
            for s, e in _runs(flag[:, b]):
                onset[s:e, b] = pos[s, b]
        return dict(
            framerate=np.asarray(framerate),
            body_names=np.array(CONTACT_BODIES),
            link_names=np.array(CONTACT_LINKS),
            contact_flag=flag,
            contact_pos_w=contact_pos.astype(np.float32),
            onset_pos_w=onset.astype(np.float32),
            contact_normal_w=np.where(flag[..., None], normal, 0.0).astype(np.float32),
            min_dist=min_dist.astype(np.float32),
            contact_speed=speed.astype(np.float32),
            support_added=added & flag,
            support_status=status,
            com_w=com.astype(np.float32),
            params=np.array(json.dumps(params)),
        )


def default_params(**overrides) -> dict:
    """Labeling parameters (stored in the sidecar as json)."""
    params = dict(
        d_on=D_ON, d_off=D_OFF, d_touch=D_TOUCH, d_on_body=D_ON_BODY, d_off_body=D_OFF_BODY, v_max=V_MAX, move_frames=MOVE_FRAMES,
        skid_frames=SKID_FRAMES, min_duration=MIN_DURATION, min_duration_foot=MIN_DURATION_FOOT, min_duration_hand=MIN_DURATION_HAND,
        min_normal_z=MIN_NORMAL_Z, simplified_collision=False,
        speed_quantile=SPEED_QUANTILE, support=True, support_d=SUPPORT_D, support_margin=SUPPORT_MARGIN, quasi_static_v=QUASI_STATIC_V, quasi_static_a=QUASI_STATIC_A,
    )  # fmt: skip
    params.update(overrides)
    return params


def _load_terrain(path: str) -> tuple[trimesh.Trimesh, TerrainDistance]:
    mesh = trimesh.load(path, force="mesh")
    return mesh, TerrainDistance(mesh)


def _viz(out_path: str, labels: dict, mesh: trimesh.Trimesh, terrain: TerrainDistance, title: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    flag, pos, fr = labels["contact_flag"], labels["contact_pos_w"], float(labels["framerate"])
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw=dict(width_ratios=[1.3, 1]))
    t = np.arange(flag.shape[0]) / fr
    for b, name in enumerate(CONTACT_BODIES):
        ax0.fill_between(t, b, b + 0.8 * flag[:, b], step="mid", color=f"C{b}")
    ax0.set_yticks(np.arange(len(CONTACT_BODIES)) + 0.4, CONTACT_BODIES)
    ax0.set_xlabel("time [s]")
    ax0.set_title(title)
    v = np.concatenate([terrain.vertices, trimesh.remesh.subdivide_to_size(terrain.vertices, terrain.faces, max_edge=0.1)[0]])
    lo, hi = v[:, :2].min(axis=0), v[:, :2].max(axis=0)
    res = 0.1
    shape = np.ceil((hi - lo) / res).astype(int) + 1
    hm = np.full(shape, np.nan)
    ij = ((v[:, :2] - lo) / res).astype(int)
    np.fmax.at(hm, (ij[:, 0], ij[:, 1]), v[:, 2])
    im = ax1.imshow(hm.T, origin="lower", extent=[lo[0], hi[0], lo[1], hi[1]], cmap="terrain")
    fig.colorbar(im, ax=ax1, label="terrain height [m]")
    for b, name in enumerate(CONTACT_BODIES):
        m = flag[:, b]
        ax1.scatter(pos[m, b, 0], pos[m, b, 1], s=6, color=f"C{b}", label=name, edgecolors="k", linewidths=0.2)
    ax1.set_aspect("equal")
    ax1.legend(fontsize=7, loc="upper right")
    ax1.set_title("contact positions (top-down)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


_LABELER: ContactLabeler | None = None
_TERRAINS: dict[str, tuple[trimesh.Trimesh, TerrainDistance]] = {}


def _get_terrain(path: str) -> tuple[trimesh.Trimesh, TerrainDistance]:
    if path not in _TERRAINS:
        _TERRAINS.clear()  # one terrain at a time: a subdivided 40 x 40 m floor is ~0.4 GB and datasets can have one terrain per clip
        _TERRAINS[path] = _load_terrain(path)
    return _TERRAINS[path]


def _label_one(job: tuple) -> tuple[str, dict | None]:
    """Label one motion (per-process labeler / terrain cache), write its sidecar; returns (rel path, labels or None if skipped)."""
    global _LABELER
    root, rel, terrain_path, _, params, overwrite = job
    out_path = os.path.join(root, rel[: -len(".npz")] + ".contacts.npz")
    if os.path.exists(out_path) and not overwrite:
        return rel, None
    if _LABELER is None:
        _LABELER = ContactLabeler(simplified_collision=bool(params.get("simplified_collision", False)))
    _, terrain = _get_terrain(terrain_path)
    motion = dict(np.load(os.path.join(root, rel), allow_pickle=True))
    labels = _LABELER.label(motion, terrain, params)
    np.savez(out_path, **labels)
    return rel, labels


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", required=True, help="folder with metadata.yaml")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--viz", default=None, help="motion path relative to the dataset folder to plot")
    parser.add_argument("--workers", type=int, default=1, help="processes over clips")
    parser.add_argument("--d-on", type=float, default=D_ON, help="feet")
    parser.add_argument("--d-off", type=float, default=D_OFF, help="feet")
    parser.add_argument("--d-touch", type=float, default=D_TOUCH, help="feet; the other bodies touch below --d-on-body")
    parser.add_argument("--d-on-body", type=float, default=D_ON_BODY, help="knees, pelvis, elbows, hands")
    parser.add_argument("--d-off-body", type=float, default=D_OFF_BODY, help="knees, pelvis, elbows, hands")
    parser.add_argument("--v-max", type=float, default=V_MAX)
    parser.add_argument("--move-frames", type=int, default=MOVE_FRAMES)
    parser.add_argument("--skid-frames", type=int, default=SKID_FRAMES)
    parser.add_argument("--min-duration", type=float, default=MIN_DURATION)
    parser.add_argument("--min-duration-foot", type=float, default=MIN_DURATION_FOOT)
    parser.add_argument("--min-duration-hand", type=float, default=MIN_DURATION_HAND)
    parser.add_argument("--min-normal-z", type=float, default=MIN_NORMAL_Z)
    parser.add_argument("--no-support", action="store_true", help="no support completion")
    parser.add_argument("--support-d", type=float, default=SUPPORT_D, help="a body may be added as a support within this distance")
    parser.add_argument("--support-margin", type=float, default=SUPPORT_MARGIN)
    parser.add_argument("--quasi-static-v", type=float, default=QUASI_STATIC_V)
    parser.add_argument("--quasi-static-a", type=float, default=QUASI_STATIC_A)
    parser.add_argument(
        "--simplified-collision", dest="simplified_collision", action="store_true", default=False,
        help="measure the labels on the analytic collision primitives of the legs / upper body (the task's simplified_collision, "
        "default off) instead of the authors' collision meshes (matching v3_final / v3_train6)",
    )
    parser.add_argument(
        "--mesh-collision", dest="simplified_collision", action="store_false",
        help="measure on the authors' collision meshes (the default; ``G1AcpTrackV1EnvCfg.simplified_collision=False``)",
    )  # fmt: skip
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU only (MuJoCo kinematics, warp BVH on the CPU): the GPU belongs to the training jobs
    root = os.path.abspath(os.path.expanduser(args.dataset))
    with open(os.path.join(root, "metadata.yaml")) as f:
        meta = yaml.safe_load(f)
    terrain_files = {t["terrain_id"]: t["terrain_file"] for t in meta["terrains"]}
    params = default_params(
        d_on=args.d_on, d_off=args.d_off, d_touch=args.d_touch, d_on_body=args.d_on_body, d_off_body=args.d_off_body, v_max=args.v_max,
        move_frames=args.move_frames, skid_frames=args.skid_frames, min_duration=args.min_duration, min_duration_foot=args.min_duration_foot,
        min_duration_hand=args.min_duration_hand, min_normal_z=args.min_normal_z, support=not args.no_support,
        simplified_collision=args.simplified_collision,
        support_d=args.support_d, support_margin=args.support_margin, quasi_static_v=args.quasi_static_v, quasi_static_a=args.quasi_static_a,
    )  # fmt: skip
    jobs = [(root, entry["motion_file"], os.path.join(root, terrain_files[entry["terrain_id"]]), entry["terrain_id"], params, args.overwrite) for entry in meta["motion_files"]]
    if args.workers > 1:
        pool = multiprocessing.Pool(args.workers)
        results = pool.imap(_label_one, jobs)
    else:
        results = map(_label_one, jobs)
    short = [b.replace("_link", "").replace("_rubber", "")[:9] for b in CONTACT_LINKS]
    print(f"{'motion':60s} {'T':>5s} " + " ".join(f"{s:>9s}" for s in short) + "   (% frames / #segments)")
    tot_frames, tot_flag, tot_segs = 0, np.zeros(len(CONTACT_BODIES)), np.zeros(len(CONTACT_BODIES), dtype=int)
    tot_added, tot_status = np.zeros(len(CONTACT_BODIES), dtype=int), np.zeros(SUPPORT_STEEP + 1, dtype=int)
    tot_steep = np.zeros(len(CONTACT_BODIES))  # contact frames on faces with normal z < STEEP_NORMAL_Z (what --min-normal-z would drop)
    for rel, labels in results:
        if labels is None:
            print(f"skip (exists) {rel}")
            continue
        flag = labels["contact_flag"]
        segs = np.array([len(_runs(flag[:, b])) for b in range(flag.shape[1])])
        tot_frames += flag.shape[0]
        tot_flag += flag.sum(axis=0)
        tot_segs += segs
        tot_steep += (flag & (labels["contact_normal_w"][..., 2] < STEEP_NORMAL_Z)).sum(axis=0)
        tot_added += labels["support_added"].sum(axis=0)
        tot_status += np.bincount(labels["support_status"], minlength=SUPPORT_STEEP + 1)
        print(f"{rel[-60:]:60s} {flag.shape[0]:5d} " + " ".join(f"{100 * flag[:, b].mean():4.0f}%/{segs[b]:<3d}" for b in range(flag.shape[1])))
        if args.viz == rel:
            tid = next(e["terrain_id"] for e in meta["motion_files"] if e["motion_file"] == rel)
            mesh, terrain = _get_terrain(os.path.join(root, terrain_files[tid]))
            labels = dict(np.load(os.path.join(root, rel[: -len(".npz")] + ".contacts.npz")))
            png = f"/tmp/acp_contacts_{os.path.basename(rel)[: -len('.npz')]}.png"
            _viz(png, labels, mesh, terrain, rel)
            print(f"viz -> {png}")
    if tot_frames:
        print(f"{'TOTAL':60s} {tot_frames:5d} " + " ".join(f"{100 * tot_flag[b] / tot_frames:4.0f}%/{tot_segs[b]:<3d}" for b in range(len(CONTACT_BODIES))))
        print(f"{f'contact frames on faces with normal z < {STEEP_NORMAL_Z}':66s} " + " ".join(f"{100 * tot_steep[b] / max(tot_flag[b], 1):4.0f}%    " for b in range(len(CONTACT_BODIES))))
        print(f"{'contact frames added by the support completion':66s} " + " ".join(f"{tot_added[b]:9d}" for b in range(len(CONTACT_BODIES))))
        print("support check (frames): " + ", ".join(f"{n}={c}" for n, c in zip(("not checked", "held", "completed", "unreachable", "no candidate", "steep"), tot_status)))


if __name__ == "__main__":
    main()
