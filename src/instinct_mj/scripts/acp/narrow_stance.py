"""Put the feet of a reference clip onto the support the clip is supposed to use, then relabel its contacts.

Two kinds of support need this:

``beam`` (default)
    Retargeted or Kimodo-generated clips keep a human stance: the ankle points sit at ``|y| = 0.07-0.12 m``. On a 0.20 m wide
    beam that puts the sole (7.4 cm wide, ankle ``+- 0.037``) partly or entirely outside the support, and since
    ``label_contacts`` projects the contact of a body onto the nearest point of the terrain box, its contact labels end up
    clamped to the beam edge. The tool moves the ankle towards the centerline (only ``y``; the fore-aft position, the height and
    the foot orientation are kept).

``stakes`` (梅花桩)
    The stakes are placed at the plan's footholds, and the retargeted clip lands 3-7 cm off them. The tool moves each stance foot
    (``x`` and ``y``) until the ankle - the plan's foothold, and what the QA's ankle error measures - sits on the stake the foot
    is standing on (``--stake-target ankle``, the default). ``--stake-target foot`` centres the 0.19 m long foot instead, which
    looks tidier from above but moves the ankle off the plan. The shift is constant over a stance, so the foot does not slide
    on the stake; it fades in and out over the swing, where the foot is in the air anyway.

Both modes solve the 6 joint angles of the leg with a small least-squares IK on the ACP G1 MJCF. The clip then has to be
relabeled:

    python -m instinct_mj.scripts.acp.narrow_stance --dataset ~/Datasets/acp/v3_sparse_more --clips '^beam/' --backup-tag beam_narrow
    python -m instinct_mj.scripts.acp.narrow_stance --dataset ~/Datasets/acp/v3_sparse_more --mode stakes --clips '^stakes/' \
        --w-temp 0.05 --stance-tol 0.015 --limit-margin 0.005 --max-move 0.08 --backup-tag stakes_align
    python -m instinct_mj.scripts.acp.label_contacts --dataset ~/Datasets/acp/v3_sparse_more --overwrite --workers 4

Details that keep the edit physically honest:

  * ``beam``: target ``min(--max-target, max(--min-target, half_width - --margin))``. The sole plus ``--margin`` of clearance has
    to fit on the support, and no stance narrower than ``--max-target`` is built (0.20 m beam -> 0.052, 0.26 m -> 0.065).
  * ``beam``: the clamp is per FRAME, not one offset per stance: inside a stance the retargeted ankle itself drifts 2-11 cm
    laterally, so a constant offset cannot keep every frame on the support. The offset track is then Hann-smoothed
    (``--smooth`` frames) and capped at ``--max-move``, so a wide swing keeps its natural arc instead of being dragged onto
    the beam.
  * ``beam``: the offset fades in / out over ``--ramp`` metres of x at the two ends of the narrow box, so the approach and the
    exit on the wide platforms keep their natural stance (that is where a wide placement belongs).
  * ``stakes``: the stance segment, not the frame, is the unit. A planted foot that moves sideways or fore-aft is foot slip, so
    the whole stance gets one offset. Segments that are not on a stake (the platforms at the two ends, a foot that the contact
    labels put in the air) are left alone.
  * ``stakes``: the offset is scaled down (1, 3/4, 1/2, 1/4) when the leg cannot follow it. The stance leg of these clips is
    nearly straight at touchdown, so pulling the foot further back than the reach allows would either break the foot
    orientation or blow up the knee. The printed ``alpha`` / residual columns say how much of the correction survived.
  * Only the 12 leg joints change. Base trajectory, timing, waist and arms are untouched, so the clip's velocity profile, its
    footstep timing and therefore its contact segments stay the same.
  * The IK keeps the foot orientation and, as a temporal term, the frame increment of the source motion. Without the latter a
    nearly straight leg (a flat direction of the Jacobian) jumps 10-20 deg between two frames. ``--w-temp`` lowers the term for
    the stake mode, where the correction is fore-aft and the term would otherwise fight the position target.

Geometry is read from the clip's ``terrain_geoms``: the beam is the box with the smallest ``size[1]`` below ``--max-box-half``
that is longer than it is wide; the stakes are the top-level boxes below ``--max-box-half`` in both directions. ``--mode``
defaults to ``auto`` (beam if there is a beam box, otherwise stakes). The clip needs its ``.contacts.npz`` sidecar (the support
frames and the before/after validation come from it). ``--dry-run`` prints the numbers and writes nothing.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys

import mujoco
import numpy as np
import yaml
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from instinct_mj.assets.acp_g1 import get_acp_g1_spec

LEG_JOINTS = {
    "left": ("left_hip_yaw_joint", "left_hip_roll_joint", "left_hip_pitch_joint", "left_knee_joint", "left_ankle_pitch_joint",
             "left_ankle_roll_joint"),  # fmt: skip
    "right": ("right_hip_yaw_joint", "right_hip_roll_joint", "right_hip_pitch_joint", "right_knee_joint", "right_ankle_pitch_joint",
              "right_ankle_roll_joint"),  # fmt: skip
}
ANKLE_BODY = {"left": "left_ankle_roll_link", "right": "right_ankle_roll_link"}

W_POS = 1.0  # ankle position residual (m)
W_ORI = 0.2  # foot orientation residual (rad): the sole stays flat and parallel to the support
W_REG = 0.005  # stay near the original joint angles: shapes the null space, keeps the edit small

ALPHAS = (1.0, 0.75, 0.5, 0.25)  # how much of the stake offset is allowed when the leg cannot follow all of it


class Rig:
    """ACP playground G1 (free base = pelvis, 23 joints) with qpos addressed by joint name."""

    def __init__(self) -> None:
        self.model = get_acp_g1_spec().compile()
        self.data = mujoco.MjData(self.model)
        self.adr = {j: int(self.model.joint(j).qposadr[0]) for leg in LEG_JOINTS.values() for j in leg}
        self.body = {s: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, b) for s, b in ANKLE_BODY.items()}
        self.range = {j: self.model.jnt_range[self.model.joint(j).id].copy() for leg in LEG_JOINTS.values() for j in leg}
        self._adr: list[int] = []
        self._cols: list[int] = []
        self.prev = np.zeros(6)  # previous frame's leg angles (temporal tie-breaker of the IK)
        self.prev0 = np.zeros(6)  # previous frame's original leg angles (the increment the edit must not disturb)

    def bind(self, motion: dict) -> None:
        """Map the clip's ``joint_pos`` columns onto the model's qpos (the clip carries 29 columns, the model 23 joints)."""
        names = [str(n) for n in motion["joint_names"]]
        joint_names = [j for leg in LEG_JOINTS.values() for j in leg]
        self._adr = [self.adr[j] for j in joint_names]
        self._cols = [names.index(j) for j in joint_names]
        self.base_pos = motion["base_pos_w"]
        self.base_quat = motion["base_quat_w"]

    def qpos(self, t: int, joints: np.ndarray) -> np.ndarray:
        q = np.zeros(self.model.nq)
        q[0:3] = self.base_pos[t]
        q[3:7] = self.base_quat[t]
        q[self._adr] = joints[self._cols]
        return q

    def fk(self, q: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        self.data.qpos[:] = q
        mujoco.mj_kinematics(self.model, self.data)
        return {s: (self.data.xpos[b].copy(), self.data.xmat[b].reshape(3, 3).copy()) for s, b in self.body.items()}

    def solve_leg(self, q0: np.ndarray, x0: np.ndarray, target_p: np.ndarray, target_R: np.ndarray, lo: np.ndarray, hi: np.ndarray,
                  adr: list[int], body: int, init: np.ndarray | None = None, w_temp: float = 0.3) -> tuple[np.ndarray, float, float]:  # fmt: skip
        """Least-squares IK of one leg: ankle position -> ``target_p``, foot orientation -> ``target_R``, stay near this frame's
        original angles ``x0``, and keep the frame-to-frame increment of the original motion (``self.prev`` / ``self.prev0``)."""
        data, model = self.data, self.model
        x_prev, delta0 = self.prev, self.prev0

        def residual(x: np.ndarray) -> np.ndarray:
            data.qpos[:] = q0
            data.qpos[adr] = x
            mujoco.mj_kinematics(model, data)
            rot = Rotation.from_matrix(data.xmat[body].reshape(3, 3) @ target_R.T).as_rotvec()
            return np.concatenate([(data.xpos[body] - target_p) * W_POS, rot * W_ORI, (x - x0) * W_REG,
                                   (x - x_prev - delta0) * w_temp])  # fmt: skip

        start = x_prev if init is None else init
        sol = least_squares(residual, np.clip(start, lo, hi), bounds=(lo, hi), xtol=1e-10, ftol=1e-10, gtol=1e-10, max_nfev=400)
        data.qpos[:] = q0
        data.qpos[adr] = sol.x
        mujoco.mj_kinematics(model, data)
        perr = float(np.linalg.norm(data.xpos[body] - target_p))
        oerr = float(np.linalg.norm(Rotation.from_matrix(data.xmat[body].reshape(3, 3) @ target_R.T).as_rotvec()))
        return sol.x, perr, oerr


def sole_half_width(rig: Rig, side: str) -> float:
    """Lateral half-width of the foot's collision geoms in the ankle body frame (capsules are along the geom's local z)."""
    bid = rig.body[side]
    half = 0.0
    for g in np.nonzero(rig.model.geom_bodyid == bid)[0]:
        if rig.model.geom_contype[g] == 0 and rig.model.geom_conaffinity[g] == 0:
            continue
        pos, quat, size = rig.model.geom_pos[g], rig.model.geom_quat[g], rig.model.geom_size[g]
        if rig.model.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE:
            axis = Rotation.from_quat(quat, scalar_first=True).as_matrix() @ np.array([0.0, 0.0, 1.0])
            half = max(half, abs(pos[1]) + size[0] + size[1] * abs(axis[1]))
        elif rig.model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
            half = max(half, abs(pos[1]) + size[1])
        elif rig.model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE:
            half = max(half, abs(pos[1]) + size[0])
    return half


def sole_centre(rig: Rig, side: str) -> np.ndarray:
    """Fore-aft centre of the foot's collision geoms in the ankle body frame (the ankle sits ~4 cm behind it)."""
    bid = rig.body[side]
    lo, hi = np.inf, -np.inf
    for g in np.nonzero(rig.model.geom_bodyid == bid)[0]:
        if rig.model.geom_contype[g] == 0 and rig.model.geom_conaffinity[g] == 0:
            continue
        pos, quat, size = rig.model.geom_pos[g], rig.model.geom_quat[g], rig.model.geom_size[g]
        if rig.model.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE:
            axis = Rotation.from_quat(quat, scalar_first=True).as_matrix() @ np.array([0.0, 0.0, 1.0])
            lo, hi = min(lo, pos[0] - size[1] * abs(axis[0]) - size[0]), max(hi, pos[0] + size[1] * abs(axis[0]) + size[0])
        elif rig.model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
            lo, hi = min(lo, pos[0] - size[0]), max(hi, pos[0] + size[0])
    return np.array([(lo + hi) / 2.0, 0.0])


def sole_lowest(rig: Rig, side: str) -> np.ndarray:
    """World positions of the sole's collision capsule surface (16 samples per capsule end)."""
    gid = rig.body[side]
    pts = []
    for g in np.nonzero(rig.model.geom_bodyid == gid)[0]:
        if rig.model.geom_contype[g] == 0 and rig.model.geom_conaffinity[g] == 0:
            continue
        if rig.model.geom_type[g] != mujoco.mjtGeom.mjGEOM_CAPSULE:
            continue
        c, R = rig.data.geom_xpos[g], rig.data.geom_xmat[g].reshape(3, 3)
        r, h = rig.model.geom_size[g][0], rig.model.geom_size[g][1]
        for s in (1, -1):
            cen = c + R[:, 2] * h * s
            for a in np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False):
                pts.append(cen + R @ np.array([np.cos(a), np.sin(a), 0.0]) * r)
    return np.array(pts)


def smooth(x: np.ndarray, k: int) -> np.ndarray:
    """Hann smoothing over ``k`` frames (edge-padded); ``k <= 1`` leaves ``x`` unchanged."""
    if k <= 1:
        return x
    w = np.hanning(k + 2)[1:-1]
    w = w / w.sum()
    pad = k // 2
    return np.convolve(np.pad(x, pad, mode="edge"), w, mode="valid")


def narrow_box(geoms: list[dict], max_half: float) -> dict | None:
    boxes = [g for g in geoms if str(g.get("type", "box")) == "box" and g["size"][1] < max_half and g["size"][0] > 2 * g["size"][1]]
    return min(boxes, key=lambda g: g["size"][1]) if boxes else None


def stake_boxes(geoms: list[dict], max_half: float) -> tuple[list[dict], float]:
    """Top-level boxes that are small in both directions (the stakes), and the top of the terrain."""
    top = max(g["pos"][2] + g["size"][2] for g in geoms)
    boxes = [g for g in geoms if abs(g["pos"][2] + g["size"][2] - top) < 1e-6
             and g["size"][0] < max_half and g["size"][1] < max_half]  # fmt: skip
    return boxes, float(top)


def runs(flag: np.ndarray, max_gap: int = 0, min_len: int = 1) -> list[tuple[int, int]]:
    """[start, stop) index runs of True, bridging gaps up to ``max_gap`` frames and dropping runs shorter than ``min_len``."""
    idx = np.nonzero(flag)[0]
    if len(idx) == 0:
        return []
    out: list[list[int]] = [[int(idx[0]), int(idx[0]) + 1]]
    for i in idx[1:]:
        if i - out[-1][1] <= max_gap:
            out[-1][1] = int(i) + 1
        else:
            out.append([int(i), int(i) + 1])
    return [(a, b) for a, b in out if b - a >= min_len]


def smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def solve_track(rig: Rig, motion: dict, fk: list[dict], qpos: np.ndarray, side: str, names: list[str],
                track: np.ndarray, w_temp: float) -> tuple[list[int], np.ndarray, np.ndarray, np.ndarray]:  # fmt: skip
    """Solve the leg IK for every frame whose offset is non-zero; ``track`` is ``(T, 2)`` of ankle offsets in xy (metres)."""
    joints = [names.index(j) for j in LEG_JOINTS[side]]
    lo = np.array([rig.range[j][0] for j in LEG_JOINTS[side]])
    hi = np.array([rig.range[j][1] for j in LEG_JOINTS[side]])
    adr = [rig.adr[j] for j in LEG_JOINTS[side]]
    cur = motion["joint_pos"][:, joints].copy()
    solved = cur.copy()
    perr = np.zeros(len(cur))
    for t in range(len(cur)):
        if not track[t].any():
            continue
        p, R = fk[t][side]
        rig.prev = solved[t - 1] if t > 0 else cur[t]
        rig.prev0 = cur[t] - (cur[t - 1] if t > 0 else cur[t])  # increment of the original motion into frame t
        target_p = p + np.array([track[t, 0], track[t, 1], 0.0])
        x, perr[t], _ = rig.solve_leg(qpos[t], cur[t], target_p, R, lo, hi, adr, rig.body[side], w_temp=w_temp)
        if perr[t] > 0.004:  # local minimum: retry from the smooth prediction (previous solution + source increment)
            x2, pe2, _ = rig.solve_leg(qpos[t], cur[t], target_p, R, lo, hi, adr, rig.body[side],
                                       init=rig.prev + rig.prev0, w_temp=w_temp)  # fmt: skip
            if pe2 < perr[t]:
                x, perr[t] = x2, pe2
        solved[t] = x
    return joints, cur, solved, perr


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", required=True, help="folder with metadata.yaml")
    parser.add_argument("--clips", default=None, help="regex on the motion_file path (default: every clip with such a support)")
    parser.add_argument("--mode", choices=("auto", "beam", "stakes"), default="auto", help="support kind (default: auto)")
    # beam mode
    parser.add_argument("--target-half", type=float, default=None, help="beam: desired |ankle y| (m); default: half-width - margin")
    parser.add_argument("--margin", type=float, default=0.048, help="beam: sole half-width (0.037) + this clearance to the edge")
    parser.add_argument("--min-target", type=float, default=0.045, help="beam: lower bound of the target half-stance")
    parser.add_argument("--max-target", type=float, default=0.065, help="beam: upper bound: do not build an unnaturally narrow stance")
    parser.add_argument("--ramp", type=float, default=0.35, help="beam: fade the offset in / out over this many metres of x at the box ends")
    # stake mode
    parser.add_argument("--merge-gap", type=int, default=5, help="stakes: bridge contact gaps up to this many frames inside a stance")
    parser.add_argument("--min-stance", type=int, default=6, help="stakes: ignore stance segments shorter than this")
    parser.add_argument("--sole-tol", type=float, default=0.03, help="stakes: ignore segments whose sole is higher than this above the stake top")
    parser.add_argument("--stake-target", choices=("ankle", "foot"), default="ankle",
                        help="stakes: put the ankle (plan foothold) or the foot's centre on the stake")  # fmt: skip
    parser.add_argument("--stance-tol", type=float, default=0.006, help="stakes: IK error (m) a scaled-down offset may leave on a stance")
    parser.add_argument("--limit-margin", type=float, default=0.02, help="stakes: keep the joints at least this far (rad) from their limits")
    # both / IK
    parser.add_argument("--max-box-half", type=float, default=0.25, help="beam: y half-width below this; stakes: both half-sizes below this")
    parser.add_argument("--tol", type=float, default=0.002, help="ignore offsets below this (m)")
    parser.add_argument("--smooth", type=int, default=5, help="Hann smoothing of the offset track (frames)")
    parser.add_argument("--max-move", type=float, default=0.12, help="cap of the correction (m): a wider swing is the source's business")
    parser.add_argument("--w-temp", type=float, default=0.3, help="weight of the source-increment term of the IK (0.1 for stakes)")
    parser.add_argument("--backup-tag", default=None, help="copy the replaced clips to <dataset>/_backup_<tag>/ first")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = os.path.abspath(os.path.expanduser(args.dataset))
    meta = yaml.safe_load(open(os.path.join(root, "metadata.yaml")))
    terrains = {t["terrain_id"]: t for t in meta["terrains"]}
    pat = re.compile(args.clips) if args.clips else None
    rig = Rig()
    sole = sole_half_width(rig, "left")
    centre = sole_centre(rig, "left")
    print(f"foot sole half-width {sole:.3f} m, sole centre {centre[0]:+.4f} m ahead of the ankle")

    for entry in meta["motion_files"]:
        rel = entry["motion_file"]
        if pat is not None and not pat.search(rel):
            continue
        geoms = yaml.safe_load(open(os.path.join(root, terrains[entry["terrain_id"]]["terrain_geoms"])))
        box = narrow_box(geoms, args.max_box_half)
        mode = args.mode
        if mode == "auto":
            mode = "beam" if box is not None else "stakes"
        if mode == "beam" and box is None:
            continue
        stakes, top = stake_boxes(geoms, args.max_box_half) if mode == "stakes" else ([], 0.0)
        if mode == "stakes" and len(stakes) < 2:
            continue

        motion = dict(np.load(os.path.join(root, rel), allow_pickle=True))
        names = [str(n) for n in motion["joint_names"]]
        T = len(motion["joint_pos"])
        rig.bind(motion)
        qpos = np.stack([rig.qpos(t, motion["joint_pos"][t]) for t in range(T)])
        fk = [rig.fk(q) for q in qpos]
        contacts = np.load(os.path.join(root, rel)[:-4] + ".contacts.npz", allow_pickle=True)
        links = [str(n) for n in contacts["link_names"]]
        flag = contacts["contact_flag"]
        new_joints = motion["joint_pos"].copy()

        if mode == "beam":
            hw = float(box["size"][1])
            x0, x1 = box["pos"][0] - box["size"][0], box["pos"][0] + box["size"][0]
            target = args.target_half if args.target_half is not None else min(args.max_target, max(args.min_target, hw - args.margin))
            for side in ("left", "right"):
                ank_y = np.array([f[side][0][1] for f in fk])
                ank_x = np.array([f[side][0][0] for f in fk])
                # per-frame clamp: the source's own lateral ankle motion is kept, the planted foot gains no slip of its own
                ramp = np.minimum((ank_x - (x0 - args.ramp)) / args.ramp, ((x1 + args.ramp) - ank_x) / args.ramp)
                weight = smoothstep(ramp)
                track1 = -np.sign(ank_y) * np.maximum(0.0, np.abs(ank_y) - target) * weight
                capped = int((np.abs(track1) > args.max_move).sum())
                track1 = np.clip(track1, -args.max_move, args.max_move)
                track1 = smooth(track1, args.smooth)
                track1[np.abs(track1) < args.tol] = 0.0
                if capped:
                    print(f"   {side:5s} capped the correction on {capped} frames (swing wider than {args.max_move * 1000:.0f} mm)")
                track = np.stack([np.zeros(T), track1], axis=1)
                joints, cur, solved, perr = solve_track(rig, motion, fk, qpos, side, names, track, args.w_temp)
                new_joints[:, joints] = solved
                d = np.degrees(np.abs(np.diff(solved, axis=0)))  # joint speed of the edited clip
                d0 = np.degrees(np.abs(np.diff(cur, axis=0)))  # joint speed of the source
                step = d.max() if T > 1 else 0.0
                print(f"   {side:5s} |y| {np.abs(ank_y).mean():.3f} (target {target:.3f}) | move max {np.abs(track1).max() * 1000:4.0f} mm "
                      f"| IK err p95 {np.percentile(perr, 95) * 1000:4.1f} mm | joint step max {step:.1f} (source {d0.max():.1f}) deg/frame")  # fmt: skip

        else:
            ctr = np.array([[g["pos"][0], g["pos"][1]] for g in stakes])
            half = np.array([[g["size"][0], g["size"][1]] for g in stakes])
            x0, x1 = float((ctr[:, 0] - half[:, 0]).min()), float((ctr[:, 0] + half[:, 0]).max())
            inside_x = (x0 - args.ramp, x1 + args.ramp)
            for side in ("left", "right"):
                # unit of the edit is the stance segment: a constant offset per stance, nothing on the platforms / in the air
                track = np.zeros((T, 2))
                filled = np.zeros(T, bool)
                segs = runs(flag[:, links.index(ANKLE_BODY[side])], max_gap=args.merge_gap, min_len=args.min_stance)
                kept, scaled, skipped = 0, 0, 0
                for a, b in segs:
                    am = min((a + b) // 2, T - 1)
                    p, R = fk[am][side]
                    rig.data.qpos[:] = qpos[am]
                    mujoco.mj_kinematics(rig.model, rig.data)
                    if not inside_x[0] <= p[0] <= inside_x[1]:  # platform at the clip ends
                        skipped += 1
                        continue
                    if sole_lowest(rig, side)[:, 2].min() - top > args.sole_tol:  # labelled contact but in the air
                        skipped += 1
                        continue
                    fc = p[:2] + R[:2, :2] @ centre
                    k = int(np.argmin(((ctr - p[:2]) ** 2).sum(1)))  # stake under the ankle, not under the toe
                    anchor = p[:2] if args.stake_target == "ankle" else fc
                    d = np.clip(ctr[k] - anchor, -args.max_move, args.max_move)
                    if np.linalg.norm(d) < args.tol:
                        skipped += 1
                        continue
                    best = None
                    for alpha in ALPHAS:  # the stance leg is nearly straight: keep only what the joints can follow
                        probe = np.zeros((T, 2))
                        probe[a:b] = alpha * d
                        _, _, x, pe = solve_track(rig, motion, fk, qpos, side, names, probe, args.w_temp)
                        lo = np.array([rig.range[j][0] for j in LEG_JOINTS[side]])
                        hi = np.array([rig.range[j][1] for j in LEG_JOINTS[side]])
                        margin = float(np.min(np.minimum(x[a:b] - lo, hi - x[a:b])))
                        best = (alpha, float(pe[a:b].max()), margin)
                        if best[1] <= args.stance_tol and margin >= args.limit_margin:
                            break
                    alpha = best[0]
                    if alpha < 1.0:
                        scaled += 1
                    kept += 1
                    track[a:b] = alpha * d
                    filled[a:b] = True
                if not filled.any():
                    print(f"   {side:5s} no stance segment on a stake (kept {kept}, skipped {skipped})")
                    continue
                # fade between two stances over the swing: the foot is in the air there, so this is a placement change
                idx = np.nonzero(filled)[0]
                for t in range(T):
                    if filled[t]:
                        continue
                    before, after = idx[idx < t], idx[idx > t]
                    if len(before) and len(after):
                        i, j = before[-1], after[0]
                        w = smoothstep(np.array([(t - i) / max(j - i, 1)]))[0]
                        track[t] = (1.0 - w) * track[i] + w * track[j]
                    else:
                        track[t] = track[before[-1]] if len(before) else track[after[0]]
                # the approach / exit on the wide platforms keeps its natural placement
                ank_x = np.array([f[side][0][0] for f in fk])
                ramp = np.minimum((ank_x - (x0 - args.ramp)) / args.ramp, ((x1 + args.ramp) - ank_x) / args.ramp)
                track *= smoothstep(ramp)[:, None]
                for c in range(2):
                    track[:, c] = smooth(track[:, c], args.smooth)
                track[np.abs(track).max(axis=1) < args.tol] = 0.0
                joints, cur, solved, perr = solve_track(rig, motion, fk, qpos, side, names, track, args.w_temp)
                new_joints[:, joints] = solved
                lo = np.array([rig.range[j][0] for j in LEG_JOINTS[side]])
                hi = np.array([rig.range[j][1] for j in LEG_JOINTS[side]])
                near_limit = int((np.minimum(solved - lo, hi - solved) < np.radians(1.0)).any(axis=1).sum())
                print(f"   {side:5s} stances {kept} (scaled {scaled}, skipped {skipped}) | move max {np.abs(track).max() * 1000:4.0f} mm "
                      f"| IK err p95 {np.percentile(perr, 95) * 1000:4.1f} mm | joints at limit {near_limit}/{T} frames")  # fmt: skip

        qpos_new = [rig.qpos(t, new_joints[t]) for t in range(T)]
        fk_new = [rig.fk(q) for q in qpos_new]
        if mode == "beam":
            print(f"== {rel}  support half-width {hw:.3f} m, x [{x0:.2f}, {x1:.2f}], {T} frames")
            for side in ("left", "right"):
                y_old = np.array([f[side][0][1] for f in fk])
                y_new = np.array([f[side][0][1] for f in fk_new])
                x_new = np.array([f[side][0][0] for f in fk_new])
                on = flag[:, links.index(ANKLE_BODY[side])] & (x_new > x0) & (x_new < x1)  # supports on the narrow box only
                outside = (np.abs(y_new) + sole - hw) > 0
                print(f"   {side:5s} |y| mean {np.abs(y_old).mean():.3f} -> {np.abs(y_new).mean():.3f} | "
                      f"on-support max {np.abs(y_new[on]).max() if on.any() else float('nan'):.3f} (edge {hw:.3f}) | "
                      f"sole outside {int(outside[on].sum())}/{int(on.sum())} support frames")  # fmt: skip
        else:
            print(f"== {rel}  {len(stakes)} stakes, side {2 * half[:, 0].mean():.2f} m, top {top:.2f} m, x [{x0:.2f}, {x1:.2f}], {T} frames")
            for side in ("left", "right"):
                on = flag[:, links.index(ANKLE_BODY[side])]
                for tag, ffk in (("before", fk), ("after ", fk_new)):
                    ank = np.array([f[side][0][:2] for f in ffk])
                    fc = np.array([f[side][0][:2] + f[side][1][:2, :2] @ centre for f in ffk])
                    d = ank[:, None, :] - ctr[None]
                    k = np.argmin((d ** 2).sum(2), axis=1)
                    ank_dist = np.linalg.norm(d[np.arange(T), k], axis=1)
                    fc_off = np.linalg.norm(fc - ctr[k], axis=1)
                    ov = (np.abs(fc - ctr[k]) - half[k]).max(1) / half[k].max()
                    sel = on & (ank[:, 0] > x0) & (ank[:, 0] < x1)  # stance frames on the stakes, not on the end platforms
                    print(f"   {side:5s} {tag} ankle -> stake {100 * ank_dist[sel].mean():4.1f} cm, "
                          f"foot centre -> stake {100 * fc_off[sel].mean():4.1f} cm, "
                          f"foot outside {100 * (ov[sel] > 0).mean():3.0f}% of {int(sel.sum())} stance frames")  # fmt: skip
        if args.dry_run:
            continue
        if args.backup_tag:
            for src in (rel, rel[:-4] + ".contacts.npz"):
                dst = os.path.join(root, f"_backup_{args.backup_tag}", src)
                if os.path.exists(os.path.join(root, src)) and not os.path.exists(dst):
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    shutil.copy2(os.path.join(root, src), dst)
        out = dict(motion)
        out["joint_pos"] = new_joints.astype(motion["joint_pos"].dtype)
        np.savez(os.path.join(root, rel), **out)

    print(f"relabel next: python -m instinct_mj.scripts.acp.label_contacts --dataset {root} --overwrite --workers 4", file=sys.stderr)


if __name__ == "__main__":
    main()
