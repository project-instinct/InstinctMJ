"""Minimal, local penetration repair of pelvis-rooted ACP motions.

By default only geometry penetration is repaired. There is no contact-attraction, foot locking,
sliding correction, retiming or source-motion denoising. Terrain distances minimize
the analytic SDF of EACH convex primitive along EACH collision capsule axis, before
taking the minimum over primitives. Final checks use float64 and a conservative
search-error bound; optimization uses refreshed closest-point witnesses.

Smooth cubic-spline pose corrections live only in padded penetration windows, with
C2 quintic envelopes and three unchanged boundary frames. Failed candidates never
replace source motions. NPZ fields outside those windows remain bit-identical;
unrepaired files, labels, metadata and terrains are copied byte-for-byte.

Examples (run GPU commands in an acp-pen-* systemd unit with MemoryMax=16G):
  python -m instinct_mj.scripts.acp.fix_penetration scan --dataset DATA --report scan.json
  python -m instinct_mj.scripts.acp.fix_penetration repair --dataset DATA --out OUT --copy-dataset --report repair.json
  python -m instinct_mj.scripts.acp.fix_penetration verify --dataset DATA --out OUT --report verify.json

Default scope is climb / box / ladder / vault. Sparse requires explicit --scope
sparse after its separately managed source revision is ready. No source is edited.
Explicit --repair-jumps additionally repairs source joint steps >0.35 rad/frame
inside local windows. Use fine knots (e.g. 0.02 s at 50 Hz); pelvis orientation
stays byte-identical and joint corrections are capped at 9.8 degrees. It does not
enable general denoising or change the default penetration-only behavior.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import torch
import yaml
from scipy.interpolate import BSpline
from scipy.spatial.transform import Rotation

from instinct_mj.assets.acp_g1 import get_acp_g1_spec
from instinct_mj.rl.acp.planner import CONTACT_LINKS
from instinct_mj.rl.supermimic.rotations import axis_angle_to_mat, quat_to_mat, so3_exp

KINDS = dict(plane=0, sphere=1, capsule=2, cylinder=4, box=6)
SPARSE = {"stakes", "stones", "paired_stones", "pallets", "sparse_mix", "sparse_seq", "beam"}
VERSION = 1


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False))
    os.replace(tmp, path)


def in_scope(rel, scope="primary"):
    cat = rel.split("/")[0]
    primary = cat.startswith(("climb_", "high-climb", "low-climb", "box_up", "box_down", "ladder", "vault"))
    return primary if scope == "primary" else cat in SPARSE if scope == "sparse" else primary or cat in SPARSE


def paired_sdf(x, size, kind):
    """Local-frame point SDF, one convex terrain primitive per row. No smoothing bias."""
    q = x.abs() - size
    box = q.clamp(min=0).norm(dim=-1) + q.amax(-1).clamp(max=0)
    cylq = torch.stack((x[..., :2].norm(dim=-1) - size[..., 0], x[..., 2].abs() - size[..., 1]), -1)
    cylinder = cylq.clamp(min=0).norm(dim=-1) + cylq.amax(-1).clamp(max=0)
    z = x[..., 2] - x[..., 2].clamp(-size[..., 1], size[..., 1])
    capsule = torch.stack((x[..., 0], x[..., 1], z), -1).norm(dim=-1) - size[..., 0]
    sphere = x.norm(dim=-1) - size[..., 0]
    return torch.where(
        kind == 0,
        x[..., 2],
        torch.where(kind == 1, sphere, torch.where(kind == 2, capsule, torch.where(kind == 4, cylinder, box))),
    )


@torch.no_grad()
def segment_minimum(a, b, size, kind, iterations=32):
    """Global convex 1-D minimization per segment/primitive pair, including endpoints.

    Returns SDF, minimizing parameter, and a conservative 1-Lipschitz search bound.
    Crucially, this never searches the nonconvex union SDF as a single function.
    """
    phi = (math.sqrt(5) - 1) / 2
    lo, hi = torch.zeros_like(a[:, 0]), torch.ones_like(a[:, 0])
    c, d = hi - phi * (hi - lo), lo + phi * (hi - lo)
    delta = b - a
    f = lambda t: paired_sdf(a + t[:, None] * delta, size, kind)
    fc, fd = f(c), f(d)
    for _ in range(iterations):
        left = fc < fd
        hi, lo = torch.where(left, d, hi), torch.where(left, lo, c)
        new = torch.where(left, hi - phi * (hi - lo), lo + phi * (hi - lo))
        fn = f(new)
        c, d, fc, fd = (
            torch.where(left, new, d),
            torch.where(left, c, new),
            torch.where(left, fn, fd),
            torch.where(left, fc, fn),
        )
    t = (lo + hi) / 2
    fmid, fa, fb = f(t), f(lo * 0), f(hi * 0 + 1)
    v = torch.stack((fmid, fa, fb), -1)
    k = v.argmin(-1)
    t = torch.where(k == 1, 0.0, torch.where(k == 2, 1.0, t))
    error = delta.norm(dim=-1) * (hi - lo) / 2
    error += 16 * torch.finfo(a.dtype).eps * (1 + a.abs().amax(-1) + b.abs().amax(-1) + size.abs().amax(-1))
    return v.amin(-1), t, error


class Terrain:
    def __init__(self, geoms, device="cpu", dtype=torch.float32):
        if not geoms or any(g["type"] not in KINDS for g in geoms):
            raise ValueError("terrain requires box / cylinder / capsule / sphere / plane primitives")
        t = lambda x: torch.as_tensor(np.asarray(x), device=device, dtype=dtype)
        self.pos = t([g["pos"] for g in geoms])
        self.rot = quat_to_mat(t([g.get("quat", [1, 0, 0, 0]) for g in geoms]))
        self.size = t([list(g["size"]) + [0.0] * (3 - len(g["size"])) for g in geoms])
        self.kind = torch.tensor([KINDS[g["type"]] for g in geoms], device=device)
        ext = self.size.clone()
        for i, g in enumerate(geoms):
            r, h = float(self.size[i, 0]), float(self.size[i, 1])
            if g["type"] == "sphere":
                ext[i] = r
            if g["type"] == "capsule":
                ext[i] = t([r, r, h + r])
            if g["type"] == "cylinder":
                ext[i] = t([r, r, h])
        world = torch.einsum("kij,kj->ki", self.rot.abs(), ext)
        world[self.kind == 0] = float("inf")
        self.low, self.high = self.pos - world, self.pos + world

    def local(self, x, idx):
        # R maps LOCAL columns to world; row-vector inverse transform is (x-pos) @ R.
        return torch.einsum("ni,nij->nj", x - self.pos[idx], self.rot[idx])

    @torch.no_grad()
    def witnesses(self, a, b, radius, iterations=32, cutoff=0.03):
        """Flattened segments -> conservative distances and all potentially near pairs.

        AABB pruning only discards pairs whose separation exceeds cutoff. Thus the
        reported positive distances are capped at cutoff, but every zero/negative
        distance and every minimum below cutoff is evaluated continuously.
        """
        low, high = torch.minimum(a, b), torch.maximum(a, b)
        sep = torch.maximum(self.low[None] - high[:, None], low[:, None] - self.high[None]).clamp(min=0)
        bound = sep.norm(dim=-1) - radius[:, None]
        row, geom = (bound <= cutoff).nonzero(as_tuple=True)
        dist = a.new_full((len(a),), cutoff)
        if len(row) == 0:
            return dist, (row, geom, a.new_empty(0))
        aa, bb = self.local(a[row], geom), self.local(b[row], geom)
        vals, ts, errs = [], [], []
        for start in range(0, len(row), 131072):
            sl = slice(start, start + 131072)
            v, t, e = segment_minimum(aa[sl], bb[sl], self.size[geom[sl]], self.kind[geom[sl]], iterations)
            vals.append(v)
            ts.append(t)
            errs.append(e)
        v, t, error = torch.cat(vals), torch.cat(ts), torch.cat(errs)
        dist.scatter_reduce_(0, row, v - radius[row] - error, reduce="amin", include_self=True)
        return dist, (row, geom, t)


class Robot:
    """Task MJCF geometry and FK, without importing the old motion repairer."""

    def __init__(self, device="cpu", dtype=torch.float32):
        self.model = m = get_acp_g1_spec(simplified_collision=True).compile()
        self.device, self.dtype = device, dtype
        t = lambda x: torch.as_tensor(np.asarray(x), device=device, dtype=dtype)
        self.t = t
        self.names = [m.joint(j).name for j in range(1, m.njnt)]
        self.ranges = t(m.jnt_range[1:])
        self.parent = m.body_parentid.tolist()
        self.offset, self.rotation = t(m.body_pos), quat_to_mat(t(m.body_quat))
        self.joints = {}
        for j in range(1, m.njnt):
            assert m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE and not np.any(m.jnt_pos[j])
            assert m.jnt_qposadr[j] == 7 + j - 1
            self.joints[int(m.jnt_bodyid[j])] = (j - 1, t(m.jnt_axis[j]))
        ids = np.nonzero((m.geom_contype != 0) | (m.geom_conaffinity != 0))[0]
        assert all(m.geom_type[g] in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_SPHERE) for g in ids)
        self.ids = ids
        self.owner = torch.tensor(m.geom_bodyid[ids], device=device)
        self.body_names = [m.body(i).name for i in range(m.nbody)]
        self.geom_names = [m.geom(g).name for g in ids]
        rot = quat_to_mat(t(m.geom_quat[ids]))
        half = t(m.geom_size[ids, 1])
        half[t(m.geom_type[ids]) == int(mujoco.mjtGeom.mjGEOM_SPHERE)] = 0
        self.seg_a = t(m.geom_pos[ids]) - rot[:, :, 2] * half[:, None]
        self.seg_b = t(m.geom_pos[ids]) + rot[:, :, 2] * half[:, None]
        self.radius = t(m.geom_size[ids, 0])
        self.contact_geoms = [
            torch.tensor(np.nonzero(m.geom_bodyid[ids] == m.body(n).id)[0], device=device) for n in CONTACT_LINKS
        ]
        h = hashlib.sha256()
        for name in (
            "geom_type",
            "geom_size",
            "geom_pos",
            "geom_quat",
            "geom_bodyid",
            "geom_contype",
            "geom_conaffinity",
            "body_parentid",
            "body_pos",
            "body_quat",
            "jnt_axis",
            "jnt_range",
        ):
            h.update(np.asarray(getattr(m, name)).tobytes())
        self.signature = h.hexdigest()

    def fk(self, p, R, q):
        ps, rs = [p * 0, p], [R * 0, R]
        for b in range(2, self.model.nbody):
            parent = self.parent[b]
            ps.append(ps[parent] + (rs[parent] @ self.offset[b, :, None]).squeeze(-1))
            rr = rs[parent] @ self.rotation[b]
            if b in self.joints:
                j, axis = self.joints[b]
                rr = rr @ axis_angle_to_mat(axis.expand(q[..., j].shape + (3,)), q[..., j])
            rs.append(rr)
        return torch.stack(ps, -2), torch.stack(rs, -3)

    def segments(self, p, R, q):
        bp, br = self.fk(p, R, q)
        a = bp[..., self.owner, :] + (br[..., self.owner, :, :] @ self.seg_a[..., None]).squeeze(-1)
        b = bp[..., self.owner, :] + (br[..., self.owner, :, :] @ self.seg_b[..., None]).squeeze(-1)
        return a, b, bp

    @torch.no_grad()
    def measure(self, motion, terrain, chunk=128):
        p, R, q, cols = self.pose(motion)
        distances = []
        for start in range(0, len(p), chunk):
            a, b, _ = self.segments(p[start : start + chunk], R[start : start + chunk], q[start : start + chunk])
            rad = self.radius.expand(len(a), -1).reshape(-1)
            d, _ = terrain.witnesses(a.reshape(-1, 3), b.reshape(-1, 3), rad, iterations=36)
            distances.append(d.reshape(len(a), -1).cpu().numpy())
        return np.concatenate(distances)

    def pose(self, motion):
        cols = [list(motion["joint_names"]).index(n) for n in self.names]
        return (
            self.t(motion["base_pos_w"]),
            quat_to_mat(self.t(motion["base_quat_w"])),
            self.t(motion["joint_pos"][:, cols]),
            cols,
        )


def windows(mask, fps, padding=0.45):
    """[start,end) windows; include three exact boundary samples beyond the C2 ramp."""
    idx = np.nonzero(mask)[0]
    if not len(idx):
        return []
    pad = int(math.ceil(padding * fps)) + 3
    spans = []
    for i in idx:
        lo, hi = max(0, int(i) - pad), min(len(mask), int(i) + pad + 1)
        if spans and lo <= spans[-1][1]:
            spans[-1][1] = max(hi, spans[-1][1])
        else:
            spans.append([lo, hi])
    return [tuple(x) for x in spans]


def repair_mask(motion, distances, repair_jumps=False):
    """Penetration frames, optionally including BOTH sides of authorized source jumps."""
    mask = np.asarray(distances).min(-1) < 0
    if repair_jumps:
        jumps = np.any(np.abs(np.diff(motion["joint_pos"], axis=0)) > 0.35, axis=1)
        mask[:-1] |= jumps
        mask[1:] |= jumps
    return mask


def window_basis(start, end, total, fps, padding=0.45, knot_seconds=0.16):
    """Cubic B-spline basis times C2 envelope; interior boundaries keep 3 exact frames."""
    n = end - start
    interior = np.arange(knot_seconds * fps, n - 1, knot_seconds * fps)
    knots = np.r_[[0.0] * 4, interior, [float(n - 1)] * 4]
    B = BSpline.design_matrix(np.arange(n), knots, 3).toarray()

    def smooth(x):
        x = np.clip(x, 0, 1)
        return x * x * x * (10 + x * (-15 + 6 * x))

    env = np.ones(n)
    if start > 0:
        env *= smooth((np.arange(n) - 2) / (padding * fps))
    if end < total:
        env *= smooth((n - 3 - np.arange(n)) / (padding * fps))
    return B * env[:, None]


@dataclass
class Job:
    rel: str
    motion: dict
    terrain: Terrain
    labels: np.ndarray
    before: np.ndarray
    spans: list
    source_sha: str
    input_sha: str


def derivative_stats(x):
    out = {}
    for order, name in ((1, "velocity"), (2, "acceleration"), (3, "jerk")):
        d = np.diff(x, order, axis=0)
        if not len(d):
            d = np.zeros((1, x.shape[-1]))
        out[name] = {
            "p99": float(np.percentile(np.abs(d), 99)),
            "max": float(np.abs(d).max()),
            "energy": float(np.mean(d * d)),
        }
    return out


def capsule_pair_distance(a, b, c, d, radius):
    """Exact segment-segment distance: valid interior stationary point plus four boundaries."""
    u, v, w = b - a, d - c, a - c
    aa = (u * u).sum(-1)
    bb = (u * v).sum(-1)
    cc = (v * v).sum(-1)
    dd = (u * w).sum(-1)
    ee = (v * w).sum(-1)
    den = aa * cc - bb * bb
    safe = den.clamp(min=1e-20)
    s = (bb * ee - cc * dd) / safe
    t = (aa * ee - bb * dd) / safe
    interior = (w + s[..., None] * u - t[..., None] * v).norm(dim=-1)
    valid = (den > 1e-14) & (s >= 0) & (s <= 1) & (t >= 0) & (t <= 1)
    interior = torch.where(valid, interior, torch.full_like(interior, float("inf")))

    def point_seg(x, lo, hi):
        vv = hi - lo
        tt = ((x - lo) * vv).sum(-1) / (vv * vv).sum(-1).clamp(min=1e-20)
        return (x - lo - tt.clamp(0, 1)[..., None] * vv).norm(dim=-1)

    return (
        torch.stack(
            (interior, point_seg(a, c, d), point_seg(b, c, d), point_seg(c, a, b), point_seg(d, a, b)), -1
        ).amin(-1)
        - radius
    )


def self_contacts(robot, motion):
    """Per-frame active MuJoCo self-contact pair depths (including model exclusions)."""
    m = robot.model
    d = mujoco.MjData(m)
    names = list(motion["joint_names"])
    cols = [names.index(n) for n in robot.names]
    out = []
    for p, r, q in zip(motion["base_pos_w"], motion["base_quat_w"], motion["joint_pos"][:, cols]):
        d.qpos[:3] = p
        d.qpos[3:7] = r
        d.qpos[7:] = q
        mujoco.mj_kinematics(m, d)
        mujoco.mj_collision(m, d)
        pairs = {}
        for c in d.contact[: d.ncon]:
            key = tuple(sorted((int(c.geom1), int(c.geom2))))
            if c.dist < 0:
                pairs[key] = min(float(c.dist), pairs.get(key, 0.0))
        out.append(pairs)
    return out


def candidate_metrics(robot, source, candidate, before, after, changed):
    p0, r0, q0, _ = robot.pose(source)
    p, r, q, _ = robot.pose(candidate)
    with torch.no_grad():
        bp0, _ = robot.fk(p0, r0, q0)
        bp, _ = robot.fk(p, r, q)
    dq = (q - q0).cpu().numpy()
    dp = (p - p0).cpu().numpy()
    result = dict(
        min_sd_before=float(before.min()),
        min_sd_after=float(after.min()),
        joint_max_deg=float(np.degrees(np.abs(dq).max())),
        root_max_cm=float(100 * np.linalg.norm(dp, axis=-1).max()),
        body_max_cm=float(100 * (bp - bp0).norm(dim=-1).max()),
        joint_limit_count=int(((q < robot.ranges[:, 0] - 1e-6) | (q > robot.ranges[:, 1] + 1e-6)).sum()),
        joint_jump_count=int((torch.diff(q, dim=0).abs() > 0.35).sum()),
    )
    out = []
    for key, limit in [
        ("joint_max_deg", 10.0),
        ("root_max_cm", 5.0),
        ("body_max_cm", 8.0),
        ("joint_limit_count", 0),
        ("joint_jump_count", 0),
    ]:
        if result[key] > limit:
            out.append(f"{key}: {result[key]:.6g} > {limit}")
    if not np.isfinite(after).all() or after.min() < 0:
        out.append("penetration or nonfinite distance")
    result["dynamics"] = {}
    for name, x0, x in [("joint", q0.cpu().numpy(), q.cpu().numpy()), ("root", p0.cpu().numpy(), p.cpu().numpy())]:
        bf, af = derivative_stats(x0), derivative_stats(x)
        result["dynamics"][name] = {"before": bf, "after": af}
        for order in ("acceleration", "jerk"):
            if af[order]["p99"] > 1.2 * bf[order]["p99"] + 1e-9:
                out.append(f"{name} {order} p99 grew >20%")
            if af[order]["max"] > 1.2 * bf[order]["max"] + 1e-8:
                out.append(f"{name} {order} peak grew >20%")
        if af["jerk"]["energy"] > 1.2 * bf["jerk"]["energy"] + 1e-12:
            out.append(f"{name} high-frequency energy grew >20%")

    # Angular motion is measured on SO(3), independent of quaternion signs.
    def angular(rs):
        rr = rs.cpu().numpy()
        return Rotation.from_matrix(rr[1:] @ rr[:-1].transpose(0, 2, 1)).as_rotvec()

    ba, aa = derivative_stats(angular(r0)), derivative_stats(angular(r))
    result["dynamics"]["root_rotation"] = {"before": ba, "after": aa}
    for key in ("velocity", "acceleration"):
        if aa[key]["p99"] > 1.2 * ba[key]["p99"] + 1e-9 or aa[key]["max"] > 1.2 * ba[key]["max"] + 1e-8:
            out.append(f"root angular {key} grew >20%")
    for key in ("joint_pos", "base_pos_w", "base_quat_w"):
        if source[key][~changed].tobytes() != candidate[key][~changed].tobytes():
            out.append("changed frame outside windows: " + key)
    if not all(np.isfinite(candidate[k]).all() for k in ("joint_pos", "base_pos_w", "base_quat_w")):
        out.append("nonfinite pose")
    # Compare PAIRS on the SAME FRAME, not just the fraction of bad frames.
    old, new = self_contacts(robot, source), self_contacts(robot, candidate)
    bad = []
    for t in np.nonzero(changed)[0]:
        for pair, depth in new[t].items():
            if depth < min(old[t].get(pair, 0.0), 0.0) - 1e-5:
                bad.append([int(t), list(pair), depth, old[t].get(pair, 0.0)])
    result["new_self_collision_count"] = len(bad)
    result["new_self_collision_examples"] = bad[:8]
    if bad:
        out.append("new/deeper self collision (>10 micrometres)")
    result["failures"] = out
    return result


def optimize_jobs(jobs, robot, args):
    """Multiple clips/windows share one batched FK and optimizer; all losses use the same frame normalization."""
    device = robot.device
    chunks = []
    bases = []
    ps = []
    rs = []
    qs = []
    offset = 0
    knots = 0
    for ji, j in enumerate(jobs):
        p, R, q, cols = robot.pose(j.motion)
        for st, en in j.spans:
            base = robot.t(window_basis(st, en, len(p), float(j.motion["framerate"]), args.padding, args.knot_seconds))
            chunks.append((ji, st, en, slice(offset, offset + en - st), slice(knots, knots + base.shape[1])))
            bases.append(base)
            ps.append(p[st:en])
            rs.append(R[st:en])
            qs.append(q[st:en])
            offset += en - st
            knots += base.shape[1]
    p0, R0, q0 = torch.cat(ps), torch.cat(rs), torch.cat(qs)
    basis = p0.new_zeros(offset, knots)
    for c, B in zip(chunks, bases):
        basis[c[3], c[4]] = B
    theta = torch.zeros(knots, 29, device=device, requires_grad=True)
    scale = robot.t([0.01] * 3 + [0.02] * 3 + [0.05] * 23)
    with torch.no_grad():
        a0, b0, bp0 = robot.segments(p0, R0, q0)
    G = len(robot.radius)
    radii = robot.radius
    # Use only pairs actually enabled by MuJoCo (observed in source / an earlier rejected
    # candidate), avoiding constraints on intentionally overlapping adjacent geoms.
    pair_terms = []
    geom_index = {int(g): i for i, g in enumerate(robot.ids)}
    for ji, j in enumerate(jobs):
        source_pairs = self_contacts(robot, j.motion)
        pairs = {pair for frame in source_pairs for pair in frame}
        seed_path = Path(args.seed_candidates) / j.rel if args.seed_candidates else None
        seed = dict(np.load(seed_path)) if seed_path and seed_path.is_file() else None
        if seed is not None:
            pairs |= {pair for frame in self_contacts(robot, seed) for pair in frame}
        pairs = sorted(pair for pair in pairs if all(g in geom_index for g in pair))
        ix = torch.tensor([geom_index[x] for x, y in pairs], device=device, dtype=torch.long)
        iy = torch.tensor([geom_index[y] for x, y in pairs], device=device, dtype=torch.long)
        for cj, st, en, sl, ks in chunks:
            if cj != ji:
                continue
            if len(pairs):
                old = capsule_pair_distance(
                    a0[sl][:, ix], b0[sl][:, ix], a0[sl][:, iy], b0[sl][:, iy], radii[ix] + radii[iy]
                )
                pair_terms.append((sl, ix, iy, old.clamp(max=0).detach()))
            if seed is not None:
                sp, sR, sq, _ = robot.pose(seed)
                dr = Rotation.from_matrix(
                    (sR[st:en] @ R0[sl].transpose(-1, -2)).cpu().numpy().astype(np.float64)
                ).as_rotvec()
                target = torch.cat((sp[st:en] - p0[sl], robot.t(dr), sq[st:en] - q0[sl]), -1) / scale
                coef = np.linalg.lstsq(basis[sl, ks].cpu().numpy(), target.cpu().numpy(), rcond=1e-6)[0]
                with torch.no_grad():
                    theta[ks] = robot.t(coef)
    # Original material-point witnesses on labelled contact bodies; per-frame, no stance anchors.
    anchors = []
    for ji, st, en, sl, _ in chunks:
        j = jobs[ji]
        aa = a0[sl].reshape(-1, 3)
        bb = b0[sl].reshape(-1, 3)
        vals, (row, gi, t) = j.terrain.witnesses(
            aa, bb, radii.expand(en - st, -1).reshape(-1), iterations=28, cutoff=float("inf")
        )
        x = aa[row] + t[:, None] * (bb[row] - aa[row])
        d = paired_sdf(j.terrain.local(x, gi), j.terrain.size[gi], j.terrain.kind[gi]) - radii[row % G]
        for bi, igs in enumerate(robot.contact_geoms):
            flags = np.nonzero(j.labels[st:en, bi])[0]
            for frame in flags:
                mask = (row // G == frame) & torch.isin(row % G, igs)
                ii = mask.nonzero().flatten()
                if not len(ii):
                    continue
                k = ii[d[ii].argmin()]
                f = int(row[k] // G) + sl.start
                g = int(row[k] % G)
                tg = int(gi[k])
                anchors.append((f, g, j.terrain, tg, t[k], x[k].detach(), d[k].detach()))
    # Group anchors by terrain to keep optimization vectorized.
    anchor_groups = []
    for j in jobs:
        arr = [x for x in anchors if x[2] is j.terrain]
        if not arr:
            continue
        fi = torch.tensor([x[0] for x in arr], device=device)
        gg = torch.tensor([x[1] for x in arr], device=device)
        tg = torch.tensor([x[3] for x in arr], device=device)
        ts = torch.stack([x[4] for x in arr])
        xs = torch.stack([x[5] for x in arr])
        ds = torch.stack([x[6] for x in arr])
        with torch.enable_grad():
            xx = xs.detach().requires_grad_()
            dd = paired_sdf(j.terrain.local(xx, tg), j.terrain.size[tg], j.terrain.kind[tg])
            normal = torch.autograd.grad(dd.sum(), xx)[0].detach()
        anchor_groups.append((j.terrain, fi, gg, tg, ts, xs, ds, normal))
    opt = torch.optim.Adam([theta], lr=args.learning_rate or (0.03 if args.repair_jumps else 0.12))
    witness = []
    best = None
    best_loss = float("inf")
    start = time.time()
    for it in range(args.iterations):
        delta = (basis @ theta) * scale
        if args.repair_jumps or args.lock_root_rotation:
            # Source jump repair needs joint freedom, not added pelvis rotations.
            delta = delta * delta.new_tensor([1] * 3 + [0] * 3 + [1] * 23)
        if args.repair_jumps:
            delta = torch.cat((delta[:, :6], delta[:, 6:].clamp(-math.radians(9.8), math.radians(9.8))), -1)
        p = p0 + delta[:, :3]
        R = so3_exp(delta[:, 3:6]) @ R0
        q = (q0 + delta[:, 6:]).clamp(robot.ranges[:, 0], robot.ranges[:, 1])
        a, b, bp = robot.segments(p, R, q)
        if it % args.refresh == 0:
            witness = []
            for ji, st, en, sl, _ in chunks:
                ter = jobs[ji].terrain
                _, (row, gi, t) = ter.witnesses(
                    a[sl].detach().reshape(-1, 3),
                    b[sl].detach().reshape(-1, 3),
                    radii.expand(en - st, -1).reshape(-1),
                    iterations=20,
                )
                witness.append((ter, row + sl.start * G, gi, t))
        aa, bb = a.reshape(-1, 3), b.reshape(-1, 3)
        pen = p.new_zeros(())
        mind = 1.0
        for ter, row, gi, t in witness:
            x = aa[row] + t[:, None] * (bb[row] - aa[row])
            d = paired_sdf(ter.local(x, gi), ter.size[gi], ter.kind[gi]) - radii[row % G]
            pen += torch.relu(args.safety - d).square().sum() / offset
            if it % args.refresh == 0:
                mind = min(mind, float(d.detach().min()))
        keep = p.new_zeros(())
        for ter, fi, gg, tg, ts, xs, ds, normal in anchor_groups:
            x = a[fi, gg] + ts[:, None] * (b[fi, gg] - a[fi, gg])
            d = paired_sdf(ter.local(x, tg), ter.size[tg], ter.kind[tg]) - radii[gg]
            disp = x - xs
            tang = disp - (disp * normal).sum(-1, keepdim=True) * normal
            # Optimize inside the final allowance, leaving room for penalty residuals and serialization.
            keep_slack = max(0.0, args.contact_slack - 0.0005)
            keep += (
                torch.relu(d - (ds.clamp(min=0) + keep_slack)).square().sum()
                + torch.relu(tang.norm(dim=-1) - keep_slack).square().sum()
            ) / offset
        fit = (
            5 * delta[:, :3].square().sum()
            + 0.2 * delta[:, 3:6].square().sum()
            + 0.02 * delta[:, 6:].square().sum()
            + 2 * (bp - bp0).square().sum() / robot.model.nbody
        ) / offset
        smooth = p.new_zeros(())
        jump_loss = p.new_zeros(())
        for _, _, _, sl, _ in chunks:
            v = delta[sl] / scale
            smooth += (
                torch.diff(v, 2, dim=0).square().sum() * 0.02 + torch.diff(v, 3, dim=0).square().sum() * 0.1
            ) / offset
            if args.repair_jumps:
                jump_loss += torch.relu(torch.diff(q[sl], dim=0).abs() - 0.335).square().sum() / offset
                # Penalize only newly increased acceleration/jerk, allowing the
                # source discontinuity itself to get substantially smoother.
                for order in (2, 3):
                    d0, d = torch.diff(q0[sl], order, dim=0), torch.diff(q[sl], order, dim=0)
                    jump_loss += 0.1 * torch.relu(d.abs() - d0.abs()).square().sum() / offset
        limits = (
            torch.relu(robot.ranges[:, 0] - q).square() + torch.relu(q - robot.ranges[:, 1]).square()
        ).sum() / offset
        sizecap = (
            torch.relu(delta[:, 6:].abs() - math.radians(10)).square().sum()
            + torch.relu(delta[:, :3].norm(dim=-1) - 0.05).square().sum()
        ) / offset
        self_pen = p.new_zeros(())
        for sl, ix, iy, old in pair_terms:
            now = capsule_pair_distance(a[sl][:, ix], b[sl][:, ix], a[sl][:, iy], b[sl][:, iy], radii[ix] + radii[iy])
            self_pen += torch.relu(old + 0.00005 - now).square().sum() / offset
        loss = (
            fit
            + smooth
            + args.pen_weight * pen
            + args.keep_weight * keep
            + 20000 * self_pen
            + 1000 * (limits + sizecap)
            + 10000 * jump_loss
        )
        if not torch.isfinite(loss):
            break
        if (
            it % args.refresh == 0
            and mind >= args.safety * 0.7
            and float(keep.detach()) < 1e-9
            and float(limits.detach()) < 1e-12
            and float(self_pen.detach()) < 1e-11
            and float(jump_loss.detach()) < 1e-9
        ):
            val = float((fit + smooth).detach())
            if val < best_loss:
                best_loss = val
                best = theta.detach().clone()
            if it >= 80:
                break
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([theta], 10.0)
        opt.step()
        if it % 50 == 0:
            print(
                f"  iter {it} frames={offset} min={mind*1000:.3f}mm loss={float(loss.detach()):.6g} elapsed={time.time()-start:.1f}s",
                flush=True,
            )
    if best is not None:
        theta = best
    with torch.no_grad():
        delta = (basis @ theta) * scale
        if args.repair_jumps or args.lock_root_rotation:
            delta = delta * delta.new_tensor([1] * 3 + [0] * 3 + [1] * 23)
        if args.repair_jumps:
            delta = torch.cat((delta[:, :6], delta[:, 6:].clamp(-math.radians(9.8), math.radians(9.8))), -1)
        p = p0 + delta[:, :3]
        R = so3_exp(delta[:, 3:6]) @ R0
        q = (q0 + delta[:, 6:]).clamp(robot.ranges[:, 0], robot.ranges[:, 1])
    results = []
    for ji, j in enumerate(jobs):
        candidate = {k: v.copy() for k, v in j.motion.items()}
        changed = np.zeros(len(candidate["base_pos_w"]), bool)
        cols = robot.pose(j.motion)[3]
        for cj, st, en, sl, _ in chunks:
            if cj != ji:
                continue
            pp = p[sl].cpu().numpy()
            rr = R[sl].cpu().numpy()
            qq = q[sl].cpu().numpy()
            # Only nonzero envelope samples are serialized; zero-envelope frames retain their exact original bits.
            active = np.any(
                window_basis(st, en, len(changed), float(j.motion["framerate"]), args.padding, args.knot_seconds) != 0,
                axis=1,
            )
            idx = np.arange(st, en)[active]
            changed[idx] = True
            quat = Rotation.from_matrix(rr.astype(np.float64)).as_quat()[:, [3, 0, 1, 2]]
            quat *= np.where((quat * j.motion["base_quat_w"][st:en]).sum(-1, keepdims=True) < 0, -1.0, 1.0)
            candidate["base_pos_w"][idx] = pp[active]
            if not (args.repair_jumps or args.lock_root_rotation):
                candidate["base_quat_w"][idx] = quat[active]
            candidate["joint_pos"][np.ix_(idx, cols)] = qq[active]
        results.append((candidate, changed))
    return results


def load_dataset(root):
    meta = yaml.safe_load((Path(root) / "metadata.yaml").read_text())
    terr = {t["terrain_id"]: t for t in meta["terrains"]}
    return meta, terr


def inputs_sha(root, entry, terr):
    rel = entry["motion_file"]
    names = ["metadata.yaml", rel, rel[:-4] + ".contacts.npz", terr["terrain_geoms"], terr["terrain_file"]]
    return hashlib.sha256(json.dumps({n: sha(Path(root) / n) for n in names}, sort_keys=True).encode()).hexdigest()


def contact_patch_stats(robot, source, candidate, terrain, labels, changed, slack=0.005):
    """Independent continuous check of labelled gaps and original supporting primitives.

    Keep the ORIGINAL supporting primitive within its original gap allowance;
    a different primitive becoming slightly closer at an intersection does not
    itself mean that support was lost. Normals are compared on that original
    primitive. Distant hysteresis tails have no unique supporting face.
    """
    frames = np.nonzero(changed)[0]
    if not len(frames):
        return dict(gap_violations=0, surface_changes=0, normal_turns=0, max_gap_growth_mm=0.0)

    def signature(motion):
        p, R, q, _ = robot.pose(motion)
        with torch.no_grad():
            a, b, _ = robot.segments(p[frames], R[frames], q[frames])
        F, G = len(a), len(robot.radius)
        K = len(terrain.kind)
        _, (row, geom, t) = terrain.witnesses(
            a.reshape(-1, 3),
            b.reshape(-1, 3),
            robot.radius.expand(F, -1).reshape(-1),
            iterations=36,
            cutoff=float("inf"),
        )
        aa, bb = a.reshape(-1, 3), b.reshape(-1, 3)
        x = aa[row] + t[:, None] * (bb[row] - aa[row])
        values = (
            paired_sdf(terrain.local(x, geom), terrain.size[geom], terrain.kind[geom]) - robot.radius[row % G]
        ).reshape(F, G, K)
        ds = []
        tg = []
        xs = []
        per_terrain = []
        for inds in robot.contact_geoms:
            vals = values[:, inds, :]
            per, gchoice = vals.min(1)
            per_terrain.append(per)
            g = inds[gchoice]
            k = torch.arange(K, device=a.device).expand(F, K)
            idx = torch.arange(F, device=a.device)[:, None] * G * K + g * K + k
            ds.append(per.amin(-1))
            tg.append(per.argmin(-1))
            xs.append(x[idx])
        xx = torch.stack(xs, 1).reshape(-1, 3).detach().requires_grad_()
        kk = torch.arange(K, device=a.device).expand(F, 9, K).reshape(-1)
        vv = paired_sdf(terrain.local(xx, kk), terrain.size[kk], terrain.kind[kk])
        normal = torch.autograd.grad(vv.sum(), xx)[0].detach()
        normal = normal / normal.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        return (
            torch.stack(ds, 1).detach(),
            torch.stack(tg, 1),
            normal.reshape(F, 9, K, 3),
            torch.stack(per_terrain, 1).detach(),
        )

    before, old_id, old_normals, old_per = signature(source)
    after, new_id, new_normals, new_per = signature(candidate)
    active = torch.as_tensor(labels[frames], device=before.device, dtype=torch.bool)
    growth = after - before.clamp(min=0)
    gap = (growth > slack + 2e-6) & active
    original_gap = new_per.gather(-1, old_id[..., None]).squeeze(-1)
    surface = (original_gap > before.clamp(min=0) + slack + 2e-6) & active & (before < 0.015)
    ix = old_id[..., None, None].expand(-1, -1, 1, 3)
    old_n = old_normals.gather(2, ix).squeeze(2)
    new_n = new_normals.gather(2, ix).squeeze(2)
    turns = ((old_n * new_n).sum(-1) < 0.5) & active & (before < 0.015) & ~surface
    return dict(
        gap_violations=int(gap.sum()),
        surface_changes=int(surface.sum()),
        normal_turns=int(turns.sum()),
        max_gap_growth_mm=float(growth[active].max() * 1000) if active.any() else 0.0,
        max_original_surface_gap_growth_mm=(
            float((original_gap - before.clamp(min=0))[active].max() * 1000) if active.any() else 0.0
        ),
        nearest_primitive_switches=int(((old_id != new_id) & active).sum()),
        surface_examples=[[int(frames[f]), int(body)] for f, body in surface.nonzero().cpu().tolist()[:8]],
        normal_examples=[[int(frames[f]), int(body)] for f, body in turns.nonzero().cpu().tolist()[:8]],
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["scan", "repair", "verify", "gallery"])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out")
    ap.add_argument("--report", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--scope", choices=["primary", "sparse", "all"], default="primary")
    ap.add_argument("--clips-file")
    ap.add_argument("--batch-clips", type=int, default=3)
    ap.add_argument("--iterations", type=int, default=350)
    ap.add_argument("--padding", type=float, default=0.45)
    ap.add_argument("--knot-seconds", type=float, default=0.16)
    ap.add_argument("--safety", type=float, default=0.0005)
    ap.add_argument("--contact-slack", type=float, default=0.003)
    ap.add_argument("--pen-weight", type=float, default=5000.0)
    ap.add_argument("--keep-weight", type=float, default=3000.0)
    ap.add_argument(
        "--seed-candidates",
        help="optional rejected candidates from an earlier pass; only initialization, never trusted acceptance",
    )
    ap.add_argument("--repair-manifest")
    ap.add_argument("--gallery-dir")
    ap.add_argument("--refresh", type=int, default=10)
    ap.add_argument("--copy-dataset", action="store_true")
    ap.add_argument(
        "--repair-jumps",
        action="store_true",
        help="also repair authorized source joint jumps >0.35 rad/frame in local windows",
    )
    ap.add_argument("--lock-root-rotation", action="store_true", help="retain source pelvis quaternions byte-for-byte")
    ap.add_argument("--learning-rate", type=float, help="Adam learning rate override for stiff contact constraints")
    args = ap.parse_args()
    torch.set_num_threads(4)
    root = Path(args.dataset).resolve()
    out = Path(args.out).resolve() if args.out else None
    if out == root:
        raise ValueError("output must differ from source")
    if args.mode != "scan" and out is None:
        raise ValueError("--out is required")
    if args.mode == "gallery":
        if not args.repair_manifest or not args.gallery_dir:
            raise ValueError("gallery requires --repair-manifest and --gallery-dir")
        from instinct_mj.scripts.acp.penetration_gallery import generate_gallery

        generate_gallery(root, out, args.repair_manifest, args.gallery_dir, args.report)
        return
    if args.copy_dataset:
        if out.exists():
            raise FileExistsError("refusing to overwrite existing dataset; use per-clip repair without --copy-dataset")
        shutil.copytree(root, out, symlinks=False)
    if out:
        out.mkdir(parents=True, exist_ok=True)
    meta, terr = load_dataset(root)
    entries = meta["motion_files"]
    accepted_rows = json.loads(Path(args.repair_manifest).read_text())["clips"] if args.repair_manifest else {}
    if len({e["motion_file"] for e in entries}) != len(entries):
        raise ValueError("duplicate motion names")
    if args.clips_file:
        names = set(Path(args.clips_file).read_text().split())
        entries = [e for e in entries if e["motion_file"] in names]
        if {e["motion_file"] for e in entries} != names:
            raise ValueError("unknown clips in selection")
    robot = Robot(args.device, torch.float64)
    opt_robot = Robot(args.device, torch.float32) if args.mode == "repair" else None
    report = dict(
        version=VERSION,
        code_sha=sha(__file__),
        model_sha=robot.signature,
        source=str(root),
        source_metadata_sha=sha(root / "metadata.yaml"),
        args=vars(args),
        clips={},
    )
    if args.mode == "verify":
        names = ["metadata.yaml"] + [e["motion_file"][:-4] + ".contacts.npz" for e in meta["motion_files"]]
        names += list({t[k] for t in meta["terrains"] for k in ("terrain_file", "terrain_geoms") if k in t})
        report["integrity_errors"] = [n for n in names if not (out / n).is_file() or sha(root / n) != sha(out / n)]
    pending = []
    start = time.time()

    def save():
        write_json(args.report, report)

    def flush():
        if not pending:
            return
        print("OPTIMIZE " + ", ".join(j.rel for j in pending), flush=True)
        optimized = optimize_jobs(pending, opt_robot, args)
        for job, (candidate, changed) in zip(pending, optimized):
            entry = next(e for e in entries if e["motion_file"] == job.rel)
            te = terr[entry["terrain_id"]]
            terrain = Terrain(yaml.safe_load((root / te["terrain_geoms"]).read_text()), args.device, torch.float64)
            after = robot.measure(candidate, terrain)
            metrics = candidate_metrics(robot, job.motion, candidate, job.before, after, changed)
            # Same-body labelled gap bound, independent of the optimizer's frozen material witnesses.
            contact_bad = 0
            for bi, inds in enumerate(robot.contact_geoms):
                ix = inds.cpu().numpy()
                flags = job.labels[:, bi]
                contact_bad += int(
                    (
                        (after[:, ix].min(-1) > np.maximum(job.before[:, ix].min(-1), 0) + args.contact_slack + 2e-6)
                        & flags
                    ).sum()
                )
            if contact_bad:
                metrics["failures"].append(f"contact gap grew on {contact_bad} frames")
            metrics.update(
                status="repaired" if not metrics["failures"] else "unresolved",
                windows=job.spans,
                changed_frames=int(changed.sum()),
                source_sha=job.source_sha,
                input_sha=job.input_sha,
            )
            metrics["candidate_min_sd_after"] = metrics["min_sd_after"]
            metrics["published_min_sd_after"] = (
                metrics["min_sd_after"] if not metrics["failures"] else metrics["min_sd_before"]
            )
            if inputs_sha(root, entry, te) != job.input_sha:
                raise RuntimeError("source changed during repair: " + job.rel)
            target = out / job.rel
            target.parent.mkdir(parents=True, exist_ok=True)
            if not metrics["failures"]:
                tmp = target.with_suffix(".tmp.npz")
                np.savez(tmp, **candidate)
                os.replace(tmp, target)
            else:
                # Preserve the authorized original; explicitly mark unresolved, never claim repaired.
                shutil.copyfile(root / job.rel, target)
                reject = Path(args.report).parent / "rejected" / job.rel
                reject.parent.mkdir(parents=True, exist_ok=True)
                np.savez(reject, **candidate)
            metrics["output_sha"] = sha(target)
            report["clips"][job.rel] = metrics
            print(
                job.rel,
                metrics["status"],
                f"{metrics['min_sd_before']*1000:.3f}->{metrics['min_sd_after']*1000:.3f} mm",
                metrics["failures"],
                flush=True,
            )
            save()
        pending.clear()

    for i, e in enumerate(entries):
        rel = e["motion_file"]
        te = terr[e["terrain_id"]]
        motion = dict(np.load(root / rel))
        source_sha = sha(root / rel)
        terrain = Terrain(yaml.safe_load((root / te["terrain_geoms"]).read_text()), args.device, torch.float64)
        if args.mode == "repair" and not in_scope(rel, args.scope):
            if out and not (out / rel).exists():
                (out / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(root / rel, out / rel)
            report["clips"][rel] = dict(
                status="deferred_sparse" if rel.split("/")[0] in SPARSE else "excluded", source_sha=source_sha
            )
            continue
        before = robot.measure(motion, terrain)
        row = dict(
            status="clean" if before.min() >= 0 else "penetrating",
            min_sd_before=float(before.min()),
            negative_frames=int((before.min(-1) < 0).sum()),
            worst_frame=int(np.unravel_index(before.argmin(), before.shape)[0]),
            worst_geom=robot.geom_names[np.unravel_index(before.argmin(), before.shape)[1]],
            source_sha=source_sha,
            in_scope=in_scope(rel, args.scope),
        )
        if args.mode == "verify":
            candidate = dict(np.load(out / rel))
            byte_equal = sha(out / rel) == source_sha
            after = before if byte_equal else robot.measure(candidate, terrain)
            changed = np.zeros(len(before), bool)
            for key in ("joint_pos", "base_pos_w", "base_quat_w"):
                changed |= np.any(
                    np.ascontiguousarray(candidate[key]).view(np.uint8).reshape(len(before), -1)
                    != np.ascontiguousarray(motion[key]).view(np.uint8).reshape(len(before), -1),
                    axis=1,
                )
            allowed = np.zeros(len(before), bool)
            declared = accepted_rows.get(rel, {}).get("windows")
            for st, en in (
                declared
                if declared is not None
                else windows(before.min(-1) < 0, float(motion["framerate"]), args.padding)
            ):
                allowed[st:en] = True
            row.update(
                candidate_metrics(robot, motion, candidate, before, after, allowed)
                if changed.any()
                else dict(min_sd_after=float(after.min()), failures=[])
            )
            _, _, q, _ = robot.pose(candidate)
            row["joint_limit_count"] = int(((q < robot.ranges[:, 0] - 1e-6) | (q > robot.ranges[:, 1] + 1e-6)).sum())
            row["joint_jump_count"] = int((torch.diff(q, dim=0).abs() > 0.35).sum())
            if changed.any() and not in_scope(rel, args.scope):
                row["failures"].append("modified clip outside authorized scope")
            for key in set(motion) - {"joint_pos", "base_pos_w", "base_quat_w"}:
                if candidate[key].tobytes() != motion[key].tobytes():
                    row["failures"].append("changed auxiliary motion field: " + key)
            if changed.any():
                labels = np.load(root / (rel[:-4] + ".contacts.npz"))["contact_flag"]
                slack = accepted_rows.get(rel, {}).get("contact_slack_m", args.contact_slack)
                contact = contact_patch_stats(robot, motion, candidate, terrain, labels, changed, slack)
                row["contact_patch"] = contact
                if any(contact[k] for k in ("gap_violations", "surface_changes", "normal_turns")):
                    row["failures"].append("lost original contact surface / gap / normal")
            row.update(
                status="modified" if changed.any() else "unchanged", output_sha=sha(out / rel), byte_equal=byte_equal
            )
        elif args.mode == "repair":
            mask = repair_mask(motion, before, args.repair_jumps)
            if mask.any():
                labels = np.load(root / (rel[:-4] + ".contacts.npz"))["contact_flag"]
                pending.append(
                    Job(
                        rel,
                        motion,
                        Terrain(yaml.safe_load((root / te["terrain_geoms"]).read_text()), args.device),
                        labels,
                        before,
                        windows(mask, float(motion["framerate"]), args.padding),
                        source_sha,
                        inputs_sha(root, e, te),
                    )
                )
                row["status"] = "pending"
            else:
                (out / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(root / rel, out / rel)
        report["clips"][rel] = row
        shown_sd = row["min_sd_after"] if args.mode == "verify" else before.min()
        print(
            f'{i+1}/{len(entries)} {rel}: {row["status"]} min={shown_sd*1000:.4f}mm elapsed={time.time()-start:.1f}s',
            flush=True,
        )
        save()
        if len(pending) >= args.batch_clips:
            flush()
    flush()
    report["elapsed_seconds"] = time.time() - start
    if args.mode == "verify":
        rows = report["clips"]
        report["negative_in_scope"] = [r for r, s in rows.items() if s["in_scope"] and s["min_sd_after"] < 0]
        report["negative_outside_scope"] = [r for r, s in rows.items() if not s["in_scope"] and s["min_sd_after"] < 0]
        report["scope_verified"] = not (
            report["integrity_errors"]
            or report["negative_in_scope"]
            or any(s["failures"] or s["joint_limit_count"] or s["joint_jump_count"] for s in rows.values())
        )
    save()


if __name__ == "__main__":
    main()
