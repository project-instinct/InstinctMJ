"""CPU audit of reference-motion dynamics, without modifying input data.

Usage::

    uv run --no-sync python -m instinct_mj.scripts.acp.check_dynamics \
        /home/duanxin/Datasets/acp/v3_train11 --clip-regex 'vault' --out /tmp/codex_dyn/train
    uv run --no-sync python -m instinct_mj.scripts.acp.check_dynamics \
        /home/duanxin/Datasets/acp/rollout_motions/vault --out /tmp/codex_dyn/positive

The input is either a dataset with metadata.yaml and analytic terrain_geoms, or
an exported rollout directory (searched recursively for motion + XML + physics
sidecars). Only --clip-regex and --out are optional. At most four CPU workers
are used; BLAS/HiGHS threads are restricted to one. No environment or GPU is
created. Long runs should be launched with systemd-run --user and MemoryMax=16G.

Each frame is retained in frames/*.npz and frames/*.csv; clips.csv, contacts.csv,
events.csv, flights.csv and conclusions.md summarize clips. Floor departure
for run-up statistics uses the lowest horizontal plane/box surface, separately
from feet touching the wall. Frame indices are zero based and
inclusive. Two frames at each end are excluded from scores, not from exports.
The nominal friction is 1.0; 0.5/1.2 sensitivity and 2/20 mm proximity are fixed
protocols. Sixteen rays circumscribe a circular Coulomb cone (1.96% relaxation).
The LP minimizes the infinity norm of the six root imbalance components, scaled
by mg (force) and mg*0.5 m (moment). A separate LP bounds actuator torques;
minimum required peak actuator ratio is minimized after fixing the optimal root
residual. Contact forces have no artificial upper bound. These are necessary
conditions, not a closed-loop tracking or unilateral contact-acceleration test.

Use calibrate(output_dirs) on verified rollout outputs to obtain a frame P99
threshold and a clip P99 bad-frame fraction. Pass the returned dictionary to
summarize(output_dir, calibration) to regenerate thresholded CSV/conclusions.
Without calibration the tool uses descriptive 0.2 screening only, explicitly
marked uncalibrated. Python helpers audit_clip(entry, out, variant=...) support
read-only native-mesh checks and synthetic perturbations (height_shift in m,
time_scale as playback speed); these are deliberately not extra CLI switches.

50 Hz differences smear 2 ms impacts. Physics cross-checks therefore compare
both the measured interval wrench against the geometric LP and against inverse
dynamics. Raw 2 ms contact points/forces also undergo an independent cone LP.
"""

from __future__ import annotations

import os

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import argparse
import csv
import json
import multiprocessing
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import mujoco
import numpy as np
import yaml
import warnings
from scipy.optimize import OptimizeWarning, linprog

from instinct_mj.assets.acp_g1 import get_acp_g1_spec

CONTACT_LINKS = (
    "left_knee_link", "right_knee_link", "pelvis", "left_ankle_roll_link",
    "right_ankle_roll_link", "left_elbow_link", "right_elbow_link",
    "left_wrist_yaw_link", "right_wrist_yaw_link",
)
TOLERANCE = 0.02
MU = 1.0
RAYS = 16
# SciPy forwards this supported HiGHS option but emits a wrapper warning.
warnings.filterwarnings("ignore", message="Unrecognized options detected.*threads", category=OptimizeWarning)


def intervals(mask: np.ndarray, min_length: int = 1) -> list[list[int]]:
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    return [[int(a), int(b - 1)] for a, b in zip(edges[::2], edges[1::2]) if b - a >= min_length]


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                         for k, v in row.items()} for row in rows)


def discover(root: Path, pattern: str | None = None) -> list[dict]:
    root = root.expanduser().resolve()
    entries = []
    if (root / "metadata.yaml").exists():
        meta = yaml.safe_load((root / "metadata.yaml").read_text())
        terrains = {t["terrain_id"]: t for t in meta["terrains"]}
        for e in meta["motion_files"]:
            p = root / e["motion_file"]
            t = terrains[e["terrain_id"]]
            entries.append(dict(name=p.stem, path=str(p), group=e.get("group", p.parent.name),
                                terrain=str(root / t["terrain_geoms"]) if t.get("terrain_geoms") else None,
                                xml=str(p.with_suffix(".xml")) if p.with_suffix(".xml").exists() else None))
    else:
        for p in sorted(root.rglob("*.npz")):
            if "." in p.stem or not p.with_suffix(".xml").exists():
                continue
            rel = p.relative_to(root)
            entries.append(dict(name=str(rel.with_suffix("")).replace("/", "__"), path=str(p),
                                group="rollout__" + str(rel.parent), terrain=None,
                                xml=str(p.with_suffix(".xml"))))
    if pattern:
        entries = [e for e in entries if (re.search(pattern, e["name"]) or re.search(pattern, e["group"]))]
    names = [e["name"] for e in entries]
    for e in entries:
        if names.count(e["name"]) > 1:
            e["name"] += "@" + Path(e["path"]).parent.name
    if not entries:
        raise ValueError(f"No matching motions in {root}")
    return entries


def build_model(entry: dict, native: bool = False):
    # Native XML is also the source of exact terrain orientation and dimensions.
    source = mujoco.MjModel.from_xml_path(entry["xml"]) if entry.get("xml") else None
    if native:
        if source is None:
            raise ValueError("Native cross-check requires an exported scene XML")
        model = source
    else:
        spec = get_acp_g1_spec(simplified_collision=True)
        if entry.get("terrain"):
            geoms = yaml.safe_load(Path(entry["terrain"]).read_text())
            for i, g in enumerate(geoms):
                spec.worldbody.add_geom(name=f"audit_terrain_{i}",
                    type=getattr(mujoco.mjtGeom, "mjGEOM_" + g["type"].upper()),
                    size=g["size"], pos=g["pos"], quat=g.get("quat", [1, 0, 0, 0]),
                    contype=0, conaffinity=1, friction=[MU, .005, .0001])
        elif source is not None:
            for i in range(source.ngeom):
                if source.geom_bodyid[i] != 0 or not (source.geom_contype[i] or source.geom_conaffinity[i]):
                    continue
                spec.worldbody.add_geom(name=f"audit_terrain_{i}", type=int(source.geom_type[i]),
                    size=source.geom_size[i], pos=source.geom_pos[i], quat=source.geom_quat[i],
                    contype=0, conaffinity=1, friction=source.geom_friction[i])
        else:
            raise ValueError(f"Missing analytic terrain / XML for {entry['name']}")
        model = spec.compile()
    pelvis = model.body("pelvis").id
    # Filter collisions to robot-world only; self forces cannot balance root wrench.
    robot = np.array([model.body_rootid[int(b)] == pelvis for b in model.geom_bodyid])
    active = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
    world = (model.geom_bodyid == 0) & active
    model.geom_contype[:] = np.where(robot & active, 1, 0)
    model.geom_conaffinity[:] = np.where(world, 1, 0)
    model.geom_margin[:] = .06
    model.geom_gap[:] = .06
    # Margins are expanded after compilation: the compiled midphase BVH can
    # otherwise cull separated pairs that are inside the audit tolerance.
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_MIDPHASE
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONSTRAINT
    return model, mujoco.MjData(model), source



def ground_geom_ids(model) -> list[int]:
    """Lowest horizontal plane/box top; separate floor takeoff from wall stepping."""
    candidates = []
    for g in np.flatnonzero(model.geom_conaffinity):
        rotation = np.zeros(9)
        mujoco.mju_quat2Mat(rotation, model.geom_quat[g])
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE:
            candidates.append((int(g), float(model.geom_pos[g, 2])))
        elif model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX and rotation[8] > .999:
            candidates.append((int(g), float(model.geom_pos[g, 2] + model.geom_size[g, 2])))
    if not candidates:
        return []
    lowest = min(z for _, z in candidates)
    return [g for g, z in candidates if z < lowest + .002]


def trajectory(model, raw, time_scale: float = 1., height_shift: float = 0.):
    dt = 1. / float(raw["framerate"]) / time_scale
    n = len(raw["joint_pos"])
    q = np.tile(model.qpos0, (n, 1))
    root = model.jnt_qposadr[0]
    q[:, root:root + 3] = raw["base_pos_w"]
    q[:, root + 2] += height_shift
    quat = raw["base_quat_w"].astype(float)
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    q[:, root + 3:root + 7] = quat
    names = list(raw["joint_names"])
    for j in range(1, model.njnt):
        q[:, model.jnt_qposadr[j]] = raw["joint_pos"][:, names.index(model.joint(j).name)]
    half = np.zeros((n - 1, model.nv))
    for i in range(n - 1):
        mujoco.mj_differentiatePos(model, half[i], dt, q[i], q[i + 1])
    v = np.vstack([half[0], (half[:-1] + half[1:]) * .5, half[-1]])
    a = np.gradient(v, dt, axis=0)
    return q, v, a, dt


def cone_matrix(model, data, contacts, mu):
    cols = []
    for point, normal, body, _distance in contacts:
        normal = np.asarray(normal, float)
        normal = normal / np.linalg.norm(normal)
        basis = np.array([1., 0, 0]) if abs(normal[0]) < .9 else np.array([0., 1, 0])
        t1 = np.cross(normal, basis)
        t1 /= np.linalg.norm(t1)
        t2 = np.cross(normal, t1)
        angles = np.arange(RAYS) * 2 * np.pi / RAYS
        rays = normal[:, None] + mu / np.cos(np.pi / RAYS) * (
            t1[:, None] * np.cos(angles) + t2[:, None] * np.sin(angles))
        jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        mujoco.mj_jac(model, data, jp, jr, point, body)
        cols.append(jp.T @ rays)
    return np.concatenate(cols, axis=1) if cols else np.zeros((model.nv, 0))


def solve_lp(required, matrix, mg, limits=None):
    nrow = len(required) if limits is not None else 6
    scale = np.r_[np.full(3, mg), np.full(3, mg * .5), limits if limits is not None else []]
    c = matrix[:nrow]
    if limits is not None:
        actuators = np.zeros((nrow, len(limits)))
        actuators[6:] = np.eye(len(limits))
        c = np.c_[c, actuators]
    scaled = c / scale[:, None]
    rhs = required[:nrow] / scale
    bounds = [(0, None)] * matrix.shape[1]
    if limits is not None:
        bounds += [(-float(x), float(x)) for x in limits]
    result = linprog(np.r_[np.zeros(c.shape[1]), 1.],
        A_ub=np.vstack([np.c_[scaled, -np.ones(nrow)], np.c_[-scaled, -np.ones(nrow)]]),
        b_ub=np.r_[rhs, -rhs], bounds=bounds + [(0, None)], method="highs",
        options={"threads": 1})
    if not result.success:
        raise RuntimeError(result.message)
    residual = required[:nrow] - c @ result.x[:-1]
    return float(result.x[-1]), residual, result.x[:matrix.shape[1]]


def torque_ratio(required, matrix, mg, limits, root_score):
    # Lexicographic second stage: retain the best root fit, minimize peak torque.
    n = matrix.shape[1]
    root = matrix[:6] / np.r_[np.full(3, mg), np.full(3, mg * .5)][:, None]
    rhs = required[:6] / np.r_[np.full(3, mg), np.full(3, mg * .5)]
    joint = matrix[6:] / limits[:, None]
    jrhs = required[6:] / limits
    eps = root_score + 1e-7
    u = np.vstack([np.c_[root, np.zeros(6)], np.c_[-root, np.zeros(6)],
                   np.c_[joint, -np.ones(len(limits))], np.c_[-joint, -np.ones(len(limits))]])
    r = linprog(np.r_[np.zeros(n), 1.], A_ub=u,
        b_ub=np.r_[rhs + eps, -rhs + eps, jrhs, -jrhs],
        bounds=[(0, None)] * (n + 1), method="highs", options={"threads": 1})
    if not r.success:
        raise RuntimeError(r.message)
    ratios = np.abs(required[6:] - matrix[6:] @ r.x[:n]) / limits
    return float(r.x[-1]), float(np.mean(ratios > 1 + 1e-5))


def raw_normal(source, data, row):
    # Raw exports store normal Z and magnitude, but not the full normal.
    # Scene geometry resolves the remaining face/edge ambiguity.
    g1, g2 = row[9:11].astype(int)
    terrain = g1 if source.geom_bodyid[g1] == 0 else g2
    rot = data.geom_xmat[terrain].reshape(3, 3)
    if source.geom_type[terrain] == mujoco.mjtGeom.mjGEOM_PLANE:
        return rot[:, 2]
    if source.geom_type[terrain] != mujoco.mjtGeom.mjGEOM_BOX:
        raise ValueError("Raw-force validation currently supports plane/box rollout terrain")
    local = rot.T @ (row[2:5] - data.geom_xpos[terrain])
    gap = np.abs(np.abs(local) - source.geom_size[terrain])
    candidates = np.flatnonzero(gap < gap.min() + .005)
    normals = [rot[:, k] * (1 if local[k] >= 0 else -1) for k in candidates]
    nz = np.clip(row[11] / max(row[8], 1e-12), -1., 1.)
    if len(candidates) == 1:
        return normals[0]
    # At box edges the true normal is a combination of adjacent face normals.
    # Recover it from stored n_z, |n|=1 and F.n=normal_force, then select the
    # solution inside the geometric normal cone. Friction remains independently
    # testable: |F_t|^2 = |F|^2 - normal_force^2 is not fixed by this recovery.
    force = row[5:8]
    horizontal = np.linalg.norm(force[:2])
    if horizontal > 1e-10:
        u = force[:2] / horizontal
        along = (row[8] - force[2] * nz) / horizontal
        across = np.sqrt(max(0., 1 - nz * nz - along * along))
        perpendicular = np.array([-u[1], u[0]])
        recovered = [np.r_[along * u + sign * across * perpendicular, nz] for sign in (-1, 1)]
        def cone_error(normal):
            components = rot.T @ normal
            allowed = np.zeros(3)
            allowed[candidates] = np.sign(local[candidates])
            return float(np.sum(np.abs(components[allowed == 0])) +
                         np.maximum(-components * allowed, 0).sum())
        return min(recovered, key=cone_error)
    return min(normals, key=lambda n: abs(n[2] - nz))


def audit_clip(entry: dict, out: str | Path, *, variant: dict | None = None) -> dict:
    variant = variant or {}
    out = Path(out)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    model, data, source = build_model(entry, variant.get("native", False))
    raw = np.load(entry["path"])
    q, v, a, dt = trajectory(model, raw, variant.get("time_scale", 1.), variant.get("height_shift", 0.))
    n = len(q)
    pelvis = model.body("pelvis").id
    mass = float(model.body_subtreemass[pelvis])
    mg = mass * np.linalg.norm(model.opt.gravity)
    limits = model.jnt_actfrcrange[1:, 1].copy()
    if np.any(limits <= 0) or len(limits) != 23:
        raise ValueError("Expected the ACP 23-DOF motor limits")
    side = Path(entry["path"]).with_suffix(".contacts.npz")
    phy_path = Path(entry["path"]).with_suffix(".physics.npz")
    physics = np.load(phy_path) if phy_path.exists() and not any(k in variant for k in ("time_scale", "height_shift")) else None
    flags = np.full((n, len(CONTACT_LINKS)), False)
    labels_present = side.exists()
    if labels_present:
        label = np.load(side)
        links = list(label["link_names"])
        flags[:] = label["contact_flag"][:, [links.index(b) for b in CONTACT_LINKS]]
    elif physics is not None:
        links = list(physics["link_names"])
        flags[:] = physics["contact_flag"][:, [links.index(b) for b in CONTACT_LINKS]]
        labels_present = True
    bodyids = [model.body(b).id for b in CONTACT_LINKS]
    bybody = {b: i for i, b in enumerate(bodyids)}
    robot_geoms = np.flatnonzero(model.geom_contype)
    terrain_geoms = np.flatnonzero(model.geom_conaffinity)
    ground_ids = ground_geom_ids(model)
    ground_indices = [i for i, g in enumerate(terrain_geoms) if g in ground_ids]
    ground_distance = np.full((n, 2), 10.)
    distances = np.full((n, len(CONTACT_LINKS)), 10.)
    com = np.zeros((n, 3))
    required = np.zeros((n, model.nv))
    min_distance = np.full(n, 10.)
    scores = np.zeros((n, 7))
    residual = np.zeros((n, 6))
    torque = np.zeros((n, 2))
    support_bodies = np.zeros((n, len(CONTACT_LINKS)), bool)
    support_count = np.zeros(n, int)
    physics_scores = np.full((n, 4), np.nan)
    measured_wrench = np.full((n, 6), np.nan)
    raw_rows = physics["raw_contacts"] if physics is not None else None
    if physics is not None:
        source_data = mujoco.MjData(source)
        # Avoid binary rounding putting step 29 in frame 2 instead of frame 2.
        raw_frame = np.floor((raw_rows[:, 0] + .1) / round((1 / float(raw["framerate"])) / float(physics["physics_dt"]))).astype(int)
    for f in range(n):
        data.qpos[:] = q[f]
        data.qvel[:] = v[f]
        data.qacc[:] = a[f]
        model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONSTRAINT
        mujoco.mj_inverse(model, data)
        # MuJoCo skips collision detection while constraints are disabled.
        model.opt.disableflags &= ~int(mujoco.mjtDisableBit.mjDSBL_CONSTRAINT)
        mujoco.mj_collision(model, data)
        required[f] = data.qfrc_inverse
        # With constraints disabled, mj_inverse excludes Coulomb frictionloss.
        required[f, 6:] += model.dof_frictionloss[6:] * np.sign(v[f, 6:])
        com[f] = data.subtree_com[pelvis]
        contacts = []
        for co in data.contact:
            g1, g2 = co.geom
            b1, b2 = model.geom_bodyid[[g1, g2]]
            if not ((b1 == 0) ^ (b2 == 0)):
                continue
            body = int(b2 if b1 == 0 else b1)
            distance = float(co.dist)
            if distance <= TOLERANCE + 1e-9:
                normal = np.array(co.frame[:3]) * (1 if b1 == 0 else -1)
                contacts.append((co.pos.copy(), normal, body, distance))
                if body in bybody:
                    support_bodies[f, bybody[body]] = True
        # Exact signed primitive distance, not a saturated contact search radius.
        segment = np.zeros(6)
        for rg in robot_geoms:
            body = int(model.geom_bodyid[rg])
            # MuJoCo mesh distance at very large distmax can return a spurious
            # zero with separated witness points; native checks need proximity only.
            distance_cap = .06 if variant.get("native", False) else 10.
            pair_distances = [mujoco.mj_geomDistance(model, data, int(rg), int(tg), distance_cap, segment)
                              for tg in terrain_geoms]
            best = min(pair_distances)
            if body in bodyids[3:5] and ground_indices:
                j = bodyids[3:5].index(body)
                ground_distance[f, j] = min(ground_distance[f, j],
                                           min(pair_distances[k] for k in ground_indices))
            min_distance[f] = min(min_distance[f], best)
            if body in bybody:
                distances[f, bybody[body]] = min(distances[f, bybody[body]], best)
        support_count[f] = len(set(c[2] for c in contacts))
        if (min_distance[f] <= TOLERANCE - 1e-6) != bool(contacts):
            raise RuntimeError(f"Distance/manifold disagreement: {entry['name']} frame {f}")
        nominal = cone_matrix(model, data, contacts, MU)
        root, res, _ = solve_lp(required[f], nominal, mg)
        scores[f, 0] = root
        residual[f] = res
        scores[f, 1] = solve_lp(required[f], nominal, mg, limits)[0]
        for col, mu in [(2, .5), (3, 1.2)]:
            scores[f, col] = solve_lp(required[f], cone_matrix(model, data, contacts, mu), mg)[0]
        strict = cone_matrix(model, data, [c for c in contacts if c[3] <= .002], MU)
        scores[f, 4] = solve_lp(required[f], strict, mg)[0]
        scores[f, 5] = solve_lp(required[f], strict, mg, limits)[0]
        labeled = cone_matrix(model, data, [c for c in contacts if c[2] in bybody and flags[f, bybody[c[2]]]], MU)
        scores[f, 6] = solve_lp(required[f], labeled, mg)[0] if labels_present else np.nan
        torque[f] = torque_ratio(required[f], nominal, mg, limits, root)
        if physics is not None:
            rr = raw_rows[raw_frame == f]
            source_data.qpos[:] = physics["qpos"][f]
            mujoco.mj_kinematics(source, source_data)
            mujoco.mj_comPos(source, source_data)
            actual = np.zeros(model.nv)
            measured_contacts = []
            subs = round((1 / float(raw["framerate"])) / float(physics["physics_dt"]))
            for row in rr:
                body = model.body(str(physics["link_names"][int(row[1])])).id
                jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
                mujoco.mj_jac(model, data, jp, jr, row[2:5], body)
                actual += jp.T @ row[5:8] / subs
                if row[8] > 1e-6:
                    measured_contacts.append((row[2:5], raw_normal(source, source_data, row), body, 0.))
            measured_wrench[f] = actual[:6]
            physics_scores[f, 0] = solve_lp(actual, nominal, mg)[0]
            physics_scores[f, 1] = solve_lp(actual, cone_matrix(model, data, measured_contacts, MU), mg)[0]
            physics_scores[f, 2] = np.max(np.abs((required[f, :6] - actual[:6]) /
                                                    np.r_[np.full(3, mg), np.full(3, mg * .5)]))
            force_sum = rr[:, 5:8].sum(axis=0) / subs if len(rr) else np.zeros(3)
            physics_scores[f, 3] = np.max(np.abs(force_sum - physics["force_w"][f].sum(axis=0)))
    com_velocity = np.gradient(com, dt, axis=0)
    com_accel = np.gradient(com_velocity, dt, axis=0)
    ballistic = np.linalg.norm(com_accel - model.opt.gravity, axis=1)
    # A derivative stencil must lie wholly inside an unsupported interval.
    flight_interior = np.convolve((min_distance > TOLERANCE).astype(int), np.ones(5), mode="same") == 5
    valid = np.arange(n) >= 2
    valid &= np.arange(n) < n - 2
    arrays = dict(qpos=q, qvel=v, qacc=a, dt=np.array(dt), mass=np.array(mass), required=required,
        root_residual=residual, scores=scores, torque=torque, com=com, com_velocity=com_velocity,
        com_acceleration=com_accel, ballistic_error=ballistic, flight_interior=flight_interior,
        min_distance=min_distance, distance=distances, ground_distance=ground_distance,
        labels=flags, support_bodies=support_bodies,
        support_count=support_count, valid=valid, physics_scores=physics_scores, measured_wrench=measured_wrench)
    destination = out / "frames" / (entry["name"] + ".npz")
    temporary = destination.with_suffix(".npz.tmp")
    with temporary.open("wb") as f:
        np.savez_compressed(f, **arrays)
    temporary.replace(destination)
    info = dict(entry, frames=n, dt=dt, mass=mass, variant=variant, labels_present=labels_present,
                physics_present=physics is not None,
                score_columns=["root_mu1_20mm", "full_mu1_20mm", "root_mu05_20mm", "root_mu12_20mm",
                               "root_mu1_2mm", "full_mu1_2mm", "root_labels_20mm"],
                physics_columns=["measured_wrench_geometry_lp", "measured_points_cone_lp",
                                 "inverse_minus_measured", "force_aggregation_error_N"])
    (out / "frames" / (entry["name"] + ".json")).write_text(json.dumps(info, indent=2))
    return info


def frame_paths(output: str | Path) -> list[Path]:
    output = Path(output)
    selection = output / "selection.json"
    selected = {e["name"] for e in json.loads(selection.read_text())} if selection.exists() else None
    return sorted(p for p in (output / "frames").glob("*.npz") if selected is None or p.stem in selected)


def calibrate(outputs: list[str | Path]) -> dict:
    roots, full, ballots = [], [], []
    arrays = []
    for output in outputs:
        for p in frame_paths(output):
            with np.load(p) as z:
                valid = z["valid"]
                roots.extend(z["scores"][valid, 0])
                full.extend(z["scores"][valid, 1])
                ballots.extend(z["ballistic_error"][valid & z["flight_interior"]])
                arrays.append(z["scores"][valid, :2].copy())
    rt, ft = float(np.quantile(roots, .99)), float(np.quantile(full, .99))
    fractions = [float(np.mean(a[:, 0] > rt)) for a in arrays]
    maxruns = [max((b - a + 1 for a, b in intervals(x[:, 0] > rt)), default=0) for x in arrays]
    return dict(calibrated=True, positive_clips=len(arrays), positive_frames=len(roots),
        root_threshold=rt, full_threshold=ft, clip_bad_fraction_threshold=float(np.quantile(fractions, .99)),
        full_clip_bad_fraction_threshold=float(np.quantile([np.mean(x[:, 1] > ft) for x in arrays], .99)),
        longest_bad_frames_p99=float(np.quantile(maxruns, .99)),
        longest_bad_frames_max=int(max(maxruns)),
        ballistic_threshold=float(np.quantile(ballots, .99)) if ballots else None,
        root_quantiles=np.quantile(roots, [.5, .9, .99, 1]).tolist(),
        full_quantiles=np.quantile(full, [.5, .9, .99, 1]).tolist())


def takeoff_events(z) -> list[dict]:
    dt, mass = float(z["dt"]), float(z["mass"])
    feet = np.min(z["ground_distance"] if "ground_distance" in z else z["distance"][:, 3:5], axis=1) <= TOLERANCE
    events = []
    for lo, hi in intervals(~feet, max(3, round(.06 / dt))):
        if lo < round(.5 / dt) or hi >= len(feet) - 2:
            continue
        # Only floor departures after meaningful support, not 1-frame chatter.
        if np.mean(feet[max(0, lo - 5):lo]) < .6:
            continue
        before = max(0, lo - round(.5 / dt))
        speed = np.linalg.norm(z["qvel"][before:lo, :2], axis=1)
        cv = z["com_velocity"]
        impulse = mass * (cv[lo] - cv[before] - np.array([0., 0., -9.81]) * (lo - before) * dt)
        # Integrate unsupported net external-force demand, independently of labels.
        miss = z["required"][lo:hi + 1, :3] * (z["min_distance"][lo:hi + 1] > TOLERANCE)[:, None]
        measured = z["measured_wrench"][before + 1:lo + 1, :3]
        evidence = {}
        if np.isfinite(measured).all():
            measured_impulse = measured.sum(axis=0) * dt
            evidence = dict(measured_impulse_Ns=measured_impulse.tolist(),
                            impulse_error_norm_Ns=float(np.linalg.norm(impulse - measured_impulse)))
        events.append(dict(takeoff_frame=lo, feet_return_frame=hi + 1, **evidence,
            runup_speed_mean=float(speed.mean()), runup_speed_p90=float(np.quantile(speed, .9)),
            runup_speed_last=float(speed[-1]), com_takeoff_vz=float(cv[lo, 2]),
            takeoff_vertical_momentum_Ns=float(mass * cv[lo, 2]),
            momentum_change_z_Ns=float(mass * (cv[lo, 2] - cv[before, 2])),
            impulse_Ns=impulse.tolist(), impulse_z_Ns=float(impulse[2]),
            unsupported_impulse_Ns=(miss.sum(axis=0) * dt).tolist()))
    return events


def ballistic_events(z) -> list[dict]:
    result = []
    dt, mass = float(z["dt"]), float(z["mass"])
    for lo, hi in intervals(z["min_distance"] > TOLERANCE, 5):
        t = np.arange(hi - lo + 1) * dt
        p = z["com"][lo:hi + 1]
        corrected = p.copy()
        corrected[:, 2] += .5 * 9.81 * t * t
        fit = np.c_[np.ones(len(t)), t]
        error = corrected - fit @ np.linalg.lstsq(fit, corrected, rcond=None)[0]
        quadratic = np.linalg.lstsq(np.c_[fit, t * t], p, rcond=None)[0]
        external = mass * (2 * quadratic[2] + np.array([0., 0., 9.81]))
        needed_v = (p[-1] - p[0] - .5 * np.array([0., 0., -9.81]) * t[-1] ** 2) / t[-1]
        missing_impulse = mass * (needed_v - z["com_velocity"][lo])
        result.append(dict(start=lo, end=hi, duration_s=float(t[-1]),
            ballistic_fit_max_mm=float(np.linalg.norm(error, axis=1).max() * 1000),
            endpoint_required_start_velocity=needed_v.tolist(),
            reference_start_velocity=z["com_velocity"][lo].tolist(),
            endpoint_impulse_deficit_Ns=missing_impulse.tolist(),
            fitted_external_force_N=external.tolist(),
            fitted_external_force_norm_N=float(np.linalg.norm(external))))
    return result


def summarize(out: str | Path, calibration: dict | None = None) -> list[dict]:
    out = Path(out)
    calibration = calibration or dict(calibrated=False, root_threshold=.2, full_threshold=.2,
                                     clip_bad_fraction_threshold=.1, ballistic_threshold=None)
    (out / "thresholds.json").write_text(json.dumps(calibration, indent=2))
    summaries, contact_rows, event_rows, flight_rows, conclusions = [], [], [], [], []
    threshold = calibration["root_threshold"]
    for path in frame_paths(out):
        info = json.loads(path.with_suffix(".json").read_text())
        z = np.load(path)
        valid, scores = z["valid"], z["scores"]
        bad = valid & (scores[:, 0] > threshold)
        runs = intervals(bad)
        longest = max(runs, key=lambda p: p[1] - p[0], default=[])
        events = takeoff_events(z)
        flights = ballistic_events(z)
        flight_rows.extend(dict(clip=info["name"], **e) for e in flights)
        # Select the flight with greatest pelvis elevation, excluding ordinary gait swings.
        primary = max(events, key=lambda e: float(z["qpos"][e["takeoff_frame"]:e["feet_return_frame"], 2].max()), default={})
        row = dict(clip=info["name"], group=info["group"], source=info["path"], frames=info["frames"],
            root_p50=float(np.quantile(scores[valid, 0], .5)), root_p90=float(np.quantile(scores[valid, 0], .9)),
            root_max=float(scores[valid, 0].max()), bad_fraction=float(bad[valid].mean()),
            longest_bad=longest, bad_intervals=runs,
            full_p90=float(np.quantile(scores[valid, 1], .9)),
            full_bad_fraction=float(np.mean(scores[valid, 1] > calibration["full_threshold"])),
            mu05_p90=float(np.quantile(scores[valid, 2], .9)), mu12_p90=float(np.quantile(scores[valid, 3], .9)),
            strict2mm_p90=float(np.quantile(scores[valid, 4], .9)),
            labels_only_p90=float(np.quantile(scores[valid, 6], .9)),
            torque_peak_ratio=float(z["torque"][valid, 0].max()),
            torque_frame_exceed_fraction=float(np.mean(z["torque"][valid, 0] > 1.00001)),
            torque_joint_exceed_fraction=float(np.mean(z["torque"][valid, 1])),
            max_unexplained_force_N=float(np.linalg.norm(z["root_residual"][valid, :3], axis=1).max()),
            max_unexplained_moment_Nm=float(np.linalg.norm(z["root_residual"][valid, 3:6], axis=1).max()),
            no_support_fraction=float(np.mean(z["min_distance"][valid] > TOLERANCE)),
            no_support_bad_intervals=intervals(bad & (z["min_distance"] > TOLERANCE)),
            penetration_max_mm=float(max(0., -z["min_distance"].min()) * 1000),
            ballistic_frames=int(np.sum(valid & z["flight_interior"])), **primary)
        flight = valid & z["flight_interior"]
        row["ballistic_p90_ms2"] = float(np.quantile(z["ballistic_error"][flight], .9)) if flight.any() else None
        row["ballistic_max_ms2"] = float(z["ballistic_error"][flight].max()) if flight.any() else None
        row["ballistic_fit_max_mm"] = max((e["ballistic_fit_max_mm"] for e in flights), default=None)
        row["root_failed"] = row["bad_fraction"] > calibration["clip_bad_fraction_threshold"]
        row["full_failed"] = row["full_bad_fraction"] > calibration.get(
            "full_clip_bad_fraction_threshold", calibration["clip_bad_fraction_threshold"])
        row["screen_failed"] = row["root_failed"] or row["full_failed"]
        row["persistent_root_failed"] = row["root_failed"] and (
            len(longest) > 0 and longest[1] - longest[0] + 1 > calibration.get("longest_bad_frames_max", 3))
        row["calibrated"] = calibration["calibrated"]
        for j, body in enumerate(CONTACT_LINKS):
            lab = valid & z["labels"][:, j]
            geo = valid & (z["distance"][:, j] <= TOLERANCE)
            gaps = z["distance"][lab, j] * 1000
            contact_rows.append(dict(clip=info["name"], body=body, label_frames=int(lab.sum()),
                label_gap_p50_mm=float(np.quantile(gaps, .5)) if len(gaps) else None,
                label_gap_p90_mm=float(np.quantile(gaps, .9)) if len(gaps) else None,
                label_gap_max_mm=float(gaps.max()) if len(gaps) else None,
                label_air_fraction=float(np.mean(gaps > 20)) if len(gaps) else None,
                support_unlabeled_fraction=float(np.mean(~z["labels"][geo, j])) if geo.any() else None,
                label_air_intervals=intervals(lab & (z["distance"][:, j] > TOLERANCE))))
        if info["physics_present"]:
            p = z["physics_scores"][valid]
            for j, key in enumerate(["measured_geometry", "measured_points", "inverse_measured", "force_aggregation"]):
                row[key + "_p99"] = float(np.nanquantile(p[:, j], .99))
                row[key + "_max"] = float(np.nanmax(p[:, j]))
        summaries.append(row)
        event_rows.extend(dict(clip=info["name"], **e) for e in events)
        frame_rows = []
        for f in range(len(valid)):
            frame_rows.append(dict(frame=f, time_s=f * float(z["dt"]), valid=bool(valid[f]),
                **{key: float(scores[f, j]) for j, key in enumerate(info["score_columns"])},
                root_bad=bool(bad[f]), force_residual_N=float(np.linalg.norm(z["root_residual"][f, :3])),
                moment_residual_Nm=float(np.linalg.norm(z["root_residual"][f, 3:6])),
                torque_peak_ratio=float(z["torque"][f, 0]), min_distance_mm=float(z["min_distance"][f] * 1000),
                support_bodies=int(z["support_count"][f]), ballistic_error_ms2=float(z["ballistic_error"][f]),
                flight_interior=bool(flight[f]),
                **{b + "_gap_mm": float(z["distance"][f, j] * 1000) for j, b in enumerate(CONTACT_LINKS)}))
        write_csv(path.with_suffix(".csv"), frame_rows)
        status = "超过正样本筛查线" if row["screen_failed"] else "处于正样本筛查范围"
        reasons = []
        if row["no_support_bad_intervals"]:
            reasons.append("无几何支撑时仍需外力")
        single = bad & (z["support_count"] == 1)
        if single.any():
            reasons.append(f"单部位支撑不能配平根部力/矩，帧 {intervals(single)}")
        if row["torque_frame_exceed_fraction"] > .01:
            reasons.append("最佳根部配平所需关节力矩超过上限")
        if row["penetration_max_mm"] > 10:
            reasons.append("存在超过10 mm的几何穿入")
        if np.any(z["labels"][valid] & (z["distance"][valid] > TOLERANCE)):
            reasons.append("部分正接触标签距表面超过20 mm")
        conclusions.append(f"- `{info['name']}`：{status}；根残差 P50/P90/max="
            f"{row['root_p50']:.3f}/{row['root_p90']:.3f}/{row['root_max']:.3f}；"
            f"超线 {row['bad_fraction']:.1%}，最长 {longest}；"
            f"无来源力峰值 {row['max_unexplained_force_N']:.1f} N；"
            f"力矩超限帧 {row['torque_frame_exceed_fraction']:.1%}。"
            f"无支撑超线区间：{row['no_support_bad_intervals']}。原因：{'; '.join(reasons) or '未发现超阈值持续根部缺陷'}。")
    write_csv(out / "clips.csv", summaries)
    write_csv(out / "contacts.csv", contact_rows)
    write_csv(out / "events.csv", event_rows)
    write_csv(out / "flights.csv", flight_rows)
    (out / "conclusions.md").write_text("# 逐段检查结论\n\n" +
        ("已按真实 rollout 校准。\n\n" if calibration["calibrated"] else "未校准：仅使用描述性筛查线 0.2。\n\n") + "\n".join(conclusions) + "\n")
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--clip-regex")
    parser.add_argument("--out", type=Path, default=Path("/tmp/codex_dyn/check"))
    args = parser.parse_args()
    entries = discover(args.dataset, args.clip_regex)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "selection.json").write_text(json.dumps(entries, indent=2))
    with ProcessPoolExecutor(max_workers=min(4, len(entries)), mp_context=multiprocessing.get_context("spawn")) as pool:
        jobs = {pool.submit(audit_clip, e, args.out): e for e in entries}
        for future in as_completed(jobs):
            result = future.result()
            print(f"{result['name']}: {result['frames']} frames", flush=True)
    summarize(args.out)
    print(f"Wrote {args.out}/clips.csv and conclusions.md", flush=True)


if __name__ == "__main__":
    main()
