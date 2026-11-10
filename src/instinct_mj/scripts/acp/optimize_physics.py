"""Optimize kinematic ACP references with CPU MuJoCo receding-horizon CEM.

Usage::

    systemd-run --user --unit=acp-codex-physopt-pilot -p MemoryMax=16G \\
        -p WorkingDirectory=$PWD $PWD/.venv/bin/python -m \\
        instinct_mj.scripts.acp.optimize_physics \\
        /home/duanxin/Datasets/acp/v3_train11 \\
        --clip-regex 'vault_gen__206_d002_s21$' --out /tmp/codex_physopt/pilot

The required dataset uses metadata.yaml and terrain_geoms. --clip-regex selects
clips; --out is required. Numerical settings are intentionally in CEMConfig,
which can also be passed to optimize_clip from Python for controlled experiments.
Five sampling workers are used by default, with one BLAS thread per process and no
CUDA. The model comes from mjlab's G1 entity, including its built-in sum-clamped
PD actuators, and ACP make_sim(). Nominal Play physics is used: no DR or delay.
Each action is a 23-DOF offset with target = default_pose + 0.25 * action.

The exported motion, physics forces, contact labels, controls, per-frame tracking
errors, model, and summary describe one uninterrupted simulation from the first
reference state. No root forces, state stitching, kinematic correction, or
reference resets are used during execution. Exported poses sample interval ends
(first at 0.02 s); physics forces average the preceding four substeps, matching
the author rollout convention. The exact initial integration state is saved for
control replay. A one-second terminal hold tests
landing stability. Failed attempts remain outputs and are marked unsuccessful;
physical execution by itself does not establish a successful vault. Original
reference arrays are retained separately for frame-aligned error plots.
"""

from __future__ import annotations

import os

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("MUJOCO_LOG_PATH", "/tmp/codex_physopt/MUJOCO_LOG.TXT")

import argparse
import csv
import json
import multiprocessing
import resource
import time
import types
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path

import mujoco
import numpy as np
import yaml

CONTACT_BODIES = ("left_knee", "right_knee", "pelvis", "left_foot", "right_foot",
                  "left_elbow", "right_elbow", "left_hand", "right_hand")
CONTACT_LINKS = ("left_knee_link", "right_knee_link", "pelvis", "left_ankle_roll_link",
                 "right_ankle_roll_link", "left_elbow_link", "right_elbow_link",
                 "left_wrist_yaw_link", "right_wrist_yaw_link")
TRACKED_LINKS = CONTACT_LINKS + ("torso_link", "left_hip_roll_link", "right_hip_roll_link",
                               "left_shoulder_roll_link", "right_shoulder_roll_link")
DT = 0.02
DECIMATION = 4


# ACP-unspecified: these pilot search settings do not change the ACP MDP.
@dataclass(frozen=True)
class CEMConfig:
    horizon: int = 40
    samples: int = 72
    iterations: int = 4
    execute: int = 4
    knot_stride: int = 5
    elite_fraction: float = 0.125
    noise_rad: float = 0.22
    action_rate_weight: float = 0.0
    workers: int = 5
    seed: int = 20261109
    hold_steps: int = 50


def select_entries(dataset: Path, pattern: str) -> list[dict]:
    import re

    meta = yaml.safe_load((dataset / "metadata.yaml").read_text())
    terrains = {t["terrain_id"]: t for t in meta["terrains"]}
    entries = []
    for entry in meta["motion_files"]:
        name = Path(entry["motion_file"]).stem
        if not re.search(pattern, name) and not re.search(pattern, entry.get("group", "")):
            continue
        terrain = terrains[entry["terrain_id"]]
        entries.append(dict(name=name, path=str(dataset / entry["motion_file"]),
                            group=entry.get("group", ""), terrain=str(dataset / terrain["terrain_geoms"])))
    if not entries:
        raise ValueError(f"No clips match {pattern!r}")
    return entries


def build_model(entry: dict):
    from instinct_mj.tasks.acp.acp_env_cfg import make_sim
    from instinct_mj.tasks.acp.config.g1.acp_tracking_cfg import G1_CFG_SIMPLIFIED_COLLISION
    from instinct_mj.tasks.acp.scene_cfg import TERRAIN_FRICTION

    # Entity builds the same two-element PD actuators and default keyframe as training.
    robot = deepcopy(G1_CFG_SIMPLIFIED_COLLISION).build()
    spec = robot.spec
    # Static primitives use their clip-local world poses (the training tile translation is omitted).
    terrain = spec.worldbody
    geoms = yaml.safe_load(Path(entry["terrain"]).read_text())
    for i, geom in enumerate(geoms):
        quat = np.asarray(geom.get("quat", [1, 0, 0, 0]), dtype=float)
        terrain.add_geom(name=f"physopt_terrain_{i}",
                         type=getattr(mujoco.mjtGeom, "mjGEOM_" + geom["type"].upper()),
                         size=np.pad(geom["size"], (0, 3 - len(geom["size"]))),
                         pos=geom["pos"], quat=quat / np.linalg.norm(quat),
                         group=0, contype=1, conaffinity=1,
                         friction=[TERRAIN_FRICTION, .005, .0001], rgba=[.55, .6, .65, 1])
    from instinct_mj.assets.acp_g1 import ACP_G1_MESHES_DIR
    spec.meshdir = ACP_G1_MESHES_DIR
    # Load the entity-generated XML once so mj_saveLastXML can export this exact model.
    simulation = make_sim()
    spec.nconmax = simulation.nconmax
    spec.njmax = simulation.njmax
    model = mujoco.MjModel.from_xml_string(spec.to_xml())
    simulation.mujoco.apply(model)
    assert model.nv == 29 and model.nu == 46
    return model, geoms


def model_arrays(model):
    pos = np.flatnonzero(model.actuator_gainprm[:, 0] > 0)
    # The velocity element has zero position bias; position elements have -kp.
    pos = pos[model.actuator_biasprm[pos, 1] < 0]
    order = np.argsort(model.actuator_trnid[pos, 0])
    pos = pos[order]
    joints = model.actuator_trnid[pos, 0]
    qadr = model.jnt_qposadr[joints]
    vadr = model.jnt_dofadr[joints]
    default = model.key_qpos[model.key("init_state").id, qadr].copy()
    kp = model.actuator_gainprm[pos, 0].copy()
    kd = np.array([-model.actuator_biasprm[model.actuator_trnid[:, 0] == j, 2].sum() for j in joints])
    bodies = np.array([model.body(n).id for n in TRACKED_LINKS])
    return dict(pos=pos, joints=joints, qadr=qadr, vadr=vadr, default=default, kp=kp, kd=kd, bodies=bodies)


def load_reference(model, entry, hold):
    raw = np.load(entry["path"])
    if abs(float(raw["framerate"]) - 50) > 1e-6:
        raise ValueError("The optimizer expects the ACP 50 Hz dataset")
    a = model_arrays(model)
    n = len(raw["joint_pos"])
    q = np.tile(model.qpos0, (n + hold, 1))
    q[:n, :3] = raw["base_pos_w"]
    quat = raw["base_quat_w"].astype(float)
    q[:n, 3:7] = quat / np.linalg.norm(quat, axis=1, keepdims=True)
    names = list(raw["joint_names"])
    for j, adr in zip(a["joints"], a["qadr"]):
        q[:n, adr] = raw["joint_pos"][:, names.index(model.joint(j).name)]
    q[n:] = q[n - 1]
    half = np.zeros((len(q) - 1, model.nv))
    for f in range(len(half)):
        mujoco.mj_differentiatePos(model, half[f], DT, q[f], q[f + 1])
    v = np.vstack([half[0], (half[:-1] + half[1:]) / 2, half[-1]])
    data = mujoco.MjData(model)
    bp = np.empty((len(q), len(TRACKED_LINKS), 3))
    cp_local = np.zeros((len(q), 9, 3))
    labels = np.zeros((len(q), 9), dtype=bool)
    sidepath = Path(entry["path"]).with_suffix(".contacts.npz")
    contact = np.load(sidepath)
    order = [list(contact["link_names"]).index(name) for name in CONTACT_LINKS]
    labels[:n] = contact["contact_flag"][:, order]
    labels[n:] = labels[n - 1]
    cp = np.zeros((len(q), 9, 3))
    cp[:n] = contact["contact_pos_w"][:, order]
    cp[n:] = cp[n - 1]
    for f in range(len(q)):
        data.qpos[:] = q[f]
        mujoco.mj_kinematics(model, data)
        bp[f] = data.xpos[a["bodies"]]
        rotations = data.xmat[a["bodies"][:9]].reshape(9, 3, 3)
        cp_local[f] = np.einsum("bji,bj->bi", rotations, cp[f] - bp[f, :9])
    # Cancel nominal damping in the seed. CEM can change every PD position target.
    target = q[:, a["qadr"]] + v[:, a["vadr"]] * a["kd"] / a["kp"]
    return dict(q=q, v=v, body=bp, labels=labels, cp=cp, cp_local=cp_local,
                target=target, original_frames=n), a


_WORKER = None


def init_worker(model_path: str, reference: dict):
    global _WORKER
    model = mujoco.MjModel.from_binary_path(model_path)
    _WORKER = (model, mujoco.MjData(model), reference, model_arrays(model))


def rollout_batch(job):
    state, start, sequences = job
    model, data, ref, a = _WORKER
    costs = np.empty(len(sequences))
    local_index = np.full(model.nbody, -1, dtype=int)
    local_index[a["bodies"][:9]] = np.arange(9)
    qerror = np.empty(model.nv)
    for sample, targets in enumerate(sequences):
        mujoco.mj_setState(model, data, state, mujoco.mjtState.mjSTATE_INTEGRATION)
        cost = 0.0
        previous_target = data.ctrl[a["pos"]].copy()
        for k, target in enumerate(targets):
            f = min(start + k + 1, len(ref["q"]) - 1)
            data.ctrl[:] = 0
            data.ctrl[a["pos"]] = target
            mujoco.mj_step(model, data, nstep=DECIMATION)
            mujoco.mj_kinematics(model, data)
            if not np.isfinite(data.qpos).all() or np.linalg.norm(data.qvel) > 250:
                cost += 1e6 * (len(targets) - k)
                break
            mujoco.mj_differentiatePos(model, qerror, 1., ref["q"][f], data.qpos)
            dp = data.qpos[:3] - ref["q"][f, :3]
            db = data.xpos[a["bodies"]] - ref["body"][f]
            value = (100 * np.dot(dp, dp) + 15 * np.dot(qerror[3:6], qerror[3:6])
                     + 2 * np.mean(qerror[6:] ** 2) + 40 * np.mean(np.sum(db ** 2, axis=1))
                     + .5 * np.mean((data.qvel[:6] - ref["v"][f, :6]) ** 2)
                     + .003 * np.mean(((target - ref["target"][min(start + k, len(ref["q"]) - 1)]) / .25) ** 2))
            mask = ref["labels"][f]
            if mask.any():
                rotations = data.xmat[a["bodies"][:9]].reshape(9, 3, 3)
                cp = data.xpos[a["bodies"][:9]] + np.einsum("bij,bj->bi", rotations, ref["cp_local"][f])
                value += 30 * np.sum((cp[mask] - ref["cp"][f, mask]) ** 2) / max(mask.sum(), 1)
            active = np.zeros(9, bool)
            for co in data.contact:
                b1, b2 = model.geom_bodyid[co.geom]
                if (b1 == 0) != (b2 == 0):
                    b = b2 if b1 == 0 else b1
                    j = local_index[b]
                    if j >= 0 and co.dist < .003:
                        active[j] = True
            value += .3 * np.count_nonzero(active != mask)
            if f >= ref["original_frames"]:
                value += (8 * np.dot(data.qvel[:3], data.qvel[:3])
                          + 2 * np.dot(data.qvel[3:6], data.qvel[3:6])
                          + .01 * np.dot(data.qvel[6:], data.qvel[6:])
                          + 8 * np.count_nonzero(~active[3:5]))
            # Keep a graded fall cost so sampling remains useful near poor references.
            value += 200 * max(0., .50 - data.qpos[2]) ** 2
            value += 10 * max(0., .4 - data.xmat[a["bodies"][2], 8]) ** 2
            if k and not ref.get("action_rate_weight", 0.):
                value += .02 * np.mean((target - targets[k - 1]) ** 2)
            residual = target - ref["target"][start + k]
            previous_residual = previous_target - ref["target"][max(0, start + k - 1)]
            value += ref.get("action_rate_weight", 0.) * np.mean(((residual - previous_residual) / DT) ** 2)
            previous_target = target
            cost += value * (2 if k == len(targets) - 1 else 1)
        costs[sample] = cost
    return costs


def get_state(model, data):
    state = np.empty(mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_INTEGRATION))
    mujoco.mj_getState(model, data, state, mujoco.mjtState.mjSTATE_INTEGRATION)
    return state


def sample_plan(pool, model, data, ref, a, start, mean, cfg, rng):
    h = len(mean)
    knots = max(2, int(np.ceil(h / cfg.knot_stride)) + 1)
    x = np.linspace(0, h - 1, knots)
    basis = np.stack([np.interp(np.arange(h), x, np.eye(knots)[j]) for j in range(knots)], axis=1)
    std = np.full((knots, len(a["pos"])), cfg.noise_rad * (.4 if start >= ref["original_frames"] else 1))
    state = get_state(model, data)
    best_cost = np.inf
    best = mean.copy()
    for _iteration in range(cfg.iterations):
        noise = rng.standard_normal((cfg.samples, knots, len(a["pos"]))) * std
        noise[cfg.samples // 2:] = -noise[:cfg.samples - cfg.samples // 2]
        samples = mean[None] + np.einsum("hk,nkj->nhj", basis, noise)
        samples[0] = best
        samples[1] = ref["target"][start:start + h]
        samples[2] = mean
        # Actions are unbounded in ACP, but useful targets need not exceed limits by >0.5 rad.
        limits = model.jnt_range[a["joints"]]
        samples = np.clip(samples, limits[:, 0] - .5, limits[:, 1] + .5)
        batches = np.array_split(samples, cfg.workers)
        costs = np.concatenate(list(pool.map(rollout_batch, [(state, start, batch) for batch in batches])))
        ix = np.argsort(costs)
        if costs[ix[0]] < best_cost:
            best_cost = float(costs[ix[0]])
            best = samples[ix[0]].copy()
        elite = samples[ix[:max(2, int(cfg.samples * cfg.elite_fraction))]]
        mean = .2 * mean + .8 * elite.mean(axis=0)
        deviation = elite - mean
        std = np.maximum(.035, .6 * std + .4 * np.std(deviation[:, np.round(x).astype(int)], axis=0))
    return best, best_cost


def physics_interval(model, data, target, a, frame, raw_rows):
    force = np.zeros((9, 3))
    max_fz = np.zeros(9)
    point_sum = np.zeros((9, 3))
    point_weight = np.zeros(9)
    bybody = {int(b): j for j, b in enumerate(a["bodies"][:9])}
    wrench = np.zeros(6)
    data.ctrl[:] = 0
    data.ctrl[a["pos"]] = target
    for sub in range(DECIMATION):
        mujoco.mj_step(model, data)
        sub_force = np.zeros((9, 3))
        for ci, co in enumerate(data.contact):
            b1, b2 = model.geom_bodyid[co.geom]
            if (b1 == 0) == (b2 == 0):
                continue
            body = int(b2 if b1 == 0 else b1)
            if body not in bybody:
                continue
            j = bybody[body]
            mujoco.mj_contactForce(model, data, ci, wrench)
            fw = co.frame.reshape(3, 3).T @ wrench[:3] * (1 if b1 == 0 else -1)
            sub_force[j] += fw
            weight = np.linalg.norm(fw)
            point_sum[j] += weight * co.pos
            point_weight[j] += weight
            # Raw format is shared with check_dynamics / rollout exports.
            raw_rows.append([frame * DECIMATION + sub, j, *co.pos, *fw, wrench[0],
                             *co.geom, wrench[0] * co.frame[2] * (1 if b1 == 0 else -1)])
        force += sub_force / DECIMATION
        max_fz = np.maximum(max_fz, np.abs(sub_force[:, 2]))
    mujoco.mj_kinematics(model, data)
    cp = data.xpos[a["bodies"][:9]].copy()
    mask = point_weight > 0
    cp[mask] = point_sum[mask] / point_weight[mask, None]
    return force, max_fz, cp


def export_labels(path: Path):
    # Invoke the existing force labeler rather than drifting from its duration rule.
    source = Path("/home/duanxin/Datasets/acp/tools/rollout/physics_sidecar.py")
    module = types.ModuleType("acp_physics_sidecar")
    # Compile the read-only source directly: do not create a bytecode cache in the dataset.
    exec(compile(source.read_text(), str(source), "exec"), module.__dict__)
    params = dict(source="physics", force_min_n=1., min_duration=.2, position_source="contact_pos_w")
    labels = module.label(str(path.with_suffix(".physics.npz")), params)
    np.savez(path.with_suffix(".contacts.npz"), **labels)
    return labels


def wall_metrics(geoms, ref, q, v, body, force_z):
    # Project the actual rotated box onto the reference approach direction.
    candidates = [g for g in geoms if g["type"] == "box" and g["pos"][2] + g["size"][2] > .15]
    if not candidates:
        return dict(crossed=False, stable=False, success=False, reason="no wall box")
    wall = min(candidates, key=lambda g: np.linalg.norm(np.asarray(g["pos"])[:2] - ref["q"][:, :2].mean(axis=0)))
    rotation = np.empty(9)
    quat = np.asarray(wall.get("quat", [1, 0, 0, 0]), float)
    mujoco.mju_quat2Mat(rotation, quat / np.linalg.norm(quat))
    rotation = rotation.reshape(3, 3)
    direction = ref["q"][-1, :2] - ref["q"][0, :2]
    direction /= np.linalg.norm(direction)
    axis = max((rotation[:2, i] for i in range(2)), key=lambda d: abs(d @ direction)).copy()
    axis *= np.sign(axis @ direction)
    extent = np.sum(np.abs(axis @ rotation[:2]) * np.asarray(wall["size"]))
    edge = float(axis @ np.asarray(wall["pos"])[:2] + extent)
    progress = q[:, :2] @ axis - edge
    feet = body[:, 3:5, :2] @ axis - edge
    crossed = bool(np.any((progress > .08) & (feet.min(axis=1) > .03) & (q[:, 2] > .5)))
    tail = slice(-25, None)
    upright = 1 - 2 * (q[:, 4] ** 2 + q[:, 5] ** 2)
    stable_frames = ((q[:, 2] > .55) & (upright > .8)
                     & (np.linalg.norm(v[:, :3], axis=1) < .35)
                     & (np.linalg.norm(v[:, 3:6], axis=1) < .7)
                     & (force_z[:, 3:5] > 1).all(axis=1))
    stable = bool(stable_frames[tail].all())
    finished = bool((progress[tail] > .08).all() and (feet[tail].min(axis=1) > .03).all())
    # A crossing must go over the wall span, not around an end.
    lateral = np.array([-axis[1], axis[0]])
    near = np.abs(progress + extent) < extent + .12
    lateral_extent = np.sum(np.abs(lateral @ rotation[:2]) * np.asarray(wall["size"]))
    over_span = bool(near.any() and np.max(np.abs((q[near, :2] - wall["pos"][:2]) @ lateral)) < lateral_extent)
    return dict(crossed=crossed, stable=stable, success=crossed and stable and finished and over_span,
                finished_far_side=finished, over_wall_span=over_span, final_root_clearance_m=float(progress[-1]),
                final_speed_ms=float(np.linalg.norm(v[-1, :3])), final_height_m=float(q[-1, 2]),
                final_upright=float(upright[-1]), wall_axis=axis.tolist(), wall_far_edge_m=edge)


def tracking_report(out, name, ref, q, v, body, a, forces, geoms):
    n = ref["original_frames"]
    root = np.linalg.norm(q[:, :3] - ref["q"][:, :3], axis=1)
    bp = np.linalg.norm(body - ref["body"], axis=2)
    joint = np.sqrt(np.mean((q[:, a["qadr"]] - ref["q"][:, a["qadr"]]) ** 2, axis=1))
    angle = 2 * np.arccos(np.clip(np.abs(np.sum(q[:, 3:7] * ref["q"][:, 3:7], axis=1)), 0, 1))
    original_speed = np.linalg.norm(ref["v"][:, :2], axis=1)
    actual_speed = np.linalg.norm(v[:, :2], axis=1)
    offset = ref.get("time_offset_s", 0.)
    rows = []
    for f in range(len(q)):
        rows.append(dict(frame=f, time_s=f * DT + offset, terminal_hold=f >= n, root_error_m=root[f],
                         body_error_m=bp[f].mean(), joint_rmse_rad=joint[f], root_angle_rad=angle[f],
                         reference_speed_ms=original_speed[f], executed_speed_ms=actual_speed[f],
                         **{b + "_error_m": bp[f, j] for j, b in enumerate(TRACKED_LINKS)}))
    with (out / f"{name}.errors.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez(out / f"{name}.reference.npz", qpos=ref["q"], qvel=ref["v"], body_pos=ref["body"])
    summary = dict(root_rmse_m=float(np.sqrt(np.mean(root[:n] ** 2))),
                   body_rmse_m=float(np.sqrt(np.mean(bp[:n] ** 2))),
                   joint_rmse_rad=float(np.sqrt(np.mean(joint[:n] ** 2))),
                   root_angle_rmse_rad=float(np.sqrt(np.mean(angle[:n] ** 2))),
                   original_frames=n, exported_frames=len(q),
                   **wall_metrics(geoms, ref, q, v, body, forces))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.arange(len(q)) * DT + offset
    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    axes[0].plot(t, root, label="root"); axes[0].plot(t, bp.mean(axis=1), label="14-body mean")
    axes[0].set_ylabel("position error (m)"); axes[0].legend()
    axes[1].plot(t, joint); axes[1].set_ylabel("joint RMSE (rad)")
    axes[2].plot(t, original_speed, label="reference"); axes[2].plot(t, actual_speed, label="executed")
    axes[2].set_ylabel("horizontal speed (m/s)"); axes[2].legend()
    axes[3].plot(t, q[:, 2], label="executed"); axes[3].plot(t, ref["q"][:, 2], label="reference")
    axes[3].set_ylabel("root height (m)"); axes[3].set_xlabel("time (s)"); axes[3].legend()
    for ax in axes:
        ax.axvline((n - 1) * DT + offset, color="k", linestyle="--", alpha=.4); ax.grid(alpha=.2)
    fig.suptitle(name); fig.tight_layout(); fig.savefig(out / f"{name}.errors.png", dpi=130); plt.close(fig)
    return summary


def optimize_clip(entry: dict, out: Path, cfg: CEMConfig = CEMConfig()):
    if not 1 <= cfg.workers <= 6:
        raise ValueError("Use one to six sampling workers")
    start_time = time.monotonic()
    cpu_start = resource.getrusage(resource.RUSAGE_SELF).ru_utime
    child_cpu_start = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime
    cgroup = Path("/sys/fs/cgroup") / Path("/proc/self/cgroup").read_text().strip().split("::")[-1].lstrip("/")
    memory_peak = 0
    out.mkdir(parents=True, exist_ok=True)
    name = entry["name"]
    model, geoms = build_model(entry)
    ref, a = load_reference(model, entry, cfg.hold_steps)
    ref["action_rate_weight"] = cfg.action_rate_weight
    model_path = out / f"{name}.mjb"
    mujoco.mj_saveModel(model, str(model_path))
    mujoco.mj_saveLastXML(str(out / f"{name}.xml"), model)
    data = mujoco.MjData(model)
    data.qpos[:] = ref["q"][0]
    data.qvel[:] = ref["v"][0]
    if cfg.action_rate_weight:
        data.ctrl[a["pos"]] = ref["target"][0]
    mujoco.mj_forward(model, data)
    # Training reset also removes initial terrain penetration. Record the exact lift.
    penetration = max((-co.dist for co in data.contact if (model.geom_bodyid[co.geom[0]] == 0)
                       != (model.geom_bodyid[co.geom[1]] == 0)), default=0.)
    lift = max(0., penetration)
    data.qpos[2] += lift
    mujoco.mj_forward(model, data)
    initial_state = get_state(model, data)
    q, v, body, controls, force, force_z, cp = [], [], [], [], [], [], []
    raw_rows, planning, executed_targets = [], [], []
    rng = np.random.default_rng(cfg.seed)
    total = len(ref["q"])
    mean = ref["target"][:min(cfg.horizon, total - 1)].copy()
    with ProcessPoolExecutor(max_workers=cfg.workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=init_worker, initargs=(str(model_path), ref)) as pool:
        f = 0
        while f < total - 1:
            horizon = min(cfg.horizon, total - 1 - f)
            tick = time.monotonic()
            plan, value = sample_plan(pool, model, data, ref, a, f, mean[:horizon], cfg, rng)
            planning.append(dict(frame=f, cost=value, wall_s=time.monotonic() - tick))
            if (cgroup / "memory.current").exists():
                memory_peak = max(memory_peak, int((cgroup / "memory.current").read_text()))
            steps = min(cfg.execute, horizon)
            for k in range(steps):
                q.append(data.qpos.copy()); v.append(data.qvel.copy())
                body.append(data.xpos[a["bodies"]].copy()); controls.append((plan[k] - a["default"]) / .25)
                executed_targets.append(plan[k].copy())
                fw, fz, point = physics_interval(model, data, plan[k], a, f + k, raw_rows)
                force.append(fw); force_z.append(fz); cp.append(point)
            f += steps
            following = min(cfg.horizon, total - 1 - f)
            mean = np.vstack([plan[steps:], ref["target"][f + horizon - steps:f + following]])[:following]
            print(f"{name} frame={f}/{total - 1} cost={value:.3f} root={data.qpos[:3].round(3).tolist()} elapsed={time.monotonic() - start_time:.1f}s", flush=True)
    q.append(data.qpos.copy()); v.append(data.qvel.copy()); body.append(data.xpos[a["bodies"]].copy())
    controls.append(controls[-1]); executed_targets.append(executed_targets[-1]); force.append(force[-1]); force_z.append(force_z[-1]); cp.append(cp[-1])
    q, v, body, force, force_z, cp = map(np.asarray, (q, v, body, force, force_z, cp))
    # Match the author's rollout convention: each pose is the interval-end state,
    # paired with forces averaged over that actual interval. No endpoint force is invented.
    q, v, body = q[1:], v[1:], body[1:]
    force, force_z, cp = force[:-1], force_z[:-1], cp[:-1]
    controls, executed_targets = controls[:-1], executed_targets[:-1]
    input_frames = ref["original_frames"]
    ref = {key: value[1:] if isinstance(value, np.ndarray) and value.ndim and len(value) == total else value
           for key, value in ref.items()}
    ref["original_frames"] -= 1
    ref["time_offset_s"] = DT
    names = np.array([model.joint(j).name for j in a["joints"]])
    path = out / f"{name}.npz"
    np.savez(path, framerate=np.array(50.), joint_names=names, joint_pos=q[:, a["qadr"]],
             base_pos_w=q[:, :3], base_quat_w=q[:, 3:7])
    np.savez(path.with_suffix(".physics.npz"), framerate=np.array(50.), physics_dt=np.array(.005),
             link_names=np.array(CONTACT_LINKS), qpos=q, qvel=v, force_w=force, force_z=force[:, :, 2],
             contact_pos_w=cp, raw_contacts=np.asarray(raw_rows).reshape(-1, 12), force_peak_z=force_z, initial_state=initial_state,
             sample_time_s=(np.arange(len(q)) + 1) * DT, state_sampling=np.array("interval_end"),
             raw_columns=np.array(["physics_step", "link_index", "x", "y", "z", "fx", "fy", "fz",
                                   "normal_force", "geom1", "geom2", "normal_force_z"]))
    np.savez(path.with_suffix(".controls.npz"), framerate=np.array(50.), joint_names=names,
             action=np.asarray(controls), target=np.asarray(executed_targets), default_pose=a["default"])
    export_labels(path)
    summary = tracking_report(out, name, ref, q, v, body, a, np.abs(force[:, :, 2]), geoms)
    summary.update(name=name, source=entry, config=asdict(cfg), initial_lift_m=lift,
                   input_frames=input_frames, reference_frame_offset=1, state_sampling="interval_end",
                   wall_seconds=time.monotonic() - start_time,
                   parent_cpu_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_utime - cpu_start,
                   children_cpu_seconds=resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime - child_cpu_start,
                   peak_cgroup_memory_mb=memory_peak / 1024 ** 2,
                   peak_parent_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                   peak_child_rss_mb=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024,
                   planning=planning)
    (out / f"{name}.summary.json").write_text(json.dumps(summary, indent=2))
    (out / f"{name}.geoms.yaml").write_text(yaml.safe_dump(geoms))
    print(json.dumps({k: v for k, v in summary.items() if k != "planning"}), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--clip-regex", default="vault_gen__206_d002_s21$|vault_gen_slow__kvs206_d002_s21$|vault_gen__206_d0_s11$|vault__cat_vault_00$")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    entries = select_entries(args.dataset.expanduser().resolve(), args.clip_regex)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "selection.json").write_text(json.dumps(entries, indent=2))
    for entry in entries:
        optimize_clip(entry, args.out)


if __name__ == "__main__":
    main()
