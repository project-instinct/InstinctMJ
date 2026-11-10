"""Analytic MDP of the ACP tracking task (``Instinct-ACP-Track-G1-v1``) shared by the simulator labels and the world model.

State tuple ``S = (p, R, v, w, q, qd)`` as in ``instinct_mj.rl.supermimic.mdp`` (root = ``pelvis``, the 23 joints of the robot in simulator
order). The *base* of the task is the PELVIS (IMU, scans, goal and contact frames, base rewards / terminations): pelvis quantities
come from the forward kinematics of the 9 contact links (``CONTACT_LINKS``, the pelvis is one of them). Everything mirrors
``instinct_mj.tasks.acp``: the ``policy`` observation group (``acp_env_cfg.make_observations``), the Table 3 rewards
(``tasks/acp/mdp/rewards.py``) and the terminations (``tasks/acp/mdp/terminations.py``); the weights, kernel scales, thresholds
and noise ranges are read from the live env.

The domain-randomization context ``xi`` of an env (what the world model conditions on) is laid out in ``XI_*``: the encoder bias
and the action delay change every episode, the rest is constant after startup.
"""

from __future__ import annotations

import torch

from instinct_mj.rl.supermimic.mdp import invariant_features
from instinct_mj.rl.supermimic.rotations import mat_to_6d, mat_to_quat, mm, mv, rot_angle_sq, yaw_mat
from instinct_mj.rl.supermimic.terrain import HeightScanner

from .kinematics import ACPKinematics
from .motion import NUM_CONTACT, PELVIS, ACPMotion
from .planner import CONTACT_LINKS, TRACKED_LINKS
from .sdf_contact import VIOLATION_D, VIOLATION_LINKS, VIOLATION_REF_CLEARANCE, VIOLATION_TAU, get_violation_sdf

FEET = [CONTACT_LINKS.index("left_ankle_roll_link"), CONTACT_LINKS.index("right_ankle_roll_link")]  # the foot_pos_too_far links
VIOLATION_IDS = [TRACKED_LINKS.index(n) for n in VIOLATION_LINKS]  # ``kin.forward_all`` rows of the contact_violation bodies
from instinct_mj.tasks.acp.mdp.observations import WIDE_SCAN_NUM_X, WIDE_SCAN_NUM_Y, wide_scan_offsets

ACT_DIM = 23
PROPRIO_DIM = 3 + 3 + ACT_DIM * 3  # base ang vel, projected gravity, joint pos, joint vel, last action (one step)
SCAN_NY, SCAN_NX = 11, 17  # planner scan, y-major (y outer, x inner), as the env
TRACKER_Y, TRACKER_X = slice(2, 9), slice(4, 13)  # tracker scan = central 7 (y) x 9 (x) block
SCAN_OFFSET = 0.5  # scan value = pelvis_z - ground_z - 0.5
POLICY_OBS_DIM = 5 * PROPRIO_DIM + 63 + 3 * NUM_CONTACT + 3
ODO_HIST = 6  # pelvis pose history length of the odometry model (``PelvisRayCastSensorCfg.history_length``)
OBS_TERMS = ("base_ang_vel", "projected_gravity", "joint_pos", "joint_vel", "last_action")

NUM_J = 23  # joints of the robot (the wrists of the playground G1 are welded); all of them are action joints
STATE_DIM = 3 + 4 + 3 + 3 + NUM_J + NUM_J
INV_DIM = 6 + 3 + 3 + NUM_J + NUM_J

# xi: encoder bias (NUM_J), then startup context [torso com offset (3), kp scale, kd scale (NUM_J each), masses / nominal (13: torso, 12 leg
# links), armature scale (NUM_J), friction (1)], then the action delay in physics steps (1: one lag per env, shared by all actuator groups)
XI_BIAS = slice(0, NUM_J)
XI_COM = slice(NUM_J, NUM_J + 3)
XI_KP = slice(NUM_J + 3, 2 * NUM_J + 3)
XI_KD = slice(2 * NUM_J + 3, 3 * NUM_J + 3)
XI_MASS = slice(3 * NUM_J + 3, 3 * NUM_J + 16)
XI_ARM = slice(3 * NUM_J + 16, 4 * NUM_J + 16)
XI_FRIC = slice(4 * NUM_J + 16, 4 * NUM_J + 17)
XI_STATIC = slice(NUM_J, 4 * NUM_J + 17)
XI_STATIC_DIM = 3 * NUM_J + 17
XI_LAG = slice(4 * NUM_J + 17, 4 * NUM_J + 18)
XI_DIM = 4 * NUM_J + 18
XI_DYN_DIM = NUM_J + 1  # encoder bias + lag
XI_FLOOR = torch.cat((torch.full((NUM_J,), 5e-3), torch.full((3,), 1e-2), torch.full((2 * NUM_J,), 0.02), torch.full((13,), 0.02),
                      torch.full((NUM_J,), 0.02), torch.full((1,), 0.05), torch.full((1,), 0.5)))  # fmt: skip
# per-component floors on the world-model error normalization (pos, rot, lin vel, ang vel, q, qd increments)
ERR_FLOOR = torch.cat((torch.full((3,), 3e-3), torch.full((3,), 5e-3), torch.full((3,), 2e-2), torch.full((3,), 5e-2),
                       torch.full((NUM_J,), 3e-3), torch.full((NUM_J,), 5e-2)))  # fmt: skip


def make_xi(static: torch.Tensor, dyn: torch.Tensor) -> torch.Tensor:
    """Full context (..., XI_DIM) from the startup part (..., XI_STATIC_DIM) and the per-episode part (..., XI_DYN_DIM) = [bias, lag]."""
    return torch.cat((dyn[..., :NUM_J], static, dyn[..., NUM_J:]), -1)


def split_xi_dyn(xi: torch.Tensor) -> torch.Tensor:
    return torch.cat((xi[..., XI_BIAS], xi[..., XI_LAG]), -1)


def wrap_to_pi(x: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(x), torch.cos(x))


def targets_to_world(est_pos: torch.Tensor, est_yaw: torch.Tensor, targets_b: torch.Tensor) -> torch.Tensor:
    """World positions (B, 9, 3) of contact targets given in the base-yaw frame of the ESTIMATED pelvis pose (the frame the planner works
    in: delayed and drifted pose of the observation it was computed from). Detached: the targets are commands, not part of the dynamics."""
    c, s = torch.cos(est_yaw), torch.sin(est_yaw)
    x, y = targets_b[..., 0], targets_b[..., 1]
    world = torch.stack((c.unsqueeze(-1) * x - s.unsqueeze(-1) * y, s.unsqueeze(-1) * x + c.unsqueeze(-1) * y, targets_b[..., 2]), -1)
    return (est_pos.unsqueeze(1) + world).detach()


def planned_contact_error(body_pos_w: torch.Tensor, pelvis_R: torch.Tensor, targets_w: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """sum_b m_b |p_exec,B_b - p^_b|^2 of ``track_planned_contact`` with the planner targets as world points: the executed body positions
    (differentiable world positions) and the targets are expressed in the base-yaw frame of the TRUE pelvis, i.e. the error is
    ``R_yaw^T (p_exec,W - p^_W)``, which keeps the gradient w.r.t. the root translation (a base-frame target would cancel it)."""
    diff = mv(yaw_mat(pelvis_R).transpose(-1, -2).unsqueeze(1), body_pos_w - targets_w)
    return (mask * (diff**2).sum(-1)).sum(-1)


def heading(R: torch.Tensor) -> torch.Tensor:
    """Yaw of a rotation matrix from its projected x axis (``yaw_of_quat``)."""
    return torch.atan2(R[..., 1, 0], R[..., 0, 0])


class ACPMDP:
    def __init__(self, kin: ACPKinematics, motion: ACPMotion, scanner: HeightScanner, env, dt: float, device: str):
        """All task constants are read from the live env (reward / termination params, observation noise, actuators, action term)."""
        self.kin, self.motion, self.scanner, self.dt, self.device = kin, motion, scanner, dt, device
        robot = env.scene["robot"]
        rm = env.reward_manager
        self.w = {n: rm.get_term_cfg(n).weight for n in rm.active_terms["rewards"]}
        self.p = {n: rm.get_term_cfg(n).params for n in rm.active_terms["rewards"]}
        tm = env.termination_manager
        self.wide_scan = False  # the runner sets it from ``wm_scan``
        self.wide_offsets = wide_scan_offsets(device)
        self.geo = None  # GeometricContact (set by the runner with ``geo_contact``)
        self.violation_sdf = get_violation_sdf(env)  # exact torso-terrain distance of the contact_violation termination
        self.contact_acc_vel = self.p["contact_acc"].get("mode", "pos2") == "vel1"  # contact_acc from the link velocities: ``lp_hist`` carries the velocities of the states -1 and 0
        self.stopgrad: tuple[str, ...] = ()  # reward terms whose gradient is cut in ``reward`` (the runner sets it from ``bptt_stopgrad_terms``)
        self.soft_tz, self.soft_tt = 0.02, 0.04  # temperatures of the soft terminations (the runner sets them from soft_term_tau_dist / soft_term_tau_tilt)
        self.tp = {n: tm.get_term_cfg(n).params for n in ("base_pos_too_far", "base_pg_too_far", "base_xy_too_far", "foot_pos_too_far")}  # the v1 failure terminations
        self.num_track = motion.track_p.shape[1]  # the SuperMimic tracked bodies (the contact links first); the kinematics evaluate all of them
        # nominal joint data (xi carries the per-env encoder bias)
        self.default_q = robot.data.default_joint_pos[0].to(device)
        limits = robot.data.soft_joint_pos_limits[0]
        self.soft_lo, self.soft_hi = limits[:, 0].to(device), limits[:, 1].to(device)
        action_term = env.action_manager.get_term("joint_pos")
        self.act_ids = action_term.target_ids.to(device)
        assert len(self.act_ids) == ACT_DIM
        self.sel = torch.zeros(NUM_J, ACT_DIM, device=device)  # action index -> joint index
        self.sel[self.act_ids, torch.arange(ACT_DIM, device=device)] = 1.0
        self.act_mask = self.sel.sum(1)
        self.act_scale_j = mv(self.sel, action_term.scale[0].to(device))
        # PD (nominal kp / kd, per-env scales are in xi); effort limits
        model = env.sim.mj_model
        self.kp_nom = torch.zeros(NUM_J, device=device)
        self.kd_nom = torch.zeros(NUM_J, device=device)
        self.effort_limit = torch.zeros(NUM_J, device=device)
        for act in robot.actuators:
            n = len(act.target_ids)
            ctrl = [int(c) for c in act.global_ctrl_ids]
            for k, j in enumerate(act.target_ids.tolist()):
                self.kp_nom[j] = float(model.actuator_gainprm[ctrl[k], 0])
                self.kd_nom[j] = float(model.actuator_gainprm[ctrl[n + k], 0])
                self.effort_limit[j] = float(abs(model.jnt_actfrcrange[int(robot.indexing.joint_ids[j])]).max())
        self.vel_limit = rm.get_term_cfg("dof_vel_limits").func.velocity_limits.to(device)
        # observation noise (uniform half-widths) of the policy group
        terms = env.cfg.observations["policy"].terms
        half = lambda name: float(terms[name].noise.n_max) if terms[name].noise is not None else 0.0  # noqa: E731
        self.proprio_noise = torch.cat([torch.full((n,), half(name)) for n, name in zip((3, 3, ACT_DIM, ACT_DIM, ACT_DIM), OBS_TERMS)]).to(device)
        self.scan_noise = half("tracker_height_scan")
        self.target_noise = float(terms["contact_targets"].params["noise"])
        self.planner_z_noise = float(env.cfg.observations["planner"].terms["planner_height_scan"].params["z_noise_std"])
        self.goal_noise = half("goal_command")
        goal_cfg = env.command_manager.get_term("acp_goal").cfg
        self.goal_min_distance = goal_cfg.min_distance
        self.goal_lookahead = goal_cfg.lookahead_range_s
        self.footprint = (motion.footprint_x, motion.footprint_y)
        geom_ids = robot.indexing.geom_ids.cpu().numpy()
        self.collision_geom_ids = torch.as_tensor(geom_ids[model.geom_contype[geom_ids] != 0], device=device).long()
        # hard joint range of the MJCF (MuJoCo enforces it as a constraint; the soft limits above are the reward's)
        jnt = robot.indexing.joint_ids.cpu().numpy()
        assert model.jnt_limited[jnt].all()
        self.hard_lo = torch.as_tensor(model.jnt_range[jnt, 0], dtype=torch.float32, device=device)
        self.hard_hi = torch.as_tensor(model.jnt_range[jnt, 1], dtype=torch.float32, device=device)
        # odometry model of the actor-side pelvis pose (``PelvisRayCastSensor``)
        sensor_cfg = env.scene["height_scanner"].cfg
        assert sensor_cfg.use_estimated_pose and sensor_cfg.history_length == ODO_HIST
        self.odo_dt = sensor_cfg.history_dt
        self.decimation, self.physics_dt = env.cfg.decimation, env.physics_dt
        self.xi_dim = XI_DIM
        self.xi_mean = torch.zeros(XI_DIM, device=device)
        self.xi_scale = torch.ones(XI_DIM, device=device)

    # ---------------- sim context ----------------
    def set_xi_stats(self, table: torch.Tensor) -> None:
        """Fixed normalization of the per-env context from the DR sample at setup."""
        self.xi_mean = table.mean(0)
        self.xi_scale = torch.maximum(table.std(0, correction=0), XI_FLOOR.to(table.device))

    def normalize_xi(self, xi: torch.Tensor) -> torch.Tensor:
        return (xi - self.xi_mean) / self.xi_scale

    def pad(self, a: torch.Tensor) -> torch.Tensor:
        """Action (..., ACT_DIM) -> joint order (..., NUM_J)."""
        return mv(self.sel, a)

    # ---------------- observations (policy group of the task) ----------------
    @staticmethod
    def base_of(fk) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pelvis (position, rotation, linear velocity, angular velocity) from the contact-link FK."""
        return fk[0][:, PELVIS], fk[1][:, PELVIS], fk[2][:, PELVIS], fk[3][:, PELVIS]

    def scan(self, base_p: torch.Tensor, base_R: torch.Tensor) -> torch.Tensor:
        """Scan (B, 187, y-major): pelvis_z - ground - 0.5 on the yaw-aligned grid of the given pelvis pose. The env's ``critic_height_scanner`` reads the
        true pose; the actor / planner scans of the env read the odometry-estimated pose (delay + drift, ``estimated_pose``), which the runner passes
        for the actor-side observations. The imagination's critic input uses the true imagined pose."""
        return self.scanner.height_scan(base_p, base_R) - SCAN_OFFSET

    def critic_scan(self, base_p: torch.Tensor, base_R: torch.Tensor) -> torch.Tensor:
        """Scan of the world model and the critic (true pelvis pose): ``wm_scan="wide"`` = the 31 x 15 forward grid (0 .. 1.5 m, +-0.35 m, 0.05 m, y-major; (B, 465)),
        "planner" = ``scan`` (17 x 11). Same encoding (pelvis_z - ground - 0.5), the exact ray cast of the scanner."""
        if not self.wide_scan:
            return self.scan(base_p, base_R)
        sc = self.scanner
        if hasattr(sc, "exact_height"):
            height = sc.height_scan(base_p, base_R, self.wide_offsets, (WIDE_SCAN_NUM_Y, WIDE_SCAN_NUM_X))
        else:  # the 0.05 m height map
            xy = base_p[:, None, :2] + mm(self.wide_offsets, yaw_mat(base_R)[..., :2, :2].transpose(-1, -2))
            height = torch.clamp(base_p[:, 2:3] - sc.ground_height(xy), max=sc.max_distance)
        return height - SCAN_OFFSET

    @property
    def critic_scan_dim(self) -> int:
        return WIDE_SCAN_NUM_X * WIDE_SCAN_NUM_Y if self.wide_scan else SCAN_NY * SCAN_NX

    @staticmethod
    def tracker_scan(scan: torch.Tensor) -> torch.Tensor:
        """Central 9 x 7 block of the planner scan (B, 63, y-major)."""
        return scan.view(-1, SCAN_NY, SCAN_NX)[:, TRACKER_Y, TRACKER_X].flatten(1)

    def proprio(self, base_R, base_w, q, qd, a_prev, xi) -> torch.Tensor:
        """Noise-free one-step proprioception (..., 75): pelvis ang vel (pelvis frame), projected gravity, biased joint pos relative to
        the default, joint vel, last action; the 23 action joints."""
        g = -base_R[..., 2, :]  # R^T (0, 0, -1)
        w_b = mv(base_R.transpose(-1, -2), base_w)
        q_rel = q[..., self.act_ids] + xi[..., XI_BIAS][..., self.act_ids] - self.default_q[self.act_ids]
        return torch.cat((w_b, g, q_rel, qd[..., self.act_ids], a_prev), -1)

    def add_noise(self, o: torch.Tensor) -> torch.Tensor:
        return o + (2 * torch.rand_like(o) - 1) * self.proprio_noise

    def targets(self, base_p, base_R, f, phase, origin) -> tuple[torch.Tensor, torch.Tensor]:
        """Reference next-onset contact targets of frame ``f`` in the pelvis yaw frame (B, 9, 3), zero where invalid, and the valid
        mask (B, 9) (``AcpReferenceContacts.targets`` of the true pose)."""
        m = self.motion
        pos_w = m.next_pos(f, phase) + origin.unsqueeze(1)
        pos_b = mv(yaw_mat(base_R).transpose(-1, -2).unsqueeze(1), pos_w - base_p.unsqueeze(1))
        valid = m.next_valid(f, phase) & (pos_b[..., 0].abs() <= self.footprint[0]) & (pos_b[..., 1].abs() <= self.footprint[1])
        return pos_b * valid.unsqueeze(-1), valid

    def noisy_targets(self, pos_b: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Uniform noise on the valid entries only (the env's ``contact_targets`` term), flattened (B, 27)."""
        return (pos_b + self.target_noise * (2 * torch.rand_like(pos_b) - 1) * valid.unsqueeze(-1)).flatten(1)

    def goal(self, base_p, base_R, goal_pos, goal_yaw) -> torch.Tensor:
        """Noise-free goal command (B, 3) (``AcpGoalCommand.command``): d_xy / max(|d_xy|, r) in the pelvis yaw frame, wrapped heading error."""
        yaw = heading(base_R)
        d_w = goal_pos[:, :2] - base_p[:, :2]
        c, s = torch.cos(yaw), torch.sin(yaw)
        d = torch.stack((c * d_w[:, 0] + s * d_w[:, 1], -s * d_w[:, 0] + c * d_w[:, 1]), -1)
        direction = d / torch.clamp(d.norm(dim=-1, keepdim=True), min=self.goal_min_distance)
        return torch.cat((direction, wrap_to_pi(goal_yaw - yaw).unsqueeze(-1)), -1)

    def goal_update(self, f, phase, origin, goal_pos, goal_yaw, goal_t):
        """The env's ``AcpGoalCommand._update_command``: an env whose remaining lookahead (goal time - reference time) dropped below the lower
        bound of the lookahead range draws a new goal U(lookahead range) seconds ahead (capped at the clip end). ``goal_t`` is the goal time in clip-local frames (seconds * fps)."""
        m = self.motion
        clip = m.clip_id[f]
        local = (f - m.clip_start[clip]).float() + phase
        passed = goal_t - local < self.goal_lookahead[0] * m.framerate
        look = torch.empty_like(local).uniform_(*self.goal_lookahead) * m.framerate
        t_new = torch.minimum(local + look, (m.clip_len[clip] - 1).float())
        pos_new, yaw_new = m.goal_target(f, torch.round(t_new).long(), origin)
        return (torch.where(passed.unsqueeze(-1), pos_new, goal_pos), torch.where(passed, yaw_new, goal_yaw),
                torch.where(passed, t_new, goal_t))  # fmt: skip

    def noisy_goal(self, g: torch.Tensor) -> torch.Tensor:
        return g + self.goal_noise * (2 * torch.rand_like(g) - 1)

    def noisy_tracker_scan(self, scan63: torch.Tensor) -> torch.Tensor:
        return scan63 + self.scan_noise * (2 * torch.rand_like(scan63) - 1)

    def policy_obs(self, o_hist: torch.Tensor, scan63: torch.Tensor, targets27: torch.Tensor, goal3: torch.Tensor) -> torch.Tensor:
        """Flat ``policy`` observation group (468) in the env's term order and flattening: the 5-step histories (oldest first, time
        major) of base_ang_vel, projected_gravity, joint_pos, joint_vel, last_action (from ``o_hist`` (B, 5, 75)), then the tracker
        height scan, the contact targets and the goal command."""
        hist = [o_hist[..., a:b].flatten(1) for a, b in ((0, 3), (3, 6), (6, 29), (29, 52), (52, 75))]
        return torch.cat((*hist, scan63, targets27, goal3), -1)

    # ---------------- odometry (actor-side pelvis pose) ----------------
    def pose_history(self, pelvis_p: torch.Tensor, pelvis_R: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Pelvis pose history (position (B, ODO_HIST, 3), quaternion (B, ODO_HIST, 4)) from pelvis poses ordered OLDEST first
        ((B, ODO_HIST, 3), (B, ODO_HIST, 3, 3)); the result is newest first, as ``PelvisRayCastSensor.history_*``."""
        return pelvis_p.flip(1), mat_to_quat(pelvis_R).flip(1)

    def push_pose(self, hist, pelvis_p: torch.Tensor, pelvis_R: torch.Tensor):
        hp, hq = hist
        return torch.cat((pelvis_p.unsqueeze(1), hp[:, :-1]), 1), torch.cat((mat_to_quat(pelvis_R).unsqueeze(1), hq[:, :-1]), 1)

    def estimated_pose(self, hist, delay: torch.Tensor, drift: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Estimated pelvis position (B, 3) and yaw (B,) exactly as ``PelvisRayCastSensor._update_estimated_pose``: the true pose ``delay``
        seconds ago (linear interpolation between the two control-step poses of the history, normalized quaternion lerp) plus the
        per-episode world drift. Only the heading of the estimated rotation is used by the actor-side transforms."""
        hp, hq = hist
        lag = (delay / self.odo_dt).clamp(0.0, ODO_HIST - 1.0)
        lower = lag.floor().long()
        upper = (lower + 1).clamp(max=ODO_HIST - 1)
        frac = (lag - lower).unsqueeze(-1)
        env = torch.arange(len(lag), device=lag.device)
        pos = (1 - frac) * hp[env, lower] + frac * hp[env, upper] + drift
        q0, q1 = hq[env, lower], hq[env, upper]
        q1 = torch.where((q0 * q1).sum(-1, keepdim=True) < 0, -q1, q1)
        q = (1 - frac) * q0 + frac * q1
        w, x, y, z = (q / q.norm(dim=-1, keepdim=True)).unbind(-1)
        return pos, torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    @staticmethod
    def yaw_rotation(yaw: torch.Tensor) -> torch.Tensor:
        """Rotation about z (B, 3, 3): the heading-only pose of the transforms that take a rotation matrix."""
        c, s = torch.cos(yaw), torch.sin(yaw)
        o, one = torch.zeros_like(c), torch.ones_like(c)
        return torch.stack((c, -s, o, s, c, o, o, o, one), -1).reshape(yaw.shape + (3, 3))

    # ---------------- joint range ----------------
    def clamp_joints(self, S: tuple, margin: float = 0.0) -> tuple:
        """MuJoCo enforces the hard joint range as a constraint: a predicted joint position beyond it is put on the range and a velocity
        pushing further out is removed (plain clamp; the gradient of a saturated joint is zero, as in the simulator). ``margin``: the range is widened by it (a NaN guard
        when the imagination does not project: ``imagine_clamp_joints=False``)."""
        p, R, v, w, q, qd = S
        hi, lo = self.hard_hi + margin, self.hard_lo - margin
        qc = torch.maximum(torch.minimum(q, hi), lo)
        out = ((q > hi) & (qd > 0)) | ((q < lo) & (qd < 0))
        return p, R, v, w, qc, torch.where(out, torch.zeros_like(qd), qd)

    def joint_violation(self, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Fraction of (sample, controlled joint) pairs beyond the soft limits of the reward and beyond the hard range."""
        qa = q[..., self.act_ids]
        soft = ((qa < self.soft_lo[self.act_ids]) | (qa > self.soft_hi[self.act_ids])).float().mean()
        hard = ((qa < self.hard_lo[self.act_ids]) | (qa > self.hard_hi[self.act_ids])).float().mean()
        return soft, hard

    # ---------------- Stage 2: planner interface ----------------
    def planner_points(self, scan: torch.Tensor, noisy: bool = True) -> torch.Tensor:
        """Planner input points (B, 187, 3) exactly as the env's ``planner`` group gives them: (x, y, pelvis_z - ground_z - 0.5), y-major in
        the base-yaw frame (``AttentionContactPlanner`` converts to physical base-relative z itself). ``noisy``: Gaussian z noise with the
        ``z_noise_std`` of the env's ``planner_height_scan`` term (0.03)."""
        h = scan + self.planner_z_noise * torch.randn_like(scan) if noisy else scan
        xy = self.scanner.offsets.unsqueeze(0).expand(scan.shape[0], -1, -1)
        return torch.cat((xy, h.unsqueeze(-1)), -1)

    # ---------------- reward / termination ----------------
    def torque_terms(self, q1, qd1, qd0, a, a_prev, xi) -> tuple[torch.Tensor, torch.Tensor]:
        """``joint_torque`` and ``torque_limits`` of the env's PD actuators with the action delay of the episode.

        The effective control of the last physics substep of a control step is the command ``lag`` substeps earlier (``DelayBuffer``; lag of
        the env in physics steps, ``xi``): the action of the previous control step if ``lag`` >= decimation, else the current
        one (in the first step of an episode the delay history holds the default pose target, i.e. the zero previous action). Position
        target = default + scale * action - encoder bias (the wrist joints hold their default).

        Both terms are evaluated from the end state of the control step, ``q1`` / ``qd1``: ``torque_limits`` is the clipping excess of the PD torque
        at the end state; ``joint_torque`` is the clipped torque applied during the LAST physics substep (the env reads ``qfrc_actuator``, which is
        computed before the substep integrates, i.e. at the start of the substep), reconstructed from the end state (semi-implicit Euler:
        q_pre = q1 - dt qd1; velocity: the mean acceleration of the control step from ``qd0``, the velocity at its start)."""
        use_prev = xi[..., XI_LAG] >= self.decimation  # (..., 1): the lag is shared by all joints
        a_eff = torch.where(use_prev, self.pad(a_prev), self.pad(a))
        target = self.default_q + self.act_scale_j * a_eff - xi[..., XI_BIAS] * self.act_mask
        kp, kd = self.kp_nom * xi[..., XI_KP], self.kd_nom * xi[..., XI_KD]
        clip = lambda tau: torch.maximum(torch.minimum(tau, self.effort_limit), -self.effort_limit)  # noqa: E731
        computed = kp * (target - q1) - kd * qd1
        limit_excess = (computed - clip(computed)).abs()[..., self.act_ids].sum(-1)
        applied = clip(kp * (target - (q1 - self.physics_dt * qd1)) - kd * (qd1 - (qd1 - qd0) / self.decimation))
        return (applied[..., self.act_ids] ** 2).sum(-1), limit_excess

    def weights(self, eta_rew: torch.Tensor) -> dict[str, torch.Tensor | float]:
        """Reward weights of the env; with the Stage 2 reward schedule the three reference-contact terms of Table 3 are multiplied by
        ``1 - eta_rew`` and the planned-contact term is ``1.5 * eta_rew``. ``track_body_pos_anchor`` (a body POSITION term, like the
        reference-contact positions, which the planner replaces) fades with ``1 - eta_rew`` as well; ``track_body_ori`` / ``track_body_lin_vel`` /
        ``track_body_ang_vel`` keep their weights (the paper keeps all other terms)."""
        w = dict(self.w)
        for n in ("track_contact_label", "track_contact_pos_base", "track_contact_pos_global", "track_body_pos_anchor"):
            w[n] = w[n] * (1 - eta_rew)
        w["track_planned_contact"] = 1.5 * eta_rew
        return w

    def reward(self, S, qd0, f, phase, origin, a, a_prev, a_prev2, xi, contact, lp_hist, ep0, H: int, planned=None, eta_rew=None, terms: bool = False, use_geo_contact: bool = False, fk=None):
        """Per-step reward (weights and ``dt`` applied) of the task for the transitions ending in ``S``. The batch is (H * B) laid out
        step-major (``stack(xs).flatten(0, 1)``). ``contact``: c_meas (B*H, 9) (probabilities in imagination), ``lp_hist`` (2, B, 9, 3): world
        positions (velocities with ``contact_acc`` mode vel1) of the contact links at the two states before the start (for ``contact_acc``), ``ep0`` (B,): episode step of the start
        state (``contact_acc`` needs three positions of the episode), ``qd0``: joint velocities at the start of every transition (same layout as ``S``), ``planned``: (world targets (H*B, 9, 3), mask) of the planner (``targets_to_world``; None: the
        reference targets, as the env evaluates the term). ``eta_rew`` None = the env's weights. ``fk``: ``kin.forward_all(*S)`` when the caller has it already."""
        m, P = self.motion, self.p
        w = self.w if eta_rew is None else self.weights(eta_rew)
        p, R, v, wv, q, qd = S
        full = self.kin.forward_all(*S) if fk is None else fk  # the SuperMimic tracked bodies: the contact links are the first NUM_CONTACT
        bp, bR, bv, bw = tuple(x[:, :NUM_CONTACT] for x in full)
        pel_p, pel_R, pel_v, pel_w = self.base_of((bp, bR, bv, bw))
        if use_geo_contact:  # imagination: the contact probability of the nine links computed from the geometry (instead of the world model's learned head)
            contact = self.geo.contact(bp, bR)
        rp, rR, rv, rw = (m.link_p[f][:, PELVIS] + origin, m.link_R[f][:, PELVIS], m.link_v[f][:, PELVIS], m.link_w[f][:, PELVIS])
        ref_links = m.link_p[f] + origin.unsqueeze(1)
        ids = self.act_ids
        sq = lambda std: std * std  # noqa: E731
        c = P["track_base_pos"]
        moving = (rv.norm(dim=-1) > c["moving_speed_threshold"]).float()
        t = {"track_base_pos": torch.exp(-moving * ((pel_p - rp) ** 2).sum(-1) / sq(c["sigma_pos"]) - rot_angle_sq(rR, pel_R) / sq(c["sigma_rot"]))}
        c = P["track_base_vel"]
        t["track_base_vel"] = torch.exp(-((pel_v - rv) ** 2).sum(-1) / sq(c["sigma_lin_vel"]) - ((pel_w - rw) ** 2).sum(-1) / sq(c["sigma_ang_vel"]))
        err_q = wrap_to_pi(q[..., ids] - m.q[f][..., ids])
        t["track_joint_pos"] = torch.exp(-(err_q**2).sum(-1) / sq(P["track_joint_pos"]["sigma"]))
        t["track_joint_vel"] = torch.exp(-((qd[..., ids] - m.qd[f][..., ids]) ** 2).sum(-1) / sq(P["track_joint_vel"]["sigma"]))
        t["track_contact_label"] = (contact * (2.0 * m.flag(f, phase).float() - 1.0)).mean(-1)
        exec_b = mv(yaw_mat(pel_R).transpose(-1, -2).unsqueeze(1), bp - pel_p.unsqueeze(1))
        ref_b = mv(yaw_mat(rR).transpose(-1, -2).unsqueeze(1), ref_links - rp.unsqueeze(1))
        t["track_contact_pos_base"] = torch.exp(-((exec_b - ref_b) ** 2).sum((-1, -2)) / sq(P["track_contact_pos_base"]["sigma"]))
        c = P["track_contact_pos_global"]
        moving = (rv.norm(dim=-1) > c["moving_speed_threshold"]).float()
        t["track_contact_pos_global"] = torch.exp(-moving * ((bp - ref_links) ** 2).sum((-1, -2)) / sq(c["sigma"]))
        # the SuperMimic body terms (supermimic_vibed mdp.py:108-131) over the tracked bodies; anchor = (robot root xy, reference root z), dR = yaw(R rR^T)
        tp, tR, tv, tw = full
        trk_p, trk_R, trk_v, trk_w = m.track_p[f], m.track_R[f], m.track_v[f], m.track_w[f]
        d_R, ref_bp = self.anchored_reference(pel_p, pel_R, rp, rR, f, origin)
        t["track_body_pos_anchor"] = torch.exp(-((ref_bp - tp) ** 2).sum(-1).mean(-1) / sq(P["track_body_pos_anchor"]["sigma"]))
        t["track_body_ori"] = torch.exp(-rot_angle_sq(mm(d_R.unsqueeze(1), trk_R), tR).mean(-1) / sq(P["track_body_ori"]["sigma"]))
        t["track_body_lin_vel"] = torch.exp(-((trk_v - tv) ** 2).sum(-1).mean(-1) / sq(P["track_body_lin_vel"]["sigma"]))
        t["track_body_ang_vel"] = torch.exp(-((trk_w - tw) ** 2).sum(-1).mean(-1) / sq(P["track_body_ang_vel"]["sigma"]))
        if planned is None:  # the reference targets in the (true) pelvis frame, as the env evaluates the term
            tgt_b, mask = self.targets(pel_p, pel_R, f, phase, origin)
            err = (mask * ((exec_b - tgt_b) ** 2).sum(-1)).sum(-1)
        else:  # planner targets: world points of the estimated frame they were generated in
            tgt_w, mask = planned
            err = planned_contact_error(bp, pel_R, tgt_w, mask)
        t["track_planned_contact"] = torch.exp(-err / (sq(P["track_planned_contact"]["sigma"]) * mask.sum(-1).float().clamp(min=1.0)))
        t["action_rate"] = ((a - a_prev) ** 2).sum(-1)
        t["action_rate_2nd"] = ((a - 2 * a_prev + a_prev2) ** 2).sum(-1)
        t["joint_torque"], t["torque_limits"] = self.torque_terms(q, qd, qd0, a, a_prev, xi)  # the 23 controlled joints
        excess = torch.relu(self.soft_lo - q) + torch.relu(q - self.soft_hi)
        t["dof_pos_limits"] = excess[..., ids].sum(-1).clamp(max=P["dof_pos_limits"]["max_value"])
        c = P["dof_vel_limits"]
        t["dof_vel_limits"] = torch.relu(qd.abs() - c["limit_ratio"] * self.vel_limit)[..., ids].clamp(max=c["max_value"]).sum(-1)
        # contact_acc: second finite difference of the world positions of the contact links over env steps (``mode="vel1"``: the difference of their world velocities)
        B = lp_hist.shape[1]
        if self.contact_acc_vel:
            vel = bv.view(H, B, NUM_CONTACT, 3)
            acc = (vel - torch.cat((lp_hist[1:], vel), 0)[:H]) / self.dt
        else:
            pos = bp.view(H, B, NUM_CONTACT, 3)
            prev = torch.cat((lp_hist[1:], pos), 0)[:H]
            prev2 = torch.cat((lp_hist, pos), 0)[:H]
            acc = (pos - 2 * prev + prev2) / self.dt**2
        valid = ((ep0.unsqueeze(0) + torch.arange(H, device=ep0.device).unsqueeze(1)) >= (1 if self.contact_acc_vel else 2)).float()
        t["contact_acc"] = (acc.norm(dim=-1).sum(-1) * valid).clamp(max=P["contact_acc"]["max_value"]).flatten()
        t = {k: v_ * w[k] for k, v_ in t.items()}
        r = sum((v_.detach() if k in self.stopgrad else v_) for k, v_ in t.items()) * self.dt  # the value of every term, the gradient of the others only
        return (r, t) if terms else r

    def anchored_reference(self, pel_p, pel_R, rp, rR, f, origin) -> tuple[torch.Tensor, torch.Tensor]:
        """SuperMimic's anchored reference: ``dR = yaw(R R*^T)`` (B, 3, 3) and the reference positions of the tracked bodies (B, K, 3) re-anchored to
        (robot root xy, reference root z) and rotated by ``dR`` (``rp`` / ``rR``: the reference pelvis in the world)."""
        d_R = yaw_mat(mm(pel_R, rR.transpose(-1, -2)))
        anchor = torch.cat((pel_p[..., :2], rp[..., 2:3]), -1).unsqueeze(1)
        return d_R, anchor + mv(d_R.unsqueeze(1), self.motion.track_p[f] + origin.unsqueeze(1) - rp.unsqueeze(1))

    def pelvis_height_error(self, bp, f, origin) -> torch.Tensor:
        """Distance of the pelvis from the reference pelvis for ``base_pos_too_far`` (its ``height_only`` flag: the vertical component only, else 3-D)."""
        diff = bp[:, PELVIS] - (self.motion.link_p[f][:, PELVIS] + origin)
        return diff[..., 2].abs() if self.tp["base_pos_too_far"].get("height_only", True) else diff.norm(dim=-1)

    def projected_gravity_error(self, bR, f) -> torch.Tensor:
        """Distance of the pelvis projected gravity from the reference's, as the env's ``base_pg_too_far`` (``z_only``: only the z component, the tilt test of the SuperMimic reference)."""
        diff = -bR[:, PELVIS, 2, :] + self.motion.link_R[f][:, PELVIS, 2, :]
        return diff[..., 2].abs() if self.tp["base_pg_too_far"]["z_only"] else diff.norm(dim=-1)

    def termination(self, S, f, origin, phase) -> dict[str, torch.Tensor]:
        """Hard versions of the failure terminations (margin to the threshold)."""
        bp, bR = self.kin.forward(*S)[:2]
        e_z = self.pelvis_height_error(bp, f, origin)
        e_pg = self.projected_gravity_error(bR, f)
        e_xy, e_foot = self.tracking_errors(bp, f, origin)
        d_v = self.violation_distance(S)
        return {
            "contact_violation": torch.where(self.violation_active(f, origin), VIOLATION_D - d_v, torch.full_like(d_v, -1.0)),
            "base_pos_too_far": e_z - self.tp["base_pos_too_far"]["distance_threshold"],
            "base_pg_too_far": e_pg - self.tp["base_pg_too_far"]["projected_gravity_threshold"],
            "base_xy_too_far": e_xy - self.tp["base_xy_too_far"]["distance_threshold"],
            "foot_pos_too_far": e_foot - self.tp["foot_pos_too_far"]["distance_threshold"],
        }

    def violation_distance(self, S, fk=None) -> torch.Tensor:
        """Smallest terrain distance (B,) of the ``contact_violation`` bodies (the torso) from the exact SDF of their collision primitives (``fk``: ``kin.forward_all``)."""
        full = self.kin.forward_all(*S) if fk is None else fk
        return self.violation_sdf.distance(full[0][:, VIOLATION_IDS], full[1][:, VIOLATION_IDS]).min(-1).values

    def violation_active(self, f, origin) -> torch.Tensor:
        """``contact_violation`` applies where the reference violation bodies are further than ``VIOLATION_REF_CLEARANCE`` from the terrain (B,) (as the env term)."""
        m = self.motion
        d = self.violation_sdf.distance(m.track_p[f][:, VIOLATION_IDS] + origin.unsqueeze(1), m.track_R[f][:, VIOLATION_IDS]).min(-1).values
        return d > VIOLATION_REF_CLEARANCE

    def tracking_errors(self, bp, f, origin) -> tuple[torch.Tensor, torch.Tensor]:
        """``base_xy_too_far`` / ``foot_pos_too_far`` errors: horizontal pelvis distance to the reference pelvis, largest 3-D foot distance to the reference foot."""
        ref = self.motion.link_p[f] + origin.unsqueeze(1)
        e_xy = (bp[:, PELVIS, :2] - ref[:, PELVIS, :2]).norm(dim=-1)
        e_foot = (bp[:, FEET] - ref[:, FEET]).norm(dim=-1).max(-1).values
        return e_xy, e_foot

    def link_deviation(self, S, f, origin, fk=None) -> torch.Tensor:
        """Largest height deviation (B,) of the nine contact links from the reference (the quantity of the ``link_pos_too_far`` termination).
        ``fk``: ``kin.forward_all(*S)`` (or ``kin.forward``) when the caller has it already."""
        bp = (self.kin.forward(*S) if fk is None else fk)[0][:, :NUM_CONTACT]
        return (bp[..., 2] - (self.motion.link_p[f][..., 2] + origin[..., 2:3])).abs().max(-1).values

    def reference_state(self, f, origin):
        """The reference as a simulator state (p, R, v, w, q, qd) at the global frame ``f`` (world positions: motion-local + ``origin``)."""
        m = self.motion
        return (m.link_p[f][:, PELVIS] + origin, m.link_R[f][:, PELVIS], m.link_v[f][:, PELVIS], m.link_w[f][:, PELVIS], m.q[f], m.qd[f])

    def hard_termination(self, S, f, origin, phase) -> torch.Tensor:
        return torch.stack([v > 0 for v in self.termination(S, f, origin, phase).values()], -1).any(-1)

    def soft_termination(self, S, f, origin, phase, tz: float | None = None, tt: float | None = None, fk=None) -> torch.Tensor:
        """Differentiable termination probability for imagined rollouts (the per-step hazard of every term is sigmoid((error - threshold) / tau), ``soft_tz`` for the distances, ``soft_tt`` for the tilt).
        ``fk``: ``kin.forward_all(*S)`` (or ``kin.forward``) when the caller has it already."""
        tz, tt = self.soft_tz if tz is None else tz, self.soft_tt if tt is None else tt
        fk = self.kin.forward_all(*S) if fk is None else fk
        bp, bR = fk[:2]
        e_z = self.pelvis_height_error(bp, f, origin)
        e_pg = self.projected_gravity_error(bR, f)
        alive = 1 - torch.sigmoid((e_z - self.tp["base_pos_too_far"]["distance_threshold"]) / tz)
        alive = alive * (1 - torch.sigmoid((e_pg - self.tp["base_pg_too_far"]["projected_gravity_threshold"]) / tt))
        e_xy, e_foot = self.tracking_errors(bp, f, origin)
        alive = alive * (1 - torch.sigmoid((e_xy - self.tp["base_xy_too_far"]["distance_threshold"]) / tz))
        alive = alive * (1 - torch.sigmoid((e_foot - self.tp["foot_pos_too_far"]["distance_threshold"]) / tz))
        alive = alive * (1 - torch.sigmoid((VIOLATION_D - self.violation_distance(S, fk)) / VIOLATION_TAU) * self.violation_active(f, origin).float())
        return 1 - alive

    # ---------------- critic ----------------
    def critic_features(self, S, f, phase, origin, xi_n, goal_pos, goal_yaw, fk=None) -> torch.Tensor:
        """Privileged critic input: invariant state, pelvis velocities, robot-vs-reference errors (joints, 9 contact links, the anchored position
        error of the 14 tracked bodies in the robot yaw frame (SuperMimic's critic input, upstream mdp.py:174), pelvis), the
        reference kinematics / contact flags / next-onset targets (noise-free), goal, 17 x 11 scan (true pose) and the
        sim context. ``fk``: ``kin.forward_all(*S)`` when the caller has it already."""
        m = self.motion
        p, R, v, w, q, qd = S
        full = self.kin.forward_all(*S) if fk is None else fk
        bp, bR, bv, bw = tuple(x[:, :NUM_CONTACT] for x in full)
        pel_p, pel_R, pel_v, pel_w = self.base_of((bp, bR, bv, bw))
        Ht = yaw_mat(pel_R).transpose(-1, -2)
        rp, rR, rv, rw = (m.link_p[f][:, PELVIS] + origin, m.link_R[f][:, PELVIS], m.link_v[f][:, PELVIS], m.link_w[f][:, PELVIS])
        ref_links = m.link_p[f] + origin.unsqueeze(1)
        ref_b = mv(Ht.unsqueeze(1), ref_links - pel_p.unsqueeze(1))
        act_b = mv(Ht.unsqueeze(1), bp - pel_p.unsqueeze(1))
        e_root = mv(Ht, rp - pel_p)
        e_track = mv(Ht.unsqueeze(1), self.anchored_reference(pel_p, pel_R, rp, rR, f, origin)[1] - full[0])  # SuperMimic critic input: anchored reference - body position, robot yaw frame
        rel = mat_to_6d(mm(pel_R.transpose(-1, -2), rR))
        ref_kin = torch.cat((mv(rR.transpose(-1, -2), rv), mv(rR.transpose(-1, -2), rw), m.q[f][..., self.act_ids], m.qd[f][..., self.act_ids]), -1)
        tgt, valid = self.targets(pel_p, pel_R, f, phase, origin)
        vel_b = torch.cat((mv(pel_R.transpose(-1, -2), pel_v), mv(pel_R.transpose(-1, -2), pel_w)), -1)
        return torch.cat((invariant_features(S), vel_b, m.q[f] - q, (ref_b - act_b).flatten(1), ref_b.flatten(1), e_track.flatten(1), e_root, rel, ref_kin,
                          m.flag(f, phase).float(), tgt.flatten(1), valid.float(), self.goal(pel_p, pel_R, goal_pos, goal_yaw),
                          self.critic_scan(pel_p, pel_R), xi_n), -1)  # fmt: skip

    def critic_dim(self) -> int:
        return INV_DIM + 6 + NUM_J + 27 + 27 + 3 * self.num_track + 3 + 6 + 52 + NUM_CONTACT + 27 + NUM_CONTACT + 3 + self.critic_scan_dim + self.xi_dim
