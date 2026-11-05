"""Analytic MDP heads of the G1 perceptive shadowing task shared by the simulator labels and the world model.

Adapted from the SuperMimic reference implementation (MIT License), Chong Zhang / ETH Zurich.

State tuple ``S = (p, R, v, w, q, qd)``: ``torso_link`` (the root / free-joint body of the G1 popsicle MJCF) position,
rotation matrix, world linear velocity of the link-frame origin, world angular velocity, joint positions and
velocities (simulator joint order). Positions are world positions (``motion-local + origin``); reference data come from
:class:`MotionLib` and are in the motion-local frame, so they are shifted by the per-sample ``origin``.

Everything here mirrors the managers of ``instinct_mj.tasks.shadowing.perceptive.perceptive_env_cfg``:
reward terms (``instinct_mj.envs.mdp.rewards``), terminations (``instinct_mj.envs.mdp.terminations``) and the
``policy`` observation group (without ``depth_image``).
"""

from __future__ import annotations

import torch

from .kinematics import G1Kinematics
from .motion_lib import MotionLib
from .rotations import mat_to_6d, mat_to_quat, mm, mv, quat_to_mat, rot_angle_sq, yaw_mat
from .terrain import HeightScanner

NUM_J = 29
STATE_DIM = 3 + 4 + 3 + 3 + NUM_J + NUM_J
# sensor noise of the task's ``policy`` group (uniform half-widths) / goal noise (position_ref, rotation_ref)
PROPRIO_NOISE = (0.05, 0.2, 0.01, 0.5, 0.0)
GOAL_POS_NOISE, GOAL_ROT_NOISE = 0.25, 0.05
SCAN_NOISE = 0.02
INV_DIM = 6 + 3 + 3 + NUM_J + NUM_J
# xi layout: default joint offsets, torso com offset, kp / kd scales, mass scales of the randomized bodies, then the
# friction (3 components) of every robot collision geom (the task randomizes them independently per geom)
XI_DQ, XI_COM, XI_KP, XI_KD, XI_MASS = slice(0, 29), slice(29, 32), slice(32, 61), slice(61, 90), slice(90, 101)
XI_FRICTION_START = 101


def flatten_state(S) -> torch.Tensor:
    p, R, v, w, q, qd = S
    return torch.cat((p, mat_to_quat(R), v, w, q, qd), -1)


def unflatten_state(x: torch.Tensor):
    nj = (x.shape[-1] - 13) // 2  # number of joints of the robot (29 for the G1 of the shadowing tasks)
    return x[..., 0:3], quat_to_mat(x[..., 3:7]), x[..., 7:10], x[..., 10:13], x[..., 13 : 13 + nj], x[..., 13 + nj : 13 + 2 * nj]


DEPTH_SHAPE = (1, 18, 32)  # processed depth image of the task's policy group (C, H, W)
DEPTH_DIM = 18 * 32


def invariant_features(S) -> torch.Tensor:
    """Heading- and position-invariant state features: tilt (6D), body-frame velocities, joints.

    (The reference version also contains the root height; on terrain the height scan carries that information.)
    """
    p, R, v, w, q, qd = S
    H = yaw_mat(R)
    Rt = R.transpose(-1, -2)
    return torch.cat(
        (mat_to_6d(mm(H.transpose(-1, -2), R)), mv(Rt, v), mv(Rt, w), q, qd),
        -1,
    )


class MDP:
    def __init__(
        self,
        kin: G1Kinematics,
        motion: MotionLib,
        scanner: HeightScanner,
        env,
        dt: float,
        device: str,
    ):
        """All task constants are read from the live env (reward / termination params, actuators, action term)."""
        self.kin, self.motion, self.scanner, self.dt, self.device = kin, motion, scanner, dt, device
        robot = env.scene["robot"]
        rm = env.reward_manager
        self.w = {n: rm.get_term_cfg(n).weight for n in rm.active_terms["rewards"]}
        self.p = {n: rm.get_term_cfg(n).params for n in rm.active_terms["rewards"]}
        tm = env.termination_manager
        self.tp = {n: tm.get_term_cfg(n).params for n in ("base_pos_too_far", "base_pg_too_far", "link_pos_too_far")}
        # nominal joint data (xi carries the per-env deviations)
        self.default_q = robot.data.default_joint_pos_nominal.to(device)
        limits = robot.data.soft_joint_pos_limits[0]
        self.soft_lo, self.soft_hi = limits[:, 0].to(device), limits[:, 1].to(device)
        action_term = env.action_manager.get_term("joint_pos")
        self.act_scale = action_term.scale[0].to(device)
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
        self.torque_joint_ids = list(rm.get_term_cfg("applied_torque_limits_by_ratio").params["asset_cfg"].joint_ids)
        self.contact_body_ids = list(rm.get_term_cfg("undesired_contacts").params["asset_cfg"].body_ids)
        link_names = list(env.scene["motion_reference"].cfg.link_of_interests)
        term_links = tm.get_term_cfg("link_pos_too_far").params["reference_cfg"].body_names
        self.term_link_ids = [link_names.index(n) for n in term_links]
        self.proprio_noise = torch.cat([torch.full((n,), s) for n, s in zip((3, 3, NUM_J, NUM_J, NUM_J), PROPRIO_NOISE)]).to(device)
        goal_dim = motion.goal_frames * (2 * NUM_J + 3 + 6)
        self.goal_noise = torch.zeros(goal_dim, device=device)
        g = motion.goal_frames
        self.goal_noise[2 * NUM_J * g : (2 * NUM_J + 3) * g] = GOAL_POS_NOISE
        self.goal_noise[(2 * NUM_J + 3) * g :] = GOAL_ROT_NOISE
        self.goal_dim = goal_dim
        geom_ids = robot.indexing.geom_ids.cpu().numpy()
        self.collision_geom_ids = torch.as_tensor(geom_ids[model.geom_contype[geom_ids] != 0], device=device).long()
        self.xi_dim = XI_FRICTION_START + 3 * len(self.collision_geom_ids)
        self.xi_mean = torch.zeros(self.xi_dim, device=device)
        self.xi_scale = torch.ones(self.xi_dim, device=device)

    # ---------------- sim context ----------------
    def set_xi_stats(self, table: torch.Tensor) -> None:
        """Fixed normalization of the (episode-constant) per-env context from the DR sample."""
        floor = torch.cat((torch.full((29,), 1e-3), torch.full((3,), 1e-2), torch.full((58,), 0.02), torch.full((11,), 0.02),
                           torch.full((self.xi_dim - XI_FRICTION_START,), 0.05)))  # fmt: skip
        self.xi_mean = table.mean(0)
        self.xi_scale = torch.maximum(table.std(0, correction=0), floor.to(table.device))

    def normalize_xi(self, xi: torch.Tensor) -> torch.Tensor:
        return (xi - self.xi_mean) / self.xi_scale

    # ---------------- observations (policy group of the task, without depth) ----------------
    def obs(self, S, a_prev, xi) -> torch.Tensor:
        """Noise-free proprioceptive observation: projected gravity, base ang vel (body), joint pos/vel (rel), last action."""
        p, R, v, w, q, qd = S
        pg = -R[..., 2, :]  # R^T (0, 0, -1)
        w_b = mv(R.transpose(-1, -2), w)
        return torch.cat((pg, w_b, q - (self.default_q + xi[..., XI_DQ]), qd, a_prev), -1)

    def add_noise(self, o: torch.Tensor) -> torch.Tensor:
        return o + (2 * torch.rand_like(o) - 1) * self.proprio_noise

    def goal(self, S, f, phase, noisy: bool = True) -> torch.Tensor:
        """The 10 future reference frames of the task (joint_pos_ref, joint_vel_ref, position_ref, rotation_ref)."""
        g = torch.cat((self.motion.goal_static(f, phase, self.default_q), self.motion.goal_rotation(f, phase, S[1])), -1)
        return g + (2 * torch.rand_like(g) - 1) * self.goal_noise if noisy else g

    def policy_obs(self, o_hist: torch.Tensor, goal: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        """Flat ``policy`` observation group of the task, in the env's term order and flattening:
        joint_pos_ref, joint_vel_ref, position_ref, rotation_ref (= ``goal``), depth_image, then the 8-step (oldest first,
        time-major) histories of projected_gravity, base_ang_vel, joint_pos, joint_vel, last_action (from ``o_hist``)."""
        hist = [o_hist[..., a:b].flatten(1) for a, b in ((0, 3), (3, 6), (6, 35), (35, 64), (64, 93))]
        return torch.cat((goal, depth.flatten(1), *hist), -1)

    def scan(self, S, noisy: bool = True) -> torch.Tensor:
        s = self.scanner.height_scan(S[0], S[1])
        return s + (2 * torch.rand_like(s) - 1) * SCAN_NOISE if noisy else s

    # ---------------- reward / termination ----------------
    def _torque(self, q, qd, a, xi) -> torch.Tensor:
        """Applied PD torque (sum-clamped to the effort limit) as MuJoCo's position + velocity actuators produce it."""
        target = self.default_q + xi[..., XI_DQ] + self.act_scale * a
        tau = self.kp_nom * xi[..., XI_KP] * (target - q) - self.kd_nom * xi[..., XI_KD] * qd
        return torch.maximum(torch.minimum(tau, self.effort_limit), -self.effort_limit)

    def reward(self, S, f, origin, a, a_prev, xi, contact_prob, terms: bool = False):
        """Per-step reward (weights and ``dt`` applied) of the task for the transition ending in ``S``."""
        m, w, P = self.motion, self.w, self.p
        p, R, v, wv, q, qd = S
        bp, bR, bv, bw = self.kin.forward(*S)
        rp, rR = m.root_p[f] + origin, m.root_R[f]
        dR = yaw_mat(mm(R, rR.transpose(-1, -2)))
        anchor = torch.cat((p[..., :2], rp[..., 2:3]), -1)
        ref_bp = anchor.unsqueeze(-2) + mv(dR.unsqueeze(-3), m.link_p[f] - m.root_p[f].unsqueeze(-2))
        ref_bR = mm(dR.unsqueeze(-3), m.link_R[f])
        e_pos = ((ref_bp - bp) ** 2).sum(-1).mean(-1)
        e_rot = rot_angle_sq(ref_bR, bR).mean(-1)
        e_lv = ((m.link_v[f] - bv) ** 2).sum(-1).mean(-1)
        e_av = ((m.link_w[f] - bw) ** 2).sum(-1).mean(-1)
        sq = lambda std: std * std  # noqa: E731
        t = {
            "base_position_imitation_gauss": torch.exp(-((p - rp) ** 2).sum(-1) / sq(P["base_position_imitation_gauss"]["std"])),
            "base_rot_imitation_gauss": torch.exp(-rot_angle_sq(rR, R) / sq(P["base_rot_imitation_gauss"]["std"])),
            "link_pos_imitation_gauss": torch.exp(-e_pos / sq(P["link_pos_imitation_gauss"]["std"])),
            "link_rot_imitation_gauss": torch.exp(-e_rot / sq(P["link_rot_imitation_gauss"]["std"])),
            "link_lin_vel_imitation_gauss": torch.exp(-e_lv / sq(P["link_lin_vel_imitation_gauss"]["std"])),
            "link_ang_vel_imitation_gauss": torch.exp(-e_av / sq(P["link_ang_vel_imitation_gauss"]["std"])),
            "action_rate_l2": ((a - a_prev) ** 2).sum(-1),
            "joint_limit": (torch.relu(self.soft_lo - q) + torch.relu(q - self.soft_hi)).sum(-1),
            "undesired_contacts": contact_prob.sum(-1),
        }
        tau = self._torque(q, qd, a, xi)[..., self.torque_joint_ids]
        limit = self.effort_limit[self.torque_joint_ids] * P["applied_torque_limits_by_ratio"].get("limit_ratio", 0.8)
        t["applied_torque_limits_by_ratio"] = (torch.relu(tau.abs() - limit) ** 2).sum(-1)
        t = {k: v_ * w[k] for k, v_ in t.items()}
        r = sum(t.values()) * self.dt
        return (r, t) if terms else r

    def termination(self, S, f, origin) -> dict[str, torch.Tensor]:
        """Hard versions of base_pos_too_far / base_pg_too_far / link_pos_too_far (the margin to the threshold)."""
        m = self.motion
        p, R = S[0], S[1]
        e_z = (p[..., 2] - (m.root_p[f][..., 2] + origin[..., 2])).abs()
        e_pg = ((-R[..., 2, :]) - (-m.root_R[f][..., 2, :])).norm(dim=-1)
        bp = self.kin.forward(*S)[0]
        e_link = (bp[..., self.term_link_ids, 2] - (m.link_p[f][..., self.term_link_ids, 2] + origin[..., 2:3])).abs()
        return {
            "base_pos_too_far": e_z - self.tp["base_pos_too_far"]["distance_threshold"],
            "base_pg_too_far": e_pg - self.tp["base_pg_too_far"]["projected_gravity_threshold"],
            "link_pos_too_far": e_link.max(-1).values - self.tp["link_pos_too_far"]["distance_threshold"],
        }

    def hard_termination(self, S, f, origin) -> torch.Tensor:
        return torch.stack([v > 0 for v in self.termination(S, f, origin).values()], -1).any(-1)

    def soft_termination(self, S, f, origin, tz: float = 0.02, tt: float = 0.04) -> torch.Tensor:
        """Differentiable termination probability for imagined rollouts."""
        m = self.motion
        p, R = S[0], S[1]
        e_z = (p[..., 2] - (m.root_p[f][..., 2] + origin[..., 2])).abs()
        e_pg = ((-R[..., 2, :]) - (-m.root_R[f][..., 2, :])).norm(dim=-1)
        bp = self.kin.forward(*S)[0]
        e_link = (bp[..., self.term_link_ids, 2] - (m.link_p[f][..., self.term_link_ids, 2] + origin[..., 2:3])).abs()
        alive = (1 - torch.sigmoid((e_z - self.tp["base_pos_too_far"]["distance_threshold"]) / tz))
        alive = alive * (1 - torch.sigmoid((e_pg - self.tp["base_pg_too_far"]["projected_gravity_threshold"]) / tt))
        alive = alive * (1 - torch.sigmoid((e_link - self.tp["link_pos_too_far"]["distance_threshold"]) / tz)).prod(-1)
        return 1 - alive

    def critic_features(self, S, f, origin, xi_n, goal, scan) -> torch.Tensor:
        """Privileged critic input: invariant state, robot-vs-reference errors, sim context, goal and terrain scan."""
        m = self.motion
        p, R, v, w, q, qd = S
        Ht = yaw_mat(R).transpose(-1, -2)
        rp, rR = m.root_p[f] + origin, m.root_R[f]
        bp = self.kin.forward(*S)[0]
        dR = yaw_mat(mm(R, rR.transpose(-1, -2)))
        anchor = torch.cat((p[..., :2], rp[..., 2:3]), -1)
        ref_bp = anchor.unsqueeze(-2) + mv(dR.unsqueeze(-3), m.link_p[f] - m.root_p[f].unsqueeze(-2))
        e_body = mv(Ht.unsqueeze(-3), ref_bp - bp).flatten(-2)
        e_root = mv(Ht, rp - p)
        rel = mat_to_6d(mm(R.transpose(-1, -2), rR))
        return torch.cat((invariant_features(S), m.q[f] - q, e_body, e_root, rel, xi_n, goal, scan), -1)

    def critic_dim(self) -> int:
        return INV_DIM + NUM_J + 14 * 3 + 3 + 6 + self.xi_dim + self.goal_dim + self.scanner.num_rays
