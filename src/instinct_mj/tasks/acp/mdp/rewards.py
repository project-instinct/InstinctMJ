"""Reward terms of the ACP tracking task (paper Table 3). The formulas are mirrored by the SuperMimic ACP runner."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.actuator import BuiltinPdActuator
from mjlab.managers import ManagerTermBase, SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply, quat_conjugate, quat_error_magnitude, quat_mul, wrap_to_pi, yaw_quat
from mjlab.utils.lab_api.string import resolve_matching_names_values

from instinct_mj.envs.mdp.rewards.regularizations import _unwrap_base_actuator

from .commands import AcpReferenceContacts, to_base_yaw_frame

if TYPE_CHECKING:
    from mjlab.entity import Entity
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers import RewardTermCfg


def measured_contact_flags(env: ManagerBasedRlEnv, command_name: str, force_threshold: float = 1.0) -> torch.Tensor:
    """c_meas_b, (N, 9) bool. ``env.cfg.contact_detect == "sdf"`` (default): the signed distance of the sphere approximation of the body to the terrain is below
    ``d_on_b`` (``rl/acp/sdf_contact.py``), the same function the imagination relaxes; ``"force_head"``: max over the sensor history window of |F_z| > threshold."""
    term: AcpReferenceContacts = env.command_manager.get_term(command_name)
    if getattr(env.cfg, "contact_detect", "force_head") == "sdf":
        from instinct_mj.rl.acp.sdf_contact import env_contact_distance, get_sdf_contact

        return get_sdf_contact(env).flags(env_contact_distance(env, term.contact_body_ids))
    force_z = term.contact_sensor.data.force_history[:, term.contact_sensor_ids, :, 2]  # (N, 9, H)
    return force_z.abs().amax(dim=-1) > force_threshold


def track_base_pos(
    env: ManagerBasedRlEnv,
    command_name: str,
    sigma_pos: float = 0.5,
    sigma_rot: float = 1.0,
    moving_speed_threshold: float = 0.1,
) -> torch.Tensor:
    """exp(-chi_mov |r_b - r*_b|^2 / sigma_p^2 - theta_e^2 / sigma_r^2), chi_mov = 1[|v*_b| > threshold]."""
    term: AcpReferenceContacts = env.command_manager.get_term(command_name)
    robot: Entity = term.robot
    moving = (term.ref_base_lin_vel_w.norm(dim=-1) > moving_speed_threshold).float()
    pos_err = (term.base_pos_w - term.ref_base_pos_w).square().sum(-1)
    rot_err = quat_error_magnitude(term.base_quat_w, term.ref_base_quat_w)
    return torch.exp(-moving * pos_err / sigma_pos**2 - rot_err.square() / sigma_rot**2)


def track_base_vel(
    env: ManagerBasedRlEnv,
    command_name: str,
    sigma_lin_vel: float = 1.0,
    sigma_ang_vel: float = 3.0,
) -> torch.Tensor:
    """exp(-|v_b - v*_b|^2 / sigma_v^2 - |w_b - w*_b|^2 / sigma_w^2), world frame."""
    term: AcpReferenceContacts = env.command_manager.get_term(command_name)
    robot: Entity = term.robot
    lin_err = (term.base_lin_vel_w - term.ref_base_lin_vel_w).square().sum(-1)
    ang_err = (term.base_ang_vel_w - term.ref_base_ang_vel_w).square().sum(-1)
    return torch.exp(-lin_err / sigma_lin_vel**2 - ang_err / sigma_ang_vel**2)


def track_joint_pos(
    env: ManagerBasedRlEnv,
    command_name: str,
    sigma: float = 2.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """exp(-|wrap_pi(q - q*)|^2 / sigma_q^2) over the action joints."""
    term: AcpReferenceContacts = env.command_manager.get_term(command_name)
    ref_q = term.motion_reference.reference_frame.joint_pos[:, 0, asset_cfg.joint_ids]
    err = wrap_to_pi(term.robot.data.joint_pos[:, asset_cfg.joint_ids] - ref_q)
    return torch.exp(-err.square().sum(-1) / sigma**2)


def track_joint_vel(
    env: ManagerBasedRlEnv,
    command_name: str,
    sigma: float = 10.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """exp(-|qd - qd*|^2 / sigma_qd^2) over the action joints."""
    term: AcpReferenceContacts = env.command_manager.get_term(command_name)
    ref_qd = term.motion_reference.reference_frame.joint_vel[:, 0, asset_cfg.joint_ids]
    err = term.robot.data.joint_vel[:, asset_cfg.joint_ids] - ref_qd
    return torch.exp(-err.square().sum(-1) / sigma**2)


def track_contact_label(env: ManagerBasedRlEnv, command_name: str, force_threshold: float = 1.0) -> torch.Tensor:
    """(1/9) sum_b c_meas_b (2 c_ref_b - 1)."""
    term: AcpReferenceContacts = env.command_manager.get_term(command_name)
    measured = measured_contact_flags(env, command_name, force_threshold).float()
    return (measured * (2.0 * term.ref_contact_flag.float() - 1.0)).mean(-1)


def track_contact_pos_base(env: ManagerBasedRlEnv, command_name: str, sigma: float = 1.0) -> torch.Tensor:
    """exp(-sum_b |p_exec,B_b - p_ref,B_b|^2 / sigma_c^2); actual / reference positions in their own base-yaw frames."""
    term: AcpReferenceContacts = env.command_manager.get_term(command_name)
    robot: Entity = term.robot
    exec_b = to_base_yaw_frame(
        robot.data.body_link_pos_w[:, term.contact_body_ids], term.base_pos_w, term.base_quat_w
    )
    ref_b = to_base_yaw_frame(term.ref_link_pos_w, term.ref_base_pos_w, term.ref_base_quat_w)
    return torch.exp(-(exec_b - ref_b).square().sum((-1, -2)) / sigma**2)


def track_contact_pos_global(
    env: ManagerBasedRlEnv,
    command_name: str,
    sigma: float = 1.0,
    moving_speed_threshold: float = 0.1,
) -> torch.Tensor:
    """exp(-chi_mov sum_b |p_exec,W_b - p_ref,W_b|^2 / sigma_c^2)."""
    term: AcpReferenceContacts = env.command_manager.get_term(command_name)
    robot: Entity = term.robot
    moving = (term.ref_base_lin_vel_w.norm(dim=-1) > moving_speed_threshold).float()
    err = (robot.data.body_link_pos_w[:, term.contact_body_ids] - term.ref_link_pos_w).square().sum((-1, -2))
    return torch.exp(-moving * err / sigma**2)


def _track_anchor(term: AcpReferenceContacts):
    """(robot body states of the tracked bodies, yaw rotation dR = yaw(R_root R*_root^T) (N, 4), reference positions re-anchored to (robot root xy, reference root z))."""
    robot: Entity = term.robot
    ids = term.track_body_ids
    d_yaw = yaw_quat(quat_mul(term.base_quat_w, quat_conjugate(term.ref_base_quat_w)))
    anchor = torch.cat((term.base_pos_w[:, :2], term.ref_base_pos_w[:, 2:3]), -1)
    ref_pos = term.ref_track("link_pos_w")
    rel = ref_pos - term.ref_base_pos_w.unsqueeze(1)
    ref_bp = anchor.unsqueeze(1) + quat_apply(d_yaw.unsqueeze(1).expand(-1, rel.shape[1], -1), rel)
    return robot, ids, d_yaw, ref_bp


def track_body_pos_anchor(env: ManagerBasedRlEnv, command_name: str, sigma: float = 0.3) -> torch.Tensor:
    """SuperMimic / BeyondMimic body-position reward (SuperMimic App. A Table A.2: sigma 0.3 m, weight 1; supermimic_vibed mdp.py:113-126) over the SuperMimic tracked bodies (``TRACKED_LINKS``: the nine
    contact links + torso, hip-roll and shoulder-roll links; user decision 2026-11-07): the reference body positions are re-anchored to the robot, ``anchor = (robot root xy, reference root z)``,
    ``dR = yaw(R_root R*_root^T)``, ``p*_b = anchor + dR (p_ref,b - p_ref,root)``, and ``r = exp(-mean_b |p*_b - p_b|^2 / sigma^2)``: the shape of the motion (relative to the robot's own
    horizontal position and heading), not the drift of the root."""
    term: AcpReferenceContacts = env.command_manager.get_term(command_name)
    robot, ids, _, ref_bp = _track_anchor(term)
    err = (ref_bp - robot.data.body_link_pos_w[:, ids]).square().sum(-1).mean(-1)
    return torch.exp(-err / sigma**2)


def track_body_ori(env: ManagerBasedRlEnv, command_name: str, sigma: float = 0.4) -> torch.Tensor:
    """SuperMimic body-orientation reward (App. A Table A.2: sigma 0.4, weight 1; supermimic_vibed mdp.py:108-131; user decision 2026-11-07): ``exp(-mean_b angle(dR R*_b, R_b)^2 / sigma^2)`` over the
    SuperMimic tracked bodies, ``dR = yaw(R_root R*_root^T)`` as in ``track_body_pos_anchor``."""
    term: AcpReferenceContacts = env.command_manager.get_term(command_name)
    robot, ids, d_yaw, _ = _track_anchor(term)
    ref_q = quat_mul(d_yaw.unsqueeze(1).expand(-1, len(ids), -1), term.ref_track("link_quat_w"))
    err = quat_error_magnitude(ref_q, robot.data.body_link_quat_w[:, ids]).square().mean(-1)
    return torch.exp(-err / sigma**2)


def track_body_lin_vel(env: ManagerBasedRlEnv, command_name: str, sigma: float = 1.0) -> torch.Tensor:
    """SuperMimic body linear-velocity reward (sigma 1.0, weight 1; supermimic_vibed mdp.py:108-131; user decision 2026-11-07): ``exp(-mean_b |v*_b - v_b|^2 / sigma^2)`` over the SuperMimic tracked
    bodies, world frame without the heading correction (as upstream)."""
    term: AcpReferenceContacts = env.command_manager.get_term(command_name)
    err = (term.ref_track("link_lin_vel_w") - term.robot.data.body_link_lin_vel_w[:, term.track_body_ids]).square().sum(-1).mean(-1)
    return torch.exp(-err / sigma**2)


def track_body_ang_vel(env: ManagerBasedRlEnv, command_name: str, sigma: float = 3.14) -> torch.Tensor:
    """SuperMimic body angular-velocity reward (sigma 3.14, weight 1; supermimic_vibed mdp.py:108-131; user decision 2026-11-07): ``exp(-mean_b |w*_b - w_b|^2 / sigma^2)`` over the SuperMimic tracked
    bodies, world frame."""
    term: AcpReferenceContacts = env.command_manager.get_term(command_name)
    err = (term.ref_track("link_ang_vel_w") - term.robot.data.body_link_ang_vel_w[:, term.track_body_ids]).square().sum(-1).mean(-1)
    return torch.exp(-err / sigma**2)


def track_planned_contact(env: ManagerBasedRlEnv, command_name: str, sigma: float = 1.0) -> torch.Tensor:
    """exp(-sum_b m_b |p_exec,B_b - p^_b|^2 / (sigma_c^2 max(1, sum_b m_b))), m_b = valid contact target mask.

    Stage 1 evaluates it against the reference contact targets (``AcpReferenceContacts``); Stage 2/3 weight it (1.5)."""
    term: AcpReferenceContacts = env.command_manager.get_term(command_name)
    robot: Entity = term.robot
    target_b, mask = term.targets(estimated=False)
    exec_b = to_base_yaw_frame(
        robot.data.body_link_pos_w[:, term.contact_body_ids], term.base_pos_w, term.base_quat_w
    )
    err = (mask * (exec_b - target_b).square().sum(-1)).sum(-1)
    return torch.exp(-err / (sigma**2 * mask.sum(-1).clamp(min=1.0)))


def joint_pos_limits_clipped(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_value: float = 30.0,
) -> torch.Tensor:
    """min(max_value, sum_j ([q_min - q]_+ + [q - q_max]_+)) with the soft joint limits."""
    robot: Entity = env.scene[asset_cfg.name]
    limits = robot.data.soft_joint_pos_limits[:, asset_cfg.joint_ids]
    q = robot.data.joint_pos[:, asset_cfg.joint_ids]
    excess = (limits[..., 0] - q).clip(min=0.0) + (q - limits[..., 1]).clip(min=0.0)
    return excess.sum(-1).clamp(max=max_value)


class dof_vel_limits(ManagerTermBase):
    """sum_j min(1, [|qd_j| - ratio qd_max_j]_+) over the joints of ``asset_cfg`` with the velocity limits of the G1 URDF
    (``velocity_limits`` regex dict)."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        robot: Entity = env.scene[cfg.params.get("asset_cfg", SceneEntityCfg("robot")).name]
        _, _, values = resolve_matching_names_values(cfg.params["velocity_limits"], list(robot.joint_names))
        self.velocity_limits = torch.tensor(values, dtype=torch.float, device=env.device)

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        velocity_limits: dict[str, float],
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        limit_ratio: float = 0.9,
        max_value: float = 1.0,
    ) -> torch.Tensor:
        robot: Entity = env.scene[asset_cfg.name]
        excess = robot.data.joint_vel[:, asset_cfg.joint_ids].abs() - limit_ratio * self.velocity_limits[asset_cfg.joint_ids]
        return excess.clip(min=0.0).clamp(max=max_value).sum(-1)


def torque_limit_excess(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """sum_j |tau_computed_j - tau_applied_j|: the PD torque before / after the actuator effort clipping.

    tau_computed = kp (target - q) + kd (0 - qd) from the controls and the gains of the model, tau_applied is the
    computed torque clipped to the joint effort limit (``jnt_actfrcrange``, the sum-clamp of the built-in PD actuators).
    Both are evaluated at the same state: the state after the last physics step, as ``qfrc_actuator`` lags it by one
    physics step (its difference to the computed torque would otherwise add the one-step change of the PD error)."""
    robot: Entity = env.scene[asset_cfg.name]
    model = env.sim.model
    ctrl = env.sim.data.ctrl
    computed = torch.zeros_like(robot.data.qfrc_actuator)
    for actuator in robot.actuators:
        base_actuator = _unwrap_base_actuator(actuator)
        if not isinstance(base_actuator, BuiltinPdActuator):
            raise TypeError(f"torque_limit_excess supports BuiltinPdActuator only, got {type(base_actuator).__name__}.")
        n = base_actuator.num_targets
        ids = actuator.global_ctrl_ids
        pos_ids, vel_ids = ids[:n], ids[n:]
        kp = model.actuator_gainprm[:, pos_ids, 0]
        kd = model.actuator_gainprm[:, vel_ids, 0]
        target_ids = actuator.target_ids
        q = robot.data.joint_pos[:, target_ids]
        qd = robot.data.joint_vel[:, target_ids]
        computed[:, target_ids] = kp * (ctrl[:, pos_ids] - q) + kd * (ctrl[:, vel_ids] - qd)
    effort_limit = model.jnt_actfrcrange[:, robot.indexing.joint_ids, 1]
    applied = torch.maximum(torch.minimum(computed, effort_limit), -effort_limit)
    return (computed - applied).abs()[:, asset_cfg.joint_ids].sum(-1)


class contact_acc(ManagerTermBase):
    """min(max_value, sum_b |p''_exec,W_b|) (Table 3): finite-difference acceleration of the 9 contact links over env steps. ``mode="pos2"``: the second difference of the link
    positions; ``mode="vel1"``: the difference of the link-origin world velocities (as the authors' framework, ``body_lin_acc_w``)."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        self.mode: str = cfg.params.get("mode", "pos2")
        assert self.mode in ("pos2", "vel1")
        term: AcpReferenceContacts = env.command_manager.get_term(cfg.params["command_name"])
        self.robot: Entity = term.robot
        self.contact_body_ids = term.contact_body_ids
        shape = (env.num_envs, len(self.contact_body_ids), 3)
        self.prev_pos = torch.zeros(shape, device=env.device)
        self.prev_prev_pos = torch.zeros(shape, device=env.device)
        self.prev_vel = torch.zeros(shape, device=env.device)
        self.num_samples = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        self.num_samples[slice(None) if env_ids is None else env_ids] = 0

    def __call__(self, env: ManagerBasedRlEnv, command_name: str, max_value: float = 1e4, mode: str = "pos2") -> torch.Tensor:
        if self.mode == "vel1":
            vel = self.robot.data.body_link_lin_vel_w[:, self.contact_body_ids]
            acc = (vel - self.prev_vel) / env.step_dt
            self.prev_vel[:] = vel
            self.num_samples += 1
            # the difference needs two velocities of the current episode (a select, not a product: the stored velocity of a blown-up episode is non-finite)
            acc = torch.where((self.num_samples >= 2).view(-1, 1, 1), acc, 0.0)
            return acc.norm(dim=-1).sum(-1).clamp(max=max_value)
        pos = self.robot.data.body_link_pos_w[:, self.contact_body_ids]
        acc = (pos - 2.0 * self.prev_pos + self.prev_prev_pos) / env.step_dt**2
        self.prev_prev_pos[:] = self.prev_pos
        self.prev_pos[:] = pos
        self.num_samples += 1
        # the second difference needs three positions of the current episode (a select, as above)
        acc = torch.where((self.num_samples >= 3).view(-1, 1, 1), acc, 0.0)
        return acc.norm(dim=-1).sum(-1).clamp(max=max_value)
