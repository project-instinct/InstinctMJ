"""Goal-reaching part of the ACP Stage 3 task (paper "Three-Stage Training Pipeline", Fig. 3, Table 3 Stage-3 column, Supplementary
"Terrains and Curriculum"): the mixed reference / goal environment type, the goal command on the procedural terrains, the terrain
curriculum, the reset events, the goal rewards and the terminations of the goal-reaching environments.

A fixed fraction of the envs follows a reference motion on the reference-anchored terrains (the Stage 2 setup, the reference terms are
masked to these envs); the others reach a goal on the procedural terrains of ``instinct_mj.tasks.acp.goal_terrains`` (the goal terms are
masked to these envs). The type of an env never changes; a goal env keeps its terrain type and moves along the 10 difficulty levels of the
curriculum. The reference motion machinery keeps running for the goal envs (it only supplies the unused reference of the env); every
reference reward / termination is masked off for them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.managers import ManagerTermBase, SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse, quat_from_euler_xyz, wrap_to_pi

from instinct_mj.envs.mdp.events.motion_reference import reset_robot_state_by_reference
from instinct_mj.envs.mdp.terminations.general import dataset_exhausted
from instinct_mj.rl.acp.planner import CONTACT_LINKS
from instinct_mj.tasks.acp.goal_terrains import BASE_STAND_HEIGHT, NUM_LEVELS, GoalArena, GoalArenaTables

from . import rewards as acp_rewards
from . import terminations as acp_terminations
from .commands import AcpGoalCommand, AcpGoalCommandCfg, AcpReferenceContacts, yaw_of_quat

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers import CurriculumTermCfg, RewardTermCfg

GOAL_COMMAND = "acp_goal"
FEET = tuple(CONTACT_LINKS.index(name) for name in ("left_ankle_roll_link", "right_ankle_roll_link"))
"""Indices of the feet in the contact links (feet without contact count in ``standatgoal``)."""


# ----------------------------------------------------------------------------------------------------------------------------------
# command
# ----------------------------------------------------------------------------------------------------------------------------------
@dataclass(kw_only=True)
class AcpStage3GoalCommandCfg(AcpGoalCommandCfg):
    """Goal command of the mixed Stage 3 env: reference lookahead goals on the reference envs, terrain goals on the goal envs."""

    goal_env_fraction: float = 0.5
    """ACP-unspecified: fraction of the envs that reach goals on the procedural terrains (spread evenly over the env ids)."""

    goal_min_distance: float = 0.5
    """Goal scaling radius r of the goal-reaching envs (Table S4: 0.5 m; ``min_distance`` = 0.3 m for the reference envs)."""

    goal_episode_s: float = 15.0
    """Maximum goal-reaching episode duration (Table S4)."""

    reset_yaw_range: float = 0.4
    """Reset yaw of the goal-reaching envs U(-0.4, 0.4) rad around the travel direction (Table S2)."""

    reset_height_range: tuple[float, float] = (0.0, 0.05)
    """Reset base height + U(0, 0.05) m (Table S2)."""

    reset_joint_range: float = 0.2
    """Reset joint positions U(-0.2, 0.2) rad around the default pose (Table S2)."""

    lying_height: float = 0.3
    """ACP-unspecified: pelvis height above the surface at which the get-up envs are dropped lying on their back / front."""

    def build(self, env: ManagerBasedRlEnv) -> AcpStage3GoalCommand:
        return AcpStage3GoalCommand(self, env)


class AcpStage3GoalCommand(AcpGoalCommand):
    """Goal command g = (d_xy / max(|d_xy|, r), wrap(psi_goal - psi)) with r = 0.3 m on the reference envs and 0.5 m on the goal envs.

    Goal envs: ``terrain_type`` (fixed) and ``level`` (curriculum) select the tile of the arena; the reset event samples the spawn pose
    and the goal (position, heading, target base height = surface + standing height) from the patches of the tile
    (:meth:`sample_reset`). ``target_pos_w[:, 2]`` is the target base height of the goal envs.
    """

    cfg: AcpStage3GoalCommandCfg

    def __init__(self, cfg: AcpStage3GoalCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self.arena: GoalArena = env.cfg.scene.arena
        self.tables: GoalArenaTables = self.arena.tables(self.device)
        n = self.num_envs
        idx = torch.arange(n, device=self.device)
        # goal envs evenly spread over the ids (the world-model hold-out envs are the last ones and contain both types)
        self.is_goal = torch.floor((idx + 1) * cfg.goal_env_fraction) > torch.floor(idx * cfg.goal_env_fraction)
        rank = torch.cumsum(self.is_goal.long(), 0) - 1
        self.terrain_type = torch.where(self.is_goal, rank % self.arena.num_types, torch.zeros_like(rank))
        """Terrain type of every goal env (balanced over the types, constant)."""
        self.level = torch.zeros(n, dtype=torch.long, device=self.device)
        """Curriculum level (0 .. 9) of every goal env."""
        self.success_ema = torch.zeros(n, device=self.device)
        """Exponentially smoothed per-env success estimate s of the curriculum."""
        self.min_xy_ok = torch.full((n,), float("inf"), device=self.device)
        """Smallest horizontal goal distance of the episode over the steps with |dz| <= 0.1 m (inf if none)."""
        self.metrics["goal_min_xy_ok"] = torch.zeros(n, device=self.device)

    # -- env type / tile --
    @property
    def ref_mask(self) -> torch.Tensor:
        return (~self.is_goal).float()

    @property
    def tile_id(self) -> torch.Tensor:
        """Arena tile (terrain type * levels + level) of every env (meaningful for the goal envs)."""
        return self.tables.tile_id(self.terrain_type, self.level)

    @property
    def radius(self) -> torch.Tensor:
        return torch.where(self.is_goal, self.cfg.goal_min_distance, self.cfg.min_distance)

    def flag(self, name: str) -> torch.Tensor:
        """Per-env tile flag (``foot``, ``stumble``, ``slip``, ``getup``, ``lying``) of the goal envs, False for the reference envs."""
        return getattr(self.tables, name)[self.tile_id] & self.is_goal

    # -- goal geometry in the world frame --
    def goal_geometry(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Horizontal distance d_g (N,), unit direction (N, 2), vertical offset dz = z_goal - z_base (N,) and heading error (N,) of the
        pelvis (base) to the goal."""
        d = self.target_pos_w[:, :2] - self.base_pos_w[:, :2]
        dist = d.norm(dim=-1)
        dz = self.target_pos_w[:, 2] - self.base_pos_w[:, 2]
        dpsi = wrap_to_pi(self.target_yaw - yaw_of_quat(self.base_quat_w))
        return dist, d / dist.clamp(min=1e-6).unsqueeze(-1), dz, dpsi

    def goal_command(self, estimated: bool) -> torch.Tensor:
        if estimated:
            base_pos, base_quat = self.odometry_sensor.estimated_pos_w, self.odometry_sensor.estimated_quat_w
        else:
            base_pos, base_quat = self.base_pos_w, self.base_quat_w
        yaw = yaw_of_quat(base_quat)
        d_w = self.target_pos_w[:, :2] - base_pos[:, :2]
        cos, sin = torch.cos(yaw), torch.sin(yaw)
        d = torch.stack((cos * d_w[:, 0] + sin * d_w[:, 1], -sin * d_w[:, 0] + cos * d_w[:, 1]), dim=-1)
        direction = d / torch.maximum(d.norm(dim=-1, keepdim=True), self.radius.unsqueeze(-1))
        heading = wrap_to_pi(self.target_yaw - yaw)
        return torch.cat((direction, heading.unsqueeze(-1)), dim=-1)

    # -- sampling --
    def sample_reset(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Spawn and goal of the goal envs ``env_ids`` on their current tile. Writes the goal; returns the spawn position (surface z,
        n x 3), the spawn heading (travel direction + U(+-reset_yaw_range)), the lying mask and the face-up mask of the get-up envs."""
        tile = self.tile_id[env_ids]
        t = self.tables
        spawn = t.sample_spawn(tile)
        goal, goal_yaw = t.sample_goal(tile)
        self.target_pos_w[env_ids] = goal + torch.tensor([0.0, 0.0, BASE_STAND_HEIGHT], device=self.device)
        self.target_yaw[env_ids] = wrap_to_pi(goal_yaw)
        self.target_time[env_ids] = 0.0
        face = t.face[tile] + (2 * torch.rand(len(env_ids), device=self.device) - 1) * self.cfg.reset_yaw_range
        lying = t.lying[tile]
        face_up = torch.rand(len(env_ids), device=self.device) < 0.5
        return spawn, face, lying, face_up

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        ref = env_ids[~self.is_goal[env_ids]]
        if len(ref) > 0:
            super()._resample_command(ref)
        self.min_xy_ok[env_ids] = float("inf")  # new episode (the goal of a goal env is written by the reset event)

    def _update_metrics(self) -> None:
        super()._update_metrics()
        dist, _, dz, _ = self.goal_geometry()
        ok = self.is_goal & (dz.abs() <= 0.1)  # horizontal curriculum distances are measured only below the vertical tolerance (Table S4)
        self.min_xy_ok = torch.where(ok, torch.minimum(self.min_xy_ok, dist), self.min_xy_ok)
        self.metrics["goal_min_xy_ok"][:] = torch.where(torch.isinf(self.min_xy_ok), torch.full_like(dist, 10.0), self.min_xy_ok)

    def _update_command(self, env_ids: torch.Tensor | None) -> None:
        if env_ids is None:
            # reference envs only: the goal of a goal env is fixed for the episode
            time = self.motion_time
            stale = ((self.target_time - time < self.cfg.lookahead_range_s[0]) & ~self.is_goal).nonzero().flatten()
            if len(stale) > 0:
                self._sample_goal(stale, time[stale])


# ----------------------------------------------------------------------------------------------------------------------------------
# curriculum
# ----------------------------------------------------------------------------------------------------------------------------------
class GoalTerrainCurriculum(ManagerTermBase):
    """Terrain curriculum of the goal-reaching envs (supplement "Terrains and Curriculum", Table S4).

    After an episode: ``s <- 0.8 s + 0.2 I_success`` (success = within 0.5 m horizontally of the goal at the required height (|dz| <= 0.1 m)
    and the episode ended without early termination); promoted one level if it succeeded and ``s > 0.5``, demoted one level if its
    minimum qualifying goal distance exceeded 1.0 m. Levels are clipped to [0, 9]."""

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        p = cfg.params
        self.ema_weight: float = p.get("ema_weight", 0.2)
        self.promote_ema: float = p.get("promote_ema", 0.5)
        self.promote_distance: float = p.get("promote_distance", 0.5)
        self.demote_distance: float = p.get("demote_distance", 1.0)
        self.command: AcpStage3GoalCommand = env.command_manager.get_term(p.get("command_name", GOAL_COMMAND))

    def update(self, ended: torch.Tensor, terminated: torch.Tensor) -> torch.Tensor:
        """Curriculum update of the goal envs ``ended`` (indices) given whether they ended in an early termination. Returns their success."""
        c = self.command
        min_xy = c.min_xy_ok[ended]
        success = (min_xy <= self.promote_distance) & ~terminated
        c.success_ema[ended] = (1 - self.ema_weight) * c.success_ema[ended] + self.ema_weight * success.float()
        promote = success & (c.success_ema[ended] > self.promote_ema)
        demote = min_xy > self.demote_distance
        c.level[ended] = (c.level[ended] + promote.long() - demote.long()).clamp(0, NUM_LEVELS - 1)
        return success

    def __call__(self, env: ManagerBasedRlEnv, env_ids: torch.Tensor, **params) -> dict[str, float]:
        """Called before the reset of ``env_ids``."""
        del params
        c = self.command
        ended = env_ids[(env.episode_length_buf[env_ids] > 0) & c.is_goal[env_ids]]
        out = {}
        if len(ended) > 0:
            success = self.update(ended, env.termination_manager.terminated[ended])
            out["episode_success"] = success.float().mean().item()
        goal = c.is_goal
        out["level_mean"] = c.level[goal].float().mean().item() if goal.any() else 0.0
        out["level_max"] = c.level[goal].max().item() if goal.any() else 0
        out["success_ema_mean"] = c.success_ema[goal].mean().item() if goal.any() else 0.0
        return out


# ----------------------------------------------------------------------------------------------------------------------------------
# events
# ----------------------------------------------------------------------------------------------------------------------------------
def reset_reference_envs(env: ManagerBasedRlEnv, env_ids: torch.Tensor, command_name: str = GOAL_COMMAND, **kwargs) -> None:
    """``reset_robot_state_by_reference`` on the reference envs among ``env_ids``."""
    term: AcpStage3GoalCommand = env.command_manager.get_term(command_name)
    ref = env_ids[~term.is_goal[env_ids]]
    if len(ref) > 0:
        reset_robot_state_by_reference(env, ref, **kwargs)


def reset_goal_envs(env: ManagerBasedRlEnv, env_ids: torch.Tensor, command_name: str = GOAL_COMMAND, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> None:
    """Reset the goal envs among ``env_ids``: tile from the curriculum level, spawn on a spawn patch of the tile, heading U(+-0.4) around the
    travel direction, base height + U(0, 0.05) m, joints around the default pose U(+-0.2), zero velocity; the get-up tiles drop the robot lying
    face up or face down; and sample the goal."""
    term: AcpStage3GoalCommand = env.command_manager.get_term(command_name)
    ids = env_ids[term.is_goal[env_ids]]
    if len(ids) == 0:
        return
    cfg = term.cfg
    robot = env.scene[asset_cfg.name]
    spawn, face, lying, face_up = term.sample_reset(ids)
    n, dev = len(ids), env.device
    zeros = torch.zeros(n, device=dev)
    pitch = torch.where(lying, torch.where(face_up, -zeros - math.pi / 2, zeros + math.pi / 2), zeros)  # chest up: -90 deg about y
    quat = quat_from_euler_xyz(zeros, pitch, face)
    height = torch.where(lying, torch.full_like(zeros, cfg.lying_height), torch.full_like(zeros, BASE_STAND_HEIGHT)) + torch.empty(n, device=dev).uniform_(*cfg.reset_height_range)
    pos = torch.cat((spawn[:, :2], (spawn[:, 2] + height).unsqueeze(-1)), -1)
    robot.write_root_link_pose_to_sim(torch.cat((pos, quat), -1), env_ids=ids)
    robot.write_root_link_velocity_to_sim(torch.zeros(n, 6, device=dev), env_ids=ids)
    limits = robot.data.soft_joint_pos_limits[ids]
    joint_pos = robot.data.default_joint_pos[ids] + cfg.reset_joint_range * (2 * torch.rand(n, robot.data.default_joint_pos.shape[-1], device=dev) - 1)
    joint_pos = torch.maximum(torch.minimum(joint_pos, limits[..., 1]), limits[..., 0])
    robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos), env_ids=ids)


# ----------------------------------------------------------------------------------------------------------------------------------
# rewards (paper Table 3, Stage 3): the goal terms apply to the goal envs, the reference terms to the reference envs
# ----------------------------------------------------------------------------------------------------------------------------------
def _goal_term(env: ManagerBasedRlEnv, command_name: str = GOAL_COMMAND) -> AcpStage3GoalCommand:
    return env.command_manager.get_term(command_name)


def _ref_only(func):
    """Reward / termination of the reference envs: ``func`` masked to the reference envs."""

    def wrapped(env: ManagerBasedRlEnv, **params):
        return func(env, **params) * _goal_term(env).ref_mask

    wrapped.__name__, wrapped.__doc__ = f"{func.__name__}_ref_only", f"``{func.__name__}`` of the reference envs only."
    return wrapped


track_base_pos_ref = _ref_only(acp_rewards.track_base_pos)
track_base_vel_ref = _ref_only(acp_rewards.track_base_vel)
track_joint_pos_ref = _ref_only(acp_rewards.track_joint_pos)
track_joint_vel_ref = _ref_only(acp_rewards.track_joint_vel)
track_contact_label_ref = _ref_only(acp_rewards.track_contact_label)
track_contact_pos_base_ref = _ref_only(acp_rewards.track_contact_pos_base)
track_contact_pos_global_ref = _ref_only(acp_rewards.track_contact_pos_global)
track_planned_contact_ref = _ref_only(acp_rewards.track_planned_contact)
track_body_pos_anchor_ref = _ref_only(acp_rewards.track_body_pos_anchor)
track_body_ori_ref = _ref_only(acp_rewards.track_body_ori)
track_body_lin_vel_ref = _ref_only(acp_rewards.track_body_lin_vel)
track_body_ang_vel_ref = _ref_only(acp_rewards.track_body_ang_vel)


def move2goal(
    env: ManagerBasedRlEnv,
    command_name: str = GOAL_COMMAND,
    radius: float = 0.5,
    min_speed: float = 0.1,
    z_speed: float = 0.1,
    z_tolerance: float = 0.2,
    stair_speed_cap: float = 0.8,
    speed_cap: float = 2.0,
) -> torch.Tensor:
    """(1 - g_xy) g_v + g_xy g_z with g_xy = 1[d_g < 0.5], g_v = 1[v_->g > 0.1] 1[|v_b,xy| <= v_max] (0.8 m/s on the stair terrains, 2.0 m/s
    elsewhere), g_z = 1[v_z sign(dz) > 0.1 or |dz| < 0.2] on the get-up terrains and 1 elsewhere."""
    term = _goal_term(env, command_name)
    dist, direction, dz, _ = term.goal_geometry()
    v = term.base_lin_vel_w
    v_xy = v[:, :2]
    cap = term.tables.v_max[term.tile_id]
    g_xy = dist < radius
    g_v = ((v_xy * direction).sum(-1) > min_speed) & (v_xy.norm(dim=-1) <= cap)
    g_z = torch.where(term.flag("getup"), (v[:, 2] * torch.sign(dz) > z_speed) | (dz.abs() < z_tolerance), torch.ones_like(g_xy))
    return (torch.where(g_xy, g_z, g_v) & term.is_goal).float()


def standatgoal(
    env: ManagerBasedRlEnv,
    command_name: str = GOAL_COMMAND,
    contacts_command_name: str = "acp_reference_contacts",
    force_threshold: float = 1.0,
    radius: float = 0.5,
    z_tolerance: float = 0.2,
    heading_tolerance: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """g_xy g_h g_psi exp(-(d_foot + d_tilt + d_q) / 3): d_foot = fraction of the feet without measured contact, d_tilt = 1 - (g_z^base)^2
    (vertical component of the projected gravity of the pelvis), d_q = mean |q_j - q_0,j| over the 23 controlled joints."""
    term = _goal_term(env, command_name)
    robot = term.robot
    dist, _, dz, dpsi = term.goal_geometry()
    gate = (dist < radius) & (dz.abs() < z_tolerance) & (dpsi.abs() < heading_tolerance) & term.is_goal
    contact = acp_rewards.measured_contact_flags(env, contacts_command_name, force_threshold)[:, list(FEET)]
    d_foot = 1.0 - contact.float().mean(-1)
    g_z = quat_apply_inverse(term.base_quat_w, robot.data.gravity_vec_w)[:, 2]
    d_tilt = 1.0 - g_z.square()
    d_q = (robot.data.joint_pos[:, asset_cfg.joint_ids] - robot.data.default_joint_pos[:, asset_cfg.joint_ids]).abs().mean(-1)
    return gate.float() * torch.exp(-(d_foot + d_tilt + d_q) / 3.0)


def _torso_id(term: AcpStage3GoalCommand) -> int:
    return term.robot.find_bodies("torso_link")[0][0]


def flat_orientation(env: ManagerBasedRlEnv, command_name: str = GOAL_COMMAND) -> torch.Tensor:
    """T_ori |g_torso,xy|^2: horizontal components of the gravity in the torso frame (ACP-unspecified terrain mask: the foot terrains)."""
    term = _goal_term(env, command_name)
    g = quat_apply_inverse(term.robot.data.body_link_quat_w[:, _torso_id(term)], term.robot.data.gravity_vec_w)
    return g[:, :2].square().sum(-1) * term.flag("foot").float()


def ang_vel_xy(env: ManagerBasedRlEnv, command_name: str = GOAL_COMMAND) -> torch.Tensor:
    """T_w |omega^W_torso,xy|^2: horizontal world angular velocity of the torso (ACP-unspecified terrain mask: the foot terrains)."""
    term = _goal_term(env, command_name)
    w = term.robot.data.body_link_ang_vel_w[:, _torso_id(term), :2]
    return w.square().sum(-1) * term.flag("foot").float()


def stumbling_links(env: ManagerBasedRlEnv, contacts_command_name: str = "acp_reference_contacts") -> torch.Tensor:
    """(N, 9) indicators 1[F_max_f,xy > 1.1 F_max_f,z + 1] of the 9 contact links, the maxima over the sensor-history window, without the
    terrain mask (the label of the learned stumbling prediction of the Stage 3 imagination)."""
    contacts: AcpReferenceContacts = env.command_manager.get_term(contacts_command_name)
    force = contacts.contact_sensor.data.force_history[:, contacts.contact_sensor_ids]  # (N, 9, H, 3)
    f_xy = force[..., :2].norm(dim=-1).amax(-1)
    f_z = force[..., 2].abs().amax(-1)
    return (f_xy > 1.1 * f_z + 1.0).float()


def stumbling(env: ManagerBasedRlEnv, command_name: str = GOAL_COMMAND, contacts_command_name: str = "acp_reference_contacts") -> torch.Tensor:
    """T_st sum_f 1[F_max_f,xy > 1.1 F_max_f,z + 1] over the 9 contact links, the maxima over the sensor-history window (ACP-unspecified:
    the monitored links are the 9 contact links, the terrain mask the foot terrains)."""
    term = _goal_term(env, command_name)
    return stumbling_links(env, contacts_command_name).sum(-1) * term.flag("stumble").float()


class slippage(ManagerTermBase):
    """T_sl sum_f 1[t_contact_f > 0.026] 1[|v_f| > 0.5] over the 9 contact links: continuous contact for longer than 0.026 s (two control
    steps of the 50 Hz contact flag) and a world speed above 0.5 m/s (ACP-unspecified: the terrain mask is the foot terrains plus the box
    and uneven-climb terrains)."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        self.command_name = cfg.params.get("command_name", GOAL_COMMAND)
        self.contacts_command_name = cfg.params.get("contacts_command_name", "acp_reference_contacts")
        self.contact_time = torch.zeros(env.num_envs, len(CONTACT_LINKS), device=env.device)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        self.contact_time[slice(None) if env_ids is None else env_ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        command_name: str = GOAL_COMMAND,
        contacts_command_name: str = "acp_reference_contacts",
        force_threshold: float = 1.0,
        min_contact_time: float = 0.026,
        max_speed: float = 0.5,
    ) -> torch.Tensor:
        term = _goal_term(env, command_name)
        contact = acp_rewards.measured_contact_flags(env, contacts_command_name, force_threshold)
        self.contact_time = (self.contact_time + env.step_dt) * contact.float()
        contacts: AcpReferenceContacts = env.command_manager.get_term(contacts_command_name)
        speed = term.robot.data.body_link_lin_vel_w[:, contacts.contact_body_ids].norm(dim=-1)
        slip = (self.contact_time > min_contact_time) & (speed > max_speed)
        return slip.float().sum(-1) * term.flag("slip").float()


# ----------------------------------------------------------------------------------------------------------------------------------
# terminations
# ----------------------------------------------------------------------------------------------------------------------------------
def _ref_only_termination(func):
    def wrapped(env: ManagerBasedRlEnv, command_name: str = GOAL_COMMAND, **params):
        return func(env, **params) & ~_goal_term(env, command_name).is_goal

    wrapped.__name__, wrapped.__doc__ = f"{func.__name__}_ref_only", f"``{func.__name__}`` of the reference envs only."
    return wrapped


dataset_exhausted_ref_only = _ref_only_termination(dataset_exhausted)


def pos_far_from_ref_ref_only(env: ManagerBasedRlEnv, command_name: str, goal_command_name: str = GOAL_COMMAND, **params) -> torch.Tensor:
    return acp_terminations.pos_far_from_ref(env, command_name, **params) & ~_goal_term(env, goal_command_name).is_goal


def projected_gravity_far_from_ref_ref_only(env: ManagerBasedRlEnv, command_name: str, goal_command_name: str = GOAL_COMMAND, **params) -> torch.Tensor:
    return acp_terminations.projected_gravity_far_from_ref(env, command_name, **params) & ~_goal_term(env, goal_command_name).is_goal


def pos_xy_far_from_ref_ref_only(env: ManagerBasedRlEnv, command_name: str, goal_command_name: str = GOAL_COMMAND, **params) -> torch.Tensor:
    return acp_terminations.pos_xy_far_from_ref(env, command_name, **params) & ~_goal_term(env, goal_command_name).is_goal


def feet_far_from_ref_ref_only(env: ManagerBasedRlEnv, command_name: str, goal_command_name: str = GOAL_COMMAND, **params) -> torch.Tensor:
    return acp_terminations.feet_far_from_ref(env, command_name, **params) & ~_goal_term(env, goal_command_name).is_goal


def goal_time_out(env: ManagerBasedRlEnv, command_name: str = GOAL_COMMAND) -> torch.Tensor:
    """Goal envs: the maximum goal-reaching episode duration (15 s, Table S4) is over."""
    term = _goal_term(env, command_name)
    return term.is_goal & (env.episode_length_buf * env.step_dt >= term.cfg.goal_episode_s)


def goal_fall(
    env: ManagerBasedRlEnv,
    command_name: str = GOAL_COMMAND,
    fall_margin: float = 0.25,
    tilt_threshold: float = 0.8,
    clearance_threshold: float = 0.3,
    scan_sensor: str = "critic_height_scanner",
) -> torch.Tensor:
    """ACP-unspecified fall of a goal env: the pelvis is below the lowest walkable surface of the tile (``fall_margin``), or, on the foot
    terrains, its projected gravity is further than ``tilt_threshold`` from upright (as ``base_pg_too_far``) or it is lower than
    ``clearance_threshold`` above the surface under it."""
    from .observations import SCAN_NUM_X, SCAN_NUM_Y, _scan_height

    term = _goal_term(env, command_name)
    z = term.base_pos_w[:, 2]
    below = z < term.tables.z_low[term.tile_id] + term.tables.origin[term.tile_id][:, 2] - fall_margin
    pg = quat_apply_inverse(term.base_quat_w, term.robot.data.gravity_vec_w)
    tilt = (pg - torch.tensor([0.0, 0.0, -1.0], device=pg.device)).norm(dim=-1) > tilt_threshold
    clearance = _scan_height(env, scan_sensor, 0.0)[:, SCAN_NUM_Y // 2, SCAN_NUM_X // 2]  # pelvis z - surface under the pelvis
    low = clearance < clearance_threshold
    return term.is_goal & (below | (term.flag("foot") & (tilt | low)))
