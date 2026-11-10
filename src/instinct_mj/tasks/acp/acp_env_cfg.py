"""Stage-1 environment of ACP (Humanoid Mobility via Attention-based Contacts, arXiv 2610.31608).

A contact-conditioned whole-body tracker over reference motions with aligned terrains. The tracker observes proprioception,
a 9 x 7 height scan, the next contact target of each of 9 contact bodies (Stage 1: read from the reference contact labels)
and a goal command; the planner (Stage 2/3) reads the 17 x 11 scan, the proprioception and the goal. Rewards follow Table 3
of the paper plus the SuperMimic body terms, domain randomization Table S2.

``make_rewards`` / ``make_terminations`` / ``make_events`` are THE definition of the v1 MDP, shared by Stages 1, 2 and 3 (Stage 3 only masks
the reference terms to the reference envs, ``acp_stage3_env_cfg``). The v1 deviations from the paper are listed in ``rl/acp/PLAN.md``.

The reward weights / parameters, terminations, action scale and PD gains of this env are read by the SuperMimic ACP runner.
Values the paper does not give are marked ``# ACP-unspecified``.
"""

import math
from dataclasses import MISSING, dataclass, field

import mjlab.envs.mdp as mdp
from mjlab.managers import CurriculumTermCfg, EventTermCfg
from mjlab.managers import ObservationGroupCfg as ObsGroupCfg
from mjlab.managers import ObservationTermCfg as ObsTermCfg
from mjlab.managers import RewardTermCfg as RewTermCfg
from mjlab.managers import SceneEntityCfg
from mjlab.managers import TerminationTermCfg as DoneTermCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.utils.noise import UniformNoiseCfg

import instinct_mj.envs.mdp as instinct_mdp
import instinct_mj.tasks.acp.mdp as acp_mdp
from instinct_mj.envs.manager_based_rl_env_cfg import InstinctRlEnvCfg
from instinct_mj.managers import MultiRewardCfg
from instinct_mj.tasks.acp.robot_limits import ACP_G1_JOINT_VELOCITY_LIMITS
from instinct_mj.tasks.acp.scene_cfg import AcpSceneCfg

PROPRIO_HISTORY_LENGTH = 5

# Velocity limits of the joints of the robot: ``robot_limits.ACP_G1_JOINT_VELOCITY_LIMITS`` (the authors' model has none; datasheet values, see there).

# The 23 controlled joints: all joints of the robot (the wrists are welded in the ACP G1; the regex keeps the wrist-free selection explicit).
ACTION_JOINT_NAMES = (r"^(?!.*wrist).*$",)

# Reward terms read the reference through this command term (SuperMimic reads the term parameters from the env).
CONTACTS_COMMAND = "acp_reference_contacts"
GOAL_COMMAND = "acp_goal"
CURRICULUM_NAME = "episode_start"


def make_commands(*, require_full_clip: bool = True) -> dict:
    """Reference contacts (Stage 1 contact targets, reference labels) and the goal command. ``require_full_clip``: the episode cap must fit the longest clip."""
    return {
        CONTACTS_COMMAND: acp_mdp.AcpReferenceContactsCfg(require_full_clip=require_full_clip),
        GOAL_COMMAND: acp_mdp.AcpGoalCommandCfg(),
    }


def make_actions() -> dict[str, mdp.JointPositionActionCfg]:
    """23 joint-position offsets around the default pose, scale 0.25."""
    return {
        "joint_pos": mdp.JointPositionActionCfg(
            entity_name="robot",
            actuator_names=ACTION_JOINT_NAMES,
            scale={".*": 0.25},
            use_default_offset=True,
        ),
    }


def make_observations() -> dict[str, ObsGroupCfg]:
    """``policy`` (tracker, 468), ``planner`` (561) and ``critic`` (353, privileged) groups."""
    action_joints = SceneEntityCfg("robot", joint_names=ACTION_JOINT_NAMES)

    policy_terms = {
        # proprioception, 5-step history (oldest first), flattened term-major
        # base = pelvis (IMU in the pelvis), as in the authors' deploy stack (g1_rsl_cpp)
        "base_ang_vel": ObsTermCfg(
            func=acp_mdp.base_ang_vel,
            params={"command_name": CONTACTS_COMMAND},
            noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05),
            history_length=PROPRIO_HISTORY_LENGTH,
        ),
        "projected_gravity": ObsTermCfg(
            func=acp_mdp.projected_gravity,
            params={"command_name": CONTACTS_COMMAND},
            noise=UniformNoiseCfg(n_min=-0.0125, n_max=0.0125),
            history_length=PROPRIO_HISTORY_LENGTH,
        ),
        "joint_pos": ObsTermCfg(
            func=mdp.joint_pos_rel,
            # joint zero-position bias (encoder bias DR) is part of the measurement
            params={"biased": True, "asset_cfg": action_joints},
            noise=UniformNoiseCfg(n_min=-0.01, n_max=0.01),
            history_length=PROPRIO_HISTORY_LENGTH,
        ),
        "joint_vel": ObsTermCfg(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": action_joints},
            noise=UniformNoiseCfg(n_min=-1.5, n_max=1.5),
            history_length=PROPRIO_HISTORY_LENGTH,
        ),
        "last_action": ObsTermCfg(func=mdp.last_action, history_length=PROPRIO_HISTORY_LENGTH),
        # 9 x 7 scan (central block of the 17 x 11 planner scan), base_z - hit_z - 0.5
        "tracker_height_scan": ObsTermCfg(
            func=acp_mdp.tracker_height_scan,
            params={"sensor_name": "height_scanner", "offset": 0.5},
            noise=UniformNoiseCfg(n_min=-0.1, n_max=0.1),
        ),
        # next contact target of the 9 contact bodies in the base-yaw frame (noise on the valid entries inside the term)
        "contact_targets": ObsTermCfg(
            func=acp_mdp.contact_targets,
            params={"command_name": CONTACTS_COMMAND, "noise": 0.1},
        ),
        "goal_command": ObsTermCfg(
            func=mdp.generated_commands,
            params={"command_name": GOAL_COMMAND},
            noise=UniformNoiseCfg(n_min=-0.2, n_max=0.2),
        ),
    }

    # The planner additionally reads the proprioception history and the goal from the policy group.
    planner_terms = {
        # noise on the z channel inside the term (Gaussian, sigma 0.03)
        "planner_height_scan": ObsTermCfg(
            func=acp_mdp.planner_height_scan,
            params={"sensor_name": "height_scanner", "offset": 0.5, "z_noise_std": 0.03},
        ),
    }

    critic_terms = {
        "base_ang_vel": ObsTermCfg(func=acp_mdp.base_ang_vel, params={"command_name": CONTACTS_COMMAND}),
        "projected_gravity": ObsTermCfg(func=acp_mdp.projected_gravity, params={"command_name": CONTACTS_COMMAND}),
        "joint_pos": ObsTermCfg(func=mdp.joint_pos_rel, params={"asset_cfg": action_joints}),
        "joint_vel": ObsTermCfg(func=mdp.joint_vel_rel, params={"asset_cfg": action_joints}),
        "last_action": ObsTermCfg(func=mdp.last_action),
        "height_scan": ObsTermCfg(
            func=acp_mdp.critic_height_scan,
            params={"sensor_name": "critic_height_scanner", "offset": 0.5},
        ),
        "base_lin_vel": ObsTermCfg(func=acp_mdp.base_lin_vel, params={"command_name": CONTACTS_COMMAND}),
        "reference_kinematics": ObsTermCfg(
            func=acp_mdp.reference_kinematics,
            params={"command_name": CONTACTS_COMMAND, "asset_cfg": action_joints},
        ),
        "reference_contact_pos": ObsTermCfg(
            func=acp_mdp.reference_contact_pos_b,
            params={"command_name": CONTACTS_COMMAND},
        ),
        "reference_contact_flags": ObsTermCfg(
            func=acp_mdp.reference_contact_flags,
            params={"command_name": CONTACTS_COMMAND},
        ),
    }

    return {
        "policy": ObsGroupCfg(terms=policy_terms, enable_corruption=True, concatenate_terms=True),
        "planner": ObsGroupCfg(terms=planner_terms, enable_corruption=True, concatenate_terms=True),
        "critic": ObsGroupCfg(terms=critic_terms, enable_corruption=False, concatenate_terms=True),
    }


def make_rewards() -> dict[str, RewTermCfg]:
    """Rewards of the v1 task (per-step reward = dt * sum_i w_i r_i; the regularizers sum over the 23 controlled joints): paper Table 3
    (``track_base_pos`` sigma 0.5, ``track_contact_pos_global`` sigma 1.0) plus the four SuperMimic r_track body terms (App. A Table A.2 /
    supermimic_vibed mdp.py:108-131, all weight 1.0, mean over the 14 tracked bodies ``TRACKED_LINKS``): ``track_body_pos_anchor`` (sigma 0.3 m;
    the reference positions re-anchored to (robot root xy, reference root z) and rotated by the yaw of R R*^T), ``track_body_ori`` (0.4),
    ``track_body_lin_vel`` (1.0), ``track_body_ang_vel`` (3.14). SuperMimic's two root terms are not added (``track_base_pos`` covers them)."""
    action_joints = SceneEntityCfg("robot", joint_names=ACTION_JOINT_NAMES)
    command = {"command_name": CONTACTS_COMMAND}
    return {
        "track_base_pos": RewTermCfg(
            func=acp_mdp.track_base_pos,
            weight=3.0,
            params={**command, "sigma_pos": 0.5, "sigma_rot": 1.0, "moving_speed_threshold": 0.1},
        ),
        "track_base_vel": RewTermCfg(
            func=acp_mdp.track_base_vel,
            weight=1.0,
            params={**command, "sigma_lin_vel": 1.0, "sigma_ang_vel": 3.0},
        ),
        "track_joint_pos": RewTermCfg(
            func=acp_mdp.track_joint_pos,
            weight=2.5,
            params={**command, "sigma": 2.0, "asset_cfg": action_joints},
        ),
        "track_joint_vel": RewTermCfg(
            func=acp_mdp.track_joint_vel,
            weight=0.5,
            params={**command, "sigma": 10.0, "asset_cfg": action_joints},
        ),
        "track_contact_label": RewTermCfg(
            func=acp_mdp.track_contact_label,
            weight=5.0,
            params={**command, "force_threshold": 1.0},
        ),
        "track_contact_pos_base": RewTermCfg(
            func=acp_mdp.track_contact_pos_base,
            weight=2.0,
            params={**command, "sigma": 1.0},
        ),
        "track_contact_pos_global": RewTermCfg(
            func=acp_mdp.track_contact_pos_global,
            weight=1.0,
            params={**command, "sigma": 1.0, "moving_speed_threshold": 0.1},
        ),
        # Stage 1: weight 0 (evaluated against the reference contact targets as a diagnostic); the runner gives it 1.5 * eta_rew in Stages 2 / 3.
        "track_planned_contact": RewTermCfg(
            func=acp_mdp.track_planned_contact,
            weight=0.0,
            params={**command, "sigma": 1.0},
        ),
        "track_body_pos_anchor": RewTermCfg(func=acp_mdp.track_body_pos_anchor, weight=1.0, params={**command, "sigma": 0.3}),
        "track_body_ori": RewTermCfg(func=acp_mdp.track_body_ori, weight=1.0, params={**command, "sigma": 0.4}),
        "track_body_lin_vel": RewTermCfg(func=acp_mdp.track_body_lin_vel, weight=1.0, params={**command, "sigma": 1.0}),
        "track_body_ang_vel": RewTermCfg(func=acp_mdp.track_body_ang_vel, weight=1.0, params={**command, "sigma": 3.14}),
        "action_rate": RewTermCfg(func=mdp.action_rate_l2, weight=-0.1),
        "action_rate_2nd": RewTermCfg(func=mdp.action_acc_l2, weight=-0.05),
        "joint_torque": RewTermCfg(
            func=instinct_mdp.joint_torques_l2,
            weight=-1e-5,
            params={"asset_cfg": action_joints},
        ),
        "torque_limits": RewTermCfg(
            func=acp_mdp.torque_limit_excess,
            weight=-1e-3,
            params={"asset_cfg": action_joints},
        ),
        "dof_pos_limits": RewTermCfg(
            func=acp_mdp.joint_pos_limits_clipped,
            weight=-10.0,
            params={"asset_cfg": action_joints, "max_value": 30.0},  # Table 3: min(30, sum_j excess)
        ),
        "dof_vel_limits": RewTermCfg(
            func=acp_mdp.dof_vel_limits,
            weight=-1.0,
            params={
                "asset_cfg": action_joints,
                "velocity_limits": ACP_G1_JOINT_VELOCITY_LIMITS,
                "limit_ratio": 0.9,
                "max_value": 1.0,
            },
        ),
        "contact_acc": RewTermCfg(
            func=acp_mdp.contact_acc,
            weight=-0.01,
            # "vel1": the difference of the link velocities (the authors' framework), not the second difference of the positions
            params={**command, "max_value": 1e4, "mode": "vel1"},  # Table 3: min(1e4, sum_b |p''_b|)
        ),
    }


def make_events() -> dict[str, EventTermCfg]:
    """Domain randomization of paper Table S2, the reset to the reference and the episode start frame."""
    return {
        "match_motion_ref_with_scene": EventTermCfg(
            func=instinct_mdp.match_motion_ref_with_scene,
            mode="startup",
            params={"motion_ref_cfg": SceneEntityCfg("motion_reference")},
        ),
        # friction U(0.3, 1.2) (one value per env for all robot geoms; ground friction see ``scene_cfg.TERRAIN_FRICTION``).
        # Restitution U(0, 0.5) is skipped: MuJoCo geoms have no restitution coefficient.
        "physics_material": EventTermCfg(
            func=mdp.dr.geom_friction,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", geom_names=".*"),
                "ranges": {0: (0.3, 1.2)},
                "operation": "abs",
                "distribution": "uniform",
                "shared_random": True,
            },
        ),
        # torso added mass U(-2, 4) kg (a point mass at the CoM: the inertia is unchanged)
        "torso_added_mass": EventTermCfg(
            func=mdp.dr.body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
                "ranges": (-2.0, 4.0),
                "operation": "add",
            },
        ),
        # leg link mass x U(0.85, 1.15) (mass and inertia scale together)
        "leg_link_mass": EventTermCfg(
            func=mdp.dr.pseudo_inertia,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=[".*_hip_.*_link", ".*_knee_link", ".*_ankle_.*_link"],
                ),
                "alpha_range": (0.5 * math.log(0.85), 0.5 * math.log(1.15)),
                "distribution": instinct_mdp.uniform_mass_scale_to_alpha,
            },
        ),
        "base_com": EventTermCfg(
            func=instinct_mdp.randomize_rigid_body_com,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
                "com_range": {"x": (-0.025, 0.025), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
            },
        ),
        "joint_armature": EventTermCfg(
            func=mdp.dr.joint_armature,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "ranges": (0.9, 1.1),
                "operation": "scale",
            },
        ),
        "randomize_actuator_gains": EventTermCfg(
            func=mdp.dr.pd_gains,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "kp_range": (0.9, 1.1),
                "kd_range": (0.9, 1.1),
                "operation": "scale",
                "distribution": "uniform",
            },
        ),
        # joint zero-position bias U(+-0.01) rad (ankles +-0.03), sampled once per env at simulation startup (paper p16 and Table S2;
        # encoder bias: the observed joint positions carry it and the action targets compensate for it)
        "joint_zero_bias": EventTermCfg(
            func=mdp.dr.encoder_bias,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(r"^(?!.*ankle).*$",)),
                "bias_range": (-0.01, 0.01),
            },
        ),
        "ankle_zero_bias": EventTermCfg(
            func=mdp.dr.encoder_bias,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*ankle.*",)),
                "bias_range": (-0.03, 0.03),
            },
        ),
        # the clip start frame (SuperMimic reset bins) must be set after the motion reference reset and before the robot reset. Start frames
        # whose reference penetrates the terrain by more than 1 cm are not used (the fallback of the reset rejection, the noise-free
        # reference, must itself be clear of the terrain).
        "episode_start": EventTermCfg(
            func=acp_mdp.reset_motion_start_phase,
            mode="reset",
            params={"curriculum_name": CURRICULUM_NAME, "legal_start_max_penetration": 0.01},
        ),
        # base pose / velocity from the reference, base height + U(0, 0.05) m, joint positions +-0.2 rad (clamped to the joint limits), no
        # root pose / velocity noise (Table S2; the SuperMimic reference noise is made for flat ground and put robots on contact poses into
        # ladder rungs / boxes). The noise of an env that starts > 1 cm inside the terrain is redrawn (up to 3 times, then the noise-free
        # reference) and the root is raised by the deepest remaining penetration (a foot / hand pushed into a box top starts on it).
        "reset_robot": EventTermCfg(
            func=instinct_mdp.reset_robot_state_by_reference,
            mode="reset",
            params={
                "motion_ref_cfg": SceneEntityCfg("motion_reference"),
                "asset_cfg": SceneEntityCfg("robot"),
                "position_offset": [0.0, 0.0, 0.0],
                "dof_vel_ratio": 1.0,
                "base_lin_vel_ratio": 1.0,
                "base_ang_vel_ratio": 1.0,
                # reset noise of zita-ch/supermimic_vibed (env.py:153-167; replaces Table S2's z + U(0, 0.05) m / joints +-0.2 rad): root position, orientation
                # (world-frame rotation), world linear / angular velocity and joint noise around the reference state (the penetration resampling / lift below stays)
                "randomize_pose_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01), "roll": (-0.1, 0.1), "pitch": (-0.1, 0.1), "yaw": (-0.2, 0.2)},
                "randomize_velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.2, 0.2), "roll": (-0.52, 0.52), "pitch": (-0.52, 0.52), "yaw": (-0.78, 0.78)},
                "randomize_joint_pos_range": (-0.1, 0.1),
                "clamp_joint_pos_to_limits": True,
                "max_penetration": 0.01,
                "max_resample": 3,
                "lift_out_of_terrain": True,
            },
        ),
        # per-episode odometry error of the pelvis pose (Table S2): drift in a 0.05 m disk (XY) / U(+-0.05) m (Z) and a
        # delay U(50, 100) ms, applied to the pose used for the actor / planner scans and the contact-target / goal transforms
        "odometry": EventTermCfg(
            func=acp_mdp.randomize_odometry,
            mode="reset",
            params={
                "sensor_name": "height_scanner",
                "xy_radius": 0.05,
                "z_range": 0.05,
                "delay_range_s": (0.05, 0.1),
            },
        ),
    }


def make_curriculum() -> dict[str, CurriculumTermCfg]:
    """Episode start (``mode``: "bins" SuperMimic reset bins attached by the runner, "uniform", "start" = phase 0 for evaluation / Stage 3)."""
    return {
        CURRICULUM_NAME: CurriculumTermCfg(
            func=acp_mdp.EpisodeStart,
            params={"mode": "bins", "uniform_max_phase": 0.98},
        ),
    }


def make_terminations() -> dict[str, DoneTermCfg]:
    """The v1 terminations (the paper gives no thresholds; the ACP authors: the robot fell or lost the reference): root height |z - z_ref| > 0.25 m
    (``base_pos_too_far``, height only) and tilt |R_zz - R_zz_ref| > 0.8 (``base_pg_too_far``, z component only), as SuperMimic (upstream mdp.py:147-161);
    the pelvis horizontally > 0.5 m from the reference pelvis (``base_xy_too_far``) and a foot > 0.5 m (3-D) from the reference foot (``foot_pos_too_far``)
    (# ACP-unspecified: 0.5 m, the success radius of the evaluation) and a contact violation, the torso (with the head) touching the terrain with > 1 N
    while the reference torso is > 5 cm from it (``contact_violation``). Time-outs (not failures): the episode cap (``time_out``), the clip end
    (``dataset_exhausted``) and the reset guard ``illegal_reset_contact``."""
    return {
        "time_out": DoneTermCfg(func=mdp.time_out, time_out=True),
        "dataset_exhausted": DoneTermCfg(
            func=instinct_mdp.dataset_exhausted,
            time_out=True,
            params={"reference_cfg": SceneEntityCfg("motion_reference"), "print_reason": False},
        ),
        # InstinctLab's perceptive shadowing term: an episode whose first 2 policy steps both see > 500 N on a body other than the feet / hands
        # started inside the terrain (reset noise on a contact pose, e.g. the ladder rungs). A time-out (not a failure of the policy): bootstrapped,
        # no failure for the bins (clip failure rate), nothing to mirror in the imagination.
        "illegal_reset_contact": DoneTermCfg(
            func=instinct_mdp.illegal_reset_contact,
            time_out=True,
            params={
                "sensor_name": "contact_forces",
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=[r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$).+$"],
                ),
                "threshold": 500,
                "episode_length_threshold": 2,
            },
        ),
        "base_pos_too_far": DoneTermCfg(
            func=acp_mdp.pos_far_from_ref,
            time_out=False,
            params={"command_name": CONTACTS_COMMAND, "distance_threshold": 0.25, "height_only": True},
        ),
        "base_pg_too_far": DoneTermCfg(
            func=acp_mdp.projected_gravity_far_from_ref,
            time_out=False,
            params={"command_name": CONTACTS_COMMAND, "projected_gravity_threshold": 0.8, "z_only": True},
        ),
        "base_xy_too_far": DoneTermCfg(
            func=acp_mdp.pos_xy_far_from_ref,
            time_out=False,
            params={"command_name": CONTACTS_COMMAND, "distance_threshold": 0.5},
        ),
        "foot_pos_too_far": DoneTermCfg(
            func=acp_mdp.feet_far_from_ref,
            time_out=False,
            params={"command_name": CONTACTS_COMMAND, "distance_threshold": 0.5},
        ),
        # the torso carries the head (the G1 has no head link); the imagination mirrors it with the exact torso-terrain distance (sdf_contact.VIOLATION_LINKS)
        "contact_violation": DoneTermCfg(
            func=acp_mdp.body_contact_violation,
            time_out=False,
            params={"command_name": CONTACTS_COMMAND, "sensor_name": "contact_forces", "asset_cfg": SceneEntityCfg("robot", body_names=["torso_link"]), "threshold": 1.0},
        ),
    }


def make_sim() -> SimulationCfg:
    """Physics of the ACP tasks, all values explicit: 200 Hz, implicit-fast Newton solver with 10 iterations / 20 line-search iterations (the authors' MJCF
    has 50; v1 keeps the 10 / 20 of the mjlab scene layer, ``scripts/acp/measure_solver.py`` compares both), pyramidal cone, sparse Jacobian, 128 CCD
    iterations (MuJoCo 3.5's native CCD can exhaust a lower ceiling on the motion-matched terrain), 128 contacts / 700 constraints per world."""
    return SimulationCfg(
        nconmax=128,
        njmax=700,
        contact_sensor_maxmatch=128,
        mujoco=MujocoCfg(
            timestep=0.005,
            integrator="implicitfast",
            impratio=1.0,
            cone="pyramidal",
            jacobian="sparse",
            solver="newton",
            iterations=10,
            tolerance=1e-8,
            ls_iterations=20,
            ls_tolerance=0.01,
            ccd_iterations=128,
            gravity=(0.0, 0.0, -9.81),
        ),
    )


@dataclass(kw_only=True)
class AcpTrackEnvCfg(InstinctRlEnvCfg):
    scene: AcpSceneCfg = MISSING
    decimation: int = 4
    """Policy 50 Hz over the 200 Hz physics (``sim.mujoco.timestep`` 0.005 s)."""
    episode_length_s: float = 120.0
    """ACP-unspecified: the episode lasts until the clip end (``dataset_exhausted``); this is only the upper bound (the command term checks that the longest clip fits)."""
    sim: SimulationCfg = field(default_factory=lambda: make_sim())
    commands: dict = field(default_factory=make_commands)
    actions: dict = field(default_factory=make_actions)
    observations: dict = field(default_factory=make_observations)
    rewards: dict = field(default_factory=lambda: MultiRewardCfg({"rewards": make_rewards()}))
    events: dict = field(default_factory=make_events)
    curriculum: dict = field(default_factory=make_curriculum)
    terminations: dict = field(default_factory=make_terminations)
    contact_detect: str = "force_head"
    """How c^meas of the nine contact bodies is decided (``track_contact_label``, the slippage / foot terms of Stage 3, the planner's hindsight labels): "force_head" (default) =
    the contact sensor's |F_z| > 1 N (the imagination then uses the world model's contact head, as the ACP authors' SuperMimic), "sdf" = the signed distance of the body collision
    capsules to the terrain primitives (exact) is below ``d_on`` (``rl/acp/sdf_contact.py``; the imagination uses its soft version). Keep it equal to the runner's ``contact_detect``."""

    def disable_randomization(self) -> None:
        """Evaluation: no domain randomization and no observation noise (the reset keeps the reference state)."""
        for name in (
            "physics_material",
            "torso_added_mass",
            "leg_link_mass",
            "base_com",
            "joint_armature",
            "randomize_actuator_gains",
            "joint_zero_bias",
            "ankle_zero_bias",
            "odometry",
        ):
            self.events[name] = None
        reset_robot = self.events["reset_robot"].params
        reset_robot["randomize_pose_range"] = {"z": (0.0, 0.0)}
        reset_robot["randomize_velocity_range"] = {}
        reset_robot["randomize_joint_pos_range"] = (0.0, 0.0)
        for group in self.observations.values():
            group.enable_corruption = False
        self.observations["policy"].terms["contact_targets"].params["noise"] = 0.0
        self.observations["planner"].terms["planner_height_scan"].params["z_noise_std"] = 0.0
