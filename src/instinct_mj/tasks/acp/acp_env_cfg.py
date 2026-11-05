"""Stage-1 environment of ACP (Humanoid Mobility via Attention-based Contacts, arXiv 2610.31608).

A contact-conditioned whole-body tracker over reference motions with aligned terrains. The tracker observes proprioception,
a 9 x 7 height scan, the next contact target of each of 9 contact bodies (Stage 1: read from the reference contact labels)
and a goal command; the planner (Stage 2/3) reads the 17 x 11 scan, the proprioception and the goal. Rewards follow Table 3
of the paper, domain randomization Table S2.

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
from mjlab.sensor import ContactMatch, ContactSensorCfg, GridPatternCfg, ObjRef, SensorCfg
from mjlab.utils.noise import UniformNoiseCfg

import instinct_mj.envs.mdp as instinct_mdp
import instinct_mj.tasks.acp.mdp as acp_mdp
import instinct_mj.tasks.shadowing.perceptive.perceptive_env_cfg as perceptual_cfg
from instinct_mj.assets.unitree_g1 import G1_29DOF_JOINT_VELOCITY_LIMITS
from instinct_mj.envs.manager_based_rl_env_cfg import InstinctRlEnvCfg
from instinct_mj.managers import MultiRewardCfg
from instinct_mj.motion_reference.motion_reference_cfg import MotionReferenceManagerCfg
from instinct_mj.rl.acp.planner import CONTACT_LINKS

PROPRIO_HISTORY_LENGTH = 5

# Velocity limits of the joints of the robot (the regex dict of the 29-DoF G1 without the wrist entries, which match no joint).
ACP_G1_JOINT_VELOCITY_LIMITS = {k: v for k, v in G1_29DOF_JOINT_VELOCITY_LIMITS.items() if "wrist" not in k}

# The 23 controlled joints: all joints of the robot (the wrists are welded in the ACP G1; the regex keeps the wrist-free selection explicit).
ACTION_JOINT_NAMES = (r"^(?!.*wrist).*$",)

# Reward terms read the reference through this command term (SuperMimic reads the term parameters from the env).
CONTACTS_COMMAND = "acp_reference_contacts"
GOAL_COMMAND = "acp_goal"
CURRICULUM_NAME = "curriculum_state_init"

# ACP-unspecified: friction of the terrain geoms. MuJoCo mixes the friction of two geoms with max(), so a ground friction
# at the lower bound of the robot friction range makes the contact friction equal to the sampled robot friction
# U(0.3, 1.2) (paper Table S2); without randomization the robot geoms keep their default friction (1.0).
TERRAIN_FRICTION = 0.3


def _edit_acp_scene_spec(spec) -> None:
    perceptual_cfg._edit_perceptive_scene_spec(spec)
    for geom in spec.body("terrain").geoms:
        geom.friction[0] = TERRAIN_FRICTION
        # the height scanner casts rays against geom group 0 only: the robot (groups 2 / 3) is invisible to it
        geom.group = 0


@dataclass(kw_only=True)
class AcpSceneCfg(perceptual_cfg.PerceptiveShadowingSceneCfg):
    """Perceptive-shadowing scene (motion-matched terrains, ``motion_reference`` sensor) with the ACP sensors."""

    def __post_init__(self):
        super().__post_init__()
        self.spec_fn = _edit_acp_scene_spec


def make_acp_scene_sensors(*, motion_reference: MotionReferenceManagerCfg) -> tuple[SensorCfg, ...]:
    """Contact sensor over the 9 contact links, the 17 x 11 height scanner and the motion reference."""
    return (
        # all robot bodies vs the terrain: the ``illegal_reset_contact`` termination (a reset into the terrain)
        ContactSensorCfg(
            name="contact_forces",
            primary=ContactMatch(mode="body", pattern=".*", entity="robot"),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("force",),
            reduce="netforce",
            history_length=4,  # one policy step (decimation 4) of physics substeps
        ),
        ContactSensorCfg(
            name="acp_contact_forces",
            primary=ContactMatch(mode="body", pattern=CONTACT_LINKS, entity="robot"),
            secondary=ContactMatch(mode="body", pattern="terrain"),  # terrain contacts only (as the reference labels): no self-contact of the links
            fields=("force",),
            reduce="netforce",
            # one policy step (decimation 4) of physics substeps; measured contact = max over the window of |F_z| > 1 N
            history_length=4,
        ),
        # contact point of the strongest contact of each contact link (the position of the hindsight labels of the planner)
        ContactSensorCfg(
            name="acp_contact_points",
            primary=ContactMatch(mode="body", pattern=CONTACT_LINKS, entity="robot"),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found", "force", "pos"),
            reduce="maxforce",
            num_slots=1,
        ),
        # actor / planner scan: 17 x 11 grid from the estimated pelvis pose (odometry delay and drift), as in the authors'
        # deploy stack (g1_rsl_cpp): rays start 5 m above the pelvis, straight down, yaw of the pelvis
        acp_mdp.PelvisRayCastSensorCfg(
            name="height_scanner",
            frame=ObjRef(type="body", name="pelvis", entity="robot"),
            pattern=GridPatternCfg(resolution=0.1, size=(1.6, 1.0)),
            ray_alignment="yaw",
            max_distance=10.0,
            exclude_parent_body=True,
            # terrain only: the terrain geoms are moved to geom group 0 (``_edit_acp_scene_spec``), the robot geoms are
            # in groups 2 / 3
            include_geom_groups=(0,),
            ray_offset=5.0,
            use_estimated_pose=True,
            debug_vis=False,
        ),
        # critic scan: the same grid from the true pelvis pose (no delay, no drift)
        acp_mdp.PelvisRayCastSensorCfg(
            name="critic_height_scanner",
            frame=ObjRef(type="body", name="pelvis", entity="robot"),
            pattern=GridPatternCfg(resolution=0.1, size=(1.6, 1.0)),
            ray_alignment="yaw",
            max_distance=10.0,
            exclude_parent_body=True,
            include_geom_groups=(0,),
            ray_offset=5.0,
            use_estimated_pose=False,
            debug_vis=False,
        ),
        motion_reference,
    )


def make_commands() -> dict:
    """Reference contacts (Stage 1 contact targets, reference labels) and the goal command."""
    return {
        CONTACTS_COMMAND: acp_mdp.AcpReferenceContactsCfg(),
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
    """Stage 1 rewards of paper Table 3 (per-step reward = dt * sum_i w_i r_i); the regularizers sum over the 23 controlled
    joints."""
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
        # Stage 1: weight 0 (evaluated against the reference contact targets as a diagnostic); Stage 2 / 3 weight 1.5.
        "track_planned_contact": RewTermCfg(
            func=acp_mdp.track_planned_contact,
            weight=0.0,
            params={**command, "sigma": 1.0},
        ),
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
            params={"asset_cfg": action_joints, "max_value": 30.0},
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
            params={**command, "max_value": 1e4},
        ),
    }


def make_events() -> dict[str, EventTermCfg]:
    """Domain randomization of paper Table S2, the reset to the reference and the CSI start phase."""
    return {
        "match_motion_ref_with_scene": EventTermCfg(
            func=instinct_mdp.match_motion_ref_with_scene,
            mode="startup",
            params={"motion_ref_cfg": SceneEntityCfg("motion_reference")},
        ),
        # friction U(0.3, 1.2) (one value per env for all robot geoms; ground friction see TERRAIN_FRICTION).
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
        # CSI: the clip start phase must be set after the motion reference reset and before the robot reset.
        "csi_start_phase": EventTermCfg(
            func=acp_mdp.reset_motion_start_phase,
            mode="reset",
            params={"curriculum_name": CURRICULUM_NAME},
        ),
        # base pose / velocity from the reference, base height + U(0, 0.05) m, joint positions +-0.2 rad
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
                "randomize_pose_range": {"z": (0.0, 0.05)},
                "randomize_velocity_range": {},
                "randomize_joint_pos_range": (-0.2, 0.2),
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
    """Curriculum State Initialization (``init_phase_mode``: "csi" Stage 1, "uniform" Stage 2, "start" evaluation)."""
    return {
        CURRICULUM_NAME: CurriculumTermCfg(
            func=acp_mdp.CurriculumStateInit,
            params={
                "mode": "csi",
                "init_phase": 0.9,
                "step": 0.05,
                "success_phase": 0.99,
                "max_phase": 0.99,
                "uniform_max_phase": 0.98,
            },
        ),
    }


def make_terminations() -> dict[str, DoneTermCfg]:
    """ACP-unspecified: the perceptive shadowing terminations (thresholds included), plus the time-out at the clip end.
    The base terms compare the pelvis (the base of the authors' deploy stack) with the reference pelvis; the link term
    covers the 9 contact links."""
    return {
        "time_out": DoneTermCfg(func=mdp.time_out, time_out=True),
        "dataset_exhausted": DoneTermCfg(
            func=instinct_mdp.dataset_exhausted,
            time_out=True,
            params={"reference_cfg": SceneEntityCfg("motion_reference"), "print_reason": False},
        ),
        # InstinctLab's perceptive shadowing term: an episode whose first 2 policy steps both see > 500 N on a body other than the feet / hands
        # started inside the terrain (reset noise on a contact pose, e.g. the ladder rungs). A time-out (not a failure of the policy): bootstrapped,
        # no CSI failure, nothing to mirror in the imagination.
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
            params={"command_name": CONTACTS_COMMAND, "projected_gravity_threshold": 0.8, "z_only": False},
        ),
        "link_pos_too_far": DoneTermCfg(
            func=instinct_mdp.link_pos_far_from_ref,
            time_out=False,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "reference_cfg": SceneEntityCfg(
                    "motion_reference",
                    body_names=list(CONTACT_LINKS),
                    preserve_order=True,
                ),
                "distance_threshold": 0.25,
                "in_base_frame": False,
                "check_at_keyframe_threshold": -1,
                "height_only": True,
                "print_reason": False,
            },
        ),
    }


@dataclass(kw_only=True)
class AcpTrackEnvCfg(InstinctRlEnvCfg):
    scene: AcpSceneCfg = MISSING
    decimation: int = 4
    commands: dict = field(default_factory=make_commands)
    actions: dict = field(default_factory=make_actions)
    observations: dict = field(default_factory=make_observations)
    rewards: dict = field(default_factory=lambda: MultiRewardCfg({"rewards": make_rewards()}))
    events: dict = field(default_factory=make_events)
    curriculum: dict = field(default_factory=make_curriculum)
    terminations: dict = field(default_factory=make_terminations)

    def __post_init__(self):
        # policy 50 Hz, PD 200 Hz
        self.decimation = 4
        # ACP-unspecified: the episode lasts until the clip end (``dataset_exhausted``); this is only the upper bound
        # (the command term checks that the longest clip fits)
        self.episode_length_s = 120.0
        self.sim.mujoco.timestep = 0.005
        self.sim.mujoco.iterations = 10
        self.sim.mujoco.ls_iterations = 20
        self.sim.mujoco.ccd_iterations = 128
        self.sim.mujoco.jacobian = "sparse"
        self.sim.nconmax = 128
        self.sim.njmax = 700
        self.sim.contact_sensor_maxmatch = 128

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
        reset_robot["randomize_joint_pos_range"] = (0.0, 0.0)
        for group in self.observations.values():
            group.enable_corruption = False
        self.observations["policy"].terms["contact_targets"].params["noise"] = 0.0
        self.observations["planner"].terms["planner_height_scan"].params["z_noise_std"] = 0.0
