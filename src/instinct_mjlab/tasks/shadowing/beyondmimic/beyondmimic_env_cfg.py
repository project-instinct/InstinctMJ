from dataclasses import field, dataclass, MISSING

import mujoco
import mjlab.envs.mdp as mdp
from mjlab.entity import EntityCfg
from mjlab.managers import CurriculumTermCfg, EventTermCfg
from mjlab.managers import ObservationGroupCfg as ObsGroupCfg
from mjlab.managers import ObservationTermCfg as ObsTermCfg
from mjlab.managers import RewardTermCfg as RewTermCfg
from mjlab.managers import SceneEntityCfg
from mjlab.managers import TerminationTermCfg as DoneTermCfg
from mjlab.scene import SceneCfg as InteractiveSceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, SensorCfg
from mjlab.terrains import TerrainImporterCfg
from mjlab.utils.spec_config import MaterialCfg, TextureCfg
from mjlab.utils.noise import UniformNoiseCfg

import instinct_mjlab.envs.mdp as instinct_mdp
from instinct_mjlab.envs.manager_based_rl_env_cfg import InstinctLabRLEnvCfg
from instinct_mjlab.monitors import (
    MonitorTermCfg,
    MotionReferenceMonitorTerm,
    ShadowingJointPosMonitorTerm,
    ShadowingJointVelMonitorTerm,
    ShadowingLinkPosMonitorTerm,
    ShadowingPositionMonitorTerm,
    ShadowingRotationMonitorTerm,
)
from instinct_mjlab.motion_reference import MotionReferenceManagerCfg

_UNDESIRED_CONTACT_BODY_NAMES = (
    "pelvis",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "waist_yaw_link",
    "waist_roll_link",
    "torso_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "left_wrist_pitch_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
)


def _edit_beyondmimic_scene_spec(spec: mujoco.MjSpec) -> None:
    """Apply skybox and ground material to the scene spec."""
    ground_texture_name = "beyondmimic_groundplane"
    ground_material_name = "beyondmimic_groundplane"

    # Bright sky theme so native viewer doesn't look like a black void.
    sky_rgb_top = (0.98, 0.99, 1.0)
    sky_rgb_horizon = (0.78, 0.86, 0.95)
    # White-ish checker ground instead of the default blue checker.
    ground_rgb1 = (0.95, 0.95, 0.95)
    ground_rgb2 = (0.88, 0.88, 0.88)
    ground_mark_rgb = (0.80, 0.80, 0.80)

    existing_skybox = next(
        tex
        for tex in spec.textures
        if tex.type == mujoco.mjtTexture.mjTEXTURE_SKYBOX
    )
    existing_skybox.builtin = mujoco.mjtBuiltin.mjBUILTIN_GRADIENT
    existing_skybox.rgb1[:] = sky_rgb_top
    existing_skybox.rgb2[:] = sky_rgb_horizon
    existing_skybox.width = 512
    existing_skybox.height = 3072

    TextureCfg(
        name=ground_texture_name,
        type="2d",
        builtin="checker",
        mark="edge",
        rgb1=ground_rgb1,
        rgb2=ground_rgb2,
        markrgb=ground_mark_rgb,
        width=300,
        height=300,
    ).edit_spec(spec)
    MaterialCfg(
        name=ground_material_name,
        texuniform=True,
        texrepeat=(4, 4),
        reflectance=0.05,
        texture=ground_texture_name,
    ).edit_spec(spec)

    spec.visual.rgba.haze[:] = (0.90, 0.94, 0.98, 1.0)
    spec.visual.headlight.ambient[:] = (0.45, 0.45, 0.45)
    spec.visual.headlight.diffuse[:] = (0.75, 0.75, 0.75)

    for geom in spec.body("terrain").geoms:
        geom.material = ground_material_name


@dataclass(kw_only=True)
class BeyondMimicSceneCfg(InteractiveSceneCfg):
    """Configuration for the BeyondMimic scene with necessary scene entities as motion reference."""

    env_spacing: float = 4.0

    # terrain
    terrain: object = field(default_factory=lambda: TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        terrain_generator=None,
        physics_material=None,
        visual_material=None,
    ))

    sensors: tuple[SensorCfg, ...] = field(default_factory=lambda: _make_beyondmimic_base_scene_sensors())

    def __post_init__(self):
        self.spec_fn = _edit_beyondmimic_scene_spec


def make_beyondmimic_scene_entities(
    *,
    robot: EntityCfg,
) -> dict[str, EntityCfg]:
    """Build BeyondMimic scene entities without bridge fields."""
    # robots
    # robot reference articulation
    # motion reference is configured as a sensor cfg ("motion_reference").
    return {"robot": robot}


def make_beyondmimic_scene_entities_with_reference(
    *,
    robot: EntityCfg,
    robot_reference: EntityCfg,
) -> dict[str, EntityCfg]:
    """Build BeyondMimic scene entities for play/debug with reference robot."""
    return {
        "robot": robot,
        "robot_reference": robot_reference,
    }


def _make_beyondmimic_undesired_contact_sensor_cfg() -> ContactSensorCfg:
    return ContactSensorCfg(
        name="undesired_contact_forces",
        primary=ContactMatch(
            mode="body",
            pattern=_UNDESIRED_CONTACT_BODY_NAMES,
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        history_length=3,
        num_slots=1,
    )


def _make_beyondmimic_base_scene_sensors() -> tuple[SensorCfg, ...]:
    return (_make_beyondmimic_undesired_contact_sensor_cfg(),)


def make_beyondmimic_scene_sensors(
    *,
    motion_reference: MotionReferenceManagerCfg,
) -> tuple[SensorCfg, ...]:
    """Build BeyondMimic scene sensors without bridge fields."""
    # lights are applied in _edit_beyondmimic_scene_spec.
    return _make_beyondmimic_base_scene_sensors() + (motion_reference,)


def make_beyondmimic_commands() -> dict[str, instinct_mdp.ShadowingCommandBaseCfg]:
    """BeyondMimic command configuration following their approach."""
    return {
        "position_ref_command": instinct_mdp.PositionRefCommandCfg(
            realtime_mode=True,
            current_state_command=False,
            anchor_frame="robot",
        ),
        "rotation_ref_command": instinct_mdp.RotationRefCommandCfg(
            realtime_mode=True,
            current_state_command=False,
            in_base_frame=True,
            rotation_mode="tannorm",
        ),
        "joint_pos_ref_command": instinct_mdp.JointPosRefCommandCfg(current_state_command=False),
        "joint_vel_ref_command": instinct_mdp.JointVelRefCommandCfg(current_state_command=False),
    }


def make_beyondmimic_actions() -> dict[str, mdp.JointPositionActionCfg]:
    """Action specifications for the BeyondMimic MDP."""
    return {
        "joint_pos": mdp.JointPositionActionCfg(
            entity_name="robot",
            actuator_names=(".*",),
        ),
    }


def make_beyondmimic_observations() -> dict[str, ObsGroupCfg]:
    """BeyondMimic observation configuration following their approach."""
    
    # Policy observations
    actor_terms = {
        "joint_pos_ref": ObsTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "joint_pos_ref_command"},
        ),
        "joint_vel_ref": ObsTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "joint_vel_ref_command"},
        ),
        "position_ref": ObsTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "position_ref_command"},
            noise=UniformNoiseCfg(n_min=-0.25, n_max=0.25),
        ),
        "rotation_ref": ObsTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "rotation_ref_command"},
            noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05),
        ),
        "base_lin_vel": ObsTermCfg(
            func=mdp.base_lin_vel,
            noise=UniformNoiseCfg(n_min=-0.5, n_max=0.5),
        ),
        "base_ang_vel": ObsTermCfg(
            func=mdp.base_ang_vel,
            noise=UniformNoiseCfg(n_min=-0.2, n_max=0.2),
        ),
        "joint_pos": ObsTermCfg(
            func=mdp.joint_pos_rel,
            noise=UniformNoiseCfg(n_min=-0.01, n_max=0.01),
        ),
        "joint_vel": ObsTermCfg(
            func=mdp.joint_vel_rel,
            noise=UniformNoiseCfg(n_min=-0.5, n_max=0.5),
        ),
        "last_action": ObsTermCfg(func=mdp.last_action),
    }

    # Critic observations
    critic_terms = {
        "joint_pos_ref": ObsTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "joint_pos_ref_command"},
        ),
        "joint_vel_ref": ObsTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "joint_vel_ref_command"},
        ),
        "position_ref": ObsTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "position_ref_command"},
        ),
        "rotation_ref": ObsTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "rotation_ref_command"},
        ),
        "link_pos": ObsTermCfg(
            func=instinct_mdp.link_pos_b,
            params={
                "asset_cfg": SceneEntityCfg(
                    name="robot",
                    body_names=MISSING,
                    preserve_order=True,
                ),
            },
        ),
        "link_rot": ObsTermCfg(
            func=instinct_mdp.link_tannorm_b,
            params={
                "asset_cfg": SceneEntityCfg(
                    name="robot",
                    body_names=MISSING,
                    preserve_order=True,
                ),
            },
        ),
        "base_lin_vel": ObsTermCfg(func=mdp.base_lin_vel),
        "base_ang_vel": ObsTermCfg(func=mdp.base_ang_vel),
        "joint_pos": ObsTermCfg(func=mdp.joint_pos_rel),
        "joint_vel": ObsTermCfg(func=mdp.joint_vel_rel),
        "last_action": ObsTermCfg(func=mdp.last_action),
    }

    return {
        "actor": ObsGroupCfg(
            terms=actor_terms,
            enable_corruption=True,
            concatenate_terms=False,
        ),
        "critic": ObsGroupCfg(
            terms=critic_terms,
            enable_corruption=False,
            concatenate_terms=False,
        ),
    }


def make_beyondmimic_rewards() -> dict[str, RewTermCfg | None]:
    """BeyondMimic reward terms following their approach."""
    return {
        "base_position_imitation_gauss": RewTermCfg(
            func=instinct_mdp.base_position_imitation_gauss,
            weight=0.5,
            params={
                "std": 0.3,
            },
        ),
        "base_rot_imitation_gauss": RewTermCfg(
            func=instinct_mdp.base_rot_imitation_gauss,
            weight=0.5,
            params={
                "std": 0.4,
                "difference_type": "axis_angle",
            },
        ),
        "motion_body_pos": RewTermCfg(
            func=instinct_mdp.link_pos_imitation_gauss,
            weight=1.0,
            params={
                "combine_method": "mean_prod",
                "in_base_frame": False,
                "in_relative_world_frame": True,
                "std": 0.3,
            },
        ),
        "motion_body_ori": RewTermCfg(
            func=instinct_mdp.link_rot_imitation_gauss,
            weight=1.0,
            params={
                "combine_method": "mean_prod",
                "in_base_frame": False,
                "in_relative_world_frame": True,
                "std": 0.4,
            },
        ),
        "motion_body_lin_vel": RewTermCfg(
            func=instinct_mdp.link_lin_vel_imitation_gauss,
            weight=1.0,
            params={
                "combine_method": "mean_prod",
                "std": 1.0,
            },
        ),
        "motion_body_ang_vel": RewTermCfg(
            func=instinct_mdp.link_ang_vel_imitation_gauss,
            weight=1.0,
            params={
                "combine_method": "mean_prod",
                "std": 3.14,
            },
        ),
        "action_rate_l2": RewTermCfg(func=mdp.action_rate_l2, weight=-0.1),
        "joint_limit": RewTermCfg(
            func=mdp.joint_pos_limits,
            weight=-10.0,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
        ),
        "undesired_contacts": RewTermCfg(
            func=instinct_mdp.undesired_contacts,
            weight=-0.1,
            params={
                "sensor_name": "undesired_contact_forces",
                "threshold": 1.0,
            },
        ),
    }


def make_beyondmimic_events() -> dict[str, EventTermCfg]:
    """BeyondMimic events config such as termination conditions."""
    return {
        "physics_material": EventTermCfg(
            func=instinct_mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                "static_friction_range": (0.3, 1.6),
                "dynamic_friction_range": (0.3, 1.2),
            },
        ),
        "add_joint_default_pos": EventTermCfg(
            func=instinct_mdp.randomize_default_joint_pos,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                "offset_distribution_params": (-0.01, 0.01),
                "operation": "add",
                "distribution": "uniform",
            },
        ),
        "base_com": EventTermCfg(
            func=instinct_mdp.randomize_rigid_body_coms,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
                "coms_x_distribution_params": (-0.025, 0.025),
                "coms_y_distribution_params": (-0.05, 0.05),
                "coms_z_distribution_params": (-0.05, 0.05),
                "distribution": "uniform",
            },
        ),
        "push_robot": EventTermCfg(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(1.0, 3.0),
            params={
                "velocity_range": {
                    "x": (-0.5, 0.5),
                    "y": (-0.5, 0.5),
                    "z": (-0.2, 0.2),
                    "roll": (-0.52, 0.52),
                    "pitch": (-0.52, 0.52),
                    "yaw": (-0.78, 0.78),
                },
            },
        ),
        "match_motion_ref_with_scene": EventTermCfg(
            func=instinct_mdp.match_motion_ref_with_scene,
            mode="startup",
            params={
                "motion_ref_cfg": SceneEntityCfg("motion_reference"),
            },
        ),
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
                # Pose randomization (+-5cm position, +-6degrees rotation)
                "randomize_pose_range": {
                    "x": (-0.05, 0.05),
                    "y": (-0.05, 0.05),
                    "z": (-0.01, 0.01),
                    "roll": (-0.1, 0.1),
                    "pitch": (-0.1, 0.1),
                    "yaw": (-0.2, 0.2),
                },
                # Velocity randomization (+-0.1 m/s linear, +-0.1 rad/s angular)
                "randomize_velocity_range": {
                    "x": (-0.5, 0.5),
                    "y": (-0.5, 0.5),
                    "z": (-0.2, 0.2),
                    "roll": (-0.52, 0.52),
                    "pitch": (-0.52, 0.52),
                    "yaw": (-0.78, 0.78),
                },
                # Joint position randomization (+-0.1 rad)
                "randomize_joint_pos_range": (-0.1, 0.1),
            },
        ),
        "bin_fail_counter_smoothing": EventTermCfg(
            func=instinct_mdp.beyondmimic_bin_fail_counter_smoothing,
            mode="interval",
            interval_range_s=(0.02, 0.02),  # every environment step
            params={
                "curriculum_name": "beyond_adaptive_sampling",
            },
        ),
    }


def make_beyondmimic_curriculum() -> dict[str, CurriculumTermCfg]:
    """BeyondMimic curriculum terms for the MDP."""
    return {
        "beyond_adaptive_sampling": CurriculumTermCfg(  # type: ignore
            func=instinct_mdp.BeyondMimicAdaptiveWeighting,
        ),
    }


def make_beyondmimic_terminations() -> dict[str, DoneTermCfg]:
    """BeyondMimic termination terms for the MDP."""
    return {
        "time_out": DoneTermCfg(func=mdp.time_out, time_out=True),
        "base_pos_too_far": DoneTermCfg(
            func=instinct_mdp.pos_far_from_ref,
            time_out=False,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "reference_cfg": SceneEntityCfg("motion_reference"),
                "distance_threshold": 0.25,
                "check_at_keyframe_threshold": -1,
                "print_reason": False,
                "height_only": True,
            },
        ),
        "base_pg_too_far": DoneTermCfg(
            func=instinct_mdp.projected_gravity_far_from_ref,
            time_out=False,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "reference_cfg": SceneEntityCfg("motion_reference"),
                "projected_gravity_threshold": 0.8,  # distance on z-axis of projected gravity
                "check_at_keyframe_threshold": -1,
                "print_reason": False,
            },
        ),
        "link_pos_too_far": DoneTermCfg(
            func=instinct_mdp.link_pos_far_from_ref,
            time_out=False,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "reference_cfg": SceneEntityCfg(
                    "motion_reference",
                    body_names=[
                        "left_ankle_roll_link",
                        "right_ankle_roll_link",
                        "left_wrist_yaw_link",
                        "right_wrist_yaw_link",
                    ],
                    preserve_order=True,
                ),
                "distance_threshold": 0.25,
                "in_base_frame": False,
                "check_at_keyframe_threshold": -1,
                "height_only": True,
                "print_reason": False,
            },
        ),
        "dataset_exhausted": DoneTermCfg(
            func=instinct_mdp.dataset_exhausted,
            time_out=True,
            params={
                "reference_cfg": SceneEntityCfg("motion_reference"),
                "print_reason": False,
            },
        ),
        "out_of_border": DoneTermCfg(
            func=instinct_mdp.terrain_out_of_bounds,
            time_out=True,
            params={"asset_cfg": SceneEntityCfg("robot"), "print_reason": False, "distance_buffer": 0.1},
        ),
    }


def make_beyondmimic_monitors() -> dict[str, MonitorTermCfg]:
    """BeyondMimic monitor configuration."""
    return {
        "dataset": MonitorTermCfg(
            func=MotionReferenceMonitorTerm,
            params=dict(
                asset_cfg=SceneEntityCfg("motion_reference"),
                sample_stat_interval=500,
                top_n_samples=5,
            ),
        ),
        "shadowing_position": MonitorTermCfg(
            func=ShadowingPositionMonitorTerm,
            params=dict(
                robot_cfg=SceneEntityCfg("robot"),
                motion_reference_cfg=SceneEntityCfg("motion_reference"),
                in_base_frame=True,
                check_at_keyframe_threshold=0.03,
            ),
        ),
        "shadowing_rotation": MonitorTermCfg(
            func=ShadowingRotationMonitorTerm,
            params=dict(
                robot_cfg=SceneEntityCfg("robot"),
                motion_reference_cfg=SceneEntityCfg("motion_reference"),
                masking=True,
            ),
        ),
        "shadowing_joint_pos": MonitorTermCfg(
            func=ShadowingJointPosMonitorTerm,
            params=dict(
                robot_cfg=SceneEntityCfg("robot"),
                motion_reference_cfg=SceneEntityCfg("motion_reference"),
                masking=True,
            ),
        ),
        "shadowing_joint_vel": MonitorTermCfg(
            func=ShadowingJointVelMonitorTerm,
            params=dict(
                robot_cfg=SceneEntityCfg("robot"),
                motion_reference_cfg=SceneEntityCfg("motion_reference"),
                masking=True,
            ),
        ),
        "shadowing_link_pos": MonitorTermCfg(
            func=ShadowingLinkPosMonitorTerm,
            params=dict(
                robot_cfg=SceneEntityCfg("robot"),
                motion_reference_cfg=SceneEntityCfg("motion_reference"),
                in_base_frame=True,
                masking=True,
            ),
        ),
    }


@dataclass(kw_only=True)
class BeyondMimicEnvCfg(InstinctLabRLEnvCfg):
    """Configuration for the BeyondMimic environment."""

    scene: BeyondMimicSceneCfg = field(default_factory=lambda: BeyondMimicSceneCfg(num_envs=4096))
    commands: dict = field(default_factory=make_beyondmimic_commands)
    actions: dict = field(default_factory=make_beyondmimic_actions)
    observations: dict = field(default_factory=make_beyondmimic_observations)
    rewards: dict = field(default_factory=make_beyondmimic_rewards)
    events: dict = field(default_factory=make_beyondmimic_events)
    curriculum: dict = field(default_factory=make_beyondmimic_curriculum)
    terminations: dict = field(default_factory=make_beyondmimic_terminations)
    monitors: dict = field(default_factory=make_beyondmimic_monitors)

    def __post_init__(self):
        # general settings
        self.decimation = 4
        self.episode_length_s = 10.0
        # simulation settings — constrain collision buffers to avoid GPU OOM
        # BeyondMimic monitors many body-ground contacts, so nconmax needs to be
        # larger than tracking (35) but bounded to prevent mujoco_warp from
        # auto-allocating enormous EPA buffers.
        self.sim.nconmax = 100
        self.sim.njmax = 300
        self.sim.mujoco.timestep = 1.0 / 50.0 / self.decimation
        self.sim.mujoco.iterations = 10
        self.sim.mujoco.ls_iterations = 20
        # Keep CCD iterations moderate to avoid large EPA buffers at 4096 envs.
        self.sim.mujoco.ccd_iterations = 80

        # All managers are already dicts, no conversion needed!
        self.run_name = "BeyondMimic"
