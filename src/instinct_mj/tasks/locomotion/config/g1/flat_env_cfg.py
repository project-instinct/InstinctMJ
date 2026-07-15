"""mjlab-native G1 locomotion environment config builders."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

import mjlab.envs.mdp as mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import CurriculumTermCfg as CurrTerm
from mjlab.managers import EventTermCfg as Event
from mjlab.managers import ObservationGroupCfg as ObsGroup
from mjlab.managers import ObservationTermCfg as ObsTerm
from mjlab.managers import RewardTermCfg as RewTerm
from mjlab.managers import SceneEntityCfg
from mjlab.managers import TerminationTermCfg as DoneTerm
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer.viewer_config import ViewerConfig

import instinct_mj.envs.mdp as instinct_mdp
import instinct_mj.tasks.locomotion.mdp as locomotion_mdp
from instinct_mj.assets.unitree_g1 import (
    G1_29DOF_TORSOBASE_POPSICLE_CFG,
    beyondmimic_action_scale,
    beyondmimic_g1_29dof_actuator_cfgs,
)
from instinct_mj.envs.manager_based_rl_env_cfg import InstinctLabRLEnvCfg

G1_CFG = G1_29DOF_TORSOBASE_POPSICLE_CFG


# ============================================================================
# Scene Configuration
# ============================================================================


@dataclass(kw_only=True)
class G1LocomotionSceneCfg(SceneCfg):
    """G1 locomotion scene in mjlab-native dataclass form."""

    def __post_init__(self) -> None:
        robot_cfg = copy.deepcopy(G1_CFG)
        robot_cfg.articulation.actuators = copy.deepcopy(beyondmimic_g1_29dof_actuator_cfgs)
        feet_contact_forces = ContactSensorCfg(
            name="feet_contact_forces",
            primary=ContactMatch(
                mode="body",
                pattern=("left_ankle_roll_link", "right_ankle_roll_link"),
                entity="robot",
            ),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found", "force"),
            reduce="maxforce",
            track_air_time=True,
            history_length=3,
        )
        base_contact_forces = ContactSensorCfg(
            name="base_contact_forces",
            primary=ContactMatch(
                mode="body",
                pattern=(
                    "torso_link",
                    ".*_shoulder_.*",
                    ".*_elbow_.*",
                    ".*_wrist_.*",
                    ".*_hip_.*",
                    ".*_knee_.*",
                ),
                entity="robot",
            ),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found", "force"),
            reduce="maxforce",
            track_air_time=False,
            history_length=3,
        )
        self.terrain = TerrainEntityCfg(terrain_type="plane")
        self.entities = {"robot": robot_cfg}
        self.sensors = (feet_contact_forces, base_contact_forces)
        self.extent = 2.0


def _scene_cfg(play: bool) -> G1LocomotionSceneCfg:
    return G1LocomotionSceneCfg(
        num_envs=1 if play else 4096,
        env_spacing=2.5,
    )


# ============================================================================
# Actions Configuration
# ============================================================================


def _actions_cfg() -> dict[str, JointPositionActionCfg]:
    return {
        "joint_pos": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=(".*",),
            scale=0.5,
            use_default_offset=True,
        )
    }


# ============================================================================
# Commands Configuration
# ============================================================================


def _commands_cfg() -> dict[str, UniformVelocityCommandCfg]:
    return {
        "base_velocity": UniformVelocityCommandCfg(
            entity_name="robot",
            resampling_time_range=(10.0, 10.0),
            rel_standing_envs=0.2,
            rel_heading_envs=0.5,
            heading_command=True,
            heading_control_stiffness=0.5,
            debug_vis=True,
            ranges=UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(-0.5, 1.0),
                lin_vel_y=(-0.5, 0.5),
                ang_vel_z=(-1.5, 1.5),
                heading=(-math.pi, math.pi),
            ),
        ),
    }


# ============================================================================
# Observations Configuration
# ============================================================================


def _observations_cfg() -> dict[str, ObsGroup]:
    policy_terms = {
        "base_ang_vel": ObsTerm(func=instinct_mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2)),
        "projected_gravity": ObsTerm(
            func=instinct_mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        ),
        "velocity_commands": ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"}),
        "joint_pos": ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01)),
        "joint_vel": ObsTerm(func=locomotion_mdp.joint_vel, noise=Unoise(n_min=-1.5, n_max=1.5)),
        "actions": ObsTerm(func=instinct_mdp.last_action),
    }
    critic_terms = {
        "base_lin_vel": ObsTerm(func=mdp.base_lin_vel),
        "base_ang_vel": ObsTerm(func=instinct_mdp.base_ang_vel),
        "projected_gravity": ObsTerm(func=instinct_mdp.projected_gravity),
        "velocity_commands": ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"}),
        "joint_pos": ObsTerm(func=mdp.joint_pos_rel),
        "joint_vel": ObsTerm(func=locomotion_mdp.joint_vel),
        "actions": ObsTerm(func=instinct_mdp.last_action),
    }
    return {
        "policy": ObsGroup(
            terms=policy_terms,
            enable_corruption=True,
            concatenate_terms=False,
        ),
        "critic": ObsGroup(
            terms=critic_terms,
            enable_corruption=False,
            concatenate_terms=False,
        ),
    }


# ============================================================================
# Rewards Configuration
# ============================================================================


def _rewards_cfg() -> dict[str, RewTerm]:
    return {
        "termination_penalty": RewTerm(func=mdp.is_terminated, weight=-200.0),
        "track_lin_vel_xy_exp": RewTerm(
            func=locomotion_mdp.track_lin_vel_xy_yaw_frame_exp,
            weight=1.0,
            params={"command_name": "base_velocity", "std": 0.5},
        ),
        "track_ang_vel_z_exp": RewTerm(
            func=locomotion_mdp.track_ang_vel_z_world_exp,
            weight=1.0,
            params={"command_name": "base_velocity", "std": 0.5},
        ),
        "feet_air_time": RewTerm(
            func=locomotion_mdp.feet_air_time_positive_biped,
            weight=1.0,
            params={
                "command_name": "base_velocity",
                "sensor_name": "feet_contact_forces",
                "threshold": 0.5,
            },
        ),
        "feet_slide": RewTerm(
            func=instinct_mdp.contact_slide,
            weight=-0.1,
            params={
                "sensor_cfg": SceneEntityCfg("feet_contact_forces"),
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=("left_ankle_roll_link", "right_ankle_roll_link"),
                ),
            },
        ),
        # "base_height_l2": RewTerm(func=mdp.base_height_l2, weight=-5.0, params={"target_height": 0.8}),
        "flat_orientation_l2": RewTerm(func=mdp.flat_orientation_l2, weight=-1.0),
        "stand_still": RewTerm(func=locomotion_mdp.stand_still, weight=-0.8, params={"command_name": "base_velocity"}),
        "dof_pos_limits": RewTerm(
            func=mdp.joint_pos_limits,
            weight=-1.0,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*_ankle_pitch_joint", ".*_ankle_roll_joint"))},
        ),
        "joint_deviation_hip": RewTerm(
            func=locomotion_mdp.joint_deviation_l1,
            weight=-0.1,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*_hip_yaw_joint", ".*_hip_roll_joint"))},
        ),
        "joint_deviation_arms": RewTerm(
            func=locomotion_mdp.joint_deviation_l1,
            weight=-0.1,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=(
                        ".*_shoulder_pitch_joint",
                        ".*_shoulder_roll_joint",
                        ".*_shoulder_yaw_joint",
                        ".*_elbow_joint",
                        ".*_wrist_roll_joint",
                        ".*_wrist_pitch_joint",
                        ".*_wrist_yaw_joint",
                    ),
                )
            },
        ),
        "joint_deviation_torso": RewTerm(
            func=locomotion_mdp.joint_deviation_l1,
            weight=-0.1,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=("waist_.*",))},
        ),
        "joint_deviation_knee": RewTerm(
            func=locomotion_mdp.joint_deviation_l1,
            weight=-0.05,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*_knee_joint",))},
        ),
        "lin_vel_z_l2": RewTerm(func=locomotion_mdp.lin_vel_z_l2, weight=-0.1),
        "action_rate_l2": RewTerm(func=mdp.action_rate_l2, weight=-0.05),
        "dof_acc_l2": RewTerm(
            func=mdp.joint_acc_l2,
            weight=-2.0e-7,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*_hip_.*", ".*_knee_joint"))},
        ),
        "dof_torques_l2": RewTerm(
            func=instinct_mdp.joint_torques_l2,
            weight=-4.0e-6,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*_hip_.*", ".*_knee_joint"))},
        ),
    }


# ============================================================================
# Terminations Configuration
# ============================================================================


def _terminations_cfg() -> dict[str, DoneTerm]:
    return {
        "time_out": DoneTerm(func=mdp.time_out, time_out=True),
        "base_contact": DoneTerm(
            func=locomotion_mdp.illegal_contact,
            time_out=False,
            params={
                "sensor_name": "base_contact_forces",
                "threshold": 1.0,
            },
        ),
    }


# ============================================================================
# Events Configuration
# ============================================================================


def _events_cfg() -> dict[str, Event]:
    return {
        "physics_material": Event(
            func=instinct_mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", geom_names=(".*",)),
                "static_friction_range": (0.25, 0.8),
                "dynamic_friction_range": (0.2, 0.6),
                "restitution_range": (0.0, 0.8),
                "num_buckets": 64,
            },
        ),
        "add_base_mass": Event(
            func=locomotion_mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
                "mass_distribution_params": (-5.0, 5.0),
                "operation": "add",
            },
        ),
        "base_external_force_torque": Event(
            func=mdp.apply_external_force_torque,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
                "force_range": (0.0, 0.0),
                "torque_range": (-0.0, 0.0),
            },
        ),
        "reset_base": Event(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
                "velocity_range": {
                    "x": (-0.5, 0.5),
                    "y": (-0.5, 0.5),
                    "z": (-0.1, 0.1),
                    "roll": (-0.5, 0.5),
                    "pitch": (-0.5, 0.5),
                    "yaw": (-0.5, 0.5),
                },
            },
        ),
        "reset_robot_joints": Event(
            func=locomotion_mdp.reset_joints_by_scale,
            mode="reset",
            params={
                "position_range": (0.8, 1.2),
                "velocity_range": (-1.0, 1.0),
            },
        ),
        "push_robot": Event(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(10.0, 15.0),
            params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
        ),
    }


# ============================================================================
# Curriculum Configuration
# ============================================================================


def _curriculum_cfg() -> dict[str, CurrTerm]:
    # "terrain_levels": CurrTerm(func=locomotion_mdp.terrain_levels_vel),
    return {}


@dataclass(kw_only=True)
class G1LocomotionFlatEnvCfg(InstinctLabRLEnvCfg):
    scene: G1LocomotionSceneCfg = field(default_factory=lambda: _scene_cfg(play=False))
    actions: dict = field(default_factory=_actions_cfg)
    commands: dict = field(default_factory=_commands_cfg)
    observations: dict = field(default_factory=_observations_cfg)
    rewards: dict = field(default_factory=lambda: {"rewards": _rewards_cfg()})
    terminations: dict = field(default_factory=_terminations_cfg)
    events: dict = field(default_factory=_events_cfg)
    curriculum: dict = field(default_factory=_curriculum_cfg)
    monitors: dict = field(default_factory=dict)
    viewer: ViewerConfig = field(
        default_factory=lambda: ViewerConfig(
            lookat=(0.0, 0.0, 0.0),
            distance=2.9,
            elevation=-10.0,
            azimuth=45.0,
            origin_type=ViewerConfig.OriginType.ASSET_ROOT,
            entity_name="robot",
        )
    )
    sim: SimulationCfg = field(
        default_factory=lambda: SimulationCfg(
            mujoco=MujocoCfg(
                timestep=0.005,
                solver="newton",
                iterations=10,
                ls_iterations=20,
                ccd_iterations=500,
            ),
        )
    )
    decimation: int = 4
    episode_length_s: float = 20.0

    def __post_init__(self) -> None:
        # All managers are already dicts, no conversion needed!
        self.sim.njmax = 300
        joint_pos_action: JointPositionActionCfg = self.actions["joint_pos"]
        joint_pos_action.scale = copy.deepcopy(beyondmimic_action_scale)
        reward_terms = self.rewards["rewards"]
        feet_air_time = reward_terms.get("feet_air_time")
        stand_still = reward_terms.get("stand_still")
        action_rate_l2 = reward_terms.get("action_rate_l2")
        joint_deviation_knee = reward_terms.get("joint_deviation_knee")
        self.run_name = "".join(
            [
                "G1Flat",
                f"_feetAirTime{feet_air_time.weight:.2f}" if feet_air_time is not None else "",
                f"_standStill{-stand_still.weight:.2f}" if stand_still is not None else "",
                f"_actionRate{-action_rate_l2.weight:.2f}" if action_rate_l2 is not None else "",
                (
                    f"_jointDeviationKnee{-joint_deviation_knee.weight:.2f}"
                    if joint_deviation_knee is not None
                    else ""
                ),
            ]
        )


def instinct_g1_locomotion_flat_env_cfg(*, play: bool = False) -> G1LocomotionFlatEnvCfg:
    """Build locomotion config in mjlab-native manager dict style."""
    cfg = G1LocomotionFlatEnvCfg()
    if play:
        cfg.scene = _scene_cfg(play=True)
        cfg.sim.mujoco.ccd_iterations = 50
        base_velocity_cmd: UniformVelocityCommandCfg = cfg.commands["base_velocity"]
        base_velocity_cmd.resampling_time_range = (2.0, 2.0)
        cfg.events["base_external_force_torque"] = None
        cfg.events["push_robot"] = None
    return cfg
