from copy import deepcopy
from dataclasses import dataclass, field
import numpy as np
import os
import yaml
from functools import partial

import mjlab.envs.mdp as mdp
import mjlab.sim as sim_utils
from mjlab.managers import CurriculumTermCfg, EventTermCfg
from mjlab.managers import ObservationGroupCfg as ObsGroupCfg
from mjlab.managers import ObservationTermCfg as ObsTermCfg
from mjlab.managers import SceneEntityCfg
from mjlab.managers import TerminationTermCfg as DoneTermCfg
from mjlab.utils.noise import UniformNoiseCfg
from mjlab.viewer.viewer_config import ViewerConfig

import instinct_mjlab.envs.mdp as instinct_mdp
import instinct_mjlab.tasks.shadowing.mdp as shadowing_mdp
import instinct_mjlab.tasks.shadowing.perceptive.perceptive_env_cfg as perceptual_cfg
from instinct_mjlab.assets.unitree_g1 import (
    G1_29DOF_TORSOBASE_POPSICLE_CFG,
    G1_29Dof_TorsoBase_symmetric_augmentation_joint_mapping,
    G1_29Dof_TorsoBase_symmetric_augmentation_joint_reverse_buf,
    G1_MJCF_PATH,
    beyondmimic_action_scale,
    beyondmimic_g1_29dof_actuator_cfgs,
    beyondmimic_g1_29dof_delayed_actuator_cfgs,
)
from instinct_mjlab.monitors import ActuatorMonitorTerm, MonitorTermCfg, ShadowingBasePosMonitorTerm
from instinct_mjlab.motion_reference import MotionReferenceManagerCfg
from instinct_mjlab.motion_reference.motion_files.aistpp_motion_cfg import AistppMotionCfg as AistppMotionCfgBase
from instinct_mjlab.motion_reference.motion_files.amass_motion_cfg import AmassMotionCfg as AmassMotionCfgBase
from instinct_mjlab.motion_reference.motion_files.terrain_motion_cfg import TerrainMotionCfg as TerrainMotionCfgBase
from instinct_mjlab.motion_reference.utils import motion_interpolate_bilinear
from instinct_mjlab.utils.motion_validation import resolve_datasets_root

G1_CFG = G1_29DOF_TORSOBASE_POPSICLE_CFG
PROPRIO_HISTORY_LENGTH = 8
_DATASETS_ROOT = resolve_datasets_root()

MOTION_FOLDER = (
    str(_DATASETS_ROOT / "20251116_50cm_kneeClimbStep1")
    # "~/Datasets/NoKov-Marslab-Motions-instinctnpz/20251116_50cm_kneeClimbStep1/20251106_diveroll4_roadRamp_noWall"
)

@dataclass(kw_only=True)
class TerrainMotionCfg(TerrainMotionCfgBase):
    path: object = field(default_factory=lambda: os.path.expanduser(MOTION_FOLDER))

    metadata_yaml: object = field(default_factory=lambda: os.path.expanduser(f"{MOTION_FOLDER}/metadata.yaml"))

    max_origins_per_motion: int = 49


    ensure_link_below_zero_ground: bool = False

    motion_start_from_middle_range: list = field(default_factory=lambda: [0.0, 0.0])

    motion_start_height_offset: float = 0.0

    motion_bin_length_s: float = 1.0

    buffer_device: str = "output_device"

    motion_interpolate_func: object = field(default_factory=lambda: motion_interpolate_bilinear)

    velocity_estimation_method: str = "frontbackward"

    env_starting_stub_sampling_strategy: str = "concat_motion_bins"


motion_reference_cfg = MotionReferenceManagerCfg(
    name="motion_reference",
    entity_name="robot",
    robot_model_path=G1_MJCF_PATH,
    link_of_interests=[
        "pelvis",
        "torso_link",
        "left_shoulder_roll_link",
        "right_shoulder_roll_link",
        "left_elbow_link",
        "right_elbow_link",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
        "left_hip_roll_link",
        "right_hip_roll_link",
        "left_knee_link",
        "right_knee_link",
        "left_ankle_roll_link",
        "right_ankle_roll_link",
    ],
    symmetric_augmentation_link_mapping=None,
    symmetric_augmentation_joint_mapping=None,
    symmetric_augmentation_joint_reverse_buf=None,
    frame_interval_s=0.1,
    update_period=0.02,
    num_frames=10,
    data_start_from="current_time",
    # set the robot_reference directly at where they are in the scene
    # DO NOT FORGET to change this when in actual training
    visualizing_robot_offset=(2.0, 0.0, 0.0),
    visualizing_robot_from="reference_frame",
    visualizing_marker_types=["relative_links", "links"],
    motion_buffers={
        "TerrainMotion": TerrainMotionCfg(),
    },
    mp_split_method="None",
)


def _make_motion_reference_cfg(*, debug_vis: bool = False) -> MotionReferenceManagerCfg:
    cfg = deepcopy(motion_reference_cfg)
    cfg.debug_vis = debug_vis
    cfg.reference_entity_name = "robot_reference" if debug_vis else None
    return cfg


def _apply_motion_matched_terrain_source(
    scene: perceptual_cfg.PerceptiveShadowingSceneCfg,
    *,
    default_path: str,
    default_metadata_yaml: str,
) -> None:
    """Bind perceptive terrain generator to motion-matched source."""
    terrain_generator = scene.terrain.terrain_generator
    terrain_cfg = perceptual_cfg.get_motion_matched_subterrain_cfg(
        terrain_generator.sub_terrains
    )
    terrain_cfg.path = default_path
    terrain_cfg.metadata_yaml = default_metadata_yaml


def make_vae_observations() -> dict[str, ObsGroupCfg]:
    """Observation specifications for the perceptive VAE MDP."""

    # Policy observations
    policy_terms = {
        "depth_image": ObsTermCfg(
            func=instinct_mdp.visualizable_image,
            # params={"sensor_cfg": SceneEntityCfg("camera"), "data_type": "distance_to_image_plane"},
            params={
                "sensor_cfg": SceneEntityCfg("camera"),
                "data_type": "distance_to_image_plane_noised_history",
                "history_skip_frames": 2,
            },
        ),

        # proprioception
        "projected_gravity": ObsTermCfg(
            func=mdp.projected_gravity,
            noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05),
            history_length=PROPRIO_HISTORY_LENGTH,
        ),
        # base_lin_vel = ObsTermCfg(func=mdp.base_lin_vel)
        "base_ang_vel": ObsTermCfg(
            func=mdp.base_ang_vel,
            noise=UniformNoiseCfg(n_min=-0.2, n_max=0.2),
            history_length=PROPRIO_HISTORY_LENGTH,
        ),
        "joint_pos": ObsTermCfg(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
            },
            noise=UniformNoiseCfg(n_min=-0.01, n_max=0.01),
            history_length=PROPRIO_HISTORY_LENGTH,
        ),
        "joint_vel": ObsTermCfg(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
            },
            noise=UniformNoiseCfg(n_min=-0.5, n_max=0.5),
            history_length=PROPRIO_HISTORY_LENGTH,
        ),
        "last_action": ObsTermCfg(func=mdp.last_action, history_length=PROPRIO_HISTORY_LENGTH),
    }

    # Critic observations
    critic_terms = {
        # Should be the same as the teacher observations.
        "joint_pos_ref": ObsTermCfg(func=mdp.generated_commands, params={"command_name": "joint_pos_ref_command"}),
        "joint_vel_ref": ObsTermCfg(func=mdp.generated_commands, params={"command_name": "joint_vel_ref_command"}),
        "position_ref": ObsTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "position_b_ref_command"},
            noise=UniformNoiseCfg(n_min=-0.25, n_max=0.25),
        ),
        "rotation_ref": ObsTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "rotation_ref_command"},
            noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05),
        ),

        "depth_image": ObsTermCfg(
            func=instinct_mdp.visualizable_image,
            # params={"sensor_cfg": SceneEntityCfg("camera"), "data_type": "distance_to_image_plane"},
            params={"sensor_cfg": SceneEntityCfg("camera"), "data_type": "distance_to_image_plane_noised"},
        ),

        # proprioception
        "projected_gravity": ObsTermCfg(
            func=mdp.projected_gravity,
            noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05),
            history_length=8,
        ),
        # base_lin_vel = ObsTermCfg(func=mdp.base_lin_vel)
        "base_ang_vel": ObsTermCfg(
            func=mdp.base_ang_vel,
            noise=UniformNoiseCfg(n_min=-0.2, n_max=0.2),
            history_length=8,
        ),
        "joint_pos": ObsTermCfg(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
            },
            noise=UniformNoiseCfg(n_min=-0.01, n_max=0.01),
            history_length=8,
        ),
        "joint_vel": ObsTermCfg(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
            },
            noise=UniformNoiseCfg(n_min=-0.5, n_max=0.5),
            history_length=8,
        ),
        "last_action": ObsTermCfg(func=mdp.last_action, history_length=8),
    }

    return {
        "policy": ObsGroupCfg(
            terms=policy_terms,
            enable_corruption=True,
            concatenate_terms=False,
        ),
        "critic": ObsGroupCfg(
            terms=critic_terms,
            enable_corruption=False,
            concatenate_terms=False,
        ),
    }

@dataclass(kw_only=True)
class G1PerceptiveVaeEnvCfg(perceptual_cfg.PerceptiveShadowingEnvCfg):
    scene: perceptual_cfg.PerceptiveShadowingSceneCfg = field(default_factory=lambda: perceptual_cfg.PerceptiveShadowingSceneCfg(
        num_envs=4096,
        entities=perceptual_cfg.make_perceptive_scene_entities(
            robot=deepcopy(G1_CFG),
        ),
        sensors=perceptual_cfg.make_perceptive_scene_sensors(
            motion_reference=_make_motion_reference_cfg(debug_vis=False),
            include_height_scanner=False,
        ),
    ))
    observations: dict = field(default_factory=make_vae_observations)

    def __post_init__(self):
        super().__post_init__()

        perceptual_cfg.remove_scene_sensor_cfg(self.scene, "height_scanner")
        camera_cfg = perceptual_cfg.get_camera_sensor_cfg(self.scene)
        robot_cfg = perceptual_cfg.get_scene_entity_cfg(self.scene, "robot")
        motion_reference_cfg = perceptual_cfg.get_motion_reference_cfg(self.scene)

        camera_cfg.data_histories["distance_to_image_plane_noised"] = 10
        self.observations["policy"].terms["depth_image"].params["history_skip_frames"] = 3
        robot_cfg.articulation.actuators = beyondmimic_g1_29dof_actuator_cfgs
        self.actions["joint_pos"].scale = beyondmimic_action_scale
        # Use sparse Jacobian explicitly to avoid dense Jacobian unsupported path for nv > 60 in mjwarp.
        self.sim.mujoco.jacobian = "sparse"

        motion_buffer = list(motion_reference_cfg.motion_buffers.values())[0]
        _apply_motion_matched_terrain_source(
            self.scene,
            default_path=motion_buffer.path,
            default_metadata_yaml=motion_buffer.metadata_yaml,
        )

        self.run_name = "g1PerceptiveVae" + "".join(
            [
                f"_propHistory{PROPRIO_HISTORY_LENGTH}",
                f"_depthHist{camera_cfg.data_histories['distance_to_image_plane_noised']}Skip{self.observations['policy'].terms['depth_image'].params['history_skip_frames']}",
            ]
        )

@dataclass(kw_only=True)
class G1PerceptiveVaeEnvCfg_PLAY(G1PerceptiveVaeEnvCfg):
    scene: perceptual_cfg.PerceptiveShadowingSceneCfg = field(default_factory=lambda: perceptual_cfg.PerceptiveShadowingSceneCfg(
        num_envs=1,
        env_spacing=2.5,
        entities=perceptual_cfg.make_perceptive_scene_entities_with_reference(
            robot=deepcopy(G1_CFG),
            robot_reference=deepcopy(G1_CFG),
        ),
        sensors=perceptual_cfg.make_perceptive_scene_sensors(
            motion_reference=_make_motion_reference_cfg(debug_vis=True),
            include_height_scanner=False,
        ),
    ))

    viewer: ViewerConfig = field(default_factory=lambda: ViewerConfig(
        lookat=(0.0, 0.0, 0.0),
        distance=3.2016,
        elevation=51.3402,
        azimuth=90.0,
        origin_type=ViewerConfig.OriginType.ASSET_BODY,
        entity_name="robot",
        body_name="torso_link",
    ))

    def __post_init__(self):
        super().__post_init__()

        motion_reference_cfg = perceptual_cfg.get_motion_reference_cfg(self.scene)
        camera_cfg = perceptual_cfg.get_camera_sensor_cfg(self.scene)

        # deactivate adaptive sampling and start from the 0.0s of the motion
        self.curriculum["beyond_adaptive_sampling"] = None
        self.events["bin_fail_counter_smoothing"] = None
        MOTION_NAME = list(motion_reference_cfg.motion_buffers.keys())[0]
        motion_reference_cfg.motion_buffers[MOTION_NAME].motion_start_from_middle_range = [0.0, 0.0]
        motion_reference_cfg.motion_buffers[MOTION_NAME].motion_bin_length_s = None
        motion_reference_cfg.motion_buffers[MOTION_NAME].env_starting_stub_sampling_strategy = "independent"
        # self.scene.motion_reference.motion_buffers[MOTION_NAME].path = (
        #     "/localhdd/Datasets/NoKov-Marslab-Motions-instinctnpz/20251115_diveRoll4_kneelClimb_jumpSit_rollVault"
        # )
        # self.scene.motion_reference.motion_buffers[MOTION_NAME].metadata_yaml = (
        #     "/localhdd/Datasets/NoKov-Marslab-Motions-instinctnpz/20251115_diveRoll4_kneelClimb_jumpSit_rollVault/metadata.yaml"
        # )
        # self.scene.terrain.terrain_generator.sub_terrains["motion_matched"].path = (
        #     self.scene.motion_reference.motion_buffers[MOTION_NAME].path
        # )
        # self.scene.terrain.terrain_generator.sub_terrains["motion_matched"].metadata_yaml = (
        #     self.scene.motion_reference.motion_buffers[MOTION_NAME].metadata_yaml
        # )

        # Use non-terrain-matching motion and plane to hack the scene.
        self.scene.terrain.terrain_generator.num_rows = 6
        self.scene.terrain.terrain_generator.num_cols = 6
        # self.scene.motion_reference.motion_buffers.pop(MOTION_NAME)
        # self.scene.motion_reference.motion_buffers["AMASSMotion"] = AMASSMotionCfg()
        # self.scene.motion_reference.motion_buffers["AMASSMotion"].motion_start_from_middle_range = [0.0, 0.0]
        # self.scene.motion_reference.motion_buffers["AMASSMotion"].motion_bin_length_s = None
        # self.scene.terrain.terrain_type = "plane"
        # self.scene.terrain.terrain_generator = None

        camera_cfg.debug_vis = True
        self.scene.terrain.collision_debug_vis = False
        self.observations["policy"].terms["depth_image"].params["debug_vis"] = True
        self.viewer.debug_vis_show_all_envs = True

        # change reset robot event with more pitch_down randomization (since the robot is facing -y axis)
        # self.events.reset_robot.params["randomize_pose_range"]["roll"] = (0.0, 0.6)

        # remove some terimation terms
        self.terminations["base_pos_too_far"] = None
        self.terminations["base_pg_too_far"] = None
        self.terminations["link_pos_too_far"] = None
        self.terminations["dataset_exhausted"].params["reset_without_notice"] = True

        # put the reference in scene and move the robot elsewhere
        # self.events.reset_robot.params["position_offset"] = [0.0, 1.0, 2.0]
        # self.scene.motion_reference.visualizing_robot_offset = (0.0, 0.0, 0.0)

        # hack the randomization range
        # self.events["add_joint_default_pos"].params["offset_distribution_params"] = (-0.05, 0.05)
        # self.events["physics_material"].params["static_friction_range"] = (2.0, 2.0)
        # self.events["physics_material"].params["dynamic_friction_range"] = (2.0, 2.0)
        # self.events["base_com"].params["coms_z_distribution_params"] = (0.15, 0.15)

        # remove some randomizations
        self.events["add_joint_default_pos"] = None
        self.events["base_com"] = None
        self.events["physics_material"] = None
        self.events["push_robot"] = None
        self.events["reset_robot"].params["randomize_pose_range"]["x"] = (0.0, 0.0)
        self.events["reset_robot"].params["randomize_pose_range"]["y"] = (0.0, 0.0)
        self.events["reset_robot"].params["randomize_pose_range"]["z"] = (0.0, 0.0)
        self.events["reset_robot"].params["randomize_pose_range"]["roll"] = (0.0, 0.0)
        self.events["reset_robot"].params["randomize_pose_range"]["pitch"] = (0.0, 0.0)
        self.events["reset_robot"].params["randomize_pose_range"]["yaw"] = (0.0, 0.0)
        self.events["reset_robot"].params["randomize_velocity_range"]["x"] = (0.0, 0.0)
        self.events["reset_robot"].params["randomize_velocity_range"]["y"] = (0.0, 0.0)
        self.events["reset_robot"].params["randomize_velocity_range"]["z"] = (0.0, 0.0)
        self.events["reset_robot"].params["randomize_velocity_range"]["roll"] = (0.0, 0.0)
        self.events["reset_robot"].params["randomize_velocity_range"]["pitch"] = (0.0, 0.0)
        self.events["reset_robot"].params["randomize_velocity_range"]["yaw"] = (0.0, 0.0)
        self.events["reset_robot"].params["randomize_joint_pos_range"] = (0.0, 0.0)

        # add some additional monitor terms
        self.monitors["shadowing_position_stats"] = MonitorTermCfg(
            func=ShadowingBasePosMonitorTerm,
            params=dict(
                robot_cfg=SceneEntityCfg("robot"),
                motion_reference_cfg=SceneEntityCfg("motion_reference"),
            ),
        )
        self.monitors["right_ankle_pitch_actuator"] = MonitorTermCfg(
            func=ActuatorMonitorTerm,
            params=dict(
                asset_cfg=SceneEntityCfg("robot", joint_names="right_ankle_pitch.*"),
            ),
        )
        self.monitors["left_ankle_pitch_actuator"] = MonitorTermCfg(
            func=ActuatorMonitorTerm,
            params=dict(
                asset_cfg=SceneEntityCfg("robot", joint_names="left_ankle_pitch.*"),
            ),
        )
        self.monitors["right_knee_actuator"] = MonitorTermCfg(
            func=ActuatorMonitorTerm,
            params=dict(
                asset_cfg=SceneEntityCfg("robot", joint_names="right_knee.*"),
            ),
        )
        self.monitors["left_knee_actuator"] = MonitorTermCfg(
            func=ActuatorMonitorTerm,
            params=dict(
                asset_cfg=SceneEntityCfg("robot", joint_names="left_knee.*"),
            ),
        )

        # add another box to the scene (to test visual generalization)
        # self.scene.distractor = RigidObjectCfg(
        #     prim_path="{ENV_REGEX_NS}/cube",
        #     spawn=sim_utils.MeshCuboidCfg(
        #         size=(1.23, 0.35, 0.6),
        #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
        #             kinematic_enabled=False,
        #             disable_gravity=False,
        #             max_depenetration_velocity=1.0,
        #         ),
        #         mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        #         collision_props=sim_utils.CollisionPropertiesCfg(),
        #         physics_material=sim_utils.material_cfg,
        #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.8, 0.3)),
        #     ),
        #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -1.0, 0.3)),
        # )
        # self.scene.camera.mesh_prim_paths.append("/World/envs/env_.*/cube/.*")

        # see the reference robot
        # self.scene.camera.mesh_prim_paths.append("/World/envs/env_.*/RobotReference/.*")
