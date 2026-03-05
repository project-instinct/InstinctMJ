# Copyright (c) 2022-2024, The Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
from copy import deepcopy

import mujoco
import yaml

import mjlab.envs.mdp as mdp
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import CurriculumTermCfg
from mjlab.managers import EventTermCfg
from mjlab.managers import ObservationGroupCfg
from mjlab.managers import ObservationTermCfg
from mjlab.managers import RewardTermCfg
from mjlab.managers import SceneEntityCfg
from mjlab.managers import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.terrains import TerrainImporterCfg
from mjlab.utils.noise import UniformNoiseCfg
from mjlab.utils.spec_config import MaterialCfg, TextureCfg
from mjlab.viewer.viewer_config import ViewerConfig

import instinct_mjlab.envs.mdp as instinct_mdp
import instinct_mjlab.tasks.shadowing.whole_body.shadowing_env_cfg as shadowing_cfg
from instinct_mjlab.assets.unitree_g1 import (
    G1_29DOF_TORSOBASE_POPSICLE_CFG,
    beyondmimic_action_scale,
)
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
from instinct_mjlab.motion_reference.motion_files.amass_motion_cfg import AmassMotionCfg as AmassMotionCfgBase
from instinct_mjlab.motion_reference.utils import motion_interpolate_bilinear
from instinct_mjlab.utils.motion_validation import resolve_datasets_root

combine_method = "prod"
G1_CFG = G1_29DOF_TORSOBASE_POPSICLE_CFG
_DATASETS_ROOT = resolve_datasets_root()

_UNDESIRED_CONTACT_SENSOR_NAME = "undesired_contact_forces"
_SELF_COLLISION_SENSOR_NAME = "self_collision"


def _edit_shadowing_scene_spec(spec: mujoco.MjSpec) -> None:
    """Apply skybox and white-ish ground material for native viewer play."""
    ground_texture_name = "whole_body_groundplane"
    ground_material_name = "whole_body_groundplane"

    sky_rgb_top = (0.98, 0.99, 1.0)
    sky_rgb_horizon = (0.78, 0.86, 0.95)
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

# MOTION_NAME = "AccadRun" # success
# _hacked_selected_file_ = "ACCAD/Male2Running_c3d/C5 - walk to run_retargetted.npz"

# MOTION_NAME = "AccadMartialBounce"
# _hacked_selected_file_ = "ACCAD/MartialArtsWalksTurns_c3d/E8 - bounce_retargetted.npz"
# MOTION_NAME = "KitStomp"
# _hacked_selected_file_ = "KIT/3/stomp_left03_retargetted.npz"

# MOTION_NAME = "AccadMartialSpin"
# _hacked_selected_file_ = "ACCAD/Male2MartialArtsKicks_c3d/G20_-__reverse_spin_cresent_right_retargetted.npz"
# MOTION_NAME = "KitStretch"  # requires balancing
# _hacked_selected_file_ = "KIT/3/streching_leg01_retargetted.npz"

MOTION_NAME = "LafanKungfu1"
_hacked_selected_files_ = ["fightAndSports1_subject1_retargetted.npz"]
# MOTION_NAME = "LafanSprint1"
# _hacked_selected_files_ = ["sprint1_subject2_retargetted.npz"]

# MOTION_NAME = "test"
# _hacked_selected_files_ = ["CMU/90/90_26_retargetted.npz"]

MOTION_NAME = "LafanFight5Files"
_path_ = os.path.expanduser("~/Xyk/Datasets/NoKov-Marslab-Motions-instinctnpz/20251016_diveroll4_single")
_hacked_selected_files_ = [
    "fight1_subject2_retargetted.npz",
    "fight1_subject3_retargetted.npz",
    "fight1_subject5_retargetted.npz",
    "fightAndSports1_subject1_retargetted.npz",
    "fightAndSports1_subject4_retargetted.npz",
]


MOTION_NAME = "LafanFiltered"
_path_ = os.path.expanduser(
    os.environ.get(
        "INSTINCT_WHOLEBODY_MOTION_PATH",
        "~/Xyk/Datasets/NoKov-Marslab-Motions-instinctnpz/20251016_diveroll4_single",
    )
)
_hacked_selected_files_ = [
    "aiming1_subject1_retargetted.npz",  # O
    "aiming1_subject4_retargetted.npz",  # O
    "aiming2_subject2_retargetted.npz",  # O
    "aiming2_subject3_retargetted.npz",  # O
    "aiming2_subject5_retargetted.npz",  # O
    "dance1_subject1_retargetted.npz",  # O
    "dance1_subject2_retargetted.npz",  # O
    "dance1_subject3_retargetted.npz",  # O
    "dance2_subject1_retargetted.npz",  # O
    "dance2_subject2_retargetted.npz",  # O
    "dance2_subject3_retargetted.npz",  # O
    "dance2_subject4_retargetted.npz",  # O
    "dance2_subject5_retargetted.npz",  # O
    "fallAndGetUp1_subject1_retargetted.npz",  # O
    "fallAndGetUp1_subject4_retargetted.npz",  # O
    "fallAndGetUp1_subject5_retargetted.npz",  # O
    "fallAndGetUp2_subject2_retargetted.npz",  # O
    "fallAndGetUp2_subject3_retargetted.npz",  # O
    "fallAndGetUp3_subject1_retargetted.npz",  # O
    "fight1_subject2_retargetted.npz",  # O
    "fight1_subject3_retargetted.npz",  # O
    "fight1_subject5_retargetted.npz",  # O
    "fightAndSports1_subject1_retargetted.npz",  # O
    "fightAndSports1_subject4_retargetted.npz",  # O
    "ground1_subject1_retargetted.npz",  # O
    "ground1_subject4_retargetted.npz",  # O
    "ground1_subject5_retargetted.npz",  # O
    "ground2_subject2_retargetted.npz",  # O
    "ground2_subject3_retargetted.npz",  # O
    "jumps1_subject1_retargetted.npz",  # O
    "jumps1_subject2_retargetted.npz",  # O
    "jumps1_subject5_retargetted.npz",  # O
    "multipleActions1_subject1_retargetted.npz",  # O
    "multipleActions1_subject2_retargetted.npz",  # O
    # "multipleActions1_subject3_retargetted.npz", # X
    # "multipleActions1_subject4_retargetted.npz", # - (some sitting pose, but seems torlerable)
    # "obstacles1_subject1_retargetted.npz", # X
    # "obstacles1_subject2_retargetted.npz", # X
    # "obstacles1_subject5_retargetted.npz", # X
    # "obstacles2_subject1_retargetted.npz", # X
    # "obstacles2_subject2_retargetted.npz", # X
    # "obstacles2_subject5_retargetted.npz", # disable all obstacles
    # "obstacles3_subject3_retargetted.npz", # disable all obstacles
    # "obstacles3_subject4_retargetted.npz", # disable all obstacles
    # "obstacles4_subject2_retargetted.npz", # disable all obstacles
    # "obstacles4_subject3_retargetted.npz", # X  # disable all obstacles
    # "obstacles4_subject4_retargetted.npz", # disable all obstacles
    # "obstacles5_subject2_retargetted.npz", # disable all obstacles
    # "obstacles5_subject3_retargetted.npz", # disable all obstacles
    # "obstacles5_subject4_retargetted.npz", # disable all obstacles
    # "obstacles6_subject1_retargetted.npz", # disable all obstacles
    # "obstacles6_subject4_retargetted.npz", # disable all obstacles
    # "obstacles6_subject5_retargetted.npz", # disable all obstacles
    "push1_subject2_retargetted.npz",  # O
    "pushAndFall1_subject1_retargetted.npz",  # O
    "pushAndFall1_subject4_retargetted.npz",  # O
    "pushAndStumble1_subject2_retargetted.npz",  # O
    "pushAndStumble1_subject3_retargetted.npz",  # O
    "pushAndStumble1_subject5_retargetted.npz",  # O
    "run1_subject2_retargetted.npz",
    "run1_subject5_retargetted.npz",
    "run2_subject1_retargetted.npz",
    "run2_subject4_retargetted.npz",
    "sprint1_subject2_retargetted.npz",
    "sprint1_subject4_retargetted.npz",
    "walk1_subject1_retargetted.npz",
    "walk1_subject2_retargetted.npz",
    "walk1_subject5_retargetted.npz",
    "walk2_subject1_retargetted.npz",
    "walk2_subject3_retargetted.npz",
    "walk2_subject4_retargetted.npz",
    "walk3_subject1_retargetted.npz",
    "walk3_subject2_retargetted.npz",
    "walk3_subject3_retargetted.npz",
    "walk3_subject4_retargetted.npz",
    "walk3_subject5_retargetted.npz",
    "walk4_subject1_retargetted.npz",  # O
]

# MOTION_NAME = "LafanGetup2S3"
# _path_ = os.path.expanduser("~/Datasets/UbisoftLAFAN1_GMR_g1_29dof_torsoBase_retargetted_instinctnpz")
# _hacked_selected_files_ = [
#     "fallAndGetUp2_subject3_retargetted.npz",
# ]

with open(f"/tmp/{MOTION_NAME}.yaml", "w") as f:
    yaml.dump(
        {
            "selected_files": _hacked_selected_files_,
        },
        f,
    )


def _make_amass_motion_cfg() -> AmassMotionCfgBase:
    return AmassMotionCfgBase(
        # path = os.path.expanduser("~/Datasets/AMASS_CMU_KIT_ACCAD_DanceDB_HumanEva_retargetted_20250702")
        # path = os.path.expanduser("~/Datasets/AMASS_SMPLX-NG_GMR_29dof_g1_torsoBase_retargetted_20250825_instinctnpz")
        # path = os.path.expanduser("~/Datasets/UbisoftLAFAN1_GMR_g1_29dof_torsoBase_retargetted_instinctnpz")
        # path = os.path.expanduser("~/Datasets/AMASS_SMPLX-NG_GMR_29dof_g1_torsoBase_retargetted_20250901_instinctnpz")
        # path = _path_
        path=_path_,
        retargetting_func=None,
        filtered_motion_selection_filepath=None,
        motion_start_from_middle_range=[0.0, 0.8],
        motion_start_height_offset=0.0,
        ensure_link_below_zero_ground=False,
        # env_starting_stub_sampling_strategy = "concat_motion_bins"
        env_starting_stub_sampling_strategy="independent",
        buffer_device="output_device",
        motion_interpolate_func=motion_interpolate_bilinear,
        velocity_estimation_method="frontbackward",
        motion_bin_length_s=1.0,
    )


# URDF path for PyTorch Kinematics (G1_CFG.spawn.asset_path equivalent)
_G1_URDF_PATH = os.path.join(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(
                    os.path.dirname(
                        os.path.dirname(os.path.abspath(__file__))
                    )
                )
            )
        )
    ),
    "assets", "resources", "unitree_g1", "urdf", "g1_29dof_torsobase_popsicle.urdf",
)


def _make_motion_reference_cfg(*, debug_vis: bool) -> MotionReferenceManagerCfg:
    return MotionReferenceManagerCfg(
        name="motion_reference",
        entity_name="robot",
        robot_model_path=_G1_URDF_PATH,
        debug_vis=debug_vis,
        reference_entity_name="robot_reference" if debug_vis else None,
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
        # symmetric_augmentation_link_mapping=[
        #     0,
        #     1,
        #     3,
        #     2,
        #     5,
        #     4,
        #     7,
        #     6,
        #     9,
        #     8,
        #     11,
        #     10,
        #     13,
        #     12,
        # ],
        # symmetric_augmentation_joint_mapping=G1_29Dof_torsoBase_symmetric_augmentation_joint_mapping,
        # symmetric_augmentation_joint_reverse_buf=G1_29Dof_torsoBase_symmetric_augmentation_joint_reverse_buf,
        symmetric_augmentation_link_mapping=None,
        symmetric_augmentation_joint_mapping=None,
        symmetric_augmentation_joint_reverse_buf=None,
        frame_interval_s=0.02,
        update_period=0.02,
        num_frames=10,
        data_start_from="current_time",
        visualizing_robot_offset=(0.0, 1.5, 0.0),
        visualizing_robot_from="reference_frame",
        motion_buffers={
            #     "CMU_KIT": AmassMotionCfg(
            #         path=os.path.expanduser("~/Datasets/AMASS_CMU_KIT_retargetted_20250702"),  # type: ignore
            #         filtered_motion_selection_filepath=os.path.expanduser(  # type: ignore
            #             "~/Datasets/AMASS_selections/CMU_KIT_weighted_retargetted_20250702.yaml",
            #         ),
            #     ),
            # "CMU_KIT_DanceDB_BioMotionLab": AmassMotionCfg(
            #     path=os.path.expanduser("~/Datasets/AMASS_CMU_KIT_DanceDB_BioMotionLab_retargetted_20250702"),  # type: ignore
            #     filtered_motion_selection_filepath=os.path.expanduser(  # type: ignore
            #         "~/Datasets/AMASS_selections/CMU_KIT_DanceDB_BioMotionLab_weighted_retargetted_20250702.yaml",
            #     ),
            # ),
            # "CMU_KIT_ACCAD_DanceDB_HumanEva": AmassMotionCfg(
            #     path=os.path.expanduser("~/Datasets/AMASS_CMU_KIT_ACCAD_DanceDB_HumanEva_retargetted_20250702"),  # type: ignore
            #     filtered_motion_selection_filepath=os.path.expanduser(  # type: ignore
            #         # "~/Datasets/AMASS_selections/CMU_KIT_ACCAD_DanceDB_HumanEva_weighted_retargetted_20250702.yaml",
            #         "~/Datasets/AMASS_selections/CMU_KIT_ACCAD_DanceDB_HumanEva_weighted_moverange_20250724_retargetted_20250702.yaml",
            #     ),
            # ),
            # "UbisoftLAFAN1_GMR": AmassMotionCfg(
            #     path=os.path.expanduser("~/Datasets/UbisoftLAFAN1_GMR_g1_29dof_torsoBase_retargetted_instinctnpz"),  # type: ignore
            #     filtered_motion_selection_filepath=None,
            # ),
            MOTION_NAME: _make_amass_motion_cfg(),
        },
        mp_split_method="Even",
    )


def _make_scene_cfg(*, play: bool, motion_reference_cfg: MotionReferenceManagerCfg) -> SceneCfg:
    undesired_contact_forces = ContactSensorCfg(
        name=_UNDESIRED_CONTACT_SENSOR_NAME,
        primary=ContactMatch(
            mode="body",
            pattern=".*",
            entity="robot",
            exclude=(
                "left_ankle_roll_link",
                "right_ankle_roll_link",
                "left_wrist_yaw_link",
                "right_wrist_yaw_link",
            ),
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        history_length=3,
    )
    self_collision = ContactSensorCfg(
        name=_SELF_COLLISION_SENSOR_NAME,
        primary=ContactMatch(mode="subtree", pattern="torso_link", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="torso_link", entity="robot"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )

    entities = {
        "robot": deepcopy(G1_CFG),
    }
    if play:
        entities["robot_reference"] = deepcopy(G1_CFG)

    return SceneCfg(
        num_envs=1 if play else 2048,
        env_spacing=2.5 if play else 4.0,
        terrain=TerrainImporterCfg(terrain_type="plane"),
        entities=entities,
        sensors=(
            undesired_contact_forces,
            self_collision,
            motion_reference_cfg,
        ),
        spec_fn=_edit_shadowing_scene_spec,
    )


def _actions_cfg() -> dict[str, mdp.JointPositionActionCfg]:
    return {
        "joint_pos": mdp.JointPositionActionCfg(
            entity_name="robot",
            actuator_names=(".*",),
            scale=deepcopy(beyondmimic_action_scale),
        ),
    }


def _commands_cfg() -> dict[str, object]:
    return {
        "position_ref_command": instinct_mdp.PositionRefCommandCfg(
            realtime_mode=True,
            current_state_command=False,
            anchor_frame="robot",
        ),
        "position_b_ref_command": instinct_mdp.PositionRefCommandCfg(
            realtime_mode=True,
            current_state_command=False,
            anchor_frame="reference",
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


def _observations_cfg(link_of_interests: list[str]) -> dict[str, ObservationGroupCfg]:
    return {
        "policy": ObservationGroupCfg(
            terms={
                # BeyondMimic specific reference observations
                "joint_pos_ref": ObservationTermCfg(
                    func=mdp.generated_commands,
                    params={"command_name": "joint_pos_ref_command"},
                ),
                "joint_vel_ref": ObservationTermCfg(
                    func=mdp.generated_commands,
                    params={"command_name": "joint_vel_ref_command"},
                ),
                "position_ref": ObservationTermCfg(
                    func=mdp.generated_commands,
                    params={"command_name": "position_b_ref_command"},
                    noise=UniformNoiseCfg(n_min=-0.25, n_max=0.25),
                ),
                "rotation_ref": ObservationTermCfg(
                    func=mdp.generated_commands,
                    params={"command_name": "rotation_ref_command"},
                    noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05),
                ),
                # proprioception
                # "base_lin_vel": ObservationTermCfg(
                #     func=mdp.base_lin_vel,
                #     noise=UniformNoiseCfg(n_min=-0.5, n_max=0.5),
                # ),
                "projected_gravity": ObservationTermCfg(
                    func=mdp.projected_gravity,
                    noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05),
                ),
                "base_ang_vel": ObservationTermCfg(
                    func=mdp.base_ang_vel,
                    noise=UniformNoiseCfg(n_min=-0.2, n_max=0.2),
                ),
                "joint_pos": ObservationTermCfg(
                    func=mdp.joint_pos_rel,
                    noise=UniformNoiseCfg(n_min=-0.01, n_max=0.01),
                ),
                "joint_vel": ObservationTermCfg(
                    func=mdp.joint_vel_rel,
                    noise=UniformNoiseCfg(n_min=-0.5, n_max=0.5),
                ),
                "last_action": ObservationTermCfg(func=mdp.last_action),
            },
            enable_corruption=True,
            concatenate_terms=False,
        ),
        "critic": ObservationGroupCfg(
            terms={
                # BeyondMimic specific reference observations
                "joint_pos_ref": ObservationTermCfg(
                    func=mdp.generated_commands,
                    params={"command_name": "joint_pos_ref_command"},
                ),
                "joint_vel_ref": ObservationTermCfg(
                    func=mdp.generated_commands,
                    params={"command_name": "joint_vel_ref_command"},
                ),
                "position_ref": ObservationTermCfg(
                    func=mdp.generated_commands,
                    params={"command_name": "position_ref_command"},
                ),
                "rotation_ref": ObservationTermCfg(
                    func=mdp.generated_commands,
                    params={"command_name": "rotation_ref_command"},
                ),
                # proprioception
                "link_pos": ObservationTermCfg(
                    func=instinct_mdp.link_pos_b,
                    params={
                        "asset_cfg": SceneEntityCfg(
                            name="robot",
                            body_names=link_of_interests,
                            preserve_order=True,
                        ),
                    },
                ),
                "link_rot": ObservationTermCfg(
                    func=instinct_mdp.link_tannorm_b,
                    params={
                        "asset_cfg": SceneEntityCfg(
                            name="robot",
                            body_names=link_of_interests,
                            preserve_order=True,
                        ),
                    },
                ),
                "base_lin_vel": ObservationTermCfg(func=mdp.base_lin_vel),
                "base_ang_vel": ObservationTermCfg(func=mdp.base_ang_vel),
                "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel),
                "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel),
                "last_action": ObservationTermCfg(func=mdp.last_action),
            },
            enable_corruption=False,
            concatenate_terms=False,
        ),
    }


def _rewards_cfg() -> dict[str, RewardTermCfg | None]:
    # Inherits rewards from whole_body.shadowing_env_cfg.
    return deepcopy(shadowing_cfg.shadowing_rewards_terms())


def _events_cfg() -> dict[str, EventTermCfg | None]:
    return {
        # startup
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
        # interval
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
        # for motion initialization and reset
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


def _curriculum_cfg() -> dict[str, CurriculumTermCfg | None]:
    return {
        "beyond_adaptive_sampling": CurriculumTermCfg(  # type: ignore[arg-type]
            func=instinct_mdp.BeyondMimicAdaptiveWeighting,
        ),
    }


def _terminations_cfg() -> dict[str, TerminationTermCfg]:
    return {
        "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
        "base_pos_too_far": TerminationTermCfg(
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
        "base_pg_too_far": TerminationTermCfg(
            func=instinct_mdp.projected_gravity_far_from_ref,
            time_out=False,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "reference_cfg": SceneEntityCfg("motion_reference"),
                "projected_gravity_threshold": 0.8,  # distance on z-axis of projected gravity
                "check_at_keyframe_threshold": -1,
                "z_only": False,  # find out useful if not z_only but beyondmimic default is z_only
                "print_reason": False,
            },
        ),
        "link_pos_too_far": TerminationTermCfg(
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
        "dataset_exhausted": TerminationTermCfg(
            func=instinct_mdp.dataset_exhausted,
            time_out=True,
            params={
                "reference_cfg": SceneEntityCfg("motion_reference"),
                "print_reason": False,
            },
        ),
        "out_of_border": TerminationTermCfg(
            func=instinct_mdp.terrain_out_of_bounds,
            time_out=True,
            params={"asset_cfg": SceneEntityCfg("robot"), "print_reason": False, "distance_buffer": 0.1},
        ),
    }


def _monitors_cfg() -> dict[str, MonitorTermCfg]:
    return {
        # joint_torque = SceneEntityCfg("monitor_joint_torque") # NOTE: hurt the performance, so not used.
        # upper_joint_stat = MonitorTermCfg(
        #     func=JointStatMonitorTerm,
        #     params=dict(
        #         asset_cfg=SceneEntityCfg(
        #             "robot",
        #             joint_names=[
        #                 ".*_shoulder_.*",
        #                 ".*_elbow_.*",
        #                 ".*_wrist_.*",
        #             ],
        #         ),
        #     ),
        # )
        # lower_joint_stat = MonitorTermCfg(
        #     func=JointStatMonitorTerm,
        #     params=dict(
        #         asset_cfg=SceneEntityCfg(
        #             "robot",
        #             joint_names=[
        #                 "waist_.*",
        #                 ".*_ankle_.*",
        #                 ".*_hip_.*",
        #             ],
        #         ),
        #     ),
        # )
        # body_stat = MonitorTermCfg(
        #     func=BodyStatMonitorTerm,
        #     params=dict(
        #         asset_cfg=SceneEntityCfg(
        #             "robot",
        #             body_names=MISSING,
        #         ),
        #     ),
        # )
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


def _viewer_cfg(play: bool) -> ViewerConfig:
    if not play:
        return ViewerConfig()
    return ViewerConfig(
        lookat=(0.0, 0.75, 0.0),
        distance=4.1231,
        elevation=14.0362,
        azimuth=0.0,
        origin_type=ViewerConfig.OriginType.ASSET_ROOT,
        entity_name="robot",
    )


def _apply_motion_buffer_curriculum(cfg: ManagerBasedRlEnvCfg, motion_reference_cfg: MotionReferenceManagerCfg) -> None:
    assert (
        len(list(motion_reference_cfg.motion_buffers.keys())) == 1
    ), "Only support single motion buffer for now"
    motion_buffer = list(motion_reference_cfg.motion_buffers.values())[0]
    if motion_buffer.motion_bin_length_s is None:
        return
    if motion_buffer.env_starting_stub_sampling_strategy == "concat_motion_bins":
        cfg.curriculum["beyond_adaptive_sampling"] = CurriculumTermCfg(  # type: ignore[arg-type]
            func=instinct_mdp.BeyondConcatMotionAdaptiveWeighting,
        )
    elif motion_buffer.env_starting_stub_sampling_strategy == "independent":
        cfg.curriculum["beyond_adaptive_sampling"] = CurriculumTermCfg(  # type: ignore[arg-type]
            func=instinct_mdp.BeyondMimicAdaptiveWeighting,
        )
    else:
        raise ValueError(
            "Unsupported env starting stub sampling method:"
            f" {motion_buffer.env_starting_stub_sampling_strategy}"
        )


def _apply_play_overrides(cfg: ManagerBasedRlEnvCfg, motion_reference_cfg: MotionReferenceManagerCfg) -> None:
    # spawn the robot randomly in the grid (instead of their terrain levels)
    cfg.scene.terrain.max_init_terrain_level = None
    # reduce the number of terrains to save memory
    if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.num_rows = 3
        cfg.scene.terrain.terrain_generator.num_cols = 3
        cfg.scene.terrain.terrain_generator.curriculum = False

    motion_reference_cfg.symmetric_augmentation_joint_mapping = None
    motion_reference_cfg.visualizing_marker_types = ["relative_links"]
    cfg.curriculum["beyond_adaptive_sampling"] = None
    cfg.events["push_robot"] = None
    cfg.events["bin_fail_counter_smoothing"] = None

    # enable print_reason option in the termination terms
    for term in cfg.terminations.values():
        if "print_reason" in term.params:
            term.params["print_reason"] = True
    # self.episode_length_s = 10.0
    # for term_name, term in self.terminations.__dict__.items():
    #     if (not term_name == "dataset_exhausted") and (not term_name == "time_out"):
    #         self.terminations.__dict__[term_name] = None

    # enable debug_vis option in commands
    cfg.commands["position_ref_command"].debug_vis = True
    cfg.commands["position_b_ref_command"].debug_vis = True
    cfg.commands["rotation_ref_command"].debug_vis = True
    cfg.commands["joint_pos_ref_command"].debug_vis = True
    cfg.commands["joint_vel_ref_command"].debug_vis = True

    # add PLAY-specific monitor term
    # self.monitors.shoulder_actuator = MonitorTermCfg(
    #     func=ActuatorMonitorTerm,
    #     params={
    #         "asset_cfg": SceneEntityCfg(name="robot", joint_names="left_shoulder_roll.*"),
    #         "torque_plot_scale": 1e-2,
    #         # "joint_vel_plot_scale": 1e-1,
    #         "joint_power_plot_scale": 1e-1,
    #     },
    # )
    # self.monitors.waist_actuator = MonitorTermCfg(
    #     func=ActuatorMonitorTerm,
    #     params={
    #         "asset_cfg": SceneEntityCfg(name="robot", joint_names="waist_roll.*"),
    #         "torque_plot_scale": 1e-2,
    #         # "joint_vel_plot_scale": 1e-1,
    #         "joint_power_plot_scale": 1e-1,
    #     },
    # )
    # self.monitors.knee_actuator = MonitorTermCfg(
    #     func=ActuatorMonitorTerm,
    #     params={
    #         "asset_cfg": SceneEntityCfg(name="robot", joint_names="left_knee.*"),
    #         "torque_plot_scale": 1e-2,
    #         # "joint_vel_plot_scale": 1e-1,
    #         "joint_power_plot_scale": 1e-1,
    #     },
    # )
    # self.monitors.reward_sum = MonitorTermCfg(
    #     func=RewardSumMonitorTerm,
    # )
    # self.monitors.reference_stat_case = MonitorTermCfg(
    #     func=ShadowingJointReferenceMonitorTerm,
    #     params=dict(
    #         reference_cfg=SceneEntityCfg(
    #             "motion_reference",
    #             joint_names=[
    #                 "left_hip_pitch.*",
    #             ],
    #         ),
    #     ),
    # )
    # self.monitors.shadowing_base_pos = MonitorTermCfg(
    #     func=ShadowingBasePosMonitorTerm,
    #     params=dict(
    #         robot_cfg=SceneEntityCfg("robot"),
    #         motion_reference_cfg=SceneEntityCfg("motion_reference"),
    #     ),
    # )


def _build_run_name(cfg: InstinctLabRLEnvCfg, motion_reference_cfg: MotionReferenceManagerCfg) -> str:
    motion_buffer = list(motion_reference_cfg.motion_buffers.values())[0]
    return "".join(
        [
            "G1Shadowing",
            f"_{MOTION_NAME}",
            (
                "_odomObs"
                if ("base_lin_vel" in cfg.observations["policy"].terms.keys())
                and cfg.commands["position_b_ref_command"].anchor_frame == "robot"
                else ""
            ),
            # (
            #     "_" + "-".join(self.scene.motion_reference.motion_buffers.keys())
            #     if self.scene.motion_reference.motion_buffers
            #     else ""
            # ),
            # (
            #     f"_proprioHist{self.observations.policy.joint_pos.history_length}"
            #     if self.observations.policy.joint_pos.history_length > 0
            #     else ""
            # ),
            # (
            #     f"_futureRef{self.scene.motion_reference.num_frames}"
            #     if self.scene.motion_reference.num_frames > 1
            #     else ""
            # ),
            # f"_FrameStartFrom{self.scene.motion_reference.data_start_from}",
            # "_forLoopMotionWeights",
            # "_forLoopMotionSample",
            ("_pgTermXYalso" if not cfg.terminations["base_pg_too_far"].params["z_only"] else ""),
            (
                "_concatMotionBins"
                if motion_buffer.env_starting_stub_sampling_strategy == "concat_motion_bins"
                else "_independentMotionBins"
            ),
            "_fixFramerate_diveroll4",
        ]
    )


def g1_plane_shadowing_env_cfg(*, play: bool = False) -> ManagerBasedRlEnvCfg:
    motion_reference_cfg = _make_motion_reference_cfg(debug_vis=play)
    scene = _make_scene_cfg(play=play, motion_reference_cfg=motion_reference_cfg)

    cfg = InstinctLabRLEnvCfg(
        scene=scene,
        actions=_actions_cfg(),
        observations=_observations_cfg(link_of_interests=list(motion_reference_cfg.link_of_interests)),
        commands=_commands_cfg(),
        rewards=_rewards_cfg(),
        events=_events_cfg(),
        curriculum=_curriculum_cfg(),
        terminations=_terminations_cfg(),
        monitors=_monitors_cfg(),
        viewer=_viewer_cfg(play),
        decimation=4,
        episode_length_s=10.0,
    )
    cfg.sim.njmax = 1200
    cfg.sim.nconmax = None
    cfg.sim.mujoco.timestep = 1.0 / 50.0 / cfg.decimation
    cfg.sim.mujoco.iterations = 10
    cfg.sim.mujoco.ls_iterations = 20

    _apply_motion_buffer_curriculum(cfg, motion_reference_cfg)
    if play:
        _apply_play_overrides(cfg, motion_reference_cfg)

    cfg.run_name = _build_run_name(cfg, motion_reference_cfg)
    return cfg


def G1PlaneShadowingEnvCfg() -> ManagerBasedRlEnvCfg:
    """Compatibility callable that returns the train env config."""

    return g1_plane_shadowing_env_cfg(play=False)


def G1PlaneShadowingEnvCfg_PLAY() -> ManagerBasedRlEnvCfg:
    """Compatibility callable that returns the play env config."""

    return g1_plane_shadowing_env_cfg(play=True)


__all__ = [
    "G1PlaneShadowingEnvCfg",
    "G1PlaneShadowingEnvCfg_PLAY",
    "g1_plane_shadowing_env_cfg",
]
