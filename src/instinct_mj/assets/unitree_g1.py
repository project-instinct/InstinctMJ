"""Customized Unitree G1 asset definitions."""

from __future__ import annotations

import copy
import os

import mujoco
from mjlab.actuator import ActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from instinct_mj.actuators import DelayedInstinctActuatorCfg, InstinctActuatorCfg

__file_dir__ = os.path.dirname(os.path.realpath(__file__))

# MJCF (XML) path – uses the local 29-dof torso-base popsicle model.
# Joint-related semantics in this module follow the MuJoCo/MJCF native order.
G1_MJCF_PATH: str = os.path.join(__file_dir__, "resources/unitree_g1/xml/g1_29dof_torsobase_popsicle.xml")
G1_MESHES_DIR: str = os.path.join(__file_dir__, "resources/unitree_g1/meshes")

"""
joint name order:
[
    'waist_pitch_joint',
    'waist_roll_joint',
    'waist_yaw_joint',
    'left_hip_pitch_joint',
    'left_hip_roll_joint',
    'left_hip_yaw_joint',
    'left_knee_joint',
    'left_ankle_pitch_joint',
    'left_ankle_roll_joint',
    'right_hip_pitch_joint',
    'right_hip_roll_joint',
    'right_hip_yaw_joint',
    'right_knee_joint',
    'right_ankle_pitch_joint',
    'right_ankle_roll_joint',
    'left_shoulder_pitch_joint',
    'left_shoulder_roll_joint',
    'left_shoulder_yaw_joint',
    'left_elbow_joint',
    'left_wrist_roll_joint',
    'left_wrist_pitch_joint',
    'left_wrist_yaw_joint',
    'right_shoulder_pitch_joint',
    'right_shoulder_roll_joint',
    'right_shoulder_yaw_joint',
    'right_elbow_joint',
    'right_wrist_roll_joint',
    'right_wrist_pitch_joint',
    'right_wrist_yaw_joint',
]
"""

# NOTE:
# Joint-order dependent buffers below follow MuJoCo/MJCF native joint order.
# This keeps all mjlab tensors and motion-reference augmentation in one order.


def get_g1_spec() -> mujoco.MjSpec:
    """Load the local g1_29dof_torsobase_popsicle.xml as MjSpec."""
    return mujoco.MjSpec.from_file(G1_MJCF_PATH)


# Initial state matching InstinctLab G1_29DOF_TORSOBASE_CFG (simplified variant).
# NOTE: pos is the root (torso_link) world position.
_SIMPLIFIED_INIT_STATE = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.82),
    joint_pos={
        ".*_hip_pitch_joint": -0.20,
        ".*_knee_joint": 0.42,
        ".*_ankle_pitch_joint": -0.23,
        ".*_elbow_joint": 0.87,
        ".*_wrist_roll_joint": 0.0,
        ".*_wrist_pitch_joint": 0.0,
        ".*_wrist_yaw_joint": 0.0,
        "left_shoulder_roll_joint": 0.16,
        "left_shoulder_pitch_joint": 0.35,
        "right_shoulder_roll_joint": -0.16,
        "right_shoulder_pitch_joint": 0.35,
    },
    joint_vel={".*": 0.0},
)

# Initial state matching InstinctLab G1_29DOF_TORSOBASE_POPSICLE_CFG.
# NOTE: pos is the root (torso_link) world position.
_POPSICLE_INIT_STATE = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.82),
    joint_pos={
        ".*_hip_pitch_joint": -0.312,
        ".*_knee_joint": 0.669,
        ".*_ankle_pitch_joint": -0.363,
        ".*_elbow_joint": 0.6,
        "left_shoulder_roll_joint": 0.2,
        "left_shoulder_pitch_joint": 0.2,
        "right_shoulder_roll_joint": -0.2,
        "right_shoulder_pitch_joint": 0.2,
    },
    joint_vel={".*": 0.0},
)


# Motor specs (from Unitree), aligned with mjlab's original g1_constants.
ROTOR_INERTIAS_5020 = (0.139e-4, 0.017e-4, 0.169e-4)
GEARS_5020 = (1, 1 + (46 / 18), 1 + (56 / 16))
ROTOR_INERTIAS_7520_14 = (0.489e-4, 0.098e-4, 0.533e-4)
GEARS_7520_14 = (1, 4.5, 1 + (48 / 22))
ROTOR_INERTIAS_7520_22 = (0.489e-4, 0.109e-4, 0.738e-4)
GEARS_7520_22 = (1, 4.5, 5)
ROTOR_INERTIAS_4010 = (0.068e-4, 0.0, 0.0)
GEARS_4010 = (1, 5, 5)

# Motor output limits (aligned with mjlab's original g1_constants).
ACTUATOR_5020_EFFORT_LIMIT = 25.0
ACTUATOR_7520_14_EFFORT_LIMIT = 88.0
ACTUATOR_7520_22_EFFORT_LIMIT = 139.0
ACTUATOR_4010_EFFORT_LIMIT = 5.0
ACTUATOR_DUAL_5020_EFFORT_LIMIT = ACTUATOR_5020_EFFORT_LIMIT * 2.0

# Following the principles of BeyondMimic, and the kp/kd computation logic.
# NOTE: These logic are still being tested, so we put them here for substitution in users Cfg class.
ARMATURE_5020 = 0.003609725
ARMATURE_7520_14 = 0.010177520
ARMATURE_7520_22 = 0.025101925
ARMATURE_4010 = 0.00425

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ**2
STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ**2
STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ**2
STIFFNESS_4010 = ARMATURE_4010 * NATURAL_FREQ**2

DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ
DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ
DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ
DAMPING_4010 = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ


G1_29DOF_TORSOBASE_DELAYED_LEGS = DelayedInstinctActuatorCfg(
    target_names_expr=(".*_hip_yaw_joint", ".*_hip_roll_joint", ".*_hip_pitch_joint"),
    effort_limit=88.0,
    velocity_limit=60.0,
    stiffness=90.0,
    damping=2.0,
    armature=0.03,
    delay_min_lag=0,
    delay_max_lag=1,
)
G1_29DOF_TORSOBASE_DELAYED_KNEES = DelayedInstinctActuatorCfg(
    target_names_expr=(".*_knee_joint",),
    effort_limit=139.0,
    velocity_limit=60.0,
    stiffness=140.0,
    damping=2.5,
    armature=0.03,
    delay_min_lag=0,
    delay_max_lag=1,
)
G1_29DOF_TORSOBASE_DELAYED_WAIST = DelayedInstinctActuatorCfg(
    target_names_expr=("waist_roll_joint", "waist_pitch_joint"),
    effort_limit=50.0,
    velocity_limit=60.0,
    stiffness=60.0,
    damping=2.5,
    armature=0.03,
    delay_min_lag=0,
    delay_max_lag=1,
)
G1_29DOF_TORSOBASE_DELAYED_WAIST_YAW = DelayedInstinctActuatorCfg(
    target_names_expr=("waist_yaw_joint",),
    effort_limit=88.0,
    velocity_limit=60.0,
    stiffness=90.0,
    damping=2.5,
    armature=0.03,
    delay_min_lag=0,
    delay_max_lag=1,
)
G1_29DOF_TORSOBASE_DELAYED_FEET = DelayedInstinctActuatorCfg(
    target_names_expr=(".*_ankle_pitch_joint", ".*_ankle_roll_joint"),
    effort_limit=20.0,
    velocity_limit=60.0,
    stiffness=20.0,
    damping=1.0,
    armature=0.03,
    delay_min_lag=0,
    delay_max_lag=1,
)
G1_29DOF_TORSOBASE_DELAYED_ARMS = DelayedInstinctActuatorCfg(
    target_names_expr=(
        ".*_shoulder_pitch_joint",
        ".*_shoulder_roll_joint",
        ".*_shoulder_yaw_joint",
        ".*_elbow_joint",
    ),
    effort_limit=25.0,
    velocity_limit=60.0,
    stiffness=25.0,
    damping=1.0,
    armature=0.03,
    delay_min_lag=0,
    delay_max_lag=1,
)
G1_29DOF_TORSOBASE_DELAYED_WRIST_ROLL = DelayedInstinctActuatorCfg(
    target_names_expr=(".*wrist_roll_joint",),
    effort_limit=25.0,
    velocity_limit=25.0,
    stiffness=25.0,
    damping=1.0,
    armature=0.03,
    delay_min_lag=0,
    delay_max_lag=1,
)
G1_29DOF_TORSOBASE_DELAYED_WRIST_PITCH_YAW = DelayedInstinctActuatorCfg(
    target_names_expr=(".*wrist_pitch_joint", ".*wrist_yaw_joint"),
    effort_limit=5.0,
    velocity_limit=25.0,
    stiffness=5.0,
    damping=0.5,
    armature=0.03,
    delay_min_lag=0,
    delay_max_lag=1,
)
g1_29dof_torsobase_delayed_actuator_cfgs: tuple[ActuatorCfg, ...] = (
    G1_29DOF_TORSOBASE_DELAYED_LEGS,
    G1_29DOF_TORSOBASE_DELAYED_KNEES,
    G1_29DOF_TORSOBASE_DELAYED_WAIST,
    G1_29DOF_TORSOBASE_DELAYED_WAIST_YAW,
    G1_29DOF_TORSOBASE_DELAYED_FEET,
    G1_29DOF_TORSOBASE_DELAYED_ARMS,
    G1_29DOF_TORSOBASE_DELAYED_WRIST_ROLL,
    G1_29DOF_TORSOBASE_DELAYED_WRIST_PITCH_YAW,
)


BEYONDMIMIC_G1_29DOF_LEGS_PITCH_YAW = InstinctActuatorCfg(
    target_names_expr=(".*_hip_pitch_joint", ".*_hip_yaw_joint"),
    effort_limit=88.0,
    velocity_limit=32.0,
    stiffness=STIFFNESS_7520_14,
    damping=DAMPING_7520_14,
    armature=ARMATURE_7520_14,
)
BEYONDMIMIC_G1_29DOF_WAIST_YAW = InstinctActuatorCfg(
    target_names_expr=("waist_yaw_joint",),
    effort_limit=88.0,
    velocity_limit=32.0,
    stiffness=STIFFNESS_7520_14,
    damping=DAMPING_7520_14,
    armature=ARMATURE_7520_14,
)
BEYONDMIMIC_G1_29DOF_LEGS_ROLL_KNEE = InstinctActuatorCfg(
    target_names_expr=(".*_hip_roll_joint", ".*_knee_joint"),
    effort_limit=139.0,
    velocity_limit=20.0,
    stiffness=STIFFNESS_7520_22,
    damping=DAMPING_7520_22,
    armature=ARMATURE_7520_22,
)
BEYONDMIMIC_G1_29DOF_FEET = InstinctActuatorCfg(
    target_names_expr=(".*_ankle_pitch_joint", ".*_ankle_roll_joint"),
    effort_limit=50.0,
    velocity_limit=37.0,
    stiffness=2.0 * STIFFNESS_5020,
    damping=2.0 * DAMPING_5020,
    armature=2.0 * ARMATURE_5020,
)
BEYONDMIMIC_G1_29DOF_WAIST = InstinctActuatorCfg(
    target_names_expr=("waist_roll_joint", "waist_pitch_joint"),
    effort_limit=50.0,
    velocity_limit=37.0,
    stiffness=2.0 * STIFFNESS_5020,
    damping=2.0 * DAMPING_5020,
    armature=2.0 * ARMATURE_5020,
)
BEYONDMIMIC_G1_29DOF_ARMS_5020 = InstinctActuatorCfg(
    target_names_expr=(
        ".*_shoulder_pitch_joint",
        ".*_shoulder_roll_joint",
        ".*_shoulder_yaw_joint",
        ".*_elbow_joint",
        ".*_wrist_roll_joint",
    ),
    effort_limit=25.0,
    velocity_limit=37.0,
    stiffness=STIFFNESS_5020,
    damping=DAMPING_5020,
    armature=ARMATURE_5020,
)
BEYONDMIMIC_G1_29DOF_ARMS_4010 = InstinctActuatorCfg(
    target_names_expr=(".*_wrist_pitch_joint", ".*_wrist_yaw_joint"),
    effort_limit=5.0,
    velocity_limit=22.0,
    stiffness=STIFFNESS_4010,
    damping=DAMPING_4010,
    armature=ARMATURE_4010,
)
beyondmimic_g1_29dof_actuator_cfgs: tuple[ActuatorCfg, ...] = (
    BEYONDMIMIC_G1_29DOF_LEGS_PITCH_YAW,
    BEYONDMIMIC_G1_29DOF_WAIST_YAW,
    BEYONDMIMIC_G1_29DOF_LEGS_ROLL_KNEE,
    BEYONDMIMIC_G1_29DOF_FEET,
    BEYONDMIMIC_G1_29DOF_WAIST,
    BEYONDMIMIC_G1_29DOF_ARMS_5020,
    BEYONDMIMIC_G1_29DOF_ARMS_4010,
)


BEYONDMIMIC_G1_29DOF_DELAYED_LEGS_PITCH_YAW = DelayedInstinctActuatorCfg(
    target_names_expr=(".*_hip_pitch_joint", ".*_hip_yaw_joint"),
    effort_limit=88.0,
    velocity_limit=32.0,
    stiffness=STIFFNESS_7520_14,
    damping=DAMPING_7520_14,
    armature=ARMATURE_7520_14,
    delay_min_lag=0,
    delay_max_lag=2,
)
BEYONDMIMIC_G1_29DOF_DELAYED_WAIST_YAW = DelayedInstinctActuatorCfg(
    target_names_expr=("waist_yaw_joint",),
    effort_limit=88.0,
    velocity_limit=32.0,
    stiffness=STIFFNESS_7520_14,
    damping=DAMPING_7520_14,
    armature=ARMATURE_7520_14,
    delay_min_lag=0,
    delay_max_lag=2,
)
BEYONDMIMIC_G1_29DOF_DELAYED_LEGS_ROLL_KNEE = DelayedInstinctActuatorCfg(
    target_names_expr=(".*_hip_roll_joint", ".*_knee_joint"),
    effort_limit=139.0,
    velocity_limit=20.0,
    stiffness=STIFFNESS_7520_22,
    damping=DAMPING_7520_22,
    armature=ARMATURE_7520_22,
    delay_min_lag=0,
    delay_max_lag=2,
)
BEYONDMIMIC_G1_29DOF_DELAYED_FEET = DelayedInstinctActuatorCfg(
    target_names_expr=(".*_ankle_pitch_joint", ".*_ankle_roll_joint"),
    effort_limit=50.0,
    velocity_limit=37.0,
    stiffness=2.0 * STIFFNESS_5020,
    damping=2.0 * DAMPING_5020,
    armature=2.0 * ARMATURE_5020,
    delay_min_lag=0,
    delay_max_lag=2,
)
BEYONDMIMIC_G1_29DOF_DELAYED_WAIST = DelayedInstinctActuatorCfg(
    target_names_expr=("waist_roll_joint", "waist_pitch_joint"),
    effort_limit=50.0,
    velocity_limit=37.0,
    stiffness=2.0 * STIFFNESS_5020,
    damping=2.0 * DAMPING_5020,
    armature=2.0 * ARMATURE_5020,
    delay_min_lag=0,
    delay_max_lag=2,
)
BEYONDMIMIC_G1_29DOF_DELAYED_ARMS_5020 = DelayedInstinctActuatorCfg(
    target_names_expr=(
        ".*_shoulder_pitch_joint",
        ".*_shoulder_roll_joint",
        ".*_shoulder_yaw_joint",
        ".*_elbow_joint",
        ".*_wrist_roll_joint",
    ),
    effort_limit=25.0,
    velocity_limit=37.0,
    stiffness=STIFFNESS_5020,
    damping=DAMPING_5020,
    armature=ARMATURE_5020,
    delay_min_lag=0,
    delay_max_lag=2,
)
BEYONDMIMIC_G1_29DOF_DELAYED_ARMS_4010 = DelayedInstinctActuatorCfg(
    target_names_expr=(".*_wrist_pitch_joint", ".*_wrist_yaw_joint"),
    effort_limit=5.0,
    velocity_limit=22.0,
    stiffness=STIFFNESS_4010,
    damping=DAMPING_4010,
    armature=ARMATURE_4010,
    delay_min_lag=0,
    delay_max_lag=2,
)
beyondmimic_g1_29dof_delayed_actuator_cfgs: tuple[ActuatorCfg, ...] = (
    BEYONDMIMIC_G1_29DOF_DELAYED_LEGS_PITCH_YAW,
    BEYONDMIMIC_G1_29DOF_DELAYED_WAIST_YAW,
    BEYONDMIMIC_G1_29DOF_DELAYED_LEGS_ROLL_KNEE,
    BEYONDMIMIC_G1_29DOF_DELAYED_FEET,
    BEYONDMIMIC_G1_29DOF_DELAYED_WAIST,
    BEYONDMIMIC_G1_29DOF_DELAYED_ARMS_5020,
    BEYONDMIMIC_G1_29DOF_DELAYED_ARMS_4010,
)


G1_29DOF_TORSOBASE_CFG = EntityCfg(
    init_state=copy.deepcopy(_SIMPLIFIED_INIT_STATE),
    spec_fn=get_g1_spec,
    articulation=EntityArticulationInfoCfg(
        actuators=tuple(copy.deepcopy(act) for act in g1_29dof_torsobase_delayed_actuator_cfgs),
        soft_joint_pos_limit_factor=0.95,
    ),
)
G1_29DOF_TORSOBASE_CLOG_CFG = EntityCfg(
    init_state=copy.deepcopy(_SIMPLIFIED_INIT_STATE),
    spec_fn=get_g1_spec,
    articulation=EntityArticulationInfoCfg(
        actuators=tuple(copy.deepcopy(act) for act in g1_29dof_torsobase_delayed_actuator_cfgs),
        soft_joint_pos_limit_factor=0.95,
    ),
)
G1_29DOF_TORSOBASE_POPSICLE_CFG = EntityCfg(
    init_state=copy.deepcopy(_POPSICLE_INIT_STATE),
    spec_fn=get_g1_spec,
    articulation=EntityArticulationInfoCfg(
        actuators=tuple(copy.deepcopy(act) for act in beyondmimic_g1_29dof_actuator_cfgs),
        soft_joint_pos_limit_factor=0.9,
    ),
)


G1_29Dof_TorsoBase_symmetric_augmentation_joint_mapping = [
    0,  # waist pitch
    1,
    2,  # waist roll / yaw
    9,  # left hip pitch -> right hip pitch
    10,  # left hip roll -> right hip roll
    11,  # left hip yaw -> right hip yaw
    12,  # left knee -> right knee
    13,  # left ankle pitch -> right ankle pitch
    14,  # left ankle roll -> right ankle roll
    3,  # right hip pitch -> left hip pitch
    4,  # right hip roll -> left hip roll
    5,  # right hip yaw -> left hip yaw
    6,  # right knee -> left knee
    7,  # right ankle pitch -> left ankle pitch
    8,  # right ankle roll -> left ankle roll
    22,  # left shoulder pitch -> right shoulder pitch
    23,  # left shoulder roll -> right shoulder roll
    24,  # left shoulder yaw -> right shoulder yaw
    25,  # left elbow -> right elbow
    26,  # left wrist roll -> right wrist roll
    27,  # left wrist pitch -> right wrist pitch
    28,  # left wrist yaw -> right wrist yaw
    15,  # right shoulder pitch -> left shoulder pitch
    16,  # right shoulder roll -> left shoulder roll
    17,  # right shoulder yaw -> left shoulder yaw
    18,  # right elbow -> left elbow
    19,  # right wrist roll -> left wrist roll
    20,  # right wrist pitch -> left wrist pitch
    21,  # right wrist yaw -> left wrist yaw
]

G1_29Dof_TorsoBase_symmetric_augmentation_joint_reverse_buf = [
    1,  # waist pitch
    -1,  # waist roll
    -1,  # waist yaw
    1,  # left hip pitch
    -1,  # left hip roll
    -1,  # left hip yaw
    1,  # left knee
    1,  # left ankle pitch
    -1,  # left ankle roll
    1,  # right hip pitch
    -1,  # right hip roll
    -1,  # right hip yaw
    1,  # right knee
    1,  # right ankle pitch
    -1,  # right ankle roll
    1,  # left shoulder pitch
    -1,  # left shoulder roll
    -1,  # left shoulder yaw
    1,  # left elbow
    -1,  # left wrist roll
    1,  # left wrist pitch
    -1,  # left wrist yaw
    1,  # right shoulder pitch
    -1,  # right shoulder roll
    -1,  # right shoulder yaw
    1,  # right elbow
    -1,  # right wrist roll
    1,  # right wrist pitch
    -1,  # right wrist yaw
]

# Ray-target helper uses membership only, but keep mjlab/MJCF body order here so
# any downstream debugging or config dumps match the current MuJoCo asset.

beyondmimic_action_scale: dict[str, float] = {}
for actuator_cfg in beyondmimic_g1_29dof_actuator_cfgs:
    effort = actuator_cfg.effort_limit
    stiffness = actuator_cfg.stiffness
    if effort is None or stiffness == 0.0:
        continue
    for joint_expr in actuator_cfg.target_names_expr:
        beyondmimic_action_scale[joint_expr] = 0.25 * effort / stiffness


__all__ = [
    "ACTUATOR_4010_EFFORT_LIMIT",
    "ACTUATOR_5020_EFFORT_LIMIT",
    "ACTUATOR_7520_14_EFFORT_LIMIT",
    "ACTUATOR_7520_22_EFFORT_LIMIT",
    "ACTUATOR_DUAL_5020_EFFORT_LIMIT",
    "ARMATURE_4010",
    "ARMATURE_5020",
    "ARMATURE_7520_14",
    "ARMATURE_7520_22",
    "DAMPING_4010",
    "DAMPING_5020",
    "DAMPING_7520_14",
    "DAMPING_7520_22",
    "DAMPING_RATIO",
    "GEARS_4010",
    "GEARS_5020",
    "GEARS_7520_14",
    "GEARS_7520_22",
    "G1_29DOF_TORSOBASE_CFG",
    "G1_29DOF_TORSOBASE_CLOG_CFG",
    "G1_29DOF_TORSOBASE_POPSICLE_CFG",
    "G1_MESHES_DIR",
    "G1_MJCF_PATH",
    "G1_29Dof_TorsoBase_symmetric_augmentation_joint_mapping",
    "G1_29Dof_TorsoBase_symmetric_augmentation_joint_reverse_buf",
    "NATURAL_FREQ",
    "ROTOR_INERTIAS_4010",
    "ROTOR_INERTIAS_5020",
    "ROTOR_INERTIAS_7520_14",
    "ROTOR_INERTIAS_7520_22",
    "STIFFNESS_4010",
    "STIFFNESS_5020",
    "STIFFNESS_7520_14",
    "STIFFNESS_7520_22",
    "beyondmimic_action_scale",
    "beyondmimic_g1_29dof_actuator_cfgs",
    "beyondmimic_g1_29dof_delayed_actuator_cfgs",
    "get_g1_spec",
    "g1_29dof_torsobase_delayed_actuator_cfgs",
]
