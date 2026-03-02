"""G1 parkour AMP task config factories.

Mirrors the original InstinctLab ``g1_parkour_target_amp_cfg.py`` using
mjlab-native factory functions.  Config classes (``G1ParkourRoughEnvCfg``,
``G1ParkourEnvCfg``, etc.) are replaced by a single factory
``instinct_g1_parkour_amp_final_cfg(play, shoe)`` that returns a fully-built
``ManagerBasedRlEnvCfg``.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.viewer.viewer_config import ViewerConfig

from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg

from instinct_mjlab.assets.unitree_g1 import (
  G1_29DOF_INSTINCTLAB_JOINT_ORDER,
  G1_29Dof_TorsoBase_symmetric_augmentation_joint_mapping,
  G1_29Dof_TorsoBase_symmetric_augmentation_joint_reverse_buf,
  G1_MJCF_PATH,
  beyondmimic_action_scale,
  beyondmimic_g1_29dof_delayed_actuator_cfgs,
  get_g1_assets,
)
from instinct_mjlab.motion_reference import MotionReferenceManagerCfg
from instinct_mjlab.motion_reference.motion_files.amass_motion_cfg import (
  AmassMotionCfg as AmassMotionCfgBase,
)
from instinct_mjlab.motion_reference.utils import motion_interpolate_bilinear
from instinct_mjlab.utils.motion_validation import resolve_datasets_root
from instinct_mjlab.tasks.parkour.config.parkour_env_cfg import (
  set_parkour_amp_observations,
  set_parkour_basic_settings,
  set_parkour_commands,
  set_parkour_curriculum,
  set_parkour_events,
  set_parkour_observations,
  set_parkour_play_overrides,
  set_parkour_rewards,
  set_parkour_scene_visual_style,
  set_parkour_scene_sensors,
  set_parkour_terminations,
  set_parkour_terrain,
)

_PARKOUR_TASK_DIR = Path(__file__).resolve().parents[2]
_PARKOUR_G1_WITH_SHOE_MJCF_PATH = (
  _PARKOUR_TASK_DIR / "mjcf" / "g1_29dof_torsoBase_popsicle_with_shoe.xml"
)
_PARKOUR_MOTION_REFERENCE_DIR = (
  resolve_datasets_root() / "data&model" / "parkour_motion_reference"
).resolve()
_PARKOUR_FILTERED_MOTION_YAML = (
  _PARKOUR_MOTION_REFERENCE_DIR / "parkour_motion_without_run.yaml"
)


# ---------------------------------------------------------------------------
# Motion reference configs (mirrors InstinctLab AmassMotionCfg)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class AmassMotionCfg(AmassMotionCfgBase):
  """Parkour AMASS motion buffer config."""

  path: str = str(_PARKOUR_MOTION_REFERENCE_DIR)
  retargetting_func: object | None = None
  filtered_motion_selection_filepath: str | None = str(_PARKOUR_FILTERED_MOTION_YAML)
  motion_start_from_middle_range: list[float] = field(default_factory=lambda: [0.0, 0.9])
  motion_start_height_offset: float = 0.0
  ensure_link_below_zero_ground: bool = False
  buffer_device: str = "output_device"
  motion_interpolate_func: object = field(default_factory=lambda: motion_interpolate_bilinear)
  velocity_estimation_method: str = "frontward"


def _make_motion_reference_cfg() -> MotionReferenceManagerCfg:
  """Build parkour motion reference manager config."""
  # Convert InstinctLab symmetric augmentation joint mapping into the
  # MuJoCo/MJCF joint index order used by mjlab entities.
  model = mujoco.MjModel.from_xml_path(G1_MJCF_PATH)
  mjlab_joint_order: list[str] = []
  for joint_id in range(model.njnt):
    if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
      continue
    joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
    assert joint_name is not None
    mjlab_joint_order.append(joint_name)

  instinct_joint_order = list(G1_29DOF_INSTINCTLAB_JOINT_ORDER)
  instinct_name_to_idx = {name: idx for idx, name in enumerate(instinct_joint_order)}
  mjlab_name_to_idx = {name: idx for idx, name in enumerate(mjlab_joint_order)}

  symmetric_joint_mapping_mjlab: list[int] = []
  symmetric_joint_reverse_buf_mjlab: list[int] = []
  for joint_name in mjlab_joint_order:
    inst_idx = instinct_name_to_idx[joint_name]
    mirrored_inst_idx = G1_29Dof_TorsoBase_symmetric_augmentation_joint_mapping[inst_idx]
    mirrored_joint_name = instinct_joint_order[mirrored_inst_idx]
    symmetric_joint_mapping_mjlab.append(mjlab_name_to_idx[mirrored_joint_name])
    symmetric_joint_reverse_buf_mjlab.append(
      G1_29Dof_TorsoBase_symmetric_augmentation_joint_reverse_buf[inst_idx]
    )

  return MotionReferenceManagerCfg(
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
    symmetric_augmentation_link_mapping=[0, 1, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12],
    symmetric_augmentation_joint_mapping=symmetric_joint_mapping_mjlab,
    symmetric_augmentation_joint_reverse_buf=symmetric_joint_reverse_buf_mjlab,
    frame_interval_s=0.02,
    update_period=0.02,
    num_frames=10,
    motion_buffers={"run_walk": AmassMotionCfg()},
    mp_split_method="Even",
  )


def _set_motion_reference_sensor(cfg: ManagerBasedRlEnvCfg) -> None:
  """Attach motion reference manager to the scene sensors."""
  motion_reference_cfg = _make_motion_reference_cfg()
  existing_sensors = tuple(
    sensor_cfg
    for sensor_cfg in cfg.scene.sensors
    if sensor_cfg.name != motion_reference_cfg.name
  )
  cfg.scene.sensors = existing_sensors + (motion_reference_cfg,)


# ---------------------------------------------------------------------------
# Shoe spec factory
# ---------------------------------------------------------------------------


def _parkour_g1_with_shoe_spec() -> mujoco.MjSpec:
  """Build MjSpec for the G1 robot with shoe mesh."""
  spec = mujoco.MjSpec.from_file(str(_PARKOUR_G1_WITH_SHOE_MJCF_PATH))
  spec.assets = get_g1_assets(spec.meshdir)
  # InstinctLab shoe URDF does not include robot-mounted lights.
  # Remove embedded per-robot lights to avoid localized over-bright spots.
  for body in spec.bodies:
    for light in tuple(body.lights):
      spec.delete(light)
  return spec


def _apply_shoe_config(cfg: ManagerBasedRlEnvCfg) -> None:
  """Apply shoe-specific adjustments to a parkour env cfg (in-place).

  Mirrors the ``ShoeConfigMixin.apply_shoe_config()`` from the original
  InstinctLab ``g1_parkour_target_amp_cfg.py``.
  """
  # Replace robot spec with shoe variant
  robot_cfg_with_shoe = copy.deepcopy(cfg.scene.entities["robot"])
  robot_cfg_with_shoe.spec_fn = _parkour_g1_with_shoe_spec
  # Keep the URDF-authored collision setup as-is to match InstinctLab semantics.
  # Even though shoe foot collision geoms now carry explicit names, we still
  # avoid reapplying asset-zoo collision overrides for parity.
  robot_cfg_with_shoe.collisions = tuple()
  cfg.scene.entities["robot"] = robot_cfg_with_shoe

  # Adjust leg volume points z-range for shoes
  leg_volume_points = next(
    sensor_cfg for sensor_cfg in cfg.scene.sensors if sensor_cfg.name == "leg_volume_points"
  )
  leg_volume_points.points_generator.z_min = -0.063
  leg_volume_points.points_generator.z_max = -0.023

  # Adjust feet_at_plane height offset for shoes
  cfg.rewards["feet_at_plane"].params["height_offset"] = 0.058


def _apply_play_overrides(cfg: ManagerBasedRlEnvCfg) -> None:
  """Apply play-mode-specific overrides to a parkour env cfg (in-place).

  Mirrors ``G1ParkourRoughEnvCfg_PLAY.__post_init__`` from the original
  InstinctLab ``g1_parkour_target_amp_cfg.py``.
  """
  # Viewer
  cfg.viewer = ViewerConfig(
    lookat=(0.0, 0.75, 0.0),
    distance=4.123105625617661,
    elevation=-14.036243467926479,
    azimuth=180.0,
    origin_type=ViewerConfig.OriginType.WORLD,
    entity_name=None,
  )
  # Keep debug visualization for all environments after replacing viewer config.
  cfg.viewer.debug_vis_show_all_envs = True


def _set_world_free_viewer(cfg: ManagerBasedRlEnvCfg) -> None:
  """Ensure viewer uses world-origin free camera instead of robot tracking."""
  cfg.viewer.origin_type = ViewerConfig.OriginType.WORLD
  cfg.viewer.entity_name = None
  cfg.viewer.body_name = None


# ---------------------------------------------------------------------------
# G1-specific actuator setup (from original env_cfgs.py)
# ---------------------------------------------------------------------------


def _set_parkour_actuators(cfg: ManagerBasedRlEnvCfg) -> None:
  """Set G1-specific actuators and action scale for parkour (in-place).

  Mirrors the original InstinctLab ``G1ParkourRoughEnvCfg.__post_init__``
  where ``beyondmimic_g1_29dof_delayed_actuator_cfgs`` and
  ``beyondmimic_action_scale`` are applied.
  """
  robot_cfg = cfg.scene.entities["robot"]
  assert robot_cfg.articulation is not None
  robot_cfg.articulation.actuators = copy.deepcopy(
    beyondmimic_g1_29dof_delayed_actuator_cfgs
  )

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  # Keep Parkour action dimensions aligned with InstinctLab 29-DoF joint order.
  joint_pos_action.actuator_names = G1_29DOF_INSTINCTLAB_JOINT_ORDER
  joint_pos_action.preserve_order = True
  joint_pos_action.scale = copy.deepcopy(beyondmimic_action_scale)


# ---------------------------------------------------------------------------
# Base parkour env builder (merges original env_cfgs.py logic)
# ---------------------------------------------------------------------------


def instinct_g1_parkour_amp_env_cfg(
  *,
  play: bool = False,
  shoe: bool = True,
) -> ManagerBasedRlEnvCfg:
  """Build the base G1 parkour AMP environment configuration.

  Mirrors the original InstinctLab ``G1ParkourRoughEnvCfg`` assembly:
  starts from the tracking base, then applies parkour-specific MDP
  settings (terrain, sensors, observations, rewards, etc.) and
  G1-specific actuators.

  Args:
    play: If True, apply play-mode overrides (fewer envs, relaxed
      termination, etc.).
    shoe: If True, apply shoe-specific adjustments (default is True,
      matching the original ``G1ParkourEnvCfg`` behavior expected by
      parkour task entry points).

  Returns:
    A ``ManagerBasedRlEnvCfg`` instance with parkour settings applied.
  """
  # Scene settings (start from tracking base with G1 robot)
  cfg = unitree_g1_flat_tracking_env_cfg(play=play, has_state_estimation=True)
  _set_world_free_viewer(cfg)
  # Match InstinctLab parkour init height (G1_CFG.init_state.pos = (0, 0, 0.9)).
  cfg.scene.entities["robot"].init_state.pos = (0.0, 0.0, 0.9)

  # Basic settings
  set_parkour_basic_settings(cfg)
  # G1-specific actuators
  _set_parkour_actuators(cfg)
  # Terrain
  set_parkour_terrain(cfg, play=play)
  # Scene visual style
  set_parkour_scene_visual_style(cfg)
  # Scene sensors
  set_parkour_scene_sensors(cfg)
  _set_motion_reference_sensor(cfg)

  # MDP settings
  set_parkour_commands(cfg)
  set_parkour_observations(cfg)
  set_parkour_amp_observations(cfg)
  set_parkour_rewards(cfg)
  set_parkour_curriculum(cfg)
  set_parkour_terminations(cfg)
  set_parkour_events(cfg)

  # Apply shoe-specific adjustments (matches original G1ParkourEnvCfg).
  if shoe:
    _apply_shoe_config(cfg)

  # general settings
  # simulation settings
  # update sensor update periods
  # lights
  if play:
    set_parkour_play_overrides(cfg)
  return cfg


# ---------------------------------------------------------------------------
# Public factory functions
# ---------------------------------------------------------------------------


def instinct_g1_parkour_amp_final_cfg(
  *,
  play: bool = False,
  shoe: bool = True,
) -> ManagerBasedRlEnvCfg:
  """Create the final G1 parkour AMP env configuration.

  Args:
    play: If True, apply play-mode overrides (fewer envs, relaxed
      termination, etc.).
    shoe: If True, apply shoe-specific adjustments (default is True,
      matching the original ``G1ParkourEnvCfg``).

  Returns:
    A fully-built ``ManagerBasedRlEnvCfg`` instance.
  """
  # Build base parkour config (already includes play overrides if requested)
  cfg = instinct_g1_parkour_amp_env_cfg(play=play, shoe=shoe)

  # Apply play-mode viewer overrides
  if play:
    _apply_play_overrides(cfg)
    _set_world_free_viewer(cfg)

  return cfg


# ---------------------------------------------------------------------------
# Backward-compatible class aliases (thin wrappers for registration)
# ---------------------------------------------------------------------------


class G1ParkourEnvCfg(ManagerBasedRlEnvCfg):
  """G1 parkour train config (with shoe)."""

  def __init__(self):
    cfg = instinct_g1_parkour_amp_final_cfg(play=False, shoe=True)
    super().__init__(**{f.name: getattr(cfg, f.name) for f in cfg.__dataclass_fields__.values()})


class G1ParkourEnvCfg_PLAY(ManagerBasedRlEnvCfg):
  """G1 parkour play config (with shoe)."""

  def __init__(self):
    cfg = instinct_g1_parkour_amp_final_cfg(play=True, shoe=True)
    super().__init__(**{f.name: getattr(cfg, f.name) for f in cfg.__dataclass_fields__.values()})
