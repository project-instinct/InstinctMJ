from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from mjlab.managers import SceneEntityCfg
from mjlab.sensor import ContactSensor, RayCastSensor
from mjlab.utils.lab_api.math import quat_apply_inverse

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def track_lin_vel_xy_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward tracking reference linear velocity (x/y in body frame).

  Mirrors the original InstinctLab ``track_lin_vel_xy_exp`` which uses
  ``env.command_manager.get_command(command_name)[:, :2]`` directly.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  lin_vel_error = torch.sum(
    torch.square(command[:, :2] - asset.data.root_link_lin_vel_b[:, :2]),
    dim=1,
  )
  return torch.exp(-lin_vel_error / max(std, 1e-6) ** 2)


def track_ang_vel_z_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward tracking reference yaw angular velocity (body frame).

  Mirrors the original InstinctLab ``track_ang_vel_z_exp`` which uses
  ``env.command_manager.get_command(command_name)[:, 2]`` directly.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  ang_vel_error = torch.square(command[:, 2] - asset.data.root_link_ang_vel_b[:, 2])
  return torch.exp(-ang_vel_error / max(std, 1e-6) ** 2)


def heading_error(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Compute heading command magnitude (InstinctLab-compatible).

  Mirrors the original: ``torch.abs(env.command_manager.get_command(command_name)[:, 2])``.
  """
  command = env.command_manager.get_command(command_name)
  return torch.abs(command[:, 2])


def dont_wait(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize standing still when there is a forward velocity command.

  Mirrors the original InstinctLab ``dont_wait`` which uses
  ``env.command_manager.get_command(command_name)[:, 0]`` directly.
  """
  asset: Entity = env.scene[asset_cfg.name]
  lin_vel_cmd_x = env.command_manager.get_command(command_name)[:, 0]
  lin_vel_x = asset.data.root_link_lin_vel_b[:, 0]

  return (lin_vel_cmd_x > 0.3) * (
    (lin_vel_x < 0.15).float() + (lin_vel_x < 0.0).float() + (lin_vel_x < -0.15).float()
  )


def stand_still(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  threshold: float = 0.15,
  offset: float = 1.0,
) -> torch.Tensor:
  """Penalize moving when there is no velocity command.

  Mirrors the original InstinctLab ``stand_still``.
  """
  asset: Entity = env.scene[asset_cfg.name]
  default_joint_pos = asset.data.default_joint_pos
  assert default_joint_pos is not None
  dof_error = torch.sum(torch.abs(asset.data.joint_pos - default_joint_pos), dim=1)

  cmd = env.command_manager.get_command(command_name)
  cmd_lin_norm = torch.norm(cmd[:, :2], dim=1)
  cmd_yaw_abs = torch.abs(cmd[:, 2])

  return (dof_error - offset) * (cmd_lin_norm < threshold) * (cmd_yaw_abs < threshold)


def feet_air_time(
  env: ManagerBasedRlEnv,
  command_name: str,
  vel_threshold: float,
  sensor_name: str,
) -> torch.Tensor:
  """Reward long steps taken by the feet for bipeds.

  Mirrors the original InstinctLab ``feet_air_time``.
  """
  contact_sensor: ContactSensor = env.scene[sensor_name]
  air_time = contact_sensor.data.current_air_time
  contact_time = contact_sensor.data.current_contact_time
  assert air_time is not None
  assert contact_time is not None

  in_contact = contact_time > 0.0
  in_mode_time = torch.where(in_contact, contact_time, air_time)
  single_stance = torch.sum(in_contact.int(), dim=1) == 1
  reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]

  # no reward for zero command
  cmd = env.command_manager.get_command(command_name)
  reward *= torch.logical_or(
    torch.norm(cmd[:, :2], dim=1) > vel_threshold,
    torch.abs(cmd[:, 2]) > vel_threshold,
  )
  return reward


def feet_slide(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  threshold: float = 1.0,
) -> torch.Tensor:
  """Penalize foot sliding speed while feet are in contact.

  Mirrors the original InstinctLab ``contact_slide``.
  ``threshold`` is the contact force threshold (Newtons) — matches the
  original semantics where ``net_forces_w_history.norm() > threshold``
  determines contact.
  """
  asset: Entity = env.scene[asset_cfg.name]
  sensor: ContactSensor = env.scene[sensor_name]

  # Original: contacts = net_forces_w_history[...].norm(dim=-1).max(dim=1)[0] > threshold
  force_history = sensor.data.force_history
  if force_history is not None:
    in_contact = torch.max(torch.linalg.vector_norm(force_history, dim=-1), dim=2)[0] > threshold
  else:
    force = sensor.data.force
    assert force is not None
    in_contact = torch.linalg.vector_norm(force, dim=-1) > threshold

  foot_vel_xy = asset.data.body_link_lin_vel_w[:, asset_cfg.body_ids, :2]
  slip_speed = torch.norm(foot_vel_xy, dim=-1)
  # Original: torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
  # No slip speed threshold/clamp — penalize raw speed
  return torch.sum(slip_speed * in_contact.float(), dim=1)


def ang_vel_xy_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return torch.sum(torch.square(asset.data.root_link_ang_vel_b[:, :2]), dim=1)


def joint_deviation_square(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  default_joint_pos = asset.data.default_joint_pos
  assert default_joint_pos is not None
  joint_error = asset.data.joint_pos[:, asset_cfg.joint_ids] - default_joint_pos[:, asset_cfg.joint_ids]
  return torch.sum(torch.square(joint_error), dim=1)


def joint_deviation_l1(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  default_joint_pos = asset.data.default_joint_pos
  assert default_joint_pos is not None
  joint_error = asset.data.joint_pos[:, asset_cfg.joint_ids] - default_joint_pos[:, asset_cfg.joint_ids]
  return torch.sum(torch.abs(joint_error), dim=1)


def link_orientation(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize non-flat link orientation using L2 squared kernel."""
  asset: Entity = env.scene[asset_cfg.name]
  link_quat = asset.data.body_link_quat_w[:, asset_cfg.body_ids[0], :]
  link_projected_gravity = quat_apply_inverse(link_quat, asset.data.gravity_vec_w)
  return torch.sum(torch.square(link_projected_gravity[:, :2]), dim=1)


def feet_orientation_contact(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  contact_force_threshold: float = 1.0,
) -> torch.Tensor:
  """Reward feet being oriented vertically when in contact with the ground."""
  asset: Entity = env.scene[asset_cfg.name]
  contact_sensor: ContactSensor = env.scene[sensor_name]

  body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]
  num_envs, num_feet = body_quat_w.shape[:2]

  gravity_w = asset.data.gravity_vec_w.unsqueeze(1).expand(-1, num_feet, -1)
  projected_gravity = quat_apply_inverse(
    body_quat_w.reshape(-1, 4), gravity_w.reshape(-1, 3)
  ).reshape(num_envs, num_feet, 3)
  orientation_error = torch.linalg.vector_norm(projected_gravity[:, :, :2], dim=-1)

  force_history = contact_sensor.data.force_history
  if force_history is not None:
    in_contact = (
      torch.max(torch.linalg.vector_norm(force_history, dim=-1), dim=2)[0]
      > contact_force_threshold
    )
  else:
    force = contact_sensor.data.force
    assert force is not None
    in_contact = torch.linalg.vector_norm(force, dim=-1) > contact_force_threshold

  return torch.sum(orientation_error * in_contact.float(), dim=1)


def feet_at_plane(
  env: ManagerBasedRlEnv,
  contact_sensor_name: str,
  left_height_scanner_name: str,
  right_height_scanner_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  height_offset: float = 0.035,
  contact_force_threshold: float = 1.0,
) -> torch.Tensor:
  """Reward feet being at certain height above the ground plane."""
  asset: Entity = env.scene[asset_cfg.name]
  body_pos_w = asset.data.body_link_pos_w

  body_ids = asset_cfg.body_ids
  if isinstance(body_ids, slice):
    body_ids = list(range(body_pos_w.shape[1]))[body_ids]
  else:
    body_ids = list(body_ids)

  contact_sensor: ContactSensor = env.scene[contact_sensor_name]
  force_history = contact_sensor.data.force_history
  if force_history is not None:
    is_contact = (
      torch.max(torch.linalg.vector_norm(force_history, dim=-1), dim=2)[0]
      > contact_force_threshold
    )
  else:
    force = contact_sensor.data.force
    assert force is not None
    is_contact = torch.linalg.vector_norm(force, dim=-1) > contact_force_threshold

  left_sensor: RayCastSensor = env.scene[left_height_scanner_name]
  right_sensor: RayCastSensor = env.scene[right_height_scanner_name]
  left_hit_z = left_sensor.data.hit_pos_w[..., 2]
  right_hit_z = right_sensor.data.hit_pos_w[..., 2]
  left_hit_z = torch.where(left_sensor.data.distances < 0.0, 0.0, left_hit_z)
  right_hit_z = torch.where(right_sensor.data.distances < 0.0, 0.0, right_hit_z)

  left_height = body_pos_w[:, body_ids[0], 2].unsqueeze(-1)
  right_height = body_pos_w[:, body_ids[1], 2].unsqueeze(-1)

  left_contact = is_contact[:, 0:1].float()
  right_contact = is_contact[:, 1:2].float()

  left_reward = torch.clamp(left_height - left_hit_z - height_offset, min=0.0, max=0.3) * left_contact
  right_reward = torch.clamp(right_height - right_hit_z - height_offset, min=0.0, max=0.3) * right_contact
  return torch.sum(left_reward, dim=-1) + torch.sum(right_reward, dim=-1)


def feet_close_xy_gauss(
  env: ManagerBasedRlEnv,
  threshold: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  std: float = 0.1,
) -> torch.Tensor:
  """Penalize when feet are too close together in the y distance."""
  asset: Entity = env.scene[asset_cfg.name]
  body_pos_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :]

  left_foot_xy = body_pos_w[:, 0, :2]
  right_foot_xy = body_pos_w[:, 1, :2]
  heading_w = asset.data.heading_w

  cos_heading = torch.cos(heading_w)
  sin_heading = torch.sin(heading_w)

  left_y = -sin_heading * left_foot_xy[:, 0] + cos_heading * left_foot_xy[:, 1]
  right_y = -sin_heading * right_foot_xy[:, 0] + cos_heading * right_foot_xy[:, 1]
  feet_distance_y = torch.abs(left_y - right_y)

  return torch.exp(-torch.clamp(threshold - feet_distance_y, min=0.0) / std**2) - 1


def volume_points_penetration(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  tolerance: float = 0.0,
) -> torch.Tensor:
  sensor = env.scene.sensors[sensor_name]
  penetration = sensor.data.penetration_offset
  points_vel = sensor.data.points_vel_w

  penetration_depth = torch.linalg.vector_norm(penetration.reshape(env.num_envs, -1, 3), dim=-1)
  in_obstacle = (penetration_depth > tolerance).float()
  points_vel_norm = torch.linalg.vector_norm(points_vel.reshape(env.num_envs, -1, 3), dim=-1)
  velocity_times_penetration = in_obstacle * (points_vel_norm + 1e-6) * penetration_depth
  return torch.sum(velocity_times_penetration, dim=-1)


def motors_power_square(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  normalize_by_stiffness: bool = True,
  normalize_by_num_joints: bool = False,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  # mjlab uses actuator_force instead of applied_torque
  power_j = asset.data.actuator_force * asset.data.joint_vel
  if normalize_by_stiffness:
    # mjlab: stiffness is configured on each BuiltinPositionActuator cfg.
    for actuator in asset.actuators:
      base_actuator = actuator.base_actuator
      target_ids = base_actuator.target_ids
      stiffness = base_actuator.cfg.stiffness
      if torch.is_tensor(stiffness):
        power_j[:, target_ids] /= stiffness.to(device=power_j.device, dtype=power_j.dtype)
      else:
        power_j[:, target_ids] /= float(stiffness)

  power_j = power_j[:, asset_cfg.joint_ids]
  power = torch.sum(torch.square(power_j), dim=-1)
  if normalize_by_num_joints and power_j.shape[-1] > 0:
    power = power / power_j.shape[-1]
  return power


def joint_vel_limits(
  env: ManagerBasedRlEnv,
  soft_ratio: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize joint velocities if they cross soft limits.

  Mirrors the original Isaac Lab ``joint_vel_limits``:
  ``abs(joint_vel) - soft_joint_vel_limits * soft_ratio`` clipped to [0, 1].
  In mjlab parkour G1, per-joint velocity limits are read directly from
  actuator cfg metadata (``velocity_limit``).
  """
  asset: Entity = env.scene[asset_cfg.name]

  joint_vel_limits = torch.zeros_like(asset.data.joint_vel)
  for actuator in asset.actuators:
    base_actuator = actuator.base_actuator
    target_ids = base_actuator.target_ids
    joint_vel_limits[:, target_ids] = float(base_actuator.cfg.velocity_limit)

  out_of_limits = (
    torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids])
    - joint_vel_limits[:, asset_cfg.joint_ids] * soft_ratio
  )
  # Clip to max error = 1 rad/s per joint to avoid huge penalties
  out_of_limits = out_of_limits.clip_(min=0.0, max=1.0)
  return torch.sum(out_of_limits, dim=1)


def applied_torque_limits_by_ratio(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  limit_ratio: float = 0.8,
) -> torch.Tensor:
  """Penalize when the applied torque exceeds a ratio of torque limits."""
  asset: Entity = env.scene[asset_cfg.name]

  if isinstance(asset_cfg.joint_ids, slice):
    selected_joint_ids = list(range(asset.num_joints))
  else:
    selected_joint_ids = list(asset_cfg.joint_ids)
  selected_joint_names = {asset.joint_names[j] for j in selected_joint_ids}

  actuator_forcerange = env.sim.model.actuator_forcerange
  if actuator_forcerange.ndim == 3:
    actuator_forcerange = actuator_forcerange[0]

  actuator_force = asset.data.actuator_force
  out_of_limits = []
  for actuator in asset.actuators:
    base_actuator = actuator.base_actuator
    target_names = list(base_actuator.target_names)
    ctrl_ids_local = base_actuator.ctrl_ids
    ctrl_ids_global = base_actuator.global_ctrl_ids
    for idx, joint_name in enumerate(target_names):
      if joint_name not in selected_joint_names:
        continue
      ctrl_id_local = int(ctrl_ids_local[idx])
      ctrl_id_global = int(ctrl_ids_global[idx])
      effort_limit = torch.max(torch.abs(actuator_forcerange[ctrl_id_global]))
      torque_abs = torch.abs(actuator_force[:, ctrl_id_local])
      out_of_limits.append(torch.clamp(torque_abs - effort_limit * limit_ratio, min=0.0))

  out_of_limits_tensor = torch.stack(out_of_limits, dim=-1)
  return torch.sum(torch.square(out_of_limits_tensor), dim=-1)


def undesired_contacts(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  threshold: float,
) -> torch.Tensor:
  contact_sensor: ContactSensor = env.scene[sensor_name]
  force_history = contact_sensor.data.force_history
  if force_history is not None:
    is_contact = torch.max(torch.linalg.vector_norm(force_history, dim=-1), dim=2)[0] > threshold
  else:
    force = contact_sensor.data.force
    assert force is not None
    is_contact = torch.linalg.vector_norm(force, dim=-1) > threshold

  return torch.sum(is_contact.float(), dim=1)
