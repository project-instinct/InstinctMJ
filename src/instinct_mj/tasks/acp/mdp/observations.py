"""Observation terms of the ACP tracking task (height scans, contact targets, reference quantities)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from mjlab.managers import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse

from .commands import AcpReferenceContacts, to_base_yaw_frame

if TYPE_CHECKING:
    from mjlab.entity import Entity

    from .sensors import PelvisRayCastSensor
    from mjlab.envs import ManagerBasedRlEnv

# Planner scan: 17 (x) x 11 (y) points, 0.1 m spacing, x in [-0.8, 0.8], y in [-0.5, 0.5], y-major (y outer, x inner).
SCAN_NUM_X = 17
SCAN_NUM_Y = 11
# World-model / critic scan (privileged, never the actor's; ``wm_scan="wide"``): 31 (x, 0 .. 1.5 m forward) x 15 (y, +-0.35 m) points at 0.05 m, y-major, in the yaw frame of the true
# pelvis (the open-source reproduction's CRITIC_SCAN, ((0, 1.5, 31), (-0.35, 0.35, 15))).
WIDE_SCAN_X = (0.0, 1.5, 31)
WIDE_SCAN_Y = (-0.35, 0.35, 15)
WIDE_SCAN_NUM_X, WIDE_SCAN_NUM_Y = WIDE_SCAN_X[2], WIDE_SCAN_Y[2]


def wide_scan_offsets(device="cpu") -> torch.Tensor:
    """(15 * 31, 2) offsets (x, y) of the wide scan in the pelvis yaw frame, y-major (y outer, x inner)."""
    gx, gy = torch.meshgrid(torch.linspace(*WIDE_SCAN_X, device=device), torch.linspace(*WIDE_SCAN_Y, device=device), indexing="xy")
    return torch.stack((gx.flatten(), gy.flatten()), -1)


# Tracker scan: the central 9 (x, x in [-0.4, 0.4]) x 7 (y, y in [-0.3, 0.3]) block of the planner scan.
TRACKER_X_SLICE = slice(4, 13)
TRACKER_Y_SLICE = slice(2, 9)
MISS_FILL_VALID_FRACTION = 0.75


def _scan_height(env: ManagerBasedRlEnv, sensor_name: str, offset: float) -> torch.Tensor:
    """pelvis_z - hit_z - offset of every ray, (N, SCAN_NUM_Y, SCAN_NUM_X).

    The rays start ``ray_offset`` above the pelvis (deploy: 5 m) and the frame z of the sensor data is the pelvis z: the
    true one for the critic scanner, the estimated (delayed, drifted) one for the actor / planner scanner."""
    sensor: PelvisRayCastSensor = env.scene[sensor_name]
    data = sensor.data
    height = data.frame_pos_w[:, 0, 2].unsqueeze(-1) - data.hit_pos_w[..., 2] - offset
    miss_height = sensor.cfg.max_distance - sensor.cfg.ray_offset - offset
    miss = (data.distances < 0).view(-1, SCAN_NUM_Y, SCAN_NUM_X)
    return fill_scan_misses(height.view(-1, SCAN_NUM_Y, SCAN_NUM_X), miss, miss_height)


def fill_scan_misses(height: torch.Tensor, miss: torch.Tensor, miss_height: float) -> torch.Tensor:
    """Scan values (N, SCAN_NUM_Y, SCAN_NUM_X) of the rays that hit; ``miss`` marks the rays that did not.

    The mesh ray cast misses rays inside narrow bands along some mesh edges (the sensor casts them a second time shifted; what still
    misses is filled here). A missed ray whose 3 x 3 neighbourhood is at least ``MISS_FILL_VALID_FRACTION`` valid is a cast artifact,
    not empty space: it takes the largest hit height of its valid neighbours (smallest encoded value, conservative). Real
    misses (nothing below the pelvis within range, the border of the terrain) have mostly missed neighbours and stay at the miss height."""
    valid = (~miss).float().unsqueeze(1)
    neighbours = F.avg_pool2d(F.pad(torch.ones_like(valid), (1, 1, 1, 1)), 3, 1) * 9 - 1  # in-bounds neighbours of every cell
    valid_neighbours = F.avg_pool2d(F.pad(valid, (1, 1, 1, 1)), 3, 1) * 9 - valid
    near = -F.max_pool2d(F.pad(torch.where(miss.unsqueeze(1), torch.full_like(valid, -1e6), -height.unsqueeze(1)), (1, 1, 1, 1), value=-1e6), 3, 1)
    fill = miss & (valid_neighbours.squeeze(1) >= MISS_FILL_VALID_FRACTION * neighbours.squeeze(1))
    height = torch.where(fill, near.squeeze(1), height)
    return torch.where(miss & ~fill, torch.full_like(height, miss_height), height)


def base_lin_vel(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Pelvis linear velocity in the pelvis frame (the base is the pelvis, as in the authors' deploy stack)."""
    term: AcpReferenceContacts = env.command_manager.get_term(command_name)
    return quat_apply_inverse(term.base_quat_w, term.base_lin_vel_w)


def base_ang_vel(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Pelvis angular velocity in the pelvis frame (the IMU of the authors' deploy stack is in the pelvis)."""
    term: AcpReferenceContacts = env.command_manager.get_term(command_name)
    return quat_apply_inverse(term.base_quat_w, term.base_ang_vel_w)


def projected_gravity(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Gravity direction in the pelvis frame."""
    term: AcpReferenceContacts = env.command_manager.get_term(command_name)
    return quat_apply_inverse(term.base_quat_w, term.robot.data.gravity_vec_w)


def critic_height_scan(
    env: ManagerBasedRlEnv, sensor_name: str = "critic_height_scanner", offset: float = 0.5
) -> torch.Tensor:
    """Full 17 x 11 scan heights of the true pelvis pose (no noise, drift or delay), (N, 187), y-major."""
    return _scan_height(env, sensor_name, offset).flatten(1)


def tracker_height_scan(env: ManagerBasedRlEnv, sensor_name: str = "height_scanner", offset: float = 0.5) -> torch.Tensor:
    """9 x 7 tracker scan = central block of the planner scan, (N, 63), y-major (y outer, x inner)."""
    return _scan_height(env, sensor_name, offset)[:, TRACKER_Y_SLICE, TRACKER_X_SLICE].flatten(1)


def planner_height_scan(
    env: ManagerBasedRlEnv,
    sensor_name: str = "height_scanner",
    offset: float = 0.5,
    z_noise_std: float = 0.03,
) -> torch.Tensor:
    """Planner input: 17 x 11 points (x, y, base_z - hit_z - offset), (N, 561), y-major; Gaussian noise on z only."""
    sensor: PelvisRayCastSensor = env.scene[sensor_name]
    height = _scan_height(env, sensor_name, offset).flatten(1)
    height = height + z_noise_std * torch.randn_like(height)
    xy = sensor.cfg.pattern.generate_rays(None, env.device)[0][:, :2]  # (rays, 2) in the yaw-aligned base frame
    return torch.cat((xy.unsqueeze(0).expand(height.shape[0], -1, -1), height.unsqueeze(-1)), dim=-1).flatten(1)


def contact_targets(env: ManagerBasedRlEnv, command_name: str, noise: float = 0.1) -> torch.Tensor:
    """Contact targets in the pelvis yaw frame (estimated pose) (N, 27), uniform +-noise on the valid entries only."""
    term: AcpReferenceContacts = env.command_manager.get_term(command_name)
    pos_b, valid = term.targets(estimated=True)
    pos_b = pos_b + noise * (2.0 * torch.rand_like(pos_b) - 1.0) * valid.unsqueeze(-1)
    return pos_b.flatten(1)


def reference_kinematics(
    env: ManagerBasedRlEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reference base lin / ang velocity in the reference base frame (3 + 3), reference joint pos / vel of the joints."""
    term: AcpReferenceContacts = env.command_manager.get_term(command_name)
    ref = term.motion_reference.reference_frame
    quat = term.ref_base_quat_w
    lin_vel_b = quat_apply_inverse(quat, term.ref_base_lin_vel_w)
    ang_vel_b = quat_apply_inverse(quat, term.ref_base_ang_vel_w)
    return torch.cat(
        (lin_vel_b, ang_vel_b, ref.joint_pos[:, 0, asset_cfg.joint_ids], ref.joint_vel[:, 0, asset_cfg.joint_ids]), dim=-1
    )


def reference_contact_pos_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Reference contact-link positions of the current reference frame in the robot's base-yaw frame, (N, 27)."""
    term: AcpReferenceContacts = env.command_manager.get_term(command_name)
    return to_base_yaw_frame(term.ref_link_pos_w, term.base_pos_w, term.base_quat_w).flatten(1)


def reference_contact_flags(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Instantaneous reference contact flags, (N, 9)."""
    term: AcpReferenceContacts = env.command_manager.get_term(command_name)
    return term.ref_contact_flag.float()
