"""Event terms of the ACP tracking task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

    from .curriculums import CurriculumStateInit
    from .sensors import PelvisRayCastSensor


def randomize_odometry(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    sensor_name: str = "height_scanner",
    xy_radius: float = 0.05,
    z_range: float = 0.05,
    delay_range_s: tuple[float, float] = (0.05, 0.1),
):
    """Per-episode odometry error of the estimated pelvis pose (paper Table S2).

    Drift: XY uniform in a disk of ``xy_radius``, Z U(+-z_range). Delay: U(delay_range_s) seconds. The delayed and drifted
    pose is used for the actor / planner scans and the contact-target and goal transforms."""
    sensor: PelvisRayCastSensor = env.scene[sensor_name]
    radius = xy_radius * torch.sqrt(torch.rand(len(env_ids), device=env.device))
    angle = 2.0 * torch.pi * torch.rand(len(env_ids), device=env.device)
    sensor.odometry_drift_w[env_ids, 0] = radius * torch.cos(angle)
    sensor.odometry_drift_w[env_ids, 1] = radius * torch.sin(angle)
    sensor.odometry_drift_w[env_ids, 2] = z_range * (2.0 * torch.rand(len(env_ids), device=env.device) - 1.0)
    sensor.odometry_delay_s[env_ids] = delay_range_s[0] + (delay_range_s[1] - delay_range_s[0]) * torch.rand(
        len(env_ids), device=env.device
    )


def reset_motion_start_phase(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    curriculum_name: str = "curriculum_state_init",
):
    """Set the clip start time of the reset envs from the initialization phase (Curriculum State Initialization).

    Must run after the motion reference reset (which samples the clip) and before the robot is reset to the reference."""
    curriculum: CurriculumStateInit = env.curriculum_manager.get_term_cfg(curriculum_name).func
    curriculum.set_start_phase(env_ids)
