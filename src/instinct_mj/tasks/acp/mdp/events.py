"""Event terms of the ACP tracking task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

    from .curriculums import EpisodeStart
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
    curriculum_name: str = "episode_start",
    legal_start_max_penetration: float | None = None,
):
    """Set the clip start time of the reset envs from the episode start term (SuperMimic reset bins / uniform / phase 0).

    ``legal_start_max_penetration`` [m]: a start frame whose noise-free reference state penetrates the terrain by more than this is moved to the nearest legal frame of the
    clip (``legal_start.LegalStarts``, measured once on the first call; the phase-0 starts of the evaluation mode are never moved), so that the fallback of the reset penetration
    rejection (the noise-free reference) is itself clear of the terrain. The moved starts are counted in ``env.legal_start_stats`` (starts, moved).

    Must run after the motion reference reset (which samples the clip) and before the robot is reset to the reference."""
    curriculum: EpisodeStart = env.curriculum_manager.get_term_cfg(curriculum_name).func
    curriculum.set_start_phase(env_ids)
    if legal_start_max_penetration is None or curriculum.mode == "start":
        return
    legal = getattr(env, "legal_starts", None)
    if legal is None:
        from .legal_start import LegalStarts

        legal = env.legal_starts = LegalStarts(env, legal_start_max_penetration, env.event_manager.get_term_cfg("reset_robot").params)
        env.legal_start_stats = torch.zeros(2, dtype=torch.float64)
        print(legal.report())
    env.legal_start_stats += torch.tensor([len(env_ids), legal.snap_starts(curriculum.motion_buffer, env_ids)], dtype=torch.float64)
