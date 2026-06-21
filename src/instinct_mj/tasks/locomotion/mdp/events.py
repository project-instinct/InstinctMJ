# Copyright (c) 2022-2025, The Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define events for the learning environment."""

from __future__ import annotations

from typing import Literal

import torch
from mjlab.entity import Entity
from mjlab.envs.mdp import dr
from mjlab.managers import SceneEntityCfg
from mjlab.utils.lab_api.math import sample_uniform


def randomize_rigid_body_mass(
    env,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    mass_distribution_params: tuple[float, float] = (-0.5, 0.5),
    operation: Literal["add", "scale", "abs"] = "add",
) -> None:
    """Randomize rigid-body mass."""
    dr.body_mass(
        env=env,
        env_ids=env_ids,
        ranges=mass_distribution_params,
        operation=operation,
        asset_cfg=asset_cfg,
    )


def reset_joints_by_scale(
    env,
    env_ids: torch.Tensor | None,
    position_range: tuple[float, float],
    velocity_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Reset joint state by scaling default joint positions and velocities."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

    asset: Entity = env.scene[asset_cfg.name]
    joint_pos = asset.data.default_joint_pos[env_ids][:, asset_cfg.joint_ids].clone()
    joint_pos *= sample_uniform(*position_range, joint_pos.shape, env.device)
    joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids][:, asset_cfg.joint_ids]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])

    joint_vel = asset.data.default_joint_vel[env_ids][:, asset_cfg.joint_ids].clone()
    joint_vel *= sample_uniform(*velocity_range, joint_vel.shape, env.device)

    joint_ids = asset_cfg.joint_ids
    if isinstance(joint_ids, list):
        joint_ids = torch.tensor(joint_ids, device=env.device)

    asset.write_joint_state_to_sim(
        joint_pos.view(len(env_ids), -1),
        joint_vel.view(len(env_ids), -1),
        env_ids=env_ids,
        joint_ids=joint_ids,
    )
