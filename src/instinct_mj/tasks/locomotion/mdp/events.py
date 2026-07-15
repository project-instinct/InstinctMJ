# Copyright (c) 2022-2025, The Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define events for the learning environment."""

from __future__ import annotations

import math
from typing import Literal

import torch
from mjlab.entity import Entity
from mjlab.envs.mdp import dr
from mjlab.managers import SceneEntityCfg
from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields
from mjlab.utils.lab_api.math import sample_uniform

from instinct_mj.envs.mdp.events.randomization import uniform_mass_scale_to_alpha


@requires_model_fields(
    "body_mass",
    "body_ipos",
    "body_inertia",
    "body_iquat",
    recompute=RecomputeLevel.set_const,
)
def randomize_rigid_body_mass(
    env,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    mass_distribution_params: tuple[float, float] = (-0.5, 0.5),
    operation: Literal["add", "scale", "abs"] = "add",
    distribution: str = "uniform",
) -> None:
    """Randomize rigid-body pseudo-inertia."""
    pseudo_inertia_distribution = uniform_mass_scale_to_alpha if distribution == "uniform" else distribution
    if operation == "scale":
        alpha_range = _mass_scale_to_alpha_range(mass_distribution_params)
        dr.pseudo_inertia(
            env=env,
            env_ids=env_ids,
            alpha_range=alpha_range,
            distribution=pseudo_inertia_distribution,
            asset_cfg=asset_cfg,
        )
        return

    asset: Entity = env.scene[asset_cfg.name]
    local_body_ids = _selected_local_body_ids(asset, asset_cfg, env.device)
    model_body_ids = asset.indexing.body_ids[local_body_ids]
    target_env_ids = (
        torch.arange(env.num_envs, device=env.device, dtype=torch.int)
        if env_ids is None
        else env_ids.to(env.device, dtype=torch.int)
    )

    for local_body_id, model_body_id in zip(local_body_ids.tolist(), model_body_ids.tolist(), strict=True):
        default_masses = _default_body_masses(env, target_env_ids, model_body_id)
        default_mass = default_masses.flatten()[0]
        if not torch.allclose(default_masses, default_mass.expand_as(default_masses)):
            raise ValueError(
                "pseudo-inertia add/abs randomization requires a shared default mass per selected body."
            )
        mass = float(default_mass.item())
        if operation == "add":
            scale_range = (
                (mass + mass_distribution_params[0]) / mass,
                (mass + mass_distribution_params[1]) / mass,
            )
        elif operation == "abs":
            scale_range = (mass_distribution_params[0] / mass, mass_distribution_params[1] / mass)
        else:
            raise ValueError(f"Unsupported mass randomization operation: {operation}")
        alpha_range = _mass_scale_to_alpha_range(scale_range)
        dr.pseudo_inertia(
            env=env,
            env_ids=target_env_ids,
            alpha_range=alpha_range,
            distribution=pseudo_inertia_distribution,
            asset_cfg=SceneEntityCfg(asset_cfg.name, body_ids=[local_body_id]),
        )


def _mass_scale_to_alpha_range(scale_range: tuple[float, float]) -> tuple[float, float]:
    if scale_range[0] <= 0.0 or scale_range[1] <= 0.0:
        raise ValueError("Pseudo-inertia mass scale range must stay positive.")
    return (0.5 * math.log(scale_range[0]), 0.5 * math.log(scale_range[1]))


def _selected_local_body_ids(asset: Entity, asset_cfg: SceneEntityCfg, device: torch.device) -> torch.Tensor:
    all_local_body_ids = torch.arange(asset.indexing.body_ids.numel(), device=device, dtype=torch.long)
    return all_local_body_ids[asset_cfg.body_ids].reshape(-1)


def _default_body_masses(env, env_ids: torch.Tensor, model_body_id: int) -> torch.Tensor:
    default_body_mass = env.sim.get_default_field("body_mass")
    if "body_mass" in env.sim.per_world_default_fields:
        return default_body_mass[env_ids, model_body_id]
    return default_body_mass[model_body_id].reshape(1)


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
