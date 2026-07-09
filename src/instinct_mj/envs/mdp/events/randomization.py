from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp import dr
from mjlab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields
from mjlab.sensor import RayCastSensor as RayCaster
from mjlab.utils.lab_api import math as math_utils

from instinct_mj.sensors.grouped_ray_caster import GroupedRayCaster

if TYPE_CHECKING:
    from mjlab.entity import Entity as Articulation
    from mjlab.entity import Entity as RigidObject
    from mjlab.sensor import CameraSensor as Camera
    from mjlab.sensor import RayCastSensor as RayCasterCamera

    from instinct_mj.sensors.grouped_ray_caster import GroupedRayCasterCamera

ManagerBasedEnv = ManagerBasedRlEnv

# DR engine builtin `add` uses defaults; this custom op adds sampled offsets to current model values.
_DR_ADD_CURRENT = dr.Operation(
    name="add_current",
    initialize=torch.zeros_like,
    combine=torch.add,
    uses_defaults=False,
)

uniform_mass_scale_to_alpha = dr.Distribution(
    name="uniform_mass_scale_to_alpha",
    sample=lambda lo, hi, shape, device: 0.5 * torch.log(
        math_utils.sample_uniform(
            torch.exp(2.0 * lo),
            torch.exp(2.0 * hi),
            shape,
            device=device,
        )
    ),
)


def _randomize_prop_by_op(
    data: torch.Tensor,
    distribution_params: tuple[float, float],
    env_ids: torch.Tensor | slice | None,
    property_ids: torch.Tensor | slice | None,
    operation: Literal["add", "scale", "abs"] = "add",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
) -> torch.Tensor:
    data_randomized = data.clone()
    env_selector = slice(None) if env_ids is None else env_ids
    property_selector = slice(None) if property_ids is None else property_ids

    if data_randomized.ndim == 1:
        index = env_selector
    elif isinstance(env_selector, torch.Tensor) and isinstance(property_selector, torch.Tensor):
        # Reshape for outer-product indexing: env_ids[:, None] x property_ids[None, :]
        index = (env_selector.long().unsqueeze(1), property_selector.long().unsqueeze(0))
    else:
        index = (env_selector, property_selector)
    target = data_randomized[index]

    if distribution == "uniform":
        sampled = math_utils.sample_uniform(
            distribution_params[0],
            distribution_params[1],
            target.shape,
            device=data_randomized.device,
        )
    elif distribution == "log_uniform":
        sampled = math_utils.sample_log_uniform(
            distribution_params[0],
            distribution_params[1],
            target.shape,
            device=data_randomized.device,
        )
    elif distribution == "gaussian":
        sampled = math_utils.sample_gaussian(
            distribution_params[0],
            distribution_params[1],
            target.shape,
            device=data_randomized.device,
        )

    if operation == "add":
        randomized_target = target + sampled
    elif operation == "scale":
        randomized_target = target * sampled
    elif operation == "abs":
        randomized_target = sampled

    data_randomized[index] = randomized_target
    return data_randomized


@requires_model_fields("body_ipos", "body_iquat", recompute=RecomputeLevel.set_const)
def randomize_rigid_body_com(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    com_range: dict[str, tuple[float, float]],
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """Randomize rigid-body COM offsets from ``com_range``."""
    axis_map = {"x": 0, "y": 1, "z": 2}
    ranges = {axis_map[axis_name]: axis_range for axis_name, axis_range in com_range.items() if axis_name in axis_map}
    if len(ranges) == 0:
        return

    # Apply sampled offsets on top of the current COM values.
    dr.body_ipos(
        env=env,
        env_ids=env_ids,
        ranges=ranges,
        asset_cfg=asset_cfg,
        distribution=distribution,
        operation=_DR_ADD_CURRENT,
        axes=sorted(ranges.keys()),
    )


def randomize_default_joint_pos(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    offset_distribution_params: tuple[float, float],
    operation: Literal["add", "scale", "abs"] = "add",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """
    Randomize the joint default positions which may be different from URDF due to calibration errors.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # save nominal value for export
    asset.data.default_joint_pos_nominal = torch.clone(asset.data.default_joint_pos[0])

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=env.device)

    # resolve joint indices
    if asset_cfg.joint_ids == slice(None):
        joint_ids = slice(None)  # for optimization purposes
    else:
        joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.int, device=env.device)

    if offset_distribution_params is not None:
        pos = asset.data.default_joint_pos.to(env.device).clone()
        pos = _randomize_prop_by_op(
            pos, offset_distribution_params, env_ids, joint_ids, operation=operation, distribution=distribution
        )[env_ids][:, joint_ids]

        if env_ids != slice(None) and joint_ids != slice(None):
            env_ids = env_ids[:, None]  # type: ignore
        asset.data.default_joint_pos[env_ids, joint_ids] = pos
        # update the offset in action since it is not updated automatically
        env.action_manager.get_term("joint_pos")._offset[env_ids, joint_ids] = pos


def randomize_ray_offsets(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    offset_pose_ranges: dict[str, tuple[float, float]],
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """
    Randomize the ray_starts and ray_directions of the sensor to mimic the sensor installation errors.

    Args:
        - offset_pose_ranges: (dict[str, tuple[float, float]])
            where keys are ["x", "y", "z", "roll", "pitch", "yaw"]
            and values are tuples representing the range for each component.
        - distribution: (str) "uniform" or "log_uniform" or "gaussian", determines the distribution of the randomization.
    """
    num_env_ids = env.scene.num_envs if env_ids is None else len(env_ids)
    # extract the used quantities (to enable type-hinting)
    sensor: RayCaster | GroupedRayCaster = env.scene[asset_cfg.name]
    ray_starts = sensor.ray_starts[env_ids]  # (num_envs, num_rays, 3)
    ray_directions = sensor.ray_directions[env_ids]  # (num_envs, num_rays, 3)
    # sample from given range
    range_list = [offset_pose_ranges.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=ray_starts.device)  # (6, 2)
    rand_samples = (
        math_utils.sample_uniform(
            ranges[:, 0],
            ranges[:, 1],
            (num_env_ids, 6),
            device=ray_starts.device,
        )[..., None, :]
        .repeat(1, sensor.num_rays, 1)
        .flatten(0, 1)
    )
    position_samples = rand_samples[..., :3]  # (num_envs * num_rays, 3)
    rotation_samples = math_utils.quat_from_euler_xyz(
        rand_samples[..., 3],
        rand_samples[..., 4],
        rand_samples[..., 5],
    )  # (num_envs * num_rays, 4) (w, x, y, z)
    # apply the randomization
    ray_starts += position_samples.reshape(ray_starts.shape)
    ray_directions = math_utils.quat_apply(rotation_samples.reshape(*ray_directions.shape[:-1], 4), ray_directions)

    sensor.ray_starts[env_ids] = ray_starts
    sensor.ray_directions[env_ids] = ray_directions


def randomize_camera_offsets(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    offset_pose_ranges: dict[str, tuple[float, float]],
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """
    Randomize the camera offset pose to mimic the sensor installation errors, which lead to imperfect camera calibration.

    Args:
        - offset_pose_ranges: (dict[str, tuple[float, float]])
            where keys are ["x", "y", "z", "roll", "pitch", "yaw"]
            and values are tuples representing the range for each component.
        - distribution: (str) "uniform" or "log_uniform" or "gaussian", determines the distribution of the randomization.
    """
    # extract the used quantities (to enable type-hinting), as well as all inherited classes
    sensor: Camera | RayCasterCamera | GroupedRayCasterCamera = env.scene[asset_cfg.name]

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=sensor._device)

    # get the camera pose
    num_sensors = sensor.data.pos_w.shape[0]
    camera_offset_pos = torch.tensor(list(sensor.cfg.offset.pos), device=sensor._device).repeat(num_sensors, 1)
    camera_quat_w = math_utils.convert_camera_frame_orientation_convention(
        torch.tensor([sensor.cfg.offset.rot], device=sensor._device),
        origin=sensor.cfg.offset.convention,
        target="world",
    )
    camera_offset_quat = camera_quat_w.repeat(num_sensors, 1)
    camera_offset_pos = camera_offset_pos[env_ids]
    camera_offset_quat = camera_offset_quat[env_ids]

    # sample from given range
    camera_offset_pos[..., 0] = _randomize_prop_by_op(
        camera_offset_pos[..., 0].unsqueeze(-1),
        offset_pose_ranges.get("x", (0.0, 0.0)),
        None,
        slice(None),
        operation="add",
        distribution=distribution,
    ).squeeze(-1)

    camera_offset_pos[..., 1] = _randomize_prop_by_op(
        camera_offset_pos[..., 1].unsqueeze(-1),
        offset_pose_ranges.get("y", (0.0, 0.0)),
        None,
        slice(None),
        operation="add",
        distribution=distribution,
    ).squeeze(-1)

    camera_offset_pos[..., 2] = _randomize_prop_by_op(
        camera_offset_pos[..., 2].unsqueeze(-1),
        offset_pose_ranges.get("z", (0.0, 0.0)),
        None,
        slice(None),
        operation="add",
        distribution=distribution,
    ).squeeze(-1)

    camera_euler_w = math_utils.euler_xyz_from_quat(camera_offset_quat)

    camera_euler_roll = _randomize_prop_by_op(
        camera_euler_w[0].unsqueeze(-1),
        offset_pose_ranges.get("roll", (0.0, 0.0)),
        None,
        slice(None),
        operation="add",
        distribution=distribution,
    ).squeeze(-1)

    camera_euler_pitch = _randomize_prop_by_op(
        camera_euler_w[1].unsqueeze(-1),
        offset_pose_ranges.get("pitch", (0.0, 0.0)),
        None,
        slice(None),
        operation="add",
        distribution=distribution,
    ).squeeze(-1)

    camera_euler_yaw = _randomize_prop_by_op(
        camera_euler_w[2].unsqueeze(-1),
        offset_pose_ranges.get("yaw", (0.0, 0.0)),
        None,
        slice(None),
        operation="add",
        distribution=distribution,
    ).squeeze(-1)

    camera_offset_quat = math_utils.quat_from_euler_xyz(
        camera_euler_roll,
        camera_euler_pitch,
        camera_euler_yaw,
    )
    camera_pos_w, camera_quat_w = sensor._compute_view_world_poses(env_ids)
    camera_pos_w += math_utils.quat_apply(camera_quat_w, camera_offset_pos)
    camera_quat_w = math_utils.quat_mul(camera_quat_w, camera_offset_quat)

    # set the new camera pose
    # Note: the offset will be updated automatically,
    # and the attachment relation is kept.
    sensor.set_world_poses(
        camera_pos_w,
        camera_quat_w,
        env_ids=env_ids,
        convention=sensor.cfg.offset.convention,
    )


def randomize_rigid_body_material(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    static_friction_range: tuple[float, float] = (0.3, 1.6),
    dynamic_friction_range: tuple[float, float] = (0.3, 1.2),
    restitution_range: tuple[float, float] = (0.0, 0.5),
    num_buckets: int = 64,
) -> None:
    """Randomize rigid body material properties through MuJoCo geom friction fields.

    MuJoCo geoms expose slide, spin, and roll friction and do not provide a per-geom
    restitution coefficient. The provided static and dynamic friction ranges are
    merged into the slide-friction interval.
    """
    # Use both friction ranges to form the slide-friction interval.
    slide_friction_range = (
        min(static_friction_range[0], dynamic_friction_range[0]),
        max(static_friction_range[1], dynamic_friction_range[1]),
    )

    # MuJoCo has no per-geom restitution coefficient.
    del num_buckets, restitution_range
    dr.geom_friction(
        env,
        env_ids=env_ids,
        ranges=slide_friction_range,
        operation="abs",
        asset_cfg=asset_cfg,
        shared_random=True,
    )
