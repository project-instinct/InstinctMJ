from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import tqdm
from mjlab.managers import SceneEntityCfg
from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields
from mjlab.utils.lab_api import math as math_utils

from instinct_mj.motion_reference.motion_reference_hoi_data import HoiMotionReferenceData, HoiMotionReferenceState

if TYPE_CHECKING:
    from mjlab.entity import Entity
    from mjlab.entity import Entity as Articulation
    from mjlab.entity import Entity as RigidObject
    from mjlab.envs import ManagerBasedRlEnv as ManagerBasedEnv

    from instinct_mj.envs.mdp import BeyondMimicAdaptiveWeighting
    from instinct_mj.motion_reference import MotionReferenceData
    from instinct_mj.motion_reference.motion_reference_manager import MotionReferenceManager


@requires_model_fields("body_gravcomp", recompute=RecomputeLevel.set_const_fixed)
def virtualize_articulation(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot_reference"),
):
    """Virtualize the robot reference (Articulation) in the simulation by removing all
    their collision shapes and setting their mass to zero.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # resolve environment ids (num_envs)
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, dtype=torch.long, device=env.device)
    else:
        env_ids = env_ids.to(dtype=torch.long, device=env.device)

    # resolve body indices (global ids in MuJoCo model arrays)
    if asset_cfg.body_ids == slice(None):
        body_ids = asset.indexing.body_ids
    else:
        body_ids_local = torch.tensor(asset_cfg.body_ids, dtype=torch.long, device=env.device)
        body_ids = asset.indexing.body_ids[body_ids_local]

    # MuJoCo does not provide the previous backend-style per-body gravity toggle.
    # gravcomp=1.0 compensates full gravity for the selected bodies.
    env_grid, body_grid = torch.meshgrid(env_ids, body_ids, indexing="ij")
    env.sim.model.body_gravcomp[env_grid, body_grid] = 1.0


def match_motion_ref_with_scene(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    motion_ref_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
):
    """Match information between motion reference and the scene, use at 'startup' stage."""
    print("[Event] Match motion reference with scene.")
    motion_ref: MotionReferenceManager = env.scene[motion_ref_cfg.name]
    motion_ref.match_scene(env.scene)


def _write_reference_reset(
    asset,
    env_ids: torch.Tensor,
    base_pos_w: torch.Tensor,
    base_quat_w: torch.Tensor,
    base_lin_vel_w: torch.Tensor,
    base_ang_vel_w: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    randomize_pose_range: dict[str, tuple[float, float]],
    randomize_velocity_range: dict[str, tuple[float, float]],
    randomize_joint_pos_range: tuple[float, float],
    clamp_joint_pos_to_limits: bool,
):
    """Draw the reset noise on the (noise-free) reference state of ``env_ids`` and write the result to the simulation."""
    # apply randomizations on the reset states if specified
    if randomize_pose_range:
        # Apply pose randomization similar to reset_root_state_uniform
        range_list = [randomize_pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=base_pos_w.device)
        rand_samples = math_utils.sample_uniform(
            ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=base_pos_w.device
        )

        # Add position randomization
        base_pos_w = base_pos_w + rand_samples[:, 0:3]

        # Add orientation randomization
        orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        base_quat_w = math_utils.quat_mul(orientations_delta, base_quat_w)

    # write the root pose to the simulation
    asset.write_root_link_pose_to_sim(torch.cat([base_pos_w, base_quat_w], dim=-1), env_ids=env_ids)

    # apply velocity randomization if specified
    if randomize_velocity_range:
        # Apply velocity randomization
        range_list = [randomize_velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=base_lin_vel_w.device)
        vel_samples = math_utils.sample_uniform(
            ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=base_lin_vel_w.device
        )

        # Add velocity randomization
        base_lin_vel_w = base_lin_vel_w + vel_samples[:, 0:3]
        base_ang_vel_w = base_ang_vel_w + vel_samples[:, 3:6]

    # write the root velocity to the simulation
    asset.write_root_link_velocity_to_sim(torch.cat([base_lin_vel_w, base_ang_vel_w], dim=-1), env_ids=env_ids)

    # apply joint position randomization if specified
    if randomize_joint_pos_range != (0.0, 0.0):
        # Apply joint position randomization
        joint_pos_noise = math_utils.sample_uniform(
            randomize_joint_pos_range[0], randomize_joint_pos_range[1], joint_pos.shape, device=joint_pos.device
        )
        joint_pos = joint_pos + joint_pos_noise
    if clamp_joint_pos_to_limits:
        limits = asset.data.joint_pos_limits[env_ids]
        joint_pos = torch.clamp(joint_pos, limits[..., 0], limits[..., 1])

    # write the joint state to the simulation
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


def terrain_penetration_depth(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    terrain_name: str = "terrain",
) -> torch.Tensor:
    """Deepest robot-vs-terrain penetration [m] of every env at its CURRENT (just written) state, shape ``(num_envs,)`` (0 = no penetrating contact).

    Runs the (CUDA-graph captured) ``sim.forward()`` on the existing data (kinematics, collision detection, constraints; no extra GPU memory) and reduces the contacts of the robot
    geoms against the terrain geoms (robot self-contacts and contacts with other entities are ignored) to the per-world minimum of
    ``contact.dist``. ``contact.dist`` is the signed distance of a contact (negative = penetration); with a zero margin only penetrating
    contacts are generated. Contacts beyond the contact buffer (``nacon`` > ``naconmax``) are not seen. Note that the derived
    quantities and the contact buffer of ALL envs are recomputed (from the same states; a ``sim.forward()`` follows every reset anyway).
    """
    import warp as wp

    sim = env.sim
    cache = getattr(env, "_penetration_geom_masks", None)
    if cache is None or cache[0] != (asset_cfg.name, terrain_name):
        ngeom = sim.mj_model.ngeom
        robot = torch.zeros(ngeom, dtype=torch.bool, device=env.device)
        terrain = torch.zeros(ngeom, dtype=torch.bool, device=env.device)
        robot[env.scene[asset_cfg.name].indexing.geom_ids.long()] = True
        terrain[env.scene[terrain_name].indexing.geom_ids.long()] = True
        cache = ((asset_cfg.name, terrain_name), robot, terrain)
        env._penetration_geom_masks = cache
    _, robot, terrain = cache

    sim.forward()  # the captured forward graph of the sim (no new Data, no new graph, no scratch memory: an eager ``mjwarp.collision`` allocates ~4 MB per env of scratch)
    contact = sim.wp_data.contact
    dist, world = wp.to_torch(contact.dist), wp.to_torch(contact.worldid).long()
    geom = wp.to_torch(contact.geom).long()  # (naconmax, 2)
    nacon = wp.to_torch(sim.wp_data.nacon)  # (1,)
    valid = torch.arange(dist.shape[0], device=dist.device) < nacon
    g0, g1 = geom[:, 0].clamp(min=0), geom[:, 1].clamp(min=0)
    pair = (robot[g0] & terrain[g1]) | (terrain[g0] & robot[g1])
    depth = torch.where(valid & pair, -dist, torch.zeros_like(dist)).clamp(min=0.0)
    out = torch.zeros(env.num_envs, dtype=dist.dtype, device=dist.device)
    out.scatter_reduce_(0, world.clamp(0, env.num_envs - 1), depth, reduce="amax")
    return out


def reset_robot_state_by_reference(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    motion_ref_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    position_offset: list = [0.0, 0.0, 0.1],
    dof_vel_ratio: float = 0.5,
    base_lin_vel_ratio: float = 0.5,
    base_ang_vel_ratio: float = 0.5,
    randomize_pose_range: dict[str, tuple[float, float]] = {},
    randomize_velocity_range: dict[str, tuple[float, float]] = {},
    randomize_joint_pos_range: tuple[float, float] = (0.0, 0.0),
    clamp_joint_pos_to_limits: bool = False,
    max_penetration: float | None = None,
    max_resample: int = 8,
    lift_out_of_terrain: bool = False,
    terrain_name: str = "terrain",
):
    """Reset robot state based on motion reference with optional randomization.

    Args:
        env: The environment instance.
        env_ids: Environment IDs to reset.
        motion_ref_cfg: Motion reference configuration.
        asset_cfg: Robot asset configuration.
        position_offset: Position offset to apply [x, y, z].
        dof_vel_ratio: Ratio to scale joint velocities.
        base_lin_vel_ratio: Ratio to scale base linear velocities.
        base_ang_vel_ratio: Ratio to scale base angular velocities.
        randomize_pose_range: Optional pose randomization ranges for ["x", "y", "z", "roll", "pitch", "yaw"].
        randomize_velocity_range: Optional velocity randomization ranges for ["x", "y", "z", "roll", "pitch", "yaw"].
        randomize_joint_pos_range: Optional joint position randomization range (min, max).
        clamp_joint_pos_to_limits: Clamp the randomized joint positions to the joint position limits.
        max_penetration: If set [m], reject-sample the reset noise against the terrain: after the noisy state is written, the deepest
            robot-vs-terrain penetration of each env is measured (``terrain_penetration_depth``); the envs that penetrate by more than
            ``max_penetration`` redraw the noise (same distributions), up to ``max_resample`` times; the envs that still penetrate fall back to
            the noise-free reference state (kept even if that penetrates: the reference is what the data defines). ``None`` (default): no check.
            The counts are accumulated in ``env.reset_penetration_stats`` (a ``(4,)`` tensor: resets, resampled, fell back, reference penetrates).
        max_resample: Number of noise redraws of the penetration rejection.
        lift_out_of_terrain: After the reset state is written (and after the rejection, if any), raise the root of every env by its deepest
            robot-vs-terrain penetration (``terrain_penetration_depth``, one pass, as the ACP reproduction Acp_supermimic_vibed ``env.py``): a foot or
            hand that the reset noise pushed into a box top starts on it instead of inside it. Vertical only: a contact from the side or from
            above (a ladder rung over the hand) is not resolved. Skipped in the reset of the env construction (step counter 0).
        terrain_name: Scene entity whose geoms are the terrain of the penetration check.
    """
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    motion_ref: MotionReferenceManager = env.scene[motion_ref_cfg.name]

    # reset the motion reference object
    # motion reference (as sensor) is already reset(ed) in scene.reset(...)
    # motion_ref.reset(env_ids)

    # reset the robot state based on motion reference data
    data: MotionReferenceData = motion_ref.data  # triggering the motion reference to update the data.
    motion_ref_init_state = motion_ref.get_init_reference_state(env_ids)

    # reset the robot state
    base_pos_w = motion_ref_init_state.base_pos_w + torch.tensor(
        position_offset, device=motion_ref_init_state.base_pos_w.device
    ).unsqueeze(0)
    base_quat_w = motion_ref_init_state.base_quat_w
    base_lin_vel_w = motion_ref_init_state.base_lin_vel_w * base_lin_vel_ratio
    base_ang_vel_w = motion_ref_init_state.base_ang_vel_w * base_ang_vel_ratio
    joint_pos = motion_ref_init_state.joint_pos
    joint_vel = motion_ref_init_state.joint_vel * dof_vel_ratio
    ref = (base_pos_w, base_quat_w, base_lin_vel_w, base_ang_vel_w, joint_pos, joint_vel)
    noise = (randomize_pose_range, randomize_velocity_range, randomize_joint_pos_range, clamp_joint_pos_to_limits)

    _write_reference_reset(asset, env_ids, *ref, *noise)
    if max_penetration is not None:
        _reject_penetrating_noise(env, asset, asset_cfg, terrain_name, env_ids, ref, noise, max_penetration, max_resample, clamp_joint_pos_to_limits)
    if lift_out_of_terrain and getattr(env, "common_step_counter", 0) > 0:
        depth = terrain_penetration_depth(env, asset_cfg, terrain_name)[env_ids]  # (runs sim.forward(): xpos below is the written state)
        pose = asset.data.root_link_pose_w[env_ids].clone()
        pose[:, 2] += depth
        asset.write_root_link_pose_to_sim(pose, env_ids=env_ids)


def _reject_penetrating_noise(env, asset, asset_cfg, terrain_name, env_ids, ref, noise, max_penetration, max_resample, clamp_joint_pos_to_limits) -> None:
    """The ``max_penetration`` rejection of ``reset_robot_state_by_reference``."""
    # penetration rejection: redraw the noise of the envs that start inside the terrain (one host sync per round: the size of the failing subset)
    sel = (terrain_penetration_depth(env, asset_cfg, terrain_name)[env_ids] > max_penetration).nonzero(as_tuple=False).squeeze(-1)
    num_resampled = len(sel)
    for _ in range(max_resample):
        if len(sel) == 0:
            break
        _write_reference_reset(asset, env_ids[sel], *(x[sel] for x in ref), *noise)
        still = terrain_penetration_depth(env, asset_cfg, terrain_name)[env_ids[sel]] > max_penetration
        sel = sel[still.nonzero(as_tuple=False).squeeze(-1)]
    num_fallback, num_ref_pen = len(sel), 0
    if num_fallback > 0:  # the noise-free reference state
        _write_reference_reset(asset, env_ids[sel], *(x[sel] for x in ref), {}, {}, (0.0, 0.0), clamp_joint_pos_to_limits)
        num_ref_pen = int((terrain_penetration_depth(env, asset_cfg, terrain_name)[env_ids[sel]] > max_penetration).sum())
    stats = getattr(env, "reset_penetration_stats", None)
    if stats is None:
        stats = env.reset_penetration_stats = torch.zeros(4, dtype=torch.float64)
    stats += torch.tensor([len(env_ids), num_resampled, num_fallback, num_ref_pen], dtype=torch.float64)


def _apply_rigid_object_states(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    object_pos: torch.Tensor,
    object_quat: torch.Tensor,
    object_lin_vel: torch.Tensor,
    object_ang_vel: torch.Tensor,
    object_validity: torch.Tensor,
    scene_object_names: list[str],
    invalid_object_pos: tuple[float, float, float] | None = None,
):
    """Helper to apply rigid object states to simulation.

    Args:
        env: The environment instance.
        env_ids: Environment IDs to update.
        object_pos: Object positions [N_env, N_obj, 3].
        object_quat: Object orientations [N_env, N_obj, 4].
        object_lin_vel: Object linear velocities [N_env, N_obj, 3].
        object_ang_vel: Object angular velocities [N_env, N_obj, 3].
        object_validity: Object validity mask [N_env, N_obj].
        scene_object_names: List of object names in the scene.
        invalid_object_pos: If not None, set invalid objects to this position (e.g. to hide them).
            If None, invalid objects are not updated.
    """
    # Ensure env_ids is on the same device as data
    if env_ids.device != object_pos.device:
        env_ids = env_ids.to(object_pos.device)

    device = object_pos.device
    identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device, dtype=object_quat.dtype)
    zero_vel = torch.zeros(3, device=device, dtype=object_lin_vel.dtype)

    # Individual objects case
    scene_entities = env.scene.entities
    for obj_idx, entity_name in enumerate(scene_object_names):
        if obj_idx >= object_validity.shape[1]:
            break

        # Use scene entity names for membership check; env.scene[key] raises KeyError for invalid keys
        # (e.g. numeric strings like '0' from object index slots that are not scene entity names)
        if entity_name not in scene_entities:
            continue
        asset: Entity = scene_entities[entity_name]

        valid_mask = object_validity[:, obj_idx]

        # Update valid objects
        if valid_mask.any():
            valid_env_ids = env_ids[valid_mask].contiguous()

            root_pose = torch.cat(
                [object_pos[valid_mask, obj_idx], object_quat[valid_mask, obj_idx]],
                dim=-1,
            )
            root_velocity = torch.cat(
                [object_lin_vel[valid_mask, obj_idx], object_ang_vel[valid_mask, obj_idx]],
                dim=-1,
            )

            if asset.is_fixed_base:
                if not asset.is_mocap:
                    raise ValueError(f"Cannot apply HOI object reference to fixed non-mocap entity '{entity_name}'.")
                asset.write_mocap_pose_to_sim(root_pose, env_ids=valid_env_ids)
            else:
                asset.write_root_link_pose_to_sim(root_pose, env_ids=valid_env_ids)
                asset.write_root_link_velocity_to_sim(root_velocity, env_ids=valid_env_ids)

        # Optionally set invalid objects to a fixed position
        if invalid_object_pos is not None and (~valid_mask).any():
            invalid_env_ids = env_ids[~valid_mask].contiguous()
            n_invalid = invalid_env_ids.shape[0]
            invalid_pos = (
                torch.tensor(invalid_object_pos, device=device, dtype=object_pos.dtype)
                .unsqueeze(0)
                .expand(n_invalid, 3)
            )
            invalid_quat = identity_quat.unsqueeze(0).expand(n_invalid, 4)
            invalid_lin_vel = zero_vel.unsqueeze(0).expand(n_invalid, 3)
            invalid_ang_vel = zero_vel.unsqueeze(0).expand(n_invalid, 3)
            root_pose = torch.cat([invalid_pos, invalid_quat], dim=-1)
            root_velocity = torch.cat([invalid_lin_vel, invalid_ang_vel], dim=-1)
            if asset.is_fixed_base:
                if not asset.is_mocap:
                    raise ValueError(f"Cannot apply HOI object reference to fixed non-mocap entity '{entity_name}'.")
                asset.write_mocap_pose_to_sim(root_pose, env_ids=invalid_env_ids)
            else:
                asset.write_root_link_pose_to_sim(root_pose, env_ids=invalid_env_ids)
                asset.write_root_link_velocity_to_sim(root_velocity, env_ids=invalid_env_ids)


def reset_rigid_objects_state_by_reference(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    motion_ref_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
):
    """Reset rigid object states in scene from HOI init reference state.

    This function intentionally uses object index order and ignores object-name matching.
    """
    motion_ref: MotionReferenceManager = env.scene[motion_ref_cfg.name]
    init_state = motion_ref.get_init_reference_state(env_ids)

    if not isinstance(init_state, HoiMotionReferenceState):
        return

    scene_object_names = getattr(motion_ref.cfg, "scene_object_names", None)
    if not scene_object_names:
        return

    # init_state is already indexed by env_ids (get_init_reference_state returns state[env_ids]),
    # so its tensors have shape [len(env_ids), ...] with row i mapping to env_ids[i].
    # Do NOT index again with env_ids - that treats env IDs (e.g. 2) as positional indices
    # and causes out-of-bounds when a subset of envs reset (e.g. env_ids=[2] -> tensor has 1 row).
    env_ids_t = torch.as_tensor(env_ids, device=init_state.object_pos_w.device)

    _apply_rigid_object_states(
        env,
        env_ids_t,
        init_state.object_pos_w,
        init_state.object_quat_w,
        init_state.object_lin_vel_w,
        init_state.object_ang_vel_w,
        init_state.object_validity,
        scene_object_names,
    )
    motion_ref.sync_reference_entity_state()


def update_rigid_objects_state_by_reference(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    motion_ref_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    invalid_object_pos: tuple[float, float, float] | None = None,
):
    """Set rigid object states from the current HOI reference frame every step.

    This function uses object index order and ignores object-name matching.

    Args:
        invalid_object_pos: If not None, set invalid (non-visible) objects to this position
            (e.g. to hide them far from the scene). If None, invalid objects are not updated.
    """
    motion_ref: MotionReferenceManager = env.scene[motion_ref_cfg.name]
    data: HoiMotionReferenceData = motion_ref.data
    scene_object_names = getattr(motion_ref.cfg, "scene_object_names", None)
    if not scene_object_names:
        return

    if not all(
        hasattr(data, attr)
        for attr in ("object_pos_w", "object_quat_w", "object_lin_vel_w", "object_ang_vel_w", "object_validity")
    ):
        return

    # Always use all env ids to avoid block index / out-of-bounds when interval event passes a subset.
    # Object pose updates must run for all envs to keep motion reference in sync.
    num_envs = env.scene.num_envs
    all_env_ids = torch.arange(num_envs, device=data.object_pos_w.device, dtype=torch.long)

    # data shape is [N, T, O, ...]; use the first target frame.
    _apply_rigid_object_states(
        env,
        all_env_ids,
        data.object_pos_w[:, 0],
        data.object_quat_w[:, 0],
        data.object_lin_vel_w[:, 0],
        data.object_ang_vel_w[:, 0],
        data.object_validity[:, 0],
        scene_object_names,
        invalid_object_pos,
    )
    motion_ref.sync_reference_entity_state()


def beyondmimic_bin_fail_counter_smoothing(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    curriculum_name: str,
):
    # Acquire the curriculum term instance, which should be a ManagerTermBase
    curriculum: BeyondMimicAdaptiveWeighting = env.curriculum_manager.get_term_cfg(curriculum_name).func

    if (
        getattr(curriculum, "enabled", True) is False
        or curriculum.motion_bin_fail_counter is None
        or curriculum.current_motion_bin_fail_counter is None
    ):
        return

    curriculum.motion_bin_fail_counter.concatenated_tensor.mul_(1 - curriculum.adaptive_alpha)
    curriculum.motion_bin_fail_counter.concatenated_tensor.add_(
        curriculum.adaptive_alpha * curriculum.current_motion_bin_fail_counter.concatenated_tensor
    )
    curriculum.current_motion_bin_fail_counter.concatenated_tensor.fill_(0)
