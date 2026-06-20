""" Additinoal common termination functions that are not implemented in mjlab. """

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch
from mjlab.managers import ManagerTermBase, ManagerTermBaseCfg, SceneEntityCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
    from mjlab.entity import Entity
    from mjlab.envs import ManagerBasedRlEnv

    from instinct_mj.motion_reference.motion_reference_manager import MotionReferenceManager


def dataset_exhausted(
    env: ManagerBasedRlEnv,
    reference_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    reset_without_notice: bool = False,
    print_reason: bool = False,
) -> torch.Tensor:
    """Check if the dataset is exhausted.

    Args:
        env: The environment object.
        reset_without_notice: whether to reset the environment without returning True.
    Returns:
        True if the dataset is exhausted, False otherwise.
    """
    motion_reference: MotionReferenceManager = env.scene[reference_cfg.name]
    return_ = torch.logical_not(
        motion_reference.data.validity[motion_reference.ALL_INDICES, motion_reference.aiming_frame_idx]
    )  # shape: [N,]
    if print_reason and return_.any():
        print("dataset_exhausted: ", return_.sum())
    if reset_without_notice:
        motion_reference.reset(env_ids=return_.nonzero(as_tuple=True)[0])
        return_[:] = False
    return return_


def terrain_out_of_bounds(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    distance_buffer: float = 3.0,
    print_reason: bool = False,
) -> torch.Tensor:
    """Terminate when the actor move too close to the edge of the terrain.

    If the actor moves too close to the edge of the terrain, the termination is activated. The distance
    to the edge of the terrain is calculated based on the size of the terrain and the distance buffer.
    """
    # In mjlab, Scene stores config in _cfg.
    terrain_type = env.scene._cfg.terrain.terrain_type
    if terrain_type == "plane":
        return torch.zeros(
            (env.num_envs,), device=env.device, dtype=torch.bool
        )  # we have infinite terrain because it is a plane
    elif terrain_type in ("generator", "hacked_generator"):
        # obtain the size of the sub-terrains
        terrain_gen_cfg = env.scene.terrain.cfg.terrain_generator
        grid_width, grid_length = terrain_gen_cfg.size
        n_rows, n_cols = terrain_gen_cfg.num_rows, terrain_gen_cfg.num_cols
        border_width = terrain_gen_cfg.border_width
        # compute the size of the map
        map_width = n_rows * grid_width + 2 * border_width
        map_height = n_cols * grid_length + 2 * border_width

        # extract the used quantities (to enable type-hinting)
        asset: Entity = env.scene[asset_cfg.name]

        # check if the agent is out of bounds
        x_out_of_bounds = torch.abs(asset.data.root_link_pos_w[:, 0]) > 0.5 * map_width - distance_buffer
        y_out_of_bounds = torch.abs(asset.data.root_link_pos_w[:, 1]) > 0.5 * map_height - distance_buffer
        return_ = torch.logical_or(x_out_of_bounds, y_out_of_bounds)
        if print_reason and return_.any():
            print(f"The base is out of the terrain border:", return_.sum())
        return return_
    else:
        raise ValueError("Received unsupported terrain type, must be one of: 'plane', 'generator', 'hacked_generator'.")


def abnormal_lin_vel(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_value: float = 40.0,  # [m/s]
):
    asset = env.scene[asset_cfg.name]
    return torch.norm(asset.data.root_link_lin_vel_w, dim=-1) > max_value


def abnormal_ang_vel(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_value: float = 40.0,  # [rad/s]
):
    asset = env.scene[asset_cfg.name]
    return torch.norm(asset.data.root_link_ang_vel_w, dim=-1) > max_value


def abnormal_joint_vel(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_value: float = 40.0,  # [rad/s]
):
    asset = env.scene[asset_cfg.name]
    return torch.any(torch.abs(asset.data.joint_vel) > max_value, dim=-1)


def abnormal_joint_acc(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_value: float = 4000.0,  # [rad/s^2]
):
    asset = env.scene[asset_cfg.name]
    return torch.any(torch.abs(asset.data.joint_acc) > max_value, dim=-1)


class illegal_reset_contact(ManagerTermBase):
    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        self.threshold = cfg.params["threshold"]
        self.sensor_name = cfg.params["sensor_name"]
        self.asset_cfg = cfg.params["asset_cfg"]
        self.print_reason = cfg.params.get("print_reason", False)
        self.episode_length_threshold = cfg.params.get("episode_length_threshold", 1)
        self.max_envs_to_print = cfg.params.get("max_envs_to_print", 8)
        self.illegal_contact_counter = torch.zeros(env.num_envs, device=env.device, dtype=torch.int)

    def _selected_body_names(self, env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> list[str]:
        asset = env.scene[asset_cfg.name]
        if isinstance(asset_cfg.body_ids, slice):
            selected_body_ids = list(range(asset.num_bodies))
        else:
            selected_body_ids = list(asset_cfg.body_ids)
        return [asset.body_names[i] for i in selected_body_ids]

    def _log_illegal_contact_details(
        self,
        env: ManagerBasedRlEnv,
        asset_cfg: SceneEntityCfg,
        body_contact_force: torch.Tensor,
        body_contact_mask: torch.Tensor,
        env_ids: torch.Tensor,
    ) -> None:
        selected_body_names = self._selected_body_names(env, asset_cfg)
        body_hit_counts = body_contact_mask[env_ids].sum(dim=0).detach().cpu().tolist()
        body_hit_summary = [
            f"{body_name}:{int(hit_count)}"
            for body_name, hit_count in zip(selected_body_names, body_hit_counts)
            if hit_count > 0
        ]

        print(f"illegal_reset_contact: {env_ids.numel()} envs")
        if body_hit_summary:
            print("  body_hit_counts:", ", ".join(body_hit_summary))

        motion_identifiers = None
        min_link_z = None
        if "motion_reference" in env.scene.sensors:
            motion_reference: MotionReferenceManager = env.scene["motion_reference"]
            motion_identifiers = motion_reference.get_current_motion_identifiers(env_ids=env_ids)
            min_link_z = motion_reference.reference_frame.link_pos_w[env_ids, 0, :, 2].amin(dim=1)

        num_envs_to_print = min(env_ids.numel(), self.max_envs_to_print)
        for print_idx in range(num_envs_to_print):
            env_id = int(env_ids[print_idx].item())
            triggered_body_local_ids = body_contact_mask[env_id].nonzero(as_tuple=False).squeeze(-1).tolist()
            body_force_summary = ", ".join(
                f"{selected_body_names[local_body_id]}:{float(body_contact_force[env_id, local_body_id].item()):.1f}"
                for local_body_id in triggered_body_local_ids
            )
            motion_identifier = motion_identifiers[print_idx] if motion_identifiers is not None else "N/A"
            min_link_z_value = float(min_link_z[print_idx].item()) if min_link_z is not None else float("nan")
            print(
                "  "
                f"env={env_id} "
                f"episode_step={int(env.episode_length_buf[env_id].item())} "
                f"contact_counter={int(self.illegal_contact_counter[env_id].item())} "
                f"motion_id={motion_identifier} "
                f"min_link_z={min_link_z_value:.4f} "
                f"bodies=[{body_force_summary}]"
            )
        if env_ids.numel() > num_envs_to_print:
            print(f"  ... omitted {env_ids.numel() - num_envs_to_print} more envs")

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        threshold: float,
        sensor_name: str,
        asset_cfg: SceneEntityCfg,
        print_reason: bool = False,
        episode_length_threshold: int = 1,
        max_envs_to_print: int = 8,
    ) -> torch.Tensor:
        """Timeout if the robot is reset with some undesired penetration with the environment.
        within the first episode_length_threshold steps.
        """
        contact_sensor: ContactSensor = env.scene.sensors[sensor_name]
        body_contact_force = torch.max(torch.norm(contact_sensor.data.force_history, dim=-1), dim=-1)[0][
            :, asset_cfg.body_ids
        ]
        body_contact_mask = body_contact_force > threshold
        contacts = torch.any(body_contact_mask, dim=1)
        self.illegal_contact_counter += contacts.int()
        return_ = torch.logical_and(
            self.illegal_contact_counter >= episode_length_threshold,
            env.episode_length_buf <= episode_length_threshold,
        )
        if return_.any() and print_reason:
            self.max_envs_to_print = max_envs_to_print
            self._log_illegal_contact_details(
                env=env,
                asset_cfg=asset_cfg,
                body_contact_force=body_contact_force,
                body_contact_mask=body_contact_mask,
                env_ids=return_.nonzero(as_tuple=True)[0],
            )
        return return_

    def reset(self, env_ids: Sequence[int] | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self.illegal_contact_counter[env_ids] = 0
