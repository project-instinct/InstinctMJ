from __future__ import annotations

import re
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import mujoco
import mujoco_warp as mjwarp
import torch
from mjlab.entity import Entity
from mjlab.entity.data import compute_velocity_from_cvel
from mjlab.sensor import Sensor
from mjlab.utils.lab_api import math as math_utils
from mjlab.utils.lab_api import string as string_utils

from .volume_points_data import VolumePointsData

if TYPE_CHECKING:
    from .volume_points_cfg import VolumePointsCfg


class VolumePoints(Sensor[VolumePointsData]):
    """Volume Points sensor for detecting volume points in a simulation."""

    def __init__(self, cfg: VolumePointsCfg):
        super().__init__()
        self.cfg = cfg

        # Initialize the volume points
        self._volume_points_pattern: torch.Tensor | None = None
        self._sensor_data: VolumePointsData | None = None
        self._virtual_obstacles: dict[str, Any] = {}

        self._device: str | None = None
        self._data: mjwarp.Data | None = None
        self._model: mjwarp.Model | None = None
        self._mj_model: mujoco.MjModel | None = None
        self._num_envs = 0
        self._ALL_INDICES = torch.empty(0, dtype=torch.long)

        self._body_ids: torch.Tensor | None = None
        self._body_names: list[str] = []
        self._num_bodies = 0

    """
    Properties
    """

    @property
    def data(self) -> VolumePointsData:
        return super().data

    @property
    def num_bodies(self) -> int:
        """Number of bodies with volume points sensors attached."""
        return self._num_bodies

    @property
    def body_names(self) -> list[str]:
        """Ordered names of bodies with volume points sensors attached."""
        return list(self._body_names)

    """
    Operations
    """

    def register_virtual_obstacles(self, virtual_obstacles: dict[str, Any]) -> None:
        """Record the edges buffer to the sensor so that the penetration data can be updated.

        NOTE: typically this is called by a startup event.

        """
        self._virtual_obstacles.update(virtual_obstacles)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None):
        # reset the timers and counters
        super().reset(env_ids)
        # NOTE: Original InstinctLab reset() has no additional logic (pass/...).
        ...

    def find_bodies(self, name_keys: str | Sequence[str], preserve_order: bool = False) -> tuple[list[int], list[str]]:
        """Find bodies in the articulation based on the name keys.

        Args:
            name_keys: A regular expression or a list of regular expressions to match the body names.
            preserve_order: Whether to preserve the order of the name keys in the output. Defaults to False.

        Returns:
            A tuple of lists containing the body indices and names.
        """
        return string_utils.resolve_matching_names(name_keys, self.body_names, preserve_order)

    """
    Implementation
    """

    def edit_spec(self, scene_spec: mujoco.MjSpec, entities: dict[str, Entity]) -> None:
        del scene_spec, entities

    def initialize(
        self,
        mj_model: mujoco.MjModel,
        model: mjwarp.Model,
        data: mjwarp.Data,
        device: str,
    ) -> None:
        self._mj_model = mj_model
        self._model = model
        self._data = data
        self._device = device
        self._num_envs = data.nworld
        self._ALL_INDICES = torch.arange(self._num_envs, device=device, dtype=torch.long)

        body_ids, body_names = self._resolve_body_ids(mj_model)
        if not body_ids:
            raise RuntimeError(
                "Failed to initialize volume points sensor for specified bodies."
                f"\n\tEntity name        : {self.cfg.entity_name}"
                f"\n\tBody patterns      : {self.cfg.body_names}"
            )
        self._body_ids = torch.tensor(body_ids, device=device, dtype=torch.long)
        self._body_names = body_names
        self._num_bodies = len(body_ids)

        # initialize the volume points data
        self._volume_points_pattern = self.cfg.points_generator.func(self.cfg.points_generator).to(device)  # (P, 3)
        self._sensor_data = VolumePointsData.make_zero(
            num_envs=self._num_envs,
            num_bodies=self._num_bodies,
            point_num_each_body=self._volume_points_pattern.shape[0],
            device=device,
        )

    def _compute_data(self) -> VolumePointsData:
        self._refresh_volume_points(self._ALL_INDICES)
        self._refresh_penetration_offset(self._ALL_INDICES)
        return self._sensor_data

    def _refresh_volume_points(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        """Refresh the volume points data. If env_ids is None, refresh all environments."""

        env_ids = self._resolve_env_ids(env_ids)
        body_poses_pos = self._data.xpos[env_ids][:, self._body_ids]  # (N_, B, 3)
        body_poses_quat = self._data.xquat[env_ids][:, self._body_ids]  # (N_, B, 4), wxyz
        body_cvel = self._data.cvel[env_ids][:, self._body_ids]  # (N_, B, 6)
        subtree_com = self._data.subtree_com[env_ids][:, self._body_ids]  # (N_, B, 3)

        body_vels = compute_velocity_from_cvel(body_poses_pos, subtree_com, body_cvel)  # (N_, B, 6)

        self._sensor_data.pos_w[env_ids] = body_poses_pos  # (N_, B, 3)
        self._sensor_data.quat_w[env_ids] = body_poses_quat  # (N_, B, 4), wxyz
        self._sensor_data.vel_w[env_ids] = body_vels[..., :3]  # (N_, B, 3)
        self._sensor_data.ang_vel_w[env_ids] = body_vels[..., 3:]  # (N_, B, 3)

        # calculate the volume points positions and velocities in world frame
        N_B = self._sensor_data.pos_w[env_ids].shape[0] * self._sensor_data.pos_w[env_ids].shape[1]  # (N_*B)
        points_pos_w = math_utils.transform_points(
            self._volume_points_pattern.unsqueeze(0).expand(N_B, -1, -1),  # (N_*B, P, 3)
            self._sensor_data.pos_w[env_ids].flatten(0, 1),  # (N_*B, 3)
            self._sensor_data.quat_w[env_ids].flatten(0, 1),  # (N_*B, 4)
        ).reshape(
            *self._sensor_data.pos_w[env_ids].shape[:2], self._sensor_data.point_num_each_body, 3
        )  # (N_, B, P, 3)
        self._sensor_data.points_pos_w[env_ids] = points_pos_w
        points_vel_w = self._sensor_data.vel_w[env_ids].unsqueeze(-2).expand_as(points_pos_w).clone()  # (N_, B, P, 3)
        points_vel_w += torch.linalg.cross(
            self._sensor_data.ang_vel_w[env_ids].unsqueeze(-2),
            (points_pos_w - self._sensor_data.pos_w[env_ids].unsqueeze(-2)),
            dim=-1,
        )  # (N_, B, P, 3)
        self._sensor_data.points_vel_w[env_ids] = points_vel_w

    def _refresh_penetration_offset(self, env_ids: Sequence[int] | torch.Tensor) -> None:
        """Refresh the penetration depth data. If env_ids is None, refresh all environments."""
        env_ids = self._resolve_env_ids(env_ids)

        penetration_offset_buf: torch.Tensor = self._sensor_data.penetration_offset[env_ids]
        penetration_offset_buf[:] = 0.0
        penetration_depth_buf = torch.zeros_like(penetration_offset_buf[..., 0])  # (N_, B, P)

        for virtual_obstacle in self._virtual_obstacles.values():
            # get the penetration offset for the virtual obstacle
            penetration_offset = virtual_obstacle.get_points_penetration_offset(
                self._sensor_data.points_pos_w[env_ids].flatten(0, 2)
            )  # (N_*B*P, 3)
            penetration_offset = penetration_offset.reshape(
                self._sensor_data.points_pos_w[env_ids].shape
            )  # (N_, B, P, 3)
            penetration_depth = torch.norm(penetration_offset, dim=-1)  # (N_, B, P)
            # update the penetration offset if the depth is greater than the current depth
            mask = penetration_depth > penetration_depth_buf  # (N_, B, P)
            penetration_depth_buf[mask] = penetration_depth[mask]
            penetration_offset_buf[mask] = penetration_offset[mask]

        self._sensor_data.penetration_offset[env_ids] = penetration_offset_buf

    def debug_vis(self, visualizer) -> None:
        if not self.cfg.debug_vis or self._sensor_data is None:
            return
        env_ids = list(visualizer.get_env_indices(self._num_envs))
        if not env_ids:
            return

        marker_cfgs = self.cfg.visualizer_cfg.markers
        sphere_cfg = marker_cfgs["sphere"]
        sphere_penetrated_cfg = marker_cfgs["sphere_penetrated"]
        normal_radius = float(sphere_cfg.radius)
        normal_color = tuple(sphere_cfg.color)
        penetrated_radius = float(sphere_penetrated_cfg.radius)
        penetrated_color = tuple(sphere_penetrated_cfg.color)

        points = self._sensor_data.points_pos_w[env_ids].reshape(-1, 3)
        penetrated = torch.norm(self._sensor_data.penetration_offset[env_ids].reshape(-1, 3), dim=-1) > 0.0

        # Keep parity with InstinctLab semantics:
        # if no penetrated points exist, add one red point placeholder.
        if not torch.any(penetrated):
            points = torch.cat([points, torch.zeros_like(points[:1])], dim=0)
            penetrated = torch.cat([penetrated, torch.tensor([True], device=points.device)], dim=0)

        normal_points = points[~penetrated]
        penetrated_points = points[penetrated]
        normal_points_np = normal_points.cpu().numpy()
        penetrated_points_np = penetrated_points.cpu().numpy()

        for point in normal_points_np:
            visualizer.add_sphere(center=point, radius=normal_radius, color=normal_color)
        for point in penetrated_points_np:
            visualizer.add_sphere(center=point, radius=penetrated_radius, color=penetrated_color)

    # -- private helpers --

    def _resolve_env_ids(self, env_ids: Sequence[int] | torch.Tensor | None) -> torch.Tensor:
        if env_ids is None:
            return self._ALL_INDICES
        return torch.as_tensor(env_ids, device=self._device, dtype=torch.long)

    def _resolve_body_ids(self, mj_model: mujoco.MjModel) -> tuple[list[int], list[str]]:
        entity_name, body_patterns = self._resolve_entity_and_patterns_from_cfg()
        excluded_patterns = self._resolve_excluded_patterns()

        body_ids: list[int] = []
        body_names: list[str] = []

        for body_id in range(1, mj_model.nbody):
            full_name = mj_model.body(body_id).name
            if not full_name:
                continue

            # Resolve entity prefix.
            if "/" in full_name:
                full_entity_name, local_body_name = full_name.split("/", 1)
            else:
                full_entity_name, local_body_name = "", full_name

            if entity_name and full_entity_name.lower() != entity_name.lower():
                continue

            if not any(re.fullmatch(pattern, local_body_name) for pattern in body_patterns):
                continue
            if any(re.fullmatch(pattern, local_body_name) for pattern in excluded_patterns):
                continue

            body_ids.append(body_id)
            body_names.append(local_body_name)

        return body_ids, body_names

    def _resolve_entity_and_patterns_from_cfg(self) -> tuple[str, list[str]]:
        entity_name = self.cfg.entity_name.strip()
        if isinstance(self.cfg.body_names, str):
            body_patterns = [self.cfg.body_names]
        else:
            body_patterns = list(self.cfg.body_names)

        return entity_name, body_patterns

    def _resolve_excluded_patterns(self) -> list[str]:
        patterns: list[str] = []
        for expr in self.cfg.filter_prim_paths_expr:
            tokens = [token for token in expr.split("/") if token]
            if not tokens:
                continue
            patterns.append(tokens[-1])
        return patterns
