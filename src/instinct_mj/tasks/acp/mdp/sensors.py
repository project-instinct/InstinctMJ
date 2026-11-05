"""Ray-cast sensor of the ACP tracking task: scan from the pelvis, optionally with the delayed / drifted odometry pose."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import mujoco_warp as mjwarp
import torch
import warp as wp
from mjlab.sensor import RayCastSensor, RayCastSensorCfg
from mujoco_warp import rays
from mjlab.utils.lab_api.math import matrix_from_quat, quat_from_matrix


@dataclass
class PelvisRayCastSensorCfg(RayCastSensorCfg):
    ray_offset: float = 5.0
    """The rays start at world z = frame z + ray_offset and point straight down (deploy ``ray_offset``), so surfaces
    above the pelvis (boxes, stairs) are seen."""

    use_estimated_pose: bool = False
    """Cast the rays from the estimated pelvis pose (odometry delay and drift) instead of the true pose."""

    retry_shift: tuple[float, float] = (0.03, 0.02)
    """World xy shift [m] of the second cast. The mesh ray cast (mujoco_warp) misses rays inside narrow bands (~1 cm) along some
    mesh edges although the ray hits the mesh (the CPU ray cast does hit); every ray that missed is cast again from the shifted origin
    and takes that hit. Only the hit height enters the scan encoding, so the shift does not change it."""

    history_dt: float = 0.02
    """Time between two sense() calls (the control step); the pose history is sampled with it."""

    history_length: int = 6
    """Number of past poses kept for the odometry delay (max delay <= (history_length - 1) * history_dt)."""

    def build(self) -> PelvisRayCastSensor:
        return PelvisRayCastSensor(self)


class PelvisRayCastSensor(RayCastSensor):
    """Ray-cast sensor on the pelvis with ``ray_offset`` and an optional odometry model (ACP Table S2).

    With ``use_estimated_pose`` the pose of the rays is the estimated pelvis pose: the true pose ``odometry_delay_s``
    ago (linear interpolation of the pose history) plus the per-episode world-frame ``odometry_drift_w`` on the
    position. ``estimated_pos_w`` / ``estimated_quat_w`` expose it for the other actor observations (contact targets,
    goal). ``data.frame_pos_w`` is the estimated position, so ``frame z - hit z`` is the scan encoding of the estimate.
    Both ``odometry_*`` buffers are written by the ``randomize_odometry`` event (zero = true pose).
    """

    cfg: PelvisRayCastSensorCfg

    def initialize(
        self,
        mj_model: mujoco.MjModel,
        model: mjwarp.Model,
        data: mjwarp.Data,
        device: str,
    ) -> None:
        super().initialize(mj_model, model, data, device)
        num_envs, length = data.nworld, self.cfg.history_length
        self.odometry_drift_w = torch.zeros(num_envs, 3, device=device)
        self.odometry_delay_s = torch.zeros(num_envs, device=device)
        self.history_pos_w = torch.zeros(num_envs, length, 3, device=device)
        self.history_quat_w = torch.zeros(num_envs, length, 4, device=device)
        self.history_quat_w[..., 0] = 1.0
        self.estimated_pos_w = torch.zeros(num_envs, 3, device=device)
        self.estimated_quat_w = torch.zeros(num_envs, 4, device=device)
        self.estimated_quat_w[:, 0] = 1.0
        self.reset_pending = torch.ones(num_envs, dtype=torch.bool, device=device)
        self.last_clock = torch.full((num_envs,), -1.0, device=device)
        # second cast of the missed rays (``retry_shift``)
        self._ray_pnt2 = wp.zeros((num_envs, self._num_rays), dtype=wp.vec3, device=device)
        self._ray_dist2 = wp.zeros((num_envs, self._num_rays), dtype=float, device=device)
        self._ray_geomid2 = wp.zeros((num_envs, self._num_rays), dtype=int, device=device)
        self._ray_normal2 = wp.zeros((num_envs, self._num_rays), dtype=wp.vec3, device=device)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        super().reset(env_ids)
        self.reset_pending[slice(None) if env_ids is None else env_ids] = True

    def _clock(self) -> torch.Tensor:
        """Simulation time of every env (it advances with every physics step, not with forward / sense calls)."""
        return self._data.time.clone()

    def _update_estimated_pose(self, pos: torch.Tensor, quat: torch.Tensor) -> None:
        # newest pose at index 0; the envs that were reset start with a history of the post-reset pose. ``sense()`` also runs inside
        # ``reset(env_ids)`` for ALL envs (an explicit partial reset after a step): the history advances once per control step, counted by the
        # simulation clock (an env whose clock did not move has not been stepped and pushes no duplicate pose, also when the robot is at rest).
        clock = self._clock()
        advance = ((clock != self.last_clock) | self.reset_pending).view(-1, 1, 1)
        self.last_clock = clock
        rolled_pos, rolled_quat = torch.roll(self.history_pos_w, 1, dims=1), torch.roll(self.history_quat_w, 1, dims=1)
        rolled_pos[:, 0] = pos
        rolled_quat[:, 0] = quat
        self.history_pos_w = torch.where(advance, rolled_pos, self.history_pos_w)
        self.history_quat_w = torch.where(advance, rolled_quat, self.history_quat_w)
        self.history_pos_w[self.reset_pending] = pos[self.reset_pending].unsqueeze(1)
        self.history_quat_w[self.reset_pending] = quat[self.reset_pending].unsqueeze(1)
        self.reset_pending[:] = False
        lag = (self.odometry_delay_s / self.cfg.history_dt).clamp(0.0, self.cfg.history_length - 1.0)
        lower = lag.floor().long()
        upper = (lower + 1).clamp(max=self.cfg.history_length - 1)
        frac = (lag - lower).unsqueeze(-1)
        env = torch.arange(len(lag), device=lag.device)
        est_pos = (1 - frac) * self.history_pos_w[env, lower] + frac * self.history_pos_w[env, upper]
        q0, q1 = self.history_quat_w[env, lower], self.history_quat_w[env, upper]
        q1 = torch.where((q0 * q1).sum(-1, keepdim=True) < 0, -q1, q1)
        est_quat = (1 - frac) * q0 + frac * q1
        self.estimated_pos_w = est_pos + self.odometry_drift_w
        self.estimated_quat_w = est_quat / est_quat.norm(dim=-1, keepdim=True)

    def prepare_rays(self) -> None:
        super().prepare_rays()
        true_pos = self._cached_frame_pos[:, 0]
        true_quat = quat_from_matrix(self._cached_frame_mat[:, 0])
        self._update_estimated_pose(true_pos, true_quat)
        B, N = true_pos.shape[0], self._num_rays_per_frame
        if self.cfg.use_estimated_pose:
            frame_pos, frame_mat = self.estimated_pos_w.unsqueeze(1), matrix_from_quat(self.estimated_quat_w).unsqueeze(1)
            rot = self._compute_alignment_rotation(frame_mat.reshape(B, 3, 3))
            origins = frame_pos + torch.einsum("bij,nj->bni", rot, self._local_offsets)
            rays = torch.einsum("bij,nj->bni", rot, self._local_directions)
            self._cached_frame_pos, self._cached_frame_mat, self._cached_world_rays = frame_pos, frame_mat, rays
        else:
            origins = self._cached_world_origins
        origins = origins.clone()
        origins[..., 2] += self.cfg.ray_offset
        self._cached_world_origins = origins
        wp.to_torch(self._ray_pnt).view(B, self._num_rays, 3).copy_(origins)
        wp.to_torch(self._ray_vec).view(B, self._num_rays, 3).copy_(self._cached_world_rays)

        pnt2 = wp.to_torch(self._ray_pnt2).view(B, self._num_rays, 3)
        pnt2.copy_(origins)
        pnt2[..., 0] += self.cfg.retry_shift[0]
        pnt2[..., 1] += self.cfg.retry_shift[1]

    def raycast_kernel(self, rc: mjwarp.RenderContext) -> None:
        super().raycast_kernel(rc)
        rays(
            m=self._model.struct,  # type: ignore[attr-defined]
            d=self._data.struct,  # type: ignore[attr-defined]
            pnt=self._ray_pnt2,
            vec=self._ray_vec,
            geomgroup=self._geomgroup,
            flg_static=True,
            bodyexclude=self._ray_bodyexclude,
            dist=self._ray_dist2,
            geomid=self._ray_geomid2,
            normal=self._ray_normal2,
            rc=rc,
        )

    def postprocess_rays(self) -> None:
        dist, dist2 = wp.to_torch(self._ray_dist), wp.to_torch(self._ray_dist2)
        retry = (dist < 0) & (dist2 >= 0)
        dist[retry] = dist2[retry]
        B = dist.shape[0]
        normal, normal2 = wp.to_torch(self._ray_normal).view(B, -1, 3), wp.to_torch(self._ray_normal2).view(B, -1, 3)
        normal[retry] = normal2[retry]
        super().postprocess_rays()
