from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch
import warp as wp
from mjlab.sensor import ObjRef, RayCastData, RayCastSensor, SensorContext
from mujoco_warp import rays

if TYPE_CHECKING:
    from .grouped_ray_caster_cfg import GroupedRayCasterCfg


@dataclass
class _AttachmentFrameMetadata:
    frame_type: Literal["body", "site", "geom"]
    obj_id: int
    body_id: int
    frame_local_pos: torch.Tensor


class GroupedRayCaster(RayCastSensor):
    """Ray-caster with per-environment mutable ray offsets/directions."""

    cfg: GroupedRayCasterCfg
    """The configuration parameters."""

    def __init__(self, cfg: GroupedRayCasterCfg):
        super().__init__(cfg)
        self._num_envs = 0
        self._ALL_INDICES = torch.empty(0, dtype=torch.long)
        self.drift: torch.Tensor | None = None
        self.ray_starts: torch.Tensor | None = None
        self.ray_directions: torch.Tensor | None = None
        self._min_distance = 0.0
        self._mesh_filter_epsilon = 0.0
        self._mesh_filter_max_hops = 1
        self._needs_filter_continue = False
        self._mesh_filter_enabled: bool = False
        self._allowed_geom_lut: torch.Tensor | None = None
        self._active_hop_capacity = 0
        self._active_hop_pnt = torch.empty(0, 0, 3)
        self._active_hop_vec = torch.empty(0, 0, 3)
        self._active_hop_dist = torch.empty(0, 0)
        self._active_hop_geomid = torch.empty(0, 0, dtype=torch.int32)
        self._active_hop_normal = torch.empty(0, 0, 3)
        self._active_hop_bodyexclude = torch.empty(0, dtype=torch.int32)
        self._ray_bodyexclude_value = -1
        self._attachment_frame_metadata: _AttachmentFrameMetadata | None = None
        self._runtime_mj_model = None
        self._runtime_model = None
        self._runtime_data = None
        self._runtime_ctx: SensorContext | None = None
        self._runtime_device: str | None = None
        self._runtime_wp_device = None

    def initialize(self, mj_model, model, data, device: str) -> None:
        super().initialize(mj_model, model, data, device)
        self._runtime_mj_model = mj_model
        self._runtime_model = model
        self._runtime_data = data
        self._runtime_device = device
        self._runtime_wp_device = wp.get_device(device)
        self._attachment_frame_metadata = self._resolve_attachment_frame_metadata(mj_model, device)

        self._min_distance = float(self.cfg.min_distance)
        if self._min_distance < 0.0:
            raise ValueError(f"min_distance must be >= 0.0, got {self.cfg.min_distance}.")

        self._mesh_filter_epsilon = float(self.cfg.mesh_filter_epsilon)
        if self._mesh_filter_epsilon <= 0.0:
            raise ValueError(f"mesh_filter_epsilon must be > 0.0, got {self.cfg.mesh_filter_epsilon}.")

        self._mesh_filter_max_hops = int(self.cfg.mesh_filter_max_hops)
        if self._mesh_filter_max_hops < 1:
            raise ValueError(f"mesh_filter_max_hops must be >= 1, got {self.cfg.mesh_filter_max_hops}.")

        self._num_envs = data.nworld
        self._ALL_INDICES = torch.arange(self._num_envs, device=device, dtype=torch.long)
        self.drift = torch.zeros(self._num_envs, 3, device=device, dtype=torch.float32)

        self.ray_starts = self._local_offsets.unsqueeze(0).repeat(self._num_envs, 1, 1).clone()
        self.ray_directions = self._local_directions.unsqueeze(0).repeat(self._num_envs, 1, 1).clone()
        ray_bodyexclude_torch = wp.to_torch(self._ray_bodyexclude)
        if ray_bodyexclude_torch.numel() > 0:
            self._ray_bodyexclude_value = int(ray_bodyexclude_torch[0].item())
        else:
            self._ray_bodyexclude_value = -1
        self._initialize_mesh_path_filter(mj_model, device)
        self._needs_filter_continue = self._mesh_filter_enabled or self._min_distance > 0.0

    @property
    def raycast_data(self) -> RayCastData:
        return RayCastSensor._compute_data(self)

    def set_context(self, ctx: SensorContext) -> None:
        super().set_context(ctx)
        self._runtime_ctx = ctx

    def prepare_rays(self) -> None:
        """PRE-GRAPH: Transform per-env local rays to world frame."""
        frame_pos, frame_mat = self._compute_attached_frame_world_pose()

        # note: we clone here because we are read-only operations
        frame_pos = frame_pos.clone()
        frame_mat = frame_mat.clone()

        rot_mat = self._compute_alignment_rotation(frame_mat)
        world_offsets = torch.einsum("bij,bnj->bni", rot_mat, self.ray_starts)
        world_origins = frame_pos.unsqueeze(1) + world_offsets
        ray_directions_w = torch.einsum("bij,bnj->bni", rot_mat, self.ray_directions)

        if self.drift is not None:
            # apply drift
            world_origins = world_origins + self.drift.unsqueeze(1)
            frame_pos = frame_pos + self.drift

        self._write_world_rays_to_backend(world_origins, ray_directions_w)

        self._cached_world_origins = world_origins
        self._cached_world_rays = ray_directions_w
        self._cached_frame_pos = frame_pos
        self._cached_frame_mat = frame_mat

    def _resolve_attachment_frame_metadata(
        self,
        mj_model,
        device: str,
    ) -> _AttachmentFrameMetadata:
        frames = self.cfg.frame
        if isinstance(frames, ObjRef):
            frames = (frames,)
        if len(frames) != 1:
            raise ValueError(
                "GroupedRayCaster currently supports exactly one attachment frame. "
                f"Received {len(frames)} frames for sensor '{self.cfg.name}'."
            )
        frame = frames[0]
        frame_name = frame.prefixed_name()
        frame_type: Literal["body", "site", "geom"] = frame.type
        if frame_type == "body":
            obj_id = mj_model.body(frame_name).id
            body_id = obj_id
            frame_local_pos = torch.zeros(3, device=device, dtype=torch.float32)
        elif frame_type == "site":
            obj_id = mj_model.site(frame_name).id
            body_id = int(mj_model.site_bodyid[obj_id])
            frame_local_pos = torch.tensor(mj_model.site_pos[obj_id], device=device, dtype=torch.float32)
        elif frame_type == "geom":
            obj_id = mj_model.geom(frame_name).id
            body_id = int(mj_model.geom_bodyid[obj_id])
            frame_local_pos = torch.tensor(mj_model.geom_pos[obj_id], device=device, dtype=torch.float32)
        else:
            raise ValueError(
                f"GroupedRayCaster frame must be 'body', 'site', or 'geom', got '{frame.type}' "
                f"for sensor '{self.cfg.name}'."
            )
        return _AttachmentFrameMetadata(
            frame_type=frame_type,
            obj_id=obj_id,
            body_id=body_id,
            frame_local_pos=frame_local_pos,
        )

    def _require_attachment_frame_metadata(self) -> _AttachmentFrameMetadata:
        assert self._attachment_frame_metadata is not None
        return self._attachment_frame_metadata

    def _compute_attached_frame_world_pose(
        self,
        env_ids: Sequence[int] | torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if env_ids is None:
            env_ids = self._ALL_INDICES
        device = self._runtime_device
        runtime_data = self._runtime_data
        metadata = self._require_attachment_frame_metadata()
        assert device is not None and runtime_data is not None
        env_ids = torch.as_tensor(env_ids, device=device, dtype=torch.long)
        frame_type = metadata.frame_type
        obj_id = metadata.obj_id
        body_id = metadata.body_id
        if frame_type == "body":
            frame_pos = runtime_data.xpos[env_ids, body_id]
            frame_mat = runtime_data.xmat[env_ids, body_id].view(-1, 3, 3)
        else:
            body_pos = runtime_data.xpos[env_ids, body_id]
            body_mat = runtime_data.xmat[env_ids, body_id].view(-1, 3, 3)
            if frame_type == "site":
                frame_mat = runtime_data.site_xmat[env_ids, obj_id].view(-1, 3, 3)
            else:
                frame_mat = runtime_data.geom_xmat[env_ids, obj_id].view(-1, 3, 3)
            # Keep InstinctLab semantics: the attached frame origin comes from the
            # parent body's full pose plus the local site/geom offset. ray_alignment
            # only affects how ray starts/directions are rotated below.
            frame_local_pos = metadata.frame_local_pos.to(dtype=body_mat.dtype)
            frame_pos = body_pos + torch.einsum("bij,j->bi", body_mat, frame_local_pos)
        return frame_pos, frame_mat

    def postprocess_rays(self) -> None:
        super().postprocess_rays()
        if not self._needs_filter_continue:
            return
        self._apply_hit_filter_and_continue()

    def _initialize_mesh_path_filter(self, mj_model, device: str) -> None:
        exprs = [expr for expr in self.cfg.mesh_prim_paths if expr]
        if len(exprs) == 0:
            self._mesh_filter_enabled = False
            self._allowed_geom_lut = None
            return

        compiled_patterns: list[re.Pattern[str]] = []
        for expr in exprs:
            normalized_expr = expr.replace("{ENV_REGEX_NS}", "/World/envs/env_\\d+")
            compiled_patterns.append(re.compile(normalized_expr))

        body_aliases = self._build_body_aliases()
        allowed_geom_ids: list[int] = []
        for geom_id in range(int(mj_model.ngeom)):
            candidates = self._build_geom_path_candidates(mj_model, geom_id, body_aliases)
            if any(pattern.search(path) is not None for pattern in compiled_patterns for path in candidates):
                allowed_geom_ids.append(geom_id)

        if len(allowed_geom_ids) == 0:
            raise RuntimeError(
                f"GroupedRayCaster mesh filter matched no MuJoCo geoms.\n\tmesh_prim_paths: {self.cfg.mesh_prim_paths}"
            )

        allowed_geom_lut = torch.zeros(int(mj_model.ngeom), device=device, dtype=torch.bool)
        allowed_geom_lut[torch.tensor(sorted(set(allowed_geom_ids)), device=device, dtype=torch.long)] = True
        self._allowed_geom_lut = allowed_geom_lut
        self._mesh_filter_enabled = True

    def _apply_hit_filter_and_continue(self) -> None:
        geom_ids = self._read_backend_hit_geom_ids().to(dtype=torch.long)
        distances, hit_pos_w, normals_w = self._get_mutable_raycast_outputs()
        hit_mask = distances >= 0.0
        if not torch.any(hit_mask):
            return

        if self._mesh_filter_enabled:
            allowed_mask = torch.zeros_like(hit_mask)
            allowed_mask[hit_mask] = self._allowed_geom_lut[geom_ids[hit_mask]]
        else:
            allowed_mask = torch.ones_like(hit_mask)

        reject_mask = hit_mask & ((~allowed_mask) | (distances <= self._min_distance))
        if not torch.any(reject_mask):
            return

        final_distances = distances.clone()
        final_hit_pos = hit_pos_w.clone()
        final_normals = normals_w.clone()

        # Rejected hits are initialized to miss.
        final_distances[reject_mask] = -1.0
        final_hit_pos[reject_mask] = self._cached_world_origins[reject_mask]
        final_normals[reject_mask] = 0.0

        world_rays = self._cached_world_rays
        eps = self._mesh_filter_epsilon
        max_hops = self._mesh_filter_max_hops

        current_origins = hit_pos_w.clone()
        current_origins[reject_mask] = hit_pos_w[reject_mask] + world_rays[reject_mask] * eps

        traveled = torch.zeros_like(distances)
        traveled[reject_mask] = distances[reject_mask] + eps
        remaining = self.cfg.max_distance - traveled
        active = reject_mask & (remaining > 0.0)
        if not torch.any(active):
            self._update_mutable_raycast_outputs(final_distances, final_hit_pos, final_normals)
            return

        for _ in range(max_hops):
            if not torch.any(active):
                break

            (
                active_env_ids,
                active_ray_ids,
                active_distances,
                active_geom_ids,
                active_normals,
            ) = self._raycast_active_rays(active=active, current_origins=current_origins, world_rays=world_rays)

            active_remaining = remaining[active_env_ids, active_ray_ids]
            active_hit = (active_distances >= 0.0) & (active_distances <= active_remaining)
            if not torch.any(active_hit):
                break

            if self._mesh_filter_enabled:
                active_allowed = torch.zeros_like(active_hit)
                active_allowed[active_hit] = self._allowed_geom_lut[active_geom_ids[active_hit]]
            else:
                active_allowed = active_hit

            active_origins = current_origins[active_env_ids, active_ray_ids]
            active_dirs = world_rays[active_env_ids, active_ray_ids]
            active_hit_pos = active_origins + active_dirs * active_distances.unsqueeze(-1)
            active_total_distances = traveled[active_env_ids, active_ray_ids] + active_distances

            active_accept = active_hit & active_allowed & (active_total_distances > self._min_distance)
            if torch.any(active_accept):
                accept_env_ids = active_env_ids[active_accept]
                accept_ray_ids = active_ray_ids[active_accept]
                final_distances[accept_env_ids, accept_ray_ids] = active_total_distances[active_accept]
                final_hit_pos[accept_env_ids, accept_ray_ids] = active_hit_pos[active_accept]
                final_normals[accept_env_ids, accept_ray_ids] = active_normals[active_accept]

            active_continue = active_hit & (~active_accept)
            if not torch.any(active_continue):
                break

            continue_env_ids = active_env_ids[active_continue]
            continue_ray_ids = active_ray_ids[active_continue]
            continue_hit_pos = active_hit_pos[active_continue]
            continue_dirs = active_dirs[active_continue]
            continue_distances = active_distances[active_continue]
            current_origins[continue_env_ids, continue_ray_ids] = continue_hit_pos + continue_dirs * eps
            traveled[continue_env_ids, continue_ray_ids] = (
                traveled[continue_env_ids, continue_ray_ids] + continue_distances + eps
            )
            remaining[continue_env_ids, continue_ray_ids] = (
                self.cfg.max_distance - traveled[continue_env_ids, continue_ray_ids]
            )
            active.zero_()
            continue_valid = remaining[continue_env_ids, continue_ray_ids] > 0.0
            if torch.any(continue_valid):
                active[continue_env_ids[continue_valid], continue_ray_ids[continue_valid]] = True

        self._update_mutable_raycast_outputs(final_distances, final_hit_pos, final_normals)

    def _ensure_active_hop_buffers(self, capacity: int) -> None:
        if capacity <= self._active_hop_capacity:
            return
        distances, _, _ = self._get_mutable_raycast_outputs()
        device = distances.device
        self._active_hop_pnt = torch.empty((self._num_envs, capacity, 3), device=device, dtype=torch.float32)
        self._active_hop_vec = torch.empty((self._num_envs, capacity, 3), device=device, dtype=torch.float32)
        self._active_hop_dist = torch.empty((self._num_envs, capacity), device=device, dtype=torch.float32)
        self._active_hop_geomid = torch.empty((self._num_envs, capacity), device=device, dtype=torch.int32)
        self._active_hop_normal = torch.empty((self._num_envs, capacity, 3), device=device, dtype=torch.float32)
        self._active_hop_bodyexclude = torch.full(
            (capacity,),
            self._ray_bodyexclude_value,
            device=device,
            dtype=torch.int32,
        )
        # Initialize once so non-active slots stay numerically safe.
        self._active_hop_pnt.zero_()
        self._active_hop_vec.zero_()
        self._active_hop_vec[..., 2] = 1.0
        self._active_hop_dist.fill_(-1.0)
        self._active_hop_geomid.fill_(-1)
        self._active_hop_normal.zero_()
        self._active_hop_capacity = capacity

    def _raycast_active_rays(
        self,
        *,
        active: torch.Tensor,
        current_origins: torch.Tensor,
        world_rays: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        active_pairs = torch.nonzero(active, as_tuple=False)
        active_env_ids = active_pairs[:, 0].to(dtype=torch.long)
        active_ray_ids = active_pairs[:, 1].to(dtype=torch.long)

        # Compute compact per-env slot ids (0..num_active_rays_in_env-1) using only active rows.
        num_active_total = active_env_ids.numel()
        flat_ids = torch.arange(num_active_total, device=active.device, dtype=torch.long)
        is_env_head = torch.ones(num_active_total, device=active.device, dtype=torch.bool)
        is_env_head[1:] = active_env_ids[1:] != active_env_ids[:-1]
        env_group_ids = torch.cumsum(is_env_head.to(dtype=torch.int64), dim=0) - 1
        env_group_heads = flat_ids[is_env_head]
        active_slot_ids = flat_ids - env_group_heads[env_group_ids]

        max_active = int(active_slot_ids.max().item()) + 1
        self._ensure_active_hop_buffers(max_active)

        active_hop_pnt = self._active_hop_pnt[:, :max_active, :]
        active_hop_vec = self._active_hop_vec[:, :max_active, :]
        active_hop_dist = self._active_hop_dist[:, :max_active]
        active_hop_geomid = self._active_hop_geomid[:, :max_active]
        active_hop_normal = self._active_hop_normal[:, :max_active, :]
        active_hop_bodyexclude = self._active_hop_bodyexclude[:max_active]

        # Only touch active slots; inactive slots are ignored after scatter-back.
        active_hop_pnt[active_env_ids, active_slot_ids] = current_origins[active_env_ids, active_ray_ids]
        active_hop_vec[active_env_ids, active_slot_ids] = world_rays[active_env_ids, active_ray_ids]
        active_hop_dist[active_env_ids, active_slot_ids] = -1.0
        active_hop_geomid[active_env_ids, active_slot_ids] = -1
        active_hop_normal[active_env_ids, active_slot_ids] = 0.0

        runtime_model = self._runtime_model
        runtime_data = self._runtime_data
        render_context = self._get_backend_render_context()
        assert runtime_model is not None and runtime_data is not None
        rays(
            m=runtime_model.struct,  # type: ignore[attr-defined]
            d=runtime_data.struct,  # type: ignore[attr-defined]
            pnt=wp.from_torch(active_hop_pnt.contiguous(), dtype=wp.vec3),
            vec=wp.from_torch(active_hop_vec.contiguous(), dtype=wp.vec3),
            geomgroup=self._geomgroup,  # pyright: ignore[reportArgumentType]
            flg_static=True,
            bodyexclude=wp.from_torch(active_hop_bodyexclude.contiguous(), dtype=wp.int32),
            dist=wp.from_torch(active_hop_dist.contiguous(), dtype=wp.float32),
            geomid=wp.from_torch(active_hop_geomid.contiguous(), dtype=wp.int32),
            normal=wp.from_torch(active_hop_normal.contiguous(), dtype=wp.vec3),
            rc=render_context,
        )

        return (
            active_env_ids,
            active_ray_ids,
            active_hop_dist[active_env_ids, active_slot_ids],
            active_hop_geomid[active_env_ids, active_slot_ids].to(dtype=torch.long),
            active_hop_normal[active_env_ids, active_slot_ids],
        )

    def _write_world_rays_to_backend(self, world_origins: torch.Tensor, world_rays: torch.Tensor) -> None:
        assert self._ray_pnt is not None and self._ray_vec is not None
        pnt_torch = wp.to_torch(self._ray_pnt).view(self._num_envs, self._num_rays, 3)
        vec_torch = wp.to_torch(self._ray_vec).view(self._num_envs, self._num_rays, 3)
        pnt_torch.copy_(world_origins)
        vec_torch.copy_(world_rays)

    def _read_backend_hit_geom_ids(self) -> torch.Tensor:
        assert self._ray_geomid is not None
        return wp.to_torch(self._ray_geomid).view(self._num_envs, self._num_rays)

    def _get_mutable_raycast_outputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self._distances is not None and self._hit_pos_w is not None and self._normals_w is not None
        return self._distances, self._hit_pos_w, self._normals_w

    def _update_mutable_raycast_outputs(
        self,
        distances: torch.Tensor,
        hit_pos_w: torch.Tensor,
        normals_w: torch.Tensor,
    ) -> None:
        backend_distances, backend_hit_pos_w, backend_normals_w = self._get_mutable_raycast_outputs()
        backend_distances.copy_(distances)
        backend_hit_pos_w.copy_(hit_pos_w)
        backend_normals_w.copy_(normals_w)

    def _get_backend_render_context(self):
        assert self._runtime_ctx is not None
        return self._runtime_ctx.render_context

    def _build_body_aliases(self) -> dict[str, set[str]]:
        body_aliases: dict[str, set[str]] = {}
        for mesh_name, link_name in self.cfg.aux_mesh_and_link_names.items():
            mesh_name = str(mesh_name).strip()
            if mesh_name == "":
                continue
            link_name = mesh_name if link_name is None else str(link_name).strip()
            if link_name == "":
                continue
            key = link_name.lower()
            if key not in body_aliases:
                body_aliases[key] = set()
            body_aliases[key].add(mesh_name)
        return body_aliases

    def _build_geom_path_candidates(
        self,
        mj_model,
        geom_id: int,
        body_aliases: dict[str, set[str]],
    ) -> list[str]:
        full_geom_name = mj_model.geom(geom_id).name or ""
        body_id = int(mj_model.geom_bodyid[geom_id])
        full_body_name = mj_model.body(body_id).name or ""

        entity_name, local_body_name = self._split_prefixed_name(full_body_name)
        _, local_geom_name = self._split_prefixed_name(full_geom_name)

        body_tokens = [local_body_name]
        body_tokens.extend(sorted(body_aliases.get(local_body_name.lower(), set())))
        body_tokens = [token for token in body_tokens if token != ""]
        if len(body_tokens) == 0:
            body_tokens = [f"body_{body_id}"]

        paths: set[str] = set()
        if full_body_name:
            paths.add(f"/{full_body_name}")
        if full_geom_name:
            paths.add(f"/{full_geom_name}")

        entity_lower = entity_name.lower()
        for body_token in body_tokens:
            if entity_name:
                paths.add(f"/World/{entity_name}/{body_token}")
                paths.add(f"/World/envs/env_0/{entity_name}/{body_token}")
                if local_geom_name:
                    paths.add(f"/World/{entity_name}/{body_token}/{local_geom_name}")
                    paths.add(f"/World/envs/env_0/{entity_name}/{body_token}/{local_geom_name}")

            if entity_lower == "robot":
                paths.add(f"/World/envs/env_0/Robot/{body_token}")
                if local_geom_name:
                    paths.add(f"/World/envs/env_0/Robot/{body_token}/{local_geom_name}")

            if (
                entity_lower in {"terrain", "ground"}
                or "ground" in full_body_name.lower()
                or "ground" in full_geom_name.lower()
            ):
                paths.add("/World/ground/")
                paths.add(f"/World/ground/{body_token}")
                if local_geom_name:
                    paths.add(f"/World/ground/{body_token}/{local_geom_name}")

        return sorted(path for path in paths if path != "")

    @staticmethod
    def _split_prefixed_name(full_name: str) -> tuple[str, str]:
        if "/" in full_name:
            return full_name.split("/", 1)
        return "", full_name
