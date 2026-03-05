from __future__ import annotations

import re
from typing import TYPE_CHECKING

import torch
import warp as wp
from mujoco_warp import rays

from mjlab.sensor import RayCastSensor

if TYPE_CHECKING:
    from .grouped_ray_caster_cfg import GroupedRayCasterCfg


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
        self._env_ids_grid = torch.empty(0, 0, dtype=torch.long)
        self._ray_ids_grid = torch.empty(0, 0, dtype=torch.long)
        self._active_hop_capacity = 0
        self._active_hop_pnt = torch.empty(0, 0, 3)
        self._active_hop_vec = torch.empty(0, 0, 3)
        self._active_hop_dist = torch.empty(0, 0)
        self._active_hop_geomid = torch.empty(0, 0, dtype=torch.int32)
        self._active_hop_normal = torch.empty(0, 0, 3)
        self._active_hop_bodyexclude = torch.empty(0, dtype=torch.int32)
        self._ray_bodyexclude_value = -1

    def initialize(self, mj_model, model, data, device: str) -> None:
        super().initialize(mj_model, model, data, device)

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
        self._env_ids_grid = self._ALL_INDICES.unsqueeze(1).expand(-1, self._num_rays)
        self._ray_ids_grid = (
            torch.arange(self._num_rays, device=device, dtype=torch.long).unsqueeze(0).expand(self._num_envs, -1)
        )
        ray_bodyexclude_torch = wp.to_torch(self._ray_bodyexclude)
        if ray_bodyexclude_torch.numel() > 0:
            self._ray_bodyexclude_value = int(ray_bodyexclude_torch[0].item())
        else:
            self._ray_bodyexclude_value = -1
        self._initialize_mesh_path_filter(mj_model, device)
        self._needs_filter_continue = self._mesh_filter_enabled or self._min_distance > 0.0

    def prepare_rays(self) -> None:
        """PRE-GRAPH: Transform per-env local rays to world frame."""
        if self._frame_type == "body":
            frame_pos = self._data.xpos[:, self._frame_body_id]
            frame_mat = self._data.xmat[:, self._frame_body_id].view(-1, 3, 3)
        elif self._frame_type == "site":
            frame_pos = self._data.site_xpos[:, self._frame_site_id]
            frame_mat = self._data.site_xmat[:, self._frame_site_id].view(-1, 3, 3)
        else:  # geom
            frame_pos = self._data.geom_xpos[:, self._frame_geom_id]
            frame_mat = self._data.geom_xmat[:, self._frame_geom_id].view(-1, 3, 3)

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

        pnt_torch = wp.to_torch(self._ray_pnt).view(self._num_envs, self._num_rays, 3)
        vec_torch = wp.to_torch(self._ray_vec).view(self._num_envs, self._num_rays, 3)
        pnt_torch.copy_(world_origins)
        vec_torch.copy_(ray_directions_w)

        self._cached_world_origins = world_origins
        self._cached_world_rays = ray_directions_w
        self._cached_frame_pos = frame_pos
        self._cached_frame_mat = frame_mat

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
                "GroupedRayCaster mesh filter matched no MuJoCo geoms."
                f"\n\tmesh_prim_paths: {self.cfg.mesh_prim_paths}"
            )

        allowed_geom_lut = torch.zeros(int(mj_model.ngeom), device=device, dtype=torch.bool)
        allowed_geom_lut[torch.tensor(sorted(set(allowed_geom_ids)), device=device, dtype=torch.long)] = True
        self._allowed_geom_lut = allowed_geom_lut
        self._mesh_filter_enabled = True

    def _apply_hit_filter_and_continue(self) -> None:
        ray_geomid_torch = wp.to_torch(self._ray_geomid).view(self._num_envs, self._num_rays)

        geom_ids = ray_geomid_torch.to(dtype=torch.long)
        hit_mask = self._distances >= 0.0
        if not torch.any(hit_mask):
            return

        if self._mesh_filter_enabled:
            allowed_mask = torch.zeros_like(hit_mask)
            allowed_mask[hit_mask] = self._allowed_geom_lut[geom_ids[hit_mask]]
        else:
            allowed_mask = torch.ones_like(hit_mask)

        reject_mask = hit_mask & ((~allowed_mask) | (self._distances <= self._min_distance))
        if not torch.any(reject_mask):
            return

        final_distances = self._distances.clone()
        final_hit_pos = self._hit_pos_w.clone()
        final_normals = self._normals_w.clone()

        # Rejected hits are initialized to miss.
        final_distances[reject_mask] = -1.0
        final_hit_pos[reject_mask] = self._cached_world_origins[reject_mask]
        final_normals[reject_mask] = 0.0

        world_rays = self._cached_world_rays
        eps = self._mesh_filter_epsilon
        max_hops = self._mesh_filter_max_hops

        current_origins = self._hit_pos_w.clone()
        current_origins[reject_mask] = self._hit_pos_w[reject_mask] + world_rays[reject_mask] * eps

        traveled = torch.zeros_like(self._distances)
        traveled[reject_mask] = self._distances[reject_mask] + eps
        remaining = self.cfg.max_distance - traveled
        active = reject_mask & (remaining > 0.0)
        if not torch.any(active):
            self._distances.copy_(final_distances)
            self._hit_pos_w.copy_(final_hit_pos)
            self._normals_w.copy_(final_normals)
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

        self._distances.copy_(final_distances)
        self._hit_pos_w.copy_(final_hit_pos)
        self._normals_w.copy_(final_normals)

    def _ensure_active_hop_buffers(self, capacity: int) -> None:
        if capacity <= self._active_hop_capacity:
            return
        device = self._distances.device
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
        self._active_hop_capacity = capacity

    def _raycast_active_rays(
        self,
        *,
        active: torch.Tensor,
        current_origins: torch.Tensor,
        world_rays: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        active_slot_ids = torch.cumsum(active.to(dtype=torch.int32), dim=1) - 1
        active_env_ids = self._env_ids_grid[active]
        active_ray_ids = self._ray_ids_grid[active]
        active_slot_ids = active_slot_ids[active].to(dtype=torch.long)
        max_active = int(active_slot_ids.max().item()) + 1
        self._ensure_active_hop_buffers(max_active)

        active_hop_pnt = self._active_hop_pnt[:, :max_active, :]
        active_hop_vec = self._active_hop_vec[:, :max_active, :]
        active_hop_dist = self._active_hop_dist[:, :max_active]
        active_hop_geomid = self._active_hop_geomid[:, :max_active]
        active_hop_normal = self._active_hop_normal[:, :max_active, :]
        active_hop_bodyexclude = self._active_hop_bodyexclude[:max_active]

        active_hop_pnt.zero_()
        active_hop_vec.zero_()
        active_hop_vec[..., 2] = 1.0
        active_hop_dist.fill_(-1.0)
        active_hop_geomid.fill_(-1)
        active_hop_normal.zero_()

        active_hop_pnt[active_env_ids, active_slot_ids] = current_origins[active]
        active_hop_vec[active_env_ids, active_slot_ids] = world_rays[active]

        rays(
            m=self._model.struct,  # type: ignore[attr-defined]
            d=self._data.struct,  # type: ignore[attr-defined]
            pnt=wp.from_torch(active_hop_pnt.contiguous(), dtype=wp.vec3),
            vec=wp.from_torch(active_hop_vec.contiguous(), dtype=wp.vec3),
            geomgroup=self._geomgroup,  # pyright: ignore[reportArgumentType]
            flg_static=True,
            bodyexclude=wp.from_torch(active_hop_bodyexclude.contiguous(), dtype=wp.int32),
            dist=wp.from_torch(active_hop_dist.contiguous(), dtype=wp.float32),
            geomid=wp.from_torch(active_hop_geomid.contiguous(), dtype=wp.int32),
            normal=wp.from_torch(active_hop_normal.contiguous(), dtype=wp.vec3),
            rc=self._ctx.render_context,
        )

        return (
            active_env_ids,
            active_ray_ids,
            active_hop_dist[active_env_ids, active_slot_ids],
            active_hop_geomid[active_env_ids, active_slot_ids].to(dtype=torch.long),
            active_hop_normal[active_env_ids, active_slot_ids],
        )

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

            if entity_lower in {"terrain", "ground"} or "ground" in full_body_name.lower() or "ground" in full_geom_name.lower():
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
