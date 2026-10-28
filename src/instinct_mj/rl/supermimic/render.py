"""Differentiable depth renderer of a task camera against the terrain AND the robot's own links, with the env's own ray-cast kernel.

The world model replaces the simulator, so the depth image of the policy is a function of the robot state, ``o = f(S)``: pose of
the camera mount + ray casting against the scene, differentiable w.r.t. the pose of the body the camera is attached to.
:meth:`DepthRenderer.render` returns the raw sensor measurement (``distance_to_image_plane`` of the whole pixel grid, in
meters); running the sensor-equivalent noise pipeline on it (``DepthPipeline.noised``) gives the policy's observation (its
random stages are fresh samples, independent of the image, i.e. reparameterized).

The measurement reproduces ``GroupedRayCasterCamera`` (``instinct_mj.sensors.grouped_ray_caster``):

- pose: camera position ``p_body + R_body offset_pos``, orientation ``R_body R(offset_rot)`` (the full body orientation;
  the cfg's ``ray_alignment`` is not used by the camera sensor); the rays are the sensor's own ``ray_starts`` /
  ``ray_directions`` of every env (pinhole intrinsics, pixel centers, plus the startup installation-error randomization);
- the same mujoco_warp ``rays`` kernel (BVH accelerated, exact mesh geometry), the same geom groups (``include_geom_groups``,
  which includes the robot's visual links, and ``exclude_parent_body``) and the same skip-and-continue loop for hits closer than
  ``min_distance`` (``mesh_filter_epsilon``, ``mesh_filter_max_hops``) as the sensor; ``distance_to_image_plane`` = hit distance
  along the ray * camera-forward component of the ray direction; rays that miss or hit farther than ``max_distance`` give
  ``max_distance`` (``depth_clipping_behavior="max"``).

Geometry: the kernel runs on a PRIVATE copy of the env's FULL model (terrain + robot; own data with one world per imagined
sample, render context and BVH; nothing is shared with the env's sensors, its CUDA graph or its streams). Per call the predicted
``qpos`` (root pose + joint angles) of every sample is written into its world, ``kinematics`` and ``refit_bvh`` run, and the
camera rays are cast per world: one cast covers the terrain and the robot's own links (self-occlusion), as in the sensor. With ``render_robot=False`` the robot's geoms are moved to a geom group that is not enabled (terrain only, no self-occlusion). The
model is the nominal one (the startup domain randomization of geom sizes / poses is not applied).

Gradient (implicit function trick): the hit distance ``t*`` and the surface normal ``n`` come from the kernel without
gradient; ``t = n . (x* - o) / (n . d)`` is evaluated WITH gradient on the ray origin ``o`` and direction ``d`` (``x*`` = hit
point, constant): the value is ``t*`` and the derivative is the exact first-order one for a locally planar surface that is
fixed in the world. The same spirit as ``volume_points_penetration`` of the parkour heads. ``n . d`` is clamped away from 0
(sign preserving) in the gradient only; misses give ``max_distance`` with zero gradient. Hits on the robot's own links (they
move with the state, not fixed in the world) have zero gradient.
"""

from __future__ import annotations

import copy

import mujoco
import mujoco_warp as mjwarp
import torch
import warp as wp
from mujoco_warp import rays

from .rotations import mat_to_quat, mv, quat_to_mat

_VEC6 = wp.types.vector(length=6, dtype=float)
ND_MIN = 0.05  # clamp of |n . d| in the gradient (grazing hits)


class Worlds:
    """``nworld`` worlds (data + render context with its BVH) of the shared model; ``cast`` shoots ``n`` rays per world."""

    def __init__(self, m, mjm, mjd, mjm_bvh, nworld: int, geomgroup, groups: list[int], device: str):
        self.m, self.geomgroup, self.device, self.nworld = m, geomgroup, device, nworld
        with wp.ScopedDevice(device):
            self.d = mjwarp.put_data(mjm, mjd, nworld=nworld, nconmax=1, njmax=1)
            self.rc = mjwarp.create_render_context(
                mjm=mjm_bvh, nworld=nworld, cam_res=None, render_rgb=None, render_depth=None, use_textures=False, use_shadows=False,
                enabled_geom_groups=groups, cam_active=[False] * mjm.ncam, render_seg=None,
            )  # fmt: skip
        self.qpos = wp.to_torch(self.d.qpos)  # (nworld, nq)
        self.no_exclude: dict[int, wp.array] = {}  # per number of rays per world

    def update(self) -> None:
        """Forward kinematics of the written ``qpos`` and BVH refit."""
        mjwarp.kinematics(self.m, self.d)
        mjwarp.refit_bvh(self.m, self.d, self.rc)

    def cast(self, pnt: torch.Tensor, vec: torch.Tensor):
        """Rays (nworld, n, 3), each in its own world: fresh outputs ``dist`` (nworld, n) (-1 = miss), ``normal``, ``geomid``."""
        W, n = pnt.shape[:2]
        if n not in self.no_exclude:
            self.no_exclude[n] = wp.from_torch(torch.full((n,), -1, dtype=torch.int32, device=self.device), dtype=wp.int32)
        dist = torch.empty(W, n, device=self.device)
        normal = torch.empty(W, n, 3, device=self.device)
        geomid = torch.empty(W, n, dtype=torch.int32, device=self.device)
        rays(
            m=self.m, d=self.d, pnt=wp.from_torch(pnt, dtype=wp.vec3), vec=wp.from_torch(vec, dtype=wp.vec3), geomgroup=self.geomgroup,
            flg_static=True, bodyexclude=self.no_exclude[n], dist=wp.from_torch(dist, dtype=wp.float32),
            geomid=wp.from_torch(geomid, dtype=wp.int32), normal=wp.from_torch(normal, dtype=wp.vec3), rc=self.rc,
        )  # fmt: skip
        return dist, normal, geomid.long()


class SceneRays:
    """mujoco_warp ``rays`` on a private copy of the env's model (see the module docstring), without host syncs.

    ``render_robot=True``: the full model, one world per sample (``capacity`` worlds): (qpos -> kinematics -> BVH refit -> rays)
    per call. ``render_robot=False``: terrain only, identical for every sample: the robot's geoms are moved to a geom group that
    is not enabled and the rays of all samples are packed into the single world of the data.

    The skip-and-continue loop of the sensor (hits closer than ``min_distance``) is needed by few samples (the camera of an
    imagined state inside or right at a surface) but by all of their rays, and the sensor runs up to ``mesh_filter_max_hops``
    passes. Here ``HOPS`` dense passes run on the ``HOP_ROWS`` samples with the most rejected rays (``topk``, no ``nonzero``);
    their rays that are not continued are replaced by rays that miss the scene at once. What the budget does not cover
    (``dropped`` samples, ``unresolved`` rays after the last pass) is reported as misses and counted."""

    HOPS = 3
    HOP_ROWS = 128

    def __init__(self, env, capacity: int, render_robot: bool, device: str):
        cam = env.scene["camera"]
        c = cam.cfg
        assert not c.exclude_parent_body and not c.mesh_prim_paths, "the camera sees its own parent body and has no mesh filter"
        idx = env.scene["robot"].indexing
        self.free_q, self.joint_q = idx.free_joint_q_adr.long(), idx.joint_q_adr.long()
        self.packed, self.device = not render_robot, device
        self.min_distance, self.eps, self.max_distance = c.min_distance, c.mesh_filter_epsilon, c.max_distance
        assert c.mesh_filter_max_hops >= self.HOPS
        mjm = copy.deepcopy(env.sim.mj_model)
        terrain = mjm.geom_bodyid == mjm.body("terrain").id
        self.terrain_geom = torch.as_tensor(terrain, device=device)  # (ngeom,) bool
        if self.packed:  # the robot's geoms are moved to a geom group that is not enabled
            mjm.geom_group[~terrain] = (max(c.include_geom_groups) + 1) % mujoco.mjNGROUP
            assert mjm.geom_group[~terrain].max() not in c.include_geom_groups
        geomgroup = _VEC6(*[-1.0 if g in c.include_geom_groups else 0.0 for g in range(mujoco.mjNGROUP)])
        mjd = mujoco.MjData(mjm)
        mujoco.mj_kinematics(mjm, mjd)
        mjm_bvh = copy.deepcopy(mjm)  # the BVH is built from the default pose, refit per call: build it with the robot far away
        mjm_bvh.qpos0[self.free_q[:3].cpu().numpy()] += 1e4  # from the terrain so that its links form their own subtree
        with wp.ScopedDevice(device):
            m = mjwarp.put_model(mjm)
        groups = sorted(c.include_geom_groups)
        self.rows = min(self.HOP_ROWS, capacity)
        self.main = Worlds(m, mjm, mjd, mjm_bvh, 1 if self.packed else capacity, geomgroup, groups, device)
        self.hop = self.main if self.packed else Worlds(m, mjm, mjd, mjm_bvh, self.rows, geomgroup, groups, device)
        self.dropped = torch.zeros((), dtype=torch.long, device=device)  # samples with rejected rays beyond the budget (summed over calls)
        self.unresolved = torch.zeros((), dtype=torch.long, device=device)  # rays still skipped after the last pass (summed over calls)
        self.away = torch.tensor([0.0, 0.0, 1.0], device=device)
        self.graphs: dict[int, tuple] = {}  # per batch size: (CUDA graph of ``_cast``, its static inputs, its static outputs)

    def _worlds_cast(self, worlds: Worlds, o: torch.Tensor, d: torch.Tensor):
        """(rows, P, 3) rays; the packed terrain has one world, the rows are flattened into it."""
        if not self.packed:
            return worlds.cast(o, d)
        return tuple(x.view(*o.shape[:2], *x.shape[2:]) for x in worlds.cast(o.reshape(1, -1, 3), d.reshape(1, -1, 3)))

    def _first_hits(self, o: torch.Tensor, d: torch.Tensor):
        """``GroupedRayCaster._apply_hit_filter_and_continue`` without mesh filter, for (rows, P, 3) rays (rows = worlds, or
        samples of the packed world): hits at distance <= ``min_distance`` are skipped and the ray continues behind them (from
        ``hit + eps``); returns the final total distance along the ray from ``o`` (-1 = miss), the normal and the geom id of the
        first accepted hit."""
        dist0, normal0, geomid = self._worlds_cast(self.main, o, d)
        reject = (dist0 >= 0) & (dist0 <= self.min_distance)
        dist, normal = torch.where(reject, -1.0, dist0), torch.where(reject.unsqueeze(-1), 0.0, normal0)
        count = reject.sum(1)
        rows = count.topk(min(self.rows, count.shape[0])).indices  # the samples with most rejected rays (the others change nothing)
        self.dropped += (count > 0).sum() - (count[rows] > 0).sum()
        if not self.packed:
            self.hop.qpos.copy_(self.main.qpos[rows])
            self.hop.update()
        so, sd, active = o[rows], d[rows], reject[rows]
        sdist, snormal, sgeom = dist[rows], normal[rows], geomid[rows]
        traveled = torch.where(active, dist0[rows] + self.eps, 0.0)  # distance covered by the continued rays
        for _ in range(self.HOPS):
            a = active.unsqueeze(-1)
            hd, hn, hg = self._worlds_cast(self.hop, torch.where(a, so + sd * traveled.unsqueeze(-1), 1e6), torch.where(a, sd, self.away))
            hit = active & (hd >= 0)  # (max_distance is far beyond the scene)
            total = traveled + hd
            accept = hit & (total > self.min_distance)
            sdist, snormal, sgeom = torch.where(accept, total, sdist), torch.where(accept.unsqueeze(-1), hn, snormal), torch.where(accept, hg, sgeom)
            active = hit & ~accept
            traveled = torch.where(active, total + self.eps, traveled)
        self.unresolved += active.sum()
        return dist.index_copy(0, rows, sdist), normal.index_copy(0, rows, snormal), geomid.index_copy(0, rows, sgeom)

    @torch.compiler.disable
    @torch.no_grad()
    def cast(self, p: torch.Tensor, R: torch.Tensor, q: torch.Tensor, o: torch.Tensor, d: torch.Tensor):
        """Total hit distance (B, P) (-1 = miss), surface normal (B, P, 3) and hit geom id (B, P) of the camera rays
        ``o + t d`` (unit ``d``, (B, P, 3)) of the robots with root pose ``(p, R)`` and joint angles ``q``. The whole sequence
        (qpos write, kinematics, refit, rays, skip-and-continue passes: ~150 small launches) is captured once per batch size
        in a CUDA graph and replayed (the launches would otherwise cost about as much CPU time as the GPU needs)."""
        B = p.shape[0]
        if B not in self.graphs:
            static = tuple(x.clone() for x in (p, R, q, o, d))
            self._cast(*static)  # warm-up: loads the warp modules, fills the allocator
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = self._cast(*static)
            self.graphs[B] = (graph, static, out)
        graph, static, out = self.graphs[B]
        for dst, src in zip(static, (p, R, q, o, d)):
            dst.copy_(src)
        graph.replay()
        return tuple(x.clone() for x in out)

    def _cast(self, p: torch.Tensor, R: torch.Tensor, q: torch.Tensor, o: torch.Tensor, d: torch.Tensor):
        B, P = o.shape[:2]
        with wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(self.device)), sync_enter=False), wp.ScopedDevice(self.device):
            if self.packed:
                return self._first_hits(o, d)
            out, W = [], self.main.nworld
            for s in range(0, B, W):
                b = min(W, B - s)
                self.main.qpos[:b, self.free_q[:3]] = p[s : s + b]
                self.main.qpos[:b, self.free_q[3:7]] = mat_to_quat(R[s : s + b])
                self.main.qpos[:b, self.joint_q] = q[s : s + b]
                self.main.update()
                pnt, vec = o[s : s + b], d[s : s + b]
                if b < W:  # unused worlds get a dummy ray
                    pnt = torch.cat((pnt, torch.full((W - b, P, 3), 1e6, device=self.device)))
                    vec = torch.cat((vec, self.away.expand(W - b, P, 3)))
                out.append(tuple(x[:b] for x in self._first_hits(pnt.contiguous(), vec.contiguous())))
            return tuple(torch.cat(x) for x in zip(*out)) if len(out) > 1 else out[0]


class DepthRenderer:
    def __init__(self, env, capacity: int, render_robot: bool, device: str):
        """Reads the camera constants from the live (unwrapped) env's ``camera`` sensor."""
        self.cam = env.scene["camera"]
        c = self.cam.cfg
        assert c.frame.type == "body" and c.frame.name == "torso_link" and c.offset.convention == "world" and c.depth_clipping_behavior == "max"
        assert c.data_types == ["distance_to_image_plane"]
        self.shape = c.pattern.height, c.pattern.width
        self.off_p = torch.tensor(c.offset.pos, dtype=torch.float32, device=device)
        self.off_R = quat_to_mat(torch.tensor(c.offset.rot, dtype=torch.float32, device=device))
        self.max_distance = c.max_distance
        self.scene = SceneRays(env, capacity, render_robot, device)

    def rays(self, p: torch.Tensor, R: torch.Tensor, env: torch.Tensor):
        """Camera rays (origins ``o`` (B, P, 3), unit directions ``d`` (B, P, 3), camera-forward component of ``d``
        (B, P)) of the cameras of envs ``env`` (B,) attached to the body with world position ``p`` (B, 3) and rotation ``R``
        (B, 3, 3); differentiable w.r.t. both."""
        Rc = R @ self.off_R
        pos = p + mv(R, self.off_p)
        starts, dirs_local = self.cam.ray_starts[env], self.cam.ray_directions[env]  # (B, P, 3), camera frame (+x forward)
        o = pos.unsqueeze(1) + mv(Rc.unsqueeze(1), starts)
        d = mv(Rc.unsqueeze(1), dirs_local)
        return o, d, dirs_local[..., 0]

    def measure(self, o, d, forward, dist, n, geomid) -> torch.Tensor:
        """Raw sensor measurement (``distance_to_image_plane`` [m], max-clipped) (B, H, W, 1) from the rays and their hits
        (``dist``, ``n``, ``geomid`` of ``SceneRays.cast``): the value is the kernel's hit distance, the gradient w.r.t. the rays is
        the implicit-function one for terrain hits and zero for hits on the robot's links (see the module docstring)."""
        hit = (dist >= 0) & (dist <= self.max_distance)
        x = o.detach() + dist.clamp(min=0).unsqueeze(-1) * d.detach()
        nd = (n * d).sum(-1)
        nd = torch.where(nd < 0, -1.0, 1.0) * nd.detach().abs().clamp(min=ND_MIN) + (nd - nd.detach())
        t_lin = (n * (x - o)).sum(-1) / nd  # = dist up to rounding, with the first-order dependence on o and d
        t = dist + (t_lin - t_lin.detach()) * self.scene.terrain_geom[geomid]  # exact value (also at grazing angles), gradient of the linearization
        return torch.where(hit, t * forward, self.max_distance).view(-1, *self.shape, 1)

    def render(self, p: torch.Tensor, R: torch.Tensor, q: torch.Tensor, env: torch.Tensor) -> torch.Tensor:
        """Raw sensor measurement (B, H, W, 1) of the cameras of envs ``env`` for the root pose ``(p, R)`` and joint angles
        ``q`` (B, nq); differentiable w.r.t. the pose (``rays`` -> ``SceneRays.cast`` -> ``measure``)."""
        o, d, forward = self.rays(p, R, env)
        dist, n, geomid = self.scene.cast(p, R, q, o.detach(), d.detach())
        return self.measure(o, d, forward, dist, n, geomid)
