"""Exact height scan of the imagined states: mujoco_warp ray casts against the terrain geoms (primitives and meshes), with the ray origins and
directions of the env's :class:`~instinct_mj.tasks.acp.mdp.sensors.PelvisRayCastSensor` (pelvis xy + yaw-aligned grid, origin ``ray_offset`` above
the pelvis, straight down, terrain geom group only, ``max_distance``).

The analytic height map of :class:`~instinct_mj.rl.supermimic.terrain.HeightScanner` (``0.05 m`` bilinear) misses thin structures such as ladder rungs;
the rays do not. The scan is the exact one; its gradient w.r.t. the pose is the exact derivative almost everywhere (1 for the root height, 0 for its
horizontal position and heading), because a ray hit is piecewise constant.
"""

from __future__ import annotations

import copy

import mujoco
import mujoco_warp as mjwarp
import torch
import warp as wp

from instinct_mj.rl.supermimic.networks import RunningNorm
from instinct_mj.rl.supermimic.render import _VEC6, Worlds
from instinct_mj.tasks.acp.mdp.observations import SCAN_NUM_X as SCAN_NX
from instinct_mj.tasks.acp.mdp.observations import SCAN_NUM_Y as SCAN_NY
from instinct_mj.tasks.acp.mdp.observations import fill_scan_misses
from instinct_mj.rl.supermimic.rotations import mm, yaw_mat
from instinct_mj.rl.supermimic.terrain import HeightScanner

TERRAIN_GROUP = 0

_SCANNERS: list = []  # the ray casters the custom op ``instinct_mj::ray_cast`` dispatches to (by index: a custom op takes tensors and plain values only)


@torch.library.custom_op("instinct_mj::ray_cast", mutates_args=())
def _ray_cast(origins: torch.Tensor, scanner: int) -> torch.Tensor:
    """Hit distance (B, K) of straight-down rays from ``origins`` (B, K, 3), -1 = no hit (see ``ExactHeightScanner._cast``). No gradient (a ray hit is piecewise constant)."""
    return _SCANNERS[scanner]._cast_impl(origins)


@_ray_cast.register_fake
def _(origins: torch.Tensor, scanner: int) -> torch.Tensor:
    return origins.new_empty(origins.shape[:2])


@torch.library.custom_op("instinct_mj::ray_cast_normal", mutates_args=())
def _ray_cast_normal(origins: torch.Tensor, scanner: int) -> tuple[torch.Tensor, torch.Tensor]:
    """As ``ray_cast`` plus the surface normal (B, K, 3) of the hit (``ExactHeightScanner.exact_height_and_slope``)."""
    return _SCANNERS[scanner]._cast_normal_impl(origins)


@_ray_cast_normal.register_fake
def _(origins: torch.Tensor, scanner: int) -> tuple[torch.Tensor, torch.Tensor]:
    return origins.new_empty(origins.shape[:2]), origins.new_empty(origins.shape)


class ExactHeightScanner(HeightScanner):
    """``HeightScanner`` whose ground heights are exact ray casts (see the module docstring); same interface (``offsets``, ``height_scan``)."""

    def __init__(self, env, heightmap_scanner: HeightScanner, sensor_name: str = "critic_height_scanner", heightmap_loader=None):
        # the grid of the heightmap scanner and the ray settings of the env's sensor
        self.__dict__.update(heightmap_scanner.__dict__)
        self._heightmap_loader = heightmap_loader
        cfg = env.scene[sensor_name].cfg
        assert cfg.include_geom_groups == (TERRAIN_GROUP,) and cfg.ray_alignment == "yaw"
        self.ray_offset, self.cast_max_distance, self.retry_shift = cfg.ray_offset, cfg.max_distance, cfg.retry_shift
        device = str(heightmap_scanner.offsets.device)
        mjm = copy.deepcopy(env.sim.mj_model)
        terrain = mjm.geom_bodyid == mjm.body("terrain").id
        mjm.geom_group[~terrain] = TERRAIN_GROUP + 5  # the robot is not part of the scene (the sensor excludes it)
        self.has_meshes = bool((mjm.geom_type[terrain] == mujoco.mjtGeom.mjGEOM_MESH).any())
        mjd = mujoco.MjData(mjm)
        mujoco.mj_kinematics(mjm, mjd)
        with wp.ScopedDevice(device):
            m = mjwarp.put_model(mjm)
        geomgroup = _VEC6(*[-1.0 if g == TERRAIN_GROUP else 0.0 for g in range(mujoco.mjNGROUP)])
        self.worlds = Worlds(m, mjm, mjd, mjm, 1, geomgroup, [TERRAIN_GROUP], device)  # one static world: the rays of all samples are packed into it
        self.worlds.update()
        self.slope_gradient = False  # the runner sets it from scan_slope_gradient
        self.scanner_id = len(_SCANNERS)
        _SCANNERS.append(self)
        self.down = torch.tensor([0.0, 0.0, -1.0], device=device)
        self.device = device

    def ground_height(self, xy: torch.Tensor) -> torch.Tensor:
        if self._heightmap_loader is not None:
            heights, x0, y0 = self._heightmap_loader()
            self.heights = torch.tensor(heights, device=self.device)[None, None]
            self.x0, self.y0 = x0, y0
            self._heightmap_loader = None
        return super().ground_height(xy)

    def _cast(self, origins: torch.Tensor) -> torch.Tensor:
        """Hit distance (B, K) of straight-down rays from ``origins`` (B, K, 3), -1 = no hit within the sensor's ``max_distance``: a custom op, so that the compiled
        graphs around it are not split (``origins`` is detached: the result never carried a gradient)."""
        return _ray_cast(origins.detach(), self.scanner_id)

    @torch.no_grad()
    def _cast_impl(self, origins: torch.Tensor) -> torch.Tensor:
        B, K = origins.shape[:2]
        with wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(self.device)), sync_enter=False), wp.ScopedDevice(self.device):
            dist, _, _ = self.worlds.cast(origins.reshape(1, -1, 3).contiguous(), self.down.expand(1, B * K, 3).contiguous())
        dist = dist.view(B, K)
        return torch.where(dist > self.cast_max_distance, -1.0, dist)

    @torch.no_grad()
    def _cast_normal_impl(self, origins: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, K = origins.shape[:2]
        with wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(self.device)), sync_enter=False), wp.ScopedDevice(self.device):
            dist, normal, _ = self.worlds.cast(origins.reshape(1, -1, 3).contiguous(), self.down.expand(1, B * K, 3).contiguous())
        dist = dist.view(B, K)
        return torch.where(dist > self.cast_max_distance, -1.0, dist), normal.view(B, K, 3)

    def ground_below(self, points: torch.Tensor) -> torch.Tensor:
        """Height (B, K) of the terrain directly below the world points (B, K, 3) (straight-down exact ray casts from ``ray_offset`` above each point, the shifted second
        cast for the rays that miss on mesh terrains); -1000 where there is none. No gradient."""
        origins = points + torch.tensor([0.0, 0.0, self.ray_offset], device=points.device)
        dist = self._cast(origins)
        if self.has_meshes:
            shift = torch.tensor([*self.retry_shift, 0.0], device=self.device)
            dist = torch.where(dist < 0, self._cast(origins + shift), dist)
        return torch.where(dist < 0, torch.full_like(dist, -1000.0), origins[..., 2] - dist)

    def exact_height(self, p: torch.Tensor, R: torch.Tensor, offsets: torch.Tensor | None = None, shape: tuple[int, int] = (SCAN_NY, SCAN_NX)) -> torch.Tensor:
        """``p_z - ground`` (B, K) of the straight-down rays of the pelvis pose ``(p, R)``, saturated at ``p_z`` minus the sensor's range
        (a ray without a hit); the rays that still miss after the shifted second cast are filled as in the env's observation. No gradient.
        ``offsets`` / ``shape`` (ny, nx): another yaw-frame grid than the planner scan (the wide world-model / critic scan)."""
        xy = p[:, None, :2] + mm(self.offsets if offsets is None else offsets, yaw_mat(R)[..., :2, :2].transpose(-1, -2))  # (B, K, 2), as HeightScanner.height_scan
        origins = torch.cat((xy, (p[:, None, 2:3] + self.ray_offset).expand(-1, xy.shape[1], 1)), -1)
        dist = self._cast(origins)
        if self.has_meshes:  # the mesh ray cast misses rays inside narrow bands along mesh edges: cast the missed ones again, shifted
            shift = torch.tensor([*self.retry_shift, 0.0], device=self.device)
            dist = torch.where(dist < 0, self._cast(origins + shift), dist)
        saturated = self.cast_max_distance - self.ray_offset
        height = (dist - self.ray_offset).view(-1, *shape)  # p_z - hit_z = ray length - ray_offset
        return fill_scan_misses(height, (dist < 0).view(-1, *shape), saturated).flatten(1)

    def exact_height_and_slope(self, p: torch.Tensor, R: torch.Tensor, offsets: torch.Tensor | None = None, shape: tuple[int, int] = (SCAN_NY, SCAN_NX)) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``exact_height`` (B, K), the horizontal positions of the rays (B, K, 2) with the gradient of ``p`` and ``R``, and the slope (B, K, 2) of the ground at the hits: dz/dx = -n_x / n_z, dz/dy = -n_y / n_z of the hit
        normal; zero where the ray misses and on near-vertical faces (|n_z| < 0.2: the ground height is not a function of the position there)."""
        xy = p[:, None, :2] + mm(self.offsets if offsets is None else offsets, yaw_mat(R)[..., :2, :2].transpose(-1, -2))
        origins = torch.cat((xy, (p[:, None, 2:3] + self.ray_offset).expand(-1, xy.shape[1], 1)), -1)
        dist, normal = _ray_cast_normal(origins.detach(), self.scanner_id)
        if self.has_meshes:
            shift = torch.tensor([*self.retry_shift, 0.0], device=self.device)
            dist2, normal2 = _ray_cast_normal((origins + shift).detach(), self.scanner_id)
            miss = dist < 0
            dist, normal = torch.where(miss, dist2, dist), torch.where(miss.unsqueeze(-1), normal2, normal)
        saturated = self.cast_max_distance - self.ray_offset
        height = (dist - self.ray_offset).view(-1, *shape)
        height = fill_scan_misses(height, (dist < 0).view(-1, *shape), saturated).flatten(1)
        nz = normal[..., 2]
        usable = (dist >= 0) & (nz.abs() > 0.2)
        slope = torch.where(usable.unsqueeze(-1), -normal[..., :2] / torch.where(usable, nz, torch.ones_like(nz)).unsqueeze(-1), torch.zeros_like(normal[..., :2]))
        return height, xy, slope

    def height_scan(self, p: torch.Tensor, R: torch.Tensor, offsets: torch.Tensor | None = None, shape: tuple[int, int] = (SCAN_NY, SCAN_NX)) -> torch.Tensor:
        """Height of the root above the ground at every ray (B, K), clamped at ``max_distance``: the exact value, whose derivative is 1 w.r.t. the root height and, by default, 0 w.r.t. its horizontal position
        and heading (the ground under a ray is piecewise constant; the slope of the 0.05 m height map at an edge is an artefact that, multiplied with the sharp response of the networks to the exact scan,
        made the BPTT gradient explode). That is the derivative on horizontal faces only: with ``slope_gradient`` the ground height has the slope of the hit face (``exact_height_and_slope``) in the gradient."""
        if self.slope_gradient:
            height, xy, slope = self.exact_height_and_slope(p, R, offsets, shape)
            ground = (p[:, 2:3] - height).detach() + (slope * (xy - xy.detach())).sum(-1)
        else:
            ground = (p[:, 2:3] - self.exact_height(p, R, offsets, shape)).detach()
        return torch.clamp(p[:, 2:3] - ground, max=self.max_distance)


class ScanFloorRunningNorm(RunningNorm):
    """``RunningNorm`` for inputs that contain exact height scans: the flat-terrain cells of the data have a tiny variance, so a rung, a step or a gap
    in an exact scan would be a 50-sigma input. The standard deviation of the ``scan_slice`` entries is floored at ``scan_floor`` metres. No value is
    clipped, as zita-ch/supermimic_vibed's ``RunningNorm``: every normalized input keeps its value and its gradient (the former +-8 sigma clip of every
    entry saturated ~26 % of the 32-step author vault rollouts and zeroed their gradients, /tmp/codex_smdiff/REPORT.md 5.2). ``floor`` is a buffer of
    the state dict; ``std()`` is the standard deviation ``forward`` divides by."""

    def __init__(self, dim: int, scan_slice: slice, scan_floor: float = 0.05, **kwargs):
        super().__init__(dim, **kwargs)
        floor = torch.zeros(dim)
        floor[scan_slice] = scan_floor
        self.register_buffer("floor", floor)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        for key in ("clip_mask", "clip"):  # buffers of the former clipped normalizer in older checkpoints
            state_dict.pop(prefix + key, None)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def std(self):
        return torch.sqrt(torch.maximum(self.var, self.floor**2) + self.eps**2)

    def forward(self, x):
        return (x - self.mean) / self.std()
