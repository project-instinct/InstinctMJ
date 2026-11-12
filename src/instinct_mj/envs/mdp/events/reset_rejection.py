"""Batched penetration rejection of ``reset_robot_state_by_reference``.

The sequential rejection (redraw the reset noise of the penetrating envs, up to ``max_resample`` times, then fall back to the noise-free reference) measures the penetration with a
full kinematics + collision pass over ALL worlds of the env's data after every round, each with a host sync. The draws of one env are independent and the penetration of a state
is a function of the state alone, so the sequential rule is equivalent to: draw ``K = 1 + max_resample`` candidates at once, take the first one that does not penetrate by more
than ``max_penetration`` (the noise-free reference if none does). All candidates (and the references) of the reset envs are then evaluated in ONE kinematics + collision pass on a
small private mujoco_warp ``Data`` (``AuxPenetrationProbe``: ``nworld`` = a power of two >= the number of states, a CUDA graph per capacity) and the chosen state is written to
the real sim once. The probe uses a private copy of the nominal model (as ``rl/acp/rays.py``): it is only valid if no event changed the collision / kinematic fields of the env
model (``AuxPenetrationProbe.build`` checks that and returns the reason otherwise; the caller then uses the sequential rejection).

Contact-buffer overflow: the contacts beyond the probe's contact buffer are not seen, so the depths of a pass that overflowed are underestimates (of the candidates AND of the
references the lift reads). The probe reports whether ANY pass of a call overflowed (``AuxPenetrationProbe.depth`` returns the flag); the caller then discards the batched result of
that call and runs the exact sequential rejection on the real sim instead (``reset_robot_state_by_reference``). Nothing is guessed from the truncated depths.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import mujoco
import mujoco_warp as mjwarp
import torch
import warp as wp

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv as ManagerBasedEnv

# model fields that kinematics + collision detection read and that a randomization event could change
_COLLISION_FIELDS = (
    "body_pos", "body_quat", "body_mocapid", "jnt_pos", "jnt_axis",
    "geom_type", "geom_contype", "geom_conaffinity", "geom_condim", "geom_dataid", "geom_priority", "geom_size", "geom_aabb", "geom_rbound",
    "geom_pos", "geom_quat", "geom_margin", "geom_gap", "mesh_pos", "mesh_quat",
)  # fmt: skip


def robot_terrain_masks(env: ManagerBasedEnv, asset_cfg, terrain_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Boolean geom masks ``(robot, terrain)`` of shape ``(ngeom,)`` (cached on the env)."""
    cache = getattr(env, "_penetration_geom_masks", None)
    if cache is None or cache[0] != (asset_cfg.name, terrain_name):
        ngeom = env.sim.mj_model.ngeom
        robot = torch.zeros(ngeom, dtype=torch.bool, device=env.device)
        terrain = torch.zeros(ngeom, dtype=torch.bool, device=env.device)
        robot[env.scene[asset_cfg.name].indexing.geom_ids.long()] = True
        terrain[env.scene[terrain_name].indexing.geom_ids.long()] = True
        cache = ((asset_cfg.name, terrain_name), robot, terrain)
        env._penetration_geom_masks = cache
    return cache[1], cache[2]


def reduce_contact_depth(data, robot: torch.Tensor, terrain: torch.Tensor, nworld: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-world deepest robot-vs-terrain penetration ``(nworld,)`` of the contacts of ``data`` (``max(-contact.dist, 0)`` over the contacts whose geom pair is one robot and one
    terrain geom; 0 = no penetrating contact) and the contact-buffer overflow flag (0-dim bool: the contacts beyond the buffer are not seen). Two buffers can overflow: the contact
    buffer itself (``nacon`` > ``naconmax``) and the broadphase pair buffer, which has the same size (``ncollision`` > ``naconmax``: the pairs beyond it never reach the narrowphase,
    so ``nacon`` stays below ``naconmax`` and does not show it)."""
    contact = data.contact
    dist, world = wp.to_torch(contact.dist), wp.to_torch(contact.worldid).long()
    geom = wp.to_torch(contact.geom).long()  # (naconmax, 2)
    nacon = wp.to_torch(data.nacon)  # (1,)
    valid = torch.arange(dist.shape[0], device=dist.device) < nacon
    g0, g1 = geom[:, 0].clamp(min=0), geom[:, 1].clamp(min=0)
    pair = (robot[g0] & terrain[g1]) | (terrain[g0] & robot[g1])
    depth = torch.where(valid & pair, -dist, torch.zeros_like(dist)).clamp(min=0.0)
    out = torch.zeros(nworld, dtype=dist.dtype, device=dist.device)
    out.scatter_reduce_(0, world.clamp(0, nworld - 1), depth, reduce="amax")
    return out, (nacon[0] > dist.shape[0]) | (wp.to_torch(data.ncollision)[0] > dist.shape[0])


def select_first_passing(depth: torch.Tensor, max_penetration: float) -> tuple[torch.Tensor, torch.Tensor]:
    """The sequential "redraw until clear" rule on the penetration depths ``(R, K)`` of K candidates per env in draw order: the index ``(R,)`` of the FIRST candidate with
    ``depth <= max_penetration`` and the fallback mask ``(R,)`` (no candidate is clear; the index is then K - 1, a valid but unused value)."""
    fail = depth > max_penetration
    first = fail.long().cumprod(dim=1).sum(dim=1)  # number of leading failures = index of the first pass (K if none)
    fallback = first >= depth.shape[1]
    return first.clamp(max=depth.shape[1] - 1), fallback


class AuxPenetrationProbe:
    """Robot-vs-terrain penetration depth of arbitrary robot states on a private small ``Data`` (see the module docstring)."""

    MIN_CAPACITY = 128
    CONTACT_BUFFER_SCALE = 2
    """The aux contact buffer holds this many times the env's contacts per world (a call with a pass that still overflows is redone sequentially by the caller, see :meth:`depth`)."""
    PAD_HEIGHT = 1000.0
    """Height [m] added to the root of the unused (padding) worlds of a pass: far above any terrain, so they produce no robot-vs-terrain contacts."""

    def __init__(self, env: ManagerBasedEnv, asset_cfg, terrain_name: str, max_capacity: int = 2048):
        sim = env.sim
        self.device, self.wp_device, self.use_graph = env.device, sim.wp_device, sim.use_cuda_graph
        self.max_capacity = max_capacity
        idx = env.scene[asset_cfg.name].indexing
        self.free_q, self.joint_q = idx.free_joint_q_adr.long(), idx.joint_q_adr.long()
        self.robot, self.terrain = robot_terrain_masks(env, asset_cfg, terrain_name)
        self.mjm = copy.deepcopy(sim.mj_model)
        self.mjd = mujoco.MjData(self.mjm)
        mujoco.mj_forward(self.mjm, self.mjd)
        # contacts per world as in the env's data x CONTACT_BUFFER_SCALE (the aux buffer is shared by the worlds of a pass)
        self.nconmax = self.CONTACT_BUFFER_SCALE * max(int(sim.wp_data.naconmax) // max(int(sim.wp_data.nworld), 1), 64)
        with wp.ScopedDevice(self.wp_device):
            self.model = mjwarp.put_model(self.mjm)
            sim.cfg.apply_wp_opt(self.model.opt)
        self.pad_qpos = torch.as_tensor(self.mjd.qpos, dtype=torch.float32, device=self.device).clone()  # the nominal pose, lifted far above the terrain
        self.pad_qpos[self.free_q[2]] += self.PAD_HEIGHT
        self.slots: dict[int, tuple] = {}  # capacity -> (data, qpos torch view, graph)
        self.num_calls = 0
        self.num_overflow_calls = 0  # calls with an overflowed pass (their batched result was discarded: sequential fallback); set by :meth:`note_overflow`

    @classmethod
    def build(cls, env: ManagerBasedEnv, asset_cfg, terrain_name: str, max_capacity: int = 2048) -> tuple[AuxPenetrationProbe | None, str]:
        """``(probe, "")`` or ``(None, reason)`` when the nominal model differs from the model the env simulates (collision / kinematic randomization), the terrain moves, or the
        robot has no free joint."""
        sim = env.sim
        idx = env.scene[asset_cfg.name].indexing
        if getattr(idx, "free_joint_q_adr", None) is None or len(idx.free_joint_q_adr) != 7:
            return None, "the robot has no free joint"
        mjm = sim.mj_model
        terrain_geoms = env.scene[terrain_name].indexing.geom_ids.long().cpu()
        terrain_bodies = torch.as_tensor(mjm.geom_bodyid, dtype=torch.long)[terrain_geoms]
        if (torch.as_tensor(mjm.body_mocapid, dtype=torch.long)[terrain_bodies] >= 0).any():
            return None, "the terrain is a mocap body"
        probe = cls(env, asset_cfg, terrain_name, max_capacity)
        changed = []
        for name in _COLLISION_FIELDS:
            env_arr, aux_arr = getattr(sim.wp_model, name, None), getattr(probe.model, name, None)
            if env_arr is None or aux_arr is None or env_arr.size == 0:
                continue
            a, b = wp.to_torch(env_arr), wp.to_torch(aux_arr)
            if a.shape != b.shape:
                if a.shape[1:] != b.shape[1:]:
                    changed.append(name)
                    continue
                b = b.expand_as(a)  # the aux model has the nominal single world: every world of the env model must equal it
            if not torch.equal(a, b):
                changed.append(name)
        if changed:
            return None, f"the env model differs from the nominal one in {changed} (randomized collision geometry)"
        return probe, ""

    def _slot(self, capacity: int):
        slot = self.slots.get(capacity)
        if slot is None:
            with wp.ScopedDevice(self.wp_device):
                data = mjwarp.put_data(self.mjm, self.mjd, nworld=capacity, nconmax=self.nconmax, njmax=max(int(self.mjd.nefc), 1))
            qpos = wp.to_torch(data.qpos)
            graph = None
            if self.use_graph:
                with wp.ScopedDevice(self.wp_device):
                    mjwarp.kinematics(self.model, data)  # eager warm-up (kernel compilation, scratch allocation) before the capture
                    mjwarp.collision(self.model, data)
                    with wp.ScopedCapture() as capture:
                        mjwarp.kinematics(self.model, data)
                        mjwarp.collision(self.model, data)
                graph = capture.graph
            slot = self.slots[capacity] = (data, qpos, graph)
        return slot

    @torch.no_grad()
    def depth(self, pose: torch.Tensor, joint_pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Deepest robot-vs-terrain penetration ``(n,)`` of the robot states (root pose ``(n, 7)`` = pos + wxyz quat, joint positions ``(n, num_joints)``) and the overflow flag
        (0-dim bool tensor on the device, no host sync: True if the contact buffer of ANY pass of this call overflowed).

        The contacts of a pass beyond the probe's contact buffer are not seen, so the depths of an overflowed pass are underestimates and must not be used: the caller reads the flag
        (:meth:`note_overflow`) and, if set, discards this result and measures sequentially. The unused (padding) worlds of a pass are parked far above the terrain."""
        n = pose.shape[0]
        out = torch.empty(n, dtype=pose.dtype, device=pose.device)
        overflowed = torch.zeros((), dtype=torch.bool, device=pose.device)
        for s in range(0, n, self.max_capacity):
            e = min(s + self.max_capacity, n)
            capacity = min(max(self.MIN_CAPACITY, 1 << (e - s - 1).bit_length()), self.max_capacity)
            data, qpos, graph = self._slot(capacity)
            qpos[: e - s, self.free_q] = pose[s:e]
            qpos[: e - s, self.joint_q] = joint_pos[s:e]
            if e - s < capacity:
                qpos[e - s :] = self.pad_qpos  # (a stale row of an earlier pass could penetrate and use up the shared contact buffer)
            with wp.ScopedDevice(self.wp_device):
                if graph is not None:
                    wp.capture_launch(graph)
                else:
                    mjwarp.kinematics(self.model, data)
                    mjwarp.collision(self.model, data)
            depth, overflow = reduce_contact_depth(data, self.robot, self.terrain, capacity)
            out[s:e] = depth[: e - s]
            overflowed |= overflow
        return out, overflowed

    def note_overflow(self, overflowed: torch.Tensor) -> bool:
        """The overflow flag of a :meth:`depth` call on the host (ONE device sync per reset call; overflow is expected to be (almost) never) and the bookkeeping of the sequential
        fallbacks, with a rate-limited print (the first fallback, then every 100th, and a one-time note after 128 clean calls)."""
        self.num_calls += 1
        over = bool(overflowed)
        if over:
            self.num_overflow_calls += 1
            if self.num_overflow_calls == 1 or self.num_overflow_calls % 100 == 0:
                print(f"[reset rejection] WARNING: the contact buffer of the auxiliary penetration probe overflowed in {self.num_overflow_calls} of {self.num_calls} reset calls "
                      f"({self.nconmax} contacts per world): those resets were redone with the sequential rejection on the env's data", flush=True)  # fmt: skip
        elif self.num_calls == 128 and self.num_overflow_calls == 0:
            print(f"[reset rejection] contact buffer of the auxiliary penetration probe: no overflow in the first {self.num_calls} reset calls ({self.nconmax} contacts per world)", flush=True)
        return over
