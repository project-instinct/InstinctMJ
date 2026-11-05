"""ACP Stage 2 / 3 planner training: episode recorder, success filters, hindsight labeling into a partitioned replay, the planner
trainer and the eta schedules (paper "Three-Stage Training Pipeline", "Hindsight Contact Labeling", Eq. 1-3, Tables S4 / S5).

Integration contract (runner hooks)
-----------------------------------
Construction (``ACPRunner.setup_training``)::

    self.recorder = EpisodeRecorder(N, device, max_steps=500)           # float16 proprio / points, see its docstring
    self.replay = PartitionedReplay(ref_capacity, tile_capacity, device)
    self.planner_trainer = PlannerTrainer(device, planner=self.planner)  # shares the module the frozen imagination planner uses

``collect`` (every real env step, right after ``env.step`` and BEFORE the done envs are reset)::

    self.recorder.step(
        proprio=obs_pre[:, :375],              # the 375 first entries of the policy observation the action was computed from
        goal=goal_pre,                         # (N, 3) the (noisy) goal command of that observation
        points=mdp.planner_points(scan_pre),   # (N, 187, 3) encoded planner points (x, y, pelvis_z - hit_z - 0.5) of that state
        contact_force_z=...,                   # (N, 9) vertical force of the CONTACT_LINKS after the step (state s1)
        body_pos_w=...,                        # (N, 9, 3) world position of the 9 contact bodies at s1 (contact point if in contact)
        pelvis_pos_w=..., pelvis_yaw=...,      # (N, 3), (N,) base pose at s1; the yaw is the heading of the base-yaw frame
        is_goal=...,                           # (N,) bool, goal-reaching env (Stage 3) vs reference-following env
        phase=phase1,                          # (N,) reference progress in [0, 1] (clip fraction) at s1, reference envs
        goal_xy=..., goal_dz=...,              # (N,) horizontal distance / |dz| of the base to the goal at s1, goal envs
        work_power=...,                        # (N,) optional, sum_j max(tau_j qd_j, 0) * dt of this control step, goal envs
        terrain_type=..., tile=...,            # (N,) long, goal envs: terrain curriculum type and tile of the episode
    )
    if len(done_ids) > 0:                      # ``done`` of any kind: terminated, timeout, end of the clip
        eps = self.recorder.finish(done_ids)   # only SUCCESSFUL episodes (paper filter), goal segments already truncated at 0.2 m
        stats = add_episodes(self.replay, eps) # hindsight labels (1 N, 10 steps, 75 steps, footprint) -> replay partitions
        self.unwrapped.reset(env_ids=done_ids)

Reference progress must be the clip fraction at s1 (the end of a clip counts as 1.0). The recorder keeps the first ``max_steps``
steps of an episode (paper: at most 500); the success statistics keep running on the whole episode.

``planner_update`` (once per iteration, after ``collect``, paper: ten optimizer updates per PPO iteration)::

    losses = self.planner_trainer.update(self.replay)    # None while len(replay) < 1000; else dict of mean loss parts

It should run in Stage 1 too once the recorder is on (planner initialization from the tracker's successful rollouts), not only for
``contact_source`` != "reference"; the recorded ``proprio`` / ``goal`` / ``points`` are the noisy values the tracker saw.

Planner inference (collection and imagination), occupancy threshold 0.5, k = 1::

    out = planner(proprio, goal, points)
    tgt, valid = AttentionContactPlanner.decode(out, points, k=1, occ_threshold=0.5)   # (B, 9, 3) in the base-yaw frame, invalid -> 0
    tgt, use_planner = mix_contacts(ref_tgt, tgt, eta_obs, generator)                  # whole command per env, no interpolation

The tracker's contact-target observation of the real env is replaced by ``tgt`` with the same mixing as in imagination. ``eta_obs`` /
``eta_rew`` follow ``eta(it, u_obs, N_ann)`` / ``eta(it, u_rew, N_ann)`` (Eq. 3, u_rew > u_obs; 1 / 1 in Stage 3, 0 / 0 in Stage 1).
The planner module is trained in place by the trainer, so a frozen imagination copy must be refreshed after ``update`` (or the
imagination must use ``planner_trainer.planner`` under ``torch.no_grad``).

``PlannerTrainer.state_dict()`` / ``load_state_dict()`` hold planner + Adam; the replay is not saved (it refills from collection).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .hindsight import HindsightReplay, episode_success_goal, episode_success_reference, label_episode
from .planner import NUM_BODIES, NUM_POINTS, PROPRIO_DIM, AttentionContactPlanner, GOAL_DIM, planner_loss

# Table S5 / S4 constants
MAX_EPISODE_STEPS = 500
GOAL_TRUNCATE_DIST = 0.2
GOAL_QUALITY_MIN_DIST = 0.5
GOAL_Z_TOL = 0.1


# ----------------------------------------------------------------------------------------------------------------------
# Recorder
# ----------------------------------------------------------------------------------------------------------------------
@dataclass
class Episode:
    """One finished episode: the stored segment (L, ...) and the running statistics of the whole episode."""

    proprio: torch.Tensor
    goal: torch.Tensor
    points: torch.Tensor
    contact_force_z: torch.Tensor
    body_pos_w: torch.Tensor
    pelvis_pos_w: torch.Tensor
    pelvis_yaw: torch.Tensor
    is_goal: bool
    start_phase: float
    end_phase: float
    min_xy_ok: float
    """Smallest horizontal goal distance over the steps with |dz| <= 0.1 m (inf if none)."""
    near_idx: int
    """First step index (episode time) within 0.2 m of the goal horizontally, -1 if none."""
    init_xy: float
    work: float
    terrain_type: int
    tile: int
    num_steps: int
    """Steps of the whole episode (the stored segment holds the first ``min(num_steps, max_steps)``)."""

    @property
    def success(self) -> bool:
        if self.is_goal:
            return episode_success_goal(self.min_xy_ok)
        return episode_success_reference(self.start_phase, self.end_phase)

    @property
    def quality(self) -> float:
        """Goal episodes: positive mechanical work per unit initial horizontal goal distance (lower is better)."""
        return goal_quality(self.work, self.init_xy)

    def segment(self) -> "Episode":
        """The part used for planner supervision: goal episodes are cut before the first step within 0.2 m of the goal."""
        if not self.is_goal or self.near_idx < 0 or self.near_idx >= self.proprio.shape[0]:
            return self
        L = self.near_idx
        return Episode(**{**self.__dict__, **{k: getattr(self, k)[:L] for k in _STEP_KEYS}})


_STEP_KEYS = ("proprio", "goal", "points", "contact_force_z", "body_pos_w", "pelvis_pos_w", "pelvis_yaw")


def goal_quality(work: float, init_xy: float) -> float:
    """Positive mechanical work normalized by the initial horizontal goal distance, denominator clamped below at 0.5 m."""
    return work / max(init_xy, GOAL_QUALITY_MIN_DIST)


class EpisodeRecorder:
    """GPU-resident per-env record of the current episode (first ``max_steps`` steps) plus running success statistics.

    Memory per env: ``max_steps * (939 * 2 + 40 * 4)`` bytes with the default float16 ``store_dtype`` of the planner inputs (proprio, goal,
    points), i.e. ~1 MB; the label geometry (forces, positions, pelvis pose) stays float32."""

    def __init__(self, num_envs: int, device: str = "cpu", max_steps: int = MAX_EPISODE_STEPS, store_dtype: torch.dtype = torch.float16):
        self.N, self.T, self.device = num_envs, max_steps, device
        f32 = dict(dtype=torch.float32, device=device)
        sd = dict(dtype=store_dtype, device=device)
        self.proprio = torch.zeros(num_envs, max_steps, PROPRIO_DIM, **sd)
        self.goal = torch.zeros(num_envs, max_steps, GOAL_DIM, **sd)
        self.points = torch.zeros(num_envs, max_steps, NUM_POINTS, 3, **sd)
        self.force = torch.zeros(num_envs, max_steps, NUM_BODIES, **f32)
        self.body_pos = torch.zeros(num_envs, max_steps, NUM_BODIES, 3, **f32)
        self.pelvis = torch.zeros(num_envs, max_steps, 3, **f32)
        self.yaw = torch.zeros(num_envs, max_steps, **f32)
        self.count = torch.zeros(num_envs, dtype=torch.long, device=device)  # steps of the current episode (not capped)
        self.is_goal = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.terrain_type = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.tile = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.start_phase = torch.zeros(num_envs, **f32)
        self.max_phase = torch.zeros(num_envs, **f32)
        self.min_xy_ok = torch.full((num_envs,), float("inf"), **f32)
        self.near_idx = torch.full((num_envs,), -1, dtype=torch.long, device=device)
        self.init_xy = torch.zeros(num_envs, **f32)
        self.work = torch.zeros(num_envs, **f32)

    @torch.no_grad()
    def step(
        self,
        proprio: torch.Tensor,
        goal: torch.Tensor,
        points: torch.Tensor,
        contact_force_z: torch.Tensor,
        body_pos_w: torch.Tensor,
        pelvis_pos_w: torch.Tensor,
        pelvis_yaw: torch.Tensor,
        is_goal: torch.Tensor,
        phase: torch.Tensor,
        goal_xy: torch.Tensor,
        goal_dz: torch.Tensor,
        work_power: torch.Tensor | None = None,
        terrain_type: torch.Tensor | None = None,
        tile: torch.Tensor | None = None,
    ) -> None:
        """Append one control step for all envs. Planner inputs are those of the observation the action was computed from; forces,
        positions, progress (``phase`` for reference envs, ``goal_xy`` / ``goal_dz`` for goal envs) are measured after the step."""
        N, c = self.N, self.count
        first = c == 0
        is_goal = is_goal.to(self.device, torch.bool)
        phase, goal_xy, goal_dz = (x.to(self.device, torch.float32) for x in (phase, goal_xy, goal_dz))
        self.is_goal = is_goal
        if terrain_type is not None:
            self.terrain_type = terrain_type.to(self.device, torch.long)
        if tile is not None:
            self.tile = tile.to(self.device, torch.long)
        # running statistics (whole episode)
        self.start_phase = torch.where(first, phase, self.start_phase)
        self.max_phase = torch.where(first, phase, torch.maximum(self.max_phase, phase))
        self.init_xy = torch.where(first, goal_xy, self.init_xy)
        ok = is_goal & (goal_dz.abs() <= GOAL_Z_TOL)
        self.min_xy_ok = torch.where(first, torch.full_like(goal_xy, float("inf")), self.min_xy_ok)
        self.min_xy_ok = torch.where(ok, torch.minimum(self.min_xy_ok, goal_xy), self.min_xy_ok)
        self.near_idx = torch.where(first, torch.full_like(c, -1), self.near_idx)
        self.near_idx = torch.where(is_goal & (self.near_idx < 0) & (goal_xy <= GOAL_TRUNCATE_DIST), c, self.near_idx)
        self.work = torch.where(first, torch.zeros_like(self.work), self.work)
        if work_power is not None:
            self.work = self.work + torch.where(is_goal, work_power.to(self.device, torch.float32), torch.zeros_like(self.work))
        # stored segment: the first T steps
        rows = (c < self.T).nonzero(as_tuple=False).squeeze(-1)
        if len(rows) > 0:
            t = c[rows]
            self.proprio[rows, t] = proprio[rows].to(self.proprio.dtype)
            self.goal[rows, t] = goal[rows].to(self.goal.dtype)
            self.points[rows, t] = points.reshape(N, NUM_POINTS, 3)[rows].to(self.points.dtype)
            self.force[rows, t] = contact_force_z[rows].to(torch.float32)
            self.body_pos[rows, t] = body_pos_w[rows].to(torch.float32)
            self.pelvis[rows, t] = pelvis_pos_w[rows].to(torch.float32)
            self.yaw[rows, t] = pelvis_yaw[rows].to(torch.float32)
        self.count = c + 1

    @torch.no_grad()
    def finish(self, env_ids: torch.Tensor, only_successful: bool = True) -> list[Episode]:
        """Return the episodes of ``env_ids`` that just ended (successful ones only by default, goal segments truncated at the first step
        within 0.2 m of the goal) and reset those envs' records."""
        env_ids = env_ids.to(self.device, torch.long).reshape(-1)
        eps: list[Episode] = []
        if len(env_ids) == 0:
            return eps
        scal = torch.stack([self.is_goal[env_ids].float(), self.start_phase[env_ids], self.max_phase[env_ids], self.min_xy_ok[env_ids],
                            self.near_idx[env_ids].float(), self.init_xy[env_ids], self.work[env_ids], self.terrain_type[env_ids].float(),
                            self.tile[env_ids].float(), self.count[env_ids].float()], dim=1).cpu().tolist()  # fmt: skip
        for e, (g, sp, mp, mx, ni, ix, w, tt, ti, n) in zip(env_ids.tolist(), scal):
            L = min(int(n), self.T)
            if L == 0:
                continue
            ep = Episode(
                proprio=self.proprio[e, :L], goal=self.goal[e, :L], points=self.points[e, :L], contact_force_z=self.force[e, :L],
                body_pos_w=self.body_pos[e, :L], pelvis_pos_w=self.pelvis[e, :L], pelvis_yaw=self.yaw[e, :L], is_goal=bool(g),
                start_phase=sp, end_phase=mp, min_xy_ok=mx, near_idx=int(ni), init_xy=ix, work=w, terrain_type=int(tt), tile=int(ti),
                num_steps=int(n),
            )  # fmt: skip
            if only_successful and not ep.success:
                continue
            ep = ep.segment()
            eps.append(Episode(**{**ep.__dict__, **{k: getattr(ep, k).clone() for k in _STEP_KEYS}}))  # the buffers are reused
        self.reset(env_ids)
        return eps

    def reset(self, env_ids: torch.Tensor) -> None:
        self.count[env_ids] = 0


# ----------------------------------------------------------------------------------------------------------------------
# Labeling into the replay
# ----------------------------------------------------------------------------------------------------------------------
@torch.no_grad()
def label_episode_samples(ep: Episode, **label_kwargs) -> dict[str, torch.Tensor]:
    """Hindsight labels of one (already truncated) episode as a replay batch."""
    target_pos, target_flag = label_episode(ep.contact_force_z, ep.body_pos_w, ep.pelvis_pos_w, ep.pelvis_yaw, **label_kwargs)
    return dict(proprio=ep.proprio.float(), goal=ep.goal.float(), points=ep.points.float(), target_pos=target_pos, target_flag=target_flag)


def add_episodes(replay: "PartitionedReplay", episodes: list[Episode], **label_kwargs) -> dict[str, float]:
    """Label the successful episodes and push them into the replay partition of their type. Returns counts for logging."""
    stats = dict(ref_episodes=0, goal_episodes=0, goal_rejected=0, samples=0)
    for ep in episodes:
        if not ep.success or ep.proprio.shape[0] == 0:
            continue
        batch = label_episode_samples(ep, **label_kwargs)
        if ep.is_goal:
            kept = replay.add_goal(batch, ep.terrain_type, ep.tile, ep.quality)
            stats["goal_episodes"] += int(kept)
            stats["goal_rejected"] += int(not kept)
        else:
            replay.add_reference(batch, ep.terrain_type, ep.tile)
            stats["ref_episodes"] += 1
        stats["samples"] += batch["proprio"].shape[0]
    return stats


# ----------------------------------------------------------------------------------------------------------------------
# Replay
# ----------------------------------------------------------------------------------------------------------------------
class _Tile:
    """Samples of one (terrain type, tile): the episodes concatenated in storage order with their quality."""

    def __init__(self):
        self.data: dict[str, torch.Tensor] | None = None
        self.quality: list[float] = []
        self.sizes: list[int] = []

    @property
    def total(self) -> int:
        return sum(self.sizes)

    def append(self, batch: dict[str, torch.Tensor], quality: float) -> None:
        self.data = batch if self.data is None else {k: torch.cat([self.data[k], batch[k]]) for k in self.data}
        self.quality.append(quality)
        self.sizes.append(batch["proprio"].shape[0])

    def evict_worst(self) -> int:
        """Drop the episode with the highest quality value (most work per distance; the oldest among equals); returns its index."""
        w = max(range(len(self.quality)), key=lambda i: (self.quality[i], -i))
        start = sum(self.sizes[:w])
        keep = torch.ones(self.total, dtype=torch.bool, device=self.data["proprio"].device)
        keep[start:start + self.sizes[w]] = False
        self.data = {k: v[keep] for k, v in self.data.items()}
        del self.quality[w], self.sizes[w]
        return w


class PartitionedReplay:
    """Planner replay of Stage 2 / 3 (paper p22, "Planner optimization and hindsight replay"): a reference partition and a goal-reaching
    partition, each stratified by (terrain type, tile). Reference samples are kept per (terrain type, tile) in a ring of ``ref_tile_capacity``
    samples; goal-reaching samples per (terrain type, tile) with capacity ``tile_capacity`` and quality-based eviction (the episodes with the
    lowest normalized work stay). Reference samples added without a terrain type / tile go to the plain ring ``ref`` (a stratum of their own).

    ``sample``: the two partitions contribute proportionally to their stored sample counts; inside a partition the terrain type is uniform,
    then the tile of that type, then the stored timestep of the tile. ``add`` is the reference-partition alias that makes it a drop-in for
    :class:`HindsightReplay`."""

    def __init__(self, ref_capacity: int, tile_capacity: int, device: str = "cpu", ref_tile_capacity: int | None = None):
        self.device, self.tile_capacity = device, tile_capacity
        self.ref_tile_capacity = ref_capacity if ref_tile_capacity is None else ref_tile_capacity
        self.ref_capacity = ref_capacity
        self.ref = HindsightReplay(0, device)  # the plain ring is allocated when the first sample without identity arrives
        self.ref_tiles: dict[int, dict[int, HindsightReplay]] = {}
        self.tiles: dict[int, dict[int, _Tile]] = {}

    @property
    def goal_size(self) -> int:
        return sum(t.total for tt in self.tiles.values() for t in tt.values())

    @property
    def ref_size(self) -> int:
        return len(self.ref) + sum(len(r) for tt in self.ref_tiles.values() for r in tt.values())

    def __len__(self) -> int:
        return self.ref_size + self.goal_size

    @property
    def reference_allocated_bytes(self) -> int:
        """Bytes currently allocated by the reference partition (the plain ring and the (type, tile) rings)."""
        rings = [self.ref] + [r for tt in self.ref_tiles.values() for r in tt.values()]
        return sum(v.numel() * v.element_size() for r in rings for v in r.data.values())

    def add_reference(self, batch: dict[str, torch.Tensor], terrain_type: int | None = None, tile: int | None = None) -> None:
        if terrain_type is None:
            if self.ref.capacity == 0:
                self.ref = HindsightReplay(self.ref_capacity, self.device)
            self.ref.add(batch)
            return
        rings = self.ref_tiles.setdefault(terrain_type, {})
        if tile not in rings:  # (not ``setdefault(tile, HindsightReplay(...))``: that would allocate a full ring on every call)
            rings[tile] = HindsightReplay(self.ref_tile_capacity, self.device)
        rings[tile].add(batch)

    add = add_reference

    def add_goal(self, batch: dict[str, torch.Tensor], terrain_type: int, tile: int, quality: float) -> bool:
        """Store a goal-reaching episode in its tile and evict the worst-quality episodes of the tile down to the capacity. Returns
        whether the new episode is still stored."""
        n = batch["proprio"].shape[0]
        if n == 0:
            return False
        keys = HindsightReplay.shapes
        batch = {k: batch[k][-self.tile_capacity:].to(self.device, torch.bool if k == "target_flag" else torch.float32) for k in keys}
        t = self.tiles.setdefault(terrain_type, {}).setdefault(tile, _Tile())
        t.append(batch, quality)
        new = len(t.sizes) - 1
        while t.total > self.tile_capacity:
            w = t.evict_worst()
            if w == new:
                return False
            new -= w < new
        return True

    def sample(self, n: int) -> dict[str, torch.Tensor]:
        n_ref, n_goal = self.ref_size, self.goal_size
        if n_ref + n_goal == 0:
            raise ValueError("cannot sample from an empty replay")
        n_g = int((torch.rand(n) < n_goal / (n_ref + n_goal)).sum())
        parts = []
        if n - n_g > 0:
            parts.append(self._sample_reference(n - n_g))
        if n_g > 0:
            parts.append(self._sample_goal(n_g))
        if len(parts) == 1:
            return parts[0]
        perm = torch.randperm(n, device=self.device)
        return {k: torch.cat([p[k] for p in parts])[perm] for k in parts[0]}

    def _sample_reference(self, n: int) -> dict[str, torch.Tensor]:
        """Terrain type uniform, then tile uniform among the tiles of the type, then a stored sample of the tile uniformly (the plain ring of
        the samples added without identity is one more stratum)."""
        strata = [[r for r in d.values() if len(r) > 0] for d in self.ref_tiles.values()]
        strata = [rings for rings in strata if rings]
        if len(self.ref) > 0:
            strata.append([self.ref])
        type_pick = torch.randint(0, len(strata), (n,))
        out: list[dict[str, torch.Tensor]] = []
        for i, rings in enumerate(strata):
            m = int((type_pick == i).sum())
            if m == 0:
                continue
            counts = torch.bincount(torch.randint(0, len(rings), (m,)), minlength=len(rings)).tolist()
            for r, c in zip(rings, counts):
                if c > 0:
                    out.append(r.sample(c))
        return {k: torch.cat([o[k] for o in out]) for k in out[0]}

    def _sample_goal(self, n: int) -> dict[str, torch.Tensor]:
        types = [tt for tt, d in self.tiles.items() if any(t.total > 0 for t in d.values())]
        type_pick = torch.randint(0, len(types), (n,))
        out: list[dict[str, torch.Tensor]] = []
        for i, tt in enumerate(types):
            m = int((type_pick == i).sum())
            if m == 0:
                continue
            tiles = [t for t in self.tiles[tt].values() if t.total > 0]
            counts = torch.bincount(torch.randint(0, len(tiles), (m,)), minlength=len(tiles)).tolist()
            for t, c in zip(tiles, counts):
                if c > 0:
                    idx = torch.randint(0, t.total, (c,), device=self.device)
                    out.append({k: v[idx] for k, v in t.data.items()})
        return {k: torch.cat([o[k] for o in out]) for k in out[0]}


# ----------------------------------------------------------------------------------------------------------------------
# Trainer
# ----------------------------------------------------------------------------------------------------------------------
class PlannerTrainer:
    """Owns the planner and its Adam optimizer (Table S5: lr 1e-4, betas (0.9, 0.999), eps 1e-8, no weight decay)."""

    def __init__(
        self,
        device: str = "cpu",
        lr: float = 1e-4,
        planner: AttentionContactPlanner | None = None,
        updates_per_iter: int = 10,
        batch_size: int = 4096,
        sub_batches: int = 8,
        min_entries: int = 1000,
    ):
        if batch_size % sub_batches != 0:
            raise ValueError("batch_size must be a multiple of sub_batches")
        self.device = device
        self.planner = (planner if planner is not None else AttentionContactPlanner()).to(device)
        self.opt = torch.optim.Adam(self.planner.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
        self.updates_per_iter, self.batch_size, self.sub_batches, self.min_entries = updates_per_iter, batch_size, sub_batches, min_entries
        self.num_updates = 0

    def update(self, replay: HindsightReplay | PartitionedReplay) -> dict[str, float] | None:
        """One planner iteration: ``updates_per_iter`` optimizer steps, each on ``batch_size`` samples accumulated over ``sub_batches``.
        Returns the mean loss parts (affordance, occupancy, occ_acc, top1_err), or None while the replay holds < ``min_entries``."""
        if len(replay) < self.min_entries:
            return None
        self.planner.train()
        acc: dict[str, torch.Tensor] = {}
        for _ in range(self.updates_per_iter):
            batch = replay.sample(self.batch_size)
            self.opt.zero_grad(set_to_none=True)
            valid_batch = batch["target_flag"].to(self.device).sum()  # Eq. 2 normalizes by the valid targets of the whole batch
            for sub in range(self.sub_batches):
                sl = slice(sub * self.batch_size // self.sub_batches, (sub + 1) * self.batch_size // self.sub_batches)
                proprio, goal, points = (batch[k][sl].to(self.device, torch.float32) for k in ("proprio", "goal", "points"))
                tpos, tflag = batch["target_pos"][sl].to(self.device, torch.float32), batch["target_flag"][sl].to(self.device)
                loss, parts = planner_loss(
                    self.planner(proprio, goal, points), points, tpos, tflag, affordance_norm=valid_batch, occupancy_weight=1.0 / self.sub_batches
                )  # the sub-batch losses sum to the loss of the whole batch
                loss.backward()
                for k, v in parts.items():
                    acc[k] = acc.get(k, 0.0) + (v if k == "affordance" else v / self.sub_batches)  # affordance: already a share of the batch
            self.opt.step()
            self.num_updates += 1
        return {k: float(v) / self.updates_per_iter for k, v in acc.items()}

    def state_dict(self) -> dict:
        return dict(planner=self.planner.state_dict(), opt=self.opt.state_dict(), num_updates=self.num_updates)

    def load_state_dict(self, state: dict) -> None:
        self.planner.load_state_dict(state["planner"])
        self.opt.load_state_dict(state["opt"])
        self.num_updates = state["num_updates"]


# ----------------------------------------------------------------------------------------------------------------------
# Schedules / mixing
# ----------------------------------------------------------------------------------------------------------------------
def eta(u: float, u_start: float, n_ann: float) -> float:
    """Eq. (3): 0 for u <= u_start, linear ramp (u - u_start) / n_ann up to u_start + n_ann, 1 afterwards."""
    if u <= u_start:
        return 0.0
    if u >= u_start + n_ann:
        return 1.0
    return (u - u_start) / n_ann


def mix_contacts(
    ref_targets: torch.Tensor | tuple[torch.Tensor, ...],
    planner_targets: torch.Tensor | tuple[torch.Tensor, ...],
    eta_obs: float,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor | tuple[torch.Tensor, ...], torch.Tensor]:
    """Per env, the complete planner command with probability ``eta_obs`` and the complete reference command otherwise (no
    interpolation). Targets are (N, ...) tensors or tuples of them (e.g. targets and validity), which share the per-env choice.
    Returns the mixed targets and the (N,) bool mask of the envs that got the planner command."""
    ref = ref_targets if isinstance(ref_targets, tuple) else (ref_targets,)
    plan = planner_targets if isinstance(planner_targets, tuple) else (planner_targets,)
    use = torch.rand(ref[0].shape[0], device=ref[0].device, generator=generator) < eta_obs
    mixed = tuple(torch.where(use.view(-1, *[1] * (r.dim() - 1)), p, r) for r, p in zip(ref, plan))
    return (mixed if isinstance(ref_targets, tuple) else mixed[0]), use
