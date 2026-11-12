"""Replay buffer of the ACP runner: the SuperMimic ``ReplayBuffer`` with the ACP fields (context, goal, action history)."""

from __future__ import annotations

import torch

from instinct_mj.rl.supermimic.buffer import ReplayBuffer
from .mdp import ACT_DIM, ODO_HIST, STATE_DIM, XI_DYN_DIM
from .motion import NUM_CONTACT


class ACPReplayBuffer(ReplayBuffer):
    window_keys = ("s", "s1", "a", "a_prev", "contact", "done", "terminated", "f1", "phase1", "origin1", "xi_dyn", "goal1")
    start_keys = ("s", "a_prev", "a_prev2", "f", "phase", "origin", "ep_step", "xi_dyn", "goal", "odo_drift", "odo_delay")
    history_back = ODO_HIST - 1
    """Steps of real history an imagination start needs: the pelvis pose history of the odometry model (the proprio histories need 4)."""
    hist_keys = ("s", "a_prev")

    def __init__(self, T: int, N: int, device: str):
        fields = {
            "s": (STATE_DIM, torch.float32),
            "s1": (STATE_DIM, torch.float32),
            "a": (ACT_DIM, torch.float32),
            "a_prev": (ACT_DIM, torch.float32),
            "a_prev2": (ACT_DIM, torch.float32),
            "contact": (NUM_CONTACT, torch.float32),  # c_meas of the 9 contact links at s1
            "origin": (3, torch.float32),
            "r": ((), torch.float32),
            "f": ((), torch.long),
            "f1": ((), torch.long),  # at s1 (pre-reset)
            "phase": ((), torch.float32),
            "phase1": ((), torch.float32),
            "origin1": (3, torch.float32),
            "terminated": ((), torch.bool),  # REAL termination (not a timeout / dataset end)
            "ep_step": ((), torch.int32),
            "done": ((), torch.bool),
            "xi_dyn": (XI_DYN_DIM, torch.float32),  # per-episode context at s: encoder bias, action delay
            "goal": (5, torch.float32),  # goal target at s: position (3), heading, goal time in clip-local frames
            "goal1": (5, torch.float32),  # ... at s1 (pre-reset)
            "odo_drift": (3, torch.float32),  # odometry drift of the episode at s
            "odo_delay": ((), torch.float32),  # odometry delay [s] of the episode at s
        }
        super().__init__(T, N, NUM_CONTACT, device, fields=fields)
        self._version = 0  # mutation counter (``add``): keys the cached complete-window tables
        self._cw_cache = None

    def eligible_starts(self, ok_fn, env_range=None, max_age=None, key: str = "f") -> torch.Tensor:
        """As ``ReplayBuffer.eligible_starts``, restricted to states whose full real history is in the buffer: ``age >= min(ep_step,
        history_back)``, so the histories (proprio, pelvis poses, contact link positions) of a start are never back-filled from the
        oldest buffered state instead of the episode start."""
        lo, hi = env_range if env_range is not None else (0, self.N)
        n_age = self.size if max_age is None else max_age + 1
        rows = self._row(torch.arange(n_age, device=self.device))
        ep = self.data["ep_step"][rows][:, lo:hi].long()
        ok = ok_fn(self.data[key][rows][:, lo:hi]) & (torch.arange(n_age, device=self.device).unsqueeze(1) >= torch.clamp(ep, max=self.history_back))
        idx = ok.nonzero(as_tuple=False)
        idx[:, 1] += lo
        return idx

    def sample_windows(self, B: int, k: int, env_range=None, balance_frac: float = 0.0):
        """As ``ReplayBuffer.sample_windows`` (uniform over the buffer); with ``balance_frac`` > 0 that fraction of the windows is instead drawn clip-balanced: every clip that has a
        buffered window start is equally likely, then one of its window starts uniformly (needs ``clip_table``: global frame -> clip, set by the runner).
        ``weight`` is 1 for every window."""
        out = super().sample_windows(B, k, env_range)
        n_bal = int(B * balance_frac)
        if n_bal == 0:
            return out
        lo, hi = env_range if env_range is not None else (0, self.N)
        n_win, n_env = self.size - k + 1, hi - lo
        rows_all = self._row(torch.arange(n_win, device=self.device))
        clip = self.clip_table[self.data["f"][rows_all][:, lo:hi]].flatten()  # (n_win * n_env,)
        weight = 1.0 / torch.bincount(clip).float()[clip]
        pick = torch.multinomial(weight, n_bal, replacement=True)
        t0, env = pick // n_env, pick % n_env + lo
        rows = self._row(t0.unsqueeze(-1) + torch.arange(k, device=self.device))
        e = env.unsqueeze(-1).expand_as(rows)
        bal = {key: self.data[key][rows, e] for key in self.window_keys}
        d = bal["done"].float()
        bal["mask"] = torch.cumprod(torch.cat((torch.ones(n_bal, 1, device=self.device), 1 - d[:, :-1]), -1), -1)
        bal["env"], bal["age0"], bal["weight"] = env, t0, torch.ones(n_bal, device=self.device)
        for key, v in bal.items():
            out[key][:n_bal] = v
        return out

    def add(self, tr: dict):
        super().add(tr)
        self._version += 1  # invalidates the cached complete-window tables

    def complete_window_tables(self, k: int, env_range=None):
        """Eligibility tables of the complete k-transition windows of the envs ``env_range``: ``(flat_prefix, base, counts, weights, any_window)``.

        A window is complete if no episode ends before its last transition. ``flat_prefix`` is the inclusive running count of the complete windows over the flattened
        ``(env, window start)`` table, ``base`` the count before every env, ``counts`` the windows of every env, ``weights`` the (float) eligibility of the env and
        ``any_window`` a python bool (the only host sync, once per rebuild). Cached under (mutation counter, ptr, size, k, env_range): the world-model batches of one
        iteration, between which the buffer does not change, share one rebuild."""
        lo, hi = env_range if env_range is not None else (0, self.N)
        key = (getattr(self, "_version", 0), self.ptr, self.size, k, lo, hi)
        cache = getattr(self, "_cw_cache", None)
        if cache is not None and cache[0] == key:
            return cache[1]
        rows_all = self._row(torch.arange(self.size, device=self.device))
        ended = self.data["done"][rows_all, lo:hi].long()
        cum = torch.cat((ended.new_zeros(1, hi - lo), ended.cumsum(0)), 0)
        n = self.size - k + 1
        valid = (cum[k - 1:k - 1 + n] == cum[:n]).T.contiguous()  # (env, window start)
        counts = valid.sum(-1)
        flat_prefix = valid.flatten().cumsum(0)
        base = counts.cumsum(0) - counts
        weights = (counts > 0).float()
        tables = (flat_prefix, base, counts, weights, bool((counts > 0).any()))
        self._cw_cache = (key, tables)
        return tables

    def episode_length_stats(self, env_range=None) -> str:
        """Short text on the episode lengths in the buffer (host syncs: for the rare no-complete-window diagnostic)."""
        lo, hi = env_range if env_range is not None else (0, self.N)
        rows = self._row(torch.arange(self.size, device=self.device))
        done = self.data["done"][rows, lo:hi]
        t = torch.arange(self.size, device=self.device).unsqueeze(1).expand_as(done)
        last = torch.where(done, t, torch.full_like(t, -1)).cummax(0).values  # row of the last done at or before every row
        prev = torch.cat((torch.full_like(last[:1], -1), last[:-1]))
        seg = (t - prev)[done & (prev >= 0)].float()  # lengths (transitions) of the episodes that start and end inside the buffer
        running = (t[-1] - last[-1]).float()  # transitions since the last episode end of every env
        txt = f"{int(done.sum())} episode ends in {self.size} steps x {hi - lo} envs"
        if seg.numel():
            txt += f"; fully buffered episodes: n {seg.numel()} length mean {seg.mean():.1f} max {int(seg.max())}"
        return txt + f"; running episodes: mean {running.mean():.1f} max {int(running.max())}"

    def sample_complete_windows(self, B: int, k: int, env_range=None):
        """Uniform eligible env, then uniform k-transition window within that env; ``None`` if no env has a complete window (the caller keeps collecting).

        Local replay stores true pre-reset s1, so done on the last transition
        is allowed. Unlike upstream state-only replay, terminal targets survive.
        """
        if not 1 <= k <= self.size:
            raise ValueError(f"need 1 <= k <= buffer size, got {k}, {self.size}")
        lo, hi = env_range if env_range is not None else (0, self.N)
        n = self.size - k + 1
        flat_prefix, base, counts, weights, any_window = ACPReplayBuffer.complete_window_tables(self, k, env_range)
        if not any_window:
            return None
        local_env = torch.multinomial(weights, B, replacement=True)
        rank = (torch.rand(B, device=self.device) * counts[local_env]).long().minimum(counts[local_env] - 1)
        # the rank-th complete window of the env: first flat position whose running count reaches base + rank + 1
        age = torch.searchsorted(flat_prefix, (base[local_env] + rank + 1), right=False) - local_env * n
        env = local_env + lo
        rows = self._row(age[:, None] + torch.arange(k, device=self.device))
        result = {key: self.data[key][rows, env[:, None]] for key in self.window_keys}
        result.update(mask=torch.ones(B, k, device=self.device), env=env, age0=age, weight=torch.ones(B, device=self.device))
        return result
