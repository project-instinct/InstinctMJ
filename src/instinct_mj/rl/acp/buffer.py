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

    def __init__(self, T: int, N: int, device: str, holdout_envs: float = 0.0):
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
        super().__init__(T, N, NUM_CONTACT, device, holdout_envs, fields=fields)

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

    def sample_windows(self, B: int, k: int, env_range=None, failure_frac: float = 0.0, failure_horizon: int = 16, balance_frac: float = 0.0, failure_first_done: bool = False):
        """As ``ReplayBuffer.sample_windows``; with ``balance_frac`` > 0 that fraction of the windows is instead drawn clip-balanced: every clip that has a
        buffered window start is equally likely, then one of its window starts uniformly (needs ``clip_table``: global frame -> clip, set by the runner).
        The importance ``weight`` of those windows is 1 (the failure-head weights only undo the failure oversampling)."""
        out = super().sample_windows(B, k, env_range, failure_frac, failure_horizon, failure_first_done)
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
