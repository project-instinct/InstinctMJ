"""Fixed-size GPU replay buffer of simulator transitions, laid out as (time, env).

Adapted from the SuperMimic reference implementation (MIT License), Chong Zhang / ETH Zurich.

``s1`` is always the state right after the physics step, i.e. BEFORE the env reset (the runner steps with
``auto_reset=False`` and resets explicitly afterwards). A transition with ``done`` therefore keeps a valid ``s1`` and is
trained on; only the transitions after it in a window (the next episode) are masked, as in the reference.
The episode-constant context ``xi`` lives in a per-env table.
"""

from __future__ import annotations

import torch

from .mdp import DEPTH_SHAPE, NUM_J, STATE_DIM


class ReplayBuffer:
    # fields returned by ``sample_windows`` / ``starts_at`` (start state and its observation history)
    window_keys = ("s", "s1", "a", "contact", "done", "terminated", "f1", "phase1", "origin1")
    start_keys = ("s", "a_prev", "f", "phase", "origin")
    hist_keys = ("s", "a_prev")

    def __init__(self, T: int, N: int, num_contact: int, device: str, holdout_envs: float = 0.0, fields: dict | None = None, depth: bool = False):
        """``holdout_envs``: fraction of the envs (the last ones) that is never sampled for training (``train_envs``) and
        only used for held-out diagnostics (``holdout_envs`` range). ``fields``: {name: (shape, dtype)} of a transition
        (default: the perceptive shadowing fields below); a task with other fields also sets the ``*_keys`` above. ``depth`` (default
        fields only): also store the env's depth images (the visual latent of ``vision="latent"``)."""
        self.T, self.N, self.device = T, N, device
        n_hold = int(N * holdout_envs)
        self.train_envs, self.holdout_envs = (0, N - n_hold), ((N - n_hold, N) if n_hold > 0 else (0, N))
        fields = fields if fields is not None else {
            "s": (STATE_DIM, torch.float32),
            "s1": (STATE_DIM, torch.float32),
            "a": (NUM_J, torch.float32),
            "a_prev": (NUM_J, torch.float32),
            "contact": (num_contact, torch.float32),  # indicator of the undesired-contact bodies at s1
            "origin": (3, torch.float32),  # terrain origin of the env's motion at s (world = local + origin)
            "r": ((), torch.float32),
            "f": ((), torch.long),  # global reference frame index at s
            "f1": ((), torch.long),  # ... at s1 (pre-reset)
            "phase1": ((), torch.float32),
            "origin1": (3, torch.float32),
            "terminated": ((), torch.bool),  # the transition ended in a REAL termination (not a timeout / dataset end)
            "phase": ((), torch.float32),  # fractional reference time at s (see MotionLib.current_frame)
            "ep_step": ((), torch.int32),
            "done": ((), torch.bool),
        }
        if depth:
            fields.update({
                "depth": (DEPTH_SHAPE, torch.float16),  # the env's (noised) processed policy ``depth_image`` at s
                "depth1": (DEPTH_SHAPE, torch.float16),  # the env's (noised) policy ``depth_image`` at s1 (pre-reset): latent target
                "depth1_clean": (DEPTH_SHAPE, torch.float16),  # CLEAN processed depth image at s1 (pre-reset): reconstruction target
            })
            self.window_keys = self.window_keys + ("depth", "depth1", "depth1_clean")
            self.start_keys = self.start_keys + ("depth",)
        self.data = {}
        for k, (shape, dt) in fields.items():
            shp = (T, N) + ((shape,) if isinstance(shape, int) else tuple(shape))
            self.data[k] = torch.zeros(shp, dtype=dt, device=device)
        self.ptr, self.size = 0, 0

    def add(self, tr: dict):
        for k in self.data:
            self.data[k][self.ptr] = tr[k]
        self.ptr = (self.ptr + 1) % self.T
        self.size = min(self.size + 1, self.T)

    def _row(self, age: torch.Tensor) -> torch.Tensor:
        """Map age-ordered time index (0 = oldest) to the storage row."""
        return (self.ptr - self.size + age) % self.T

    def sample_windows(self, B: int, k: int, env_range: tuple[int, int] | None = None, failure_frac: float = 0.0, failure_horizon: int = 16):
        """k-step windows; returns dict of (B, k, ...) tensors, the env index (B,), a validity mask (B, k) and, for the
        importance-weighted heads, ``weight`` (B,) that undoes the failure oversampling.

        A fraction ``failure_frac`` of the windows is drawn from the windows with a real termination at one of the
        ``failure_horizon`` transitions starting at the window start (so the approach to the failure is covered); the rest is
        uniform. ``weight`` = uniform density / mixture density, i.e. 1 / ((1 - q) + q * W / F) for windows in the failure set
        (W = number of windows, F = size of the failure set, q = failure share) and 1 / (1 - q) otherwise."""
        lo, hi = env_range if env_range is not None else (0, self.N)
        n_win, n_env = self.size - k + 1, hi - lo
        n_fail = int(B * failure_frac)
        t0 = torch.randint(0, n_win, (B,), device=self.device)
        env = torch.randint(lo, hi, (B,), device=self.device)
        weight = torch.ones(B, device=self.device)
        if n_fail > 0:
            # no host syncs: the failure set is kept as a mask, the failure windows are drawn with ``multinomial`` over it
            rows_all = self._row(torch.arange(self.size, device=self.device))
            term = self.data["terminated"][rows_all][:, lo:hi].int()
            cum = torch.nn.functional.pad(term.cumsum(0), (0, 0, 1, 0))  # (size + 1, n): terminations in rows < t
            t0_all = torch.arange(n_win, device=self.device)
            t1_all = (t0_all + failure_horizon).clamp(max=self.size)
            fail_set = (cum[t1_all] - cum[t0_all]) > 0  # (n_win, n_env)
            n_failure_windows = fail_set.sum()
            probs = torch.where(n_failure_windows > 0, fail_set.flatten().float(), torch.ones(n_win * n_env, device=self.device))
            pick = torch.multinomial(probs, n_fail, replacement=True)
            t0[:n_fail], env[:n_fail] = pick // n_env, pick % n_env + lo
            perm = torch.randperm(B, device=self.device)
            t0, env = t0[perm], env[perm]
            # importance weights undoing the oversampling (q = 0 when there is no failure window: plain uniform sampling)
            q = (n_fail / B) * (n_failure_windows > 0).float()
            in_fail = fail_set[t0, env - lo]
            weight = torch.where(in_fail, 1.0 / ((1 - q) + q * fail_set.numel() / n_failure_windows.clamp(min=1)), 1.0 / (1 - q))
        ages = t0.unsqueeze(-1) + torch.arange(k, device=self.device)
        rows = self._row(ages)
        e = env.unsqueeze(-1).expand_as(rows)
        out = {key: self.data[key][rows, e] for key in self.window_keys}
        # transition j is valid if no episode ended at transitions 0..j-1 (the done transition itself is valid)
        d = out["done"].float()
        out["mask"] = torch.cumprod(torch.cat((torch.ones(B, 1, device=self.device), 1 - d[:, :-1]), -1), -1)
        out["env"], out["age0"] = env, t0
        out["weight"] = weight
        return out

    def eligible_starts(
        self, ok_fn, env_range: tuple[int, int] | None = None, max_age: int | None = None, key: str = "f"
    ) -> torch.Tensor:
        """(M, 2) ``(age, env)`` pairs of all buffered states for which ``ok_fn(data[key])`` holds (default: the global
        reference frame index)."""
        lo, hi = env_range if env_range is not None else (0, self.N)
        rows = self._row(torch.arange(self.size if max_age is None else max_age + 1, device=self.device))
        ok = ok_fn(self.data[key][rows][:, lo:hi])  # (n_age, n_env)
        idx = ok.nonzero(as_tuple=False)
        idx[:, 1] += lo
        return idx

    def starts_at(self, age: torch.Tensor, env: torch.Tensor, hist: int):
        """Start points for imagination plus the indices of their observation history (oldest first)."""
        row = self._row(age)
        ep_step = self.data["ep_step"][row, env].long()
        back = torch.arange(hist - 1, -1, -1, device=self.device)  # hist-1 ... 0
        back = torch.minimum(back.unsqueeze(0), ep_step.unsqueeze(-1))  # stay inside the episode
        back = torch.minimum(back, age.unsqueeze(-1))  # stay inside the buffer
        hrows = self._row(age.unsqueeze(-1) - back)
        henv = env.unsqueeze(-1).expand_as(hrows)
        start = {k: self.data[k][row, env] for k in self.start_keys}
        hist_d = {k: self.data[k][hrows, henv] for k in self.hist_keys}
        # slots with the same ``back`` are the same (backfilled) observation
        hist_d["slot"] = (back.unsqueeze(2) == back.unsqueeze(1)).float().argmax(-1)
        start["env"], hist_d["env"] = env, env
        return start, hist_d
