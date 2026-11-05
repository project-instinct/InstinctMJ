"""Curriculum State Initialization (CSI) of the ACP tracking task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers import ManagerTermBase

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers import CurriculumTermCfg

    from instinct_mj.motion_reference.motion_files.terrain_motion import TerrainMotion
    from instinct_mj.motion_reference.motion_reference_manager import MotionReferenceManager


class CurriculumStateInit(ManagerTermBase):
    """Per-motion initialization phase phi_m: an episode of motion m starts at phase phi_m of the clip.

    ``mode="csi"``: phi_m starts at ``init_phase``; an episode of motion m starts exactly at phi_m. After an episode of motion
    m ends, phi_m -= step if it reached phase >= ``success_phase`` else phi_m += step (the mean update of the episodes of a motion that
    end in the same env step, then clipped to [0, max_phase]). ``mode="uniform"`` (Stage 2): the start phase is
    U(0, 0.98). ``mode="start"``: the clip starts at phase 0 (evaluation / play). The phase is only updated in ``csi`` mode.
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        params = cfg.params
        self.mode: str = params["mode"]
        if self.mode not in ("csi", "uniform", "start"):
            raise ValueError(f"init_phase_mode must be 'csi', 'uniform' or 'start', got '{self.mode}'.")
        self.step: float = params.get("step", 0.05)
        self.success_phase: float = params.get("success_phase", 0.99)
        self.max_phase: float = params.get("max_phase", 0.99)
        self.uniform_max_phase: float = params.get("uniform_max_phase", 0.98)
        self.motion_reference: MotionReferenceManager = env.scene[params.get("reference_name", "motion_reference")]
        self.motion_buffer: TerrainMotion = next(iter(self.motion_reference.motion_buffers.values()))
        self.phase = torch.full(
            (self.motion_buffer.num_trajectories,), params.get("init_phase", 0.9), device=env.device
        )
        self.success_rate = 0.0
        self.injected = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)  # episodes that were moved to the reference (runner option)
        self.drifted = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)  # episodes that left the reference by more than the strict link threshold (runner option)
        self.return_sampling = False
        self.bins = False

    def enable_return_sampling(self, gamma: float, max_return: float) -> None:
        """Return-based clip sampling (SuperMimic App. D.1; runner option ``clip_sampling="return"``): from now on the clip of a reset env is drawn with
        p_c = 0.5 u_c + 0.4 softmax_c(-25 G_c / G_max) + 0.1 h_c, written into the motion weights the motion buffer samples from. ``max_return``: G_max, the
        largest attainable discounted return; u_c is proportional to the clip duration (uniform over the time of the library), h_c is the EMA (0.999 per
        reset call) of the past softmax weights, G_c the running estimate (rate 0.2) of the discounted return of the episodes of clip c."""
        sequences = self.motion_buffer._all_motion_sequences
        length_s = (sequences.buffer_length / sequences.framerate).to(self._env.device).float()
        self.return_gamma, self.max_return = gamma, max_return
        self.clip_u = length_s / length_s.sum()
        self.clip_G = torch.zeros(len(self.phase), device=self._env.device)
        self.clip_h = torch.full((len(self.phase),), 1.0 / len(self.phase), device=self._env.device)
        self.clip_p = self.clip_u.clone()
        self.return_sampling = True

    def enable_bin_sampling(self, gamma: float, max_return: float, v2: bool = False, difficulty_k: float = 0.0) -> None:
        """SuperMimic's own reset method (App. D.1; runner option ``reset_sampling="bins"``) instead of the CSI start phases: every clip is cut into bins of
        1 s (``bin_frames`` frames), each bin b keeps a running return estimate G_b (``update_bin_returns``, attributed to the START bin of the episode) and a reset
        env draws a bin with p_b = 0.5 / B + 0.4 softmax_b(-25 G_b / G_max) + 0.1 w_b / sum(w) (w: EMA 0.999 of the past softmax weights) and starts at a frame
        uniform in [bin start, min(bin start + bin_frames, last frame)). The clip goes through the motion buffer's own sampling (terrain origin, tile): its weights
        are the clip marginals of p_b and ``set_start_phase`` then draws the bin of the chosen clip with p_b restricted to it, which is exactly the joint p_b.
        ``v2`` (runner option ``bins_v2``): the bins cover the valid start frames 0 .. T-2 and a tail bin of less than ``bin_frames / 2`` frames is merged into the previous one."""
        sequences = self.motion_buffer._all_motion_sequences
        fps, length = sequences.framerate.to(self._env.device).float(), sequences.buffer_length.to(self._env.device)
        assert bool((fps == fps[0]).all()), "the bins are 1 s of frames: one frame rate for all clips"
        self.bin_frames, self.clip_frames, self.clip_fps = int(round(float(fps[0]))), length, fps
        self.bins_v2 = v2
        self.difficulty_k = difficulty_k
        self.clip_fail = torch.full((len(length),), 0.5, device=self._env.device)  # running failure rate of the episodes of every clip (non-time-out terminations)
        if v2:
            starts = (length - 1).clamp(min=1)  # valid start frames 0 .. T-2
            full, tail = starts // self.bin_frames, starts % self.bin_frames
            merge = (tail < self.bin_frames // 2) & (full > 0)
            num_bins = torch.where(tail == 0, full, torch.where(merge, full, full + 1))
        else:
            num_bins = torch.ceil(length / self.bin_frames).long()
        self.bin_clip = torch.repeat_interleave(torch.arange(len(length), device=self._env.device), num_bins)
        self.bin_local = torch.arange(len(self.bin_clip), device=self._env.device) - (torch.cumsum(num_bins, 0) - num_bins)[self.bin_clip]
        self.bin_first = self.bin_local * self.bin_frames  # first start frame of the bin and the number of start frames it covers
        if v2:
            last_bin = self.bin_local == (num_bins[self.bin_clip] - 1)
            self.bin_size = torch.where(last_bin, starts[self.bin_clip] - self.bin_first, torch.full_like(self.bin_first, self.bin_frames))
        else:
            self.bin_size = (torch.minimum(self.bin_first + self.bin_frames, length[self.bin_clip] - 1) - self.bin_first).clamp(min=0)
        self.max_return = max_return
        self.bin_G = torch.zeros(len(self.bin_clip), device=self._env.device)
        self.bin_w_avg = torch.full_like(self.bin_G, 1.0 / len(self.bin_G))
        self.bin_p = torch.full_like(self.bin_G, 1.0 / len(self.bin_G))
        self.start_bin = torch.full((self._env.num_envs,), -1, dtype=torch.long, device=self._env.device)  # start bin of the running episode (-1: not drawn by the bins)
        self.bins = True

    def update_clip_failures(self, motion_ids: torch.Tensor, failed: torch.Tensor, rate: float = 0.02) -> None:
        """f_c <- f_c + rate (mean(failed of the episodes of c that end now) - f_c): the running failure rate (non-time-out terminations) behind ``difficulty_k``."""
        sums = torch.zeros_like(self.clip_fail).index_add_(0, motion_ids, failed)
        counts = torch.zeros_like(self.clip_fail).index_add_(0, motion_ids, torch.ones_like(failed))
        updated = counts > 0
        self.clip_fail[updated] += rate * (sums[updated] / counts[updated] - self.clip_fail[updated])

    def update_bin_returns(self, bins: torch.Tensor, returns: torch.Tensor, old_weight: float = 0.8) -> None:
        """``update_bin_returns`` of the reference implementation: G_b <- 0.8 G_b + 0.2 mean(returns of the episodes that started in b and end now)."""
        sums = torch.zeros_like(self.bin_G).index_add_(0, bins, returns)
        counts = torch.zeros_like(self.bin_G).index_add_(0, bins, torch.ones_like(returns))
        updated = counts > 0
        self.bin_G[updated] = old_weight * self.bin_G[updated] + (1 - old_weight) * sums[updated] / counts[updated]

    def update_bin_probabilities(self) -> None:
        """One ``sample_frames`` of the reference implementation (once per reset call): p_b of the next resets and the clip weights of the motion buffer."""
        soft = torch.softmax(-25.0 * self.bin_G / self.max_return, 0)
        self.bin_w_avg.mul_(0.999).add_(0.001 * soft)
        difficulty = (1.0 + self.difficulty_k * self.clip_fail)[self.bin_clip]  # weight of the uniform half: 1 for every bin without difficulty_k
        self.bin_p = 0.5 * difficulty / difficulty.sum() + 0.4 * soft + 0.1 * self.bin_w_avg / self.bin_w_avg.sum()
        self.motion_buffer._motion_weights[:] = torch.zeros(len(self.phase), device=self._env.device).index_add_(0, self.bin_clip, self.bin_p)

    def update_clip_returns(self, motion_ids: torch.Tensor, returns: torch.Tensor, old_weight: float = 0.8) -> None:
        """``update_bin_returns`` of the reference implementation per clip: G_c <- 0.8 G_c + 0.2 mean(returns of the episodes of c that end now)."""
        sums = torch.zeros_like(self.clip_G).index_add_(0, motion_ids, returns)
        counts = torch.zeros_like(self.clip_G).index_add_(0, motion_ids, torch.ones_like(returns))
        updated = counts > 0
        self.clip_G[updated] = old_weight * self.clip_G[updated] + (1 - old_weight) * sums[updated] / counts[updated]

    def update_clip_probabilities(self) -> None:
        """One ``sample_frames`` of the reference implementation: the clip probabilities p_c of the next resets (called once per reset call)."""
        soft = torch.softmax(-25.0 * self.clip_G / self.max_return, 0)
        self.clip_h.mul_(0.999).add_(0.001 * soft)
        self.clip_p = 0.5 * self.clip_u + 0.4 * soft + 0.1 * self.clip_h / self.clip_h.sum()
        self.motion_buffer._motion_weights[:] = self.clip_p

    def register_injection(self, env_ids: torch.Tensor) -> None:
        """The runner moved these envs to the reference state (reference injection): each counts as ONE failure of its clip now (averaged per clip like the
        episode ends), and the episode is not evaluated again when it ends."""
        new = env_ids[~self.injected[env_ids]]  # an episode counts at most ONE failure, however often it is moved to the reference
        if self.mode == "csi" and len(new) > 0:
            motion_ids = self.motion_buffer.assigned_env_motion_selection[new]
            total = torch.zeros(len(self.phase), device=self._env.device).index_add_(0, motion_ids, torch.full((len(new),), self.step, device=self._env.device))
            count = torch.zeros(len(self.phase), device=self._env.device).index_add_(0, motion_ids, torch.ones(len(new), device=self._env.device))
            self.phase = (self.phase + total / count.clamp(min=1.0)).clamp(0.0, self.max_phase)
        self.injected[env_ids] = True

    def set_start_phase(self, env_ids: torch.Tensor) -> None:
        """Write the clip start time of the envs that start a new episode (after the clip has been sampled)."""
        motion_ids = self.motion_buffer.assigned_env_motion_selection[env_ids]
        if self.bins:
            bins = torch.multinomial(self.bin_p.unsqueeze(0) * (self.bin_clip.unsqueeze(0) == motion_ids.unsqueeze(1)), 1).squeeze(-1)
            self.start_bin[env_ids] = bins
            frame = self.bin_first[bins] + (torch.rand(len(env_ids), device=self._env.device) * self.bin_size[bins].float()).long()  # (at least one frame before the clip end)
            assigned_ids = self.motion_buffer.env_ids_to_assigned_ids(env_ids)
            self.motion_buffer.motion_buffer_start_time_s[assigned_ids] = (frame.float() + 0.01) / self.clip_fps[motion_ids]  # (+0.01: the float time floors to this frame)
            return
        if self.mode == "csi":
            start_phase = self.phase[motion_ids]
        elif self.mode == "uniform":
            start_phase = self.uniform_max_phase * torch.rand(len(env_ids), device=self._env.device)
        else:
            start_phase = torch.zeros(len(env_ids), device=self._env.device)
        length_s = self.motion_reference.complete_motion_lengths[env_ids]
        assigned_ids = self.motion_buffer.env_ids_to_assigned_ids(env_ids)
        self.motion_buffer.motion_buffer_start_time_s[assigned_ids] = start_phase * length_s

    def __call__(self, env: ManagerBasedRlEnv, env_ids: torch.Tensor, **params) -> dict[str, float]:
        """Update phi_m with the episodes that end now (called before the reset of ``env_ids``)."""
        del params  # the term parameters are read at construction
        # episodes that actually ran (the first reset has none)
        ended = env_ids[env.episode_length_buf[env_ids] > 0]
        ended = ended[~self.injected[ended]]  # moved to the reference during the episode: already counted as a failure
        self.injected[env_ids] = False
        drifted = self.drifted[ended]  # strict success (runner option ``csi_strict_link_eps``): the episode never left the reference by more than the threshold
        self.drifted[env_ids] = False
        if self.return_sampling and len(env_ids) > 0:
            self.update_clip_probabilities()
        if self.bins and len(env_ids) > 0:
            self.update_bin_probabilities()
        if len(ended) > 0 and not self.bins:  # (no CSI phase update with the bins)
            motion_ids = self.motion_buffer.assigned_env_motion_selection[ended]
            length_s = self.motion_reference.complete_motion_lengths[ended]
            start_s = self.motion_buffer.motion_buffer_start_time_s[self.motion_buffer.env_ids_to_assigned_ids(ended)]
            end_phase = (start_s + env.episode_length_buf[ended] * env.step_dt) / length_s
            success = ((end_phase >= self.success_phase) & ~drifted).float()
            self.success_rate = success.mean().item()
            if self.mode == "csi":
                # ACP-unspecified: the +-step updates of the episodes of a motion that finish in the same env step are AVERAGED, so a motion
                # moves by at most one ``step`` per control step. Summing them (one update per finished episode) makes the phases oscillate
                # between the bounds when many parallel envs run the same motion (45 envs per clip in the trial run).
                delta = torch.where(success > 0.5, -self.step, self.step)
                total = torch.zeros(len(self.phase), device=env.device).index_add_(0, motion_ids, delta)
                count = torch.zeros(len(self.phase), device=env.device).index_add_(0, motion_ids, torch.ones_like(delta))
                total = total / count.clamp(min=1.0)
                self.phase = (self.phase + total).clamp(0.0, self.max_phase)
        return {
            "phase_mean": self.phase.mean().item(),
            "phase_min": self.phase.min().item(),
            "phase_max": self.phase.max().item(),
            "success_rate": self.success_rate,
        }
