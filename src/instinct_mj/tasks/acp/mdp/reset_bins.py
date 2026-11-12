"""SuperMimic's episode start sampling (App. D.1) of the ACP tracking task: the reset bins."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from instinct_mj.motion_reference.motion_files.terrain_motion import TerrainMotion


class ResetBins:
    """SuperMimic's reset method (App. D.1): every clip is cut into bins of 1 s (``bin_frames`` frames), each bin b keeps a running return estimate G_b
    (``update_bin_returns``, attributed to the START bin of the episode) and a reset env draws a bin with
    p_b = 0.5 / B + 0.4 softmax_b(-25 G_b / G_max) + 0.1 w_b / sum(w) (w: EMA 0.999 of the past softmax weights) and starts at a frame
    uniform in [bin start, min(bin start + bin_frames, last frame)). The clip goes through the motion buffer's own sampling (terrain origin, tile): its weights
    are the clip marginals of p_b and ``set_start_phase`` then draws the bin of the chosen clip with p_b restricted to it, which is exactly the joint p_b.

    ``v2``: the bins cover the valid start frames 0 .. T-2 and a tail bin of less than ``bin_frames / 2`` frames is merged into the previous one (:class:`ReturnBins` uses it).
    ``difficulty_k``: the uniform half is weighted by 1 + k f_c, f_c being the running failure rate of the clip (``update_clip_failures``).
    The SuperMimic ACP runner uses only the subclass :class:`ReturnBins`; this class is its base (bin layout, start-frame draw).
    """

    def __init__(self, motion_buffer: TerrainMotion, num_envs: int, device, max_return: float, v2: bool = False, difficulty_k: float = 0.0):
        self.motion_buffer, self.device = motion_buffer, device
        sequences = motion_buffer._all_motion_sequences
        fps, length = sequences.framerate.to(device).float(), sequences.buffer_length.to(device)
        assert bool((fps == fps[0]).all()), "the bins are 1 s of frames: one frame rate for all clips"
        self.num_clips = len(length)
        self.bin_frames, self.clip_frames, self.clip_fps = int(round(float(fps[0]))), length, fps
        self.v2, self.difficulty_k, self.max_return = v2, difficulty_k, max_return
        self.clip_fail = torch.full((self.num_clips,), 0.5, device=device)  # running failure rate of the episodes of every clip (non-time-out terminations)
        if v2:
            starts = (length - 1).clamp(min=1)  # valid start frames 0 .. T-2
            full, tail = starts // self.bin_frames, starts % self.bin_frames
            merge = (tail < self.bin_frames // 2) & (full > 0)
            num_bins = torch.where(tail == 0, full, torch.where(merge, full, full + 1))
        else:
            num_bins = torch.ceil(length / self.bin_frames).long()
        self.bin_clip = torch.repeat_interleave(torch.arange(self.num_clips, device=device), num_bins)
        self.bin_local = torch.arange(len(self.bin_clip), device=device) - (torch.cumsum(num_bins, 0) - num_bins)[self.bin_clip]
        self.bin_first = self.bin_local * self.bin_frames  # first start frame of the bin and the number of start frames it covers
        if v2:
            last_bin = self.bin_local == (num_bins[self.bin_clip] - 1)
            self.bin_size = torch.where(last_bin, starts[self.bin_clip] - self.bin_first, torch.full_like(self.bin_first, self.bin_frames))
        else:
            self.bin_size = (torch.minimum(self.bin_first + self.bin_frames, length[self.bin_clip] - 1) - self.bin_first).clamp(min=0)
        self.bin_G = torch.zeros(len(self.bin_clip), device=device)
        self.bin_w_avg = torch.full_like(self.bin_G, 1.0 / len(self.bin_G))
        self.bin_p = torch.full_like(self.bin_G, 1.0 / len(self.bin_G))
        self.start_bin = torch.full((num_envs,), -1, dtype=torch.long, device=device)  # start bin of the running episode (-1: not drawn by the bins)

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

    def update_probabilities(self) -> None:
        """One ``sample_frames`` of the reference implementation (once per reset call): p_b of the next resets and the clip weights of the motion buffer."""
        soft = torch.softmax(-25.0 * self.bin_G / self.max_return, 0)
        self.bin_w_avg.mul_(0.999).add_(0.001 * soft)
        difficulty = (1.0 + self.difficulty_k * self.clip_fail)[self.bin_clip]  # weight of the uniform half: 1 for every bin without difficulty_k
        self.bin_p = 0.5 * difficulty / difficulty.sum() + 0.4 * soft + 0.1 * self.bin_w_avg / self.bin_w_avg.sum()
        self.motion_buffer._motion_weights[:] = torch.zeros(self.num_clips, device=self.device).index_add_(0, self.bin_clip, self.bin_p)

    def set_start_phase(self, env_ids: torch.Tensor) -> None:
        """Draw the start bin of the envs that start a new episode (after the clip has been sampled) and write the clip start time."""
        motion_ids = self.motion_buffer.assigned_env_motion_selection[env_ids]
        bins = torch.multinomial(self.bin_p.unsqueeze(0) * (self.bin_clip.unsqueeze(0) == motion_ids.unsqueeze(1)), 1).squeeze(-1)
        self.start_bin[env_ids] = bins
        frame = self.bin_first[bins] + (torch.rand(len(env_ids), device=self.device) * self.bin_size[bins].float()).long()  # (at least one frame before the clip end)
        assigned_ids = self.motion_buffer.env_ids_to_assigned_ids(env_ids)
        self.motion_buffer.motion_buffer_start_time_s[assigned_ids] = (frame.float() + 0.01) / self.clip_fps[motion_ids]  # (+0.01: the float time floors to this frame)


class ReturnBins(ResetBins):
    """The official HybridRobotics/supermimic ``ReturnSampler`` (runners/reset_sampling.py) on the 1 s bins of :class:`ResetBins` (``v2`` layout: every bin has >= 1 start
    frame). Every ``horizon`` env steps the buffered per-step rewards turn into discounted returns (lambda 1, backwards), which continue where an episode ends with the value
    of the state after the step: 0 after a fall, the mean V after the clip end, V of the next frame's bin after a time-out or at the buffer end. Each env's visits to one bin
    within one episode give one sample (their mean return); V_b is the exact mean of a bin's first ``memory`` samples, older samples then count as ``memory``. Reset distribution:
    explore / B + (1 - explore - keep) softmax(-steepness V / v_max) + keep effort / sum(effort), effort = the summed square of each bin's focus. Unmeasured bins read v_max."""

    def __init__(self, motion_buffer, num_envs: int, device, max_return: float, gamma: float = 0.99, horizon: int = 20, memory: int = 256, steepness: float = 25.0,
                 explore: float = 0.5, keep: float = 0.1):
        super().__init__(motion_buffer, num_envs, device, max_return, v2=True, difficulty_k=0.0)
        self.gamma = gamma
        assert explore + keep < 1.0, "explore + keep must leave room for the focus"
        self.horizon, self.memory, self.steepness, self.explore, self.keep = horizon, memory, steepness, explore, keep
        nb = len(self.bin_clip)
        self.num_bins_clip = torch.bincount(self.bin_clip, minlength=self.num_clips)
        self.bin_offset = torch.cumsum(self.num_bins_clip, 0) - self.num_bins_clip
        self.value = torch.full((nb,), float(max_return), device=device)
        self.bin_G = self.value  # the same tensor: the logs and the sidecar (by clip and bin index) read / restore V_b as G_b
        self.count = torch.zeros(nb, device=device)
        self.effort = torch.zeros(nb, dtype=torch.float64, device=device)
        shape = (num_envs, horizon)
        self._bins_buf = torch.zeros(shape, dtype=torch.long, device=device)
        self._next_bins = torch.zeros(shape, dtype=torch.long, device=device)
        self._rewards = torch.zeros(shape, device=device)
        self._terminated = torch.zeros(shape, dtype=torch.bool, device=device)
        self._clip_end = torch.zeros(shape, dtype=torch.bool, device=device)
        self._timed_out = torch.zeros(shape, dtype=torch.bool, device=device)
        self._step = 0

    def bin_of(self, clip: torch.Tensor, local_frame: torch.Tensor) -> torch.Tensor:
        """Bin of a clip-local frame (frames past the last bin's range count to the last bin)."""
        b = torch.minimum(local_frame.clamp(min=0) // self.bin_frames, self.num_bins_clip[clip] - 1)
        return self.bin_offset[clip] + b

    def record(self, clip: torch.Tensor, local_frame: torch.Tensor, next_local_frame: torch.Tensor, reward: torch.Tensor, terminated: torch.Tensor, clip_end: torch.Tensor,
               timed_out: torch.Tensor) -> None:
        """Buffer one env step (the frame before the step, the reward, how the episode ended, if it did); every ``horizon`` steps update V and the reset distribution."""
        col = self._step
        self._bins_buf[:, col] = self.bin_of(clip, local_frame)
        self._next_bins[:, col] = self.bin_of(clip, next_local_frame)
        self._rewards[:, col] = torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)
        self._terminated[:, col], self._clip_end[:, col], self._timed_out[:, col] = terminated, clip_end, timed_out
        self._step += 1
        if self._step < self.horizon:
            return
        self._step = 0
        self._update()
        self._distribution()

    def _dones(self) -> torch.Tensor:
        return self._terminated | self._clip_end | self._timed_out

    def _returns(self, gamma: float) -> torch.Tensor:
        next_value = torch.where(self._terminated, 0.0, torch.where(self._clip_end, self.value.mean(), self.value[self._next_bins]))
        dones = self._dones()
        returns = torch.empty_like(self._rewards)
        next_return = next_value[:, -1]
        for t in reversed(range(self.horizon)):
            next_return = torch.where(dones[:, t], next_value[:, t], next_return)
            returns[:, t] = self._rewards[:, t] + gamma * next_return
            next_return = returns[:, t]
        return returns

    def _update(self) -> None:
        returns = self._returns(self.gamma)
        dones = self._dones()
        num_envs, horizon = dones.shape
        nb = len(self.bin_clip)
        episode = torch.cumsum(dones.long(), dim=1) - dones.long()
        envs = torch.arange(num_envs, device=dones.device).unsqueeze(1)
        key = (envs * horizon + episode) * nb + self._bins_buf
        keys, inverse, visits = torch.unique(key.reshape(-1), return_inverse=True, return_counts=True)
        samples = torch.zeros(keys.numel(), device=returns.device).scatter_add_(0, inverse, returns.reshape(-1)) / visits
        bins = keys % nb
        new = torch.zeros_like(self.count).scatter_add_(0, bins, torch.ones_like(samples))
        total = torch.zeros_like(self.value).scatter_add_(0, bins, samples)
        self.count = self.count.clamp(max=self.memory) + new
        self.value.copy_(torch.where(new > 0, self.value + (total - new * self.value) / self.count.clamp(min=1), self.value))

    def _distribution(self) -> None:
        focus = torch.softmax(-self.steepness * self.value / self.max_return, dim=0)
        self.effort += focus.double().square()
        keep = (self.effort / self.effort.sum()).float()
        self.bin_p = self.explore / len(self.bin_clip) + (1.0 - self.explore - self.keep) * focus + self.keep * keep

    def update_probabilities(self) -> None:
        """Clip weights of the motion buffer from the current bin distribution (updated by ``record``)."""
        self.motion_buffer._motion_weights[:] = torch.zeros(self.num_clips, device=self.device).index_add_(0, self.bin_clip, self.bin_p)
