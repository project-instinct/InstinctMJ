"""Episode start of the ACP tracking task: the ACP paper's Curriculum State Initialization (mode ``csi``), SuperMimic's reset bins (``reset_bins.ResetBins``) or a uniform / fixed start phase."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers import ManagerTermBase

from .reset_bins import ResetBins

CSI_SUCCESS_XY = 0.5  # m, horizontal pelvis error at the episode end below which a CSI episode can count as a success (the evaluation's ``--xy-end``)
_CONTACTS_COMMAND = "acp_reference_contacts"  # (private: the package star-imports this module)

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers import CurriculumTermCfg

    from instinct_mj.motion_reference.motion_files.terrain_motion import TerrainMotion
    from instinct_mj.motion_reference.motion_reference_manager import MotionReferenceManager


class EpisodeStart(ManagerTermBase):
    """Where in its clip an episode starts. ``mode="bins"`` (default): the runner attaches a :class:`ResetBins` (``enable_bins``, SuperMimic App. D.1) which draws the
    clip and the start frame; without it (no runner: the env alone) the start is uniform like ``mode="uniform"``. ``mode="uniform"``: the start phase is
    U(0, ``uniform_max_phase``). ``mode="start"``: the clip starts at phase 0 (evaluation / play / Stage 3), the bins are not used even when attached.

    ``mode="csi"`` (the paper's Stage 1 curriculum, Table S4; the runner switches to it with ``reset_sampling="csi"`` through :meth:`enable_csi`): every clip m has an initialization
    phase phi_m (initially ``init_phase`` = 0.9) and an episode of m starts exactly at phi_m of the clip. When episodes of m end, phi_m -= ``step`` (0.05) if the episode
    reached phase >= ``success_phase`` (0.99) and += ``step`` otherwise, clipped to [0, ``max_phase`` = 0.99]; the updates of the episodes of a clip that end in the same call are
    averaged (ACP-unspecified: with many parallel envs per clip, summing them makes the phases oscillate between the bounds). ACP-unspecified ``success_on_timeout``: the v1
    tasks cap the episodes at 10 s, so an episode that starts earlier than 10 s before the clip end of a long clip can never reach phase 0.99; an episode that ends by the cap
    without a failure termination counts as a success too (otherwise the phases of long clips never go below 1 - 10 s / length). ACP-unspecified: a success also needs the pelvis
    within ``CSI_SUCCESS_XY`` (0.5 m, ``eval_tracker --xy-end``) of the reference pelvis horizontally at the episode end; the v1 terminations have no horizontal term, so without it
    an episode that stays alive but stalls (e.g. in front of a vault box while the reference goes over it) would move the phase back."""

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        params = cfg.params
        self.mode: str = params["mode"]
        if self.mode not in ("bins", "uniform", "start", "csi"):
            raise ValueError(f"episode start mode must be 'bins', 'uniform', 'start' or 'csi', got '{self.mode}'.")
        self.uniform_max_phase: float = params.get("uniform_max_phase", 0.98)
        self.motion_reference: MotionReferenceManager = env.scene[params.get("reference_name", "motion_reference")]
        self.motion_buffer: TerrainMotion = next(iter(self.motion_reference.motion_buffers.values()))
        self.bins: ResetBins | None = None
        self.phase = torch.full((self.motion_buffer.num_trajectories,), 0.9, device=env.device)  # csi: initialization phase of every clip
        self.csi_step, self.csi_success_phase, self.csi_max_phase, self.csi_success_on_timeout = 0.05, 0.99, 0.99, True
        self.success_rate = 0.0
        self.phase_cap: torch.Tensor | None = None  # per-clip upper limit of the CSI phase (None: ``csi_max_phase`` only)

    @property
    def uses_bins(self) -> bool:
        """True when the start of the episodes is drawn by the attached bins."""
        return self.mode == "bins" and self.bins is not None

    def enable_csi(self, init_phase: float = 0.9, step: float = 0.05, success_phase: float = 0.99, max_phase: float = 0.99, success_on_timeout: bool = True) -> None:
        """Switch to the Curriculum State Initialization (runner option ``reset_sampling="csi"``, paper Table S4: initial phase 0.9, completion 0.99, adjustment 0.05, range [0, 0.99])."""
        self.mode = "csi"
        self.phase[:] = init_phase if self.phase_cap is None else torch.minimum(torch.full_like(self.phase, init_phase), self.phase_cap)
        self.csi_step, self.csi_success_phase, self.csi_max_phase, self.csi_success_on_timeout = step, success_phase, max_phase, success_on_timeout

    def set_phase_cap(self, cap: torch.Tensor) -> None:
        """Per-clip upper limit of the CSI phase (the runner passes the last start phase that still leaves the imagination horizon inside the clip)."""
        self.phase_cap = cap.to(self.phase.device)
        self.phase = self.clamp_phase(self.phase)

    def clamp_phase(self, phase: torch.Tensor) -> torch.Tensor:
        phase = phase.clamp(0.0, self.csi_max_phase)
        return phase if self.phase_cap is None else torch.minimum(phase, self.phase_cap)

    def enable_bins(self, max_return: float, v2: bool = False, difficulty_k: float = 0.0) -> ResetBins:
        """Attach the reset bins (runner options ``bins_v2``, ``bins_difficulty_k``); ``max_return``: G_max, the largest attainable discounted return."""
        self.bins = ResetBins(self.motion_buffer, self._env.num_envs, self._env.device, max_return, v2, difficulty_k)
        return self.bins

    def set_start_phase(self, env_ids: torch.Tensor) -> None:
        """Write the clip start time of the envs that start a new episode (after the clip has been sampled)."""
        if self.uses_bins:
            self.bins.set_start_phase(env_ids)
            return
        if self.mode == "start":
            start_phase = torch.zeros(len(env_ids), device=self._env.device)
        elif self.mode == "csi":
            start_phase = self.phase[self.motion_buffer.assigned_env_motion_selection[env_ids]]
        else:
            start_phase = self.uniform_max_phase * torch.rand(len(env_ids), device=self._env.device)
        length_s = self.motion_reference.complete_motion_lengths[env_ids]
        assigned_ids = self.motion_buffer.env_ids_to_assigned_ids(env_ids)
        self.motion_buffer.motion_buffer_start_time_s[assigned_ids] = start_phase * length_s

    def __call__(self, env: ManagerBasedRlEnv, env_ids: torch.Tensor, **params) -> dict[str, float]:
        """Update the bin probabilities once per reset call (called before the reset of ``env_ids``)."""
        del params  # the term parameters are read at construction
        if self.uses_bins and len(env_ids) > 0:
            self.bins.update_probabilities()
        if self.mode == "csi":
            ended = env_ids[env.episode_length_buf[env_ids] > 0]  # episodes that actually ran (the first reset has none)
            if len(ended) > 0:
                motion_ids = self.motion_buffer.assigned_env_motion_selection[ended]
                length_s = self.motion_reference.complete_motion_lengths[ended]
                start_s = self.motion_buffer.motion_buffer_start_time_s[self.motion_buffer.env_ids_to_assigned_ids(ended)]
                end_phase = (start_s + env.episode_length_buf[ended] * env.step_dt) / length_s
                tm = env.termination_manager
                term = lambda name: tm.get_term(name)[ended].bool()  # noqa: E731
                failed = env.reset_terminated[ended].bool()  # a real failure termination (root height / tilt)
                exhausted, capped = term("dataset_exhausted"), term("time_out")  # the clip end / the episode cap (10 s in v1)
                # counted: the real episode ends (failure, clip end, cap); illegal_reset_contact and external / manual resets (no done flag) do not move the phase
                counted = (failed | exhausted | capped) & ~term("illegal_reset_contact")
                contacts = env.command_manager.get_term(_CONTACTS_COMMAND)
                tracked = (contacts.base_pos_w[ended, :2] - contacts.ref_base_pos_w[ended, :2]).norm(dim=-1) < CSI_SUCCESS_XY  # ACP-unspecified: alive AND on the reference
                # success: no failure, on the reference, and the target phase / the clip end (short clips never reach .99 of their last frame) / the cap (``csi_success_on_timeout``)
                success = counted & ~failed & tracked & ((end_phase >= self.csi_success_phase) | exhausted | (capped & self.csi_success_on_timeout))
                if counted.any():
                    self.success_rate = success[counted].float().mean().item()
                    delta = torch.where(success, -self.csi_step, self.csi_step) * counted
                    total = torch.zeros_like(self.phase).index_add_(0, motion_ids, delta)
                    count = torch.zeros_like(self.phase).index_add_(0, motion_ids, counted.float())
                    self.phase = self.clamp_phase(self.phase + total / count.clamp(min=1.0))
        return {}
