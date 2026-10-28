"""Runner config of the SuperMimic-style trainer (hyperparameters follow Table A.5 of the paper / reference code)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(kw_only=True)
class SuperMimicRunnerCfg:
    """Hyperparameters of :class:`SuperMimicRunner`, mixed into the task's agent cfg (the runner receives its flat ``to_dict()``).
    ``seed``, ``max_iterations``, ``save_interval`` and ``log_interval`` are the ones of ``InstinctRlOnPolicyRunnerCfg``."""

    # ---- simulation / collection (Algorithm A.1: collect -> k-step world-model fit -> H-step BPTT policy + GAE critic) ----
    buffer_steps: int = 1000
    """Replay buffer length in vectorized env steps, (T, N) layout (default 1000)."""
    utd: float = 0.4
    """Updates per vectorized simulator step (default 0.4)."""
    warmup_steps: int = 64
    """Simulator steps collected before any update (default 64)."""
    goal_frames: int = 10
    """Number of future reference frames in the goal; task: ``MotionReferenceManagerCfg.num_frames`` (default 10)."""
    hist: int = 8
    """Proprio observation history length; task: ``PROPRIO_HISTORY_LENGTH`` (default 8)."""

    # ---- world model ----
    wm_window: int = 8
    """Window length k of the multi-step world-model fit (default 8)."""
    wm_batches: int = 10
    """World-model gradient steps per update (default 10)."""
    wm_batch: int = 2048
    """Windows per world-model batch (default 2048)."""
    wm_failure_frac: float = 0.25
    """Fraction of each world-model batch drawn from windows that start within ``wm_failure_horizon`` steps before a real
    termination; dynamics, depth and failure heads see them, the failure head importance-weights them (default 0.25)."""
    wm_failure_horizon: int = 16
    """Steps before a real termination in which a window counts as a failure window (default 16)."""
    wm_holdout_envs: float = 0.1
    """Fraction of envs never used for world-model / head training, only for held-out calibration diagnostics (default 0.1)."""
    wm_only_iters: int = 500
    """World-model-only iterations at the start, no policy update (default 500)."""

    # ---- visual latent (part of the world-model state; used by the parkour runner only, perceptive shadowing renders the images) ----
    latent_windows: int = 2048
    """Windows (of ``wm_batch``) on which the k-step latent prediction loss is computed; every window still rolls out with
    its latent (the dynamics take it as input), only the target images of these are encoded (default 2048)."""
    latent_weight: float = 1.0
    """Weight of the k-step latent prediction loss (in units of the latent variance) relative to the dynamics loss (default 1.0)."""
    latent_recon_images: int = 512
    """Images per world-model batch for the clean-image reconstruction loss that trains the policy's depth encoder (default 512)."""
    latent_recon_weight: float = 1.0
    """Weight of the reconstruction (L1) loss (default 1.0)."""

    # ---- failure head ----
    fail_head: bool = True
    """Use the learned failure ("continue") head in imagination, ``d = 1 - (1 - d_analytic) (1 - p_fail)``; the head is trained
    either way and the gradient through ``p_fail`` is kept (default True)."""
    fail_head_warmup_iters: int = 300
    """Iterations after the head was created before it affects the policy; it is trained from the start (default 300)."""
    fail_windows: int = 512
    """Windows (of ``wm_batch``) that train the head, the feature computation runs the FK (default 512)."""
    fail_clip: float = 0.95
    """Upper clip of ``p_fail`` in imagination (default 0.95)."""
    fail_pos_weight: float = 1.0
    """BCE positive weight; 1 because the failure oversampling is importance-weighted to keep the head calibrated (default 1.0)."""
    resume_init_new_heads: bool = False
    """Allow resuming from a sidecar written before the failure head existed: the head is created fresh (default False)."""

    # ---- policy / critic ----
    horizon: int = 32
    """Imagination horizon H of the BPTT policy update (default 32)."""
    pi_batch: int = 1024
    """Imagination starts per policy update (default 1024)."""
    gamma: float = 0.99
    """Discount factor (default 0.99)."""
    lam: float = 0.95
    """GAE lambda of the critic targets (default 0.95)."""
    action_noise: float = 0.1
    """Std of the Gaussian exploration noise added to the actions during real-env collection (default 0.1)."""
    critic_epochs: int = 5
    """Critic fitting epochs per policy update (default 5)."""
    critic_minibatches: int = 4
    """Critic minibatches per epoch (default 4)."""
    no_critic: bool = False
    """Ablation: no critic, i.e. no bootstrap value term in the return and no critic fit (default False)."""

    # ---- schedule ----
    lr: float = 2e-4
    """Initial learning rate (default 2e-4)."""
    lr_decay_iters: int = 250_000
    """Iterations of the cosine decay from ``lr`` to ``lr * lr_final_ratio``, constant afterwards (default 250000)."""
    lr_final_ratio: float = 0.03
    """Final learning rate as a fraction of ``lr`` (default 0.03)."""

    # ---- performance ----
    amp: str = "bf16"
    """Autocast dtype of the MLP matmuls, "bf16" or "none" (default "bf16")."""
    compile: bool = True
    """Use ``torch.compile`` on the imagination / world-model functions (default True)."""
    compile_policy: bool = True
    """``torch.compile`` the imagination step, the reward / termination and the critic-feature functions, i.e. the
    level-batched FK (default True)."""
    policy_autocast: bool = False
    """bf16 autocast of the actor MLP in imagination only; collection / play stay fp32 (default False)."""
    checkpoint_imagination: bool = False
    """Gradient checkpointing of every imagination step (recompute in backward): less memory, more compute (default False)."""
    render_robot: bool = True
    """Imagined depth images include the robot's own links (self-occlusion, as the env camera); False = terrain only, which is
    slightly cheaper per imagination step (default True)."""
    heightmap_resolution: float = 0.05
    """Resolution [m] of the global terrain heightmap used by the analytic height scan (default 0.05)."""
    heightmap_cache_dir: str = "~/.cache/instinct_mj/supermimic_heightmaps"
    """Directory caching the terrain heightmaps (default "~/.cache/instinct_mj/supermimic_heightmaps")."""

    def to_dict(self) -> dict[str, Any]:
        """Plain dict of all fields (what the runner receives)."""
        return asdict(self)
