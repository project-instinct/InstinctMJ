"""Stage 3 hyperparameters of the ACP SuperMimic runner (:class:`~instinct_mj.rl.acp.stage3_runner.ACPStage3Runner`): dual critic (reference
Vm, goal-reaching Vfs), goal-reaching planner replay. Paper: Table S4 (curriculum), Table S5 (replay), Table S6 (Stage-3 critic)."""

from __future__ import annotations

from dataclasses import dataclass

from .config import ACPRunnerCfg


@dataclass(kw_only=True)
class ACPStage3Cfg(ACPRunnerCfg):
    vfs_lr: float = 1e-3
    """Fixed learning rate of the goal-reaching critic Vfs (Table S6: 1e-3, not decayed)."""
    vfs_warmup_iters: int = 500
    """Iterations in which Vfs is trained but the goal-reaching imagination starts are excluded from the actor loss (Table S6: 500)."""
    pi_goal_fraction: float = 0.5
    """ACP-unspecified: share of the ``pi_batch`` imagination starts that are goal-reaching starts (the rest are reference starts)."""
    stumble_lr: float = 1e-3
    """ACP-unspecified: learning rate of the stumbling head (the learned force condition of the Stage 3 stumbling penalty in imagination)."""
    stumble_eval_states: int = 16384
    """Newest buffered transitions of the goal envs the stumbling-head diagnostics are computed on (ACP-unspecified; bounds the cost per log interval)."""
    stumble_eval_chunk: int = 2048
    """Transitions per chunk (forward kinematics, scan and head) of the stumbling-head diagnostics."""
    return_norm_momentum: float = 0.99
    """ACP-unspecified: momentum of the running standard deviation of the discounted return of each regime (reference / goal-reaching), the
    BPTT counterpart of the separate advantage normalization of Table S6."""
    goal_soft_tau_dist: float = 0.05
    """ACP-unspecified: temperature [m] of the sigmoid relaxation of the distance gates of the goal rewards in imagination."""
    goal_soft_tau_speed: float = 0.05
    """ACP-unspecified: temperature [m/s] of the relaxed speed gates."""
    goal_soft_tau_height: float = 0.02
    """ACP-unspecified: temperature [m] of the relaxed height gates."""
    goal_soft_tau_angle: float = 0.05
    """ACP-unspecified: temperature [rad] of the relaxed heading gates."""
    planner_tile_capacity: int = 2000
    """ACP-unspecified: samples per (terrain type, tile) of the goal partition of the planner replay (quality-based eviction above it)."""
