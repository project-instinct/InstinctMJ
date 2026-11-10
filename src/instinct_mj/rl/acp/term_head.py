"""Learned termination head of the ACP world model (SuperMimic paper Sec. 3.2: ``d̂_t = d_ω(ŝ_t, g_t)``, fit with BCE against the simulator's termination label on the k-step replay windows).

The head reads the privileged critic features of the (imagined) state: invariant state, pelvis / contact-link errors against the reference (the goal ``g_t`` of a motion-tracking task), reference
flags, the terrain scan and the sim context. It sits next to the 9-link contact logits of the world model but is a separate module with its own optimizer: the dynamics (and the checkpoints of
runs that never had the head) stay untouched. The last layer is zero-initialized with a low-probability bias, so an untrained head predicts a termination probability of ``sigmoid(TERM_BIAS_INIT)`` per step.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from instinct_mj.rl.supermimic.networks import mlp, run_mlp

from .rays import ScanFloorRunningNorm

TERM_BIAS_INIT = -5.0
"""ACP-unspecified (the paper does not give the initialization): sigmoid(-5) = 0.7 % per step, i.e. a survival of 0.80 over the 32 imagined steps before the head has learned anything."""


class TerminationHead(nn.Module):
    """Logit (B,) that the transition into a state is a REAL termination (``reset_terminated``: time-outs, the clip end and the episode cap are not terminations)."""

    def __init__(self, in_dim: int, scan_slice: slice):
        super().__init__()
        self.norm = ScanFloorRunningNorm(in_dim, scan_slice)
        self.net = mlp(in_dim, [256, 128], 1)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, TERM_BIAS_INIT)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return run_mlp(self.net, self.norm(x)).squeeze(-1)
