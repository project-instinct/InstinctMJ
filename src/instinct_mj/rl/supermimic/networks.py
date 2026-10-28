"""World model, latent decoder, failure head and critic (paper Table A.4). The policy is the task's own EncoderActorCritic
actor.

Adapted from the SuperMimic reference implementation (MIT License), Chong Zhang / ETH Zurich.
Task-specific changes: the world model also takes the local terrain height scan, predicts per-body contact logits (used for
the ``undesired_contacts`` reward) and, optionally (``z_dim > 0``), carries the visual latent z of the policy's depth
encoder as part of its state, so the depth policy can be trained in imagination without rendering images. Without z
(``z_dim = 0``, the default) the state is the robot state only and the depth images are rendered (``render.py``).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .mdp import INV_DIM, NUM_J, invariant_features
from .rotations import mm, mv, so3_exp, so3_log, yaw_mat


def mlp(inp: int, hidden: list[int], out: int, act=nn.ELU, layer_norm: bool = False) -> nn.Sequential:
    layers, d = [], inp
    for h in hidden:
        layers.append(nn.Linear(d, h))
        if layer_norm:
            layers.append(nn.LayerNorm(h))
        layers.append(act())
        d = h
    layers.append(nn.Linear(d, out))
    return nn.Sequential(*layers)


class RunningNorm(nn.Module):
    """Running mean/std normalizer; updated from data without gradients, frozen after ``max_count``."""

    def __init__(self, dim: int, max_count: float = 2e8, eps: float = 1e-2):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.zeros(()))
        self.max_count, self.eps = max_count, eps
        self._host_count = 0.0  # host-side mirror of ``count`` (avoids a device sync per update)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
        self._host_count = float(self.count)

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        x = x.reshape(-1, x.shape[-1]).float()
        if self._host_count >= self.max_count:
            return
        n = x.shape[0]
        self._host_count += n
        m, v = x.mean(0), x.var(0, unbiased=False)
        tot = self.count + n
        delta = m - self.mean
        self.mean += delta * n / tot
        self.var = (self.var * self.count + v * n + delta**2 * self.count * n / tot) / tot
        self.count = tot

    def forward(self, x):
        return (x - self.mean) / torch.sqrt(self.var + self.eps**2)

    def std(self):
        return torch.sqrt(self.var + self.eps**2)


WM_OUT = 3 + 3 + 3 + 3 + NUM_J + NUM_J

# bf16 autocast for MLP matmuls only (state integration, FK, rewards stay fp32)
AMP_DTYPE: torch.dtype | None = None


def run_mlp(net: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if AMP_DTYPE is None:
        return net(x)
    with torch.autocast("cuda", dtype=AMP_DTYPE):
        return net(x).float()


class WorldModel(nn.Module):
    """Deterministic dynamics (s_{t+1}, z_{t+1}) = f(s_t, z_t, a_t, xi, terrain scan): one MLP predicts the frame-local
    state increments, the logits of the per-body contact indicators at s_{t+1} and the increment of the visual latent z
    (the latent of the policy's depth encoder). z is normalized with running statistics kept here (``update_z_stats``).
    With ``z_dim = 0`` there is no latent: ``step`` takes ``z=None`` and returns ``None`` in its place."""

    def __init__(self, scan_dim: int, num_contact: int, xi_dim: int, z_dim: int = 0, hidden=(2048, 2048, 2048)):
        super().__init__()
        self.in_dim = INV_DIM + NUM_J + xi_dim + scan_dim
        self.num_contact, self.z_dim = num_contact, z_dim
        self.net = mlp(self.in_dim + z_dim, list(hidden), WM_OUT + num_contact + z_dim, act=nn.SiLU, layer_norm=True)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        with torch.no_grad():
            self.net[-1].bias[WM_OUT : WM_OUT + num_contact] = -3.0  # contacts are rare
        self.in_norm = RunningNorm(self.in_dim)
        self.out_norm = RunningNorm(WM_OUT, eps=1e-4)
        self.register_buffer("z_mean", torch.zeros(z_dim))
        self.register_buffer("z_std", torch.ones(z_dim))
        self.register_buffer("dz_std", torch.ones(z_dim))

    @torch.no_grad()
    def update_z_stats(self, z: torch.Tensor, dz: torch.Tensor, w: torch.Tensor, rate: float = 0.01):
        """EMA of the latent mean / std and of the std of its one-step increments (``w``: validity weights of ``dz``)."""
        z, dz, w = z.reshape(-1, self.z_dim), dz.reshape(-1, self.z_dim), w.reshape(-1, 1)
        self.z_mean.lerp_(z.mean(0), rate)
        self.z_std.lerp_(z.std(0).clamp(min=1e-3), rate)
        self.dz_std.lerp_(((dz**2 * w).sum(0) / w.sum().clamp(min=1)).sqrt().clamp(min=1e-3), rate)

    def inputs(self, S, a, xi_n, scan):
        return torch.cat((invariant_features(S), a, xi_n, scan), -1)

    @staticmethod
    def targets(S, S1):
        """Frame-local increments from S to S1 (the regression targets)."""
        p, R, v, w, q, qd = S
        p1, R1, v1, w1, q1, qd1 = S1
        H = yaw_mat(R)
        Rt = R.transpose(-1, -2)
        return torch.cat(
            (
                mv(H.transpose(-1, -2), p1 - p),
                so3_log(mm(Rt, R1)),
                mv(Rt, v1 - v),
                mv(Rt, w1 - w),
                q1 - q,
                qd1 - qd,
            ),
            -1,
        )

    def step(self, S, a, xi_n, scan, z=None):
        """Returns the next state tuple, the contact logits at the next state and the next visual latent (None if
        ``z_dim = 0``)."""
        p, R, v, w, q, qd = S
        x = self.in_norm(self.inputs(S, a, xi_n, scan))
        y = run_mlp(self.net, x if self.z_dim == 0 else torch.cat((x, (z - self.z_mean) / self.z_std), -1))
        d = y[..., :WM_OUT] * self.out_norm.std() + self.out_norm.mean
        H = yaw_mat(R)
        dp, dth, dv, dw, dq, dqd = d.split((3, 3, 3, 3, NUM_J, NUM_J), -1)
        p1 = p + mv(H, dp)
        R1 = mm(R, so3_exp(dth))
        v1 = v + mv(R, dv)
        w1 = w + mv(R, dw)
        z1 = None if self.z_dim == 0 else z + y[..., WM_OUT + self.num_contact :] * self.dz_std
        return (p1, R1, v1, w1, q + dq, qd + dqd), y[..., WM_OUT : WM_OUT + self.num_contact], z1

    @staticmethod
    def state_error(S_hat, S):
        """Frame-local state error components (same layout as ``targets``)."""
        p, R, v, w, q, qd = S
        ph, Rh, vh, wh, qh, qdh = S_hat
        H = yaw_mat(R)
        Rt = R.transpose(-1, -2)
        return torch.cat(
            (
                mv(H.transpose(-1, -2), ph - p),
                so3_log(mm(Rt, Rh)),
                mv(Rt, vh - v),
                mv(Rt, wh - w),
                qh - q,
                qdh - qd,
            ),
            -1,
        )


class LatentDecoder(nn.Module):
    """Training-only decoder of the visual latent to the CLEAN processed depth image (``channels`` x 18 x 32, e.g. a stack
    of frames): the reconstruction loss is the only training signal of the policy's depth encoder (no collapse of the
    latent prediction loss). Never used in imagination."""

    def __init__(self, latent_dim: int, channels: int = 1):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(latent_dim, 32 * 5 * 8), nn.ELU())
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(32, 32, 3, 2, 1, output_padding=(0, 1)), nn.ELU(),  # -> (9, 16)
            nn.ConvTranspose2d(32, 16, 3, 2, 1, output_padding=(1, 1)), nn.ELU(),  # -> (18, 32)
            nn.Conv2d(16, channels, 3, 1, 1),
        )  # fmt: skip

    def forward(self, z):
        with torch.autocast("cuda", dtype=AMP_DTYPE or torch.float32, enabled=AMP_DTYPE is not None):
            return torch.sigmoid(self.dec(self.fc(z).view(-1, 32, 5, 8))).float()


class FailureHead(nn.Module):
    """Dreamer-style "continue" predictor: P(the transition into this state is a REAL termination), from the privileged critic
    features of the (true or imagined) state (invariant state, robot-vs-reference errors, xi, goal, terrain scan)."""

    def __init__(self, in_dim: int):
        super().__init__()
        self.norm = RunningNorm(in_dim)
        self.net = mlp(in_dim, [256, 128], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Logits (B,)."""
        return run_mlp(self.net, self.norm(x)).squeeze(-1)


class Critic(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = mlp(in_dim, [256, 128], 1)
        self.norm = RunningNorm(in_dim)

    def forward(self, x):
        return run_mlp(self.net, self.norm(x)).squeeze(-1)
