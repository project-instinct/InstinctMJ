"""Attention-based contact planner (ACP paper Fig. 2C, Table S3) and its training loss (Eq. 1-2).

Per body, a FiLM-conditioned query attends over embedded terrain points; the attention weights are the affordance
distribution alpha_{b,i} over the height-scan points and a head predicts whether the body makes a new contact.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Height-scan grid: L=17 points along x by W=11 along y, spacing 0.1 m (1.6 x 1.0 m). Points are flattened y-major
# (y outer, x inner), i.e. index = iy * SCAN_L + ix. The planner input uses the authors' deploy encoding
# (x, y, z_enc) with z_enc = pelvis_z - hit_z - 0.5; ``points_physical`` gives z = hit_z - pelvis_z.
SCAN_L, SCAN_W = 17, 11
NUM_POINTS = SCAN_L * SCAN_W
# Semantic body order (the env maps the names to links).
# order of the authors' exported planner (inferred by probing policy.onnx of the ACP playground)
CONTACT_BODIES = ("left_knee", "right_knee", "pelvis", "left_foot", "right_foot", "left_elbow", "right_elbow", "left_hand", "right_hand")
CONTACT_LINKS = (
    "left_knee_link", "right_knee_link", "pelvis", "left_ankle_roll_link", "right_ankle_roll_link",
    "left_elbow_link", "right_elbow_link", "left_wrist_yaw_link", "right_wrist_yaw_link",
)  # fmt: skip
NUM_BODIES = len(CONTACT_BODIES)
PROPRIO_DIM = 5 * (3 + 3 + 23 + 23 + 23)  # five-step history
GOAL_DIM = 3


def points_physical(points: torch.Tensor) -> torch.Tensor:
    """Encoded scan points (x, y, z_enc) -> physical points with z = hit_z - pelvis_z = -z_enc - 0.5."""
    return torch.cat([points[..., :2], -points[..., 2:] - 0.5], dim=-1)


def _mlp(dims: list[int], last_act: bool = False) -> nn.Sequential:
    """Linear layers with ReLU in between (and after the last one if ``last_act``)."""
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2 or last_act:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class AttentionContactPlanner(nn.Module):
    def __init__(self, d: int = 96, num_heads: int = 32):
        super().__init__()
        self.point_embed = nn.Linear(3, 16)
        self.height_conv = nn.Sequential(nn.Conv2d(1, 16, 5, padding=2), nn.ELU(), nn.Conv2d(16, 48, 5, padding=2), nn.ELU())
        self.map_proj = nn.Linear(64, d)
        self.context_mlp = _mlp([d, 64, 64])
        self.shared_mlp = _mlp([PROPRIO_DIM + GOAL_DIM + 64, 128, 128, d])
        self.body_embed = nn.Parameter(torch.randn(NUM_BODIES, d) * 0.1)
        self.film = _mlp([d, 64, 2 * d])
        self.attn = nn.MultiheadAttention(embed_dim=d, num_heads=num_heads, dropout=0.0, batch_first=True)
        self.occ_head = _mlp([d + 64, 512, 512, 512, 1])

    def forward(self, proprio: torch.Tensor, goal: torch.Tensor, points: torch.Tensor) -> dict[str, torch.Tensor]:
        """proprio (B, 375), goal (B, 3), points (B, 187, 3) encoded (x, y, z_enc) -> alpha (B, 9, 187), occ_logit (B, 9).

        The CNN sees the z_enc channel as (11, 17) (y rows, x cols), following the paper intent; the authors' export
        reshaped it to (17, 11) (a transpose quirk)."""
        B = points.shape[0]
        emb = F.elu(self.point_embed(points))  # (B, P, 16)
        hmap = self.height_conv(points[..., 2].reshape(B, 1, SCAN_W, SCAN_L))  # (B, 48, W, L)
        hmap = hmap.reshape(B, 48, NUM_POINTS).transpose(1, 2)  # (B, P, 48)
        feat = self.map_proj(torch.cat([emb, hmap], dim=-1))  # (B, P, d) keys / values
        pooled = self.context_mlp(feat).max(dim=1).values  # (B, 64)
        s = self.shared_mlp(torch.cat([proprio, goal, pooled], dim=-1))  # (B, d)
        gamma, beta = self.film(self.body_embed).chunk(2, dim=-1)  # (9, d) each
        query = gamma[None] * s[:, None] + beta[None]  # (B, 9, d)
        att, alpha = self.attn(query, feat, feat, need_weights=True, average_attn_weights=True)
        occ_logit = self.occ_head(torch.cat([att, pooled[:, None].expand(-1, NUM_BODIES, -1)], dim=-1)).squeeze(-1)
        return dict(alpha=alpha, occ_logit=occ_logit)

    @staticmethod
    @torch.no_grad()
    def decode(out: dict[str, torch.Tensor], points: torch.Tensor, k: int = 1, temperature: float = 1.0, occ_threshold: float = 0.5) -> tuple[torch.Tensor, torch.Tensor]:
        """Contact targets (B, 9, 3) and validity (B, 9). ``points`` are the encoded scan points; targets are physical
        (base-yaw frame, z relative to pelvis). Invalid targets (occupancy <= threshold) are zero vectors."""
        points = points_physical(points)
        top_a, top_i = out["alpha"].topk(k, dim=-1)  # (B, 9, k)
        w = torch.softmax(torch.log(top_a.clamp_min(1e-12)) / temperature, dim=-1)
        pts = torch.gather(points[:, None].expand(-1, NUM_BODIES, -1, -1), 2, top_i[..., None].expand(-1, -1, -1, 3))  # (B, 9, k, 3)
        targets = (w[..., None] * pts).sum(dim=2)
        valid = torch.sigmoid(out["occ_logit"]) > occ_threshold
        return targets * valid[..., None], valid

    @torch.no_grad()
    def selected_targets(self, proprio: torch.Tensor, goal: torch.Tensor, points: torch.Tensor, use: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Skip unused observation rows when a smaller padded batch is available.

        Layers do not mix samples (no batch normalization or dropout). Pad with dummy rows to
        128-row buckets, at least 1024: small cuDNN/GEMM batches otherwise use different TF32
        algorithms. On checkpoint-1500 inputs these buckets preserve all decoded decisions
        and limit logit differences to float32 rounding. This is only used when no planned
        reward needs the omitted rows. Full point noise and the choice mask are drawn first.
        """
        ids = use.nonzero(as_tuple=True)[0]
        B, selected = points.shape[0], ids.numel()
        targets = points.new_zeros(B, NUM_BODIES, 3)
        valid = torch.zeros(B, NUM_BODIES, device=points.device, dtype=torch.bool)
        if selected == 0:
            return targets, valid
        padded = max(1024, ((selected + 127) // 128) * 128)
        if padded >= B:
            return self.decode(self(proprio, goal, points), points)
        inputs = []
        for x in (proprio, goal, points):
            rows = x.index_select(0, ids)
            inputs.append(torch.cat((rows, rows.new_zeros(padded - selected, *rows.shape[1:])), dim=0))
        out = {key: value[:selected] for key, value in self(*inputs).items()}
        tgt, ok = self.decode(out, inputs[2][:selected])
        targets.index_copy_(0, ids, tgt)
        valid.index_copy_(0, ids, ok)
        return targets, valid


def planner_loss(out: dict[str, torch.Tensor], points: torch.Tensor, target_pos: torch.Tensor, target_flag: torch.Tensor, sigma: float = 0.1, affordance_norm: torch.Tensor | None = None, occupancy_weight: float = 1.0) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Eq. (1)-(2): soft-target cross-entropy on alpha (valid targets only) + occupancy BCE.

    points (N, 187, 3) encoded scan points, target_pos (N, 9, 3) physical (z = hit_z - pelvis_z), target_flag (N, 9) bool.

    The affordance term is normalized by the number of valid targets (``affordance_norm``, default: those of this batch). A batch accumulated
    over sub-batches passes the valid count of the WHOLE batch to every sub-batch (and ``occupancy_weight`` = 1 / sub-batches), so that the
    sum of the sub-batch losses is the loss of the whole batch."""
    points = points_physical(points)
    alpha, logit = out["alpha"], out["occ_logit"]
    flag = target_flag.float()
    d2 = (points[:, None] - target_pos[:, :, None]).pow(2).sum(-1)  # (N, 9, P)
    p_star = torch.softmax(-d2 / (2 * sigma**2), dim=-1)
    ce = -(p_star * torch.log(alpha.clamp_min(1e-8))).sum(-1)  # (N, 9)
    affordance = (flag * ce).sum() / (flag.sum() if affordance_norm is None else affordance_norm).clamp_min(1.0)
    occupancy = F.binary_cross_entropy_with_logits(logit, flag)
    loss = 1.0 * affordance + occupancy_weight * occupancy
    with torch.no_grad():
        acc = ((logit > 0) == target_flag).float().mean()
        top = torch.gather(points[:, None].expand(-1, NUM_BODIES, -1, -1), 2, alpha.argmax(-1)[..., None, None].expand(-1, -1, 1, 3)).squeeze(2)
        err = (top - target_pos).norm(dim=-1)
        pos_err = (err * flag).sum() / flag.sum().clamp_min(1.0)
    return loss, dict(affordance=affordance.detach(), occupancy=occupancy.detach(), occ_acc=acc, top1_err=pos_err)
