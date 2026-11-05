"""Hindsight contact labeling (ACP paper, Table S5): label each step with the body's next new contact in the episode."""

from __future__ import annotations

import torch


def label_episode(
    contact_force_z: torch.Tensor,
    body_pos_w: torch.Tensor,
    base_pos_w: torch.Tensor,
    base_yaw: torch.Tensor,
    force_threshold: float = 1.0,
    min_segment: int = 10,
    horizon: int = 75,
    footprint: tuple[tuple[float, float], tuple[float, float]] = ((-0.8, 0.8), (-0.5, 0.5)),
) -> tuple[torch.Tensor, torch.Tensor]:
    """contact_force_z (T, 9), body_pos_w (T, 9, 3), base_pos_w (T, 3), base_yaw (T,), all at 50 Hz.

    Returns target_pos (T, 9, 3) in the gravity-aligned, yaw-rotated, base-centered frame of each step and
    target_flag (T, 9). The target of step t is the position at the first contact onset in (t, t + horizon]."""
    T, nb = contact_force_z.shape
    dev = contact_force_z.device
    contact = contact_force_z.abs() > force_threshold
    # drop contact segments shorter than ``min_segment`` steps
    prev = torch.cat([torch.zeros_like(contact[:1]), contact[:-1]])
    run_id = torch.cumsum((contact & ~prev).long(), dim=0) * contact  # 0 = no contact
    counts = torch.zeros(T + 1, nb, dtype=torch.long, device=dev).scatter_add_(0, run_id, torch.ones_like(run_id))
    keep = contact & (counts.gather(0, run_id) >= min_segment)
    onset = keep & ~torch.cat([torch.zeros_like(keep[:1]), keep[:-1]])
    # next onset strictly after t: reverse cumulative min over the onset indices
    idx = torch.arange(T, device=dev)[:, None].expand(T, nb)
    rc = torch.where(onset, idx, torch.full_like(idx, T)).flip(0).cummin(0).values.flip(0)
    nxt = torch.cat([rc[1:], torch.full_like(rc[:1], T)])  # (T, 9)
    exists = nxt <= idx + horizon
    exists &= nxt < T
    p_w = body_pos_w[nxt.clamp(max=T - 1), torch.arange(nb, device=dev)[None]]  # (T, 9, 3)
    d = p_w - base_pos_w[:, None]
    c, s = torch.cos(base_yaw)[:, None], torch.sin(base_yaw)[:, None]
    p_b = torch.stack([c * d[..., 0] + s * d[..., 1], -s * d[..., 0] + c * d[..., 1], d[..., 2]], dim=-1)
    (x0, x1), (y0, y1) = footprint
    flag = exists & (p_b[..., 0] >= x0) & (p_b[..., 0] <= x1) & (p_b[..., 1] >= y0) & (p_b[..., 1] <= y1)
    return p_b * flag[..., None], flag


def episode_success_reference(start_phase: float, end_phase: float) -> bool:
    """Reference-tracking episode is kept if it starts at the beginning and reaches the end of the motion."""
    return start_phase <= 0.1 and end_phase >= 0.9


def episode_success_goal(min_xy_dist_while_z_ok: float) -> bool:
    """Goal-reaching episode is kept if the base got within 0.5 m horizontally while within 0.1 m vertically."""
    return min_xy_dist_while_z_ok <= 0.5


class HindsightReplay:
    """GPU ring buffer of labeled planner samples."""

    # TODO: terrain-tile partitioning and quality-based eviction.
    shapes = dict(proprio=(375,), goal=(3,), points=(187, 3), target_pos=(9, 3), target_flag=(9,))

    def __init__(self, capacity: int, device: str = "cpu"):
        self.capacity, self.device = capacity, device
        self.data = {k: torch.zeros(capacity, *s, dtype=torch.bool if k == "target_flag" else torch.float32, device=device) for k, s in self.shapes.items()}
        self.ptr, self.size = 0, 0

    def __len__(self) -> int:
        return self.size

    def add(self, batch: dict[str, torch.Tensor]) -> None:
        n = batch["proprio"].shape[0]
        if n > self.capacity:  # keep the newest ``capacity`` samples
            batch = {k: v[-self.capacity:] for k, v in batch.items()}
            n = self.capacity
        idx = (self.ptr + torch.arange(n, device=self.device)) % self.capacity
        for k, buf in self.data.items():
            buf[idx] = batch[k].to(self.device, buf.dtype)
        self.ptr = (self.ptr + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, n: int) -> dict[str, torch.Tensor]:
        idx = torch.randint(0, self.size, (n,), device=self.device)
        return {k: v[idx] for k, v in self.data.items()}
