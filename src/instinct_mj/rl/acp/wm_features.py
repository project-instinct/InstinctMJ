"""Extra inputs of the ACP world model: the PD torque the actuators actually deliver and the clearance of the nine contact links
(``ACPRunnerCfg.wm_physics_inputs``, default off: the plain world model is unchanged).

Motivation (research/ds_proposal.md): the world model is given the *position target* of the PD actuators (the action) and the terrain
scan, but neither the torque those targets produce (which depends on ``q``, ``qd``, the per-env ``kp`` / ``kd`` scales and the action
delay) nor how far every contact link is from the ground. Both are known exactly at training time and are recomputed from the
imagined state in the imagination, so this is a pure input-representation change: same network, same losses, same training target.
STRIDE (Kotecha et al., arXiv 2603.08478) takes the applied torque ``tau`` as a *known input* of the structured dynamics; Parmar et
al. (arXiv 2103.15406) show that a network fed only a smoothed view of a stiff interaction learns the conditional mean of the modes.

Parts, all appended after the ``(scan, previous action)`` context of :class:`~instinct_mj.rl.acp.runner.ACPRunner`:

- ``torque`` (23): ``clip(kp * (target - q) - kd * qd)`` of the controlled joints, with the action delay of the episode applied exactly
  as :meth:`instinct_mj.rl.acp.mdp.ACPMDP.torque_terms` does (the env's PD law, effort limits, encoder bias).
- ``clearance`` (9): height of the origin of every contact link above the terrain below it (the 0.05 m height map: cheap and
  differentiable, the scan already carries the exact geometry).
- ``clearance_vel`` (9): vertical velocity of those link origins (world frame; ``kin.forward`` gives it for free).
"""

from __future__ import annotations

import torch

from .mdp import ACT_DIM, NUM_CONTACT, XI_BIAS, XI_KD, XI_KP, XI_LAG

PARTS = ("torque", "clearance", "clearance_vel")
PART_DIM = {"torque": ACT_DIM, "clearance": NUM_CONTACT, "clearance_vel": NUM_CONTACT}


def parse(spec: str | tuple[str, ...]) -> tuple[str, ...]:
    """Comma-separated config value -> tuple of part names (unknown names raise)."""
    parts = tuple(p for p in (spec.split(",") if isinstance(spec, str) else spec) if p)
    if unknown := set(parts) - set(PARTS):
        raise ValueError(f"wm_physics_inputs: unknown parts {sorted(unknown)} (choose from {PARTS})")
    return parts


def dim(spec: str | tuple[str, ...]) -> int:
    """Number of extra inputs of ``spec``."""
    return sum(PART_DIM[p] for p in parse(spec))


def pd_torque(mdp, q: torch.Tensor, qd: torch.Tensor, a: torch.Tensor, a_prev: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
    """(..., 23) PD torque at the start of a control step, as the env's actuators deliver it: position target
    ``default + scale * a_eff - bias``, the command ``lag`` physics steps earlier (``acp_training`` / ``DelayBuffer``), clipped to the
    effort limits. Same formula as ``ACPMDP.torque_terms`` evaluated at ``(q, qd)`` instead of at the end of the substep."""
    use_prev = xi[..., XI_LAG] >= mdp.decimation  # the lag is shared by all joints
    a_eff = torch.where(use_prev, mdp.pad(a_prev), mdp.pad(a))
    target = mdp.default_q + mdp.act_scale_j * a_eff - xi[..., XI_BIAS] * mdp.act_mask
    kp, kd = mdp.kp_nom * xi[..., XI_KP], mdp.kd_nom * xi[..., XI_KD]
    tau = torch.clamp(kp * (target - q) - kd * qd, -mdp.effort_limit, mdp.effort_limit)
    return tau[..., mdp.act_ids]


def features(mdp, S, a: torch.Tensor, a_prev: torch.Tensor, xi: torch.Tensor, fk, spec: str | tuple[str, ...]) -> torch.Tensor:
    """(..., sum(PART_DIM[p])) extra world-model inputs. ``fk``: the forward kinematics of ``S`` (positions, rotations, linear and angular
    velocities of the nine contact links), computed by the caller anyway for the height scan."""
    parts = parse(spec)
    out = []
    if "torque" in parts:
        out.append(pd_torque(mdp, S[4], S[5], a, a_prev, xi))
    if "clearance" in parts or "clearance_vel" in parts:
        bp, bv = fk[0], fk[2]  # (..., 9, 3) link origins and their linear velocities
        if "clearance" in parts:
            out.append(bp[..., 2] - mdp.scanner.ground_height(bp[..., :2]))
        if "clearance_vel" in parts:
            out.append(bv[..., 2])
    return torch.cat(out, -1)
