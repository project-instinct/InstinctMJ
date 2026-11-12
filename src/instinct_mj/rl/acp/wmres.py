"""AHAC-style contact-triggered truncation of the imagination BPTT (``ACPRunnerCfg.wmres_cut*``, default off).

Motivation (``research/glm_proposal.md`` §3): the policy gradient of the H-step imagined return is biased exactly when the
rollout crosses a contact discontinuity (Suh et al. 2022, arXiv 2202.00817: the bias of a first-order gradient grows with the
probability of crossing; it does not average out). The Codex diagnosis of this run (it 1000) confirms the scalar symptom: the
imagined reward at contact switches is biased by about +0.54 and swapping the contact source for the oracle leaves it at +0.53,
i.e. the multi-step state prediction through the switch is what is wrong, not the contact head. AHAC (Georgiev et al. 2024,
arXiv 2405.17784) answers the same problem for differentiable simulators: detect the contact event (their trigger is a change of
the contact force beyond a stiffness threshold C), stop the backpropagation there and continue the return with the critic value
of the last clean state; the actor never differentiates through the event, the critic (fitted on real data with GAE) carries it.

Here the trigger is the world-model contact head of the current run (``geo_contact=False``: the head's probabilities feed the
imagined contact rewards, so a threshold crossing of ``sigmoid(logits)`` is the visible contact event of the imagination). The
truncation: at the first event step ``t*`` of a start, the event step's reward is scaled by ``1 - alpha`` (default 1: dropped),
the steps after it get zero continuation, and ``J`` gains ``gamma^t* * c_t* * V(s_t*)`` of the last pre-event state -- the same
per-(step, start) bootstrap pattern as the horizon bootstrap, reusing the same critic. The critic fit and its GAE targets are
NOT truncated (AHAC fits the critic on the full trajectory), so the value that takes over at the event still learns through it.

Everything here is inert unless ``wmres_cut`` is true and the iteration is past ``wmres_cut_warmup_iters`` (the contact head is
noise before it is trained).
"""

from __future__ import annotations

import torch


def cut_step_from_logits(logits: torch.Tensor, theta: float, flux: float, valid: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """First contact-event step per start from the imagined contact logits.

    ``logits``: (H, B, C) contact logits of the imagined transitions (row t is the prediction of the transition t, i.e. the
    contact of the state reached by it). An event fires at step t >= 1 when any link's probability crosses the threshold
    ``theta`` between t-1 and t (and moves by more than ``flux``, the anti-dither equivalent of AHAC's stiffness threshold C).
    ``valid``: optional (H, B) 0/1 mask; steps that are invalid (beyond the clip end of the reference) cannot fire.

    Returns ``(t_star, fired)``: t_star (B,) long in [1, H-1], meaningful only where ``fired`` (B,) is true.
    """
    p = torch.sigmoid(logits.detach().float())
    cross = (p[1:] > theta) != (p[:-1] > theta)  # (H-1, B, C)
    if flux > 0:
        cross = cross & ((p[1:] - p[:-1]).abs() > flux)
    ev = cross.any(-1)  # (H-1, B): event of the transition t, rows t = 1 .. H-1
    if valid is not None:
        ev = ev & (valid[1:] > 0.5)
    fired, first = ev.max(0)  # first event per start (0 when none)
    return first + 1, fired


def cut_termination(t_star: torch.Tensor, fired: torch.Tensor, HZ: int, B: int) -> torch.Tensor:
    """The cut as a termination indicator ``d_cut`` (HZ, B): 1 at the event step of each fired start, else 0."""
    d_cut = torch.zeros(HZ, B, device=t_star.device)
    b = fired.nonzero(as_tuple=False).squeeze(-1)
    d_cut[t_star[b], b] = 1.0
    return d_cut


def parse_caps(spec: str, names: list[str], HZ: int, device) -> torch.Tensor:
    """Per-clip horizon caps from the ``wmres_cap_clips`` spec "prefix=H[,prefix=H...]" (a clip matches the FIRST prefix it starts with).

    Returns a long tensor (n_clips,): the trusted horizon of every clip, ``HZ`` where no prefix matches. Names that no cap uses stay at HZ so the
    config is inert for them; an unused prefix (no clip matches) raises, as does an out-of-range H.
    """
    caps = torch.full((len(names),), HZ, dtype=torch.long, device=device)
    used = torch.zeros(len(names), dtype=torch.bool, device=device)
    for part in filter(None, (p.strip() for p in spec.split(","))):
        prefix, _, h = part.partition("=")
        if not prefix or not h.strip().isdigit():
            raise ValueError(f'wmres_cap_clips: bad entry "{part}" (want "prefix=H")')
        H = int(h)
        if not 1 <= H < HZ:
            raise ValueError(f'wmres_cap_clips: H={H} of "{prefix}" must be in [1, {HZ - 1}]')
        hit = [i for i, n in enumerate(names) if n.startswith(prefix)]
        if not hit:
            raise ValueError(f'wmres_cap_clips: no clip name starts with "{prefix}" (have e.g. {names[:3]})')
        first = [i for i in hit if not used[i]]  # a clip matches the first prefix only
        caps[first] = H
        used[first] = True
    return caps
