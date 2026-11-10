"""Reference motion access of the ACP runner: the SuperMimic ``MotionLib`` plus the contact tables of the env.

The contact tables are read from the env's ``acp_reference_contacts`` command term (``AcpReferenceContacts``: instantaneous
reference contact flags and the next-contact-onset world position / existence of every body, per sidecar frame). They live on the
time base of the contact sidecars (which may differ from the motion's frame rate); the lookup floors the reference time of the
env exactly like ``AcpReferenceContacts.frame_index``, so the imagined observation and rewards use exactly the env's labels.
"""

from __future__ import annotations

import torch

from instinct_mj.rl.supermimic.motion_lib import MotionLib
from instinct_mj.tasks.acp.mdp.commands import AcpReferenceContacts

from .planner import CONTACT_LINKS

PELVIS = CONTACT_LINKS.index("pelvis")
NUM_CONTACT = len(CONTACT_LINKS)


class ACPMotion(MotionLib):
    """``MotionLib`` over the ``TerrainMotion`` buffer of the task (link arrays are in ``CONTACT_LINKS`` order) with the per-motion
    contact tables of the env: ``flag`` (instantaneous reference contact, bool), ``next_valid`` (the body has a contact onset within the
    lookahead horizon) and ``next_pos`` (motion-local position of that next onset; world = local + origin), all (9,) per sidecar frame.
    The accessors take the global motion frame ``f`` and the fractional reference phase of ``MotionLib.current_frame``."""

    def __init__(self, manager, contacts: AcpReferenceContacts, device: str):
        super().__init__(manager, device)
        if self.link_p.shape[1] > NUM_CONTACT:  # the SuperMimic tracked bodies after the contact links (SM-CH5TB): ``link_*`` stay the contact links, ``track_*`` have all
            self.track_p, self.track_R, self.track_v, self.track_w = self.link_p, self.link_R, self.link_v, self.link_w
            self.link_p, self.link_R, self.link_v, self.link_w = (x[:, :NUM_CONTACT] for x in (self.link_p, self.link_R, self.link_v, self.link_w))
        assert contacts.table_flag.shape[0] == len(self.clip_len)
        self.table_fps, self.table_num_frames, self.table_age = contacts.table_fps, contacts.table_num_frames, contacts.table_age
        self.table_flag, self.table_valid, self.table_pos = contacts.table_flag, contacts.table_valid, contacts.table_pos_w
        self.footprint_x, self.footprint_y = contacts.cfg.footprint_x, contacts.cfg.footprint_y

    def table_index(self, f: torch.Tensor, phase: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(motion id, sidecar frame) of the reference time ``(f - clip_start + phase) / framerate`` (``frame_index`` of the env)."""
        motion = self.clip_id[f]
        t = ((f - self.clip_start[motion]).double() + phase.double()) / self.framerate
        frame = torch.floor(t * self.table_fps[motion].double() + 1e-3).long()  # floor + tolerance, as the env
        return motion, torch.minimum(frame.clamp(min=0), self.table_num_frames[motion] - 1)

    def flag(self, f: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        return self.table_flag[self.table_index(f, phase)]

    def age(self, f: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        return self.table_age[self.table_index(f, phase)]

    def next_valid(self, f: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        return self.table_valid[self.table_index(f, phase)]

    def next_pos(self, f: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        return self.table_pos[self.table_index(f, phase)]

    def pelvis_yaw(self, f: torch.Tensor) -> torch.Tensor:
        """Heading of the reference pelvis (``yaw_of_quat``: atan2 of the projected x axis)."""
        R = self.link_R[f][:, PELVIS]
        return torch.atan2(R[:, 1, 0], R[:, 0, 0])

    def goal_target(self, f: torch.Tensor, local_frame: torch.Tensor, origin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Goal target (world position of the reference pelvis, heading) at the clip-local frame ``local_frame`` (N,) of the
        clip of the global frame ``f`` (N,) (the env's ``AcpGoalCommand._sample_goal``)."""
        g = self.clip_start[self.clip_id[f]] + local_frame
        return self.link_p[g][:, PELVIS] + origin, self.pelvis_yaw(g)
