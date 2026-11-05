"""Termination terms of the ACP tracking task on the pelvis (the base of the authors' deploy stack)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.utils.lab_api.math import quat_apply_inverse

from .commands import AcpReferenceContacts, to_base_yaw_frame

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def pos_far_from_ref(
    env: ManagerBasedRlEnv,
    command_name: str,
    distance_threshold: float = 0.25,
    height_only: bool = True,
) -> torch.Tensor:
    """The pelvis is further than ``distance_threshold`` from the reference pelvis (height only if ``height_only``)."""
    term: AcpReferenceContacts = env.command_manager.get_term(command_name)
    diff = term.base_pos_w - term.ref_base_pos_w
    distance = diff[:, 2].abs() if height_only else diff.norm(dim=-1)
    return distance > distance_threshold


def pos_xy_far_from_ref(env: ManagerBasedRlEnv, command_name: str, distance_threshold: float = 0.5) -> torch.Tensor:
    """The pelvis is further than ``distance_threshold`` horizontally from the reference pelvis (a robot that stalls in front of an obstacle)."""
    term: AcpReferenceContacts = env.command_manager.get_term(command_name)
    return (term.base_pos_w[:, :2] - term.ref_base_pos_w[:, :2]).norm(dim=-1) > distance_threshold


def contact_far_from_ref(env: ManagerBasedRlEnv, command_name: str, distance_threshold: float = 0.3, grace_frames: int = 2, frame: str = "world", upright_cos: float = 0.707) -> torch.Tensor:
    """Contact-consistent termination: a contact link is further than ``distance_threshold`` (3-D) from its reference position while the reference is in contact with it; the first
    ``grace_frames`` frames of every reference contact segment are ignored (touch-down timing). ``frame="world"``: world positions (``track_contact_pos_global``'s quantity);
    ``frame="base"``: the actual and the reference positions in their own pelvis-yaw frames (``track_contact_pos_base``'s quantity), free of the horizontal drift of the root. The pelvis-yaw
    frame is ill-conditioned when the pelvis is not upright: the termination only looks at the frames where the REFERENCE pelvis is upright (projected gravity z < -``upright_cos``, tilt < 45 deg)."""
    term: AcpReferenceContacts = env.command_manager.get_term(command_name)
    pos = term.robot.data.body_link_pos_w[:, term.contact_body_ids]
    if frame == "base":
        dist = (to_base_yaw_frame(pos, term.base_pos_w, term.base_quat_w) - to_base_yaw_frame(term.ref_link_pos_w, term.ref_base_pos_w, term.ref_base_quat_w)).norm(dim=-1)
    else:
        assert frame == "world"
        dist = (pos - term.ref_link_pos_w).norm(dim=-1)
    ref_pg_z = quat_apply_inverse(term.ref_base_quat_w, term.robot.data.gravity_vec_w)[:, 2]
    active = term.ref_contact_flag & (term.ref_contact_age > grace_frames) & (ref_pg_z < -upright_cos).unsqueeze(-1)
    return (active & (dist > distance_threshold)).any(-1)


def projected_gravity_far_from_ref(
    env: ManagerBasedRlEnv,
    command_name: str,
    projected_gravity_threshold: float = 0.8,
    z_only: bool = False,
) -> torch.Tensor:
    """The projected gravity of the pelvis is further than the threshold from that of the reference pelvis."""
    term: AcpReferenceContacts = env.command_manager.get_term(command_name)
    gravity = term.robot.data.gravity_vec_w
    pg = quat_apply_inverse(term.base_quat_w, gravity)
    ref_pg = quat_apply_inverse(term.ref_base_quat_w, gravity)
    diff = (pg[:, 2] - ref_pg[:, 2]).abs() if z_only else torch.norm(pg - ref_pg, dim=-1)
    return diff > projected_gravity_threshold
