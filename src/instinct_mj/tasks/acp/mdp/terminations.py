"""Termination terms of the ACP tracking task on the pelvis (the base of the authors' deploy stack)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse

from mjlab.utils.lab_api.math import matrix_from_quat

from instinct_mj.rl.acp.planner import CONTACT_LINKS, TRACKED_LINKS
from instinct_mj.rl.acp.sdf_contact import VIOLATION_LINKS, VIOLATION_REF_CLEARANCE, get_violation_sdf

from .commands import AcpReferenceContacts

FEET = (CONTACT_LINKS.index("left_ankle_roll_link"), CONTACT_LINKS.index("right_ankle_roll_link"))

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
    """The pelvis is horizontally further than ``distance_threshold`` from the reference pelvis (the robot lost the reference)."""
    term: AcpReferenceContacts = env.command_manager.get_term(command_name)
    return (term.base_pos_w[:, :2] - term.ref_base_pos_w[:, :2]).norm(dim=-1) > distance_threshold


def feet_far_from_ref(env: ManagerBasedRlEnv, command_name: str, distance_threshold: float = 0.5) -> torch.Tensor:
    """A foot (ankle roll link) is further than ``distance_threshold`` (3-D) from the reference foot (the robot lost the reference)."""
    term: AcpReferenceContacts = env.command_manager.get_term(command_name)
    feet = list(FEET)
    pos = term.robot.data.body_link_pos_w[:, [term.contact_body_ids[i] for i in feet]]
    return (pos - term.ref_link_pos_w[:, feet]).norm(dim=-1).max(-1).values > distance_threshold


def body_contact_violation(env: ManagerBasedRlEnv, command_name: str, sensor_name: str, asset_cfg: SceneEntityCfg, threshold: float = 1.0) -> torch.Tensor:
    """A body that must not touch the terrain (``asset_cfg``: the torso, which carries the head) has a terrain contact force above ``threshold`` within the last
    policy step (the max over the physics substeps of the sensor history) while the reference body is further than ``VIOLATION_REF_CLEARANCE`` from the terrain
    (exact SDF of its collision primitives at the reference pose): the contacts of get-up / crawl references are no violation."""
    force = env.scene.sensors[sensor_name].data.force_history.norm(dim=-1).max(-1).values[:, asset_cfg.body_ids]
    ref = env.command_manager.get_term(command_name).motion_reference.reference_frame
    ids = [TRACKED_LINKS.index(n) for n in VIOLATION_LINKS]
    d_ref = get_violation_sdf(env).distance(ref.link_pos_w[:, 0, ids], matrix_from_quat(ref.link_quat_w[:, 0, ids])).min(-1).values
    return (force > threshold).any(-1) & (d_ref > VIOLATION_REF_CLEARANCE)


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
