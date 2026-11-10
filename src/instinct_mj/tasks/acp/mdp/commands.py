"""Command terms of the ACP tracking task: reference contacts (Stage 1 contact targets) and the goal command."""

from __future__ import annotations

import json
import os
from copy import deepcopy as _deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch
import yaml
from mjlab.managers import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import quat_apply_inverse, wrap_to_pi, yaw_quat

from instinct_mj.rl.acp.planner import CONTACT_BODIES, CONTACT_LINKS, TRACKED_LINKS

if TYPE_CHECKING:
    from mjlab.entity import Entity
    from mjlab.envs import ManagerBasedRlEnv

    from instinct_mj.motion_reference.motion_files.terrain_motion import TerrainMotion
    from instinct_mj.motion_reference.motion_reference_manager import MotionReferenceManager

NUM_CONTACT_BODIES = len(CONTACT_BODIES)


def contacts_sidecar_path(motion_file: str) -> str:
    """Path of the contact sidecar ``<stem>.contacts.npz`` next to a motion file ``<stem>.npz``."""
    return os.path.splitext(motion_file)[0] + ".contacts.npz"


def yaw_of_quat(quat: torch.Tensor) -> torch.Tensor:
    """Heading (yaw) from the horizontal projection of the base x-axis, quat in (w, x, y, z)."""
    w, x, y, z = quat.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def to_base_yaw_frame(points_w: torch.Tensor, base_pos_w: torch.Tensor, base_quat_w: torch.Tensor) -> torch.Tensor:
    """p_B = Rz(-yaw)(p_w - base_pos): gravity aligned (yaw only), z relative to the base. points_w: (N, K, 3)."""
    yaw_q = yaw_quat(base_quat_w).unsqueeze(1).expand(-1, points_w.shape[1], -1)
    return quat_apply_inverse(yaw_q, points_w - base_pos_w.unsqueeze(1))


def reference_origin(motion_reference: MotionReferenceManager) -> torch.Tensor:
    """Per-env origin the reference motion is placed with (the terrain origin of the motion), (N, 3)."""
    buffer = next(iter(motion_reference.motion_buffers.values()))
    return buffer._get_motion_based_origin(motion_reference.env_origins, motion_reference.ALL_INDICES)


@dataclass(kw_only=True)
class AcpCommandCfg(CommandTermCfg):
    """Common fields of the ACP command terms."""

    resampling_time_range: tuple[float, float] = (1e9, 1e9)

    motion_reference: str = "motion_reference"
    """Name of the motion reference sensor."""

    asset_name: str = "robot"

    odometry_sensor: str = "height_scanner"
    """Sensor carrying the estimated pelvis pose (odometry delay and drift) that the actor-side transforms use."""

    def __deepcopy__(self, memo):
        """Custom deepcopy to avoid ABC metadata pickling issues in Python 3.13+ (as ``ShadowingCommandBaseCfg``)."""
        cls = type(self)
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if isinstance(v, type) or k.startswith("_abc"):
                setattr(result, k, v)
            else:
                setattr(result, k, _deepcopy(v, memo))
        return result


class AcpCommandBase(CommandTerm):
    """Reference-time bookkeeping shared by the ACP command terms."""

    cfg: AcpCommandCfg

    def __init__(self, cfg: AcpCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self.robot: Entity = env.scene[cfg.asset_name]
        self.motion_reference: MotionReferenceManager = env.scene[cfg.motion_reference]
        self.motion_buffer: TerrainMotion = next(iter(self.motion_reference.motion_buffers.values()))
        # The base is the pelvis (IMU, scan, goal and contact frames), as in the authors' deploy stack (g1_rsl_cpp); the
        # asset root is the pelvis too (mjlab's G1). The reference pelvis is a link of the motion reference.
        self.odometry_sensor = env.scene[cfg.odometry_sensor]
        self.pelvis_id = self.robot.find_bodies("pelvis")[0][0]
        self.ref_pelvis_id = CONTACT_LINKS.index("pelvis")

    @property
    def base_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_link_pos_w[:, self.pelvis_id]

    @property
    def base_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_link_quat_w[:, self.pelvis_id]

    @property
    def base_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_link_lin_vel_w[:, self.pelvis_id]

    @property
    def base_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_link_ang_vel_w[:, self.pelvis_id]

    @property
    def ref_base_pos_w(self) -> torch.Tensor:
        return self.motion_reference.reference_frame.link_pos_w[:, 0, self.ref_pelvis_id]

    @property
    def ref_base_quat_w(self) -> torch.Tensor:
        return self.motion_reference.reference_frame.link_quat_w[:, 0, self.ref_pelvis_id]

    @property
    def ref_base_lin_vel_w(self) -> torch.Tensor:
        return self.motion_reference.reference_frame.link_lin_vel_w[:, 0, self.ref_pelvis_id]

    @property
    def ref_base_ang_vel_w(self) -> torch.Tensor:
        return self.motion_reference.reference_frame.link_ang_vel_w[:, 0, self.ref_pelvis_id]

    @property
    def motion_ids(self) -> torch.Tensor:
        """Motion clip of every env, (N,)."""
        return self.motion_buffer.assigned_env_motion_selection

    @property
    def motion_time(self) -> torch.Tensor:
        """Time (s) in the clip of the current reference frame, (N,)."""
        return self.motion_buffer.motion_buffer_start_time_s + self._env.episode_length_buf * self._env.step_dt


@dataclass(kw_only=True)
class AcpReferenceContactsCfg(AcpCommandCfg):
    """Stage 1 contact targets: the next contact onset of every contact body read from the reference contact labels."""

    contact_sensor: str = "acp_contact_forces"
    """Contact sensor over the contact links."""

    lookahead_s: float = 1.5
    """A body targets its next contact onset within this horizon (paper: 75 steps at 50 Hz)."""

    require_full_clip: bool = True
    """The episode must last until the clip end: the longest clip has to fit in ``episode_length_s`` (False: the episodes are capped, e.g. the SuperMimic reset variant)."""

    allow_label_collision_mismatch: bool = False
    """Accept contact sidecars labelled with another ``simplified_collision`` than the env simulates (default False: raise). v1 sets it True with ``simplified_collision=True``: the
    labels (on the authors' meshes) are kept as they are, not relabelled (user decision 2026-11-08); a note is printed once."""

    footprint_x: float = 0.8
    footprint_y: float = 0.5
    """Targets outside |x| <= footprint_x, |y| <= footprint_y (base-yaw frame) are invalid (zero vector)."""

    def build(self, env: ManagerBasedRlEnv) -> AcpReferenceContacts:
        return AcpReferenceContacts(self, env)


class AcpReferenceContacts(AcpCommandBase):
    """Reference contact labels / next-onset targets of the 9 contact bodies (order of ``CONTACT_BODIES``).

    Per motion the "next contact onset" table is precomputed from the sidecar ``<stem>.contacts.npz``: for frame t the
    smallest onset t' (first frame of a contact segment) with t < t' <= t + ``lookahead_s``, and its world position
    ``contact_pos_w[t', b]``. The command is the target positions in the robot's base-yaw frame.
    """

    cfg: AcpReferenceContactsCfg

    def __init__(self, cfg: AcpReferenceContactsCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self.contact_body_ids = self.robot.find_bodies(CONTACT_LINKS, preserve_order=True)[0]
        self.contact_sensor = env.scene[cfg.contact_sensor]
        # the sensor lists its primaries in model body order: index of every contact link (CONTACT_LINKS order)
        self.contact_sensor_ids = [self.contact_sensor.primary_names.index(name) for name in CONTACT_LINKS]
        self._build_tables()
        self.metrics["target_valid_frac"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["ref_contact_frac"] = torch.zeros(self.num_envs, device=self.device)

    def _build_tables(self) -> None:
        cfg = self.motion_buffer.cfg
        with open(os.path.expanduser(cfg.metadata_yaml)) as f:
            motion_files = [os.path.join(cfg.path, m["motion_file"]) for m in yaml.safe_load(f)["motion_files"]]
        if len(motion_files) != self.motion_buffer.num_trajectories:
            raise ValueError("The motion files of the metadata do not match the motion buffer trajectories.")
        sidecars = []
        for motion_file in motion_files:
            path = contacts_sidecar_path(motion_file)
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"Missing contact sidecar '{path}' for motion '{motion_file}'. The ACP tracking task needs the"
                    " per-motion contact labels (framerate, body_names, contact_flag, contact_pos_w)."
                )
            sidecars.append(path)

        num_motions = len(sidecars)
        fps = torch.zeros(num_motions)
        num_frames = torch.zeros(num_motions, dtype=torch.long)
        loaded = []
        for i, path in enumerate(sidecars):
            data = np.load(path)
            if [str(n) for n in data["body_names"]] != list(CONTACT_BODIES):
                raise ValueError(f"'{path}' body_names {list(data['body_names'])} != CONTACT_BODIES {CONTACT_BODIES}.")
            # the labels must be measured on the collision geometry the env simulates (v3_train6 / v3_final: the authors' meshes)
            labelled_on = json.loads(str(data["params"])).get("simplified_collision") if "params" in data else None
            simulated = getattr(self._env.cfg, "simplified_collision", None)
            if labelled_on is not None and simulated is not None and bool(labelled_on) != bool(simulated) and self.cfg.allow_label_collision_mismatch:
                if not getattr(AcpReferenceContacts, "_mismatch_noted", False):
                    AcpReferenceContacts._mismatch_noted = True
                    print(f"[contacts] sidecars labelled with simplified_collision={labelled_on}, env simulates simplified_collision={simulated}: accepted (allow_label_collision_mismatch)", flush=True)
            elif labelled_on is not None and simulated is not None and bool(labelled_on) != bool(simulated):
                raise ValueError(
                    f"'{path}' was labelled with simplified_collision={labelled_on}, but the env simulates simplified_collision="
                    f"{simulated}. Re-label the dataset (``label_contacts --overwrite``) or set the env flag to match."
                )
            flag = torch.as_tensor(data["contact_flag"], dtype=torch.bool)
            pos = torch.as_tensor(data["contact_pos_w"], dtype=torch.float)
            fps[i] = float(data["framerate"])
            num_frames[i] = flag.shape[0]
            loaded.append((flag, pos))
        max_frames = int(num_frames.max())
        flag_tab = torch.zeros(num_motions, max_frames, NUM_CONTACT_BODIES, dtype=torch.bool)
        valid_tab = torch.zeros(num_motions, max_frames, NUM_CONTACT_BODIES, dtype=torch.bool)
        pos_tab = torch.zeros(num_motions, max_frames, NUM_CONTACT_BODIES, 3)
        for i, (flag, pos) in enumerate(loaded):
            T = flag.shape[0]
            horizon = int(self.cfg.lookahead_s * float(fps[i]) + 1e-6)
            onset = flag & ~torch.cat((torch.ones_like(flag[:1]), flag[:-1]), dim=0)
            frames = torch.arange(T).unsqueeze(-1).expand(T, NUM_CONTACT_BODIES)
            onset_idx = torch.where(onset, frames, torch.full_like(frames, T + horizon + 1))
            after = torch.cat((onset_idx[1:], torch.full_like(onset_idx[:1], T + horizon + 1)), dim=0)
            next_onset = torch.flip(torch.cummin(torch.flip(after, [0]), dim=0).values, [0])  # smallest onset > t
            valid = (next_onset < T) & (next_onset - frames <= horizon)
            flag_tab[i, :T] = flag
            valid_tab[i, :T] = valid
            pos_tab[i, :T] = torch.gather(pos, 0, next_onset.clamp(max=T - 1).unsqueeze(-1).expand(-1, -1, 3))
        self.table_fps = fps.to(self.device)
        self.table_num_frames = num_frames.to(self.device)
        self.table_flag = flag_tab.to(self.device)
        age_tab = torch.zeros(num_motions, max_frames, NUM_CONTACT_BODIES, dtype=torch.long, device=self.device)  # frames since the onset of the contact segment (1 at the first flag frame), 0 without contact
        for t in range(max_frames):
            age_tab[:, t] = (age_tab[:, t - 1] * (t > 0) + 1) * self.table_flag[:, t].long()
        self.table_age = age_tab
        self.table_valid = valid_tab.to(self.device)
        self.table_pos_w = pos_tab.to(self.device)
        self.max_clip_length_s = float((num_frames / fps).max())
        if self.cfg.require_full_clip and self.max_clip_length_s > self._env.max_episode_length_s:
            raise ValueError(
                f"The longest clip ({self.max_clip_length_s:.1f}s) is longer than episode_length_s"
                f" ({self._env.max_episode_length_s}s); the episode must last until the clip end."
            )

    # -- reference at the current time --

    @property
    def frame_index(self) -> torch.Tensor:
        """Sidecar frame of the current reference time, (N,)."""
        motion_ids = self.motion_ids
        # floor (not round): a still-future onset must not be skipped and the instantaneous flags must not advance early.
        # The product is evaluated in float64 with a small tolerance against the float32 rounding of the clip time.
        frame = torch.floor(self.motion_time.double() * self.table_fps[motion_ids].double() + 1e-3).long()
        return torch.minimum(frame.clamp(min=0), self.table_num_frames[motion_ids] - 1)

    @property
    def ref_contact_flag(self) -> torch.Tensor:
        """Instantaneous reference contact flags, (N, 9) bool."""
        return self.table_flag[self.motion_ids, self.frame_index]

    @property
    def ref_contact_age(self) -> torch.Tensor:
        """Frames (of the sidecar) since the onset of the current reference contact segment of every body (1 at the first contact frame, 0 without contact), (N, 9)."""
        return self.table_age[self.motion_ids, self.frame_index]

    @property
    def ref_target_pos_w(self) -> torch.Tensor:
        """World position (with the motion origin) of the next contact onset of every body, (N, 9, 3)."""
        pos = self.table_pos_w[self.motion_ids, self.frame_index]
        return pos + reference_origin(self.motion_reference).unsqueeze(1)

    @property
    def ref_target_exists(self) -> torch.Tensor:
        """Whether the body has a next contact onset within the horizon, (N, 9) bool."""
        return self.table_valid[self.motion_ids, self.frame_index]

    @property
    def ref_link_pos_w(self) -> torch.Tensor:
        """Reference world positions of the 9 contact links, (N, 9, 3)."""
        return self.motion_reference.reference_frame.link_pos_w[:, 0, :NUM_CONTACT_BODIES]

    @property
    def track_body_ids(self) -> list[int]:
        """Robot body ids of the SuperMimic tracked bodies (``TRACKED_LINKS``; the motion reference must carry them, SM-CH5TB)."""
        if not hasattr(self, "_track_body_ids"):
            assert self.motion_reference.cfg.link_of_interests is not None and list(self.motion_reference.cfg.link_of_interests) == list(TRACKED_LINKS), "the motion reference does not carry the tracked bodies"
            self._track_body_ids = self.robot.find_bodies(TRACKED_LINKS, preserve_order=True)[0]
        return self._track_body_ids

    def ref_track(self, name: str) -> torch.Tensor:
        """Reference ``link_pos_w`` / ``link_quat_w`` / ``link_lin_vel_w`` / ``link_ang_vel_w`` of the 14 tracked bodies, (N, 14, ...)."""
        return getattr(self.motion_reference.reference_frame, name)[:, 0]

    def targets(self, estimated: bool) -> tuple[torch.Tensor, torch.Tensor]:
        """Contact targets in the pelvis yaw frame, (N, 9, 3) with invalid entries zeroed, and the valid mask (N, 9).

        With ``estimated`` the pelvis pose is the delayed and drifted odometry pose (what the policy observes), else the
        true pose."""
        if estimated:
            base_pos, base_quat = self.odometry_sensor.estimated_pos_w, self.odometry_sensor.estimated_quat_w
        else:
            base_pos, base_quat = self.base_pos_w, self.base_quat_w
        pos_b = to_base_yaw_frame(self.ref_target_pos_w, base_pos, base_quat)
        valid = (
            self.ref_target_exists
            & (pos_b[..., 0].abs() <= self.cfg.footprint_x)
            & (pos_b[..., 1].abs() <= self.cfg.footprint_y)
        )
        return pos_b * valid.unsqueeze(-1), valid

    @property
    def command(self) -> torch.Tensor:
        """Observed contact targets (estimated pelvis pose), flattened, (N, 27)."""
        return self.targets(estimated=True)[0].flatten(1)

    def _update_metrics(self) -> None:
        self.metrics["target_valid_frac"][:] = self.targets(estimated=False)[1].float().mean(-1)
        self.metrics["ref_contact_frac"][:] = self.ref_contact_flag.float().mean(-1)

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        pass

    def _update_command(self, env_ids: torch.Tensor | None) -> None:
        pass


@dataclass(kw_only=True)
class AcpGoalCommandCfg(AcpCommandCfg):
    """Reference goal: the reference base pose a random time ahead of the current reference time."""

    lookahead_range_s: tuple[float, float] = field(default_factory=lambda: (0.5, 1.5))
    """The goal is the reference base ``U(lookahead_range_s)`` seconds ahead (capped at the clip end)."""

    min_distance: float = 0.3
    """The planar goal direction is d_xy / max(|d_xy|, min_distance)."""

    def build(self, env: ManagerBasedRlEnv) -> AcpGoalCommand:
        return AcpGoalCommand(self, env)


class AcpGoalCommand(AcpCommandBase):
    """Goal command g = (d_xy / max(|d_xy|, r), wrap(psi_target - psi_robot)) in the robot's base-yaw frame."""

    cfg: AcpGoalCommandCfg

    def __init__(self, cfg: AcpGoalCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self.target_time = torch.zeros(self.num_envs, device=self.device)
        self.target_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.target_yaw = torch.zeros(self.num_envs, device=self.device)
        self.metrics["goal_distance"] = torch.zeros(self.num_envs, device=self.device)

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        # Only called on reset: the new episode starts at the clip start time (the episode length buffer still holds the
        # previous episode at this point).
        self._sample_goal(env_ids, self.motion_buffer.motion_buffer_start_time_s[env_ids])

    def _sample_goal(self, env_ids: torch.Tensor, time: torch.Tensor) -> None:
        sequences = self.motion_buffer._all_motion_sequences
        motion_ids = self.motion_ids[env_ids]
        fps = sequences.framerate[motion_ids].to(self.device).float()
        end_time = (sequences.buffer_length[motion_ids].to(self.device) - 1) / fps
        lookahead = torch.empty(len(env_ids), device=self.device).uniform_(*self.cfg.lookahead_range_s)
        target_time = torch.minimum(time + lookahead, end_time)
        frame = torch.round(target_time * fps).long()
        origin = reference_origin(self.motion_reference)[env_ids]
        self.target_time[env_ids] = target_time
        self.target_pos_w[env_ids] = sequences.link_pos_w[motion_ids, frame][:, self.ref_pelvis_id].to(self.device) + origin
        self.target_yaw[env_ids] = yaw_of_quat(sequences.link_quat_w[motion_ids, frame][:, self.ref_pelvis_id].to(self.device))

    def goal_command(self, estimated: bool) -> torch.Tensor:
        """Goal command, (N, 3), from the estimated (delayed, drifted) or the true pelvis pose."""
        if estimated:
            base_pos, base_quat = self.odometry_sensor.estimated_pos_w, self.odometry_sensor.estimated_quat_w
        else:
            base_pos, base_quat = self.base_pos_w, self.base_quat_w
        yaw = yaw_of_quat(base_quat)
        d_w = self.target_pos_w[:, :2] - base_pos[:, :2]
        cos, sin = torch.cos(yaw), torch.sin(yaw)
        d = torch.stack((cos * d_w[:, 0] + sin * d_w[:, 1], -sin * d_w[:, 0] + cos * d_w[:, 1]), dim=-1)
        direction = d / torch.clamp(d.norm(dim=-1, keepdim=True), min=self.cfg.min_distance)
        heading = wrap_to_pi(self.target_yaw - yaw)
        return torch.cat((direction, heading.unsqueeze(-1)), dim=-1)

    @property
    def command(self) -> torch.Tensor:
        """Goal command of the actor, (N, 3), from the estimated pelvis pose."""
        return self.goal_command(estimated=True)

    def _update_metrics(self) -> None:
        self.metrics["goal_distance"][:] = (self.target_pos_w[:, :2] - self.base_pos_w[:, :2]).norm(dim=-1)

    def _update_command(self, env_ids: torch.Tensor | None) -> None:
        if env_ids is None:
            # ACP-unspecified: keep the remaining lookahead within [lookahead_range_s[0], lookahead_range_s[1]]: the goal is
            # resampled U(lookahead_range_s) ahead of the current reference time when the remaining time drops below
            # the lower bound (capped at the clip end).
            time = self.motion_time
            stale = (self.target_time - time < self.cfg.lookahead_range_s[0]).nonzero().flatten()
            if len(stale) > 0:
                self._sample_goal(stale, time[stale])
