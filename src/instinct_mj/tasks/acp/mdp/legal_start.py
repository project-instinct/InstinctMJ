"""Legal episode start frames of the ACP tracking task: the frames whose noise-free reference state does not penetrate the terrain.

The reset (``reset_robot_state_by_reference`` with ``max_penetration``) redraws the noise of an env that starts inside the terrain and, after ``max_resample`` draws, falls
back to the noise-free reference state. That fallback is only a clean start if the reference frame itself is clear of the terrain, so the start frames whose reference penetrates
by more than ``max_penetration`` are never used as an episode start: ``LegalStarts`` measures every frame of every clip once (collision detection of the written reference state
against the clip terrain, batches of ``num_envs`` frames) and ``reset_motion_start_phase`` moves a drawn illegal start to the nearest legal frame of the same clip.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers import SceneEntityCfg

from instinct_mj.envs.mdp.events.motion_reference import _write_reference_reset, reference_reset_state, terrain_penetration_depth

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


class LegalStarts:
    """Per-frame reference penetration of all clips and the nearest-legal-frame table (flat over the clips: ``offset[clip] + frame``)."""

    def __init__(self, env: ManagerBasedRlEnv, max_penetration: float, reset_params: dict, motion_ref_cfg=None):
        motion_ref_cfg = motion_ref_cfg or SceneEntityCfg("motion_reference")
        self.max_penetration = max_penetration
        device = env.device
        motion_ref = env.scene[motion_ref_cfg.name]
        buf = next(iter(motion_ref.motion_buffers.values()))
        seq = buf._all_motion_sequences
        length, fps = seq.buffer_length.to(device).long(), seq.framerate.to(device).float()
        self.num_clips = len(length)
        self.offset = torch.cumsum(length, 0) - length
        total = int(length.sum())
        clip = torch.repeat_interleave(torch.arange(self.num_clips, device=device), length)
        frame = torch.arange(total, device=device) - self.offset[clip]
        self.depth = self._measure(env, motion_ref, buf, clip, frame, fps, reset_params)
        self.legal = self.depth <= max_penetration
        self.length = length
        self.clip_names = list(getattr(buf, "_all_motion_files", range(self.num_clips)))
        n_legal = torch.zeros(self.num_clips, device=device).index_add_(0, clip, self.legal.float())
        self.illegal_clips = (n_legal == 0).nonzero().squeeze(-1).tolist()  # no legal start frame at all: reported and never sampled (``snap_starts`` redirects their envs)
        self.dead_clip = n_legal == 0
        self.clip_illegal_rate = 1.0 - n_legal / length
        # nearest legal frame of the same clip (the table of a clip without a legal frame maps every frame to itself)
        idx = torch.arange(total, device=device)
        big = total + 1
        prev = torch.where(self.legal, idx, torch.full_like(idx, -big))
        prev = torch.cummax(prev, 0).values  # last legal flat index at or before (may belong to the previous clip)
        nxt = torch.where(self.legal, idx, torch.full_like(idx, big + total))
        nxt = torch.flip(torch.cummin(torch.flip(nxt, [0]), 0).values, [0])  # first legal flat index at or after
        start, end = self.offset[clip], self.offset[clip] + length[clip]
        prev_ok, nxt_ok = prev >= start, nxt < end
        use_prev = prev_ok & (~nxt_ok | (idx - prev <= nxt - idx))
        self.snap = torch.where(use_prev, prev, torch.where(nxt_ok, nxt, idx))
        self.frame_in_clip = frame

    @torch.no_grad()
    def _measure(self, env, motion_ref, buf, clip, frame, fps, p) -> torch.Tensor:
        """Deepest robot-vs-terrain penetration of the noise-free reference state of every (clip, frame); the state of the env is restored afterwards."""
        asset = env.scene["robot"]
        N = env.num_envs
        sim_data = env.sim.data
        saved = [x.clone() for x in (sim_data.qpos, sim_data.qvel)]
        sel, start, origins = buf._assigned_env_motion_selection.clone(), buf._motion_buffer_start_time_s.clone(), buf._assigned_env_origins.clone()
        out = torch.empty(len(clip), device=env.device)
        for s in range(0, len(clip), N):
            c, f = clip[s : s + N], frame[s : s + N]
            ids = torch.arange(len(c), device=env.device)
            assigned = buf.env_ids_to_assigned_ids(ids).to(buf.buffer_device)
            buf._assigned_env_motion_selection[assigned] = c.to(buf.buffer_device)
            buf._sample_assigned_env_origins(assigned)
            buf._motion_buffer_start_time_s[assigned] = ((f.float() + 0.01) / fps[c]).to(buf.buffer_device)
            ref = reference_reset_state(motion_ref, ids, p["position_offset"], p["dof_vel_ratio"], p["base_lin_vel_ratio"], p["base_ang_vel_ratio"])
            _write_reference_reset(asset, ids, *ref, {}, {}, (0.0, 0.0), p.get("clamp_joint_pos_to_limits", False))
            out[s : s + len(c)] = terrain_penetration_depth(env)[: len(c)]
        buf._assigned_env_motion_selection[:], buf._motion_buffer_start_time_s[:], buf._assigned_env_origins[:] = sel, start, origins
        sim_data.qpos[:], sim_data.qvel[:] = saved
        env.sim.forward()
        return out

    def snap_starts(self, motion_buffer, env_ids: torch.Tensor) -> int:
        """Move the drawn start frame of ``env_ids`` to the nearest legal frame of its clip; returns the number of moved starts."""
        assigned = motion_buffer.env_ids_to_assigned_ids(env_ids)
        clip = motion_buffer.assigned_env_motion_selection[assigned]
        dead = self.dead_clip[clip]
        if dead.any():  # a clip without any legal frame is never an episode: the env takes another clip (by the buffer's clip weights) at a uniform frame, snapped below
            weights = motion_buffer._motion_weights.to(self.dead_clip.device) * (~self.dead_clip)
            new_clip = torch.multinomial(weights, int(dead.sum()), replacement=True)
            sel = assigned[dead]
            motion_buffer._assigned_env_motion_selection[sel] = new_clip.to(motion_buffer.buffer_device)
            motion_buffer._sample_assigned_env_origins(sel)
            frame = (torch.rand(len(sel), device=new_clip.device) * (self.length[new_clip] - 1)).long()
            motion_buffer._motion_buffer_start_time_s[sel] = ((frame.float() + 0.01) / motion_buffer._all_motion_sequences.framerate[new_clip]).to(motion_buffer.buffer_device)
            clip = motion_buffer.assigned_env_motion_selection[assigned]
        fps = motion_buffer._all_motion_sequences.framerate[clip]
        time = motion_buffer.motion_buffer_start_time_s[assigned]
        frame = torch.floor(time * fps).long().clamp(min=0)
        frame = torch.minimum(frame, self.length[clip] - 1)
        flat = self.offset[clip] + frame
        target = self.snap[flat]
        moved = target != flat
        if moved.any():
            new_frame = target - self.offset[clip]
            motion_buffer.motion_buffer_start_time_s[assigned] = torch.where(moved, (new_frame.float() + 0.01) / fps, time)
        return int(moved.sum())

    def report(self, top: int = 20) -> str:
        n = int((~self.legal).sum())
        lines = [f"[legal-starts] {n}/{len(self.legal)} reference frames penetrate the terrain by > {self.max_penetration * 100:.1f} cm ({100 * n / len(self.legal):.2f} %), "
                 f"{len(self.illegal_clips)} clips without a legal start frame"]
        for c in self.illegal_clips:
            lines.append(f"  no legal start: clip {c} {self.clip_names[c]} (deepest frame {float(self.depth[self.offset[c] : self.offset[c] + self.length[c]].max()) * 100:.1f} cm)")
        worst = torch.topk(self.clip_illegal_rate, min(top, self.num_clips))
        for r, c in zip(worst.values.tolist(), worst.indices.tolist()):
            if r > 0:
                lines.append(f"  clip {c} {r * 100:5.1f} % illegal frames: {self.clip_names[c]}")
        return "\n".join(lines)
