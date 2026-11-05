"""Differentiable forward kinematics of the G1 in torch, built from the compiled MJCF (root body: ``torso_link``).

Adapted from the SuperMimic reference implementation (MIT License), Chong Zhang / ETH Zurich.

The same FK computes tracked-body poses and velocities for logged simulator states and for world-model predictions,
so the analytic reward head matches the reward the simulator transitions are labeled with.
"""

from __future__ import annotations

import mujoco
import numpy as np
import torch

from instinct_mj.assets.unitree_g1 import G1_MJCF_PATH

from .rotations import axis_angle_to_mat, mm, mv, quat_to_mat


class G1Kinematics:
    def __init__(self, device: str, joint_names: list[str], tracked: list[str], mjcf_path: str = G1_MJCF_PATH):
        """
        Args:
            joint_names: joint order of the simulator (``robot.joint_names``), used for ``q`` / ``qd``.
            tracked: bodies (names without namespace prefix) returned by :meth:`forward`, in order.
            mjcf_path: the G1 MJCF (its root body is the free-joint body whose pose ``forward`` takes; default: the torso-rooted popsicle).
        """
        m = mujoco.MjModel.from_xml_path(mjcf_path)
        self.device = device
        jname_to_idx = {n: i for i, n in enumerate(joint_names)}
        bodies, parent, joint_idx, axes, offs_p, offs_R = [], [], [], [], [], []
        mj_to_local = {}
        for b in range(1, m.nbody):
            name = m.body(b).name
            if m.body_jntnum[b] == 0:
                continue  # fixed links are not tracked
            jid = m.body_jntadr[b]
            mj_to_local[b] = len(bodies)
            bodies.append(name)
            if m.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
                parent.append(-1)
                joint_idx.append(-1)
            else:
                assert np.allclose(m.jnt_pos[jid], 0.0), f"joint {m.joint(jid).name} is not anchored at its body origin"
                assert m.jnt_type[jid] == mujoco.mjtJoint.mjJNT_HINGE
                parent.append(mj_to_local[m.body_parentid[b]])
                joint_idx.append(jname_to_idx[m.joint(jid).name])
            axes.append(m.jnt_axis[jid].copy())
            offs_p.append(m.body_pos[b].copy())
            offs_R.append(m.body_quat[b].copy())
        self.body_names = bodies
        self.parent = parent
        self.joint_idx = joint_idx
        self.axes = torch.tensor(np.array(axes), dtype=torch.float32, device=device)
        self.off_p = torch.tensor(np.array(offs_p), dtype=torch.float32, device=device)
        self.off_R = quat_to_mat(torch.tensor(np.array(offs_R), dtype=torch.float32, device=device))
        self.tracked_idx = [bodies.index(n) for n in tracked]
        self.num_bodies = len(bodies)
        # Level-batched evaluation: all bodies of the same tree depth are computed with one batched op. Bodies are re-ordered
        # level by level (``pos`` = position of a body in that order); ``parent_pos`` indexes into the running level-ordered state.
        level: list[int] = []
        for i in range(len(bodies)):
            level.append(0 if parent[i] < 0 else level[parent[i]] + 1)
        order = sorted(range(len(bodies)), key=lambda i: level[i])
        pos = {b: k for k, b in enumerate(order)}
        self.levels = []
        for lv in range(1, max(level) + 1):
            idx = [b for b in order if level[b] == lv]
            self.levels.append({
                "off_p": self.off_p[idx], "off_R": self.off_R[idx], "axes": self.axes[idx],
                "parent_pos": torch.tensor([pos[parent[b]] for b in idx], device=device),
                "joint": torch.tensor([joint_idx[b] for b in idx], device=device),
            })  # fmt: skip
        assert order[0] == 0 and level[0] == 0 and sum(1 for l in level if l == 0) == 1, "a single root body is expected"
        self.tracked_pos = [pos[i] for i in self.tracked_idx]

    def forward(self, p, R, v, w, q, qd):
        """Tracked-body world positions, rotations, linear (link-origin) and angular velocities.

        Args:
            p, R, v, w: root (torso_link) position (B,3), rotation (B,3,3), world lin/ang velocity (B,3).
            q, qd: joint positions/velocities (B,29) in simulator joint order.
        """
        P, Rs, V, W = p.unsqueeze(1), R.unsqueeze(1), v.unsqueeze(1), w.unsqueeze(1)  # level-ordered (B, bodies so far, ...)
        for lv in self.levels:
            par, j = lv["parent_pos"], lv["joint"]
            Pp, Rp, Vp, Wp = P[:, par], Rs[:, par], V[:, par], W[:, par]
            pi = Pp + mv(Rp, lv["off_p"])
            R_off = mm(Rp, lv["off_R"])
            Ri = mm(R_off, axis_angle_to_mat(lv["axes"], q[:, j]))
            axis_w = mv(R_off, lv["axes"])
            wi = Wp + axis_w * qd[:, j].unsqueeze(-1)
            vi = Vp + torch.cross(Wp, pi - Pp, dim=-1)
            P, Rs, V, W = torch.cat((P, pi), 1), torch.cat((Rs, Ri), 1), torch.cat((V, vi), 1), torch.cat((W, wi), 1)
        t = self.tracked_pos
        return P[:, t], Rs[:, t], V[:, t], W[:, t]
