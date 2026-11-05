"""``G1Kinematics`` with fixed (joint-less) tracked bodies such as ``left_rubber_hand`` / ``right_rubber_hand``.

``G1Kinematics`` only evaluates bodies that own a joint. A fixed tracked body is evaluated here from its nearest jointed
ancestor through the constant offset (position / rotation) composed along the chain of fixed bodies in between; its velocity
is rigid-body transported from the ancestor (``v = v_a + w_a x (p - p_a)``, ``w = w_a``).
"""

from __future__ import annotations

import mujoco
import numpy as np
import torch

from instinct_mj.rl.supermimic.kinematics import G1Kinematics
from instinct_mj.rl.supermimic.rotations import mm, mv


class ACPKinematics(G1Kinematics):
    def __init__(self, device: str, joint_names: list[str], tracked: list[str], mjcf_path: str):
        """``tracked``: any bodies of the MJCF ``mjcf_path`` (names without namespace prefix), returned by :meth:`forward` in order; the root
        body of the MJCF is the pelvis (the free-joint body)."""
        m = mujoco.MjModel.from_xml_path(mjcf_path)
        anchors: list[str] = []  # jointed bodies evaluated by the base class
        self._spec: list[tuple[int, torch.Tensor | None, torch.Tensor | None]] = []  # (anchor index, offset p, offset R)
        for name in tracked:
            b = m.body(name).id
            off_p, off_R = np.zeros(3), np.eye(3)
            while m.body_jntnum[b] == 0:  # compose the offset up to the nearest jointed ancestor
                R_b = np.zeros(9)
                mujoco.mju_quat2Mat(R_b, m.body_quat[b])
                off_p = m.body_pos[b] + R_b.reshape(3, 3) @ off_p
                off_R = R_b.reshape(3, 3) @ off_R
                b = m.body_parentid[b]
            anchor = m.body(b).name
            if anchor not in anchors:
                anchors.append(anchor)
            fixed = anchor != name
            self._spec.append((
                anchors.index(anchor),
                torch.tensor(off_p, dtype=torch.float32, device=device) if fixed else None,
                torch.tensor(off_R, dtype=torch.float32, device=device) if fixed else None,
            ))
        super().__init__(device, joint_names, anchors, mjcf_path)

    def forward(self, p, R, v, w, q, qd):
        """Positions (B, K, 3), rotations (B, K, 3, 3), link-origin velocities and angular velocities (B, K, 3) of the tracked
        bodies in the order of ``tracked``."""
        Pa, Ra, Va, Wa = super().forward(p, R, v, w, q, qd)
        P, Rs, V, W = [], [], [], []
        for i, off_p, off_R in self._spec:
            if off_p is None:
                P.append(Pa[:, i]), Rs.append(Ra[:, i]), V.append(Va[:, i]), W.append(Wa[:, i])
            else:
                r = mv(Ra[:, i], off_p)
                P.append(Pa[:, i] + r), Rs.append(mm(Ra[:, i], off_R)), V.append(Va[:, i] + torch.cross(Wa[:, i], r, dim=-1)), W.append(Wa[:, i])
        return torch.stack(P, 1), torch.stack(Rs, 1), torch.stack(V, 1), torch.stack(W, 1)
