"""Geometric contact of the nine contact links: a differentiable contact probability computed from the (imagined) robot state and the real terrain, so that the
contact rewards of the imagination do not depend on the world model's learned contact head.

For every contact link a set of surface points of its collision geometry is kept (mesh vertices thinned by farthest-point sampling, the end points of the foot
capsules with their radius). A point's clearance is its height above the terrain directly below it (exact ray casts against the terrain geoms:
``ExactHeightScanner.ground_below``) minus the radius of its primitive; the clearance of a link is the minimum over its points, and
``contact = sigmoid(-(clearance - d0) / tau)``. The terrain height is piecewise constant, so the gradient is exact: 1 w.r.t. the point height (through the forward
kinematics to the joints) and 0 w.r.t. the horizontal position. Contacts with vertical faces (a hand against a wall) are not seen by this vertical clearance."""

from __future__ import annotations

import mujoco
import numpy as np
import torch

from instinct_mj.rl.supermimic.rotations import mm, mv


def link_surface_points(mjcf_path: str, links: list[str], points_per_link: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Surface points (L, P, 3) in the link frames and their primitive radii (L, P) for the collision geoms (``contype`` > 0) of the links."""
    model = mujoco.MjModel.from_xml_path(mjcf_path)
    pts_l, rad_l = [], []
    for name in links:
        b = model.body(name).id
        pts, rad = [], []
        for g in np.nonzero((model.geom_bodyid == b) & (model.geom_contype > 0))[0]:
            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, model.geom_quat[g])
            R = R.reshape(3, 3)
            if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                m = model.geom_dataid[g]
                v = model.mesh_vert[model.mesh_vertadr[m] : model.mesh_vertadr[m] + model.mesh_vertnum[m]]
                aabb_c, aabb_h = model.geom_aabb[g, :3], model.geom_aabb[g, 3:]
                assert np.allclose(v.min(0), aabb_c - aabb_h, atol=2e-3) and np.allclose(v.max(0), aabb_c + aabb_h, atol=2e-3), f"{name}: mesh vertices not in the geom frame"
                pts.append(model.geom_pos[g] + v @ R.T), rad.append(np.zeros(len(v)))
            elif model.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE:
                ends = np.array([[0, 0, -model.geom_size[g, 1]], [0, 0, model.geom_size[g, 1]]])
                pts.append(model.geom_pos[g] + ends @ R.T), rad.append(np.full(2, model.geom_size[g, 0]))
            else:
                raise NotImplementedError(f"{name}: geom type {mujoco.mjtGeom(model.geom_type[g]).name}")
        pts, rad = np.concatenate(pts), np.concatenate(rad)
        keep = [int(np.argmin(pts[:, 2]))]  # farthest-point sampling, starting from the lowest point
        dmin = np.linalg.norm(pts - pts[keep[0]], axis=1)
        while len(keep) < min(points_per_link, len(pts)):
            keep.append(int(np.argmax(dmin)))
            dmin = np.minimum(dmin, np.linalg.norm(pts - pts[keep[-1]], axis=1))
        keep = keep + [keep[0]] * (points_per_link - len(keep))  # (a link with fewer points repeats its first)
        pts_l.append(pts[keep]), rad_l.append(rad[keep])
    return torch.tensor(np.stack(pts_l), dtype=torch.float32), torch.tensor(np.stack(rad_l), dtype=torch.float32)


class GeometricContact:
    def __init__(self, scanner, mjcf_path: str, links: list[str], device: str, points_per_link: int = 16, d0: float = 0.01, tau: float = 0.01):
        self.scanner, self.d0, self.tau = scanner, d0, tau
        pts, rad = link_surface_points(mjcf_path, links, points_per_link)
        self.points, self.radius = pts.to(device), rad.to(device)

    def clearance(self, bp: torch.Tensor, bR: torch.Tensor) -> torch.Tensor:
        """(B, L) height of the lowest surface point of every link above the terrain below it (negative: the point is inside the terrain)."""
        w = bp.unsqueeze(2) + mv(bR.unsqueeze(2), self.points.unsqueeze(0))  # (B, L, P, 3)
        B, L, P = w.shape[:3]
        ground = self.scanner.ground_below(w.detach().reshape(B, L * P, 3)).view(B, L, P)
        return (w[..., 2] - ground - self.radius).min(-1).values

    def contact(self, bp: torch.Tensor, bR: torch.Tensor, d0: float | None = None, tau: float | None = None) -> torch.Tensor:
        """(B, L) differentiable contact probability sigmoid(-(clearance - d0) / tau)."""
        return torch.sigmoid(-(self.clearance(bp, bR) - (self.d0 if d0 is None else d0)) / (self.tau if tau is None else tau))
