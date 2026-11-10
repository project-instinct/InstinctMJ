"""Signed-distance contact of the nine ACP contact bodies (``contact_detect = "sdf"``): one differentiable function for the environment and the imagination.

    d_b = min_i min_{t in [0,1]} ( SDF_terrain(p_b + R_b (a_i + t (b_i - a_i))) - r_i )   (capsule i of body b: the collision capsules / spheres of the model, EXACT, ``body_spheres.segments``)
    c^meas_b = [ d_b < d_on_b ]                                (environment and evaluation, hard)
    c_b      = sigmoid( -(max(d_b, d_floor) - d_on_b) / tau )   (imagination, soft, one-sided: flat below the surface, so penetration is not rewarded)

The SDF of every terrain primitive is convex, hence so is its restriction to a segment: the minimum over the segment is found by a golden-section search (error of micrometres), not by a sphere
discretisation (which missed up to 5.8 mm on the feet). Only mesh collision geoms (``simplified_collision=False``) still use fitted spheres.

The terrain SDF is analytic: the primitive geoms of the terrain body of the compiled model (box, cylinder, capsule, sphere, plane; any orientation; world pose
including the tile translation, so the same function serves every env and every tile of the arena). The geoms are bucketed in a uniform xy grid
(``CELL`` m cells, every geom listed in each cell its bounding box, grown by the reach of the body spheres, touches); a body looks up the cell of its link origin and
evaluates only the K geoms of that cell. The gradient w.r.t. the capsule end points is the analytic SDF gradient at the minimizer (envelope theorem), kept as a custom autograd Function
so that nothing but the (segments, 3) normal and the (segments,) parameter is stored for the backward pass.

The search runs in a NVIDIA Warp kernel, one thread per (segment, geom of the cell) pair with the 24 golden-section iterations in registers (the torch version was bound by kernel launches: ~28 ms
per call at 2048 envs); a second kernel evaluates the analytic unit normal of the chosen geom at the minimizer. Both launch on torch's current CUDA stream, without host synchronisation.
"""

from __future__ import annotations

import mujoco
import numpy as np
import torch
import warp as wp

from . import body_spheres
from .planner import CONTACT_LINKS

wp.init()  # idempotent

CELL = 2.0  # m
PAD_KIND = -1
_KINDS = {int(mujoco.mjtGeom.mjGEOM_PLANE): 0, int(mujoco.mjtGeom.mjGEOM_SPHERE): 1, int(mujoco.mjtGeom.mjGEOM_CAPSULE): 2, int(mujoco.mjtGeom.mjGEOM_CYLINDER): 4, int(mujoco.mjtGeom.mjGEOM_BOX): 6}
PLANE, SPHERE, CAPSULE, CYLINDER, BOX = 0, 1, 2, 4, 6

# Per-body "on" distance [m] (``CONTACT_LINKS`` order), see PLAN.md ("SDF contact"): chosen from the distance distribution of the simulated contacts (|F_z| > 1 N) and the reference labels.
D_ON = (0.002,) * 9
# Bodies that must never touch the terrain (``contact_violation``): the torso with the head (the G1 has no head link). The imagination ends an episode where their distance
# is below VIOLATION_D (the d_on of the contact links, ~ the env's 1 N), with the hazard sigmoid((VIOLATION_D - d) / VIOLATION_TAU).
VIOLATION_LINKS = ("torso_link",)
VIOLATION_D = 0.002  # m
VIOLATION_TAU = 0.002  # m
# Only a contact the reference does not have is a violation: the reference torso must be further than this from the terrain (get-up / crawl frames lie on it).
VIOLATION_REF_CLEARANCE = 0.05  # m
TAU = 0.001  # m, temperature of the soft contact (recalibrated: p = .88 at the surface, .5 at d_on, .02 at 6 mm; see PLAN.md "SDF contact")
D_FLOOR = 0.0  # m, the soft contact does not distinguish depths below the surface (see ``SdfContact.logits``)


_ITERS = 24
_INV_PHI = 0.6180339887498949
_PAD = 1.0e3  # distance of a pad slot


@wp.func
def _sgn(v: float) -> float:
    r = float(0.0)
    if v > 0.0:
        r = 1.0
    elif v < 0.0:
        r = -1.0
    return r


@wp.func
def _sdf_local(l: wp.vec3, sz: wp.vec3, kind: int) -> float:
    """Signed distance of the point l (geom frame) to a primitive of half sizes sz; same formulas (incl. the 1e-12 under the roots) as the torch reference of the tests."""
    out = float(_PAD)
    if kind == 6:  # box
        q = wp.vec3(wp.abs(l[0]) - sz[0], wp.abs(l[1]) - sz[1], wp.abs(l[2]) - sz[2])
        qc = wp.vec3(wp.max(q[0], 0.0), wp.max(q[1], 0.0), wp.max(q[2], 0.0))
        out = wp.sqrt(wp.dot(qc, qc) + 1.0e-12) + wp.min(wp.max(q[0], wp.max(q[1], q[2])), 0.0)
    elif kind == 4:  # cylinder (axis z)
        qr = wp.sqrt(l[0] * l[0] + l[1] * l[1] + 1.0e-12) - sz[0]
        qz = wp.abs(l[2]) - sz[1]
        a = wp.max(qr, 0.0)
        b = wp.max(qz, 0.0)
        out = wp.sqrt(a * a + b * b + 1.0e-12) + wp.min(wp.max(qr, qz), 0.0)
    elif kind == 2:  # capsule (axis z)
        dz = l[2] - wp.max(wp.min(l[2], sz[1]), -sz[1])
        out = wp.sqrt(l[0] * l[0] + l[1] * l[1] + dz * dz + 1.0e-12) - sz[0]
    elif kind == 1:  # sphere
        out = wp.sqrt(wp.dot(l, l) + 1.0e-12) - sz[0]
    elif kind == 0:  # plane z = 0
        out = l[2]
    return out


@wp.func
def _sdf_grad_local(l: wp.vec3, sz: wp.vec3, kind: int) -> wp.vec3:
    """Analytic gradient of ``_sdf_local`` w.r.t. l (a subgradient where the SDF is not differentiable); zero for a pad slot."""
    g = wp.vec3(0.0, 0.0, 0.0)
    if kind == 6:
        q = wp.vec3(wp.abs(l[0]) - sz[0], wp.abs(l[1]) - sz[1], wp.abs(l[2]) - sz[2])
        qc = wp.vec3(wp.max(q[0], 0.0), wp.max(q[1], 0.0), wp.max(q[2], 0.0))
        s = wp.sqrt(wp.dot(qc, qc) + 1.0e-12)
        sg = wp.vec3(_sgn(l[0]), _sgn(l[1]), _sgn(l[2]))
        g = wp.cw_mul(qc / s, sg)
        m = wp.max(q[0], wp.max(q[1], q[2]))
        if m <= 0.0:  # inside: the face of the largest (least negative) q
            if q[0] >= q[1] and q[0] >= q[2]:
                g = g + wp.vec3(sg[0], 0.0, 0.0)
            elif q[1] >= q[2]:
                g = g + wp.vec3(0.0, sg[1], 0.0)
            else:
                g = g + wp.vec3(0.0, 0.0, sg[2])
    elif kind == 4:
        r = wp.sqrt(l[0] * l[0] + l[1] * l[1] + 1.0e-12)
        qr = r - sz[0]
        qz = wp.abs(l[2]) - sz[1]
        a = wp.max(qr, 0.0)
        b = wp.max(qz, 0.0)
        s = wp.sqrt(a * a + b * b + 1.0e-12)
        gr = a / s
        gz = b / s
        if wp.max(qr, qz) <= 0.0:
            if qr >= qz:
                gr = gr + 1.0
            else:
                gz = gz + 1.0
        g = wp.vec3(gr * l[0] / r, gr * l[1] / r, gz * _sgn(l[2]))
    elif kind == 2:
        dz = l[2] - wp.max(wp.min(l[2], sz[1]), -sz[1])
        s = wp.sqrt(l[0] * l[0] + l[1] * l[1] + dz * dz + 1.0e-12)
        g = wp.vec3(l[0], l[1], dz) / s
    elif kind == 1:
        g = l / wp.sqrt(wp.dot(l, l) + 1.0e-12)
    elif kind == 0:
        g = wp.vec3(0.0, 0.0, 1.0)
    return g


@wp.func
def _sdf_at(a: wp.vec3, b: wp.vec3, t: float, p: wp.vec3, R: wp.mat33, sz: wp.vec3, kind: int) -> float:
    """SDF of the geom (pose p, R = geom_xmat, local -> world) at a + t (b - a)."""
    return _sdf_local(wp.transpose(R) * (a + t * (b - a) - p), sz, kind)


@wp.kernel
def _seg_pair_kernel(
    A: wp.array(dtype=wp.vec3),
    B: wp.array(dtype=wp.vec3),
    cell: wp.array(dtype=wp.int32),
    count: wp.array(dtype=wp.int32),
    table: wp.array2d(dtype=wp.int32),
    pos: wp.array(dtype=wp.vec3),
    rot: wp.array(dtype=wp.mat33),
    size: wp.array(dtype=wp.vec3),
    kind: wp.array(dtype=wp.int32),
    d_out: wp.array2d(dtype=float),
    t_out: wp.array2d(dtype=float),
):
    """Thread (i, k): minimum over t in [0, 1] of the SDF of the k-th geom of the cell of segment i along the segment (golden-section search, as the torch reference)."""
    i, k = wp.tid()
    c = cell[i]
    if k >= count[c]:  # pad slot
        d_out[i, k] = _PAD
        t_out[i, k] = 0.5
        return
    g = table[c, k]
    a = A[i]
    b = B[i]
    p = pos[g]
    R = rot[g]
    sz = size[g]
    kd = kind[g]
    inv = float(_INV_PHI)
    lo = float(0.0)
    hi = float(1.0)
    pc = hi - inv * (hi - lo)
    pd = lo + inv * (hi - lo)
    fc = _sdf_at(a, b, pc, p, R, sz, kd)
    fd = _sdf_at(a, b, pd, p, R, sz, kd)
    for _it in range(_ITERS):
        if fc < fd:  # the minimum lies in [lo, pd]
            hi = pd
            pn = hi - inv * (hi - lo)
            fn = _sdf_at(a, b, pn, p, R, sz, kd)
            pd = pc
            fd = fc
            pc = pn
            fc = fn
        else:
            lo = pc
            pn = lo + inv * (hi - lo)
            fn = _sdf_at(a, b, pn, p, R, sz, kd)
            pc = pd
            fc = fd
            pd = pn
            fd = fn
    t = 0.5 * (lo + hi)
    d_out[i, k] = _sdf_at(a, b, t, p, R, sz, kd)
    t_out[i, k] = t


@wp.kernel
def _normal_kernel(
    A: wp.array(dtype=wp.vec3),
    B: wp.array(dtype=wp.vec3),
    t: wp.array(dtype=float),
    geom: wp.array(dtype=wp.int32),
    pos: wp.array(dtype=wp.vec3),
    rot: wp.array(dtype=wp.mat33),
    size: wp.array(dtype=wp.vec3),
    kind: wp.array(dtype=wp.int32),
    x_out: wp.array(dtype=wp.vec3),
    n_out: wp.array(dtype=wp.vec3),
):
    """Minimizer x* = A + t* (B - A) and the unit normal (SDF gradient, world frame) there of the chosen geom."""
    i = wp.tid()
    x = A[i] + t[i] * (B[i] - A[i])
    g = geom[i]
    R = rot[g]
    x_out[i] = x
    n_out[i] = R * _sdf_grad_local(wp.transpose(R) * (x - pos[g]), size[g], kind[g])


class TerrainGrid:
    """The terrain primitives of a compiled model, bucketed in a uniform xy grid."""

    def __init__(self, model: mujoco.MjModel, device: str, reach: float, body: str = "terrain"):
        data = mujoco.MjData(model)
        mujoco.mj_kinematics(model, data)
        bid = model.body(body).id
        ids = [g for g in np.nonzero(model.geom_bodyid == bid)[0] if model.geom_contype[g] or model.geom_conaffinity[g]]
        bad = sorted({mujoco.mjtGeom(model.geom_type[g]).name for g in ids if int(model.geom_type[g]) not in _KINDS})
        if bad:
            raise NotImplementedError(f"the SDF contact supports box / cylinder / capsule / sphere / plane terrain geoms, found {bad} (mesh terrains, e.g. the datasets without geoms.yaml: use contact_detect='force_head')")
        G = len(ids)
        pos = data.geom_xpos[ids].copy()
        rot = data.geom_xmat[ids].reshape(G, 3, 3).copy()
        size = model.geom_size[ids].copy()
        kind = np.array([_KINDS[int(model.geom_type[g])] for g in ids])
        # world-axis half extents (xy bounding box) of every geom
        half = np.zeros((G, 2))
        for i in range(G):
            if kind[i] == BOX:
                half[i] = np.abs(rot[i, :2, :]) @ size[i]
            elif kind[i] == CYLINDER:  # axis z: radius disc + half height along the axis
                half[i] = np.abs(rot[i, :2, 2]) * size[i, 1] + size[i, 0] * np.sqrt(np.clip(1 - rot[i, :2, 2] ** 2, 0, 1))
            elif kind[i] == CAPSULE:
                half[i] = np.abs(rot[i, :2, 2]) * size[i, 1] + size[i, 0]
            elif kind[i] == SPHERE:
                half[i] = size[i, 0]
            else:
                half[i] = np.inf
        self.num_geoms = G
        self.kinds = tuple(sorted(set(int(k) for k in kind)))
        planes = np.nonzero(kind == PLANE)[0]
        finite = np.nonzero(kind != PLANE)[0]
        lo, hi = (pos[finite, :2] - half[finite] - reach), (pos[finite, :2] + half[finite] + reach)
        self.x0, self.y0 = float(lo[:, 0].min()), float(lo[:, 1].min())
        self.nx, self.ny = int(np.ceil((hi[:, 0].max() - self.x0) / CELL)), int(np.ceil((hi[:, 1].max() - self.y0) / CELL))
        cells: dict[int, list[int]] = {}
        for j, i in enumerate(finite):
            ix0, ix1 = int((lo[j, 0] - self.x0) // CELL), int((hi[j, 0] - self.x0) // CELL)
            iy0, iy1 = int((lo[j, 1] - self.y0) // CELL), int((hi[j, 1] - self.y0) // CELL)
            for ix in range(ix0, min(ix1, self.nx - 1) + 1):
                for iy in range(iy0, min(iy1, self.ny - 1) + 1):
                    cells.setdefault(1 + ix * self.ny + iy, []).append(int(i))
        self.K = max([len(v) for v in cells.values()], default=0) + len(planes)
        table = np.full((1 + self.nx * self.ny, max(self.K, 1)), G, dtype=np.int64)  # row 0: outside the grid; G: the pad geom
        for c, v in cells.items():
            table[c, : len(v) + len(planes)] = v + list(planes)
        table[0, : len(planes)] = planes
        self.table = torch.as_tensor(table, device=device)
        self.count = (self.table < G).sum(1)  # geoms per cell (the rows are filled from the front)
        # parameters with the pad geom appended
        f = lambda a, fill: torch.as_tensor(np.concatenate([a, fill], 0), dtype=torch.float32, device=device)  # noqa: E731
        self.pos, self.rot, self.size = f(pos, np.zeros((1, 3))), f(rot, np.eye(3)[None]), f(size, np.ones((1, 3)))
        self.kind = torch.as_tensor(np.concatenate([kind, [PAD_KIND]]), dtype=torch.int64, device=device)
        self._upload()

    def _upload(self):
        """Warp views (zero-copy) of the tensors, with int32 indices; the torch tensors are kept as attributes so that the memory stays alive."""
        self._table32, self._count32, self._kind32 = self.table.int().contiguous(), self.count.int().contiguous(), self.kind.int().contiguous()
        self._pos, self._rot, self._size = self.pos.contiguous(), self.rot.contiguous(), self.size.contiguous()
        self._wp = (
            wp.from_torch(self._count32, dtype=wp.int32),
            wp.from_torch(self._table32, dtype=wp.int32),
            wp.from_torch(self._pos, dtype=wp.vec3),
            wp.from_torch(self._rot, dtype=wp.mat33),
            wp.from_torch(self._size, dtype=wp.vec3),
            wp.from_torch(self._kind32, dtype=wp.int32),
        )

    def cells(self, xy: torch.Tensor) -> torch.Tensor:
        """Cell row of every xy (B, 2) (0: outside the grid)."""
        ix = torch.floor((xy[:, 0] - self.x0) / CELL).long()
        iy = torch.floor((xy[:, 1] - self.y0) / CELL).long()
        inside = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
        return torch.where(inside, 1 + ix * self.ny + iy, torch.zeros_like(ix))

    def gather(self, cell: torch.Tensor, k: int):
        """Geom parameters (pos, rot, size, kind) of the first ``k`` geoms of the given cells: (B, k, ...)."""
        idx = self.table[cell, :k]
        return self.pos[idx], self.rot[idx], self.size[idx], self.kind[idx]


@torch.compiler.disable  # a clean graph break: dynamo never traces into Warp
def _seg_min(A: torch.Tensor, Bq: torch.Tensor, grid: TerrainGrid, cell: torch.Tensor, need_normal: bool):
    """Exact distance of the segments A-B (N, 3) to the min over the geoms of their cell (``grid.table[cell]``): per geom the minimum over t in [0, 1] of the (convex) signed distance of the
    primitive along the segment, found by golden-section search (24 iterations: the segment is located to 1e-5 of its length, i.e. a few micrometres), then the min over the geoms.
    Returns d (N,), (with ``need_normal``) the minimizing point x* (N, 3), the parameter t* (N,), and (with ``need_normal``) the unit normal there (N, 3), the gradient of d w.r.t. x*
    (0 where the nearest slot is a pad); without ``need_normal`` x* and the normal are None."""
    N, K = A.shape[0], grid.table.shape[1]
    A, Bq, cell32 = A.detach().contiguous(), Bq.detach().contiguous(), cell.int().contiguous()
    device = wp.device_from_torch(A.device)
    stream = wp.stream_from_torch(torch.cuda.current_stream(A.device))
    count, table, pos, rot, size, kind = grid._wp
    d_pair, t_pair = A.new_empty(N, K), A.new_empty(N, K)
    wA, wB, wc = wp.from_torch(A, dtype=wp.vec3), wp.from_torch(Bq, dtype=wp.vec3), wp.from_torch(cell32, dtype=wp.int32)
    wp.launch(_seg_pair_kernel, dim=(N, K), inputs=[wA, wB, wc, count, table, pos, rot, size, kind, wp.from_torch(d_pair), wp.from_torch(t_pair)], device=device, stream=stream)
    d, j = d_pair.min(1)
    t = t_pair.gather(1, j[:, None]).squeeze(1)
    if not need_normal:
        return d, None, t, None
    x, g = A.new_empty(N, 3), A.new_empty(N, 3)
    geom = grid._table32[cell, j].contiguous()
    wp.launch(_normal_kernel, dim=N, inputs=[wA, wB, wp.from_torch(t), wp.from_torch(geom, dtype=wp.int32), pos, rot, size, kind, wp.from_torch(x, dtype=wp.vec3), wp.from_torch(g, dtype=wp.vec3)], device=device, stream=stream)
    return d, x, t, g


class _SegMinSdf(torch.autograd.Function):
    """d = min over the geoms and over the segment of the SDF; the gradient w.r.t. the end points is the unit normal at the minimizer (envelope theorem: x* = A + t* (B - A), t* fixed)."""

    @staticmethod
    def forward(ctx, A, Bq, grid, cell):
        need = ctx.needs_input_grad[0] or ctx.needs_input_grad[1]
        d, _, t, g = _seg_min(A, Bq, grid, cell, need)
        if need:
            ctx.save_for_backward(g, t)
        return d

    @staticmethod
    def backward(ctx, grad_out):
        g, t = ctx.saved_tensors
        gg = grad_out.unsqueeze(-1) * g
        return gg * (1 - t).unsqueeze(-1), gg * t.unsqueeze(-1), None, None


@torch.compiler.disable
def _seg_min_diff(A: torch.Tensor, Bq: torch.Tensor, grid: TerrainGrid, cell: torch.Tensor) -> torch.Tensor:
    return _SegMinSdf.apply(A, Bq, grid, cell)


class SdfContact:
    """Exact capsule-SDF contact of the nine contact bodies against the terrain of a compiled model (see the module docstring)."""

    def __init__(self, terrain_model: mujoco.MjModel, device: str, d_on=D_ON, tau: float = TAU, segments: dict | None = None, simplified: bool = True, d_floor: float = D_FLOOR,
                 links: tuple[str, ...] = CONTACT_LINKS):
        segments = segments if segments is not None else body_spheres.segments(simplified, links=links)
        t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32, device=device)  # noqa: E731
        self.seg_a = torch.cat([t(segments[k][0]) for k in links])  # (S, 3), link frame
        self.seg_b = torch.cat([t(segments[k][1]) for k in links])
        self.radii = torch.cat([t(segments[k][2]) for k in links])
        sizes = [len(segments[k][2]) for k in links]
        reach = max(float(torch.maximum(self.seg_a.norm(dim=-1), self.seg_b.norm(dim=-1)).max() + self.radii.max()), 0.0) + 0.05
        self.grid = TerrainGrid(terrain_model, device, reach)
        self.owner = torch.repeat_interleave(torch.arange(len(sizes), device=device), torch.tensor(sizes, device=device))
        ends = np.cumsum([0] + sizes)
        self.bounds = [(int(ends[i]), int(ends[i + 1])) for i in range(len(sizes))]
        self.d_on = torch.as_tensor(d_on, dtype=torch.float32, device=device)
        self.tau = tau
        self.d_floor = d_floor
        self.device = device

    def distance(self, bp: torch.Tensor, bR: torch.Tensor, with_point: bool = False):
        """d_b (B, 9) for the link poses bp (B, 9, 3), bR (B, 9, 3, 3) in the world frame; differentiable w.r.t. both. ``with_point``: also the closest terrain point
        of the nearest capsule (B, 9, 3) (no gradient). All capsules of all bodies are evaluated as one flat batch, each against the geoms of the cell of its body."""
        B = bp.shape[0]
        owner = self.owner
        A = bp[:, owner] + torch.einsum("bsij,sj->bsi", bR[:, owner], self.seg_a)  # (B, S, 3)
        Bq = bp[:, owner] + torch.einsum("bsij,sj->bsi", bR[:, owner], self.seg_b)
        cell = self.grid.cells(bp[:, :, :2].detach().reshape(-1, 2)).view(B, -1)[:, owner]  # (B, S)
        sd, g, xs = self._seg_min_points(A.reshape(-1, 3), Bq.reshape(-1, 3), cell.reshape(-1), with_point)
        sd = sd.view(B, -1)
        v = sd - self.radii
        ds, pts = [], []
        for a, b in self.bounds:
            vb = v[:, a:b]
            if with_point:
                j = vb.argmin(-1, keepdim=True)
                ds.append(vb.gather(1, j).squeeze(1))
                j3 = j.unsqueeze(-1).expand(-1, 1, 3)
                xj, gj = xs.view(B, -1, 3)[:, a:b].gather(1, j3).squeeze(1), g.view(B, -1, 3)[:, a:b].gather(1, j3).squeeze(1)
                sj = sd[:, a:b].gather(1, j).squeeze(1)
                pts.append((xj - sj.unsqueeze(-1) * gj).detach())
            else:
                ds.append(vb.min(-1).values)
        d = torch.stack(ds, 1)
        return (d, torch.stack(pts, 1)) if with_point else d

    def _seg_min_points(self, A: torch.Tensor, Bq: torch.Tensor, cell: torch.Tensor, with_point: bool):
        """min over the geoms of the cell of every segment (N rows) of the SDF along the segment -> (N,) (with ``with_point`` also the normal and the minimizing point (N, 3), no gradient)."""
        if with_point:
            d, x, _, g = _seg_min(A, Bq, self.grid, cell, True)
            return d, g, x
        return _seg_min_diff(A, Bq, self.grid, cell), None, None

    def logits(self, d: torch.Tensor, tau: float | None = None) -> torch.Tensor:
        """Logit of the soft contact probability. One-sided: the distance is floored at ``d_floor`` (default 0 = the surface), so below it the probability is flat and its gradient is zero: the
        surrogate does not keep paying for penetration (a sigmoid of the raw distance would, e.g. p = .65 at the surface, .95 only 12 mm inside)."""
        return -(d.clamp(min=self.d_floor) - self.d_on) / (self.tau if tau is None else tau)

    def flags(self, d: torch.Tensor) -> torch.Tensor:
        return d < self.d_on


def get_sdf_contact(env, tau: float | None = None) -> SdfContact:
    """The SDF contact of a live env (built once, cached on the unwrapped env): the terrain geoms of its compiled model, the fitted body spheres."""
    env = env.unwrapped
    if getattr(env, "_acp_sdf_contact", None) is None:
        env._acp_sdf_contact = SdfContact(env.sim.mj_model, str(env.device), simplified=bool(getattr(env.cfg, "simplified_collision", False)))
    if tau is not None:
        env._acp_sdf_contact.tau = tau
    return env._acp_sdf_contact


def get_violation_sdf(env) -> SdfContact:
    """The exact terrain distance of the bodies that must never touch the terrain (``VIOLATION_LINKS``: the torso, which carries the head), built once per env:
    the imagination's mirror of the ``contact_violation`` termination."""
    env = env.unwrapped
    if getattr(env, "_acp_violation_sdf", None) is None:
        env._acp_violation_sdf = SdfContact(env.sim.mj_model, str(env.device), d_on=(VIOLATION_D,) * len(VIOLATION_LINKS), simplified=bool(getattr(env.cfg, "simplified_collision", False)),
                                            links=VIOLATION_LINKS)
    return env._acp_violation_sdf


def env_contact_distance(env, contact_body_ids, with_point: bool = False):
    """d_b (N, 9) of the live env's robot from its current link poses."""
    from mjlab.utils.lab_api.math import matrix_from_quat

    robot = env.scene["robot"]
    bp = robot.data.body_link_pos_w[:, contact_body_ids]
    bR = matrix_from_quat(robot.data.body_link_quat_w[:, contact_body_ids])
    return get_sdf_contact(env).distance(bp, bR, with_point)
