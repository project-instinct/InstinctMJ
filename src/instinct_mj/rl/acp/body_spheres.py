"""Sphere approximation of the collision geometry of the nine ACP contact bodies (offline fit; used only for the contact decision, never for the simulation).

Every contact body is one link (``CONTACT_LINKS``). Its collision geoms (``contype`` > 0 and owned by exactly that body, the set the ``acp_contact_forces`` sensor
matches: convex mesh hulls of the authors' model, the foot capsules, spheres) are covered by a union of spheres in the link frame, so that

    d_b(x) = min_i ( SDF_terrain(R c_i + p) - r_i )

is the signed distance of the body to the terrain up to the fit error. The fit is a greedy set cover of dense surface samples of the geometry by candidate
spheres that are (almost) inscribed in it (centre ``rho`` below a surface sample, radius = the distance of the centre to the surface + ``out_tol``), so that the union
never pokes out of the geometry by more than ``out_tol`` (outward error) and the surface samples not covered within ``in_tol`` are reported (inward error).
``fit_report`` returns both errors per body. The fitted spheres are cached in ``body_spheres.npz`` next to this file (written by ``scripts/acp/fit_body_spheres.py``)."""

from __future__ import annotations

import hashlib
import os

import mujoco
import numpy as np
from scipy.spatial import ConvexHull, cKDTree

from .planner import CONTACT_LINKS

CACHE = os.path.join(os.path.dirname(__file__), "body_spheres.npz")  # the authors' collision meshes
CACHE_SIMPLIFIED = os.path.join(os.path.dirname(__file__), "body_spheres_simplified.npz")  # ``simplified_collision=True`` (analytic capsules / spheres, the legs' meshes)
SPHERE_BUDGET = {  # the knee/shin hull is long and tapered, the foot is 7 thin capsules, the hand is two small hulls
    "left_knee_link": 56, "right_knee_link": 56, "pelvis": 40, "left_ankle_roll_link": 64, "right_ankle_roll_link": 64,
    "left_elbow_link": 32, "right_elbow_link": 32, "left_wrist_yaw_link": 32, "right_wrist_yaw_link": 32,
}
DEFAULT_BUDGET = 32
OUT_TOL = 0.0005  # m, a sphere may exceed the geometry by this
IN_TOL = 0.004  # m, a mesh-hull surface sample within this of the union counts as covered
IN_TOL_ANALYTIC = 0.0015  # m, the same for the surface samples of the analytic capsules / spheres (the spheres on the axis are exact, only edges between them are cut)


class _Geom:
    """One collision geom in the link frame: ``interior(x)`` = distance of x to the surface (> 0 inside, < 0 outside, exact inside) and surface samples with normals."""

    def __init__(self, model: mujoco.MjModel, g: int, rng: np.random.Generator, n_samples: int = 8000):
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, model.geom_quat[g])
        R, pos, size = R.reshape(3, 3), model.geom_pos[g].copy(), model.geom_size[g].copy()
        self.kind = int(model.geom_type[g])
        if self.kind == mujoco.mjtGeom.mjGEOM_MESH:
            m = model.geom_dataid[g]
            v = model.mesh_vert[model.mesh_vertadr[m] : model.mesh_vertadr[m] + model.mesh_vertnum[m]].astype(np.float64)
            v = pos + v @ R.T  # (the mesh vertices are in the geom frame, see ``geo_contact`` assert)
            hull = ConvexHull(v)
            self.eq = hull.equations  # n . x + d <= 0 inside
            tri = v[hull.simplices]
            area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
            idx = rng.choice(len(tri), n_samples, p=area / area.sum())
            u = rng.random((n_samples, 2))
            flip = u.sum(1) > 1
            u[flip] = 1 - u[flip]
            self.samples = tri[idx, 0] + u[:, :1] * (tri[idx, 1] - tri[idx, 0]) + u[:, 1:] * (tri[idx, 2] - tri[idx, 0])
            self.normals = self.eq[idx, :3]
            self.samples = np.concatenate([self.samples, v[hull.vertices]])
            self.normals = np.concatenate([self.normals, np.zeros((len(hull.vertices), 3))])  # (vertices: normal filled by the nearest sample below)
            nn = cKDTree(self.samples[:n_samples]).query(v[hull.vertices])[1]
            self.normals[n_samples:] = self.normals[nn]
        elif self.kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
            self.r, half = float(size[0]), float(size[1])
            self.a, self.b = pos - half * R[:, 2], pos + half * R[:, 2]
            # side + two hemispheres
            n_side = n_samples // 2
            t, th = rng.uniform(-half, half, n_side), rng.uniform(0, 2 * np.pi, n_side)
            d = np.stack([np.cos(th), np.sin(th), np.zeros(n_side)], 1)
            side = np.stack([np.zeros(n_side), np.zeros(n_side), t], 1) + self.r * d
            cap = rng.normal(size=(n_samples // 2, 3))
            cap /= np.linalg.norm(cap, axis=1, keepdims=True)
            cap_c = np.where(cap[:, 2:3] > 0, half, -half) * np.array([0, 0, 1.0])
            loc = np.concatenate([side, cap_c + self.r * cap])
            nrm = np.concatenate([d, cap])
            self.samples, self.normals = pos + loc @ R.T, nrm @ R.T
        elif self.kind == mujoco.mjtGeom.mjGEOM_SPHERE:
            self.r, self.c = float(size[0]), pos
            nrm = rng.normal(size=(n_samples, 3))
            nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
            self.samples, self.normals = pos + self.r * nrm, nrm
        else:
            raise NotImplementedError(f"geom type {mujoco.mjtGeom(self.kind).name}")

    def candidates(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Exact spheres of an analytic geom: a capsule = spheres of its own radius centred on the axis (zero outward error, at spacing r / 2), a sphere = itself; None for a mesh."""
        if self.kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
            n = max(2, int(np.ceil(np.linalg.norm(self.b - self.a) / (0.5 * self.r))) + 1)
            t = np.linspace(0, 1, n)[:, None]
            return self.a + t * (self.b - self.a), np.full(n, self.r)
        if self.kind == mujoco.mjtGeom.mjGEOM_SPHERE:
            return self.c[None], np.array([self.r])
        return None

    def interior(self, x: np.ndarray) -> np.ndarray:
        if self.kind == mujoco.mjtGeom.mjGEOM_MESH:
            return -(x @ self.eq[:, :3].T + self.eq[:, 3]).max(1)
        if self.kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
            ab = self.b - self.a
            t = np.clip(((x - self.a) @ ab) / (ab @ ab), 0, 1)
            return self.r - np.linalg.norm(x - (self.a + t[:, None] * ab), axis=1)
        return self.r - np.linalg.norm(x - self.c, axis=1)


def body_geoms(model: mujoco.MjModel, link: str, rng: np.random.Generator) -> list[_Geom]:
    b = model.body(link).id
    ids = [g for g in np.nonzero((model.geom_bodyid == b) & (model.geom_contype > 0))[0]]
    return [_Geom(model, int(g), rng) for g in ids]


def fit_body(geoms: list[_Geom], budget: int, rng: np.random.Generator, out_tol: float = OUT_TOL, in_tol: float = IN_TOL, in_tol_analytic: float = IN_TOL_ANALYTIC) -> tuple[np.ndarray, np.ndarray]:
    """Greedy cover of the union surface by near-inscribed spheres. Returns centres (S, 3) and radii (S,) in the link frame."""
    interior = lambda x: np.max([g.interior(x) for g in geoms], axis=0)  # noqa: E731
    S = np.concatenate([g.samples for g in geoms])
    N = np.concatenate([g.normals for g in geoms])
    T = np.concatenate([np.full(len(g.samples), in_tol if g.kind == mujoco.mjtGeom.mjGEOM_MESH else in_tol_analytic) for g in geoms])  # coverage tolerance of every sample
    keep = interior(S) < 0.001  # surface of the union: samples not buried inside another geom
    S, N, T = S[keep], N[keep], T[keep]
    cand_c, cand_r = [], []
    is_mesh = np.concatenate([np.full(len(g.samples), g.kind == mujoco.mjtGeom.mjGEOM_MESH) for g in geoms])[keep]
    mesh_idx = np.nonzero(is_mesh)[0]
    sub = rng.choice(mesh_idx, min(len(mesh_idx), 2500), replace=False) if len(mesh_idx) else mesh_idx
    for depth in (0.002, 0.004, 0.007, 0.011, 0.016, 0.024, 0.035, 0.05):
        c = S[sub] - depth * N[sub]
        r = interior(c)
        ok = r > 0.001
        cand_c.append(c[ok]), cand_r.append(r[ok] + out_tol)
    for g in geoms:  # exact spheres of the analytic geoms
        if (cand := g.candidates()) is not None:
            cand_c.append(cand[0]), cand_r.append(cand[1])
    C, Rr = np.concatenate(cand_c), np.concatenate(cand_r)
    uncovered = np.ones(len(S), bool)
    chosen_c, chosen_r = [], []
    # coverage of a candidate = samples within (r + in_tol) of its centre (the sphere surface is that close)
    tree = cKDTree(S)
    cov = []
    for ix, c, r in zip(tree.query_ball_point(C, Rr + T.max(), workers=-1), C, Rr):
        ix = np.asarray(ix, dtype=int)
        cov.append(ix[(np.linalg.norm(S[ix] - c, axis=1) - r) <= T[ix]] if len(ix) else ix)
    for _ in range(budget):
        if not uncovered.any():
            break
        gains = np.array([uncovered[ix].sum() for ix in cov])
        best = int(gains.argmax())
        if gains[best] == 0:
            break
        uncovered[cov[best]] = False
        chosen_c.append(C[best]), chosen_r.append(Rr[best])
    return np.array(chosen_c), np.array(chosen_r)


def fit_report(model: mujoco.MjModel, link: str, centers: np.ndarray, radii: np.ndarray, rng: np.random.Generator) -> dict:
    """Inward error (surface samples of the geometry outside the union; > 0 = the spheres are inside the geometry there) and outward error (points of the sphere surfaces
    outside the geometry), in metres."""
    geoms = body_geoms(model, link, rng)
    S = np.concatenate([g.samples for g in geoms])
    interior = lambda x: np.max([g.interior(x) for g in geoms], axis=0)  # noqa: E731
    S = S[interior(S) < 0.001]
    d_union = np.min(np.linalg.norm(S[:, None] - centers[None], axis=2) - radii[None], axis=1)
    inward = np.clip(d_union, 0, None)
    P = rng.normal(size=(300, 3))
    P /= np.linalg.norm(P, axis=1, keepdims=True)
    P = (centers[:, None] + radii[:, None, None] * P[None]).reshape(-1, 3)
    d_to_union = np.min(np.linalg.norm(P[:, None] - centers[None], axis=2) - radii[None], axis=1)  # (points on the union surface: ~0; inner ones negative)
    P = P[d_to_union > -1e-6]
    outward = np.clip(-interior(P), 0, None)  # depth outside the geometry (exact for the faces of a hull; the sphere-surface point is at most this far from it)
    return {"n": len(radii), "in_max": float(inward.max()), "in_p99": float(np.quantile(inward, 0.99)), "in_mean": float(inward.mean()), "out_max": float(outward.max()), "out_p99": float(np.quantile(outward, 0.99))}


def fit_all(simplified: bool = False, seed: int = 0) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, dict]]:
    from instinct_mj.assets.acp_g1 import get_acp_g1_spec

    model = get_acp_g1_spec(simplified_collision=simplified).compile()
    fits, reports = {}, {}
    for link in CONTACT_LINKS:
        rng = np.random.default_rng(seed)
        c, r = fit_body(body_geoms(model, link, rng), SPHERE_BUDGET.get(link, DEFAULT_BUDGET), rng)
        fits[link] = (c, r)
        reports[link] = fit_report(model, link, c, r, np.random.default_rng(seed + 1))
    return fits, reports


def segments(simplified: bool = True, seed: int = 0, links: tuple[str, ...] = CONTACT_LINKS) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Exact collision primitives of the nine contact links in the link frame, as capsule segments ``(A, B, r)`` (centre-line end points (S, 3) x2, radius (S,)): a capsule geom is its own
    segment, a sphere geom a degenerate one (A = B). ``sdf_contact`` computes the terrain distance of a segment exactly, so the robot side has no approximation error. Mesh geoms (only in the
    ``simplified_collision=False`` model) have no exact form: they are covered by the fitted spheres (degenerate segments) as before."""
    from instinct_mj.assets.acp_g1 import get_acp_g1_spec

    model = get_acp_g1_spec(simplified_collision=simplified).compile()
    out = {}
    for link in links:
        rng = np.random.default_rng(seed)
        geoms = body_geoms(model, link, rng)
        A, B, R = [], [], []
        for g in geoms:
            if g.kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
                A.append(g.a), B.append(g.b), R.append(g.r)
            elif g.kind == mujoco.mjtGeom.mjGEOM_SPHERE:
                A.append(g.c), B.append(g.c), R.append(g.r)
        meshes = [g for g in geoms if g.kind == mujoco.mjtGeom.mjGEOM_MESH]
        if meshes:
            c, r = fit_body(meshes, SPHERE_BUDGET.get(link, DEFAULT_BUDGET), rng)
            A += list(c)
            B += list(c)
            R += list(r)
        out[link] = (np.array(A), np.array(B), np.array(R))
    return out


def cache_path(simplified: bool) -> str:
    return CACHE_SIMPLIFIED if simplified else CACHE


def signature(model: mujoco.MjModel) -> str:
    """Hash of the collision geoms (type, size, pose, mesh vertices) of the nine contact links of a compiled model: the cached spheres are valid for exactly this geometry."""
    h = hashlib.sha1()
    for link in CONTACT_LINKS:
        b = model.body(link).id
        for g in np.nonzero((model.geom_bodyid == b) & (model.geom_contype > 0))[0]:
            h.update(np.array([model.geom_type[g]], dtype=np.int64).tobytes())
            for a in (model.geom_size[g], model.geom_pos[g], model.geom_quat[g]):
                h.update(np.round(a, 6).tobytes())
            if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                m = model.geom_dataid[g]
                h.update(np.round(model.mesh_vert[model.mesh_vertadr[m] : model.mesh_vertadr[m] + model.mesh_vertnum[m]], 5).tobytes())
    return h.hexdigest()


def save(fits: dict, simplified: bool = False, sig: str = "") -> None:
    np.savez(cache_path(simplified), **{f"{k}/c": v[0] for k, v in fits.items()}, **{f"{k}/r": v[1] for k, v in fits.items()}, signature=np.array(sig))


def load(simplified: bool = False) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """The cached spheres of the model with / without ``simplified_collision``; refitted (and the cache rewritten) when the collision geoms of the compiled model changed since they were fitted
    (the geometry is read from the compiled model, nothing is hard-coded)."""
    from instinct_mj.assets.acp_g1 import get_acp_g1_spec

    path = cache_path(simplified)
    sig = signature(get_acp_g1_spec(simplified_collision=simplified).compile())
    if os.path.isfile(path):
        z = np.load(path)
        if "signature" in z and str(z["signature"]) == sig:
            return {k: (z[f"{k}/c"], z[f"{k}/r"]) for k in CONTACT_LINKS}
    print(f"[body_spheres] collision geometry changed (or no cache): refitting the spheres of the {'simplified' if simplified else 'mesh'} model", flush=True)
    fits, _ = fit_all(simplified)
    try:
        save(fits, simplified, sig)
    except OSError:
        pass
    return fits
