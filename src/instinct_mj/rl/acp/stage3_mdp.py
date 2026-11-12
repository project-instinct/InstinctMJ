"""Analytic MDP of the ACP Stage 3 task (``Instinct-ACP-Stage3-G1-v1``): the goal-reaching terms next to the tracking terms of ``mdp.py``.

``ACPStage3MDP`` adds to ``ACPMDP`` what the goal-reaching environments need in imagination and for the labels of the real transitions:
the goal rewards of paper Table 3 (``move2goal``, ``standatgoal``, ``flat_orientation``, ``ang_vel_xy``, ``slippage``; ``stumbling`` needs
contact forces, which the world model does not predict: on real transitions it is the measured count, in imagination the expected count of a
learned ``StumbleHead``), the fall termination, the goal
command with the radius 0.5 m, and the goal-reaching critic features (the Vfs input: proprioceptive state, goal command, horizontal
distance, vertical offset, elapsed time over the horizon, the planner's contacts, the full height scan, the DR context).

The goal rewards are indicator functions in the paper. In imagination they are differentiated through, so each indicator ``1[x > 0]`` is
relaxed to ``sigmoid(x / tau)`` (``hard=False``; ACP-unspecified temperatures, ``ACPStage3Cfg.goal_soft_tau_*``); ``hard=True`` evaluates the
indicators exactly, as the env's reward terms do (parity check on real states).

The terrain heights come from a ``RegionHeightScanner``: one dense heightmap for the reference-anchored terrains (the compact grid of
motion-matched tiles) and one for the arena of procedural terrains; a dense map over their joint bounding box would be ten times larger.
"""

from __future__ import annotations

import copy

import mujoco
import numpy as np
import torch

from instinct_mj.rl.supermimic.mdp import invariant_features
from instinct_mj.rl.supermimic.terrain import HeightScanner, load_or_build_heightmap
from instinct_mj.tasks.acp.goal_terrains import ARENA_GEOM_PREFIX, FALL_MARGIN
from instinct_mj.tasks.acp.mdp.goal import FEET

from .kinematics import ACPKinematics
from .rays import ExactHeightScanner
from .mdp import ACPMDP, INV_DIM, SCAN_NX, SCAN_NY, heading, wrap_to_pi
from .motion import NUM_CONTACT, PELVIS, ACPMotion
from .planner import TRACKED_LINKS
from .sim import CONTACTS_COMMAND, make_kinematics, HEIGHT_SCANNER_MAX_DISTANCE, HEIGHT_SCANNER_RESOLUTION, HEIGHT_SCANNER_SIZE

REF_ONLY_TERMS = (
    "track_base_pos", "track_base_vel", "track_joint_pos", "track_joint_vel", "track_contact_label", "track_contact_pos_base",
    "track_contact_pos_global", "track_planned_contact", "track_body_pos_anchor", "track_body_ori", "track_body_lin_vel", "track_body_ang_vel",
)  # fmt: skip
"""Reward terms of the reference-following envs (masked off on the goal-reaching envs; the planned-contact term is on all envs when
evaluated with the planner targets)."""
GOAL_TERMS = ("move2goal", "standatgoal", "flat_orientation", "stumbling", "slippage", "ang_vel_xy")
"""Reward terms of the goal-reaching envs (masked off on the reference envs)."""


# ----------------------------------------------------------------------------------------------------------------------------------
# heightmaps
# ----------------------------------------------------------------------------------------------------------------------------------
def _model_with_terrain_geoms(mj_model: mujoco.MjModel, keep_arena: bool) -> mujoco.MjModel:
    """Copy of the model in which the arena geoms (``keep_arena``) or the other terrain geoms are moved to the world body: only the
    remaining ones belong to the ``terrain`` body that the heightmap builder casts rays against."""
    m = copy.deepcopy(mj_model)
    terrain = m.body("terrain").id
    names = np.array([mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or "" for i in range(m.ngeom)])
    arena = np.char.startswith(names, ARENA_GEOM_PREFIX)
    drop = (m.geom_bodyid == terrain) & (~arena if keep_arena else arena)
    m.geom_bodyid[drop] = 0
    return m


def _region_ground_height(self, xy: torch.Tensor) -> torch.Tensor:
    out = HeightScanner.ground_height(self, xy)
    for region, b in zip(self.extra, self.bounds):
        inside = (xy[..., 0] >= b[0]) & (xy[..., 0] <= b[1]) & (xy[..., 1] >= b[2]) & (xy[..., 1] <= b[3])
        out = torch.where(inside, region.ground_height(xy), out)
    return out


class RegionHeightScanner(HeightScanner):
    """``HeightScanner`` over several dense heightmaps of disjoint regions (the first is the default and clamps at its border); a point
    inside the bounds of a later region reads that region's map."""

    def __init__(self, regions: list[tuple[np.ndarray, float, float]], resolution: float, size, pattern_resolution: float, max_distance: float, device: str):
        super().__init__(*regions[0], resolution, size, pattern_resolution, max_distance, device)
        self.extra = [HeightScanner(h, x0, y0, resolution, size, pattern_resolution, max_distance, device) for h, x0, y0 in regions[1:]]
        self.bounds = [
            torch.tensor([x0, x0 + resolution * (h.shape[1] - 1), y0, y0 + resolution * (h.shape[0] - 1)], device=device) for h, x0, y0 in regions[1:]
        ]

    ground_height = _region_ground_height


class ExactRegionHeightScanner(ExactHeightScanner):
    """``ExactHeightScanner`` (exact ray-cast value, height-map gradient) whose height-map gradient reads the region maps."""

    ground_height = _region_ground_height


def build_stage3_mdp(env, heightmap_resolution: float, heightmap_cache_dir: str, device: str, cfg: dict, exact_scan: bool = False):
    """FK, motion library, two-region heightmap scanner and the analytic Stage 3 MDP for the live (unwrapped) env."""
    manager = env.scene["motion_reference"]
    motion = ACPMotion(manager, env.command_manager.get_term(CONTACTS_COMMAND), device)
    joints = list(env.scene["robot"].joint_names)
    kin = make_kinematics(env, manager, motion, device)
    kin_torso = ACPKinematics(device, joints, ["torso_link"], manager.cfg.robot_model_path)
    metadata_yaml = next(iter(manager.motion_buffers.values())).cfg.metadata_yaml
    model = env.sim.mj_model
    # the heightmap of the reference-anchored terrains is the one of the Stage 1 / 2 runs (same terrain geoms: shared cache)
    regions = [
        load_or_build_heightmap(_model_with_terrain_geoms(model, keep_arena=False), metadata_yaml, heightmap_resolution, heightmap_cache_dir),
        load_or_build_heightmap(_model_with_terrain_geoms(model, keep_arena=True), None, heightmap_resolution, heightmap_cache_dir),
    ]
    scanner = RegionHeightScanner(regions, heightmap_resolution, HEIGHT_SCANNER_SIZE, HEIGHT_SCANNER_RESOLUTION, HEIGHT_SCANNER_MAX_DISTANCE, device)
    if exact_scan:
        scanner = ExactRegionHeightScanner(env, scanner)
    return motion, kin, scanner, ACPStage3MDP(kin, kin_torso, motion, scanner, env, env.step_dt, device, cfg)


# ----------------------------------------------------------------------------------------------------------------------------------
# MDP
# ----------------------------------------------------------------------------------------------------------------------------------
GOAL_CRITIC_EXTRA = 3 + 1 + 1 + 1 + 3 * NUM_CONTACT + NUM_CONTACT
"""Goal command (3), horizontal distance, vertical offset, elapsed time, planned contact targets (27) and their validity (9)."""


class ACPStage3MDP(ACPMDP):
    def __init__(self, kin: ACPKinematics, kin_torso: ACPKinematics, motion: ACPMotion, scanner, env, dt: float, device: str, cfg: dict):
        super().__init__(kin, motion, scanner, env, dt, device)
        self.kin_torso = kin_torso
        goal = env.command_manager.get_term("acp_goal")
        self.tables = goal.tables
        self.goal_radius = goal.cfg.goal_min_distance
        self.goal_episode_s = goal.cfg.goal_episode_s
        self.tau = dict(dist=cfg["goal_soft_tau_dist"], speed=cfg["goal_soft_tau_speed"], height=cfg["goal_soft_tau_height"], angle=cfg["goal_soft_tau_angle"])
        gw = env.reward_manager.get_term_cfg
        self.gp = {n: gw(n).params for n in ("move2goal", "standatgoal", "slippage")}
        self.feet = torch.tensor(list(FEET), device=device)  # (a device tensor: indexing with a Python list copies the indices from the host, not allowed inside a CUDA graph capture)
        # the torso's rotation / angular velocity come out of the tracked-body FK when the motion tracks the torso (rows of ``kin.forward_all``), else of ``kin_torso``
        self.torso_row = TRACKED_LINKS.index("torso_link") if "torso_link" in TRACKED_LINKS[: motion.track_p.shape[1]] else None
        self.down = torch.tensor([0.0, 0.0, -1.0], device=device)  # (a constant: no host-to-device copy inside a CUDA graph capture)

    # ---------------- goal command / geometry ----------------
    def goal_command(self, base_p, base_R, goal_pos, goal_yaw) -> torch.Tensor:
        """Noise-free goal command of a goal-reaching env (``AcpStage3GoalCommand.goal_command``): radius 0.5 m."""
        yaw = heading(base_R)
        d_w = goal_pos[:, :2] - base_p[:, :2]
        c, s = torch.cos(yaw), torch.sin(yaw)
        d = torch.stack((c * d_w[:, 0] + s * d_w[:, 1], -s * d_w[:, 0] + c * d_w[:, 1]), -1)
        direction = d / torch.clamp(d.norm(dim=-1, keepdim=True), min=self.goal_radius)
        return torch.cat((direction, wrap_to_pi(goal_yaw - yaw).unsqueeze(-1)), -1)

    def _gate(self, x: torch.Tensor, tau: float, hard: bool) -> torch.Tensor:
        return (x > 0).float() if hard else torch.sigmoid(x / tau)

    # ---------------- rewards ----------------
    def goal_terms(self, S, contact, contact_prev, goal_pos, goal_yaw, tile, hard: bool, stumbling: torch.Tensor | None = None, fk=None) -> dict[str, torch.Tensor]:
        """Weighted goal terms (per row, ``dt`` not applied) of the transitions ending in ``S`` for goal envs on ``tile`` (rows,).
        ``contact`` / ``contact_prev``: c_meas of the nine links now and one control step earlier (probabilities in imagination),
        ``stumbling``: the count of stumbling links (measured on real transitions, the head's expected count in imagination; None: 0).
        ``fk``: ``kin.forward_all(*S)`` when the caller has it already (the imagination does; the torso then comes out of it too)."""
        T, tau, w = self.tables, self.tau, self.w
        if fk is None:
            bp, bR, bv, bw = self.kin.forward(*S)
            _, tor_R, _, tor_w = (x[:, 0] for x in self.kin_torso.forward(*S))
        else:
            bp, bR, bv, bw = (x[:, :NUM_CONTACT] for x in fk)
            if self.torso_row is None:
                _, tor_R, _, tor_w = (x[:, 0] for x in self.kin_torso.forward(*S))
            else:
                tor_R, tor_w = fk[1][:, self.torso_row], fk[3][:, self.torso_row]
        pel_p, pel_R, pel_v = bp[:, PELVIS], bR[:, PELVIS], bv[:, PELVIS]
        d = goal_pos[:, :2] - pel_p[:, :2]
        dist = torch.sqrt(d.square().sum(-1) + 1e-10)
        dz = goal_pos[:, 2] - pel_p[:, 2]
        dpsi = wrap_to_pi(goal_yaw - heading(pel_R))
        v_xy = pel_v[:, :2]
        speed = torch.sqrt(v_xy.square().sum(-1) + 1e-10)
        toward = (v_xy * d).sum(-1) / dist
        cap, getup = T.v_max[tile], T.getup[tile]
        p = self.gp["move2goal"]
        g_xy = self._gate(p["radius"] - dist, tau["dist"], hard)
        g_v = self._gate(toward - p["min_speed"], tau["speed"], hard) * self._gate(cap - speed, tau["speed"], hard)
        sign = torch.sign(dz) if hard else torch.tanh(dz / tau["height"])
        g_rise = self._gate(pel_v[:, 2] * sign - p["z_speed"], tau["speed"], hard)
        g_near = self._gate(p["z_tolerance"] - dz.abs(), tau["height"], hard)
        g_z = torch.where(getup, 1.0 - (1.0 - g_rise) * (1.0 - g_near), torch.ones_like(g_rise))
        t = {"move2goal": (1.0 - g_xy) * g_v + g_xy * g_z}
        p = self.gp["standatgoal"]
        g_h = self._gate(p["z_tolerance"] - dz.abs(), tau["height"], hard)
        g_psi = self._gate(p["heading_tolerance"] - dpsi.abs(), tau["angle"], hard)
        g_xy_s = self._gate(p["radius"] - dist, tau["dist"], hard)
        d_foot = 1.0 - contact[:, self.feet].mean(-1)
        d_tilt = 1.0 - pel_R[:, 2, 2].square()  # vertical component of the projected gravity of the pelvis is -R_zz
        d_q = (S[4][:, self.act_ids] - self.default_q[self.act_ids]).abs().mean(-1)
        t["standatgoal"] = g_xy_s * g_h * g_psi * torch.exp(-(d_foot + d_tilt + d_q) / 3.0)
        t["flat_orientation"] = tor_R[:, 2, :2].square().sum(-1) * T.foot[tile].float()  # gravity in the torso frame: -R_t[2, :]
        t["ang_vel_xy"] = tor_w[:, :2].square().sum(-1) * T.foot[tile].float()
        p = self.gp["slippage"]
        fast = self._gate(bv.norm(dim=-1) - p["max_speed"], tau["speed"], hard)  # (rows, 9)
        t["slippage"] = (contact * contact_prev * fast).sum(-1) * T.slip[tile].float()  # contact for two control steps = more than 0.026 s
        t["stumbling"] = (torch.zeros_like(dist) if stumbling is None else stumbling) * T.stumble[tile].float()
        return {k: v * w[k] for k, v in t.items()}

    def reward_mixed(
        self, S, qd0, f, phase, origin, a, a_prev, a_prev2, xi, contact, contact_prev, lp_hist, ep0, H: int, is_goal: torch.Tensor,
        goal_pos, goal_yaw, tile, planned=None, eta_rew=None, hard: bool = False, stumbling=None, fk=None,
    ):  # fmt: skip
        """Per-step reward (weights and ``dt`` applied) of the Stage 3 task for a batch of mixed rows: the tracking terms on the reference
        rows, the goal terms on the goal rows, the regularizers on all (``ACPMDP.reward`` + ``goal_terms``). Returns (reward, terms). ``fk``: ``kin.forward_all(*S)`` when the caller has it already."""
        _, t = super().reward(S, qd0, f, phase, origin, a, a_prev, a_prev2, xi, contact, lp_hist, ep0, H, planned, eta_rew, terms=True, fk=fk)
        ref, gol = (~is_goal).float(), is_goal.float()
        # ``track_planned_contact`` applies to all envs with the planner targets (``planned``); the env itself only knows the reference targets
        # (``planned is None``): there it is evaluated on the reference envs only
        for k in REF_ONLY_TERMS:
            if planned is None or k != "track_planned_contact":
                t[k] = t[k] * ref
        for k, v in self.goal_terms(S, contact, contact_prev, goal_pos, goal_yaw, tile, hard, stumbling, fk).items():
            t[k] = v * gol
        return sum(t.values()) * self.dt, t

    # ---------------- termination ----------------
    def goal_termination(self, S, tile, tz: float = 0.02, tt: float = 0.04, hard: bool = False, tilt_threshold: float = 0.8, clearance: float = 0.3, fk=None):
        """Fall of a goal env (``goal_fall``): the pelvis below the lowest walkable surface of its tile, or, on the foot terrains, tilted
        beyond ``tilt_threshold`` or lower than ``clearance`` above the surface. Soft (probability) unless ``hard``. ``fk``: ``kin.forward_all(*S)`` when the caller has it already."""
        T = self.tables
        bp, bR = (self.kin.forward(*S) if fk is None else fk)[:2]
        pel_p, pel_R = bp[:, PELVIS], bR[:, PELVIS]
        z_low = T.z_low[tile] + T.origin[tile][:, 2] - FALL_MARGIN
        tilt = (-pel_R[:, 2, :] - self.down).norm(dim=-1)
        clr = pel_p[:, 2] - self.scanner.ground_height(pel_p[:, None, :2]).squeeze(-1)
        foot = T.foot[tile].float()
        if hard:
            return ((pel_p[:, 2] < z_low) | ((foot > 0) & ((tilt > tilt_threshold) | (clr < clearance)))).float()
        alive = (1 - torch.sigmoid((z_low - pel_p[:, 2]) / tz)) * (1 - foot * torch.sigmoid((tilt - tilt_threshold) / tt)) * (1 - foot * torch.sigmoid((clearance - clr) / tz))
        return 1 - alive

    # ---------------- value bound ----------------
    def max_step_reward(self, eta_rew: float, names=None, goal: bool = False) -> float:
        """Largest per-step reward of the objective of one critic. Reference critic Vm (``goal=False``): the tracking terms of ``ACPMDP.reward`` (the goal terms are not in its
        return). Goal critic Vfs (``goal=True``, ``reward_mixed`` on goal rows): the goal terms, the regularizers and the planned-contact term (``planned`` targets); the other
        reference-only terms are masked off."""
        if names is None:
            names = [n for n in self.w if n not in GOAL_TERMS] if not goal else [n for n in self.w if n in GOAL_TERMS or n not in REF_ONLY_TERMS or n == "track_planned_contact"]
        return super().max_step_reward(eta_rew, names)

    # ---------------- critic ----------------
    def goal_critic_features(self, S, goal_pos, goal_yaw, t_norm, planned_b, planned_valid, xi_n, fk=None) -> torch.Tensor:
        """Input of Vfs (privileged, noise-free): invariant state (tilt, body-frame velocities, joints), goal command g, horizontal
        distance d_g, vertical offset dz = z_goal - z_base, elapsed time over the horizon, the planner's predicted contacts (targets in the
        base-yaw frame and validity), the full critic scan (``critic_scan``) of the true pose and the DR context. ``fk``: ``kin.forward_all(*S)`` when the caller has it already."""
        bp, bR = (self.kin.forward(*S) if fk is None else fk)[:2]
        pel_p, pel_R = bp[:, PELVIS], bR[:, PELVIS]
        dist = (goal_pos[:, :2] - pel_p[:, :2]).norm(dim=-1, keepdim=True)
        dz = (goal_pos[:, 2] - pel_p[:, 2]).unsqueeze(-1)
        return torch.cat((invariant_features(S), self.goal_command(pel_p, pel_R, goal_pos, goal_yaw), dist, dz, t_norm.unsqueeze(-1),
                          planned_b.flatten(1), planned_valid.float(), self.critic_scan(pel_p, pel_R), xi_n), -1)  # fmt: skip

    def goal_critic_dim(self) -> int:
        return INV_DIM + GOAL_CRITIC_EXTRA + self.critic_scan_dim + self.xi_dim
