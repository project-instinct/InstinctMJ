"""Reads the simulator state / context of every env from the live mjlab ACP tracking env."""

from __future__ import annotations

import torch

from instinct_mj.rl.supermimic.sim import SimReader
from instinct_mj.rl.supermimic.terrain import HeightScanner, load_or_build_heightmap
from instinct_mj.tasks.acp.mdp.rewards import measured_contact_flags

from .kinematics import ACPKinematics
from .mdp import NUM_J, XI_ARM, XI_BIAS, XI_COM, XI_DIM, XI_FRIC, XI_KD, XI_KP, XI_LAG, XI_MASS, XI_STATIC, XI_STATIC_DIM, ACPMDP
from .motion import ACPMotion
from .rays import ExactHeightScanner
from .planner import CONTACT_LINKS, TRACKED_LINKS

CONTACTS_COMMAND = "acp_reference_contacts"
GOAL_COMMAND = "acp_goal"
# The task's ``height_scanner``: GridPattern 1.6 x 1.0 m @ 0.1 m, max_distance 5 m (planner scan, 17 x 11).
HEIGHT_SCANNER_SIZE = (1.6, 1.0)
HEIGHT_SCANNER_RESOLUTION = 0.1
HEIGHT_SCANNER_MAX_DISTANCE = 5.0


class ACPSimReader(SimReader):
    """``SimReader`` (exact ``qpos`` / ``qvel`` state) with the ACP quantities: measured contacts, the goal target
    and the domain-randomization context. ``SimReader.__init__`` reads the perceptive shadowing terms of the env, so the
    state indices are set here."""

    def __init__(self, env):
        self.env = env
        self.robot = env.scene["robot"]
        idx = self.robot.indexing
        self.free_q = idx.free_joint_q_adr.long()
        self.free_v = idx.free_joint_v_adr.long()
        self.joint_q = idx.joint_q_adr.long()
        self.joint_v = idx.joint_v_adr.long()
        self.contacts = env.command_manager.get_term(CONTACTS_COMMAND)
        self.goal_term = env.command_manager.get_term(GOAL_COMMAND)
        self.contact_threshold = env.reward_manager.get_term_cfg("track_contact_label").params["force_threshold"]
        self.has_delay = all(actuator._delay_buffer is not None for actuator in self.robot.actuators)

    def contact(self) -> torch.Tensor:
        """c_meas (N, 9) in ``CONTACT_LINKS`` order: max over the sensor history window of |F_z| > threshold (the reward's definition)."""
        return measured_contact_flags(self.env, CONTACTS_COMMAND, self.contact_threshold).float()

    def goal(self, fps: float) -> torch.Tensor:
        """Goal target (N, 5): world position of the reference pelvis, heading, goal time in clip-local frames."""
        g = self.goal_term
        return torch.cat((g.target_pos_w, g.target_yaw.unsqueeze(-1), (g.target_time * fps).unsqueeze(-1)), -1)

    def dynamic_context(self) -> torch.Tensor:
        """Per-episode context (N, NUM_J + 1): encoder bias of the joints and the action delay in physics steps (one lag per env, shared by
        all actuator groups: ``SharedDelayPdActuatorCfg``)."""
        bias = torch.zeros(self.env.num_envs, NUM_J, device=self.env.device)
        ids = self.env.action_manager.get_term("joint_pos").target_ids
        bias[:, ids] = self.robot.data.encoder_bias[:, ids]
        lag = self.robot.actuators[0]._delay_buffer.current_lags.float() if self.has_delay else torch.zeros_like(bias[:, 0])
        return torch.cat((bias, lag.unsqueeze(-1)), -1)

    def odometry(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Odometry error of the episode: world drift (N, 3) and delay [s] (N,) of the estimated pelvis pose."""
        sensor = self.env.scene["height_scanner"]
        return sensor.odometry_drift_w.clone(), sensor.odometry_delay_s.clone()

    def static_context(self, mdp: ACPMDP) -> torch.Tensor:
        """Startup domain randomization of every env (N, XI_STATIC_DIM), see ``mdp.XI_*``: torso CoM offset, kp / kd scales, masses of the torso and
        the leg links over nominal, armature scale, friction."""
        env, robot = self.env, self.robot
        model, mj = env.sim.model, env.sim.mj_model
        dev = env.device
        xi = torch.zeros(env.num_envs, XI_STATIC_DIM + NUM_J + 1, device=dev)  # laid out as ``make_xi`` with zero bias / lag, cut below
        torso = int(robot.indexing.body_ids[robot.body_names.index("torso_link")])
        xi[:, XI_COM] = model.body_ipos[:, torso] - torch.as_tensor(mj.body_ipos[torso], device=dev)
        for act in robot.actuators:
            m_ = len(act.target_ids)
            ctrl = act.global_ctrl_ids.long()
            joints = act.target_ids.long()
            xi[:, XI_KP.start + joints] = model.actuator_gainprm[:, ctrl[:m_], 0] / mdp.kp_nom[joints]
            xi[:, XI_KD.start + joints] = model.actuator_gainprm[:, ctrl[m_:], 0] / mdp.kd_nom[joints]
        mass_ids, _ = robot.find_bodies(["torso_link", ".*_hip_.*_link", ".*_knee_link", ".*_ankle_.*_link"])
        body_ids = robot.indexing.body_ids[mass_ids].long()
        xi[:, XI_MASS] = model.body_mass[:, body_ids] / torch.as_tensor(mj.body_mass[body_ids.cpu().numpy()], device=dev, dtype=torch.float32)
        dofs = self.joint_v
        xi[:, XI_ARM] = model.dof_armature[:, dofs] / torch.as_tensor(mj.dof_armature[dofs.cpu().numpy()], device=dev, dtype=torch.float32)
        xi[:, XI_FRIC] = model.geom_friction[:, mdp.collision_geom_ids[:1], 0]
        return xi[:, XI_STATIC]


DEGENERATE_HALF_WIDTH = 1e-3  # official supermimic sim_context: half-widths at or below this are constant channels (scale 1, no float-noise amplification)


def dr_range_xi_stats(reader: "ACPSimReader") -> tuple[torch.Tensor, torch.Tensor]:
    """(mean, scale) (XI_DIM,) of the context from the DR ranges of the env's events (official supermimic ``SimContext``: range midpoint / half-width,
    half-widths <= 1e-3 get scale 1), in the units ``static_context`` / ``dynamic_context`` report: encoder bias [rad], torso CoM offset [m], kp / kd / mass /
    armature over nominal, friction (absolute), action delay [physics steps]. An event that is absent leaves its channels at the nominal value (half-width 0)."""
    env, robot = reader.env, reader.robot
    events = env.cfg.events
    lo, hi = torch.zeros(XI_DIM), torch.zeros(XI_DIM)
    lo[XI_KP] = hi[XI_KP] = lo[XI_KD] = hi[XI_KD] = lo[XI_MASS] = hi[XI_MASS] = lo[XI_ARM] = hi[XI_ARM] = 1.0

    def rng(slc, a, b):
        lo[slc], hi[slc] = float(a), float(b)

    joint_names = list(robot.joint_names)
    ids = env.action_manager.get_term("joint_pos").target_ids.tolist()  # robot joint indices, as ``dynamic_context`` (bias[:, ids] = encoder_bias[:, ids])
    import re
    for name in ("joint_zero_bias", "ankle_zero_bias"):
        ev = events.get(name)
        if ev is None:
            continue
        patterns = ev.params["asset_cfg"].joint_names
        patterns = (patterns,) if isinstance(patterns, str) else tuple(patterns)
        b_lo, b_hi = ev.params["bias_range"]
        for j in ids:
            if any(re.match(p_, joint_names[j]) for p_ in patterns):
                lo[XI_BIAS.start + j], hi[XI_BIAS.start + j] = float(b_lo), float(b_hi)
    if (ev := events.get("base_com")) is not None:
        for k, axis in enumerate("xyz"):
            a, b = ev.params["com_range"].get(axis, (0.0, 0.0))
            lo[XI_COM.start + k], hi[XI_COM.start + k] = float(a), float(b)
    if (ev := events.get("randomize_actuator_gains")) is not None:
        assert ev.params.get("operation", "scale") == "scale"
        rng(XI_KP, *ev.params["kp_range"])
        rng(XI_KD, *ev.params["kd_range"])
    mass_ids, mass_names = robot.find_bodies(["torso_link", ".*_hip_.*_link", ".*_knee_link", ".*_ankle_.*_link"])
    if (ev := events.get("torso_added_mass")) is not None:
        assert ev.params.get("operation") == "add"
        torso = int(robot.indexing.body_ids[robot.body_names.index("torso_link")])
        m0 = float(env.sim.mj_model.body_mass[torso])
        k = mass_names.index("torso_link")
        a, b = ev.params["ranges"]
        lo[XI_MASS.start + k], hi[XI_MASS.start + k] = 1.0 + a / m0, 1.0 + b / m0
    if (ev := events.get("leg_link_mass")) is not None:  # pseudo_inertia: mass scale exp(2 alpha), alpha drawn so that the scale is uniform
        import math
        a, b = ev.params["alpha_range"]
        for k, n_ in enumerate(mass_names):
            if n_ != "torso_link":
                lo[XI_MASS.start + k], hi[XI_MASS.start + k] = math.exp(2 * a), math.exp(2 * b)
    if (ev := events.get("joint_armature")) is not None:
        assert ev.params.get("operation") == "scale"
        rng(XI_ARM, *ev.params["ranges"])
    if (ev := events.get("physics_material")) is not None:
        assert ev.params.get("operation") == "abs"
        rng(XI_FRIC, *ev.params["ranges"][0])
    else:
        lo[XI_FRIC] = hi[XI_FRIC] = float(env.sim.mj_model.geom_friction[0, 0])
    if reader.has_delay:
        cfg = robot.actuators[0].cfg
        rng(XI_LAG, getattr(cfg, "delay_min_lag", 0), getattr(cfg, "delay_max_lag", 0))
    half = (hi - lo) / 2
    scale = torch.where(half > DEGENERATE_HALF_WIDTH, half, torch.ones_like(half))
    return ((lo + hi) / 2).to(env.device), scale.to(env.device)


def make_kinematics(env, manager, motion: ACPMotion, device: str) -> ACPKinematics:
    """Forward kinematics of the tracked bodies of the motion reference (the nine contact links first, then the SuperMimic tracked bodies)."""
    tracked = list(TRACKED_LINKS[: motion.track_p.shape[1]])
    return ACPKinematics(device, list(env.scene["robot"].joint_names), tracked, manager.cfg.robot_model_path, num_primary=len(CONTACT_LINKS))


def build_mdp(env, heightmap_resolution: float, heightmap_cache_dir: str, device: str, exact_scan: bool = False, lazy_exact_heightmap: bool = False):
    """Instantiate FK, motion library, height scanner (the 0.05 m height map, or with ``exact_scan`` exact ray casts with its gradient) and analytic
    MDP for the live (unwrapped) env."""
    manager = env.scene["motion_reference"]
    motion = ACPMotion(manager, env.command_manager.get_term(CONTACTS_COMMAND), device)
    kin = make_kinematics(env, manager, motion, device)
    metadata_yaml = next(iter(manager.motion_buffers.values())).cfg.metadata_yaml
    loader = None
    if exact_scan and lazy_exact_heightmap:
        import numpy as np
        # Only pattern offsets are read before ExactHeightScanner takes over. The real map is loaded on demand.
        heights, x0, y0 = np.zeros((2, 2), dtype=np.float32), 0.0, 0.0
        loader = lambda: load_or_build_heightmap(env.sim.mj_model, metadata_yaml, heightmap_resolution, heightmap_cache_dir)
        print('[exact scan] deferring unused auxiliary heightmap; ground_height will load the real map on demand', flush=True)
    else:
        heights, x0, y0 = load_or_build_heightmap(env.sim.mj_model, metadata_yaml, heightmap_resolution, heightmap_cache_dir)
    scanner = HeightScanner(
        heights, x0, y0, heightmap_resolution, HEIGHT_SCANNER_SIZE, HEIGHT_SCANNER_RESOLUTION, HEIGHT_SCANNER_MAX_DISTANCE, device
    )
    if exact_scan:
        scanner = ExactHeightScanner(env, scanner, heightmap_loader=loader)
    return motion, kin, scanner, ACPMDP(kin, motion, scanner, env, env.step_dt, device)
