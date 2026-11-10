"""Reads the simulator state / context of every env from the live mjlab ACP tracking env."""

from __future__ import annotations

import torch

from instinct_mj.rl.supermimic.sim import SimReader
from instinct_mj.rl.supermimic.terrain import HeightScanner, load_or_build_heightmap
from instinct_mj.tasks.acp.mdp.rewards import measured_contact_flags

from .kinematics import ACPKinematics
from .mdp import NUM_J, XI_ARM, XI_COM, XI_FRIC, XI_KD, XI_KP, XI_MASS, XI_STATIC, XI_STATIC_DIM, ACPMDP
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
