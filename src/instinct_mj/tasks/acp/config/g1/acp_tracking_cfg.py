import os
from copy import deepcopy
from dataclasses import dataclass, field
from functools import partial

import yaml
from mjlab.actuator import ActuatorCfg
from mjlab.managers import TerminationTermCfg as DoneTermCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.viewer.viewer_config import ViewerConfig

import instinct_mj.tasks.acp.acp_env_cfg as acp_cfg
import instinct_mj.tasks.acp.mdp as acp_mdp
from instinct_mj.tasks.acp.mdp.actuators import SharedDelayPdActuatorCfg
from instinct_mj.assets.acp_g1 import get_acp_g1_spec, write_acp_g1_mjcf
from instinct_mj.motion_reference import MotionSequence
from instinct_mj.motion_reference.motion_files.terrain_motion import TerrainMotion
from instinct_mj.motion_reference.motion_files.terrain_motion_cfg import TerrainMotionCfg as TerrainMotionCfgBase
from instinct_mj.motion_reference.motion_reference_cfg import MotionReferenceManagerCfg
from instinct_mj.motion_reference.utils import motion_interpolate_bilinear
from instinct_mj.rl.acp.planner import CONTACT_LINKS

# NOTE: Change this if your local ACP motion dataset lives elsewhere. The folder holds the motion files, a
# `metadata.yaml` and, next to every motion `<stem>.npz`, its contact labels `<stem>.contacts.npz` (framerate, body_names,
# contact_flag, contact_pos_w); a missing sidecar raises an error.
MOTION_FOLDER = "~/Datasets/acp/acp_motions_v1_50hz"

# Action delay U(0, 20) ms (Table S2): the lag is in physics steps (PD 200 Hz, 5 ms), 0..4 steps, one lag per env shared by all actuator groups
# (``SharedDelayPdActuatorCfg``), resampled on reset only.
_DELAY_RESET_ONLY_PERIOD = 1_000_000
_MAX_DELAY_STEPS = 4


def _pd(joints: tuple[str, ...], kp: float, kd: float, effort_limit: float) -> SharedDelayPdActuatorCfg:
    return SharedDelayPdActuatorCfg(
        target_names_expr=joints,
        stiffness=kp,
        damping=kd,
        effort_limit=effort_limit,
        # armature, joint damping and friction loss stay those of the MJCF
        delay_min_lag=0,
        delay_max_lag=_MAX_DELAY_STEPS,
        delay_update_period=_DELAY_RESET_ONLY_PERIOD,
        delay_per_env_phase=False,
    )


# Kinematics of the motion reference (forward kinematics of the contact links from the pelvis pose).
ACP_G1_MJCF_PATH = write_acp_g1_mjcf()


# PD gains (kp, kd) of the authors' deploy config; effort limits and armatures as the BeyondMimic G1 asset.
acp_g1_actuator_cfgs: tuple[ActuatorCfg, ...] = (
    # PD gains (kp, kd) of the authors' deploy config (REPORT.md section 3.4); effort limits = ctrlrange of the playground MJCF motors
    _pd((".*_hip_pitch_joint", ".*_hip_yaw_joint"), 88.0, 2.0, 88.0),
    _pd((".*_hip_roll_joint",), 139.0, 3.0, 88.0),
    _pd((".*_knee_joint",), 139.0, 3.0, 139.0),
    _pd((".*_ankle_pitch_joint", ".*_ankle_roll_joint"), 35.0, 0.8, 50.0),
    _pd(("waist_yaw_joint",), 88.0, 2.0, 88.0),
    _pd(("waist_roll_joint", "waist_pitch_joint"), 50.0, 1.0, 50.0),
    _pd((".*_shoulder_pitch_joint",), 35.0, 0.8, 25.0),
    _pd((".*_shoulder_roll_joint", ".*_shoulder_yaw_joint", ".*_elbow_joint"), 40.0, 1.0, 25.0),
)

# Default pose (policy names); the root is the pelvis. The wrists are welded in this model (23 joints).
G1_CFG = EntityCfg(
    init_state=EntityCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.78),
        joint_pos={
            ".*_hip_pitch_joint": -0.3,
            ".*_knee_joint": 0.62,
            ".*_ankle_pitch_joint": -0.33,
            ".*_shoulder_pitch_joint": 0.35,
            "left_shoulder_roll_joint": 0.16,
            "right_shoulder_roll_joint": -0.16,
            ".*_elbow_joint": 0.30,
        },
        joint_vel={".*": 0.0},
    ),
    spec_fn=get_acp_g1_spec,
    articulation=EntityArticulationInfoCfg(
        actuators=tuple(deepcopy(act) for act in acp_g1_actuator_cfgs),
        # ACP-unspecified: soft joint limit factor (as the G1 perceptive shadowing asset)
        soft_joint_pos_limit_factor=0.9,
    ),
)

# Same robot with the torso / arms / pelvis / legs collision meshes replaced by analytic capsules / spheres; the feet keep the authors'
# capsules, they define the ``foot`` contact body (see assets/acp_g1.py and COLLISION_SIMPLIFY.md). Default off: G1_CFG above is
# unchanged, so existing tasks / checkpoints are unaffected. A task can opt in with ``G1AcpTrackEnvCfg(simplified_collision=True)`` (or
# by passing this cfg as ``entities={"robot": ...}``).
G1_CFG_SIMPLIFIED_COLLISION = deepcopy(G1_CFG)
G1_CFG_SIMPLIFIED_COLLISION.spec_fn = partial(get_acp_g1_spec, simplified_collision=True)


def apply_simplified_collision(cfg) -> None:
    """Replace the collision spec of a ``cfg.scene`` robot entity by ``G1_CFG_SIMPLIFIED_COLLISION``'s (for cfgs that do not go through
    ``G1AcpTrackEnvCfg``, e.g. Stage 3)."""
    cfg.scene.entities["robot"].spec_fn = partial(get_acp_g1_spec, simplified_collision=True)


class AcpTerrainMotion(TerrainMotion):
    """``TerrainMotion`` whose files are all in the retargeted format (joint_pos, base_pos_w, base_quat_w, framerate, joint_names): the
    ACP datasets name their motions freely (``..._c00.npz``, ``..._t2.npz``), while the base class picks the reader from the file name."""

    def _read_motion_file(self, motion_file_idx: int) -> MotionSequence:
        return self._read_retargetted_motion_file(self._all_motion_files[motion_file_idx])


@dataclass(kw_only=True)
class TerrainMotionCfg(TerrainMotionCfgBase):
    class_type: type = AcpTerrainMotion

    path: object = field(default_factory=lambda: os.path.expanduser(MOTION_FOLDER))

    metadata_yaml: object = field(default_factory=lambda: os.path.expanduser(f"{MOTION_FOLDER}/metadata.yaml"))

    max_origins_per_motion: int = 49

    # The robot is reset to the reference pose (+ U(0, 0.05) m base height by the reset event).
    ensure_link_below_zero_ground: bool = False

    motion_start_height_offset: float = 0.0

    # The clip is uniformly sampled, the start phase is set by the CSI curriculum (``reset_motion_start_phase``).
    motion_start_from_middle_range: list = field(default_factory=lambda: [0.0, 0.0])

    motion_bin_length_s: float | None = None

    buffer_device: str = "output_device"

    motion_interpolate_func: object = field(default_factory=lambda: motion_interpolate_bilinear)

    velocity_estimation_method: str = "frontbackward"

    env_starting_stub_sampling_strategy: str = "independent"


motion_reference_cfg = MotionReferenceManagerCfg(
    name="motion_reference",
    entity_name="robot",
    robot_model_path=ACP_G1_MJCF_PATH,
    # the 9 contact links, in the order of CONTACT_BODIES
    link_of_interests=list(CONTACT_LINKS),
    symmetric_augmentation_link_mapping=None,
    symmetric_augmentation_joint_mapping=None,
    symmetric_augmentation_joint_reverse_buf=None,
    # only the current reference frame is used (rewards, terminations, contact targets, goal)
    frame_interval_s=0.02,
    update_period=0.02,
    num_frames=1,
    data_start_from="current_time",
    visualizing_robot_offset=(2.0, 0.0, 0.0),
    visualizing_robot_from="reference_frame",
    visualizing_marker_types=["links"],
    motion_buffers={
        "TerrainMotion": TerrainMotionCfg(),
    },
    mp_split_method="None",
)


def _set_motion_terrain(env_cfg: acp_cfg.AcpTrackEnvCfg) -> None:
    motion_reference_cfg_ = next(sensor for sensor in env_cfg.scene.sensors if sensor.name == "motion_reference")
    motion_buffer = motion_reference_cfg_.motion_buffers["TerrainMotion"]
    motion_buffer.metadata_yaml = os.path.join(motion_buffer.path, "metadata.yaml")
    terrain_cfg = env_cfg.scene.terrain.terrain_generator.sub_terrains["motion_matched"]
    terrain_cfg.path = motion_buffer.path
    terrain_cfg.metadata_yaml = motion_buffer.metadata_yaml
    # Every terrain of the dataset needs at least one tile, or its motions never get an origin and are resampled away. The motion
    # matching reads the terrain index from the tile difficulty (int(difficulty * num_terrains)), which the random grid draws
    # uniformly (9 tiles for 29 terrains). The curriculum layout is one column of ``num_rows`` tiles with difficulty row / (rows - 1),
    # i.e. exactly one tile per terrain for ``num_rows = num_terrains``.
    with open(os.path.expanduser(motion_buffer.metadata_yaml)) as f:
        num_terrains = len(yaml.safe_load(f)["terrains"])
    terrain_generator = env_cfg.scene.terrain.terrain_generator
    terrain_generator.curriculum = True
    terrain_generator.num_rows = num_terrains
    terrain_generator.num_cols = 1


@dataclass(kw_only=True)
class G1AcpTrackEnvCfg(acp_cfg.AcpTrackEnvCfg):
    scene: acp_cfg.AcpSceneCfg = field(
        default_factory=lambda: acp_cfg.AcpSceneCfg(
            num_envs=1024,
            entities={"robot": deepcopy(G1_CFG)},
            sensors=acp_cfg.make_acp_scene_sensors(motion_reference=deepcopy(motion_reference_cfg)),
        )
    )
    # Opt-in (default off): torso / arm / pelvis / leg collision meshes -> analytic primitives, the authors' foot capsules unchanged
    # (COLLISION_SIMPLIFY.md).
    simplified_collision: bool = False

    def __post_init__(self):
        super().__post_init__()
        if self.simplified_collision:
            apply_simplified_collision(self)
        _set_motion_terrain(self)
        self.run_name = "g1AcpTrack"


@dataclass(kw_only=True)
class G1AcpTrackNoLinkTermEnvCfg(G1AcpTrackEnvCfg):
    """Track without the end-effector (``link_pos_too_far``) termination: only the root height / tilt terminations remain, as in SuperMimic (paper
    appendix: "we do not terminate on end-effector deviation"). The ACP paper gives no thresholds for these terminations."""

    def __post_init__(self):
        super().__post_init__()
        self.terminations.pop("link_pos_too_far", None)  # (__post_init__ may run again on a copy)
        self.run_name = "g1AcpTrackNoLinkTerm"


def apply_supermimic_deltas(cfg, episode_length_s: float | None) -> None:
    """The deltas of the SuperMimic variants of the ACP tasks (paper App. D.1 / the reference implementation ``env.py``): no end-effector (``link_pos_too_far``) termination,
    the tilt termination on the z component only (|R_zz - ref R_zz| > 0.8), soft joint limits = joint limits, ``contact_acc`` from the difference of the link velocities (the
    authors' framework), the reset noise of the ACP paper (Table S2: base height + U(0, 0.05) m, joint positions +-0.2 rad, clamped to the joint limits; no root
    position / rotation / velocity noise; user decision 2026-11-04: the SuperMimic reference implementation's noise (root xy +-0.05 m, rotation vector +-(0.1, 0.1, 0.2) rad,
    root velocities +-(0.5, 0.5, 0.2) m/s / +-(0.52, 0.52, 0.78) rad/s, joints +-0.1 rad) is made for flat ground and put robots on contact poses into ladder rungs / boxes;
    the noise of an env that starts > 1 cm inside the terrain is redrawn (``max_penetration``) and the root is then raised by the deepest
    remaining penetration, ``lift_out_of_terrain``) and, with ``episode_length_s``, the episode cap (time-out, bootstrapped)."""
    cfg.terminations.pop("link_pos_too_far", None)  # (__post_init__ may run again on a copy)
    cfg.terminations["base_pg_too_far"].params["z_only"] = True  # the tilt test of the reference implementation (z component of the projected gravity)
    cfg.scene.entities["robot"].articulation.soft_joint_pos_limit_factor = 1.0  # the reference implementation's joint-limit reward
    cfg.rewards["rewards"]["contact_acc"].params["mode"] = "vel1"
    if episode_length_s is not None:
        cfg.episode_length_s = episode_length_s
        cfg.commands[acp_cfg.CONTACTS_COMMAND].require_full_clip = False  # the longest clips are longer than the episode cap
    cfg.events["reset_robot"].params.update({
        "randomize_pose_range": {"z": (0.0, 0.05)},
        "randomize_velocity_range": {},
        "randomize_joint_pos_range": (-0.2, 0.2),
        "clamp_joint_pos_to_limits": True,
        # 32-env check on v3_train4 (uniform start phase): this noise alone puts 24 % of the box-climb / 37 % of the ladder / 12 % of the vault starts > 1 cm into
        # the terrain; the vertical lift alone leaves 8 / 18 / 1.6 % (side and overhead contacts: rungs); the rejection (redraw, then the noise-free
        # reference) plus the lift leaves 0 / 0.4 / 0 % (resampled 27 / 35 / 12 %, fallback <= 0.9 %)
        "max_penetration": 0.01,
        "max_resample": 3,  # most penetrating draws are fixed by the first redraw; the rest fall back to the noise-free reference
        "lift_out_of_terrain": True,  # a foot / hand the noise pushed into a box top starts on it (Acp_supermimic_vibed env.py)
    })


@dataclass(kw_only=True)
class G1AcpTrackSMEnvCfg(G1AcpTrackNoLinkTermEnvCfg):
    """NoLinkTerm with the SuperMimic deltas (``apply_supermimic_deltas``) and the episode cap of 10 s. Used with the runner option ``reset_sampling="bins"``."""

    def __post_init__(self):
        super().__post_init__()
        apply_supermimic_deltas(self, 10.0)
        self.run_name = "g1AcpTrackSM"


@dataclass(kw_only=True)
class G1AcpStage2EnvCfg(G1AcpTrackEnvCfg):
    """Stage 2: the episodes start at a uniform phase U(0, 0.98) of the clip (instead of the Curriculum State Initialization)."""

    def __post_init__(self):
        super().__post_init__()
        self.curriculum["curriculum_state_init"].params["mode"] = "uniform"
        self.run_name = "g1AcpStage2"


@dataclass(kw_only=True)
class G1AcpStage2SMEnvCfg(G1AcpStage2EnvCfg):
    """Stage 2 with the SuperMimic deltas and an episode cap of 20 s (the initial phases come from the runner option ``reset_sampling="bins"``, half of whose distribution is
    uniform over the clip time). 20 s instead of the 10 s of Stage 1: the hindsight planner labels need episodes that progress >= 90 % of a clip, and the longest clip is
    18.7 s (with 10 s, the 10 clips longer than 11.1 s would never produce planner samples)."""

    def __post_init__(self):
        super().__post_init__()
        apply_supermimic_deltas(self, 20.0)
        self.run_name = "g1AcpStage2SM"


@dataclass(kw_only=True)
class G1AcpTrackSMXYEnvCfg(G1AcpTrackSMEnvCfg):
    """Track SM plus the termination ``base_xy_too_far``: the episode ends when the pelvis is further than ``base_xy_threshold`` [m] horizontally from the reference pelvis
    (the SM tilt / height terminations do not fire on a robot that stands still in front of an obstacle). The imagination's analytic termination mirrors it."""

    base_xy_threshold: float = 1.0  # [m]: chosen from the final4 eval (model_13750): 2-6 % of the episodes of the solved box / gap / get_up / stairs / walk clips exceed it (0.5 m: 10-23 %), 70 % of the vault episodes (the stall) do

    def __post_init__(self):
        super().__post_init__()
        self.terminations["base_xy_too_far"] = DoneTermCfg(
            func=acp_mdp.pos_xy_far_from_ref, time_out=False, params={"command_name": acp_cfg.CONTACTS_COMMAND, "distance_threshold": self.base_xy_threshold}
        )
        self.run_name = "g1AcpTrackSMXY"


@dataclass(kw_only=True)
class G1AcpTrackSMBMEnvCfg(G1AcpTrackSMEnvCfg):
    """Track SM with the termination set of the BeyondMimic / ACP tracking task: root height (0.25 m), tilt (z component, 0.8) and the contact-link height deviation ``link_pos_too_far``
    exactly as the original Track task (the 9 contact links, height only, 0.25 m); SuperMimic drops the end-effector term. The imagination's analytic terminations mirror all three."""

    def __post_init__(self):
        super().__post_init__()
        self.terminations["link_pos_too_far"] = acp_cfg.make_terminations()["link_pos_too_far"]
        self.run_name = "g1AcpTrackSMBM"


@dataclass(kw_only=True)
class G1AcpTrackSMCEnvCfg(G1AcpTrackSMXYEnvCfg):
    """Track SM-XY (height 0.25 m, z-tilt 0.8, ``base_xy_too_far`` 1.0 m) plus the contact-consistent termination ``contact_far_from_ref``: a contact link further than
    ``contact_distance_threshold`` [m] (3-D, world) from its reference position while the reference is in contact with it (after the first ``contact_grace_frames`` frames of the
    contact segment). The imagination mirrors all four."""

    contact_distance_threshold: float = 0.55  # [m], frame "base": from the final4xy eval (model_18500): the smallest D with <= 5 % cut of the successful episodes of the solved clips (4.0 % at 0.55; 0.5: 6.4 %)
    contact_grace_frames: int = 2
    contact_frame: str = "base"
    contact_upright_cos: float = 0.707

    def __post_init__(self):
        super().__post_init__()
        self.terminations["contact_far_from_ref"] = DoneTermCfg(
            func=acp_mdp.contact_far_from_ref, time_out=False,
            params={"command_name": acp_cfg.CONTACTS_COMMAND, "distance_threshold": self.contact_distance_threshold, "grace_frames": self.contact_grace_frames, "frame": self.contact_frame, "upright_cos": self.contact_upright_cos},
        )
        self.run_name = "g1AcpTrackSMC"


@dataclass(kw_only=True)
class G1AcpTrackSMCH5EnvCfg(G1AcpTrackSMCEnvCfg):
    """Track SM-C with the base height termination relaxed from 0.25 m to ``base_height_threshold`` (ACP-unspecified; 0.25 m is SuperMimic's flat-ground value).
    The termination statistics of the SM-C run (model_25750, from the clip start) showed the height term ending 53 % of all episodes (box 73 %, ladder 67 %,
    stepping 70 %): a robot that lags the reference while climbing is cut before it can catch up; the contact and xy terms now cover the placement / drift
    failures, the relaxed height term still ends 'staying below the obstacle'. The imagination reads the threshold from this cfg."""

    base_height_threshold: float = 0.5

    def __post_init__(self):
        super().__post_init__()
        self.terminations["base_pos_too_far"].params["distance_threshold"] = self.base_height_threshold
        self.run_name = "g1AcpTrackSMCH5"


@dataclass(kw_only=True)
class G1AcpTrackSMCH5X5EnvCfg(G1AcpTrackSMCH5EnvCfg):
    """Track SM-CH5 with the horizontal pelvis termination at 0.5 m instead of 1.0 m: the ACP evaluation counts an episode as a success only within 0.5 m of the
    goal horizontally (acp2.txt:273-276), so the training ends an episode once the pelvis leaves that tolerance (user decision; a policy that lags or stalls before an
    obstacle no longer keeps collecting reward). The imagination reads the threshold from this cfg."""

    base_xy_threshold: float = 0.5

    def __post_init__(self):
        super().__post_init__()
        self.run_name = "g1AcpTrackSMCH5X5"


@dataclass(kw_only=True)
class G1AcpTrackSMCH5X5PenEnvCfg(G1AcpTrackSMCH5X5EnvCfg):
    """SM-CH5X5 with penetration rejection of the reset noise (user decision after the non-finite investigation: the SuperMimic reset noise put 32 % of
    the ladder start samples > 1 cm, up to 8.4 cm, into the terrain). After the noisy state is written, the envs whose deepest robot-vs-terrain penetration
    (collision detection, self-contacts ignored) exceeds ``reset_max_penetration`` redraw the noise (same distributions, same clip and start frame), up to
    ``reset_max_resample`` times; the envs still penetrating then start from the noise-free reference state (kept even if that penetrates: it is the data).
    Reset only (nothing during the episode): no change of the rewards / terminations / observations, nothing to mirror in the imagination."""

    reset_max_penetration: float = 0.01  # [m]
    reset_max_resample: int = 8

    def __post_init__(self):
        super().__post_init__()
        self.events["reset_robot"].params.update({"max_penetration": self.reset_max_penetration, "max_resample": self.reset_max_resample})
        self.run_name = "g1AcpTrackSMCH5X5Pen"


@dataclass(kw_only=True)
class G1AcpTrackEnvCfg_PLAY(G1AcpTrackEnvCfg):
    """Evaluation: no domain randomization / observation noise / action delay, every clip starts at its beginning."""

    scene: acp_cfg.AcpSceneCfg = field(
        default_factory=lambda: acp_cfg.AcpSceneCfg(
            num_envs=1,
            env_spacing=2.5,
            entities={"robot": deepcopy(G1_CFG)},
            sensors=acp_cfg.make_acp_scene_sensors(motion_reference=deepcopy(motion_reference_cfg)),
        )
    )

    viewer: ViewerConfig = field(
        default_factory=lambda: ViewerConfig(
            lookat=(0.0, 0.0, 0.0),
            distance=2.1213,
            elevation=45.0,
            azimuth=0.0,
            origin_type=ViewerConfig.OriginType.WORLD,
            entity_name="robot",
            body_name="torso_link",
        )
    )

    def __post_init__(self):
        super().__post_init__()
        self.disable_randomization()
        self.curriculum["curriculum_state_init"].params["mode"] = "start"
        for actuator in self.scene.entities["robot"].articulation.actuators:
            actuator.delay_max_lag = 0
        self.scene.terrain.terrain_generator.num_rows = 1
        self.scene.terrain.terrain_generator.num_cols = 1
        self.run_name += "_play"


@dataclass(kw_only=True)
class G1AcpTrackSMCH5EnvCfg_PLAY(G1AcpTrackEnvCfg_PLAY, G1AcpTrackSMCH5EnvCfg):
    """Evaluation of the SM-CH5 checkpoints (``G1AcpTrackEnvCfg_PLAY`` settings on the SM-CH5 deltas): the episode lasts until the clip end and only the fall terminations end it
    early (as in ``record_rollouts``); the terrain grid keeps one tile per terrain of the dataset (the clip is chosen by the motion weights of the CSI)."""

    def __post_init__(self):
        super().__post_init__()
        _set_motion_terrain(self)  # (the play settings of the base class reduce the grid to one tile)
        self.episode_length_s = 120.0
        self.events["reset_robot"].params["randomize_velocity_range"] = {}
        for name in ("contact_far_from_ref", "base_xy_too_far", "link_pos_too_far"):
            self.terminations.pop(name, None)
        self.run_name = "g1AcpTrackSMCH5_play"

