import os
from copy import deepcopy
from dataclasses import dataclass, field
from functools import partial

import yaml
from mjlab.actuator import ActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.viewer.viewer_config import ViewerConfig

import instinct_mj.tasks.acp.acp_env_cfg as acp_cfg
import instinct_mj.tasks.acp.scene_cfg as scene_cfg
from instinct_mj.tasks.acp.mdp.actuators import SharedDelayPdActuatorCfg
from instinct_mj.tasks.acp.terrain_layout import CompactFiledTerrainGenerator
from instinct_mj.assets.acp_g1 import get_acp_g1_spec, write_acp_g1_mjcf
from instinct_mj.motion_reference import MotionSequence
from instinct_mj.motion_reference.motion_files.terrain_motion import TerrainMotion
from instinct_mj.motion_reference.motion_files.terrain_motion_cfg import TerrainMotionCfg as TerrainMotionCfgBase
from instinct_mj.motion_reference.motion_reference_cfg import MotionReferenceManagerCfg
from instinct_mj.motion_reference.utils import motion_interpolate_bilinear
from instinct_mj.rl.acp.planner import TRACKED_LINKS

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
        # ACP-unspecified: the soft joint limits (the range of the joint-limit reward, the reset clamp) are 0.9 of the joint range, as in the
        # SuperMimic reference implementation (robot.py:144) and the authors' ACP robot (acp/g1.py:172, mdp.py:49)
        soft_joint_pos_limit_factor=0.9,
    ),
)

# Same robot with the torso / arms / pelvis / legs collision meshes replaced by analytic capsules / spheres; the feet keep the authors'
# capsules, they define the ``foot`` contact body (see assets/acp_g1.py and COLLISION_SIMPLIFY.md). Default off: the ACP datasets
# (``v3_final`` / ``v3_train6``) are labelled on the authors' collision meshes, so labels and simulation must measure the same surface
# (``label_contacts`` defaults to the meshes as well). Opt in only with sidecars labelled by ``label_contacts --simplified-collision``.
def get_acp_g1_spec_simplified():
    """The spec of the simplified collision model (a named function, not a ``partial``: the env cfg is dumped to yaml by name)."""
    return get_acp_g1_spec(simplified_collision=True)


G1_CFG_SIMPLIFIED_COLLISION = deepcopy(G1_CFG)
G1_CFG_SIMPLIFIED_COLLISION.spec_fn = get_acp_g1_spec_simplified
# Same model as a flat MJCF for the analytic contact term of the world-model rollout (``GeometricContact`` reads the surface points of
# the collision geoms from it; the forward kinematics is unchanged, only the collision primitives differ).
ACP_G1_MJCF_PATH_SIMPLIFIED = write_acp_g1_mjcf(simplified_collision=True)


def apply_simplified_collision(cfg) -> None:
    """Replace the collision spec of a ``cfg.scene`` robot entity by ``G1_CFG_SIMPLIFIED_COLLISION``'s (for cfgs that do not go through
    ``G1AcpTrackV1EnvCfg``, e.g. Stage 3)."""
    cfg.scene.entities["robot"].spec_fn = get_acp_g1_spec_simplified
    # the motion reference also feeds ``robot_model_path`` to ``GeometricContact`` (the imagined contact clearance), which must
    # measure the same primitives as the simulation
    for sensor in cfg.scene.sensors:
        if getattr(sensor, "name", None) == "motion_reference":
            sensor.robot_model_path = ACP_G1_MJCF_PATH_SIMPLIFIED


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

    # The clip is uniformly sampled, the start frame is set by the episode start term (``reset_motion_start_phase``).
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
    # the 14 SuperMimic tracked bodies: the nine contact links (in the order of CONTACT_BODIES, they come first) + torso, both hip-roll and
    # shoulder-roll links (the ``track_body_*`` rewards)
    link_of_interests=list(TRACKED_LINKS),
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


def make_scene_cfg(scene_class: type, *, num_envs: int, env_spacing: float = 4.0, num_tiles: int = scene_cfg.NUM_TILES_PER_DATASET_TERRAIN):
    """The G1 scene: the robot, the ACP sensors with the G1 motion reference, the terrain of ``scene_cfg.make_acp_terrain``."""
    return scene_class(
        num_envs=num_envs,
        env_spacing=env_spacing,
        entities={"robot": deepcopy(G1_CFG)},
        sensors=scene_cfg.make_acp_scene_sensors(motion_reference=deepcopy(motion_reference_cfg)),
        terrain=scene_cfg.make_acp_terrain(num_tiles),
    )


def _set_motion_terrain(env_cfg: acp_cfg.AcpTrackEnvCfg) -> None:
    """Point the terrain generator at the dataset of the motion buffer and give it one tile per terrain of the dataset (unless ``num_rows`` is set).

    Every terrain of the dataset needs at least one tile, or its motions never get an origin and are resampled away. The motion matching reads the
    terrain index from the tile difficulty (int(difficulty * num_terrains)); the curriculum layout is one column of ``num_rows`` tiles with difficulty
    row / (rows - 1), i.e. exactly one tile per terrain for ``num_rows = num_terrains`` (the placeholder ``scene_cfg.NUM_TILES_PER_DATASET_TERRAIN``).
    The tile size / layout class are those of the scene (``scene_cfg.make_acp_terrain``)."""
    motion_reference_cfg_ = next(sensor for sensor in env_cfg.scene.sensors if sensor.name == "motion_reference")
    motion_buffer = motion_reference_cfg_.motion_buffers["TerrainMotion"]
    motion_buffer.metadata_yaml = os.path.join(motion_buffer.path, "metadata.yaml")
    terrain_generator = env_cfg.scene.terrain.terrain_generator
    terrain_cfg = terrain_generator.sub_terrains["motion_matched"]
    terrain_cfg.path = motion_buffer.path
    terrain_cfg.metadata_yaml = motion_buffer.metadata_yaml
    if not (terrain_generator.curriculum and terrain_generator.num_cols == 1 and terrain_generator.class_type is CompactFiledTerrainGenerator):
        raise ValueError("The ACP terrain needs the curriculum layout of one column of tiles on the compact grid (scene_cfg.make_acp_terrain).")
    if terrain_generator.num_rows == scene_cfg.NUM_TILES_PER_DATASET_TERRAIN:
        with open(os.path.expanduser(motion_buffer.metadata_yaml)) as f:
            terrain_generator.num_rows = len(yaml.safe_load(f)["terrains"])


@dataclass(kw_only=True)
class G1AcpTrackV1EnvCfg(acp_cfg.AcpTrackEnvCfg):
    """Stage 1 of the v1 task (``Instinct-ACP-Track-G1-v1``): ``acp_cfg.make_rewards`` / ``make_terminations`` / ``make_events`` as they are (paper Table 3 +
    SuperMimic body terms, SuperMimic terminations, Table S2 reset noise with penetration rejection) and the SuperMimic episode cap of 10 s (time-out,
    bootstrapped; the clips are longer than the cap, so the command term does not require a full clip). Use it with the runner options of
    ``scripts/acp/run_stage1.sh`` (SuperMimic clip-end semantics, reset bins)."""

    scene: scene_cfg.AcpSceneCfg = field(default_factory=lambda: make_scene_cfg(scene_cfg.AcpSceneCfg, num_envs=1024))
    # v1 default ON (user decision 2026-11-08): the upper body / arms / pelvis / wrists / hands (r = 0.04 sphere) are analytic capsules / spheres, the legs and the authors' foot capsules stay;
    # dynamics and mass properties are unchanged (COLLISION_SIMPLIFY.md). The contact sidecars of the datasets are labelled on the authors' meshes and are NOT relabelled, so the command
    # term is told to accept the mismatch (``allow_label_collision_mismatch``, set below); ``simplified_collision=False`` restores the meshes (and the strict sidecar check).
    simplified_collision: bool = True

    # SuperMimic episode cap (time-out); the clips are longer than it, so the command term does not require a full clip
    episode_length_s: float = 10.0
    commands: dict = field(default_factory=lambda: acp_cfg.make_commands(require_full_clip=False))
    run_name: str = "g1AcpTrackV1"

    def __post_init__(self):
        # only derived settings: the collision model of the robot / the reference kinematics and the dataset terrain
        if self.simplified_collision:
            apply_simplified_collision(self)
            self.commands[acp_cfg.CONTACTS_COMMAND].allow_label_collision_mismatch = True
        _set_motion_terrain(self)


@dataclass(kw_only=True)
class G1AcpStage2V1EnvCfg(G1AcpTrackV1EnvCfg):
    """Stage 2 (``Instinct-ACP-Stage2-G1-v1``): Stage 1 with an episode cap of 20 s. The hindsight planner labels need episodes that progress >= 90 % of a clip and
    the longest clip is 18.7 s (with 10 s, the clips longer than 11.1 s would never produce planner samples). The planner / eta schedules live in the runner."""

    episode_length_s: float = 20.0
    run_name: str = "g1AcpStage2V1"


@dataclass(kw_only=True)
class G1AcpTrackV1EnvCfg_PLAY(G1AcpTrackV1EnvCfg):
    """Evaluation: no domain randomization / observation noise / action delay; the episode lasts until the clip end (120 s cap) and only the fall terminations end it
    early (as in ``record_rollouts``); the terrain grid keeps one tile per terrain of the dataset (the clip is chosen by the motion weights of the reset bins)."""

    scene: scene_cfg.AcpSceneCfg = field(default_factory=lambda: make_scene_cfg(scene_cfg.AcpSceneCfg, num_envs=1, env_spacing=2.5))

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

    episode_length_s: float = 120.0
    run_name: str = "g1AcpTrackV1_play"

    def __post_init__(self):
        super().__post_init__()
        self.disable_randomization()
        self.curriculum["episode_start"].params["mode"] = "start"
        for actuator in self.scene.entities["robot"].articulation.actuators:
            actuator.delay_max_lag = 0
        self.events["reset_robot"].params["randomize_velocity_range"] = {}
