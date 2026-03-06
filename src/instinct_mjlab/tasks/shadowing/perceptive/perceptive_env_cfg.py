from dataclasses import field, dataclass, MISSING
import math

import mujoco
import mjlab.envs.mdp as mdp
import mjlab.sim as sim_utils
from mjlab.entity import EntityCfg
from mjlab.managers import CurriculumTermCfg, EventTermCfg
from mjlab.managers import ObservationGroupCfg as ObsGroupCfg
from mjlab.managers import ObservationTermCfg as ObsTermCfg
from mjlab.managers import RewardTermCfg as RewTermCfg
from mjlab.managers import SceneEntityCfg
from mjlab.managers import TerminationTermCfg as DoneTermCfg
from mjlab.scene import SceneCfg as InteractiveSceneCfg
from mjlab.sensor import (
    ContactMatch,
    ContactSensorCfg,
    GridPatternCfg,
    ObjRef,
    PinholeCameraPatternCfg,
    RayCastSensorCfg,
    SensorCfg,
)
from mjlab.terrains import FlatPatchSamplingCfg
from mjlab.utils.spec_config import MaterialCfg, TextureCfg
from mjlab.utils.noise import UniformNoiseCfg

import instinct_mjlab.envs.mdp as instinct_mdp
from instinct_mjlab.envs.manager_based_rl_env_cfg import InstinctLabRLEnvCfg
from instinct_mjlab.monitors import (
    MonitorTermCfg,
    MotionReferenceMonitorTerm,
    ShadowingJointPosMonitorTerm,
    ShadowingJointVelMonitorTerm,
    ShadowingLinkPosMonitorTerm,
    ShadowingPositionMonitorTerm,
    ShadowingRotationMonitorTerm,
)
from instinct_mjlab.motion_reference import MotionReferenceManagerCfg
from instinct_mjlab.sensors.grouped_ray_caster import GroupedRayCasterCameraCfg
from instinct_mjlab.sensors.noisy_camera import NoisyGroupedRayCasterCameraCfg
from instinct_mjlab.tasks.shadowing import mdp as shadowing_mdp
from instinct_mjlab.terrains.terrain_generator_cfg import FiledTerrainGeneratorCfg
from instinct_mjlab.terrains.terrain_importer_cfg import TerrainImporterCfg
from instinct_mjlab.terrains.trimesh import MotionMatchedTerrainCfg
from instinct_mjlab.utils.noise import (
    CropAndResizeCfg,
    DepthArtifactNoiseCfg,
    DepthContourNoiseCfg,
    DepthNormalizationCfg,
    DepthSkyArtifactNoiseCfg,
    GaussianBlurNoiseCfg,
    RangeBasedGaussianNoiseCfg,
    StereoTooCloseNoiseCfg,
)

# PROPRIO_HISTORY_LENGTH = 0
PROPRIO_HISTORY_LENGTH = 8
_UNDESIRED_CONTACT_BODY_REGEX = (
    r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$).+$"
)

def get_motion_matched_subterrain_cfg(sub_terrains: dict[str, object]) -> MotionMatchedTerrainCfg:
    """Return the required motion-matched subterrain config."""
    terrain_cfg = sub_terrains["motion_matched"]
    return terrain_cfg  # type: ignore[return-value]


def _edit_perceptive_scene_spec(spec: mujoco.MjSpec) -> None:
    """Apply skybox and terrain material to the scene spec."""
    ground_texture_name = "perceptive_groundplane"
    ground_material_name = "perceptive_groundplane"

    # Bright sky theme so native viewer doesn't look like a black void.
    sky_rgb_top = (0.98, 0.99, 1.0)
    sky_rgb_horizon = (0.78, 0.86, 0.95)
    # Gray-white striped checker style (matte and slightly dimmer for terrain).
    ground_rgb1 = (0.74, 0.76, 0.78)
    ground_rgb2 = (0.64, 0.66, 0.68)
    ground_mark_rgb = (0.60, 0.62, 0.64)

    # If a skybox already exists in the attached specs, patch it directly.
    # Otherwise, create one.
    existing_skybox = None
    for tex in spec.textures:
        if tex.type == mujoco.mjtTexture.mjTEXTURE_SKYBOX:
            existing_skybox = tex
            break

    if existing_skybox is not None:
        existing_skybox.builtin = mujoco.mjtBuiltin.mjBUILTIN_GRADIENT
        existing_skybox.rgb1[:] = sky_rgb_top
        existing_skybox.rgb2[:] = sky_rgb_horizon
        existing_skybox.width = 512
        existing_skybox.height = 3072
    else:
        TextureCfg(
            name="perceptive_skybox",
            type="skybox",
            builtin="gradient",
            rgb1=sky_rgb_top,
            rgb2=sky_rgb_horizon,
            width=512,
            height=3072,
        ).edit_spec(spec)

    TextureCfg(
        name=ground_texture_name,
        type="2d",
        builtin="checker",
        mark="edge",
        rgb1=ground_rgb1,
        rgb2=ground_rgb2,
        markrgb=ground_mark_rgb,
        width=300,
        height=300,
    ).edit_spec(spec)
    MaterialCfg(
        name=ground_material_name,
        texuniform=True,
        texrepeat=(5, 5),
        reflectance=0.0,
        texture=ground_texture_name,
    ).edit_spec(spec)

    spec.visual.rgba.haze[:] = (0.90, 0.94, 0.98, 1.0)
    spec.visual.headlight.ambient[:] = (0.28, 0.28, 0.28)
    spec.visual.headlight.diffuse[:] = (0.36, 0.36, 0.36)
    spec.visual.headlight.specular[:] = (0.0, 0.0, 0.0)
    # Increase shadow map resolution to eliminate jagged shadow edges (default is 4096).
    spec.visual.quality.shadowsize = 8192

    terrain_body = spec.body("terrain")
    for light in terrain_body.lights:
        # Soft terrain fill light: brighter than pure headlight mode while
        # avoiding hard specular highlights.
        light.castshadow = False
        light.ambient[:] = (0.12, 0.12, 0.12)
        light.diffuse[:] = (0.24, 0.24, 0.24)
        light.specular[:] = (0.0, 0.0, 0.0)
    for geom in terrain_body.geoms:
        geom.material = ground_material_name
        geom.rgba[:] = (ground_rgb1[0], ground_rgb1[1], ground_rgb1[2], 1.0)

    # Ensure reference robot never participates in contacts.
    # Some G1 XML geoms are unnamed, so name-pattern CollisionCfg cannot reliably
    # disable all reference collisions in this task.
    for geom in spec.geoms:
        parent = geom.parent
        body_name = parent.name or ""
        # Handle both direct entity names ("robot_reference/...") and nested
        # names that include additional namespace prefixes (".../robot_reference/...").
        is_reference_body = (
            body_name == "robot_reference"
            or body_name.startswith("robot_reference/")
            or "/robot_reference/" in body_name
        )
        if is_reference_body:
            original_contype = int(geom.contype)
            original_conaffinity = int(geom.conaffinity)
            collision_enabled = (original_contype != 0) or (original_conaffinity != 0)
            geom_name = (geom.name or "").lower()
            collision_name_hint = ("collision" in geom_name) or ("_col" in geom_name)
            hide_collision_geom = collision_enabled and (
                geom.type != mujoco.mjtGeom.mjGEOM_MESH or collision_name_hint
            )
            geom.contype = 0
            geom.conaffinity = 0
            if hide_collision_geom:
                # Hide collision bodies in motion-reference visualization.
                geom.group = 4
                geom.rgba[:] = (0.0, 0.0, 0.0, 0.0)
            else:
                # Keep reference visual geoms visible while excluding them from
                # depth camera rays (camera include groups are 0 and 2).
                geom.group = 1


@dataclass(kw_only=True)
class PerceptiveShadowingSceneCfg(InteractiveSceneCfg):
    """Configuration for the BeyondMimic scene with necessary scene entities as motion reference."""

    env_spacing: float = 4.0

    # terrain
    terrain: object = field(default_factory=lambda: TerrainImporterCfg(
        terrain_type="hacked_generator",
        terrain_generator=FiledTerrainGeneratorCfg(
            size=(9, 12),
            border_width=0.0,
            border_height=0.0,
            num_rows=7,
            num_cols=7,
            add_lights=True,
            sub_terrains={
                # MotionMatchedTerrainCfg keeps motion-terrain pairing from metadata.yaml.
                # Use CoACD collision to avoid MuJoCo mesh convex-hull filling issues.
                "motion_matched": MotionMatchedTerrainCfg(
                    proportion=1.0,
                    path="PLACEHOLDER",  # Will be overridden in concrete env cfg __post_init__
                    metadata_yaml="PLACEHOLDER",  # Will be overridden in concrete env cfg __post_init__
                    collision_coacd=True,
                    # Use CoACD hulls directly as rendered terrain mesh (instead of source STL mesh).
                    collision_coacd_visualize_collision_hulls=True,
                    collision_coacd_threshold=0.04,
                    # Keep hull geometry detailed to avoid blocky/disconnected visual artifacts.
                    collision_coacd_decimate=False,
                    collision_coacd_max_ch_vertex=256,
                    collision_coacd_resolution=3000,
                    # Keep stable runtime/caching behavior.
                    collision_coacd_log_level="off",
                    collision_coacd_use_disk_cache=True,
                    collision_coacd_prewarm_all=True,
                    collision_coacd_prewarm_workers=0,
                    collision_coacd_geom_margin=0.0,
                    collision_coacd_z_offset=0.0,
                    collision_coacd_auto_align_top_surface=True,
                ),
            },
        ),
    ))

    # sensors
    sensors: tuple[SensorCfg, ...] = field(default_factory=lambda: _make_perceptive_base_scene_sensors())


    def __post_init__(self):
        self.spec_fn = _edit_perceptive_scene_spec
        if "robot" not in self.entities:
            raise ValueError("PerceptiveShadowingSceneCfg requires entity 'robot'.")
        if not any(sensor_cfg.name == "motion_reference" for sensor_cfg in self.sensors):
            raise ValueError("PerceptiveShadowingSceneCfg requires sensor 'motion_reference'.")
        motion_reference_cfg = get_motion_reference_cfg(self)
        if (not motion_reference_cfg.debug_vis) and ("robot_reference" in self.entities):
            del self.entities["robot_reference"]


def make_perceptive_scene_entities(
    *,
    robot: EntityCfg,
) -> dict[str, EntityCfg]:
    """Build perceptive scene entities without bridge fields."""
    # robots
    # robot reference articulation
    # motion reference is configured as a sensor cfg ("motion_reference").
    return {"robot": robot}


def make_perceptive_scene_entities_with_reference(
    *,
    robot: EntityCfg,
    robot_reference: EntityCfg,
) -> dict[str, EntityCfg]:
    """Build perceptive scene entities for play/debug with reference robot."""
    return {
        "robot": robot,
        "robot_reference": robot_reference,
    }


def _make_perceptive_height_scanner_sensor_cfg() -> RayCastSensorCfg:
    return RayCastSensorCfg(
        name="height_scanner",
        frame=ObjRef(type="body", name="torso_link", entity="robot"),
        pattern=GridPatternCfg(resolution=0.1, size=(1.6, 1.0)),
        ray_alignment="yaw",
        max_distance=30.0,
        debug_vis=False,
    )


def _make_perceptive_camera_sensor_cfg() -> NoisyGroupedRayCasterCameraCfg:
    return NoisyGroupedRayCasterCameraCfg(
        name="camera",
        frame=ObjRef(type="body", name="torso_link", entity="robot"),
        pattern=PinholeCameraPatternCfg(
            height=int(270 / 10),
            width=int(480 / 10),
            fovy=58.0,
        ),
        focal_length=1.0,
        horizontal_aperture=2 * math.tan(math.radians(87) / 2),  # fovx
        vertical_aperture=2 * math.tan(math.radians(58) / 2),  # fovy
        ray_alignment="base",
        include_geom_groups=(0, 2),
        exclude_parent_body=False,
        offset=NoisyGroupedRayCasterCameraCfg.OffsetCfg(
            pos=(
                0.04764571478 + 0.0039635 - 0.0042 * math.cos(math.radians(48)),
                0.015,
                0.46268178553 - 0.044 + 0.0042 * math.sin(math.radians(48)) + 0.016,
            ),
            rot=(
                math.cos(math.radians(0.5) / 2) * math.cos(math.radians(48) / 2),
                math.sin(math.radians(0.5) / 2),
                math.sin(math.radians(48) / 2),
                0.0,
            ),
            convention="world",
        ),
        data_types=["distance_to_image_plane"],
        mesh_filter_max_hops=24,
        noise_pipeline={
            # "depth_contour_noise": DepthContourNoiseCfg(
            #     contour_threshold=1.8,  # in [m]
            #     maxpool_kernel_size=1,
            # ),
            # "depth_artifact_noise": DepthArtifactNoiseCfg(),
            # "reflection_artifact_noise": DepthArtifactNoiseCfg(noise_value=30.0),
            # "stereo_noise": RangeBasedGaussianNoiseCfg(
            #     max_value=1.2,
            #     min_value=0.12,
            #     noise_std=0.02,
            # ),
            # "sky_artifact_noise": DepthSkyArtifactNoiseCfg(),
            # "gaussian_blur_noise": GaussianBlurNoiseCfg(
            #     kernel_size=3,
            #     sigma=0.5,
            # ),
            # "stereo_too_close_noise": StereoTooCloseNoiseCfg(),
            # These last two noise model will affect the processing on the onboard device.
            "normalize": DepthNormalizationCfg(
                depth_range=(0.0, 2.0),
                normalize=True,
            ),
            "crop_and_resize": CropAndResizeCfg(
                crop_region=(2, 2, 2, 2),
                resize_shape=(18, 32),
            ),
        },
        # data_histories={"distance_to_image_plane": 5},
        update_period=1 / 60,
        debug_vis=False,
        depth_clipping_behavior="max",  # clip to the maximum value
        min_distance=0.05,
        max_distance=1e6,
    )


def _make_perceptive_contact_forces_sensor_cfg() -> ContactSensorCfg:
    return ContactSensorCfg(
        name="contact_forces",
        primary=ContactMatch(mode="body", pattern=".*", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="maxforce",
        history_length=3,
        track_air_time=True,
    )


def _make_perceptive_base_scene_sensors(*, include_height_scanner: bool = True) -> tuple[SensorCfg, ...]:
    sensor_list: list[SensorCfg] = [_make_perceptive_contact_forces_sensor_cfg()]
    if include_height_scanner:
        sensor_list.append(_make_perceptive_height_scanner_sensor_cfg())
    sensor_list.append(_make_perceptive_camera_sensor_cfg())
    return tuple(sensor_list)


def make_perceptive_scene_sensors(
    *,
    motion_reference: MotionReferenceManagerCfg,
    include_height_scanner: bool = True,
) -> tuple[SensorCfg, ...]:
    """Build perceptive scene sensors without bridge fields."""
    # lights are applied in _edit_perceptive_scene_spec.
    return _make_perceptive_base_scene_sensors(include_height_scanner=include_height_scanner) + (motion_reference,)


def get_scene_entity_cfg(scene: InteractiveSceneCfg, entity_name: str) -> EntityCfg:
    """Get scene entity config by name."""
    return scene.entities[entity_name]


def get_scene_sensor_cfg(scene: InteractiveSceneCfg, sensor_name: str) -> SensorCfg:
    """Get scene sensor config by name."""
    return next(
        sensor_cfg
        for sensor_cfg in scene.sensors
        if sensor_cfg.name == sensor_name
    )


def remove_scene_sensor_cfg(scene: InteractiveSceneCfg, sensor_name: str) -> None:
    """Remove a sensor cfg from scene.sensors by name."""
    scene.sensors = tuple(
        sensor_cfg
        for sensor_cfg in scene.sensors
        if sensor_cfg.name != sensor_name
    )


def get_motion_reference_cfg(scene: InteractiveSceneCfg) -> MotionReferenceManagerCfg:
    """Get motion reference sensor cfg from scene."""
    sensor_cfg = get_scene_sensor_cfg(scene, "motion_reference")
    return sensor_cfg  # type: ignore[return-value]


def get_camera_sensor_cfg(scene: InteractiveSceneCfg) -> NoisyGroupedRayCasterCameraCfg:
    """Get camera sensor cfg from scene."""
    sensor_cfg = get_scene_sensor_cfg(scene, "camera")
    return sensor_cfg  # type: ignore[return-value]


def make_perceptive_commands() -> dict[str, instinct_mdp.ShadowingCommandBaseCfg]:
    """Perceptive shadowing command configuration."""
    return {
        "position_ref_command": instinct_mdp.PositionRefCommandCfg(
            realtime_mode=True,
            current_state_command=False,
            anchor_frame="robot",
        ),
        "position_b_ref_command": instinct_mdp.PositionRefCommandCfg(
            realtime_mode=True,
            current_state_command=False,
            anchor_frame="reference",
        ),
        "rotation_ref_command": instinct_mdp.RotationRefCommandCfg(
            realtime_mode=True,
            current_state_command=False,
            in_base_frame=True,
            rotation_mode="tannorm",
        ),
        "joint_pos_ref_command": instinct_mdp.JointPosRefCommandCfg(
            current_state_command=False,
            asset_cfg=SceneEntityCfg(
                "robot",
            ),
        ),
        "joint_vel_ref_command": instinct_mdp.JointVelRefCommandCfg(
            current_state_command=False,
            asset_cfg=SceneEntityCfg(
                "robot",
            ),
        ),
    }



def make_actions() -> dict[str, mdp.JointPositionActionCfg]:
    """Action specifications for the MDP."""
    return {
        "joint_pos": mdp.JointPositionActionCfg(
            entity_name="robot",
            actuator_names=(".*",),
            scale=0.5,
        ),
    }



def make_observations() -> dict[str, ObsGroupCfg]:
    """Observation specifications for the perceptive shadowing MDP."""

    # Policy observations
    actor_terms = {
        # Currently, just a dummy observation
        "joint_pos_ref": ObsTermCfg(func=mdp.generated_commands, params={"command_name": "joint_pos_ref_command"}),
        "joint_vel_ref": ObsTermCfg(func=mdp.generated_commands, params={"command_name": "joint_vel_ref_command"}),
        "position_ref": ObsTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "position_b_ref_command"},
            noise=UniformNoiseCfg(n_min=-0.25, n_max=0.25),
        ),
        "rotation_ref": ObsTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "rotation_ref_command"},
            noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05),
        ),

        # height_scan = ObsTermCfg(
        #     func=mdp.height_scan,
        #     params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        #     clip=[-20.0, 20.0],
        # )
        "depth_image": ObsTermCfg(
            func=instinct_mdp.visualizable_image,
            # params={"sensor_cfg": SceneEntityCfg("camera"), "data_type": "distance_to_image_plane"},
            params={"sensor_cfg": SceneEntityCfg("camera"), "data_type": "distance_to_image_plane_noised"},
        ),

        # proprioception
        "projected_gravity": ObsTermCfg(
            func=mdp.projected_gravity,
            noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05),
            history_length=PROPRIO_HISTORY_LENGTH,
        ),
        # base_lin_vel = ObsTermCfg(func=mdp.base_lin_vel)
        "base_ang_vel": ObsTermCfg(
            func=mdp.base_ang_vel,
            noise=UniformNoiseCfg(n_min=-0.2, n_max=0.2),
            history_length=PROPRIO_HISTORY_LENGTH,
        ),
        "joint_pos": ObsTermCfg(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
            },
            noise=UniformNoiseCfg(n_min=-0.01, n_max=0.01),
            history_length=PROPRIO_HISTORY_LENGTH,
        ),
        "joint_vel": ObsTermCfg(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
            },
            noise=UniformNoiseCfg(n_min=-0.5, n_max=0.5),
            history_length=PROPRIO_HISTORY_LENGTH,
        ),
        "last_action": ObsTermCfg(
            func=mdp.last_action,
            history_length=PROPRIO_HISTORY_LENGTH,
        ),
    }

    # Critic observations for BeyondMimic.
    critic_terms = {
        # BeyondMimic specific reference observations
        "joint_pos_ref": ObsTermCfg(func=mdp.generated_commands, params={"command_name": "joint_pos_ref_command"}),
        "joint_vel_ref": ObsTermCfg(func=mdp.generated_commands, params={"command_name": "joint_vel_ref_command"}),
        "position_ref": ObsTermCfg(func=mdp.generated_commands, params={"command_name": "position_ref_command"}),

        # proprioception
        "link_pos": ObsTermCfg(
            func=instinct_mdp.link_pos_b,
            params={"asset_cfg": SceneEntityCfg(name="robot", body_names=MISSING, preserve_order=True)},
        ),
        "link_rot": ObsTermCfg(
            func=instinct_mdp.link_tannorm_b,
            params={"asset_cfg": SceneEntityCfg(name="robot", body_names=MISSING, preserve_order=True)},
        ),

        "height_scan": ObsTermCfg(
            func=mdp.height_scan,
            params={"sensor_name": "height_scanner"},
            clip=[-20.0, 20.0],
        ),

        "base_lin_vel": ObsTermCfg(
            func=mdp.base_lin_vel,
            history_length=PROPRIO_HISTORY_LENGTH,
        ),
        "base_ang_vel": ObsTermCfg(
            func=mdp.base_ang_vel,
            history_length=PROPRIO_HISTORY_LENGTH,
        ),
        "joint_pos": ObsTermCfg(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
            },
            history_length=PROPRIO_HISTORY_LENGTH,
        ),
        "joint_vel": ObsTermCfg(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
            },
            history_length=PROPRIO_HISTORY_LENGTH,
        ),
        "last_action": ObsTermCfg(
            func=mdp.last_action,
            history_length=PROPRIO_HISTORY_LENGTH,
        ),
    }

    return {
        "actor": ObsGroupCfg(
            terms=actor_terms,
            enable_corruption=True,
            concatenate_terms=False,
        ),
        "critic": ObsGroupCfg(
            terms=critic_terms,
            enable_corruption=False,
            concatenate_terms=False,
        ),
    }



def make_rewards() -> dict[str, RewTermCfg]:
    """Reward specifications for the perceptive shadowing MDP."""
    return {
        "base_position_imitation_gauss": RewTermCfg(
            func=instinct_mdp.base_position_imitation_gauss,
            weight=0.5,
            params={
                "std": 0.3,
            },
        ),

        "base_rot_imitation_gauss": RewTermCfg(
            func=instinct_mdp.base_rot_imitation_gauss,
            weight=0.5,
            params={
                "std": 0.4,
                "difference_type": "axis_angle",
            },
        ),

        "link_pos_imitation_gauss": RewTermCfg(
            func=instinct_mdp.link_pos_imitation_gauss,
            weight=1.0,
            params={
                "combine_method": "mean_prod",
                "in_base_frame": False,
                "in_relative_world_frame": True,
                "std": 0.3,
            },
        ),

        "link_rot_imitation_gauss": RewTermCfg(
            func=instinct_mdp.link_rot_imitation_gauss,
            weight=1.0,
            params={
                "combine_method": "mean_prod",
                "in_base_frame": False,
                "in_relative_world_frame": True,
                "std": 0.4,
            },
        ),

        "link_lin_vel_imitation_gauss": RewTermCfg(
            func=instinct_mdp.link_lin_vel_imitation_gauss,
            weight=1.0,
            params={
                "combine_method": "mean_prod",
                "std": 1.0,
            },
        ),

        "link_ang_vel_imitation_gauss": RewTermCfg(
            func=instinct_mdp.link_ang_vel_imitation_gauss,
            weight=1.0,
            params={
                "combine_method": "mean_prod",
                "std": 3.14,
            },
        ),

        "action_rate_l2": RewTermCfg(func=mdp.action_rate_l2, weight=-0.1),

        "joint_limit": RewTermCfg(
            func=mdp.joint_pos_limits,
            weight=-10.0,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
        ),

        "undesired_contacts": RewTermCfg(
            func=instinct_mdp.undesired_contacts,
            weight=-0.1,
            params={
                "sensor_name": "contact_forces",
                "asset_cfg": SceneEntityCfg("robot", body_names=[_UNDESIRED_CONTACT_BODY_REGEX]),
                "threshold": 1.0,
            },
        ),

        "applied_torque_limits_by_ratio": RewTermCfg(
            func=instinct_mdp.applied_torque_limits_by_ratio,
            weight=-0.05,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        ".*ankle.*",
                        ".*wrist.*",
                    ],
                )
            },
        ),
    }



def make_events() -> dict[str, EventTermCfg]:
    """Event specifications for the perceptive shadowing MDP."""
    return {
        # domain rand
        "physics_material": EventTermCfg(
            func=mdp.dr.geom_friction,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", geom_names=".*"),
                "ranges": {
                    0: (1.25, 2.0),
                    1: (1.2, 1.8),
                    2: (0.0, 0.5),
                },
                "operation": "abs",
                "distribution": "uniform",
            },
        ),

        "add_joint_default_pos": EventTermCfg(
            func=instinct_mdp.randomize_default_joint_pos,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                "offset_distribution_params": (-0.01, 0.01),
                "operation": "add",
                "distribution": "uniform",
            },
        ),

        "base_com": EventTermCfg(
            func=instinct_mdp.randomize_rigid_body_coms,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
                "coms_x_distribution_params": (-0.025, 0.025),
                "coms_y_distribution_params": (-0.05, 0.05),
                "coms_z_distribution_params": (-0.05, 0.05),
                "distribution": "uniform",
            },
        ),

        "randomize_ray_offsets": EventTermCfg(
            func=instinct_mdp.randomize_ray_offsets,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("camera"),
                "offset_pose_ranges": {
                    "x": (-0.01, 0.01),
                    "y": (-0.01, 0.01),
                    "z": (-0.01, 0.01),
                    "roll": (-math.radians(2), math.radians(2)),
                    "pitch": (-math.radians(10), math.radians(10)),
                    "yaw": (-math.radians(2), math.radians(2)),
                },
                "distribution": "uniform",
            },
        ),

        "randomize_actuator_gains": EventTermCfg(
            func=mdp.dr.pd_gains,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "kp_range": (0.8, 1.2),
                "kd_range": (0.9, 1.1),
                "operation": "scale",
                "distribution": "uniform",
            },
        ),

        "randomize_rigid_body_mass": EventTermCfg(
            func=mdp.dr.body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=[
                        "torso_link",
                        "left_ankle.*",
                        "right_ankle.*",
                        "left_wrist.*",
                        "right_wrist.*",
                    ],
                ),
                "ranges": (0.8, 1.2),
                "operation": "scale",
                "distribution": "uniform",
            },
        ),

        "match_motion_ref_with_scene": EventTermCfg(
            func=instinct_mdp.match_motion_ref_with_scene,
            mode="startup",
            params={
                "motion_ref_cfg": SceneEntityCfg("motion_reference"),
            },
        ),

        "reset_robot": EventTermCfg(
            func=instinct_mdp.reset_robot_state_by_reference,
            mode="reset",
            params={
                "motion_ref_cfg": SceneEntityCfg("motion_reference"),
                "asset_cfg": SceneEntityCfg("robot"),
                # reset with position offset to put the robot_reference in scene.
                "position_offset": [0.0, 0.0, 0.0],
                "dof_vel_ratio": 1.0,
                "base_lin_vel_ratio": 1.0,
                "base_ang_vel_ratio": 1.0,
                # Pose randomization (+-5cm position, +-6degrees rotation)
                "randomize_pose_range": {
                    "x": (-0.15, 0.15),
                    "y": (-0.15, 0.15),
                    "z": (0.0, 0.0),
                },
                # Velocity randomization (+-0.1 m/s linear, +-0.1 rad/s angular)
                "randomize_velocity_range": {},
                # Joint position randomization (+-0.1 rad)
                "randomize_joint_pos_range": (-0.1, 0.1),
            },
        ),

        "bin_fail_counter_smoothing": EventTermCfg(
            func=instinct_mdp.beyondmimic_bin_fail_counter_smoothing,
            mode="interval",
            interval_range_s=(0.02, 0.02),  # every environment step
            params={
                "curriculum_name": "beyond_adaptive_sampling",
            },
        ),

        # "push_robot": EventTermCfg(
        #     func=mdp.push_by_setting_velocity,
        #     mode="interval",
        #     interval_range_s=(0.5, 2.0),
        #     params={
        #         "asset_cfg": SceneEntityCfg("robot"),
        #         "velocity_range": {
        #             "x": (-0.5, 0.5),
        #             "y": (-0.5, 0.5),
        #             "z": (-0.2, 0.2),
        #             "roll": (-0.52, 0.52),
        #             "pitch": (-0.52, 0.52),
        #             "yaw": (-0.78, 0.78),
        #         },
        #     },
        # ),
    }


def make_curriculum() -> dict[str, CurriculumTermCfg]:
    """Curriculum specifications for the perceptive shadowing MDP."""
    return {
        "beyond_adaptive_sampling": CurriculumTermCfg(  # type: ignore
            func=instinct_mdp.BeyondConcatMotionAdaptiveWeighting,
        ),
    }



def make_terminations() -> dict[str, DoneTermCfg]:
    """Termination specifications for the perceptive shadowing MDP."""
    return {
        "time_out": DoneTermCfg(func=mdp.time_out, time_out=True),

        "illegal_reset_contact": DoneTermCfg(
            func=instinct_mdp.illegal_reset_contact,
            time_out=True,
            params={
                "sensor_name": "contact_forces",
                "asset_cfg": SceneEntityCfg("robot", body_names=[_UNDESIRED_CONTACT_BODY_REGEX]),
                "threshold": 500,
                "episode_length_threshold": 2,
            },
        ),

        "base_pos_too_far": DoneTermCfg(
            func=instinct_mdp.pos_far_from_ref,
            time_out=False,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "reference_cfg": SceneEntityCfg("motion_reference"),
                "distance_threshold": 0.25,
                "check_at_keyframe_threshold": -1,
                "print_reason": False,
                "height_only": True,
            },
        ),

        "base_pg_too_far": DoneTermCfg(
            func=instinct_mdp.projected_gravity_far_from_ref,
            time_out=False,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "reference_cfg": SceneEntityCfg("motion_reference"),
                "projected_gravity_threshold": 0.8,
                "check_at_keyframe_threshold": -1,
                "z_only": False,
                "print_reason": False,
            },
        ),

        "link_pos_too_far": DoneTermCfg(
            func=instinct_mdp.link_pos_far_from_ref,
            time_out=False,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "reference_cfg": SceneEntityCfg(
                    "motion_reference",
                    body_names=[
                        "left_ankle_roll_link",
                        "right_ankle_roll_link",
                        "left_wrist_yaw_link",
                        "right_wrist_yaw_link",
                    ],
                    preserve_order=True,
                ),
                "distance_threshold": 0.25,
                "in_base_frame": False,
                "check_at_keyframe_threshold": -1,
                "height_only": True,
                "print_reason": False,
            },
        ),

        "dataset_exhausted": DoneTermCfg(
            func=instinct_mdp.dataset_exhausted,
            time_out=True,
            params={
                "reference_cfg": SceneEntityCfg("motion_reference"),
                "print_reason": False,
            },
        ),

        "out_of_border": DoneTermCfg(
            func=instinct_mdp.terrain_out_of_bounds,
            time_out=True,
            params={"asset_cfg": SceneEntityCfg("robot"), "print_reason": False, "distance_buffer": 0.1},
        ),
    }



def make_monitors() -> dict[str, MonitorTermCfg]:
    """Monitor specifications for the perceptive shadowing MDP."""
    return {
        "dataset": MonitorTermCfg(
            func=MotionReferenceMonitorTerm,
            params=dict(
                asset_cfg=SceneEntityCfg("motion_reference"),
                sample_stat_interval=500,
                top_n_samples=5,
            ),
        ),

        "shadowing_position": MonitorTermCfg(
            func=ShadowingPositionMonitorTerm,
            params=dict(
                robot_cfg=SceneEntityCfg("robot"),
                motion_reference_cfg=SceneEntityCfg("motion_reference"),
                in_base_frame=True,
                check_at_keyframe_threshold=0.03,
            ),
        ),

        "shadowing_rotation": MonitorTermCfg(
            func=ShadowingRotationMonitorTerm,
            params=dict(
                robot_cfg=SceneEntityCfg("robot"),
                motion_reference_cfg=SceneEntityCfg("motion_reference"),
                masking=True,
            ),
        ),

        "shadowing_joint_pos": MonitorTermCfg(
            func=ShadowingJointPosMonitorTerm,
            params=dict(
                robot_cfg=SceneEntityCfg("robot"),
                motion_reference_cfg=SceneEntityCfg("motion_reference"),
                masking=True,
            ),
        ),

        "shadowing_joint_vel": MonitorTermCfg(
            func=ShadowingJointVelMonitorTerm,
            params=dict(
                robot_cfg=SceneEntityCfg("robot"),
                motion_reference_cfg=SceneEntityCfg("motion_reference"),
                masking=True,
            ),
        ),

        "shadowing_link_pos_b": MonitorTermCfg(
            func=ShadowingLinkPosMonitorTerm,
            params=dict(
                robot_cfg=SceneEntityCfg("robot"),
                motion_reference_cfg=SceneEntityCfg("motion_reference"),
                in_base_frame=True,
                masking=True,
            ),
        ),

        "shadowing_link_pos_w": MonitorTermCfg(
            func=ShadowingLinkPosMonitorTerm,
            params=dict(
                robot_cfg=SceneEntityCfg("robot"),
                motion_reference_cfg=SceneEntityCfg("motion_reference"),
                in_base_frame=False,
                masking=True,
            ),
        ),
    }



@dataclass(kw_only=True)
class PerceptiveShadowingEnvCfg(InstinctLabRLEnvCfg):
    scene: PerceptiveShadowingSceneCfg = field(default_factory=lambda: PerceptiveShadowingSceneCfg())
    decimation: int = 4
    commands: dict = field(default_factory=make_perceptive_commands)
    actions: dict = field(default_factory=make_actions)
    observations: dict = field(default_factory=make_observations)
    rewards: dict = field(default_factory=make_rewards)
    events: dict = field(default_factory=make_events)
    curriculum: dict = field(default_factory=make_curriculum)
    terminations: dict = field(default_factory=make_terminations)
    monitors: dict = field(default_factory=make_monitors)

    def __post_init__(self):
        # All managers are already dicts, no conversion needed!
        # general settings
        self.decimation = 4
        self.episode_length_s = 10.0
        # simulation settings
        self.sim.mujoco.timestep = 1.0 / 50.0 / self.decimation
        # Use task-level solver caps consistent with mjlab examples instead of
        # relying on MujocoCfg global defaults (100/50).
        self.sim.mujoco.iterations = 10
        self.sim.mujoco.ls_iterations = 20
        self.sim.nconmax = 128
        self.sim.njmax = 512
