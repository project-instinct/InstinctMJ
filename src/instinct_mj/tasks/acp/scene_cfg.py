"""Scene of the ACP tasks: motion-matched terrain tiles, the robot, the sensors and the appearance of the scene.

Everything is defined here (no scene or terrain factory of another task is used): the terrain importer / generator with the tile size of
``terrain_layout.TILE_SIZE``, the contact and height-scan sensors, the spec edit (sky, ground material, terrain friction, geom group of the
ray-cast sensors) and the scene class on mjlab's generic ``SceneCfg``. The motion-reference sensor and the robot entity are passed in by the
robot-specific cfg (``config/g1/acp_tracking_cfg.py``).
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import mujoco
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, GridPatternCfg, ObjRef, SensorCfg
from mjlab.utils.spec_config import MaterialCfg, TextureCfg

import instinct_mj.tasks.acp.mdp as acp_mdp
from instinct_mj.motion_reference.motion_reference_cfg import MotionReferenceManagerCfg
from instinct_mj.rl.acp.planner import CONTACT_LINKS
from instinct_mj.tasks.acp.terrain_layout import TILE_BORDER_WIDTH, TILE_SIZE, CompactFiledTerrainGenerator
from instinct_mj.terrains.terrain_generator_cfg import FiledTerrainGeneratorCfg
from instinct_mj.terrains.terrain_importer_cfg import TerrainImporterCfg
from instinct_mj.terrains.trimesh import MotionMatchedTerrainCfg

# ACP-unspecified: friction of the terrain geoms. MuJoCo mixes the friction of two geoms with max(), so a ground friction
# at the lower bound of the robot friction range makes the contact friction equal to the sampled robot friction
# U(0.3, 1.2) (paper Table S2); without randomization the robot geoms keep their default friction (1.0).
TERRAIN_FRICTION = 0.3

NUM_TILES_PER_DATASET_TERRAIN = 0
"""``terrain_generator.num_rows`` placeholder: one tile per terrain of the dataset (resolved from the metadata by the task cfg)."""


def make_acp_terrain(num_tiles: int = NUM_TILES_PER_DATASET_TERRAIN) -> TerrainImporterCfg:
    """One tile of ``TILE_SIZE`` per terrain of the dataset, laid out on the compact grid (``terrain_layout``); the curriculum layout is a single
    column of ``num_rows`` tiles with difficulty row / (rows - 1), i.e. exactly one tile per terrain, and the motion matching reads the terrain index
    from the tile difficulty. ``num_tiles`` = ``NUM_TILES_PER_DATASET_TERRAIN``: one tile per dataset terrain (resolved by the task cfg); a smaller
    number keeps only the first tiles (evaluation). ``path`` / ``metadata_yaml`` are those of the dataset (set by the task cfg)."""
    return TerrainImporterCfg(
        terrain_type="hacked_generator",
        terrain_generator=FiledTerrainGeneratorCfg(
            class_type=CompactFiledTerrainGenerator,
            size=TILE_SIZE,
            border_width=TILE_BORDER_WIDTH,
            border_height=0.0,
            curriculum=True,
            num_rows=num_tiles,
            num_cols=1,
            add_lights=True,
            sub_terrains={
                "motion_matched": MotionMatchedTerrainCfg(
                    proportion=1.0,
                    path="PLACEHOLDER",  # the dataset folder (task cfg)
                    metadata_yaml="PLACEHOLDER",
                    # the terrains, the motions and the contact labels share one frame (tile centre = env origin)
                    use_input_origin_frame=True,
                    # CoACD hulls for the collision (MuJoCo mesh collision fills non-convex meshes) and as rendered terrain; terrains given as
                    # primitive geoms (all of the ACP datasets) need none of it
                    collision_coacd=True,
                    collision_coacd_visualize_collision_hulls=True,
                    collision_coacd_threshold=0.04,
                    collision_coacd_decimate=False,
                    collision_coacd_max_ch_vertex=256,
                    collision_coacd_resolution=3000,
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
    )


def _edit_acp_scene_spec(spec: mujoco.MjSpec) -> None:
    """Sky, ground material, headlight and light of the terrain, terrain friction and geom group, hidden collision geoms of a reference robot."""
    ground_texture_name = "acp_groundplane"
    ground_material_name = "acp_groundplane"

    sky_rgb_top = (0.98, 0.99, 1.0)
    sky_rgb_horizon = (0.78, 0.86, 0.95)
    ground_rgb1 = (0.74, 0.76, 0.78)
    ground_rgb2 = (0.64, 0.66, 0.68)
    ground_mark_rgb = (0.60, 0.62, 0.64)

    existing_skybox = next((tex for tex in spec.textures if tex.type == mujoco.mjtTexture.mjTEXTURE_SKYBOX), None)
    if existing_skybox is not None:
        existing_skybox.builtin = mujoco.mjtBuiltin.mjBUILTIN_GRADIENT
        existing_skybox.rgb1[:] = sky_rgb_top
        existing_skybox.rgb2[:] = sky_rgb_horizon
        existing_skybox.width = 512
        existing_skybox.height = 3072
    else:
        TextureCfg(
            name="acp_skybox", type="skybox", builtin="gradient", rgb1=sky_rgb_top, rgb2=sky_rgb_horizon, width=512, height=3072
        ).edit_spec(spec)

    TextureCfg(
        name=ground_texture_name, type="2d", builtin="checker", mark="edge", rgb1=ground_rgb1, rgb2=ground_rgb2,
        markrgb=ground_mark_rgb, width=300, height=300,
    ).edit_spec(spec)  # fmt: skip
    MaterialCfg(name=ground_material_name, texuniform=True, texrepeat=(5, 5), reflectance=0.0, texture=ground_texture_name).edit_spec(spec)

    spec.visual.rgba.haze[:] = (0.90, 0.94, 0.98, 1.0)
    spec.visual.headlight.ambient[:] = (0.28, 0.28, 0.28)
    spec.visual.headlight.diffuse[:] = (0.36, 0.36, 0.36)
    spec.visual.headlight.specular[:] = (0.0, 0.0, 0.0)
    spec.visual.quality.shadowsize = 8192  # no jagged shadow edges

    terrain_body = spec.body("terrain")
    for light in terrain_body.lights:  # soft fill light without hard highlights
        light.castshadow = False
        light.ambient[:] = (0.12, 0.12, 0.12)
        light.diffuse[:] = (0.24, 0.24, 0.24)
        light.specular[:] = (0.0, 0.0, 0.0)
    for geom in terrain_body.geoms:
        geom.material = ground_material_name
        geom.rgba[:] = (ground_rgb1[0], ground_rgb1[1], ground_rgb1[2], 1.0)

    # a reference robot (motion-reference debug visualisation) never participates in contacts and is hidden from the depth rays
    for geom in spec.geoms:
        body_name = geom.parent.name or ""
        if body_name == "robot_reference" or body_name.startswith("robot_reference/") or "/robot_reference/" in body_name:
            collision_enabled = int(geom.contype) != 0 or int(geom.conaffinity) != 0
            geom_name = (geom.name or "").lower()
            hide_collision_geom = collision_enabled and (geom.type != mujoco.mjtGeom.mjGEOM_MESH or "collision" in geom_name or "_col" in geom_name)
            geom.contype = 0
            geom.conaffinity = 0
            if hide_collision_geom:
                geom.group = 4
                geom.rgba[:] = (0.0, 0.0, 0.0, 0.0)
            else:
                geom.group = 1

    for geom in terrain_body.geoms:
        geom.friction[0] = TERRAIN_FRICTION
        # the height scanner casts rays against geom group 0 only: the robot (groups 2 / 3) is invisible to it
        geom.group = 0


@dataclass(kw_only=True)
class AcpSceneCfg(SceneCfg):
    """ACP scene: the dataset's terrain tiles (``make_acp_terrain``), the ``robot`` entity and the sensors incl. the ``motion_reference`` sensor."""

    env_spacing: float = 4.0
    terrain: object = field(default_factory=make_acp_terrain)
    sensors: tuple[SensorCfg, ...] = ()
    """Pass ``make_acp_scene_sensors(motion_reference=...)`` (required)."""
    spec_fn: Callable[[mujoco.MjSpec], None] | None = _edit_acp_scene_spec

    def __post_init__(self):
        if "robot" not in self.entities:
            raise ValueError("AcpSceneCfg requires the entity 'robot'.")
        motion_reference = next((sensor for sensor in self.sensors if sensor.name == "motion_reference"), None)
        if motion_reference is None:
            raise ValueError("AcpSceneCfg requires the sensor 'motion_reference'.")
        if not motion_reference.debug_vis:
            self.entities.pop("robot_reference", None)


def make_acp_scene_sensors(*, motion_reference: MotionReferenceManagerCfg) -> tuple[SensorCfg, ...]:
    """Contact sensor over the 9 contact links, the 17 x 11 height scanner and the motion reference."""
    return (
        # all robot bodies vs the terrain: the ``illegal_reset_contact`` termination (a reset into the terrain)
        ContactSensorCfg(
            name="contact_forces",
            primary=ContactMatch(mode="body", pattern=".*", entity="robot"),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("force",),
            reduce="netforce",
            history_length=4,  # one policy step (decimation 4) of physics substeps
        ),
        ContactSensorCfg(
            name="acp_contact_forces",
            primary=ContactMatch(mode="body", pattern=CONTACT_LINKS, entity="robot"),
            secondary=ContactMatch(mode="body", pattern="terrain"),  # terrain contacts only (as the reference labels): no self-contact of the links
            fields=("force",),
            reduce="netforce",
            # one policy step (decimation 4) of physics substeps; measured contact = max over the window of |F_z| > 1 N
            history_length=4,
        ),
        # contact point of the strongest contact of each contact link (the position of the hindsight labels of the planner)
        ContactSensorCfg(
            name="acp_contact_points",
            primary=ContactMatch(mode="body", pattern=CONTACT_LINKS, entity="robot"),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found", "force", "pos"),
            reduce="maxforce",
            num_slots=1,
        ),
        # actor / planner scan: 17 x 11 grid from the estimated pelvis pose (odometry delay and drift), as in the authors'
        # deploy stack (g1_rsl_cpp): rays start 5 m above the pelvis, straight down, yaw of the pelvis
        acp_mdp.PelvisRayCastSensorCfg(
            name="height_scanner",
            frame=ObjRef(type="body", name="pelvis", entity="robot"),
            pattern=GridPatternCfg(resolution=0.1, size=(1.6, 1.0)),
            ray_alignment="yaw",
            max_distance=10.0,
            exclude_parent_body=True,
            # terrain only: the terrain geoms are moved to geom group 0 (``_edit_acp_scene_spec``), the robot geoms are
            # in groups 2 / 3
            include_geom_groups=(0,),
            ray_offset=5.0,
            use_estimated_pose=True,
            debug_vis=False,
        ),
        # critic scan: the same grid from the true pelvis pose (no delay, no drift)
        acp_mdp.PelvisRayCastSensorCfg(
            name="critic_height_scanner",
            frame=ObjRef(type="body", name="pelvis", entity="robot"),
            pattern=GridPatternCfg(resolution=0.1, size=(1.6, 1.0)),
            ray_alignment="yaw",
            max_distance=10.0,
            exclude_parent_body=True,
            include_geom_groups=(0,),
            ray_offset=5.0,
            use_estimated_pose=False,
            debug_vis=False,
        ),
        motion_reference,
    )

