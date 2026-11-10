"""G1 Stage-3 environment of ACP: reference tracking mixed with goal reaching on the procedural terrains."""

from dataclasses import dataclass, field

from mjlab.viewer.viewer_config import ViewerConfig

import instinct_mj.tasks.acp.acp_env_cfg as acp_cfg
from instinct_mj.tasks.acp.acp_stage3_env_cfg import AcpStage3EnvCfg, AcpStage3SceneCfg

from .acp_tracking_cfg import _set_motion_terrain, apply_simplified_collision, make_scene_cfg


@dataclass(kw_only=True)
class G1AcpStage3V1EnvCfg(AcpStage3EnvCfg):
    """Stage 3 of the v1 task (``Instinct-ACP-Stage3-G1-v1``): ``goal_env_fraction`` of the envs reach goals on the procedural terrains (10 levels,
    curriculum), the rest tracks the reference motions from phase 0 (the reference-anchored terrains) with the same rewards / terminations / reset as
    Stages 1 / 2 (``acp_cfg.make_*``, masked to the reference envs). No episode cap: the reference envs have to reach 90 % of the clip for the planner
    replay (the longest clip is 18.7 s, the 120 s of the base cfg is only an upper bound); the goal envs have their own 15 s time-out."""

    scene: AcpStage3SceneCfg = field(default_factory=lambda: make_scene_cfg(AcpStage3SceneCfg, num_envs=1024))
    run_name: str = "g1AcpStage3V1"

    simplified_collision: bool = True
    """As in Stage 1 (v1 default: analytic collision primitives for the upper body, sidecar labels not relabelled)."""

    def __post_init__(self):
        # only derived settings: the collision model of the robot / the reference kinematics and the dataset terrain
        if self.simplified_collision:
            apply_simplified_collision(self)
            self.commands[acp_cfg.CONTACTS_COMMAND].allow_label_collision_mismatch = True
        _set_motion_terrain(self)


@dataclass(kw_only=True)
class G1AcpStage3V1EnvCfg_PLAY(G1AcpStage3V1EnvCfg):
    """Evaluation: one goal-reaching env, no domain randomization / observation noise / action delay."""

    # a single reference tile (the goal-reaching env needs only the arena)
    scene: AcpStage3SceneCfg = field(default_factory=lambda: make_scene_cfg(AcpStage3SceneCfg, num_envs=1, env_spacing=2.5, num_tiles=1))
    run_name: str = "g1AcpStage3V1_play"
    viewer: ViewerConfig = field(
        default_factory=lambda: ViewerConfig(
            lookat=(0.0, 0.0, 0.0), distance=2.1213, elevation=45.0, azimuth=0.0,
            origin_type=ViewerConfig.OriginType.WORLD, entity_name="robot", body_name="torso_link",
        )
    )  # fmt: skip

    def __post_init__(self):
        super().__post_init__()
        self.disable_randomization()
        self.commands[acp_cfg.GOAL_COMMAND].goal_env_fraction = 1.0
        for actuator in self.scene.entities["robot"].articulation.actuators:
            actuator.delay_max_lag = 0

