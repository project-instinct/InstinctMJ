"""G1 Stage-3 environment of ACP: reference tracking mixed with goal reaching on the procedural terrains."""

from copy import deepcopy
from dataclasses import dataclass, field

from mjlab.viewer.viewer_config import ViewerConfig

import instinct_mj.tasks.acp.acp_env_cfg as acp_cfg
from instinct_mj.tasks.acp.acp_stage3_env_cfg import AcpStage3EnvCfg, AcpStage3SceneCfg

from .acp_tracking_cfg import G1_CFG, _set_motion_terrain, apply_supermimic_deltas, motion_reference_cfg


@dataclass(kw_only=True)
class G1AcpStage3EnvCfg(AcpStage3EnvCfg):
    """Stage 3: ``goal_env_fraction`` of the envs reach goals on the procedural terrains (10 levels, curriculum), the rest tracks the
    reference motions from phase 0 (the reference-anchored terrains)."""

    scene: AcpStage3SceneCfg = field(
        default_factory=lambda: AcpStage3SceneCfg(
            num_envs=1024,
            entities={"robot": deepcopy(G1_CFG)},
            sensors=acp_cfg.make_acp_scene_sensors(motion_reference=deepcopy(motion_reference_cfg)),
        )
    )

    def __post_init__(self):
        super().__post_init__()
        _set_motion_terrain(self)
        self.run_name = "g1AcpStage3"


@dataclass(kw_only=True)
class G1AcpStage3SMEnvCfg(G1AcpStage3EnvCfg):
    """Stage 3 with the SuperMimic deltas (``apply_supermimic_deltas``) for its reference envs (reset noise, tilt / link terminations, joint limits, contact_acc); no episode cap: the reference
    envs start at phase 0 and have to reach 90 % of the clip for the planner replay, the goal envs have their own reset and 15 s time-out."""

    def __post_init__(self):
        super().__post_init__()
        apply_supermimic_deltas(self, None)
        self.run_name = "g1AcpStage3SM"


@dataclass(kw_only=True)
class G1AcpStage3EnvCfg_PLAY(G1AcpStage3EnvCfg):
    """Evaluation: one goal-reaching env, no domain randomization / observation noise / action delay."""

    scene: AcpStage3SceneCfg = field(
        default_factory=lambda: AcpStage3SceneCfg(
            num_envs=1,
            env_spacing=2.5,
            entities={"robot": deepcopy(G1_CFG)},
            sensors=acp_cfg.make_acp_scene_sensors(motion_reference=deepcopy(motion_reference_cfg)),
        )
    )
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
        self.scene.terrain.terrain_generator.num_rows = 1
        self.scene.terrain.terrain_generator.num_cols = 1
        self.run_name += "_play"
