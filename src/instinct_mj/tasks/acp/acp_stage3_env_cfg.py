"""Stage-3 environment of ACP: reference tracking mixed with goal reaching on procedural terrains (paper Fig. 3, Table 3 Stage-3 column).

A fixed fraction of the envs follows a reference motion on the reference-anchored terrains (the Stage 2 setup, initial phase 0) and the rest
reaches goals on the procedural terrains of ``goal_terrains.py`` (10 difficulty levels per terrain type, success-gated promotion). The
mimic tracking terms are masked to the reference envs, the goal terms (``move2goal``, ``standatgoal`` and the goal style terms) to the goal
envs; the reference-contact terms are disabled and ``track_planned_contact`` keeps its weight 1.5 (the planner targets are supplied by the
runner). Values the paper does not give are marked ``# ACP-unspecified``.
"""

from __future__ import annotations

from dataclasses import MISSING, dataclass, field

import numpy as np
from mjlab.managers import CurriculumTermCfg, EventTermCfg
from mjlab.managers import RewardTermCfg as RewTermCfg
from mjlab.managers import SceneEntityCfg
from mjlab.managers import TerminationTermCfg as DoneTermCfg

import instinct_mj.tasks.acp.acp_env_cfg as acp_cfg
import instinct_mj.tasks.acp.mdp as acp_mdp
from instinct_mj.managers import MultiRewardCfg
from instinct_mj.tasks.acp.goal_terrains import GoalArena, GoalArenaCfg
from instinct_mj.tasks.acp.mdp import goal as goal_mdp
from instinct_mj.tasks.acp.scene_cfg import AcpSceneCfg, _edit_acp_scene_spec
from instinct_mj.tasks.acp.terrain_layout import grid_bounds


@dataclass(kw_only=True)
class AcpStage3SceneCfg(AcpSceneCfg):
    """ACP scene with the arena of procedural goal-reaching terrains added to the terrain body, beside the motion-matched reference tiles.

    The arena is placed from the actual reference tile grid (``terrain_layout.grid_bounds`` of the terrain generator's final tile count and size, read when the
    layout is used), so the simulated geometry, the arena tables (spawn / goal), and the analytic height maps (built from the compiled model) share one layout."""

    arena_cfg: GoalArenaCfg = field(default_factory=GoalArenaCfg)
    # ``self.arena`` (a ``GoalArena``) is built in ``__post_init__`` and deliberately not a dataclass field: the command-line parser would rebuild a field
    # from the signature of ``GoalArena`` and lose the link to the reference tile grid.

    def __post_init__(self):
        super().__post_init__()
        self.arena = GoalArena(self.arena_cfg, reference_bounds=self.reference_bounds)
        self.spec_fn = self.edit_spec

    def reference_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """(lo, hi) xy rectangle of the reference tile grid."""
        generator = self.terrain.terrain_generator
        if generator.num_rows <= 0:
            raise ValueError("The reference terrain has no tile count yet (the task cfg resolves it from the dataset).")
        return grid_bounds(generator.num_rows * generator.num_cols, tuple(generator.size))

    def edit_spec(self, spec) -> None:
        self.arena.add_to_spec(spec)  # before the terrain edits (friction, geom group of the ray-cast sensors)
        _edit_acp_scene_spec(spec)


def make_stage3_commands() -> dict:
    return {
        acp_cfg.CONTACTS_COMMAND: acp_mdp.AcpReferenceContactsCfg(),
        acp_cfg.GOAL_COMMAND: goal_mdp.AcpStage3GoalCommandCfg(),
    }


def make_stage3_rewards() -> dict[str, RewTermCfg]:
    """Paper Table 3, Stage 3: the reference tracking terms of ``acp_cfg.make_rewards`` (the same definition as Stages 1 / 2, incl. the SuperMimic
    body terms) on the reference envs, the three reference-contact terms disabled (``1 - eta_rew = 0``; ``track_body_pos_anchor`` fades with them
    in the runner, ``ACPMDP.weights``), ``track_planned_contact`` 1.5 (evaluated against the reference targets by the env, the runner's reward
    uses the planner targets), the goal terms on the goal envs and the goal style terms."""
    rewards = acp_cfg.make_rewards()
    for name, func in (
        ("track_base_pos", goal_mdp.track_base_pos_ref), ("track_base_vel", goal_mdp.track_base_vel_ref),
        ("track_joint_pos", goal_mdp.track_joint_pos_ref), ("track_joint_vel", goal_mdp.track_joint_vel_ref),
        ("track_contact_label", goal_mdp.track_contact_label_ref), ("track_contact_pos_base", goal_mdp.track_contact_pos_base_ref),
        ("track_contact_pos_global", goal_mdp.track_contact_pos_global_ref), ("track_planned_contact", goal_mdp.track_planned_contact_ref),
        ("track_body_pos_anchor", goal_mdp.track_body_pos_anchor_ref), ("track_body_ori", goal_mdp.track_body_ori_ref),
        ("track_body_lin_vel", goal_mdp.track_body_lin_vel_ref), ("track_body_ang_vel", goal_mdp.track_body_ang_vel_ref),
    ):  # fmt: skip
        rewards[name].func = func
    for name in ("track_contact_label", "track_contact_pos_base", "track_contact_pos_global"):
        rewards[name].weight = 0.0
    rewards["track_planned_contact"].weight = 1.5
    rewards.update({
        "move2goal": RewTermCfg(func=goal_mdp.move2goal, weight=40.0, params={"radius": 0.5, "min_speed": 0.1, "z_speed": 0.1,
                                                                              "z_tolerance": 0.2, "stair_speed_cap": 0.8, "speed_cap": 2.0}),
        "standatgoal": RewTermCfg(func=goal_mdp.standatgoal, weight=20.0, params={
            "force_threshold": 1.0, "radius": 0.5, "z_tolerance": 0.2, "heading_tolerance": 0.5,
            "asset_cfg": SceneEntityCfg("robot", joint_names=acp_cfg.ACTION_JOINT_NAMES)}),
        "flat_orientation": RewTermCfg(func=goal_mdp.flat_orientation, weight=-10.0),
        "stumbling": RewTermCfg(func=goal_mdp.stumbling, weight=-10.0),
        "slippage": RewTermCfg(func=goal_mdp.slippage, weight=-10.0, params={"force_threshold": 1.0, "min_contact_time": 0.026, "max_speed": 0.5}),
        "ang_vel_xy": RewTermCfg(func=goal_mdp.ang_vel_xy, weight=-0.1),
    })  # fmt: skip
    return rewards


def make_stage3_events() -> dict[str, EventTermCfg]:
    """The Stage 2 events with the reference reset restricted to the reference envs and the goal-env reset (tile, spawn, goal) after it."""
    events = acp_cfg.make_events()
    out: dict[str, EventTermCfg] = {}
    for name, event in events.items():
        if name == "reset_robot":
            event = EventTermCfg(func=goal_mdp.reset_reference_envs, mode="reset", params={**event.params, "command_name": acp_cfg.GOAL_COMMAND})
            out[name] = event
            out["reset_goal"] = EventTermCfg(func=goal_mdp.reset_goal_envs, mode="reset", params={"command_name": acp_cfg.GOAL_COMMAND, "asset_cfg": SceneEntityCfg("robot")})
        else:
            out[name] = event
    return out


def make_stage3_curriculum() -> dict[str, CurriculumTermCfg]:
    """Episode start of the reference envs (Stage 3: initial phase 0, mode "start") and the terrain curriculum of the goal envs."""
    curriculum = acp_cfg.make_curriculum()
    curriculum[acp_cfg.CURRICULUM_NAME].params["mode"] = "start"
    curriculum["goal_terrain"] = CurriculumTermCfg(
        func=goal_mdp.GoalTerrainCurriculum,
        params={"command_name": acp_cfg.GOAL_COMMAND, "ema_weight": 0.2, "promote_ema": 0.5, "promote_distance": 0.5, "demote_distance": 1.0},
    )
    return curriculum


def make_stage3_terminations() -> dict[str, DoneTermCfg]:
    """The v1 reference terminations (``acp_cfg.make_terminations``) masked to the reference envs; goal envs: the 15 s time-out and the fall."""
    terminations = acp_cfg.make_terminations()
    terminations["dataset_exhausted"].func = goal_mdp.dataset_exhausted_ref_only
    terminations["base_pos_too_far"].func = goal_mdp.pos_far_from_ref_ref_only
    terminations["base_pg_too_far"].func = goal_mdp.projected_gravity_far_from_ref_ref_only
    terminations["base_xy_too_far"].func = goal_mdp.pos_xy_far_from_ref_ref_only
    terminations["foot_pos_too_far"].func = goal_mdp.feet_far_from_ref_ref_only
    terminations["goal_time_out"] = DoneTermCfg(func=goal_mdp.goal_time_out, time_out=True)
    terminations["goal_fall"] = DoneTermCfg(func=goal_mdp.goal_fall, time_out=False)
    return terminations


@dataclass(kw_only=True)
class AcpStage3EnvCfg(acp_cfg.AcpTrackEnvCfg):
    scene: AcpStage3SceneCfg = MISSING
    commands: dict = field(default_factory=make_stage3_commands)
    rewards: dict = field(default_factory=lambda: MultiRewardCfg({"rewards": make_stage3_rewards()}))
    events: dict = field(default_factory=make_stage3_events)
    curriculum: dict = field(default_factory=make_stage3_curriculum)
    terminations: dict = field(default_factory=make_stage3_terminations)
