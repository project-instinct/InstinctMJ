"""Instinct-RL configs for G1 perceptive shadowing tasks."""

from instinct_mj.rl import InstinctRlOnPolicyRunnerCfg
from instinct_mj.tasks.shadowing.perceptive.config.g1.agents.instinct_rl_ppo_cfg import (
    G1PerceptiveShadowingPPORunnerCfg,
)
from instinct_mj.tasks.shadowing.perceptive.config.g1.agents.instinct_rl_vae_cfg import (
    G1PerceptiveVaePPORunnerCfg,
)


def g1_perceptive_shadowing_instinct_rl_cfg() -> InstinctRlOnPolicyRunnerCfg:
    return G1PerceptiveShadowingPPORunnerCfg()


def g1_perceptive_shadowing_one_motion_instinct_rl_cfg() -> InstinctRlOnPolicyRunnerCfg:
    cfg = G1PerceptiveShadowingPPORunnerCfg()
    cfg.experiment_name = "g1_perceptive_shadowing_one_motion"
    return cfg


def g1_perceptive_vae_instinct_rl_cfg() -> InstinctRlOnPolicyRunnerCfg:
    return G1PerceptiveVaePPORunnerCfg()
