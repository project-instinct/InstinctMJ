"""Instinct-RL configs for G1 locomotion tasks."""

from instinct_mj.rl import InstinctRlOnPolicyRunnerCfg
from instinct_mj.tasks.locomotion.config.g1.agents.instinct_rl_ppo_cfg import G1FlatPPORunnerCfg


def g1_locomotion_instinct_rl_cfg() -> InstinctRlOnPolicyRunnerCfg:
    return G1FlatPPORunnerCfg()
