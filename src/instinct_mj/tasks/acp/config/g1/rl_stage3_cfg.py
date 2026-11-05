"""Instinct-RL configuration of the G1 ACP Stage 3 task (SuperMimic ``ACPStage3Runner``)."""

from __future__ import annotations

import os
from dataclasses import dataclass

from instinct_mj.rl.acp.stage3_config import ACPStage3Cfg

from .rl_cfg import G1AcpTrackRunnerCfg


@dataclass(kw_only=True)
class G1AcpStage3RunnerCfg(G1AcpTrackRunnerCfg, ACPStage3Cfg):
    """Stage 3: reference tracking + goal reaching with the planner throughout (eta_obs = eta_rew = 1, paper "Stage 3 uses predicted
    contacts throughout"), the planner keeps training on the partitioned replay. Starts from a Stage 2 checkpoint (``--agent.load-run <abs
    run dir> --agent.load-checkpoint model_<it>.pt``: actor, normalizer, world model, critic Vm, failure head, planner) and counts its
    iterations from 0."""

    experiment_name: str = "g1_acp_stage3"
    max_iterations: int = 10000
    contact_source: str = "planner"
    train_planner: bool = True
    restart_iteration: bool = True
    wm_only_iters: int = 0  # the world model and the failure head come from Stage 2 (they keep training on all data)
    fail_head_warmup_iters: int = 0

    def __post_init__(self):
        super().__post_init__()
        self.run_name = "stage3" + (f"_GPU{os.environ['CUDA_VISIBLE_DEVICES']}" if "CUDA_VISIBLE_DEVICES" in os.environ else "")


def g1_acp_stage3_instinct_rl_cfg() -> G1AcpStage3RunnerCfg:
    return G1AcpStage3RunnerCfg()
