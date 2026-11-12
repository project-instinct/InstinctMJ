"""Instinct-RL configuration of the G1 ACP tracking task (SuperMimic ``ACPRunner``)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from instinct_mj.rl import InstinctRlActorCriticCfg, InstinctRlNormalizerCfg, InstinctRlOnPolicyRunnerCfg
from instinct_mj.rl.acp.config import ACPRunnerCfg



@dataclass(kw_only=True)
class G1AcpTrackRunnerCfg(InstinctRlOnPolicyRunnerCfg, ACPRunnerCfg):
    """Agent cfg of the ACP tracking task: the actor MLP (512-256-128 ELU on the ``policy`` group, 468 -> 23) and its observation
    normalizer plus the ``ACPRunnerCfg`` hyperparameters (flat, ``--agent.<field>``). The runner's critic is its own network on
    privileged features, so there is no critic normalizer, no PPO algorithm and no ``num_steps_per_env``."""

    policy: InstinctRlActorCriticCfg = field(
        default_factory=lambda: InstinctRlActorCriticCfg(
            class_name="ActorCritic",
            init_noise_std=1.0,
            actor_hidden_dims=(512, 256, 128),
            critic_hidden_dims=(512, 256, 128),
            activation="elu",
        )
    )
    algorithm: None = None
    normalizers: dict = field(default_factory=lambda: {"policy": InstinctRlNormalizerCfg()})
    num_steps_per_env: None = None
    max_iterations: int = 30000
    save_interval: int = 1000
    log_interval: int = 10
    experiment_name: str = "g1AcpTrackV1"
    load_run: str | None = None
    policy_observation_group: str = "policy"
    critic_observation_group: str = "critic"

    def __post_init__(self):
        self.resume = self.load_run is not None
        self.run_name = "supermimic" + "".join(
            [
                f"_GPU{os.environ.get('CUDA_VISIBLE_DEVICES')}" if "CUDA_VISIBLE_DEVICES" in os.environ else "",
            ]
        )


def g1_acp_track_instinct_rl_cfg() -> G1AcpTrackRunnerCfg:
    return G1AcpTrackRunnerCfg()


@dataclass(kw_only=True)
class G1AcpStage2RunnerCfg(G1AcpTrackRunnerCfg):
    """Stage 2: the tracker follows the contact commands of the planner (paper Eq. 3) while the planner is trained on the hindsight
    labels of the successful episodes. Starts from a Stage 1 checkpoint (``--agent.load-run <abs run dir> --agent.load-checkpoint
    model_<it>.pt``: actor, normalizer, world model, critic, failure head) and counts its iterations from 0."""

    experiment_name: str = "g1AcpStage2V1"
    max_iterations: int = 10000
    contact_source: str = "mixed"
    # ACP-unspecified (the paper gives the form of the schedules, not u_obs / u_rew / N_ann): the planner trains alone for the first 1000
    # iterations (``planner_updates_per_iter`` = 1 optimizer step each) before the tracker sees any of its commands (eta_obs leaves 0); the reward keeps following the
    # reference until 3000, when the tracker has adapted to the planner commands for ~2000 iterations (u_rew > u_obs, Eq. 3); both ramp over
    # the same 4000 iterations, so eta_obs = 1 at iteration 5000 and eta_rew = 1 at 7000. Counted from the start of Stage 2.
    eta_obs_start_iter: int = 1000
    eta_rew_start_iter: int = 3000
    eta_ann_iters: int = 4000
    train_planner: bool = True
    planner_updates_per_iter: int = 1  # one planner optimizer step per iteration (v1 decision; the ``ACPRunnerCfg`` default is 10, Table S5)
    restart_iteration: bool = True
    wm_only_iters: int = 0  # the world model comes from Stage 1

    def __post_init__(self):
        super().__post_init__()
        self.run_name = "stage2" + (f"_GPU{os.environ['CUDA_VISIBLE_DEVICES']}" if "CUDA_VISIBLE_DEVICES" in os.environ else "")


def g1_acp_stage2_instinct_rl_cfg() -> G1AcpStage2RunnerCfg:
    return G1AcpStage2RunnerCfg()

