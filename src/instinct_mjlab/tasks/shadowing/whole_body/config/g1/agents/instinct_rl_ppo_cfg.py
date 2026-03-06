import os

from instinct_mjlab.rl import (
    InstinctRlActorCriticCfg,
    InstinctRlOnPolicyRunnerCfg,
    InstinctRlPpoAlgorithmCfg,
)
from instinct_mjlab.tasks.config.rl_utils import default_policy_critic_normalizers


def _shadowing_policy_cfg() -> InstinctRlActorCriticCfg:
    return InstinctRlActorCriticCfg(
        class_name="MoEActorCritic",
        init_noise_std=1.0,
        actor_hidden_dims=(256, 256, 128),
        critic_hidden_dims=(256, 256, 128),
        activation="elu",
        num_moe_experts=8,
        moe_gate_hidden_dims=(128, 64),
    )


def _shadowing_algorithm_cfg() -> InstinctRlPpoAlgorithmCfg:
    return InstinctRlPpoAlgorithmCfg(
        class_name="PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


def g1_shadowing_ppo_runner_cfg() -> InstinctRlOnPolicyRunnerCfg:
    run_name = "".join(
        [
            "_MoEPolicy",
            f"_GPU{os.environ.get('CUDA_VISIBLE_DEVICES')}" if "CUDA_VISIBLE_DEVICES" in os.environ else "",
        ]
    )
    return InstinctRlOnPolicyRunnerCfg(
        policy=_shadowing_policy_cfg(),
        algorithm=_shadowing_algorithm_cfg(),
        normalizers=default_policy_critic_normalizers(),
        num_steps_per_env=24,
        max_iterations=50000,
        save_interval=1000,
        log_interval=10,
        experiment_name="g1_shadowing",
        run_name=run_name,
        resume=False,
        load_run=".*",
        policy_observation_group="actor",
        critic_observation_group="critic",
    )


def g1_multi_reward_shadowing_ppo_runner_cfg() -> InstinctRlOnPolicyRunnerCfg:
    cfg = g1_shadowing_ppo_runner_cfg()
    cfg.algorithm.advantage_mixing_weights = (0.7, 0.3)
    cfg.run_name += "_Adv622"
    return cfg


def G1ShadowingPPORunnerCfg() -> InstinctRlOnPolicyRunnerCfg:
    """Compatibility callable that returns the whole-body shadowing PPO runner config."""

    return g1_shadowing_ppo_runner_cfg()


def G1MultiRewardShadowingPPORunnerCfg() -> InstinctRlOnPolicyRunnerCfg:
    """Compatibility callable that returns the multi-reward PPO runner config."""

    return g1_multi_reward_shadowing_ppo_runner_cfg()
