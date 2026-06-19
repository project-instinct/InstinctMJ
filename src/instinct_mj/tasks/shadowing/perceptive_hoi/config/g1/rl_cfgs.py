"""Instinct-RL configs for G1 perceptive HOI shadowing tasks.

Self-contained, mjlab-native port of the upstream
``perceptive_hoi/config/g1/agents/instinct_rl_ppo_cfg.py``
(``G1PerceptiveHoiShadowingPPORunnerCfg``). The hyperparameters intentionally
match the upstream HOI config exactly; upstream HOI shares the same numbers as
the perceptive shadowing config and only differs by ``experiment_name``.
"""

from __future__ import annotations

import os

from instinct_mj.rl import (
    InstinctRlActorCriticCfg,
    InstinctRlOnPolicyRunnerCfg,
    InstinctRlPpoAlgorithmCfg,
)
from instinct_mj.tasks.config.rl_utils import default_policy_critic_normalizers


def _perceptive_hoi_depth_encoder_cfg() -> dict[str, dict]:
    return {
        "depth_image": {
            "class_name": "Conv2dHeadModel",
            "component_names": ["depth_image"],
            "output_size": 32,
            "takeout_input_components": True,
            "channels": [32, 32],
            "kernel_sizes": [3, 3],
            "strides": [1, 1],
            "hidden_sizes": [32],
            "paddings": [1, 1],
            "nonlinearity": "ReLU",
            "use_maxpool": False,
        }
    }


def g1_perceptive_hoi_shadowing_instinct_rl_cfg() -> InstinctRlOnPolicyRunnerCfg:
    # EncoderActorCritic with the depth encoder. Mirrors upstream
    # G1PerceptiveHoiShadowingPPORunnerCfg / PolicyCfg / AlgorithmCfg.
    load_run = None
    run_name = "".join(
        [
            f"_GPU{os.environ.get('CUDA_VISIBLE_DEVICES')}" if "CUDA_VISIBLE_DEVICES" in os.environ else "",
        ]
    )
    return InstinctRlOnPolicyRunnerCfg(
        policy=InstinctRlActorCriticCfg(
            class_name="EncoderActorCritic",
            init_noise_std=1.0,
            actor_hidden_dims=(512, 256, 128),
            critic_hidden_dims=(512, 256, 128),
            activation="elu",
            encoder_configs=_perceptive_hoi_depth_encoder_cfg(),
            critic_encoder_configs=None,
        ),
        algorithm=InstinctRlPpoAlgorithmCfg(
            class_name="PPO",
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.005,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        normalizers=default_policy_critic_normalizers(),
        num_steps_per_env=24,
        max_iterations=50_000,
        save_interval=100,
        log_interval=10,
        experiment_name="g1_perceptive_hoi_shadowing",
        run_name=run_name,
        resume=load_run is not None,
        load_run=load_run,
        policy_observation_group="policy",
        critic_observation_group="critic",
    )
