"""Instinct-RL configs for G1 locomotion tasks."""

from __future__ import annotations

from instinct_mjlab.rl import (
  InstinctRlActorCriticCfg,
  InstinctRlOnPolicyRunnerCfg,
  InstinctRlPpoAlgorithmCfg,
)
from instinct_mjlab.tasks.config.rl_utils import default_policy_critic_normalizers


def g1_locomotion_instinct_rl_cfg() -> InstinctRlOnPolicyRunnerCfg:
  return InstinctRlOnPolicyRunnerCfg(
    policy=InstinctRlActorCriticCfg(
      class_name="ActorCritic",
      init_noise_std=1.0,
      actor_hidden_dims=(256, 128, 128),
      critic_hidden_dims=(256, 128, 128),
      activation="elu",
    ),
    algorithm=InstinctRlPpoAlgorithmCfg(
      class_name="PPO",
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.008,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      optimizer_class_name="AdamW",
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
      clip_min_std=1.0e-12,
    ),
    normalizers=default_policy_critic_normalizers(),
    num_steps_per_env=24,
    max_iterations=5_000,
    save_interval=1_000,
    log_interval=10,
    experiment_name="g1_locomotion_flat",
    run_name="",
    policy_observation_group="actor",
    critic_observation_group="critic",
  )
