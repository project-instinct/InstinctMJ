"""Instinct-RL configs for G1 perceptive shadowing tasks."""

from __future__ import annotations

from instinct_mj.rl import (
    InstinctRlActorCriticCfg,
    InstinctRlNormalizerCfg,
    InstinctRlOnPolicyRunnerCfg,
    InstinctRlPpoAlgorithmCfg,
)
from instinct_mj.tasks.config.rl_utils import default_policy_critic_normalizers


def _perceptive_depth_encoder_cfg() -> dict[str, dict]:
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


def g1_perceptive_shadowing_instinct_rl_cfg() -> InstinctRlOnPolicyRunnerCfg:
    # Use EncoderActorCritic with the depth encoder.
    return InstinctRlOnPolicyRunnerCfg(
        policy=InstinctRlActorCriticCfg(
            class_name="EncoderActorCritic",
            init_noise_std=1.0,
            actor_hidden_dims=(512, 256, 128),
            critic_hidden_dims=(512, 256, 128),
            activation="elu",
            encoder_configs=_perceptive_depth_encoder_cfg(),
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
        save_interval=1_000,
        log_interval=10,
        experiment_name="g1_perceptive_shadowing",
        policy_observation_group="policy",
        critic_observation_group="critic",
    )


def g1_perceptive_shadowing_one_motion_instinct_rl_cfg() -> InstinctRlOnPolicyRunnerCfg:
    cfg = g1_perceptive_shadowing_instinct_rl_cfg()
    cfg.experiment_name = "g1_perceptive_shadowing_one_motion"
    return cfg


def _perceptive_vae_teacher_policy_cfg() -> dict[str, object]:
    return {
        "init_noise_std": 1.0,
        "actor_hidden_dims": [512, 256, 128],
        "critic_hidden_dims": [512, 256, 128],
        "activation": "elu",
        "encoder_configs": {
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
        },
        "critic_encoder_configs": None,
        "obs_format": {
            "policy": {
                "joint_pos_ref": (10, 29),
                "joint_vel_ref": (10, 29),
                "position_ref": (10, 3),
                "rotation_ref": (10, 6),
                "depth_image": (1, 18, 32),
                "projected_gravity": (24,),
                "base_ang_vel": (24,),
                "joint_pos": (232,),
                "joint_vel": (232,),
                "last_action": (232,),
            },
            "critic": {
                "joint_pos_ref": (10, 29),
                "joint_vel_ref": (10, 29),
                "position_ref": (10, 3),
                "link_pos": (14, 3),
                "link_rot": (14, 6),
                "height_scan": (187,),
                "base_lin_vel": (24,),
                "base_ang_vel": (24,),
                "joint_pos": (232,),
                "joint_vel": (232,),
                "last_action": (232,),
            },
        },
        "num_actions": 29,
        "num_rewards": 1,
    }


def g1_perceptive_vae_instinct_rl_cfg() -> InstinctRlOnPolicyRunnerCfg:
    return InstinctRlOnPolicyRunnerCfg(
        policy=InstinctRlActorCriticCfg(
            class_name="EncoderVaeActorCritic",
            init_noise_std=1.0e-4,
            critic_hidden_dims=(512, 256, 128),
            activation="elu",
            encoder_configs=_perceptive_depth_encoder_cfg(),
            critic_encoder_configs=None,
            vae_encoder_kwargs={
                "hidden_sizes": [256, 128, 64],
                "nonlinearity": "ELU",
            },
            vae_decoder_kwargs={
                "hidden_sizes": [512, 256, 128],
                "nonlinearity": "ELU",
            },
            vae_latent_size=16,
            vae_input_subobs_components=("parallel_latent_0_depth_image",),
            vae_aux_subobs_components=(
                "projected_gravity",
                "base_ang_vel",
                "joint_pos",
                "joint_vel",
                "last_action",
            ),
        ),
        algorithm=InstinctRlPpoAlgorithmCfg(
            class_name="VaeDistill",
            kl_loss_func="kl_divergence",
            kl_loss_coef=1.0,
            using_ppo=False,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
            teacher_act_prob=0.2,
            teacher_policy_class_name="EncoderActorCritic",
            teacher_policy=_perceptive_vae_teacher_policy_cfg(),
            teacher_logdir=(
                "~/your/path/to/instinct_mj_logs/instinct_rl/g1_perceptive_shadowing/"
                "your_teacher_run"
            ),
        ),
        normalizers={
            # NOTE: No critic normalizer, must be loaded from the teacher policy.
            "policy": InstinctRlNormalizerCfg(),
        },
        num_steps_per_env=24,
        max_iterations=50_000,
        save_interval=1_000,
        log_interval=10,
        experiment_name="g1_perceptive_vae",
        policy_observation_group="policy",
        critic_observation_group="critic",
    )
