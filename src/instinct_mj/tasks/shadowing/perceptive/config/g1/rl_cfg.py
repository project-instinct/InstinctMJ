"""Instinct-RL configurations for G1 perceptive shadowing tasks."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from instinct_mj.envs.mdp.observations.exteroception import visualizable_image
from instinct_mj.rl import (
    InstinctRlActorCriticCfg,
    InstinctRlConv2dHeadCfg,
    InstinctRlEncoderActorCriticCfg,
    InstinctRlEncoderVaeActorCriticCfg,
    InstinctRlNormalizerCfg,
    InstinctRlOnPolicyRunnerCfg,
    InstinctRlPpoAlgorithmCfg,
)
from instinct_mj.rl.supermimic.config import SuperMimicRunnerCfg


@dataclass(kw_only=True)
class Conv2dHeadEncoderCfg:
    @dataclass(kw_only=True)
    class DepthImageEncoderCfg(InstinctRlConv2dHeadCfg):
        channels: list = field(default_factory=lambda: [32, 32])

        kernel_sizes: list = field(default_factory=lambda: [3, 3])

        strides: list = field(default_factory=lambda: [1, 1])

        paddings: list = field(default_factory=lambda: [1, 1])

        hidden_sizes: list = field(
            default_factory=lambda: [
                32,
            ]
        )

        nonlinearity: str = "ReLU"

        use_maxpool: bool = False

        output_size: int = 32

        component_names: list = field(default_factory=lambda: ["depth_image"])

        takeout_input_components: bool = True

    depth_image: object = field(default_factory=DepthImageEncoderCfg)


@dataclass(kw_only=True)
class PolicyCfg(InstinctRlEncoderActorCriticCfg):
    init_noise_std: float = 1.0

    actor_hidden_dims: list = field(default_factory=lambda: [512, 256, 128])

    critic_hidden_dims: list = field(default_factory=lambda: [512, 256, 128])

    activation: str = "elu"

    encoder_configs: object = field(default_factory=lambda: Conv2dHeadEncoderCfg())

    critic_encoder_configs: object | None = None  # No encoder for critic


# No encoder for critic


@dataclass(kw_only=True)
class NormalizersCfg:
    policy: InstinctRlNormalizerCfg = field(default_factory=lambda: InstinctRlNormalizerCfg())
    critic: InstinctRlNormalizerCfg = field(default_factory=lambda: InstinctRlNormalizerCfg())


@dataclass(kw_only=True)
class G1PerceptiveShadowingRunnerCfg(InstinctRlOnPolicyRunnerCfg, SuperMimicRunnerCfg):
    """Agent cfg of the perceptive shadowing task, trained with SuperMimic (``SuperMimicRunner``): the policy architecture
    and observation normalizers of the task plus the SuperMimic hyperparameters (flat, ``--agent.<field>``). There is no
    PPO algorithm: ``algorithm`` and ``num_steps_per_env`` stay unset."""

    policy: PolicyCfg = field(default_factory=lambda: PolicyCfg())
    algorithm: None = None
    normalizers: NormalizersCfg = field(default_factory=lambda: NormalizersCfg())

    num_steps_per_env: None = None

    max_iterations: int = 320_000

    save_interval: int = 2000

    log_interval: int = 50

    experiment_name: str = "g1_perceptive_shadowing"

    load_run: str | None = None

    def __post_init__(self):
        self.resume = self.load_run is not None
        self.run_name = "supermimic" + "".join(
            [
                f"_GPU{os.environ.get('CUDA_VISIBLE_DEVICES')}" if "CUDA_VISIBLE_DEVICES" in os.environ else "",
            ]
        )


@dataclass(kw_only=True)
class VaeConv2dHeadEncoderCfg:
    @dataclass(kw_only=True)
    class DepthImageEncoderCfg(InstinctRlConv2dHeadCfg):
        channels: list = field(default_factory=lambda: [32, 32])

        kernel_sizes: list = field(default_factory=lambda: [3, 3])

        strides: list = field(default_factory=lambda: [1, 1])

        paddings: list = field(default_factory=lambda: [1, 1])

        hidden_sizes: list = field(
            default_factory=lambda: [
                32,
            ]
        )

        nonlinearity: str = "ReLU"

        use_maxpool: bool = False

        output_size: int = 32

        component_names: list = field(default_factory=lambda: ["depth_image"])

        takeout_input_components: bool = True

    depth_image: object = field(default_factory=DepthImageEncoderCfg)


@dataclass(kw_only=True)
class VaeTeacherPolicyCfg(InstinctRlEncoderActorCriticCfg):
    init_noise_std: float = 1.0

    actor_hidden_dims: list = field(default_factory=lambda: [512, 256, 128])

    critic_hidden_dims: list = field(default_factory=lambda: [512, 256, 128])

    activation: str = "elu"

    encoder_configs: object = field(default_factory=lambda: VaeConv2dHeadEncoderCfg())

    critic_encoder_configs: object | None = None


@dataclass(kw_only=True)
class VaePolicyCfg(InstinctRlEncoderVaeActorCriticCfg):
    init_noise_std: float = 1e-4

    critic_hidden_dims: list = field(default_factory=lambda: [512, 256, 128])

    activation: str = "elu"

    encoder_configs: object = field(default_factory=lambda: VaeConv2dHeadEncoderCfg())

    vae_encoder_kwargs: dict = field(
        default_factory=lambda: {
            "hidden_sizes": [256, 128, 64],
            "nonlinearity": "ELU",
        }
    )

    vae_decoder_kwargs: dict = field(
        default_factory=lambda: {
            "hidden_sizes": [512, 256, 128],
            "nonlinearity": "ELU",
        }
    )

    vae_latent_size: int = 16

    vae_input_subobs_components: list = field(
        default_factory=lambda: [
            "parallel_latent_0_depth_image",  # based on the encoder_configs in Conv2dHeadEncoderCfg
            # "projected_gravity",
            # "base_ang_vel",
            # "joint_pos",
            # "joint_vel",
            # "last_action",
        ]
    )

    vae_aux_subobs_components: list = field(
        default_factory=lambda: [
            # "parallel_latent_0_depth_image",
            "projected_gravity",
            "base_ang_vel",
            "joint_pos",
            "joint_vel",
            "last_action",
        ]
    )


@dataclass(kw_only=True)
class VaeAlgorithmCfg(InstinctRlPpoAlgorithmCfg):
    class_name: str = "VaeDistill"

    kl_loss_func: str = "kl_divergence"

    kl_loss_coef: float = 1.0

    using_ppo: bool = False

    num_learning_epochs: int = 5

    num_mini_batches: int = 4

    learning_rate: float = 1e-3

    # PPO parameters should not affect anything.
    schedule: str = "adaptive"

    gamma: float = 0.99

    lam: float = 0.95

    desired_kl: float = 0.01

    max_grad_norm: float = 1.0

    teacher_act_prob: float = 0.2

    # update_times_scale = 20 * int(1e3)

    teacher_policy_class_name: object = field(default_factory=lambda: InstinctRlEncoderActorCriticCfg().class_name)

    teacher_policy: dict = field(
        default_factory=lambda: {
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
    )
    teacher_logdir: object = field(
        default_factory=lambda: os.path.expanduser(
            "~/Xyk/Project-Instinct/InstinctMJ/logs/instinct_rl/g1_perceptive_shadowing/2026-03-07_15-29-45"
        )
    )


@dataclass(kw_only=True)
class VaeNormalizersCfg:
    policy: InstinctRlNormalizerCfg = field(default_factory=lambda: InstinctRlNormalizerCfg())
    # critic: InstinctRlNormalizerCfg = InstinctRlNormalizerCfg()
    # NOTE: No critic normalizer, must be loaded from the teacher policy.


@dataclass(kw_only=True)
class G1PerceptiveVaePPORunnerCfg(InstinctRlOnPolicyRunnerCfg):
    policy: VaePolicyCfg = field(default_factory=lambda: VaePolicyCfg())
    algorithm: VaeAlgorithmCfg = field(default_factory=lambda: VaeAlgorithmCfg())
    normalizers: VaeNormalizersCfg = field(default_factory=lambda: VaeNormalizersCfg())

    num_steps_per_env: int = 24

    max_iterations: int = 50000

    save_interval: int = 1000

    log_interval: int = 10

    experiment_name: str = "g1_perceptive_vae"

    load_run: object | None = None

    def __post_init__(self):
        self.resume = self.load_run is not None
        self.run_name = "".join(
            [
                f"_GPU{os.environ.get('CUDA_VISIBLE_DEVICES')}" if "CUDA_VISIBLE_DEVICES" in os.environ else "",
            ]
        )


def g1_perceptive_shadowing_latent_instinct_rl_cfg() -> InstinctRlOnPolicyRunnerCfg:
    """SuperMimic with the ``latent`` vision method: the world model predicts the visual latent of the depth encoder, which is
    trained by a reconstruction decoder; nothing is rendered in imagination. Setting of the 96.9% run (no failure head)."""
    cfg = G1PerceptiveShadowingRunnerCfg()
    cfg.vision = "latent"
    cfg.fail_head = False
    cfg.wm_failure_frac = 0.0
    cfg.wm_holdout_envs = 0.0
    cfg.max_iterations = 8000
    return cfg


def g1_perceptive_shadowing_render_instinct_rl_cfg() -> InstinctRlOnPolicyRunnerCfg:
    """SuperMimic with the ``render`` vision method: the depth image of every imagined state is rendered; the depth encoder gets
    no policy gradient, it is trained by a reconstruction decoder and a value head annealed to 0 over iterations 3000 -> 6000."""
    cfg = G1PerceptiveShadowingRunnerCfg()
    cfg.vision = "render"
    cfg.aux_recon_weight = 1.0
    cfg.aux_value_weight = 1.0
    cfg.aux_value_anneal_start = 3000
    cfg.aux_value_anneal_end = 6000
    cfg.max_iterations = 8000
    return cfg


def g1_perceptive_shadowing_one_motion_instinct_rl_cfg() -> InstinctRlOnPolicyRunnerCfg:
    cfg = g1_perceptive_shadowing_render_instinct_rl_cfg()
    cfg.max_iterations = G1PerceptiveShadowingRunnerCfg.max_iterations
    cfg.experiment_name = "g1_perceptive_shadowing_one_motion"
    return cfg


def g1_perceptive_vae_instinct_rl_cfg() -> InstinctRlOnPolicyRunnerCfg:
    return G1PerceptiveVaePPORunnerCfg()
