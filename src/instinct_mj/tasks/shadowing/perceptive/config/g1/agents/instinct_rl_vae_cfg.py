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

    depth_image: object = field(default_factory=lambda: DepthImageEncoderCfg())


@dataclass(kw_only=True)
class PolicyCfg(InstinctRlEncoderActorCriticCfg):
    init_noise_std: float = 1.0

    actor_hidden_dims: list = field(default_factory=lambda: [512, 256, 128])

    critic_hidden_dims: list = field(default_factory=lambda: [512, 256, 128])

    activation: str = "elu"

    encoder_configs: object = field(default_factory=lambda: Conv2dHeadEncoderCfg())

    critic_encoder_configs: object | None = None


@dataclass(kw_only=True)
class VaePolicyCfg(InstinctRlEncoderVaeActorCriticCfg):
    encoder_configs: object = field(default_factory=lambda: Conv2dHeadEncoderCfg())

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
class AlgorithmCfg(InstinctRlPpoAlgorithmCfg):
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
class NormalizersCfg:
    policy: InstinctRlNormalizerCfg = field(default_factory=lambda: InstinctRlNormalizerCfg())
    # critic: InstinctRlNormalizerCfg = InstinctRlNormalizerCfg()
    # NOTE: No critic normalizer, must be loaded from the teacher policy.


@dataclass(kw_only=True)
class G1PerceptiveVaePPORunnerCfg(InstinctRlOnPolicyRunnerCfg):
    policy: VaePolicyCfg = field(default_factory=lambda: VaePolicyCfg())
    algorithm: AlgorithmCfg = field(default_factory=lambda: AlgorithmCfg())
    normalizers: NormalizersCfg = field(default_factory=lambda: NormalizersCfg())

    num_steps_per_env: int = 24

    max_iterations: int = 50000

    save_interval: int = 1000

    log_interval: int = 10

    experiment_name: str = "g1_perceptive_vae"

    load_run: object | None = None

    def __post_init__(self):
        super().__post_init__()
        self.resume = self.load_run is not None
        self.run_name = "".join(
            [
                f"_GPU{os.environ.get('CUDA_VISIBLE_DEVICES')}" if "CUDA_VISIBLE_DEVICES" in os.environ else "",
            ]
        )
