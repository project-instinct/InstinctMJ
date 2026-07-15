import os
from dataclasses import dataclass, field

from instinct_mj.envs.mdp.observations.exteroception import visualizable_image
from instinct_mj.rl import (
    InstinctRlActorCriticCfg,
    InstinctRlConv2dHeadCfg,
    InstinctRlEncoderActorCriticCfg,
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
class AlgorithmCfg(InstinctRlPpoAlgorithmCfg):
    class_name: str = "PPO"

    value_loss_coef: float = 1.0

    use_clipped_value_loss: bool = True

    clip_param: float = 0.2

    entropy_coef: float = 0.005

    num_learning_epochs: int = 5

    num_mini_batches: int = 4

    learning_rate: float = 1e-3

    schedule: str = "adaptive"

    gamma: float = 0.99

    lam: float = 0.95

    desired_kl: float = 0.01

    max_grad_norm: float = 1.0


@dataclass(kw_only=True)
class NormalizersCfg:
    policy: InstinctRlNormalizerCfg = field(default_factory=lambda: InstinctRlNormalizerCfg())
    critic: InstinctRlNormalizerCfg = field(default_factory=lambda: InstinctRlNormalizerCfg())


@dataclass(kw_only=True)
class G1PerceptiveShadowingPPORunnerCfg(InstinctRlOnPolicyRunnerCfg):
    policy: PolicyCfg = field(default_factory=lambda: PolicyCfg())
    algorithm: AlgorithmCfg = field(default_factory=lambda: AlgorithmCfg())
    normalizers: NormalizersCfg = field(default_factory=lambda: NormalizersCfg())

    num_steps_per_env: int = 24

    max_iterations: int = 50000

    save_interval: int = 1000

    log_interval: int = 10

    experiment_name: str = "g1_perceptive_shadowing"

    load_run: object | None = None

    def __post_init__(self):
        self.resume = self.load_run is not None
        self.run_name = "".join(
            [
                f"_GPU{os.environ.get('CUDA_VISIBLE_DEVICES')}" if "CUDA_VISIBLE_DEVICES" in os.environ else "",
            ]
        )
