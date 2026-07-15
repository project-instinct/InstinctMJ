from dataclasses import dataclass, field

from instinct_mj.rl import (
    InstinctRlActorCriticCfg,
    InstinctRlNormalizerCfg,
    InstinctRlOnPolicyRunnerCfg,
    InstinctRlPpoAlgorithmCfg,
)


@dataclass(kw_only=True)
class PolicyCfg(InstinctRlActorCriticCfg):
    init_noise_std: float = 1.0

    actor_hidden_dims: list = field(default_factory=lambda: [256, 128, 128])

    critic_hidden_dims: list = field(default_factory=lambda: [256, 128, 128])

    activation: str = "elu"


@dataclass(kw_only=True)
class AlgorithmCfg(InstinctRlPpoAlgorithmCfg):
    class_name: str = "PPO"
    value_loss_coef: float = 1.0
    use_clipped_value_loss: bool = True
    clip_param: float = 0.2
    entropy_coef: float = 0.008
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
class G1FlatPPORunnerCfg(InstinctRlOnPolicyRunnerCfg):
    policy: PolicyCfg = field(default_factory=lambda: PolicyCfg())
    algorithm: AlgorithmCfg = field(default_factory=lambda: AlgorithmCfg())
    normalizers: NormalizersCfg = field(default_factory=lambda: NormalizersCfg())
    num_steps_per_env: int = 24
    max_iterations: int = 5000
    save_interval: int = 1000
    log_interval: int = 10
    experiment_name: str = "g1_locomotion_flat"

    load_run: object | None = None

    def __post_init__(self):
        self.resume = self.load_run is not None
        self.run_name = ""
