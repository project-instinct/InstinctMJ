"""Instinct-RL configuration dataclasses for mjlab integration."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Sequence


def _to_plain_dict(obj: Any) -> Any:
    """Recursively convert dataclass objects into plain dictionaries."""
    if is_dataclass(obj):
        result = {}
        for item in fields(obj):
            value = getattr(obj, item.name)
            if value is None:
                continue
            result[item.name] = _to_plain_dict(value)
        return result
    if isinstance(obj, dict):
        return {key: _to_plain_dict(value) for key, value in obj.items()}
    if isinstance(obj, tuple):
        return [_to_plain_dict(value) for value in obj]
    if isinstance(obj, list):
        return [_to_plain_dict(value) for value in obj]
    return obj


@dataclass
class InstinctRlActorCriticCfg:
    class_name: str = "ActorCritic"
    init_noise_std: float = 1.0
    actor_hidden_dims: tuple[int, ...] = (256, 128, 128)
    critic_hidden_dims: tuple[int, ...] = (256, 128, 128)
    activation: str = "elu"
    num_moe_experts: int | None = None
    moe_gate_hidden_dims: tuple[int, ...] | None = None
    encoder_configs: dict[str, Any] | None = None
    critic_encoder_configs: dict[str, Any] | str | None = None
    vae_encoder_kwargs: dict[str, Any] | None = None
    vae_decoder_kwargs: dict[str, Any] | None = None
    vae_latent_size: int | None = None
    vae_input_subobs_components: tuple[str, ...] | None = None
    vae_aux_subobs_components: tuple[str, ...] | None = None


@dataclass
class InstinctRlPpoAlgorithmCfg:
    class_name: str = "PPO"
    value_loss_coef: float = 1.0
    use_clipped_value_loss: bool = True
    clip_param: float = 0.2
    entropy_coef: float = 0.005
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    learning_rate: float = 1.0e-3
    optimizer_class_name: str = "AdamW"
    schedule: str = "adaptive"
    gamma: float = 0.99
    lam: float = 0.95
    advantage_mixing_weights: float | tuple[float, ...] = 1.0
    desired_kl: float = 0.01
    max_grad_norm: float = 1.0
    clip_min_std: float = 1.0e-12
    kl_loss_func: str | None = None
    kl_loss_coef: float | None = None
    using_ppo: bool | None = None
    teacher_policy_class_name: str | None = None
    teacher_policy: dict[str, Any] | None = None
    teacher_logdir: str | None = None
    label_action_with_critic_obs: bool | None = None
    teacher_act_prob: str | float | None = None
    update_times_scale: int | None = None
    distillation_loss_coef: float | str | None = None
    distill_target: str | None = None
    actor_state_key: str | None = None
    reference_state_key: str | None = None
    discriminator_class_name: str | None = None
    discriminator_kwargs: dict[str, Any] | None = None
    discriminator_optimizer_class_name: str | None = None
    discriminator_optimizer_kwargs: dict[str, Any] | None = None
    discriminator_reward_coef: float | None = None
    discriminator_reward_type: str | None = None
    discriminator_loss_func: str | None = None
    discriminator_loss_coef: float | None = None
    discriminator_gradient_penalty_coef: float | None = None
    discriminator_weight_decay_coef: float | None = None
    discriminator_logit_weight_decay_coef: float | None = None
    discriminator_gradient_torlerance: float | None = None
    discriminator_backbone_gradient_only: bool | None = None


@dataclass
class InstinctRlNormalizerCfg:
    class_name: str = "EmpiricalNormalization"


@dataclass
class InstinctRlOnPolicyRunnerCfg:
    seed: int = 42
    device: str = "cuda:0"
    num_steps_per_env: int = 24
    max_iterations: int = 30_000
    policy: InstinctRlActorCriticCfg = field(default_factory=InstinctRlActorCriticCfg)
    algorithm: InstinctRlPpoAlgorithmCfg = field(default_factory=InstinctRlPpoAlgorithmCfg)
    normalizers: dict[str, InstinctRlNormalizerCfg] = field(default_factory=dict)
    save_interval: int = 500
    log_interval: int = 1
    experiment_name: str = "instinct_mj"
    run_name: str = ""
    resume: bool = False
    load_run: str = ".*"
    load_checkpoint: str = "model_.*.pt"
    ckpt_manipulator: str | None = None
    ckpt_manipulator_kwargs: dict[str, Any] = field(default_factory=dict)
    policy_observation_group: str = "policy"
    critic_observation_group: str = "critic"

    def to_dict(self) -> dict[str, Any]:
        return _to_plain_dict(self)


# ---------------------------------------------------------------------------
# Compound policy configuration classes
#
# These thin subclasses set the correct ``class_name`` that instinct_rl uses
# to resolve the actual policy / algorithm implementation at runtime.
# The base ``InstinctRlActorCriticCfg`` already carries all optional fields
# (encoder, MoE, VAE, estimator, …) so no mixin is required.
# ---------------------------------------------------------------------------


@dataclass
class InstinctRlActorCriticRecurrentCfg(InstinctRlActorCriticCfg):
    class_name: str = "ActorCriticRecurrent"
    rnn_type: str = "gru"
    """The type of RNN to use. Default is GRU."""
    rnn_hidden_size: int = 256
    """The hidden size of the RNN."""
    rnn_num_layers: int = 1
    """The number of layers in the RNN."""
    multireward_multirnn: bool = False
    """Whether to use multiple RNN critics for multiple rewards."""


@dataclass
class InstinctRlMoEActorCriticCfg(InstinctRlActorCriticCfg):
    class_name: str = "MoEActorCritic"
    num_moe_experts: int | None = 8
    moe_gate_hidden_dims: tuple[int, ...] | None = ()


@dataclass
class InstinctRlVaeActorCriticCfg(InstinctRlActorCriticCfg):
    class_name: str = "VaeActor"


@dataclass
class InstinctRlEncoderActorCriticCfg(InstinctRlActorCriticCfg):
    """Encoder actor-critic: uses ``encoder_configs`` from the base class."""

    class_name: str = "EncoderActorCritic"


@dataclass
class InstinctRlEncoderActorCriticRecurrentCfg(InstinctRlActorCriticRecurrentCfg):
    """Encoder + recurrent actor-critic."""

    class_name: str = "EncoderActorCriticRecurrent"


@dataclass
class InstinctRlEncoderMoEActorCriticCfg(InstinctRlActorCriticCfg):
    """Encoder + MoE actor-critic."""

    class_name: str = "EncoderMoEActorCritic"
    num_moe_experts: int | None = 8
    moe_gate_hidden_dims: tuple[int, ...] | None = ()


@dataclass
class InstinctRlEncoderVaeActorCriticCfg(InstinctRlActorCriticCfg):
    """Encoder + VAE actor-critic."""

    class_name: str = "EncoderVaeActorCritic"


@dataclass
class EstimatorActorCriticCfg(InstinctRlActorCriticCfg):
    """Estimator actor-critic."""

    class_name: str = "EstimatorActorCritic"
    estimator_obs_components: list[str] = field(default_factory=list)
    estimator_target_components: list[str] = field(default_factory=list)
    estimator_configs: Any = None
    replace_state_prob: float = 1.0


@dataclass
class EstimatorActorCriticRecurrentCfg(InstinctRlActorCriticRecurrentCfg):
    """Estimator + recurrent actor-critic."""

    class_name: str = "EstimatorActorCriticRecurrent"
    estimator_obs_components: list[str] = field(default_factory=list)
    estimator_target_components: list[str] = field(default_factory=list)
    estimator_configs: Any = None
    replace_state_prob: float = 1.0
