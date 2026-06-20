"""Noise configuration classes for mjlab."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Callable, ClassVar, Literal, Optional, Sequence

import torch
from typing_extensions import override

from instinct_mj.utils.noise import noise_model

# Type alias for noise parameters: scalar or per-dimension values.
NoiseParam = float | tuple[float, ...]


@dataclass(kw_only=True)
class NoiseCfg(abc.ABC):
    """Base configuration for a noise term."""

    operation: Literal["add", "scale", "abs"] = "add"

    # Cache for converted tensors, keyed by device string.
    _tensor_cache: dict[str, dict[str, torch.Tensor]] = field(default_factory=dict, init=False, repr=False)

    def _get_cached_tensor(self, name: str, value: NoiseParam, device: torch.device) -> torch.Tensor:
        """Get a cached tensor for the given parameter on the specified device."""
        device_key = str(device)
        if device_key not in self._tensor_cache:
            self._tensor_cache[device_key] = {}
        if name not in self._tensor_cache[device_key]:
            self._tensor_cache[device_key][name] = torch.tensor(value, device=device)
        return self._tensor_cache[device_key][name]

    @abc.abstractmethod
    def apply(self, data: torch.Tensor) -> torch.Tensor:
        """Apply noise to the input data."""


@dataclass
class ConstantNoiseCfg(NoiseCfg):
    bias: NoiseParam = 0.0

    @override
    def apply(self, data: torch.Tensor) -> torch.Tensor:
        bias = self._get_cached_tensor("bias", self.bias, data.device)

        if self.operation == "add":
            return data + bias
        elif self.operation == "scale":
            return data * bias
        elif self.operation == "abs":
            return torch.zeros_like(data) + bias
        else:
            raise ValueError(f"Unsupported noise operation: {self.operation}")


@dataclass
class UniformNoiseCfg(NoiseCfg):
    n_min: NoiseParam = -1.0
    n_max: NoiseParam = 1.0

    def __post_init__(self):
        if isinstance(self.n_min, float) and isinstance(self.n_max, float):
            if self.n_min >= self.n_max:
                raise ValueError(f"n_min ({self.n_min}) must be less than n_max ({self.n_max})")

    @override
    def apply(self, data: torch.Tensor) -> torch.Tensor:
        n_min = self._get_cached_tensor("n_min", self.n_min, data.device)
        n_max = self._get_cached_tensor("n_max", self.n_max, data.device)

        # Generate uniform noise in [0, 1) and scale to [n_min, n_max).
        noise = torch.rand_like(data) * (n_max - n_min) + n_min

        if self.operation == "add":
            return data + noise
        elif self.operation == "scale":
            return data * noise
        elif self.operation == "abs":
            return noise
        else:
            raise ValueError(f"Unsupported noise operation: {self.operation}")


@dataclass
class GaussianNoiseCfg(NoiseCfg):
    mean: NoiseParam = 0.0
    std: NoiseParam = 1.0

    def __post_init__(self):
        if isinstance(self.std, float) and self.std <= 0:
            raise ValueError(f"std ({self.std}) must be positive")

    @override
    def apply(self, data: torch.Tensor) -> torch.Tensor:
        mean = self._get_cached_tensor("mean", self.mean, data.device)
        std = self._get_cached_tensor("std", self.std, data.device)

        # Generate standard normal noise and scale.
        noise = mean + std * torch.randn_like(data)

        if self.operation == "add":
            return data + noise
        elif self.operation == "scale":
            return data * noise
        elif self.operation == "abs":
            return noise
        else:
            raise ValueError(f"Unsupported noise operation: {self.operation}")


##
# Noise models.
##


@dataclass(kw_only=True)
class NoiseModelCfg:
    """Configuration for a noise model."""

    noise_cfg: NoiseCfg

    class_type: ClassVar[type[noise_model.NoiseModel]] = noise_model.NoiseModel

    def __init_subclass__(cls, class_type: type[noise_model.NoiseModel]):
        cls.class_type = class_type


@dataclass(kw_only=True)
class NoiseModelWithAdditiveBiasCfg(NoiseModelCfg, class_type=noise_model.NoiseModelWithAdditiveBias):
    """Configuration for an additive Gaussian noise with bias model."""

    bias_noise_cfg: NoiseCfg | None = None
    sample_bias_per_component: bool = True

    def __post_init__(self):
        if self.bias_noise_cfg is None:
            raise ValueError("bias_noise_cfg must be specified for NoiseModelWithAdditiveBiasCfg")


##
# Image noise models.
##


@dataclass(kw_only=True)
class ImageNoiseCfg:
    func: (
        Callable[[torch.Tensor, ImageNoiseCfg, torch.Tensor | Sequence[int]], torch.Tensor]
        | type[noise_model.ImageNoiseModel]
    ) = noise_model.ImageNoiseModel
    """The callable function to apply noise to the image.
    The function should take three arguments:
      - the image in shape (N_, H, W, C) where N_ = len(env_ids)
      - the cfg object (as this dataclass object)
      - the env_ids tensor for specifying the environment ids
    """

    device: str | torch.device = "cpu"


@dataclass(kw_only=True)
class DepthContourNoiseCfg(ImageNoiseCfg):
    contour_threshold: float = 2.0
    maxpool_kernel_size: int = 1
    func: Callable[[torch.Tensor, DepthContourNoiseCfg, torch.Tensor | Sequence[int]], torch.Tensor] = (
        noise_model.depth_contour_noise
    )


@dataclass(kw_only=True)
class DepthArtifactNoiseCfg(ImageNoiseCfg):
    artifacts_prob: float = 0.0001  # should be very low
    artifacts_height_mean_std: list[float] = field(default_factory=lambda: [2, 0.5])
    artifacts_width_mean_std: list[float] = field(default_factory=lambda: [2, 0.5])
    noise_value: float = 0.0
    func: Callable[[torch.Tensor, DepthArtifactNoiseCfg, torch.Tensor | Sequence[int]], torch.Tensor] = (
        noise_model.depth_artifact_noise
    )


@dataclass(kw_only=True)
class DepthSteroNoiseCfg(ImageNoiseCfg):
    stero_far_distance: float = 2.0
    stero_min_distance: float = 0.12
    stero_far_noise_std: float = 0.08
    stero_near_noise_std: float = 0.02

    stero_full_block_artifacts_prob: float = 0.001
    stero_full_block_values: list[float] = field(default_factory=lambda: [0.0, 0.25, 0.5, 1.0, 3.0])
    stero_full_block_height_mean_std: list[float] = field(default_factory=lambda: [62, 1.5])
    stero_full_block_width_mean_std: list[float] = field(default_factory=lambda: [3, 0.01])

    stero_half_block_spark_prob: float = 0.02
    stero_half_block_value: int = 3000

    func: Callable[[torch.Tensor, DepthSteroNoiseCfg, torch.Tensor | Sequence[int]], torch.Tensor] = (
        noise_model.depth_stero_noise
    )


@dataclass(kw_only=True)
class DepthSkyArtifactNoiseCfg(ImageNoiseCfg):
    sky_artifacts_prob: float = 0.0001
    sky_artifacts_far_distance: float = 2.0
    sky_artifacts_values: list[float] = field(default_factory=lambda: [0.6, 1.0, 1.2, 1.5, 1.8])
    sky_artifacts_height_mean_std: list[float] = field(default_factory=lambda: [2, 3.2])
    sky_artifacts_width_mean_std: list[float] = field(default_factory=lambda: [2, 3.2])

    func: Callable[[torch.Tensor, DepthSkyArtifactNoiseCfg, torch.Tensor | Sequence[int]], torch.Tensor] = (
        noise_model.depth_sky_artifact_noise
    )


@dataclass(kw_only=True)
class LatencyNoiseCfg(ImageNoiseCfg):
    history_length: int = 5

    sample_frequency: str | None = None
    sample_frequency_steps: int = 50
    sample_frequency_steps_offset: int = 5

    sample_probability: float = 0.1

    latency_distribution: str | None = "constant"
    latency_range: tuple[int, int] = (1, history_length)

    latency_mean_std: tuple[float, float] = (3, 1)

    latency_choices: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])
    latency_choices_probabilities: list[float] | None = None

    latency_steps: int = 5

    func: type[noise_model.LatencyNoiseModel] = noise_model.LatencyNoiseModel


@dataclass(kw_only=True)
class DepthNormalizationCfg(ImageNoiseCfg):
    """Configuration for normalizing depth values to a specific range."""

    depth_range: tuple[float, float] = (0.0, 10.0)
    normalize: bool = True
    output_range: tuple[float, float] = (0.0, 1.0)
    func: Callable[[torch.Tensor, DepthNormalizationCfg, torch.Tensor | Sequence[int]], torch.Tensor] = (
        noise_model.depth_normalization
    )


@dataclass(kw_only=True)
class CropAndResizeCfg(ImageNoiseCfg):
    """Configuration for cropping and resizing images."""

    crop_region: tuple[int, int, int, int] = (0, 0, 0, 0)
    resize_shape: tuple[int, int] | None = None
    func: Callable[[torch.Tensor, CropAndResizeCfg, torch.Tensor | Sequence[int]], torch.Tensor] = (
        noise_model.crop_and_resize
    )


@dataclass(kw_only=True)
class BlindSpotNoiseCfg(ImageNoiseCfg):
    """Configuration for adding blind spot noise (zeroing out regions of the image)."""

    crop_region: tuple[int, int, int, int] = (0, 0, 0, 0)
    func: Callable[[torch.Tensor, BlindSpotNoiseCfg, torch.Tensor | Sequence[int]], torch.Tensor] = (
        noise_model.blind_spot_noise
    )


@dataclass(kw_only=True)
class GaussianBlurNoiseCfg(ImageNoiseCfg):
    """Configuration for adding Gaussian blur noise to images."""

    kernel_size: int = 3
    sigma: float = 1.0
    func: Callable[[torch.Tensor, GaussianBlurNoiseCfg, torch.Tensor | Sequence[int]], torch.Tensor] = (
        noise_model.gaussian_blur_noise
    )


@dataclass(kw_only=True)
class RandomGaussianNoiseCfg(ImageNoiseCfg):
    """Configuration for adding random Gaussian noise to images."""

    probability: float = 0.1
    noise_mean: float = 0.0
    noise_std: float = 1.0
    func: Callable[[torch.Tensor, RandomGaussianNoiseCfg, torch.Tensor | Sequence[int]], torch.Tensor] = (
        noise_model.random_gaussian_noise
    )


@dataclass(kw_only=True)
class RangeBasedGaussianNoiseCfg(ImageNoiseCfg):
    """Configuration for adding range-based Gaussian noise to images."""

    min_value: float | None = None
    max_value: float | None = None
    noise_std: float = 1.0
    func: Callable[[torch.Tensor, RangeBasedGaussianNoiseCfg, torch.Tensor | Sequence[int]], torch.Tensor] = (
        noise_model.range_based_gaussian_noise
    )


@dataclass(kw_only=True)
class StereoTooCloseNoiseCfg(ImageNoiseCfg):
    """Configuration for adding stereo-too-close noise to images."""

    close_threshold: float = 0.12

    full_block_height_mean_std: tuple[float, float] = (62, 1.5)
    full_block_width_mean_std: tuple[float, float] = (3, 0.01)
    full_block_values: list[float] = field(default_factory=lambda: [0.0, 0.25, 0.5, 1.0, 3.0])
    full_block_artifacts_prob: float = 0.008

    half_block_height_mean_std: tuple[float, float] = (2, 3.2)
    half_block_width_mean_std: tuple[float, float] = (2, 3.2)
    half_block_value: float = 30
    half_block_spark_prob: float = 0.02

    func: Callable[[torch.Tensor, StereoTooCloseNoiseCfg, torch.Tensor | Sequence[int]], torch.Tensor] = (
        noise_model.stereo_too_close_noise
    )


@dataclass(kw_only=True)
class SensorDeadNoiseCfg(ImageNoiseCfg):
    """Configuration for adding sensor dead behavior, which might be autonomous restarted.
    Thus causing some frames of non-refreshed data.
    """

    dead_probability: float = 0.01
    """The probability of the sensor dead."""

    dead_frames: int | list[int] = 90
    """The number of frames to be non-refreshed (before the sensor is restarted).
    Can be a single number or a list of numbers to be uniformly selected from.
    """

    func: type[noise_model.SensorDeadNoiseModel] = noise_model.SensorDeadNoiseModel
