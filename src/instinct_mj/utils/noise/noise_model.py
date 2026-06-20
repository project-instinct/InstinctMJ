"""Noise models for mjlab."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Sequence

import torch
import torch.nn.functional as F
from torchvision.transforms import GaussianBlur
from typing_extensions import override

if TYPE_CHECKING:
    from instinct_mj.utils.noise import noise_cfg


class NoiseModel:
    """Base class for noise models."""

    def __init__(self, noise_model_cfg: noise_cfg.NoiseModelCfg, num_envs: int, device: str):
        self._noise_model_cfg = noise_model_cfg
        self._num_envs = num_envs
        self._device = device

        # Validate configuration.
        if not hasattr(noise_model_cfg, "noise_cfg") or noise_model_cfg.noise_cfg is None:
            raise ValueError("NoiseModelCfg must have a valid noise_cfg")

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        """Reset noise model state. Override in subclasses if needed."""

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Apply noise to input data."""
        return self._noise_model_cfg.noise_cfg.apply(data)


class NoiseModelWithAdditiveBias(NoiseModel):
    """Noise model with additional additive bias that is constant for the duration
    of the entire episode."""

    def __init__(
        self,
        noise_model_cfg: noise_cfg.NoiseModelWithAdditiveBiasCfg,
        num_envs: int,
        device: str,
    ):
        super().__init__(noise_model_cfg, num_envs, device)

        # Validate bias configuration.
        if not hasattr(noise_model_cfg, "bias_noise_cfg") or noise_model_cfg.bias_noise_cfg is None:
            raise ValueError("NoiseModelWithAdditiveBiasCfg must have a valid bias_noise_cfg")

        self._bias_noise_cfg = noise_model_cfg.bias_noise_cfg
        self._sample_bias_per_component = noise_model_cfg.sample_bias_per_component

        # Initialize bias tensor.
        self._bias = torch.zeros((num_envs, 1), device=self._device)
        self._num_components: int | None = None
        self._bias_initialized = False

    @override
    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        """Reset bias values for specified environments."""
        indices = slice(None) if env_ids is None else env_ids
        # Sample new bias values.
        self._bias[indices] = self._bias_noise_cfg.apply(self._bias[indices])

    def _initialize_bias_shape(self, data_shape: torch.Size) -> None:
        """Initialize bias tensor shape based on data and configuration."""
        if self._sample_bias_per_component and not self._bias_initialized:
            *_, self._num_components = data_shape
            # Expand bias to match number of components.
            self._bias = self._bias.repeat(1, self._num_components)
            self._bias_initialized = True
            # Resample bias with new shape.
            self.reset()

    @override
    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Apply noise and additive bias to input data."""
        self._initialize_bias_shape(data.shape)
        noisy_data = super().__call__(data)
        return noisy_data + self._bias


def _torch_rand_float(
    lower: float,
    upper: float,
    shape: tuple[int, ...],
    device: str | torch.device,
) -> torch.Tensor:
    return torch.rand(shape, device=device) * (upper - lower) + lower


class ImageNoiseModel:
    """Base image-noise model."""

    def __init__(
        self,
        cfg: noise_cfg.ImageNoiseCfg,
        num_envs: int = 1,
        device: str | torch.device = "cpu",
    ):
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = device

    def __call__(
        self,
        data: torch.Tensor,
        cfg: noise_cfg.ImageNoiseCfg,
        env_ids: torch.Tensor | Sequence[int],
    ) -> torch.Tensor:
        return data

    def reset(self, env_ids: Sequence[int] | None = None):
        pass


def depth_contour_noise(
    data: torch.Tensor,
    cfg: noise_cfg.DepthContourNoiseCfg,
    env_ids: torch.Tensor | Sequence[int],
) -> torch.Tensor:
    device = cfg.device
    contour_detection_kernel = torch.zeros(
        (8, 1, 3, 3),
        dtype=torch.float32,
        device=device,
    )
    contour_detection_kernel[0, :, 1, 1] = 0.5
    contour_detection_kernel[0, :, 0, 0] = -0.5
    contour_detection_kernel[1, :, 1, 1] = 0.1
    contour_detection_kernel[1, :, 0, 1] = -0.1
    contour_detection_kernel[2, :, 1, 1] = 0.5
    contour_detection_kernel[2, :, 0, 2] = -0.5
    contour_detection_kernel[3, :, 1, 1] = 1.2
    contour_detection_kernel[3, :, 1, 0] = -1.2
    contour_detection_kernel[4, :, 1, 1] = 1.2
    contour_detection_kernel[4, :, 1, 2] = -1.2
    contour_detection_kernel[5, :, 1, 1] = 0.5
    contour_detection_kernel[5, :, 2, 0] = -0.5
    contour_detection_kernel[6, :, 1, 1] = 0.1
    contour_detection_kernel[6, :, 2, 1] = -0.1
    contour_detection_kernel[7, :, 1, 1] = 0.5
    contour_detection_kernel[7, :, 2, 2] = -0.5

    original_shape = data.shape
    if data.dim() == 4 and data.shape[-1] == 1:
        data = data.permute(0, 3, 1, 2)
    elif data.dim() == 3:
        data = data.unsqueeze(1)

    mask = (
        F.max_pool2d(
            torch.abs(F.conv2d(data, contour_detection_kernel, padding=1)).max(dim=1, keepdim=True)[0],
            kernel_size=cfg.maxpool_kernel_size,
            stride=1,
            padding=int(cfg.maxpool_kernel_size / 2),
        )
        > cfg.contour_threshold
    )

    data = data.clone()
    data[mask] = 0.0

    if len(original_shape) == 4 and original_shape[-1] == 1:
        data = data.permute(0, 2, 3, 1)
    elif len(original_shape) == 3:
        data = data.squeeze(1)

    return data


def depth_artifact_noise(
    data: torch.Tensor,
    cfg: noise_cfg.DepthArtifactNoiseCfg,
    env_ids: torch.Tensor | Sequence[int],
) -> torch.Tensor:
    return _add_depth_artifacts(
        data,
        artifacts_prob=cfg.artifacts_prob,
        artifacts_height_mean_std=cfg.artifacts_height_mean_std,
        artifacts_width_mean_std=cfg.artifacts_width_mean_std,
        device=cfg.device,
        noise_value=cfg.noise_value,
    )


def depth_stero_noise(
    data: torch.Tensor,
    cfg: noise_cfg.DepthSteroNoiseCfg,
    env_ids: torch.Tensor | Sequence[int],
) -> torch.Tensor:
    device = cfg.device
    n, h, w, _ = data.shape
    far_mask = data > cfg.stero_far_distance
    too_close_mask = data < cfg.stero_min_distance
    near_mask = (~far_mask) & (~too_close_mask)

    far_noise = _torch_rand_float(0.0, cfg.stero_far_noise_std, (n, h * w), device=device).view(n, h, w, 1)
    far_noise = far_noise * far_mask
    data += far_noise

    near_noise = _torch_rand_float(0.0, cfg.stero_near_noise_std, (n, h * w), device=device).view(n, h, w, 1)
    near_noise = near_noise * near_mask
    data += near_noise

    vertical_block_mask = too_close_mask.sum(dim=-3, keepdim=True) > (too_close_mask.shape[-3] * 0.6)

    full_block_mask = vertical_block_mask & too_close_mask
    half_block_mask = (~vertical_block_mask) & too_close_mask

    for pixel_value in random.sample(
        cfg.stero_full_block_values,
        len(cfg.stero_full_block_values),
    ):
        artifacts_buffer = torch.ones_like(data)
        artifacts_buffer = _add_depth_artifacts(
            artifacts_buffer,
            cfg.stero_full_block_artifacts_prob,
            cfg.stero_full_block_height_mean_std,
            cfg.stero_full_block_width_mean_std,
            device=device,
        )
        data[full_block_mask] = ((1 - artifacts_buffer) * pixel_value)[full_block_mask]

    half_block_spark = (
        _torch_rand_float(0.0, 1.0, (n, h * w), device=device).view(n, h, w, 1) < cfg.stero_half_block_spark_prob
    )
    data[half_block_mask] = (half_block_spark.to(torch.float32) * cfg.stero_half_block_value)[half_block_mask]

    return data


def depth_sky_artifact_noise(
    data: torch.Tensor,
    cfg: noise_cfg.DepthSkyArtifactNoiseCfg,
    env_ids: torch.Tensor | Sequence[int],
) -> torch.Tensor:
    if data.dim() == 4 and data.shape[-1] == 1:
        data = data.permute(0, 3, 1, 2)
        restore_channels_last = True
    else:
        restore_channels_last = False

    device = cfg.device
    _, _, h, _ = data.shape

    possible_to_sky_mask = data > cfg.sky_artifacts_far_distance

    def _recognize_top_down_seeing_sky(too_far_mask):
        num_too_far_above = too_far_mask.cumsum(dim=-2)
        all_too_far_above_threshold = torch.arange(h, device=device).view(1, 1, h, 1)
        all_too_far_above = num_too_far_above > all_too_far_above_threshold
        return all_too_far_above

    to_sky_mask = _recognize_top_down_seeing_sky(possible_to_sky_mask)
    isinf_mask = data.isinf()

    for pixel_value in random.sample(
        cfg.sky_artifacts_values,
        len(cfg.sky_artifacts_values),
    ):
        artifacts_buffer = torch.ones_like(data)
        artifacts_buffer = _add_depth_artifacts(
            artifacts_buffer.permute(0, 2, 3, 1),
            cfg.sky_artifacts_prob,
            cfg.sky_artifacts_height_mean_std,
            cfg.sky_artifacts_width_mean_std,
            device=device,
        ).permute(0, 3, 1, 2)
        data[to_sky_mask & (~isinf_mask)] *= artifacts_buffer[to_sky_mask & (~isinf_mask)]
        data[to_sky_mask & isinf_mask & (artifacts_buffer < 1)] = 0.0
        data[to_sky_mask] += ((1 - artifacts_buffer) * pixel_value)[to_sky_mask]

    if restore_channels_last:
        data = data.permute(0, 2, 3, 1)

    return data


def depth_normalization(
    data: torch.Tensor,
    cfg: noise_cfg.DepthNormalizationCfg,
    env_ids: torch.Tensor | Sequence[int],
) -> torch.Tensor:
    if data.dim() == 4 and data.shape[-1] == 1:
        data = data.permute(0, 3, 1, 2)

    min_depth = cfg.depth_range[0]
    max_depth = cfg.depth_range[1]
    data = data.clip(min_depth, max_depth)

    if cfg.normalize:
        data = (data - min_depth) / (max_depth - min_depth)
        data = data * (cfg.output_range[1] - cfg.output_range[0]) + cfg.output_range[0]

    if len(data.shape) == 4 and data.shape[1] == 1:
        data = data.permute(0, 2, 3, 1)

    return data


class LatencyNoiseModel(ImageNoiseModel):
    def __init__(self, cfg: noise_cfg.LatencyNoiseCfg, num_envs, device):
        super().__init__(cfg, num_envs, device)
        from instinct_mj.utils.buffers.async_delay_buffer import AsyncDelayBuffer

        if cfg.latency_distribution == "choice" and max(cfg.latency_choices) > cfg.history_length:
            raise RuntimeError(f"Latency choices {cfg.latency_choices} exceed the history length {cfg.history_length}.")
        if cfg.latency_distribution == "constant" and cfg.latency_steps > cfg.history_length:
            raise RuntimeError(f"Latency steps {cfg.latency_steps} exceed the history length {cfg.history_length}.")
        if (cfg.latency_choices == "uniform" or cfg.latency_choices == "normal") and (
            (cfg.latency_range[1] > cfg.history_length)
            or (cfg.latency_range[0] < 0)
            or cfg.latency_range[0] > cfg.latency_range[1]
        ):
            raise RuntimeError(f"Latency range {cfg.latency_range} is invalid.")

        self.delay_buffer = AsyncDelayBuffer(cfg.history_length, num_envs, device)
        self.cfg = cfg
        self.num_envs = num_envs

        self.env_step_counters = torch.zeros(num_envs, dtype=torch.int, device=device)
        self.last_resample_steps = torch.zeros(num_envs, dtype=torch.int, device=device)

        if cfg.sample_frequency == "every_n_steps":
            self.resample_intervals = self._generate_resample_intervals()

        self._resample_delays(torch.arange(num_envs, device=device))

    def __call__(self, data, cfg, env_ids: torch.Tensor | Sequence[int]):
        if isinstance(env_ids, Sequence):
            env_ids = torch.tensor(env_ids, device=self.device)

        if data.shape[0] != len(env_ids):
            raise RuntimeError(
                f"Data batch shape {data.shape[0]} does not match the number of environments {len(env_ids)}."
            )

        self.env_step_counters[env_ids] += 1
        should_resample = self._should_resample(env_ids)

        if torch.any(should_resample):
            resample_env_ids = env_ids[should_resample]
            self._resample_delays(resample_env_ids)
            self.last_resample_steps[resample_env_ids] = self.env_step_counters[resample_env_ids]

        delayed = self.delay_buffer.compute(data, batch_ids=env_ids.tolist())
        return delayed

    def _generate_resample_intervals(self, env_ids: Sequence[int] | None = None):
        base_interval = self.cfg.sample_frequency_steps
        offset_range = self.cfg.sample_frequency_steps_offset
        if env_ids is None:
            offsets = torch.randint(-offset_range, offset_range + 1, (self.num_envs,), device=self.device)
        else:
            offsets = torch.randint(-offset_range, offset_range + 1, (len(env_ids),), device=self.device)
        intervals = base_interval + offsets
        intervals = intervals.clamp(min=1)
        return intervals

    def _should_resample(self, env_ids: torch.Tensor):
        if self.cfg.sample_frequency is not None:
            if self.cfg.sample_frequency == "every_n_steps":
                current_steps = self.env_step_counters[env_ids]
                last_resample_steps = self.last_resample_steps[env_ids]
                intervals = self.resample_intervals[env_ids]
                return current_steps - last_resample_steps >= intervals
            elif self.cfg.sample_frequency == "random_with_probability":
                prob = self.cfg.sample_probability
                return torch.rand(len(env_ids), device=self.device) < prob

        return torch.zeros(len(env_ids), dtype=torch.bool, device=self.device)

    def _resample_delays(self, env_ids: torch.Tensor):
        num_envs_to_resample = len(env_ids)

        if self.cfg.latency_distribution == "uniform":
            new_delays = torch.randint(
                self.cfg.latency_range[0],
                self.cfg.latency_range[1] + 1,
                (num_envs_to_resample,),
                dtype=torch.int,
                device=self.device,
            )
        elif self.cfg.latency_distribution == "normal":
            new_delays = (
                torch.normal(
                    mean=self.cfg.latency_mean_std[0],
                    std=self.cfg.latency_mean_std[1],
                    size=(num_envs_to_resample,),
                    device=self.device,
                )
                .round()
                .int()
                .clamp(
                    min=self.cfg.latency_range[0],
                    max=self.cfg.latency_range[1],
                )
            )
        elif self.cfg.latency_distribution == "choice":
            choices = torch.tensor(self.cfg.latency_choices, dtype=torch.int, device=self.device)
            if self.cfg.latency_choices_probabilities is not None:
                prob = torch.tensor(self.cfg.latency_choices_probabilities, device=self.device)
                indices = torch.multinomial(prob, num_envs_to_resample, replacement=True)
            else:
                indices = torch.randint(0, len(choices), (num_envs_to_resample,), device=self.device)
            new_delays = choices[indices]
        elif self.cfg.latency_distribution == "constant":
            new_delays = torch.full(
                (num_envs_to_resample,), self.cfg.latency_steps, dtype=torch.int, device=self.device
            )
        else:
            raise RuntimeError(f"Unknown latency distribution: {self.cfg.latency_distribution}")

        self.delay_buffer.set_time_lag(new_delays, env_ids.tolist())

    def reset(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = list(range(self.num_envs))

        env_ids_tensor = torch.tensor(env_ids, device=self.device)

        self.env_step_counters[env_ids_tensor] = 0
        self.last_resample_steps[env_ids_tensor] = 0

        if self.cfg.sample_frequency == "every_n_steps":
            new_intervals = self._generate_resample_intervals(env_ids)
            self.resample_intervals[env_ids_tensor] = new_intervals

        self.delay_buffer.reset(env_ids)
        self._resample_delays(env_ids_tensor)


def _add_depth_artifacts(
    data: torch.Tensor,
    artifacts_prob: float,
    artifacts_height_mean_std: list[float] | tuple[float, float],
    artifacts_width_mean_std: list[float] | tuple[float, float],
    device: str | torch.device,
    noise_value: float = 0.0,
):
    n, h, w, _ = data.shape

    def _clip(values: torch.Tensor, dim: int):
        return torch.clip(values, 0.0, float((h, w)[dim]))

    artifacts_mask = _torch_rand_float(0.0, 1.0, (n, h * w), device=device).view(n, h, w) < artifacts_prob
    artifacts_mask = artifacts_mask & (data[:, :, :, 0] > 0.0)
    artifacts_coord = torch.nonzero(artifacts_mask).to(torch.float32)

    if len(artifacts_coord) == 0:
        return data

    artifacts_size = (
        torch.clip(
            artifacts_height_mean_std[0]
            + torch.randn((artifacts_coord.shape[0],), device=device) * artifacts_height_mean_std[1],
            0.0,
            float(h),
        ),
        torch.clip(
            artifacts_width_mean_std[0]
            + torch.randn((artifacts_coord.shape[0],), device=device) * artifacts_width_mean_std[1],
            0.0,
            float(w),
        ),
    )

    artifacts_top = _clip(artifacts_coord[:, 1] - artifacts_size[0] / 2, 0)
    artifacts_left = _clip(artifacts_coord[:, 2] - artifacts_size[1] / 2, 1)
    artifacts_bottom = _clip(artifacts_coord[:, 1] + artifacts_size[0] / 2, 0)
    artifacts_right = _clip(artifacts_coord[:, 2] + artifacts_size[1] / 2, 1)

    env_ids = artifacts_coord[:, 0].long()
    env_onehot = torch.zeros((len(artifacts_coord), n), device=device)
    env_onehot[torch.arange(len(artifacts_coord)), env_ids] = 1.0

    num_artifacts = len(artifacts_coord)
    tops_expanded = artifacts_top[:, None, None]
    lefts_expanded = artifacts_left[:, None, None]
    bottoms_expanded = artifacts_bottom[:, None, None]
    rights_expanded = artifacts_right[:, None, None]

    source_patch = torch.zeros((num_artifacts, 1, 25, 25), device=device)
    source_patch[:, :, 1:24, 1:24] = 1.0

    grid = torch.zeros((num_artifacts, h, w, 2), device=device)
    grid[..., 0] = torch.linspace(-1, 1, w, device=device).view(1, 1, w)
    grid[..., 1] = torch.linspace(-1, 1, h, device=device).view(1, h, 1)
    grid[..., 0] = (grid[..., 0] * w + w - rights_expanded - lefts_expanded) / (rights_expanded - lefts_expanded)
    grid[..., 1] = (grid[..., 1] * h + h - bottoms_expanded - tops_expanded) / (bottoms_expanded - tops_expanded)

    all_artifacts = F.grid_sample(
        source_patch, grid, mode="bilinear", padding_mode="zeros", align_corners=False
    ).squeeze(1)

    final_masks = torch.einsum("an,ahw->nhw", env_onehot, all_artifacts)
    final_masks = torch.clamp(final_masks, 0, 1)

    data = data.squeeze(-1)
    data = data * (1 - final_masks) + final_masks * noise_value
    data = data.unsqueeze(-1)

    return data


def crop_and_resize(
    data: torch.Tensor,
    cfg: noise_cfg.CropAndResizeCfg,
    env_ids: torch.Tensor | Sequence[int],
) -> torch.Tensor:
    crop_region = cfg.crop_region
    start_up = crop_region[0]
    end_down = data.shape[1] - crop_region[1]
    start_left = crop_region[2]
    end_right = data.shape[2] - crop_region[3]
    cropped = data[:, start_up:end_down, start_left:end_right, :]

    if cfg.resize_shape is None:
        return cropped

    cropped = cropped.permute(0, 3, 1, 2)
    resized = F.interpolate(cropped, size=cfg.resize_shape, mode="bilinear", align_corners=False)
    resized = resized.permute(0, 2, 3, 1)
    return resized


def blind_spot_noise(
    data: torch.Tensor,
    cfg: noise_cfg.BlindSpotNoiseCfg,
    env_ids: torch.Tensor | Sequence[int],
) -> torch.Tensor:
    crop_region = cfg.crop_region
    start_up = crop_region[0]
    end_down = data.shape[1] - crop_region[1]
    start_left = crop_region[2]
    end_right = data.shape[2] - crop_region[3]

    data_modified = data.clone()
    data_modified[:, :start_up, :, :] = 0.0
    data_modified[:, end_down:, :, :] = 0.0
    data_modified[:, :, :start_left, :] = 0.0
    data_modified[:, :, end_right:, :] = 0.0
    return data_modified


def gaussian_blur_noise(
    data: torch.Tensor,
    cfg: noise_cfg.GaussianBlurNoiseCfg,
    env_ids: torch.Tensor | Sequence[int],
) -> torch.Tensor:
    data = data.permute(0, 3, 1, 2)
    blur_transform = GaussianBlur(kernel_size=cfg.kernel_size, sigma=cfg.sigma)
    blurred = blur_transform(data)
    blurred = blurred.permute(0, 2, 3, 1)
    return blurred


def random_gaussian_noise(
    data: torch.Tensor,
    cfg: noise_cfg.RandomGaussianNoiseCfg,
    env_ids: torch.Tensor | Sequence[int],
) -> torch.Tensor:
    n, h, w, c = data.shape
    noise = torch.randn((n, h, w, c), device=cfg.device) * cfg.noise_std + cfg.noise_mean
    if random.random() < cfg.probability:
        noisy_data = data + noise
    else:
        noisy_data = data

    return noisy_data


def _recognize_top_down_too_close(too_close_mask: torch.Tensor):
    vertical_too_close = too_close_mask.sum(dim=-3, keepdim=True) > (too_close_mask.shape[-3] * 0.6)
    return vertical_too_close


def range_based_gaussian_noise(
    data: torch.Tensor,
    cfg: noise_cfg.RangeBasedGaussianNoiseCfg,
    env_ids: torch.Tensor | Sequence[int],
) -> torch.Tensor:
    n, h, w, c = data.shape
    noise = torch.randn((n, h, w, c), device=data.device) * cfg.noise_std

    apply_mask = torch.ones((n, h, w, c), device=data.device, dtype=bool)
    if cfg.min_value is not None:
        apply_mask = apply_mask & (data >= cfg.min_value)
    if cfg.max_value is not None:
        apply_mask = apply_mask & (data <= cfg.max_value)

    noisy_data = data + noise * apply_mask
    return noisy_data


def stereo_too_close_noise(
    data: torch.Tensor,
    cfg: noise_cfg.StereoTooCloseNoiseCfg,
    env_ids: torch.Tensor | Sequence[int],
) -> torch.Tensor:
    n, h, w, c = data.shape
    too_close_mask = data < cfg.close_threshold
    vertical_block_mask = _recognize_top_down_too_close(too_close_mask)
    full_block_mask = vertical_block_mask & too_close_mask
    half_block_mask = (~vertical_block_mask) & too_close_mask

    for pixel_value in random.sample(
        cfg.full_block_values,
        len(cfg.full_block_values),
    ):
        artifacts_buffer = torch.ones_like(data)
        artifacts_buffer = _add_depth_artifacts(
            artifacts_buffer,
            cfg.full_block_artifacts_prob,
            cfg.full_block_height_mean_std,
            cfg.full_block_width_mean_std,
            device=data.device,
        )
        data[full_block_mask] = ((1 - artifacts_buffer) * pixel_value)[full_block_mask]

    half_block_spark = (
        _torch_rand_float(
            0.0,
            1.0,
            (n, h * w),
            device=data.device,
        ).view(n, h, w, 1)
        < cfg.half_block_spark_prob
    )
    data[half_block_mask] = (half_block_spark.to(torch.float32) * cfg.half_block_value)[half_block_mask]

    return data


class SensorDeadNoiseModel(ImageNoiseModel):
    def __init__(self, cfg: noise_cfg.SensorDeadNoiseCfg, num_envs, device):
        """Simulating when the sensor is dead and restarting, this may lead to several frames of non-refreshed data."""
        super().__init__(cfg, num_envs, device)
        self._data_buffer = None
        self._remain_dead_frames = torch.zeros(num_envs, device=device)
        self._dead_frames_options = (
            self.cfg.dead_frames
            if isinstance(self.cfg.dead_frames, int)
            else torch.tensor(self.cfg.dead_frames, device=device)
        )

    def __call__(self, data, cfg: noise_cfg.SensorDeadNoiseCfg, env_ids: torch.Tensor | Sequence[int]):
        if isinstance(env_ids, Sequence):
            env_ids = torch.tensor(env_ids, device=self.device)
        else:
            env_ids = env_ids.to(self.device)

        if data.shape[0] != len(env_ids):
            raise RuntimeError(
                f"Data batch shape {data.shape[0]} does not match the number of environments {len(env_ids)}."
            )

        if self._data_buffer is None:
            self._data_buffer = torch.zeros_like(data[0]).unsqueeze(0).repeat(self.num_envs, *([1] * (data.ndim - 1)))

        # determine if the sensor is dead this time.
        could_be_dead_mask = self._remain_dead_frames[env_ids] <= 0
        dead_this_time_mask = torch.logical_and(
            torch.rand(env_ids.shape[0], device=self.device) < self.cfg.dead_probability,
            could_be_dead_mask,
        )
        dead_frames = (
            self.cfg.dead_frames
            if isinstance(self.cfg.dead_frames, int)
            else self._dead_frames_options[
                torch.randint(
                    len(self._dead_frames_options),
                    size=(len(env_ids),),
                    device=self.device,
                )
            ]
        )
        self._remain_dead_frames[env_ids] = torch.where(
            dead_this_time_mask, dead_frames, self._remain_dead_frames[env_ids] - 1
        )
        self._remain_dead_frames[env_ids].clamp_(min=0)

        # refresh the data buffer if it is not dead.
        data_to_refresh_mask = self._remain_dead_frames[env_ids] <= 0
        self._data_buffer[env_ids[data_to_refresh_mask]] = data[data_to_refresh_mask]
        return self._data_buffer[env_ids]

    def reset(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = list(range(self.num_envs))
        self._remain_dead_frames[env_ids] = 0
        if self._data_buffer is not None:
            self._data_buffer[env_ids] = 0
