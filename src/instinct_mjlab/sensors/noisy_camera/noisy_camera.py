from __future__ import annotations

import inspect
from collections.abc import Sequence

import torch
from prettytable import PrettyTable

from mjlab.utils.lab_api.string import string_to_callable

from instinct_mjlab.utils.buffers.async_circular_buffer import AsyncCircularBuffer
from instinct_mjlab.utils.noise import ImageNoiseCfg


class NoisyCameraMixin:  # as a subclass of SensorBase
    """
    This mixin class adds noise to the camera data.
    """

    def __str__(self) -> str:
        return_ = super().__str__()
        noise_info_table = PrettyTable()
        noise_info_table.field_names = ["Noise Name", "Noise Cfg Name"]
        for noise_name, noise_cfg in self.cfg.noise_pipeline:  # type: ignore
            noise_info_table.add_row([noise_name, noise_cfg.__name__])
        return_ += "\n" + str(noise_info_table)
        history_info_table = PrettyTable()
        history_info_table.field_names = ["History Name", "History Length"]
        for history_name, history_length in self.cfg.data_histories.items():  # type: ignore
            history_info_table.add_row([history_name, history_length])
        return_ += "\n" + str(history_info_table)
        return return_

    """
    Noise Pipeline
    """

    def build_noise_pipeline(self):
        self.noise_pipeline: Sequence[ImageNoiseCfg] | list[ImageNoiseCfg] = []
        """Build the noise pipeline based on the configuration."""
        assert self._device is not None

        for noise_name, noise_cfg in self.cfg.noise_pipeline.items():  # type: ignore
            # Check if the noise configuration is valid
            if not isinstance(noise_cfg, ImageNoiseCfg):
                raise ValueError(f"Invalid noise configuration for {noise_name}: {noise_cfg}")

            noise_cfg.device = self._device
            # Ensure the device is set correctly if the function is not a class

            if isinstance(noise_cfg.func, str):
                noise_cfg.func = string_to_callable(noise_cfg.func)

            if inspect.isclass(noise_cfg.func):
                # If the function is a class, instantiate it
                noise_cfg.func = noise_cfg.func(noise_cfg, num_envs=self._num_envs, device=self._device)

            # Add the noise configuration to the pipeline
            self.noise_pipeline.append(noise_cfg)

        # apply the noise pipeline to the initialized output buffers for noised output
        for data_type in self.cfg.data_types:
            self._camera_data.output[f"{data_type}_noised"] = self.apply_noise_pipeline(
                self._camera_data.output[data_type], env_ids=self._ALL_INDICES
            )

    def apply_noise_pipeline(self, data: torch.Tensor, env_ids: torch.Tensor | Sequence[int]) -> torch.Tensor:
        """Apply noise to the data(image).
        ## NOTE: The input data is only for selected envs (by env_ids).
        Args:
            data: The data to which noise will be applied. if Image, the shape should be (N_, H, W, C) for all environments.
        """
        # Check if the noise sequence is built
        if self.noise_pipeline is None:
            raise RuntimeError("Noise sequence not built. Call build_noise_pipeline() first.")

        # Keep noised output decoupled when no noise op is configured.
        if len(self.noise_pipeline) == 0:
            return data.clone()

        # Apply noise to the image by calling the noise pipeline one by one.
        for noise_cfg in self.noise_pipeline:
            data = noise_cfg.func(data, noise_cfg, env_ids)  # type: ignore

        return data

    def apply_noise_pipeline_to_all_data_types(self, env_ids: torch.Tensor | Sequence[int]):
        """Apply the noise pipeline to all data types."""
        for data_type in self.cfg.data_types:
            self._camera_data.output[f"{data_type}_noised"][env_ids] = self.apply_noise_pipeline(
                self._camera_data.output[data_type][env_ids], env_ids=env_ids
            )

    def reset_noise_pipeline(self, env_ids: Sequence[int] | None = None):
        """Reset the noise pipeline for the specified environment IDs."""
        if self.noise_pipeline is None:
            raise RuntimeError("Noise sequence not built. Call build_noise_pipeline() first.")

        for noise_cfg in self.noise_pipeline:
            if hasattr(noise_cfg.func, "reset"):
                noise_cfg.func.reset(env_ids)

    """
    History Buffers
    """

    def build_history_buffers(self):
        """Build the history buffers for the specified data types."""
        assert self._device is not None
        self.output_history_buffers: dict[str, AsyncCircularBuffer] = dict()

        for data_type, history_length in self.cfg.data_histories.items():
            self.output_history_buffers[data_type] = AsyncCircularBuffer(
                history_length, self._num_envs, self._device
            )
            data_shape = self._camera_data.output[data_type].shape
            self._camera_data.output[f"{data_type}_history"] = torch.zeros(
                (data_shape[0], history_length, *data_shape[1:]), device=self._device
            )

    def update_history_buffers(self, env_ids: torch.Tensor | Sequence[int]):
        """Append the history buffers for the specified data types and update the result in self._camera_data.output.
        Only configured data types will be appended, so only env_ids are needed. Please call this function after all
        outputs are computed.
        """
        for data_type in self.cfg.data_histories.keys():
            self.output_history_buffers[data_type].append(self._camera_data.output[data_type][env_ids], env_ids)
            self._camera_data.output[f"{data_type}_history"][env_ids] = self.output_history_buffers[data_type].__getitem__(
                batch_ids=env_ids
            )

    def reset_history_buffers(self, env_ids: torch.Tensor | Sequence[int] | None):
        """Reset the history buffers for the specified data types."""
        for data_type in self.cfg.data_histories.keys():
            self.output_history_buffers[data_type].reset(env_ids)
