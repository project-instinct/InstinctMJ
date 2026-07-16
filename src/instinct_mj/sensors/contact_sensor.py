from __future__ import annotations

from dataclasses import dataclass

import torch
from mjlab.sensor import ContactSensor, ContactSensorCfg


class ForceThresholdContactSensor(ContactSensor):
    """mjlab contact sensor with InstinctLab force-threshold air-time semantics."""

    cfg: ForceThresholdContactSensorCfg

    def _update_air_time_tracking(self) -> None:
        assert self._air_time_state is not None

        contact_data = self._extract_sensor_data()
        assert contact_data.force is not None

        assert self._data is not None
        current_time = self._data.time
        elapsed_time = current_time - self._air_time_state.last_time
        elapsed_time = elapsed_time.unsqueeze(-1)

        is_contact = torch.linalg.vector_norm(contact_data.force, dim=-1) > self.cfg.force_threshold

        state = self._air_time_state
        is_first_contact = (state.current_air_time > 0) & is_contact
        is_first_detached = (state.current_contact_time > 0) & ~is_contact

        state.last_air_time[:] = torch.where(
            is_first_contact,
            state.current_air_time + elapsed_time,
            state.last_air_time,
        )
        state.current_air_time[:] = torch.where(
            ~is_contact,
            state.current_air_time + elapsed_time,
            torch.zeros_like(state.current_air_time),
        )

        state.last_contact_time[:] = torch.where(
            is_first_detached,
            state.current_contact_time + elapsed_time,
            state.last_contact_time,
        )
        state.current_contact_time[:] = torch.where(
            is_contact,
            state.current_contact_time + elapsed_time,
            torch.zeros_like(state.current_contact_time),
        )

        state.last_time[:] = current_time


@dataclass(kw_only=True)
class ForceThresholdContactSensorCfg(ContactSensorCfg):
    """Contact sensor config with a force threshold for air/contact timing."""

    class_type: type = ForceThresholdContactSensor

    force_threshold: float = 1.0
    """Net contact-force threshold in newtons used only for air/contact timing."""

    def build(self) -> ForceThresholdContactSensor:
        return ForceThresholdContactSensor(self)
