"""Position-velocity PD actuators of the ACP tracking task whose action delay is ONE lag per env shared by all actuator groups."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.actuator import BuiltinPdActuator, BuiltinPdActuatorCfg
from mjlab.utils.buffers import DelayBuffer

if TYPE_CHECKING:
    from mjlab.entity import Entity


class FollowerDelayBuffer(DelayBuffer):
    """Delay buffer of one actuator group whose lag is the lag of the leader group's buffer (same env, same physics step)."""

    def __init__(self, leader: DelayBuffer, **kwargs):
        super().__init__(**kwargs)
        self.leader = leader

    def _update_lags(self) -> None:
        # the leader group computes before its followers within a physics step (the entity applies the actuators in order)
        self._current_lags = self.leader.current_lags.clone()
        self._step_count += 1


@dataclass(kw_only=True)
class SharedDelayPdActuatorCfg(BuiltinPdActuatorCfg):
    """``BuiltinPdActuatorCfg`` whose delay lag is shared with the first actuator of the entity (paper Table S2: one action delay
    U(0, 20) ms per env). mjlab fuses the delay buffers of delayed actuators only for ``BuiltinActuatorType``, which ``BuiltinPdActuator``
    is not, so without this every group would draw its own lag."""

    def build(self, entity: Entity, target_ids: list[int], target_names: list[str]) -> SharedDelayPdActuator:
        return SharedDelayPdActuator(self, entity, target_ids, target_names)


class SharedDelayPdActuator(BuiltinPdActuator):
    def _init_delay_buffer(self, num_envs: int, device: str) -> None:
        if not self.has_delay:
            return
        leader = self.entity._actuators[0]
        if leader is self:
            super()._init_delay_buffer(num_envs, device)
            return
        assert isinstance(leader, SharedDelayPdActuator) and leader._delay_buffer is not None
        self._delay_buffer = FollowerDelayBuffer(
            leader._delay_buffer,
            min_lag=self.cfg.delay_min_lag,
            max_lag=self.cfg.delay_max_lag,
            batch_size=num_envs,
            device=device,
            hold_prob=self.cfg.delay_hold_prob,
            update_period=self.cfg.delay_update_period,
            per_env_phase=self.cfg.delay_per_env_phase,
        )
