from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from mjlab.actuator import BuiltinPositionActuatorCfg, DelayedActuatorCfg


@dataclass(kw_only=True)
class InstinctActuatorCfg(BuiltinPositionActuatorCfg):
  """Builtin position actuator config with joint velocity limit metadata."""

  velocity_limit: float


@dataclass(kw_only=True)
class DelayedInstinctActuatorCfg(DelayedActuatorCfg):
  """Delayed wrapper for position actuator cfg with velocity limit metadata."""

  base_cfg: InstinctActuatorCfg
  delay_target: Literal["position"] = "position"

  @property
  def velocity_limit(self) -> float:
    return self.base_cfg.velocity_limit
