from __future__ import annotations

from dataclasses import dataclass

from mjlab.actuator import BuiltinPdActuatorCfg


@dataclass(kw_only=True)
class InstinctActuatorCfg(BuiltinPdActuatorCfg):
    """Builtin PD actuator config with joint velocity limit metadata."""

    velocity_limit: float


@dataclass(kw_only=True)
class DelayedInstinctActuatorCfg(InstinctActuatorCfg):
    """PD actuator cfg with mjlab-native integrated command delay fields."""
