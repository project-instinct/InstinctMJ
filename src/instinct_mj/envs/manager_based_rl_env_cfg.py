from __future__ import annotations

from dataclasses import dataclass, field

from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnvCfg
from mjlab.managers import RewardTermCfg
from mjlab.viewer.viewer_config import ViewerConfig


@dataclass(kw_only=True)
class InstinctLabRLEnvCfg(ManagerBasedRlEnvCfg):
    """Configuration for a reinforcement learning environment with the manager-based workflow."""

    rewards: dict[str, dict[str, RewardTermCfg | str | None]] = field(default_factory=dict)
    """Reward groups consumed directly by :class:`MultiRewardManager`."""

    viewer: ViewerConfig = field(default_factory=ViewerConfig)
    """Viewer Settings."""

    run_name: str = ""
    """Environment-specific suffix used for experiment log directories."""

    # monitor settings
    monitors: dict = field(default_factory=dict)
    """Monitor Settings.

  Please refer to the `instinct_mj.monitors.MonitorManager` class for more details.
  """
