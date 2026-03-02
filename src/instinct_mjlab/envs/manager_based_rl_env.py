from __future__ import annotations

from collections.abc import Sequence

import torch
from prettytable import PrettyTable

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers import RewardTermCfg
from mjlab.sim import Simulation
from mjlab.utils.logging import print_info
from mjlab.viewer.debug_visualizer import DebugVisualizer
from mjlab.viewer.offscreen_renderer import OffscreenRenderer

from instinct_mjlab.managers import MultiRewardCfg, MultiRewardManager
from instinct_mjlab.monitors import MonitorManager
from instinct_mjlab.envs.scene import InstinctScene


class InstinctRlEnv(ManagerBasedRlEnv):
  """This class adds additional logging mechanism on sensors to get more
  comprehensive running statistics.
  """

  def __init__(
    self,
    cfg,
    device: str,
    render_mode: str | None = None,
    **kwargs,
  ) -> None:
    del kwargs  # Unused.
    self.cfg = cfg
    if self.cfg.seed is not None:
      self.cfg.seed = self.seed(self.cfg.seed)
    self._sim_step_counter = 0
    self.extras = {}
    self.obs_buf = {}

    # Use InstinctScene so terrain cfg.class_type is honored (e.g. hacked_generator importer).
    self.scene = InstinctScene(self.cfg.scene, device=device)
    self.sim = Simulation(
      num_envs=self.scene.num_envs,
      cfg=self.cfg.sim,
      model=self.scene.compile(),
      device=device,
    )

    self.scene.initialize(
      mj_model=self.sim.mj_model,
      model=self.sim.model,
      data=self.sim.data,
    )
    if self.scene.sensor_context is not None:
      self.sim.set_sensor_context(self.scene.sensor_context)

    print_info("")
    table = PrettyTable()
    table.title = "Base Environment"
    table.field_names = ["Property", "Value"]
    table.align["Property"] = "l"
    table.align["Value"] = "l"
    table.add_row(["Number of environments", self.num_envs])
    table.add_row(["Environment device", self.device])
    table.add_row(["Environment seed", self.cfg.seed])
    table.add_row(["Physics step-size", self.physics_dt])
    table.add_row(["Environment step-size", self.step_dt])
    print_info(table.get_string())
    print_info("")

    self.common_step_counter = 0
    self.episode_length_buf = torch.zeros(
      cfg.scene.num_envs, device=device, dtype=torch.long
    )
    self.render_mode = render_mode
    self._offline_renderer: OffscreenRenderer | None = None
    if self.render_mode == "rgb_array":
      renderer = OffscreenRenderer(
        model=self.sim.mj_model, cfg=self.cfg.viewer, scene=self.scene
      )
      renderer.initialize()
      self._offline_renderer = renderer
    self.metadata["render_fps"] = 1.0 / self.step_dt

    self.load_managers()
    self.setup_manager_visualizers()

  def load_managers(self) -> None:
    # Route Instinct tasks through MultiRewardManager so reward logging matches
    # InstinctLab conventions:
    #   Episode_Reward/rewards_<term>/{max_episode_len_s,sum,timestep}
    reward_group_cfg = self._as_multi_reward_cfg(self.cfg.rewards)
    if reward_group_cfg is not None:
      self.cfg.rewards = {}

    super().load_managers()

    # Replace parent reward manager with MultiRewardManager when requested.
    if reward_group_cfg is not None:
      self.cfg.rewards = reward_group_cfg
      self.reward_manager = MultiRewardManager(
        self.cfg.rewards, self, scale_by_dt=self.cfg.scale_rewards_by_dt
      )
      print_info(f"[INFO] {self.reward_manager}")

    monitor_cfg = getattr(self.cfg, "monitors", None)
    if monitor_cfg is None:
      monitor_cfg = {}
    self.monitor_manager = MonitorManager(monitor_cfg, self)

  @staticmethod
  def _as_multi_reward_cfg(rewards_cfg):
    """Convert reward config into a multi-reward group config when possible.

    - Keep existing MultiRewardCfg as-is.
    - For flat dict[str, RewardTermCfg], wrap into {"rewards": ...}.
    - For grouped dicts (dict[str, dict[str, RewardTermCfg]]), keep as-is.
    """
    if isinstance(rewards_cfg, MultiRewardCfg):
      return rewards_cfg
    if isinstance(rewards_cfg, dict):
      first_non_none = next(
        (value for value in rewards_cfg.values() if value is not None),
        None,
      )
      if first_non_none is None or isinstance(first_non_none, RewardTermCfg):
        return {"rewards": rewards_cfg}
      return rewards_cfg
    return None

  def setup_manager_visualizers(self) -> None:
    super().setup_manager_visualizers()
    self.manager_visualizers["monitor_manager"] = self.monitor_manager

  def step(self, action: torch.Tensor):
    obs, reward, terminated, truncated, extras = super().step(action)
    if getattr(self, "monitor_manager", None) is not None:
      monitor_infos = self.monitor_manager.update(dt=self.step_dt)
      extras.setdefault("step", {})
      extras["step"].update(monitor_infos)
    return obs, reward, terminated, truncated, extras

  def update_visualizers(self, visualizer: DebugVisualizer) -> None:
    # Play configs can opt into visualizing debug overlays for all environments
    # (instead of only the currently selected env index in the viewer).
    viewer_cfg = getattr(self.cfg, "viewer", None)
    if bool(getattr(viewer_cfg, "debug_vis_show_all_envs", False)):
      visualizer.show_all_envs = True
    super().update_visualizers(visualizer)
    terrain = getattr(self.scene, "terrain", None)
    if terrain is not None:
      terrain.debug_vis(visualizer)

  def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
    if env_ids is None:
      env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.int64)
    if isinstance(env_ids, Sequence):
      env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.int64)

    monitor_infos = None
    if getattr(self, "monitor_manager", None) is not None:
      monitor_infos = self.monitor_manager.reset(env_ids, is_episode=True)

    super()._reset_idx(env_ids)

    if monitor_infos is not None:
      self.extras["log"] = self.extras.get("log", {})
      self.extras["log"].update(monitor_infos)

  """
  Properties.
  """

  @property
  def num_rewards(self) -> int:
    return getattr(self.reward_manager, "num_rewards", 1)
