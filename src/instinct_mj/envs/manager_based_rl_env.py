from __future__ import annotations

from collections.abc import Sequence

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers import (
    ActionManager,
    CommandManager,
    CurriculumManager,
    EventManager,
    MetricsManager,
    NullCommandManager,
    NullCurriculumManager,
    NullMetricsManager,
    NullRecorderManager,
    ObservationManager,
    RecorderManager,
    TerminationManager,
)
from mjlab.sim import Simulation
from mjlab.utils.logging import print_info
from mjlab.viewer.debug_visualizer import DebugVisualizer
from mjlab.viewer.offscreen_renderer import OffscreenRenderer
from prettytable import PrettyTable

from instinct_mj.envs.scene import InstinctScene
from instinct_mj.managers import MultiRewardManager
from instinct_mj.monitors import MonitorManager


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
        self._instinct_body_lin_acc_cache: dict[str, dict[str, object]] = {}
        self.extras = {}
        self.obs_buf = {}
        # Initialize the manual-reset state here because InstinctRlEnv
        # customizes scene construction instead of calling ManagerBasedRlEnv.__init__.
        self._manual_reset_pending = torch.zeros(self.cfg.scene.num_envs, dtype=torch.bool, device=device)

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
        self.episode_length_buf = torch.zeros(cfg.scene.num_envs, device=device, dtype=torch.long)
        self.render_mode = render_mode
        self._offline_renderer: OffscreenRenderer | None = None
        if self.render_mode == "rgb_array":
            renderer = OffscreenRenderer(model=self.sim.mj_model, cfg=self.cfg.viewer, scene=self.scene)
            renderer.initialize()
            self._offline_renderer = renderer
        self.metadata["render_fps"] = 1.0 / self.step_dt

        self.load_managers()
        self.setup_manager_visualizers()

    def load_managers(self) -> None:
        """Load managers in mjlab order with InstinctLab multi-reward logging."""
        # Event manager (required before everything else for domain randomization).
        self.event_manager = EventManager(self.cfg.events, self)
        print_info(f"[INFO] {self.event_manager}")

        self.sim.expand_model_fields(self.event_manager.domain_randomization_fields)

        # Command manager must precede observations since observations may use commands.
        if len(self.cfg.commands) > 0:
            self.command_manager = CommandManager(self.cfg.commands, self)
        else:
            self.command_manager = NullCommandManager()
        print_info(f"[INFO] {self.command_manager}")

        self.action_manager = ActionManager(self.cfg.actions, self)
        print_info(f"[INFO] {self.action_manager}")
        self.observation_manager = ObservationManager(self.cfg.observations, self)
        print_info(f"[INFO] {self.observation_manager}")

        self.termination_manager = TerminationManager(self.cfg.terminations, self)
        print_info(f"[INFO] {self.termination_manager}")
        self.reward_manager = MultiRewardManager(
            self.cfg.rewards,
            self,
            scale_by_dt=self.cfg.scale_rewards_by_dt,
        )
        print_info(f"[INFO] {self.reward_manager}")
        if len(self.cfg.curriculum) > 0:
            self.curriculum_manager = CurriculumManager(self.cfg.curriculum, self)
        else:
            self.curriculum_manager = NullCurriculumManager()
        print_info(f"[INFO] {self.curriculum_manager}")
        if len(self.cfg.metrics) > 0:
            self.metrics_manager = MetricsManager(self.cfg.metrics, self)
        else:
            self.metrics_manager = NullMetricsManager()
        print_info(f"[INFO] {self.metrics_manager}")
        if len(self.cfg.recorders) > 0:
            self.recorder_manager = RecorderManager(self.cfg.recorders, self)
        else:
            self.recorder_manager = NullRecorderManager()
        print_info(f"[INFO] {self.recorder_manager}")

        self._configure_gym_env_spaces()

        self.monitor_manager = MonitorManager(self.cfg.monitors, self)
        print_info(f"[INFO] Monitor Manager: {self.monitor_manager}")

    def setup_manager_visualizers(self) -> None:
        super().setup_manager_visualizers()
        self.manager_visualizers["monitor_manager"] = self.monitor_manager

    def step(self, action: torch.Tensor):
        obs, reward, terminated, truncated, extras = super().step(action)
        monitor_infos = self.monitor_manager.update(dt=self.step_dt)
        extras.setdefault("step", {})
        extras["step"].update(monitor_infos)
        return obs, reward, terminated, truncated, extras

    def update_visualizers(self, visualizer: DebugVisualizer) -> None:
        super().update_visualizers(visualizer)
        terrain = self.scene.terrain
        if terrain is not None:
            terrain.debug_vis(visualizer)

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor) -> None:
        if isinstance(env_ids, Sequence):
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.int64)
        else:
            env_ids = env_ids.to(device=self.device, dtype=torch.int64)

        monitor_infos = self.monitor_manager.reset(env_ids, is_episode=True)

        super()._reset_idx(env_ids)

        self.extras["log"] = self.extras.get("log", {})
        self.extras["log"].update(monitor_infos)

    """
  Properties.
  """

    @property
    def num_rewards(self) -> int:
        return getattr(self.reward_manager, "num_rewards", 1)
