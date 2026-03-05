from __future__ import annotations

import math
import torch
from abc import abstractmethod
from prettytable import PrettyTable
from typing import TYPE_CHECKING, Sequence

from mjlab.managers import ManagerBase, ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRLEnv
    from mjlab.viewer.debug_visualizer import DebugVisualizer


class MonitorSensor:
    """Base class for monitoring the environment in scene sensors.
    Need implementation of `_update_buffers_impl` to acquire data in each sim step.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        self.cfg = cfg
        self._env: ManagerBasedRLEnv = env
        self._initialize_impl()

    @property
    def device(self):
        return self._env.device

    def _initialize_impl(self):
        # placeholder
        pass

    def update(self, dt: float):
        # placeholder
        del dt

    def debug_vis(self, visualizer):
        # placeholder
        del visualizer

    # def _update_buffers_impl(self, env_ids: Sequence[int]):

    def reset(self, env_ids: Sequence[int] | slice | None = None):
        # placeholder
        del env_ids

    @abstractmethod
    def get_log(self, is_episode=False) -> dict[str, float]:
        pass


class MonitorTerm(ManagerTermBase):
    """Base class for monitoring the environment in scene managers. Typically gets data
    in each env step.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        self.cfg = cfg
        super().__init__(env)
        self._env: ManagerBasedRLEnv = env

    def update(self, *args, **kwargs):
        """Called after each environment step to acquire data from the environment."""
        pass

    def reset_idx(self, env_ids: Sequence[int] | slice):
        # placeholder
        pass

    def debug_vis(self, visualizer):
        # placeholder
        del visualizer

    @abstractmethod
    def get_log(self, is_episode=False) -> dict[str, float]:
        pass


class MonitorManager(ManagerBase):
    _env: ManagerBasedRLEnv

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        self.cfg = cfg
        self._terms: dict[str, MonitorTerm | MonitorSensor] = dict()
        self._manager_owned_sensors: set[str] = set()
        super().__init__(env=env)

        self._monitor_data = dict()

    def __str__(self) -> str:
        """Returns: A string representation for reward manager."""
        msg = f"<MonitorManager> contains {len(self._terms)} active groups.\n"

        # create table for term information
        table = PrettyTable()
        table.title = "Active Monitor Terms"
        table.field_names = ["Index", "Name", "Func"]
        # set alignment of table columns
        table.align["Group"] = "l"
        table.align["Weight"] = "r"
        # add info on each term
        index = 0
        for term_name in self._terms.keys():
            term = self._terms[term_name]
            if isinstance(term, (MonitorTerm, MonitorSensor)):
                func = term.__class__.__name__
            else:
                func = term.__name__
            table.add_row([index, term_name, func])
            index += 1
        # convert table to string
        msg += table.get_string()
        msg += "\n"

        return msg

    """
    Properties.
    """

    @property
    def active_terms(self) -> dict[str, MonitorTerm | MonitorSensor]:
        return self._terms

    @property
    def has_debug_vis_implementation(self) -> bool:
        """Whether the command terms have debug visualization implemented."""
        # check if function raises NotImplementedError
        return True

    def debug_vis(self, visualizer: "DebugVisualizer") -> None:
        """Render monitor debug visuals for terms that implement them."""
        for term in self._terms.values():
            term.debug_vis(visualizer)

    """
    Operations
    """

    def update(self, *args, dt, **kwargs):
        monitor_infos = dict()
        for term_name, term in self._terms.items():
            if isinstance(term, MonitorTerm):
                term.update(*args, dt=dt, **kwargs)
            # monitor sensors owned by monitor manager need explicit updates
            elif isinstance(term, MonitorSensor) and term_name in self._manager_owned_sensors:
                term.update(dt=dt)
            # other monitor sensors (from scene) are updated by sensor manager
            log = term.get_log(is_episode=False)
            for key, value in log.items():
                monitor_infos[f"Step_Monitor/{term_name}_{key}"] = value
        return monitor_infos

    def reset(self, env_ids: Sequence[int], is_episode=False):
        monitor_infos = dict()
        for term_name, term in self._terms.items():
            if isinstance(term, MonitorTerm):
                term.reset_idx(env_ids)
            elif isinstance(term, MonitorSensor) and term_name in self._manager_owned_sensors:
                term.reset(env_ids)
            log = term.get_log(is_episode=is_episode)
            for key, value in log.items():
                monitor_infos[f"Episode_Monitor/{term_name}_{key}"] = value
        return monitor_infos

    def get_active_iterable_terms(self, env_idx: int) -> Sequence[tuple[str, Sequence[float]]]:
        """Get the active iterable terms for the given environment index."""
        iterable_terms = []
        for term_name, term in self.active_terms.items():
            if isinstance(term, MonitorTerm):
                term_data = term.get_log(is_episode=False)
                term_data_list = []
                for data in term_data.values():
                    if isinstance(data, torch.Tensor):
                        data = data.cpu().item()
                    if not (math.isnan(data) or math.isinf(data)):
                        term_data_list.append(data)
                    else:
                        term_data_list.append(0.0)
                if term_data_list:
                    iterable_terms.append((term_name, term_data_list))
        return iterable_terms

    def _prepare_terms(self):
        from .monitor_cfg import MonitorSensorCfg

        # check if config is dict already
        if isinstance(self.cfg, dict):
            cfg_items = self.cfg.items()
        else:
            cfg_items = self.cfg.__dict__.items()
        for term_name, term_cfg in cfg_items:
            if term_cfg is None:
                continue
            if isinstance(term_cfg, SceneEntityCfg):
                term = self._env.scene[term_cfg.name]
            elif isinstance(term_cfg, MonitorSensorCfg):
                term = term_cfg.class_type(term_cfg, self._env)
                self._manager_owned_sensors.add(term_name)
            else:
                self._resolve_common_term_cfg(term_name, term_cfg)
                term = term_cfg.func
            self._terms[term_name] = term
