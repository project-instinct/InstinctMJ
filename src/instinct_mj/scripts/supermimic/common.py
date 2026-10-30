"""Helpers shared by the SuperMimic scripts / tests: dataset override and env construction for the perceptive shadowing task.

Training and play go through the registry (``instinct-train`` / ``instinct-play`` with ``Instinct-Perceptive-Shadowing-Render-G1-v0`` /
``-Latent-G1-v0`` and their ``-Play-v0``), whose runner is ``SuperMimicRunner``; the dataset is the task's ``MOTION_FOLDER``.
"""

from __future__ import annotations

import os

import instinct_mj.tasks  # noqa: F401
from instinct_mj.tasks.registry import load_env_cfg, load_env_cls, load_vecenv_cls
from instinct_mj.tasks.shadowing.perceptive.config.g1.perceptive_shadowing_cfg import (
    TerrainMotionCfg,
    _maybe_disable_single_motion_binning,
)
from instinct_mj.tasks.shadowing.perceptive.perceptive_env_cfg import make_curriculum, make_events

TASK_ID = "Instinct-Perceptive-Shadowing-Render-G1-v0"


def apply_motion_folder(cfg, motion_folder: str | None, restore_baseline_sampling: bool = True) -> None:
    """Override the dataset of the task's env cfg (shared by training and evaluation).

    ``restore_baseline_sampling``: re-enable the training bin sampling / curriculum (the Play cfg must keep its own)."""
    if motion_folder is None:
        return
    motion_reference_cfg = next(sensor_cfg for sensor_cfg in cfg.scene.sensors if sensor_cfg.name == "motion_reference")
    motion_buffer = next(iter(motion_reference_cfg.motion_buffers.values()))
    # same as ``G1PerceptiveShadowingEnvCfg.__post_init__``
    motion_buffer.path = os.path.expanduser(motion_folder)
    motion_buffer.metadata_yaml = os.path.join(motion_buffer.path, "metadata.yaml")
    terrain_cfg = cfg.scene.terrain.terrain_generator.sub_terrains["motion_matched"]
    terrain_cfg.path = motion_buffer.path
    terrain_cfg.metadata_yaml = motion_buffer.metadata_yaml
    if not restore_baseline_sampling:
        return
    # ``load_env_cfg`` already ran ``__post_init__`` for the default dataset, which may have disabled the bin
    # sampling (single-motion datasets). Restore the baseline sampling before re-evaluating for the new dataset.
    baseline = TerrainMotionCfg()
    motion_buffer.motion_start_from_middle_range = baseline.motion_start_from_middle_range
    motion_buffer.motion_bin_length_s = baseline.motion_bin_length_s
    motion_buffer.env_starting_stub_sampling_strategy = baseline.env_starting_stub_sampling_strategy
    cfg.curriculum["beyond_adaptive_sampling"] = make_curriculum()["beyond_adaptive_sampling"]
    cfg.events["bin_fail_counter_smoothing"] = make_events()["bin_fail_counter_smoothing"]
    if _maybe_disable_single_motion_binning(motion_buffer):
        cfg.curriculum["beyond_adaptive_sampling"] = None
        cfg.events["bin_fail_counter_smoothing"] = None


def build_env_cfg(task: str, num_envs: int, motion_folder: str | None):
    """The task's (training) env cfg with the number of envs and, optionally, the dataset overridden. The camera noise models
    are part of the task cfg and ``auto_reset=False`` is set by ``SuperMimicRunner.setup_training``."""
    cfg = load_env_cfg(task)
    cfg.scene.num_envs = num_envs
    apply_motion_folder(cfg, motion_folder)
    # NOTE: the task's ``height_scanner`` / critic ``height_scan`` stay in the env although the observation is constant
    # (the sensor only includes geom group 0, which is empty; a task bug, not fixed here): removing the critic term
    # would change the critic input size of the ``EncoderActorCritic`` and break ``runner.load`` in Play. The world
    # model / critic use the analytic heightmap scan instead.
    return cfg


def make_vec_env(task: str, cfg, device: str):
    env = load_env_cls(task)(cfg=cfg, device=device, render_mode=None)
    return load_vecenv_cls(task)(env, policy_group="policy", critic_group="critic")
