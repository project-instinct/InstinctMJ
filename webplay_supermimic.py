"""Launch the viser web viewer for the SuperMimic-trained perceptive shadowing policy.

Usage: python webplay_supermimic.py [checkpoint]
(default: the newest model_*.pt of the newest SuperMimic run under logs/instinct_rl/g1_perceptive_shadowing/; the dataset is the
task's ``MOTION_FOLDER``, see ``perceptive_shadowing_cfg.py``)
"""
import glob
import os
import re
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

import yaml

import instinct_mj.scripts.instinct_rl.play as playmod
import instinct_mj.tasks  # noqa: F401  (register tasks)
from instinct_mj.scripts.instinct_rl.play import PlayConfig, run_play
from instinct_mj.tasks.shadowing.perceptive.config.g1.perceptive_shadowing_cfg import MOTION_FOLDER as _TASK_MOTION_FOLDER

MOTION_FOLDER = os.path.expanduser(_TASK_MOTION_FOLDER)
RUNS_GLOB = "logs/instinct_rl/g1_perceptive_shadowing/*supermimic*"

_real_load = playmod.load_env_cfg


def _patched_load(task_id, play=False):
    cfg = _real_load(task_id, play=play)
    # one terrain row per terrain of the dataset, so every motion has its terrain in the scene
    with open(os.path.join(MOTION_FOLDER, "metadata.yaml"), encoding="utf-8") as f:
        num_terrains = len(yaml.safe_load(f)["terrains"])
    generator = cfg.scene.terrain.terrain_generator
    generator.curriculum, generator.num_rows, generator.num_cols = True, num_terrains, 1
    cfg.episode_length_s = 30.0  # play each motion to its end
    # no reference visualization (reference robot and link markers)
    cfg.scene.entities.pop("robot_reference", None)
    motion_reference_cfg = next(s for s in cfg.scene.sensors if s.name == "motion_reference")
    motion_reference_cfg.debug_vis = False
    motion_reference_cfg.reference_entity_name = None
    return cfg


playmod.load_env_cfg = _patched_load

if len(sys.argv) > 1:
    CKPT = sys.argv[1]
else:
    runs = sorted(glob.glob(RUNS_GLOB))
    if not runs:
        raise FileNotFoundError(f"No SuperMimic run matches {RUNS_GLOB}; pass a checkpoint path.")
    CKPT = max(glob.glob(f"{runs[-1]}/model_*.pt"), key=lambda p: int(re.search(r"model_(\d+)", p).group(1)))
print(f"[webplay] checkpoint: {CKPT}", flush=True)

_real_rl_cfg = playmod.load_instinct_rl_cfg


def _patched_rl_cfg(task_id):
    # the depth encoder as trained: its nonlinearity from the run's params/agent.yaml (written by instinct-train)
    cfg = _real_rl_cfg(task_id)
    params = os.path.join(os.path.dirname(CKPT), "params", "agent.yaml")
    if os.path.isfile(params):
        with open(params, encoding="utf-8") as f:
            enc = yaml.safe_load(f)["policy"]["encoder_configs"]["depth_image"]
        cfg.policy.encoder_configs.depth_image.nonlinearity = enc["nonlinearity"]
    return cfg


playmod.load_instinct_rl_cfg = _patched_rl_cfg
cfg = PlayConfig(viewer="viser", num_envs=1, checkpoint_file=CKPT)
run_play("Instinct-Perceptive-Shadowing-G1-v0", cfg)
