"""Loads a SuperMimic checkpoint (``OnPolicyRunner`` checkpoint format) through the play path (``instinct_mj.scripts.instinct_rl.play.run_play``: the task's registered
``SuperMimicRunner.load`` + ``get_inference_policy`` + headless rollout + ``export_as_onnx``) on the Play task.

The only patch is the dataset location of the Play env cfg (the task's default folder does not exist locally).

    .venv/bin/python tests/supermimic/check_play_load.py <checkpoint.pt> --motion-folder ~/Datasets/20251116_50cm_kneeClimbStep1
"""

from __future__ import annotations

import argparse
import os

import instinct_mj.scripts.instinct_rl.play as play

parser = argparse.ArgumentParser()
parser.add_argument("checkpoint")
parser.add_argument("--motion-folder", required=True)
parser.add_argument("--steps", type=int, default=300)
args = parser.parse_args()

original_load_env_cfg = play.load_env_cfg


def load_env_cfg_with_dataset(task_id, play=False):
    cfg = original_load_env_cfg(task_id, play=play)
    folder = os.path.expanduser(args.motion_folder)
    motion_reference_cfg = next(s for s in cfg.scene.sensors if s.name == "motion_reference")
    buf = next(iter(motion_reference_cfg.motion_buffers.values()))
    buf.path, buf.metadata_yaml = folder, os.path.join(folder, "metadata.yaml")
    terrain_cfg = cfg.scene.terrain.terrain_generator.sub_terrains["motion_matched"]
    terrain_cfg.path, terrain_cfg.metadata_yaml = buf.path, buf.metadata_yaml
    return cfg


play.load_env_cfg = load_env_cfg_with_dataset
play.run_play(
    "Instinct-Perceptive-Shadowing-Render-G1-Play-v0",
    play.PlayConfig(checkpoint_file=args.checkpoint, viewer="none", max_steps=args.steps, export_onnx=True),
)
