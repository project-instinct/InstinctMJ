"""``instinct-train`` for the ACP tasks on another dataset folder: ``ACP_DATASET=<dir> python -m instinct_mj.scripts.acp.train_on_dataset <task> ...``
(the motion folder of the task config is overridden for this process only; the task default stays)."""

import os

import instinct_mj.tasks.acp.config.g1.acp_tracking_cfg as acp_g1

dataset = os.path.expanduser(os.environ["ACP_DATASET"])
buffer = acp_g1.motion_reference_cfg.motion_buffers["TerrainMotion"]
buffer.path = dataset
buffer.metadata_yaml = os.path.join(dataset, "metadata.yaml")

from instinct_mj.scripts.instinct_rl.train import main  # noqa: E402

main()
