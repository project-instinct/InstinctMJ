# InstinctMJ

[![mjlab](https://img.shields.io/badge/framework-mjlab-4C7AF2.svg)](https://github.com/mujocolab/mjlab)
[![MuJoCo Warp](https://img.shields.io/badge/simulator-MuJoCo%20Warp-silver.svg)](https://github.com/google-deepmind/mujoco_warp)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://docs.python.org/3/)
[![Platform](https://img.shields.io/badge/platform-linux--x86__64-orange.svg)](https://releases.ubuntu.com/)
[![instinct_rl](https://img.shields.io/badge/training-instinct__rl-brightgreen.svg)](https://github.com/project-instinct/instinct_rl)

## Overview

This repository is the `mjlab-native port` of [InstinctLab](https://github.com/project-instinct/InstinctLab), serving as the environment side of [Project-Instinct](https://project-instinct.github.io/).

We aim to industrialize Reinforcement Learning for humanoid whole-body control, with task families implemented on top of `mjlab` and integrated into the Project-Instinct training workflow.

**Key Features:**

- `Standalone package` Work outside the core `mjlab` repository while keeping task development self-contained.
- `Task suite` Provide locomotion, shadowing, perceptive, and parkour task families for humanoid control on `mjlab`.
- `Unified ecosystem` Integrate directly with [instinct_rl](https://github.com/project-instinct/instinct_rl) for train / play / export workflows.
- `Structured outputs` Keep experiment logs under `logs/instinct_rl/<experiment_name>/<timestamp_run>/` to match the Project-Instinct workflow.

**Keywords:** mjlab, mujoco-warp, instinct_rl, humanoid

## Warning

This codebase is under [CC BY-NC 4.0 license](LICENSE). You may not use the material for commercial purposes, for example to advertise commercial products or redistribute the code as part of a commercial offering.

## Attention

Do not directly use checkpoints trained outside `InstinctMJ` with `InstinctMJ`.

- `InstinctMJ` loads the robot from XML / MJCF, and the resulting joint order is not the same as the joint order used in IsaacLab.
- Policy inputs / outputs tied to joint ordering are therefore not directly checkpoint-compatible across different simulator setups.
- Please release and use weights trained in `InstinctMJ` for `InstinctMJ` tasks.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) and [CONTRIBUTOR_AGREEMENT.md](CONTRIBUTOR_AGREEMENT.md) for contribution requirements.

## Installation

- Install `mjlab` first by following the upstream setup instructions in the [mjlab repository](https://github.com/mujocolab/mjlab).
- This repository is currently validated against `mjlab` commit `6abd0ebfde7ef07b3ecb6032c0efe3fda2794c5c`.

- Install `instinct_rl` by following the [instinct_rl README](https://github.com/project-instinct/instinct_rl/blob/main/README.md).
  TL; DR:

  ```bash
  git clone https://github.com/project-instinct/instinct_rl.git
  python -m pip install -e instinct_rl
  ```

- Clone this repository into any common workspace directory as a sibling of `mjlab` and `instinct_rl`:

  ```bash
  mkdir -p <workspace_dir>
  cd <workspace_dir>

  # Option 1: HTTPS
  git clone https://github.com/mujocolab/mjlab.git
  git clone https://github.com/project-instinct/instinct_rl.git
  git clone https://github.com/project-instinct/InstinctMJ.git
  cd mjlab
  git checkout 6abd0ebfde7ef07b3ecb6032c0efe3fda2794c5c
  cd ..

  # Option 2: SSH
  # git clone git@github.com:mujocolab/mjlab.git
  # git clone git@github.com:project-instinct/instinct_rl.git
  # git clone git@github.com:project-instinct/InstinctMJ.git
  ```

- Install the package with `uv`:

  ```bash
  cd InstinctMJ
  uv sync
  ```

- If you are actively developing against sibling `mjlab` / `instinct_rl` checkouts,
  reinstall those local repositories in editable mode after `uv sync`.
  Otherwise, `uv` will keep using the git-pinned sources from `pyproject.toml` / `uv.lock`.

  ```bash
  uv pip install --python .venv/bin/python --no-deps -e ../mjlab -e ../instinct_rl
  ```

- Or install editable packages from the public repositories with `pip`:

  ```bash
  pip install -e "git+https://github.com/mujocolab/mjlab.git@6abd0ebfde7ef07b3ecb6032c0efe3fda2794c5c#egg=mjlab"
  pip install -e "git+https://github.com/project-instinct/instinct_rl.git#egg=instinct_rl"
  pip install -e .
  ```

- After installation, you can run the training workflow directly with `instinct_rl`-style commands:

  ```bash
  instinct-train Instinct-Locomotion-Flat-G1-v0
  instinct-play Instinct-Locomotion-Flat-G1-Play-v0 --load-run <run_name>
  ```

## Set up IDE (Optional)

If VSCode / Pylance misses local imports in a multi-repository workspace, add these paths to `.vscode/settings.json`:

```json
{
  "python.analysis.extraPaths": [
    "<workspace_dir>/InstinctMJ/src",
    "<workspace_dir>/mjlab/src",
    "<workspace_dir>/instinct_rl"
  ]
}
```

## Task Suite

Registered task IDs:

- `Instinct-Locomotion-Flat-G1-v0`
- `Instinct-Locomotion-Flat-G1-Play-v0`
- `Instinct-BeyondMimic-Plane-G1-v0`
- `Instinct-BeyondMimic-Plane-G1-Play-v0`
- `Instinct-Shadowing-WholeBody-Plane-G1-v0`
- `Instinct-Shadowing-WholeBody-Plane-G1-Play-v0`
- `Instinct-Perceptive-Shadowing-G1-v0`
- `Instinct-Perceptive-Shadowing-G1-Play-v0`
- `Instinct-Perceptive-Vae-G1-v0`
- `Instinct-Perceptive-Vae-G1-Play-v0`
- `Instinct-Parkour-Target-Amp-G1-v0`
- `Instinct-Parkour-Target-Amp-G1-Play-v0`

Use the CLI to inspect the full list at any time:

```bash
instinct-list-envs
instinct-list-envs shadowing
```

## Quick Start

Train:

```bash
instinct-train Instinct-Locomotion-Flat-G1-v0
instinct-train Instinct-Perceptive-Shadowing-G1-v0
```

Play (`--load-run` is required):

```bash
instinct-play Instinct-Locomotion-Flat-G1-Play-v0 --load-run <run_name>
instinct-play Instinct-Perceptive-Shadowing-G1-Play-v0 --load-run <run_name>
```

Play perceptive shadowing with released weights:

```bash
instinct-play Instinct-Perceptive-Shadowing-G1-Play-v0 \
  --load-run <downloaded_run_dir> \
  --checkpoint-file <checkpoint_file>
```

Pretrained weights:

- Google Drive: [Pretrained weights](https://drive.google.com/drive/folders/1RPjbZjurknhlvlj9dxAUkARkcEyexQvF?usp=sharing)

Export ONNX for parkour:

```bash
instinct-play Instinct-Parkour-Target-Amp-G1-Play-v0 --load-run <run_name> --export-onnx
```

Play parkour with released weights:

```bash
instinct-play Instinct-Parkour-Target-Amp-G1-Play-v0 \
  --load-run <downloaded_run_dir> \
  --checkpoint-file <checkpoint_file>
```

Parkour pretrained weights:

- Google Drive: [Parkour pretrained weights](https://drive.google.com/drive/folders/1B2AP5MEC5hDF7w5ws9oIiRKStL_gRxhE?usp=drive_link)

Before training or playing parkour tasks, update the local dataset root in
`src/instinct_mj/tasks/parkour/config/g1/g1_parkour_target_amp_cfg.py`:

```python
_PARKOUR_DATASET_DIR = os.path.expanduser("~/your/path/to/parkour_motion_reference")
```

If your filtered motion list is stored elsewhere, also update
`filtered_motion_selection_filepath` in the same file. See
`src/instinct_mj/tasks/parkour/README.md` for the task-specific notes.

Module form is also available when console scripts are not on `PATH`:

```bash
python -m instinct_mj.scripts.instinct_rl.train Instinct-Locomotion-Flat-G1-v0
python -m instinct_mj.scripts.instinct_rl.play Instinct-Locomotion-Flat-G1-Play-v0 --load-run <run_name>
python -m instinct_mj.scripts.list_envs
```

## Code Formatting

We use `pre-commit` for formatting and hygiene checks.

Install `pre-commit`:

```bash
pip install pre-commit
```

Run all checks:

```bash
pre-commit run --all-files
```

Or use the local helper command:

```bash
instinct-format
```

To enable hooks on every commit:

```bash
pre-commit install
```

## Train Your Own Projects

To preserve your own experiments and logs, it is usually better to create your own task package or repository and reuse the task patterns from `InstinctMJ`.

If you want to add a new task directly in this repository:

- Create a new folder under `src/instinct_mj/tasks/<your_project>/`.
- Add `__init__.py` at each package level.
- Register tasks with `register_instinct_task()`.
- Keep the environment config and `instinct_rl` config colocated in the task package.

Example registration pattern:

```python
from instinct_mj.tasks.registry import register_instinct_task

from .my_env_cfg import MyEnvCfg, MyEnvCfg_PLAY
from .rl_cfgs import my_instinct_rl_cfg

register_instinct_task(
    task_id="Instinct-My-Task-v0",
    env_cfg_factory=MyEnvCfg,
    play_env_cfg_factory=MyEnvCfg_PLAY,
    instinct_rl_cfg_factory=my_instinct_rl_cfg,
)
```
