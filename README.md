# InstinctMJ

[![mjlab](https://img.shields.io/badge/framework-mjlab-4C7AF2.svg)](https://github.com/mujocolab/mjlab)
[![MuJoCo Warp](https://img.shields.io/badge/simulator-MuJoCo%20Warp-silver.svg)](https://github.com/google-deepmind/mujoco_warp)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://docs.python.org/3/)
[![Platform](https://img.shields.io/badge/platform-linux--x86__64-orange.svg)](https://releases.ubuntu.com/)
[![instinct_rl](https://img.shields.io/badge/training-instinct__rl-brightgreen.svg)](https://github.com/project-instinct/instinct_rl)

`InstinctMJ` builds InstinctLab task families on top of `mjlab`, which combines Isaac Lab's manager-based API with MuJoCo Warp, a GPU-accelerated version of MuJoCo.
The package focuses on behavior-preserving task migration, expressed in native `mjlab` managers, scenes, and task registration for `instinct_rl`.

**Key Features:**

- `Native mjlab stack` Build tasks with `mjlab`'s manager-based API and run them on the GPU through `mujoco_warp`, without a runtime IsaacLab compatibility layer.
- `Behavior-preserving migration` Port locomotion, shadowing, perceptive, and parkour tasks while keeping task semantics aligned with InstinctLab.
- `Standalone task package` Keep the task suite outside the core `mjlab` repository while registering environments through the `mjlab.tasks` entry point.
- `Project-Instinct workflow` Plug directly into `instinct_rl` for train / play / export workflows, with logs under `logs/instinct_rl/<experiment_name>/<timestamp_run>/`.

**Keywords:** mjlab, mujoco-warp, instinct_rl, humanoid

## Warning

This codebase is under [CC BY-NC 4.0 license](LICENSE). As with InstinctLab, you may not use the material for commercial purposes, for example to advertise commercial products or redistribute the code as part of a commercial offering.

## Attention

Do not directly use checkpoints trained in InstinctLab with `InstinctMJ`.

- `InstinctMJ` loads the robot from XML / MJCF, and the resulting joint order is not the same as the joint order used in IsaacLab.
- Even when the task semantics are migrated one-to-one, policy inputs / outputs tied to joint ordering are therefore not directly checkpoint-compatible.
- In practice, InstinctLab weights should be treated as incompatible with `InstinctMJ` unless you perform an explicit, validated joint-order remapping.
- Please release and use weights trained in `InstinctMJ` for `InstinctMJ` tasks.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) and [CONTRIBUTOR_AGREEMENT.md](CONTRIBUTOR_AGREEMENT.md) for contribution requirements.

## Installation

- Install `mjlab` first by following the upstream setup instructions in the [mjlab repository](https://github.com/mujocolab/mjlab).

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

- Or install editable packages with `pip`:

  ```bash
  pip install -e ../mjlab
  pip install -e ../instinct_rl
  pip install -e .
  ```

- After installation, you can run the training workflow directly with `instinct_rl`-style commands:

  ```bash
  instinct-train Instinct-Locomotion-Flat-G1-v0
  instinct-play Instinct-Locomotion-Flat-G1-Play-v0 --load-run <run_name>
  ```

## Documentation of Critical Components

- [instinct_rl Documentation](https://github.com/project-instinct/instinct_rl/blob/main/README.md)
- [mjlab Repository](https://github.com/mujocolab/mjlab)
- [MuJoCo Warp Repository](https://github.com/google-deepmind/mujoco_warp)
- Local `mjlab` reference in this workspace: `../mjlab`
- Original InstinctLab repository: `https://github.com/project-instinct/InstinctLab`
- Original InstinctLab README: `https://github.com/project-instinct/InstinctLab/blob/main/README.md`
- Shadowing documentation: `src/instinct_mj/tasks/shadowing/README.md`
- BeyondMimic documentation: `src/instinct_mj/tasks/shadowing/beyondmimic/README.md`
- Parkour documentation: `src/instinct_mj/tasks/parkour/README.md`

### Set up IDE (Optional)

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
  --checkpoint <checkpoint_file>
```

Pretrained weights:

- Google Drive: `<add_link_here>`

Export ONNX for parkour:

```bash
instinct-play Instinct-Parkour-Target-Amp-G1-Play-v0 --load-run <run_name> --export-onnx
```

Module form is also available when console scripts are not on `PATH`:

```bash
python -m instinct_mj.scripts.instinct_rl.train Instinct-Locomotion-Flat-G1-v0
python -m instinct_mj.scripts.instinct_rl.play Instinct-Locomotion-Flat-G1-Play-v0 --load-run <run_name>
python -m instinct_mj.scripts.list_envs
```

## Code Formatting

We use `pre-commit` for formatting and hygiene checks, similar to the original InstinctLab workflow.

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

## Repository Layout

- `src/instinct_mj/tasks` — task registration and family-specific configs
- `src/instinct_mj/envs` — environment wrappers, manager extensions, and shared MDP terms
- `src/instinct_mj/motion_reference` — motion data loaders, buffers, and reference managers
- `src/instinct_mj/assets` — MuJoCo robot assets and resource files
- `src/instinct_mj/scripts` — train, play, visualization, and data-processing entry points

## Data and Outputs

- Training logs are written to `logs/instinct_rl/<experiment_name>/<timestamp_run>/`
- Play videos are saved under `videos/play/` in the selected run directory

## Relationship to InstinctLab

`InstinctMJ` is the MuJoCo Warp / `mjlab` counterpart to InstinctLab in the Project-Instinct ecosystem.

Reference links:

- Original repository: `https://github.com/project-instinct/InstinctLab`
- Original README: `https://github.com/project-instinct/InstinctLab/blob/main/README.md`
- Local reference in this workspace: `../InstinctLab`
