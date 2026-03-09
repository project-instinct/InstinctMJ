# Shadowing Task

Usage notes for the `InstinctMJ` shadowing task family on `mjlab`.

## Prerequisite

Install `mjlab` and `instinct_rl` source code first (see `InstinctMJ/README.md` for full setup), then install this package so `instinct-train` and `instinct-play` are available.

## Basic Usage Guidelines

### BeyondMimic Shadowing

**Task IDs:**
- `Instinct-BeyondMimic-Plane-G1-v0` (train)
- `Instinct-BeyondMimic-Plane-G1-Play-v0` (play)

This task follows the BeyondMimic whole-body tracking setup.

1. Go to `beyondmimic/config/g1/beyondmimic_plane_cfg.py` and update the local motion dataset settings here:

    ```python
    MOTION_NAME = "LafanSprint1"
    _hacked_selected_file_ = "sprint1_subject2_retargetted.npz"
    path=os.path.expanduser("~/Xyk/Datasets/UbisoftLAFAN1_GMR_g1_29dof_torsoBase_retargetted_instinctnpz")
    ```

    - `MOTION_NAME`: Motion setup name used by the config.
    - `_hacked_selected_file_`: The motion file to load, relative to the dataset root.
    - `path=os.path.expanduser(...)`: The local dataset root you need to change on your machine.

2. Train the policy:
```bash
instinct-train Instinct-BeyondMimic-Plane-G1-v0
```

3. Play trained policy (`--load-run` is required; absolute path is recommended, or use `--agent random` for an untrained policy):
```bash
instinct-play Instinct-BeyondMimic-Plane-G1-Play-v0 --load-run <run_name>
```

### Whole Body Shadowing

**Task IDs:**
- `Instinct-Shadowing-WholeBody-Plane-G1-v0` (train)
- `Instinct-Shadowing-WholeBody-Plane-G1-Play-v0` (play)

1. Go to `whole_body/config/g1/plane_shadowing_cfg.py` and update the motion dataset settings used by the active block:

    ```python
    MOTION_NAME = "LafanFiltered"
    _hacked_selected_files_ = [
        ...
    ]
    path=os.path.expanduser("~/Xyk/Datasets/NoKov-Marslab-Motions-instinctnpz/20251016_diveroll4_single")
    ```

    - `MOTION_NAME`: Motion setup name used by the config.
    - `_hacked_selected_files_`: One or more motion files to load, relative to the dataset root.
    - `path=os.path.expanduser(...)`: The actual active dataset root used by `motion_buffers[MOTION_NAME]`.
    - `_path_`: Exists in some preset blocks, but if you use the current active config you should update the `path=...` inside `motion_buffers`.

2. Train the policy:
```bash
instinct-train Instinct-Shadowing-WholeBody-Plane-G1-v0
```

3. Play trained policy (`--load-run` is required; absolute path is recommended, or use `--agent random` for an untrained policy):
```bash
instinct-play Instinct-Shadowing-WholeBody-Plane-G1-Play-v0 --load-run <run_name>
```

### Perceptive Shadowing

**Task IDs:**
- `Instinct-Perceptive-Shadowing-G1-v0` (train)
- `Instinct-Perceptive-Shadowing-G1-Play-v0` (play)

1. Go to `perceptive/config/g1/perceptive_shadowing_cfg.py` and update the local dataset root here:

    ```python
    MOTION_FOLDER = "~/Xyk/Datasets/20251116_50cm_kneeClimbStep1"
    ```

    The motion buffer and terrain generator read `MOTION_FOLDER` and the `metadata.yaml` under that directory.

    - `MOTION_FOLDER`: The local folder containing the motion files and `metadata.yaml`.

2. Train the policy:
```bash
instinct-train Instinct-Perceptive-Shadowing-G1-v0
```

3. Play trained policy (`--load-run` is required; absolute path is recommended, or use `--agent random` for an untrained policy):
```bash
instinct-play Instinct-Perceptive-Shadowing-G1-Play-v0 --load-run <run_name>
```

4. Current maintained setup notes in this workspace (as of `2026-03-09`):

    - Pretrained weights: [Google Drive folder](https://drive.google.com/drive/folders/1RPjbZjurknhlvlj9dxAUkARkcEyexQvF?usp=sharing)
    - Play command:

```bash
instinct-play Instinct-Perceptive-Shadowing-G1-Play-v0 \
  --load-run <downloaded_run_dir> \
  --checkpoint-file <checkpoint_file>
```

### Perceptive VAE

**Task IDs:**
- `Instinct-Perceptive-Vae-G1-v0` (train)
- `Instinct-Perceptive-Vae-G1-Play-v0` (play)

1. Go to `perceptive/config/g1/perceptive_vae_cfg.py` and update the local dataset root here:

    ```python
    MOTION_FOLDER = "~/Xyk/Datasets/20251116_50cm_kneeClimbStep1"
    ```

    The VAE motion buffer and terrain generator read `MOTION_FOLDER` and the `metadata.yaml` under that directory.

    - `MOTION_FOLDER`: The local folder containing the motion files and `metadata.yaml`.

2. Train the policy:
```bash
instinct-train Instinct-Perceptive-Vae-G1-v0
```

3. Play trained policy (`--load-run` is required; absolute path is recommended, or use `--agent random` for an untrained policy):
```bash
instinct-play Instinct-Perceptive-Vae-G1-Play-v0 --load-run <run_name>
```

## Common Options

- `--num-envs`: Number of parallel environments (default varies by task)
- `--load-run`: Run name/path pattern to select a checkpoint for play
- `--checkpoint-file`: Explicit checkpoint file to load inside the selected run
- `--agent`: Action source for play, e.g. `trained`, `random`, or `zero`
- `--device`: Runtime device, e.g. `cuda:0`
- `--viewer`: Viewer backend (`none`/`native` for train, `auto`/`native`/`viser`/`none` for play)
- `--video`: Record training/playback videos

Module form (if console scripts are not available):

```bash
python -m instinct_mj.scripts.instinct_rl.train Instinct-BeyondMimic-Plane-G1-v0
python -m instinct_mj.scripts.instinct_rl.play Instinct-BeyondMimic-Plane-G1-Play-v0 --load-run <run_name>
```
