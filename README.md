# InstinctMJ · ACP with SuperMimic

[![mjlab](https://img.shields.io/badge/framework-mjlab-4C7AF2.svg)](https://github.com/mujocolab/mjlab)
[![MuJoCo Warp](https://img.shields.io/badge/simulator-MuJoCo%20Warp-silver.svg)](https://github.com/google-deepmind/mujoco_warp)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://docs.python.org/3/)
[![instinct_rl](https://img.shields.io/badge/training-instinct__rl-brightgreen.svg)](https://github.com/project-instinct/instinct_rl)

The `acp` branch reproduces **ACP**, *Humanoid Mobility via Attention-based Contacts*
([arXiv 2610.31608](https://arxiv.org/abs/2610.31608)), on the Unitree G1 in `mjlab` / MuJoCo Warp, with the paper's PPO replaced by
**SuperMimic** ([HybridRobotics/supermimic](https://github.com/HybridRobotics/supermimic)): a learned world model and backpropagation
through imagined rollouts. The ACP MDP and contact planner follow the paper; open choices are marked `# ACP-unspecified` and listed
in [src/instinct_mj/rl/acp/PLAN.md](src/instinct_mj/rl/acp/PLAN.md).

## Stages

| Stage | Task | Launcher | What is trained |
|---|---|---|---|
| 1 | `Instinct-ACP-Track-G1-v1` | `src/instinct_mj/scripts/acp/run_stage1.sh` | the contact-conditioned tracker on the reference contact targets, from scratch |
| 2 | `Instinct-ACP-Stage2-G1-v1` | `src/instinct_mj/scripts/acp/run_stage2.sh` | the attention contact planner (hindsight labels) while the tracker moves from the reference to the planner's contact commands (`eta_obs` / `eta_rew` schedules) |
| 3 | `Instinct-ACP-Stage3-G1-v1` | `src/instinct_mj/scripts/acp/run_stage3.sh` | reference tracking and goal reaching with the planner throughout, two critics (Vm, Vfs) |

Each stage starts from the previous stage's checkpoint. The play variants (`...-Play-v1`) run the trained policy.

## Learning loop

The runner (`src/instinct_mj/rl/acp/runner.py`, Stage 3: `stage3_runner.py`) follows the official SuperMimic:

- world model: LayerNorm + GELU MLP predicting velocity increments, semi-implicit integration, trained on complete 8-step windows
  with the official generalized-coordinate loss (plus the ACP contact head);
- policy: 32-step BPTT through the world model, critic bootstrap, λ-returns clamped at the analytic value bound;
- Muon + fused Adam, constant learning rate 2e-4, TF32, replay ratio 16;
- episode starts from the official `ReturnSampler`; the reference stays at the clip's last frame.

The world-model and policy updates run as CUDA graphs.

## Installation

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # if uv is not installed
uv sync
```

- Python 3.10–3.13, Linux x86_64.
- `mjlab 1.6.0` must be next to this directory as `../mjlab` (installed editable from that path).
- `instinct_rl 1.0.2` is installed from the commit recorded in `uv.lock`.

## Data

`ACP_DATASET` points to a `TerrainMotion` folder: `metadata.yaml` (motion files, terrain ids, groups), the motion `.npz` files with
their `.contacts.npz` contact labels, and `terrains/` (STL plus analytic `geoms.yaml`). The datasets are not released with this branch.

## Usage

```bash
# Stage 1 (systemd-run unit; every agent parameter is spelled out in the script)
ACP_DATASET=/path/to/dataset NUM_ENVS=4096 MAX=20000 src/instinct_mj/scripts/acp/run_stage1.sh
# Stage 2 / 3 from the previous stage
ACP_DATASET=/path/to/dataset STAGE1_CKPT=/abs/path/model_<it>.pt src/instinct_mj/scripts/acp/run_stage2.sh
ACP_DATASET=/path/to/dataset STAGE2_CKPT=/abs/path/model_<it>.pt src/instinct_mj/scripts/acp/run_stage3.sh
# resume a stopped run: RESUME_CKPT=/abs/path/model_<it>.pt; ACP_FOREGROUND=1 runs without systemd-run

# per-clip evaluation from the clip start
ACP_DATASET=/path/to/dataset uv run --no-sync python -m instinct_mj.scripts.acp.eval_tracker <run>/model_<it>.pt --mode robust --out eval.json
# live policy viewer (viser), with the planner's contact targets for Stage 2 / 3 checkpoints
uv run --no-sync python webplay_acp.py policy --checkpoint <run>/model_<it>.pt --dataset /path/to/dataset --port 8192
```

Runs are written to `logs/instinct_rl/<experiment_name>/`: `model_<it>.pt` (policy, planner) and the `acp_state_<it>.pt` sidecar
(world model, critics, optimizers) needed to resume.

## Code map

- `src/instinct_mj/tasks/acp/`: the ACP tasks (registered in `config/g1/__init__.py`); rewards, terminations and resets are defined
  in `acp_env_cfg.make_*`.
- `src/instinct_mj/rl/acp/`: the SuperMimic runner (`runner.py`, `stage3_runner.py`), the analytic MDP of the imagination (`mdp.py`,
  `stage3_mdp.py`), the planner (`planner.py`) and its hindsight training (`stage2.py`, `hindsight.py`), optimizers and CUDA-graph
  updates (`optim.py`), the plan and the paper's numbers (`PLAN.md`).
- `src/instinct_mj/scripts/acp/`: launchers, evaluation and data tools.
- `src/instinct_mj/rl/supermimic/`: the world model and helpers shared with the perceptive shadowing tasks.

When a reward, termination or observation of the env changes, the matching analytic term of the imagination in `rl/acp/mdp.py`
(Stage 3: `stage3_mdp.py`) has to change with it; `python -m instinct_mj.scripts.acp.check_stage3` compares the two for Stage 3.

## Other tasks in this repository

The perceptive shadowing parkour tasks trained with SuperMimic (`Instinct-Perceptive-Shadowing-Render-G1-v0`,
`Instinct-Perceptive-Shadowing-Latent-G1-v0`) are described in
[src/instinct_mj/rl/supermimic/README.md](src/instinct_mj/rl/supermimic/README.md).

## Acknowledgements

Built on **ACP**, **SuperMimic** (Liao et al., CoRL 2026; [HybridRobotics/supermimic](https://github.com/HybridRobotics/supermimic))
and [**supermimic_vibed**](https://github.com/zita-ch/supermimic_vibed) by Chong Zhang (ETH Zurich, MIT License). Not affiliated with
the ACP or SuperMimic authors.

```bibtex
@misc{acp2026,
  title         = {Humanoid Mobility via Attention-based Contacts},
  year          = {2026},
  eprint        = {2610.31608},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  url           = {https://arxiv.org/abs/2610.31608}
}

@misc{liao2026supermimic,
  title         = {SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model},
  author        = {Liao, Qiayuan and Shao, Yiyang and Long, Junfeng and Zakka, Kevin and Horowitz, Roberto and Abbeel, Pieter and Sreenath, Koushil},
  year          = {2026},
  eprint        = {2610.28480},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  note          = {CoRL 2026},
  url           = {https://arxiv.org/abs/2610.28480}
}

@misc{zhang2026supermimicvibed,
  title        = {supermimic\_vibed: an open reproduction of SuperMimic},
  author       = {Zhang, Chong},
  year         = {2026},
  howpublished = {\url{https://github.com/zita-ch/supermimic_vibed}}
}
```

## License

[CC BY-NC 4.0](LICENSE): no commercial use. Contributions: see [CONTRIBUTING.md](CONTRIBUTING.md) and
[CONTRIBUTOR_AGREEMENT.md](CONTRIBUTOR_AGREEMENT.md).

Checkpoints trained outside InstinctMJ are not compatible: the joint order comes from the MJCF and differs from IsaacLab.
