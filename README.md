# InstinctMJ · ACP with SuperMimic

[![mjlab](https://img.shields.io/badge/framework-mjlab-4C7AF2.svg)](https://github.com/mujocolab/mjlab)
[![MuJoCo Warp](https://img.shields.io/badge/simulator-MuJoCo%20Warp-silver.svg)](https://github.com/google-deepmind/mujoco_warp)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://docs.python.org/3/)
[![instinct_rl](https://img.shields.io/badge/training-instinct__rl-brightgreen.svg)](https://github.com/project-instinct/instinct_rl)

The `acp` branch reproduces **ACP**, *Humanoid Mobility via Attention-based Contacts*
([arXiv 2610.31608](https://arxiv.org/abs/2610.31608)), on the Unitree G1 in `mjlab` / MuJoCo Warp. The paper's PPO is replaced by
**SuperMimic** as released by its authors ([HybridRobotics/supermimic](https://github.com/HybridRobotics/supermimic)): the simulator
only collects data, a learned world model predicts the robot state, and the policy is trained by backpropagation through 32-step
imagined rollouts. The ACP MDP (observations, rewards, domain randomization, the attention-based contact planner) follows the paper;
choices the paper leaves open are marked `# ACP-unspecified` in the code and listed in
[src/instinct_mj/rl/acp/PLAN.md](src/instinct_mj/rl/acp/PLAN.md). InstinctMJ is the `mjlab` port of
[InstinctLab](https://github.com/project-instinct/InstinctLab) ([Project-Instinct](https://project-instinct.github.io/)).

## Stages

| Stage | Task | Launcher | What is trained |
|---|---|---|---|
| 1 | `Instinct-ACP-Track-G1-v1` | `src/instinct_mj/scripts/acp/run_stage1.sh` | the contact-conditioned tracker on the reference contact targets, from scratch |
| 2 | `Instinct-ACP-Stage2-G1-v1` | `src/instinct_mj/scripts/acp/run_stage2.sh` | the attention contact planner (hindsight labels) while the tracker moves from the reference to the planner's contact commands (`eta_obs` / `eta_rew` schedules) |
| 3 | `Instinct-ACP-Stage3-G1-v1` | `src/instinct_mj/scripts/acp/run_stage3.sh` | reference tracking and goal reaching with the planner throughout, two critics (Vm, Vfs) |

Each stage starts from the previous stage's checkpoint. The play variants (`...-Play-v1`) run the trained policy.

## Learning loop (official SuperMimic)

The runner (`src/instinct_mj/rl/acp/runner.py`, Stage 3: `stage3_runner.py`) implements the official SuperMimic loop only:

- **World model**: Linear → LayerNorm → GELU MLP (3 × 2048) that predicts the velocity increments; root pose and joints are
  integrated semi-implicitly with the substep coefficient (D + 1) / (2D). It is trained on complete 8-transition windows with the
  official generalized-coordinate loss (root position, root rotation as a chordal distance, joint positions, joint velocities, fixed
  nominal errors), plus the ACP contact head.
- **Policy**: 32-step BPTT through the world model with survival masking and a critic bootstrap; λ-returns (γ 0.99, λ 0.95) clamped
  at the analytic value bound of the current reward weights.
- **Critic**: LayerNorm + GELU MLP on raw features (Stage 3: Vm and Vfs, each with its own value bound).
- **Optimizers**: Muon on the hidden Linear weights, fused Adam on the rest, constant learning rate 2e-4, TF32 matmuls.
- **Schedule**: replay ratio 16 (`utd = num_envs × 16 / (10 × 2048 × 8)`).
- **Episode starts**: the official `ReturnSampler` over 1 s bins of every clip, time-outs taking precedence over terminations.
- **Domain context**: encoded by the ranges of the env's DR events.
- **Clip ends**: the reference stays at the clip's last frame, and imagination may start from any buffered state.

Speed-ups that leave the computation unchanged (verified against the previous code up to float rounding):

- CUDA graphs for the world-model update and for the Stage 1 / 2 / 3 policy updates.
- The planner attention computed as broadcast multiply-sums, compiled in training.
- A single batched penetration check of the reset noise candidates.
- The env observation group that training never reads is not computed.

Throughput: about 1.05 iterations / s at 4096 envs on an RTX 3090 (Stage 1); about 1.7 iterations / s at 2048 envs on an RTX 4090.

## Results

Vault set, 155 clips, evaluated from the clip start with domain randomization, observation noise and 0.1 action noise
(`eval_tracker.py --mode robust`, 8 episodes per clip):

| Stage 1 at iteration 7500 | success | strict success | clips with success >= 0.75 |
|---|---|---|---|
| supermimic_vibed-style loop (previous version of this branch) | 0.764 | 0.436 | 109 |
| official SuperMimic options, first batch | **0.933** | **0.868** | **153** |

Strict success additionally requires the nine contact links to stay within 0.35 m of the reference height. The current code includes
all official options; its full-dataset results are not reported yet.

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
# resume a stopped run of the same stage
RESUME_CKPT=/abs/path/model_<it>.pt src/instinct_mj/scripts/acp/run_stage1.sh
# ACP_FOREGROUND=1 runs in the foreground instead of systemd-run; UNIT, LOG, EXPERIMENT_NAME, SAVE_INTERVAL, MEMORY_MAX are overridable

# per-clip evaluation from the clip start
ACP_DATASET=/path/to/dataset uv run --no-sync python -m instinct_mj.scripts.acp.eval_tracker <run>/model_<it>.pt --mode robust --out eval.json
# live policy viewer (viser), with the planner's contact targets for Stage 2 / 3 checkpoints
uv run --no-sync python webplay_acp.py policy --checkpoint <run>/model_<it>.pt --dataset /path/to/dataset --port 8192
```

Runs are written to `logs/instinct_rl/<experiment_name>/`. A checkpoint is `model_<it>.pt` (actor, normalizers, planner; directly
playable) plus the `acp_state_<it>.pt` sidecar (world model, critics, optimizers, reset bins) needed to resume training.

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

This branch builds on **ACP**, on **SuperMimic** (Liao et al., CoRL 2026) and its official release
[HybridRobotics/supermimic](https://github.com/HybridRobotics/supermimic), whose learning loop the runner follows, and on
[**supermimic_vibed**](https://github.com/zita-ch/supermimic_vibed) by Chong Zhang (ETH Zurich, MIT License), the open reproduction
from which the world model, replay buffer, rotation utilities and training loop were first adapted. Many thanks to the authors.
This branch is not affiliated with the ACP or SuperMimic authors.

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
