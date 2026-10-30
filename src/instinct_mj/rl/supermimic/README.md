# SuperMimic for perceptive shadowing (G1)

Trains the G1 depth-camera parkour policy inside a learned world model, following
[SuperMimic](https://arxiv.org/abs/2610.28480) (Liao et al., CoRL 2026) and its open reproduction
[supermimic_vibed](https://github.com/zita-ch/supermimic_vibed). The simulator only collects data; the policy is trained by
backpropagation through 32-step imagined rollouts of a differentiable world model, with analytic rewards and a critic. No PPO.

![two vision methods](assets/two_methods.png)

## Two methods

| Task | How the depth image enters imagination |
|---|---|
| `Instinct-Perceptive-Shadowing-Render-G1-v0` | The world model predicts the robot state; the depth image of every imagined state is rendered by ray casting (MuJoCo Warp, the env camera's kernel and noise). The CNN encoder is trained separately by a reconstruction decoder and a value head annealed to 0. |
| `Instinct-Perceptive-Shadowing-Latent-G1-v0` | The world model also predicts the latent of the CNN encoder; nothing is rendered. The encoder and its reconstruction decoder are trained together with the world model. |

In both, the actor sees the encoder feature through a stop-gradient. The policy network is the task's own, so checkpoints of either
method play and export (`instinct-play`, ONNX) the same way. Settings are in `tasks/shadowing/perceptive/config/g1/rl_cfg.py`.

## Results

One RTX 3090, 1024 envs, 8000 iterations, the 10 motions of `20251116_50cm_kneeClimbStep1`, one training seed.

| Method | Success | Link pos error | Joint pos error |
|---|---|---|---|
| Render | 94.9% | 0.046 m | 0.143 rad |
| Latent | 96.9% | 0.044 m | 0.134 rad |

Render: 3 seeds x 256 envs, about 2.7 h. Latent: 128 envs, from the original run before the two methods were merged into one code
base. Without the value head, Render reaches 91.0%; training the encoder by the policy gradient alone (an earlier variant, removed)
degrades the encoder below a random one.

## Usage

The dataset is the task's `MOTION_FOLDER` (default `~/Datasets/20251116_50cm_kneeClimbStep1`, from the
[release](https://github.com/project-instinct/InstinctMJ/releases/latest)).

```bash
# train (8000 iterations); resume with --agent.load-run <run> --agent.load-checkpoint model_<it>.pt
instinct-train Instinct-Perceptive-Shadowing-Render-G1-v0 --num-envs 1024
instinct-train Instinct-Perceptive-Shadowing-Latent-G1-v0 --num-envs 1024
# evaluate (success rate per motion, results in <run>/eval/)
python -m instinct_mj.scripts.supermimic.evaluate --checkpoint <run>/model_<it>.pt --motion-folder <dataset> --num-envs 128
# web viewer
python webplay_supermimic.py <run>/model_<it>.pt
```

Any field of `config.py` (`SuperMimicRunnerCfg`) can be overridden with `--agent.<field>`, e.g. `--agent.pi-batch 2048`. Single GPU.

## Files

| File | Content |
|---|---|
| `runner.py` | `SuperMimicRunner`: collection, world-model fit, imagination / BPTT, critic, logging, save / load |
| `networks.py` | World model, latent decoder, value head, failure head, critic |
| `mdp.py`, `kinematics.py`, `rotations.py`, `terrain.py` | Analytic rewards / terminations / observations, differentiable FK, terrain height scan |
| `render.py`, `depth.py` | Depth renderer (env's ray-cast kernel), camera noise pipeline |
| `buffer.py`, `sim.py`, `motion_lib.py` | Replay buffer, env readers, reference motions |
| `config.py` | All hyperparameters |

## Acknowledgements

Builds on **SuperMimic** (Liao et al., CoRL 2026) and on [**supermimic_vibed**](https://github.com/zita-ch/supermimic_vibed) by
Chong Zhang (ETH Zurich), from which the world model, replay buffer, rotation utilities and training loop were adapted (MIT License,
see `LICENSE-supermimic_vibed`). Not affiliated with the SuperMimic authors.

```bibtex
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
