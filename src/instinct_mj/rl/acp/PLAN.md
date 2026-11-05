# ACP reproduction plan (InstinctMJ, branch `acp`)

Paper: *Humanoid Mobility via Attention-based Contacts* (He, Wen, Zhang et al., arXiv 2610.31608).
Original: Isaac Lab + PPO. Here: mjlab / MuJoCo Warp + **SuperMimic** (world model + BPTT through imagined
rollouts) for the tracker; the contact planner stays supervised (it never used PPO).

## What the paper fixes exactly (implemented verbatim)

| Item | Value |
|---|---|
| Robot | Unitree G1 (authors' playground MJCF: pelvis-rooted, 23 joints, wrists welded), PD at 200 Hz, policy 50 Hz |
| Action | 23 joint-position offsets x 0.25 + default pose |
| Tracker input (468) | 5-step proprio history 375 = 5 x (ang vel 3, gravity 3, q 23, qd 23, a_prev 23), 9x7 scan (0.8x0.6 m, 0.1 m), 9x3 contact targets, goal 3 |
| Tracker / critics | MLP 512-256-128 ELU; critic inputs 353 (reference) / 298 (goal-reaching) |
| Planner | Table S3, 786,993 params (our layer sum matches exactly), 17x11 scan of 3-D points, 1.6x1.0 m |
| Planner loss | Eq. 1-2: Gaussian soft target (sigma 0.1 m) masked CE + occupancy BCE; Adam 1e-4; 10 updates / iteration, 4096 samples, 8 sub-batches, min 1000 entries |
| Hindsight labels | Fz > 1 N, drop segments < 10 steps, next onset within 75 steps, footprint [-0.8,0.8]x[-0.5,0.5], success-only episodes |
| Goal command | d_xy / max(|d_xy|, r) (r = 0.3 ref / 0.5 goal-reaching) + wrapped heading; ref target 0.5-1.5 s ahead |
| Rewards | Table 3 (weights, kernels sigma_p 0.5, sigma_r 1, sigma_v 1, sigma_w 3, sigma_q 2, sigma_qd 10, sigma_c 1) |
| Curricula | Stage 1 CSI (phase 0.9 -> -0.05 on reaching 0.99, +0.05 else, clip [0, 0.99]); Stage 2 uniform phase; Stage 3 10-level terrain curriculum, EMA 0.2, promote s > 0.5, demote > 1.0 m |
| Stage 2 schedules | eta_obs / eta_rew linear ramps (Eq. 3), u_rew > u_obs |
| DR | Table S2 |

## What the paper leaves open (our choices, marked in code as `# ACP-unspecified`)

- Terminations (not given): base height / gravity / contact-body deviation from the reference as in the shadowing tasks, plus
  timeout; goal-reaching: fall (base too low / tilt) + 15 s timeout.
- Contact-body points: link frames of pelvis, `*_wrist_yaw_link` (hand, carries the half-sphere collision), `*_elbow_link`,
  `*_knee_link`, `*_ankle_roll_link` (foot); the contact position is the link-geom contact point when in contact, else the link origin.
- u_obs, u_rew, N_ann: not reported; set from the iteration budget (defaults in config).
- Motion data: the 13.2 min corpus is not released. We start from the public G1 data (InstinctMJ kneeClimbStep set with
  terrain meshes, OmniRetarget box climbing, LAFAN1-G1 walk / get-up) and label contacts with our own tool (kinematic replay
  against the clip terrain, same 9 bodies). Swap in the authors' data when available.

## PPO -> SuperMimic mapping

| PPO element (paper) | SuperMimic equivalent here |
|---|---|
| Rollout 24 steps, 5 epochs x 8 minibatches | collect -> k-step world-model fit -> H = 32 step BPTT actor update + GAE(lambda) critic |
| Rewards from the simulator | the same Table 3 terms computed analytically on imagined states (FK); contact-label and slippage terms use the world model's 9-body contact head |
| Dual critics Vm / Vfs, separate advantage normalization | two critics on the same imagined batch, routed by env type; returns normalized per regime |
| 500-iteration warm-up of Vfs | Vfs trained, goal-reaching starts excluded from the actor loss for the first 500 iterations |
| Entropy / KL-adaptive lr | exploration noise during collection, cosine lr |
| Planner 10 updates per PPO iteration | default 10 per SuperMimic iteration; the Stage-2 run uses 1 (`--agent.planner-updates-per-iter 1`, 2026-11-05): a PPO iteration of Table S6 holds 115,200 transitions, a SuperMimic iteration ~5,100, so 1 update is ~2x the paper's updates per transition |

The world model state is (p, R, v, w, q[29], qd[29]); context: the 17x11 height scan, DR vector xi, previous action. In
imagination the observations (scan, contact targets from the reference or the frozen planner, goal command) are recomputed from
each imagined state, so no rendering is needed (scan is analytic from the cached terrain heightmap).

## Phases

1. **Data**: contact labeler -> per-clip `contact_flag (T,9)`, `contact_pos_w (T,9,3)` next to each motion npz.
2. **Env** `Instinct-ACP-Track-G1-v0` (mjlab, dict managers): obs / rewards / CSI / DR above; reference contact targets command.
3. **Stage 1 runner**: `rl/acp/runner.py` on top of `rl/supermimic` (world model, buffer, critic reused).
4. **Planner + Stage 2**: `rl/acp/planner.py`, `hindsight.py`, eta schedules, planner-in-the-loop.
5. **Stage 3**: procedural goal-reaching terrains (platforms, stones, gaps, beams, pallets, ladders), goal rewards, Vfs, fine-tune buffer.
6. **Eval / sim2sim**: success / contact error per terrain group; run in the playground MJCF (`~/Datasets/acp/playground`).

## Stage 3 (implemented: `Instinct-ACP-Stage3-G1-v0`)

Mixed env, half reference envs (Stage 2 setup, initial phase 0, mimic terms masked to them) and half goal-reaching envs on the procedural arena
of `tasks/acp/goal_terrains.py` (17 terrain types x 10 levels, tile = (type, level); the arena is added to the `terrain` body of the scene, beside
the motion-matched tiles). Goal command, curriculum, resets, goal rewards / terminations: `tasks/acp/mdp/goal.py`; env cfg:
`tasks/acp/acp_stage3_env_cfg.py`; runner: `rl/acp/stage3_runner.py` (`ACPStage3Runner(ACPRunner)`), analytic goal MDP and two-region heightmap:
`rl/acp/stage3_mdp.py`; hyperparameters `rl/acp/stage3_config.py`. Checks: `scripts/acp/check_goal_terrains.py` (no simulator),
`scripts/acp/check_stage3.py` (env, heightmap vs ray casts, reward parity, curriculum).

Runner = Stage 2 + (a) dual critic: Vm (reference, as before) and Vfs (goal command, d_g, dz, elapsed time / 15 s, planner contacts, scan, xi),
Vfs fixed lr 1e-3, trained but excluded from the actor loss for the first 500 iterations; (b) goal-env imagination `_imagine_goal`: fixed goal,
planner targets (frozen), analytic goal rewards with sigmoid-relaxed indicator gates, soft fall termination; (c) separate return normalization
(running std of each regime's discounted return); (d) goal-reaching planner replay: admitted if within 0.5 m (horizontal) at |dz| <= 0.1 m,
truncated at 0.2 m, partition keyed by (terrain type, level) with eviction of the highest work / distance episode; batches proportional to the
stored counts; (e) the world model trains on all windows, the failure head on reference windows only.

Not implemented / simplified (see the `ACP-unspecified` comments): `stumbling` is evaluated on real transitions only (the world model has no force
output); no learned failure head for goal envs (analytic soft fall only); the first goal-env transition of an imagined rollout assumes the
previous-step contact equals the current one (slippage); ladder rungs / rails are 4 cm boxes, thinner than the 5 cm heightmap resolution can resolve
exactly; single process (no multi-GPU).
