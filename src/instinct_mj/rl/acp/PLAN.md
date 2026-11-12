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
| Hindsight labels | Fz > 1 N (v1 code: the same SDF distance criterion as the env's c^meas, see "SDF contact"), drop segments < 10 steps, next onset within 75 steps, footprint [-0.8,0.8]x[-0.5,0.5], success-only episodes |
| Goal command | d_xy / max(|d_xy|, r) (r = 0.3 ref / 0.5 goal-reaching) + wrapped heading; ref target 0.5-1.5 s ahead |
| Rewards | Table 3 (weights, kernels sigma_p 0.5, sigma_r 1, sigma_v 1, sigma_w 3, sigma_q 2, sigma_qd 10, sigma_c 1) |
| Curricula | Paper: Stage 1 CSI (phase 0.9 -> -0.05 on reaching 0.99, +0.05 else, clip [0, 0.99]); Stage 2 uniform phase. **Stage 1 now uses the supermimic_vibed reset bins instead of ACP CSI (2026-11-09, `run_stage1.sh` default `RESET_SAMPLING=bins`, `bins_v2=False`, `bins_difficulty_k=0`); CSI remains available with `RESET_SAMPLING=csi`**; Stage 3 starts at phase 0 and has the 10-level terrain curriculum, EMA 0.2, promote s > 0.5, demote > 1.0 m |
| Stage 2 schedules | eta_obs / eta_rew linear ramps (Eq. 3), u_rew > u_obs |
| DR | Table S2 |

## What the paper leaves open (our choices, marked in code as `# ACP-unspecified`)

- Terminations (not given): v1 uses SuperMimic's (see the v1 section); goal-reaching envs: fall (base too low / tilt) + 15 s timeout.
- Contact-body points: one link per contact body - pelvis, `*_wrist_yaw_link` (hand, carries the palm sphere), `*_elbow_link`,
  `*_knee_link`, `*_ankle_roll_link` (foot); the contact position is the link-geom contact point when in contact, else the link origin.
  The points are sampled on the collision geometry the task simulates, i.e. the authors' collision meshes (`G1AcpTrackV1EnvCfg`
  keeps them; `simplified_collision=True` swaps in analytic capsules / spheres as an opt-in) - labels and simulation measure one
  surface.
- u_obs, u_rew, N_ann: not reported; set from the iteration budget (defaults in config).
- Motion data: the 13.2 min corpus is not released. We start from the public G1 data (InstinctMJ kneeClimbStep set with
  terrain meshes, OmniRetarget box climbing, LAFAN1-G1 walk / get-up) and label contacts with our own tool (kinematic replay
  against the clip terrain, same 9 bodies, `v3_final` / `v3_train6` labelled on the collision meshes). Swap in the authors' data
  when available.

## PPO -> SuperMimic mapping

| PPO element (paper) | SuperMimic equivalent here |
|---|---|
| Rollout 24 steps, 5 epochs x 8 minibatches | collect -> k-step world-model fit -> H = 32 step BPTT actor update + GAE(lambda) critic |
| Rewards from the simulator | the same Table 3 terms computed analytically on imagined states (FK); contact-label and slippage terms use the soft sphere-SDF contact of the predicted state (`contact_detect="sdf"`; "force_head": the world model's 9-body contact head) |
| Dual critics Vm / Vfs, separate advantage normalization | two critics on the same imagined batch, routed by env type; returns normalized per regime |
| 500-iteration warm-up of Vfs | Vfs trained, goal-reaching starts excluded from the actor loss for the first 500 iterations |
| Entropy / KL-adaptive lr | exploration noise during collection, constant lr (the official supermimic has no lr schedule) |
| Planner 10 updates per PPO iteration | `ACPRunnerCfg` default 10 per SuperMimic iteration; `G1AcpStage2RunnerCfg` (and `run_stage2.sh`) uses 1 (2026-11-05, v1): a PPO iteration of Table S6 holds 115,200 transitions, a SuperMimic iteration ~5,100, so 1 update is ~2x the paper's updates per transition |

The world model state is (p, R, v, w, q[29], qd[29]); context: the 17x11 height scan, DR vector xi, previous action. In
imagination the observations (scan, contact targets from the reference or the frozen planner, goal command) are recomputed from
each imagined state, so no rendering is needed (scan is analytic from the cached terrain heightmap).

## v1 task set (the only registered ACP tasks)

| Task id | Env cfg | Runner | Launcher |
|---|---|---|---|
| `Instinct-ACP-Track-G1-v1` (+ `-Play-v1`) | `G1AcpTrackV1EnvCfg` (`_PLAY`), run name `g1AcpTrackV1` | `ACPRunner`, `G1AcpTrackRunnerCfg`, experiment `g1AcpTrackV1` | `scripts/acp/run_stage1.sh` |
| `Instinct-ACP-Stage2-G1-v1` | `G1AcpStage2V1EnvCfg` = Stage 1 + episode cap 20 s, `g1AcpStage2V1` | `ACPRunner`, `G1AcpStage2RunnerCfg`, `g1AcpStage2V1` | `scripts/acp/run_stage2.sh` (needs `STAGE1_CKPT`) |
| `Instinct-ACP-Stage3-G1-v1` (+ `-Play-v1`) | `G1AcpStage3V1EnvCfg`, `g1AcpStage3V1` | `ACPStage3Runner`, `G1AcpStage3RunnerCfg`, `g1AcpStage3V1` | `scripts/acp/run_stage3.sh` (needs `STAGE2_CKPT`) |

The three stages share ONE definition of the MDP: `acp_env_cfg.make_rewards()` / `make_terminations()` / `make_events()` (the base env cfg lists the final terms
explicitly; there are no override layers). Stage 2 only changes the episode cap (20 s: the hindsight labels need episodes that progress >= 90 % of a clip and the longest
clip is 18.7 s). Stage 3 masks the reference terms to the reference envs (`goal.py`: `_ref_only` wrappers, `stage3_mdp.REF_ONLY_TERMS`), adds the paper's goal terms and
the goal envs' own `goal_fall` / `goal_time_out`; no episode cap besides the 120 s upper bound (the reference envs start at phase 0 and must reach 90 % of the clip).
All run with `AcpRlEnv` (`tasks/acp/env.py`).

Final v1 configuration (every deviation from the paper or from SuperMimic with its reason):

- **Rewards** = ACP Table 3 (`track_base_pos` sigma 0.5, `track_contact_pos_global` sigma 1.0, ...) + SuperMimic's `track_body_pos_anchor` (0.3 m) / `track_body_ori` (0.4) /
  `track_body_lin_vel` (1.0) / `track_body_ang_vel` (3.14), weight 1.0 each, mean over the 14 tracked bodies (`TRACKED_LINKS`: the nine contact links + torso, hip-roll and
  shoulder-roll links). Reason: the SuperMimic body terms anchor the whole body, not only the contact links (user decision 2026-11-07); SuperMimic's two root terms are
  not added (`track_base_pos` covers them). `contact_acc` is the difference of the link velocities (`mode="vel1"`, the authors' framework).
- **eta_rew hand-over** (`ACPMDP.weights`): the three reference-contact terms of Table 3 and `track_body_pos_anchor` (a body-position term, the planner replaces the
  reference positions) fade with `1 - eta_rew`; `track_body_ori` / `track_body_lin_vel` / `track_body_ang_vel` keep their weight (the paper changes only the three contact
  terms).
- **Terminations** = SuperMimic's: failure only on the pelvis height (`base_pos_too_far`, |z - z_ref| > 0.25 m, height only) and tilt (`base_pg_too_far`, |R_zz - R_zz_ref| > 0.8, z only); time-outs (bootstrapped,
  not failures): the 10 s cap (20 s in Stage 2), the clip end (`dataset_exhausted`), the reset guard `illegal_reset_contact`. The paper gives no thresholds; SuperMimic App. D.1
  does not terminate on end-effector deviation. The earlier variants (`link_pos_too_far`, `base_xy_too_far`, `contact_far_from_ref`, height 0.5 m) were removed.
  The imagination mirrors the active terms (`ACPMDP.termination`, `height_only` honoured).
- **Episode starts**: Stage 1: supermimic_vibed reset bins (launcher default), with the paper's CSI available via `RESET_SAMPLING=csi`; Stage 2 / 3 launchers: reset bins; reset noise = zita-ch/supermimic_vibed (2026-11-09, user; was Table S2's base height + U(0, 0.05) m /
  joints +-0.2 rad): root position U(+-0.05, +-0.05, +-0.01) m, world-frame rotation (roll, pitch, yaw) U(+-0.1, +-0.1, +-0.2) rad, world linear velocity U(+-0.5, +-0.5, +-0.2) m/s,
  angular velocity U(+-0.52, +-0.52, +-0.78) rad/s, joints U(+-0.1) rad clamped to the joint limits (the world model sees the states around the reference); a start whose noise
  leaves the robot > 1 cm inside the terrain is redrawn (3 times, then the noise-free reference) and the root is lifted by the deepest remaining penetration (this handles the
  contact poses at ladder rungs / boxes that the reference noise can push into the terrain). Stage 1 bins keep the sampled frame without legal-start snapping; CSI and Stage 2 retain the 1 cm legal-start table.
- **Soft joint limits** = 0.9 x the joint range (SuperMimic `robot.py:144`, authors' `acp/g1.py:172`); the imagination reads `soft_joint_pos_limits` from the live env.
- **Fresh kinematics**: `AcpRlEnv` calls `sim.forward()` before the termination manager (mjlab computes rewards / terminations on `xpos` / `xvel` of the state before the last physics
  substep). Only the ACP tasks; the shared mjlab step and the other tasks are unchanged. The (since removed) `check_mdp_parity` showed position terms <= 1e-5 relative to the env.
- **Critic input** (`ACPMDP.critic_features`, `critic_dim`): + the anchored position error `ref_bp - bp` of the 14 tracked bodies in the robot yaw frame (42 dims, SuperMimic mdp.py:174);
  checkpoints of earlier versions do not load.
- **World-model contact input** (`wm_ref_contacts`): the contact command the tracker's observation actually carries (27 values + validity): stored at collection (`wm_cmd` in the buffer), in
  the imagination the part of the assembled policy observation (reference + noise, planner, or the Stage 2 mix). Stage 3 supports it (`_imagine_goal` passes the planner command).
- **Planner**: the imagined planner scan noise reads the env's `z_noise_std`; Stage 2 trains the planner with 1 optimizer step per iteration.
- **Env reward / measured contact in Stage 1 (2026-11-08)**: the actor and the critic train on the imagined rewards only; the env reward feeds the reset bins and the Stage 2 / 3
  reward sync. With CSI (Stage 1) the runner therefore does not evaluate the env's reward manager (`ACPRunner.env_reward`) and does not
  read the measured contact (`read_contact`; only Stage 3 and the "force_head" contact head use it). The env reward (`sim/reward`, `sim/reward_term/*`, `sim_r`) is not logged in any stage. Measured at 2048 envs on the 3090 (v3_final, launcher config): 190 ms per
  collection step, of which the reward manager 33 ms and the second SDF evaluation of `reader.contact()` 27 ms (the SDF contact itself ~28 ms per evaluation).
- **Launch scripts (review 2026-11-07, overnight/ACP_SM_FUSION_REVIEW.md, CODEX_FUSION_REVIEW.md)**: all three stages start imagination only where f + H <= clip end
  (`imagine_past_clip_end` and the other clip-end switches False, as SuperMimic's default), plain uniform bin half (`bins_difficulty_k` 0), critic 256-128, no contact command /
  contact feedback inputs to the world model (`wm_ref_contacts` / `wm_contact_feedback` False: the command has no causal effect on the next state given the action);
  Stage 3 trains the planner with 1 optimizer step per iteration like Stage 2.
- **Known gaps**: Stage 2 starts raising eta_obs while the planner has had only ~1000 updates (no warm-up / metric gate yet); the Stage 1 reference contact segments of feet / hands
  may be 4 frames while the hindsight labels need 10 steps (Table S5); the Stage 3 goal imagination does not stop at the 15 s goal episode end.

## SDF contact (`contact_detect`, 2026-11-08)

Replaces "env: contact force > 1 N / imagination: learned contact head" by ONE differentiable function used by both (`rl/acp/sdf_contact.py`, `body_spheres.py`):
`d_b = min over the collision capsules / spheres i of b, and over the points of the capsule axis, of (SDF_terrain(point) - r_i)` (EXACT, see "P1-7" below); env / evaluation `c^meas_b = d_b < d_on_b`; imagination `c_b = sigmoid(-(max(d_b, 0) - d_on_b) / tau)`,
d_on = 2 mm, tau = 1 mm (`sdf_tau`). Switch: env cfg `contact_detect` (default "sdf") = agent `contact_detect` (must be equal, checked), "force_head" keeps the old behaviour.

- **Bodies**: the nine contact links; spheres = the collision geoms owned by exactly that link (what the contact sensor matches): convex mesh hulls of the authors' model (knee, pelvis, elbow, the two wrist_yaw hulls) and the 7 foot capsules.
  Offline greedy cover of dense surface samples by near-inscribed spheres (`scripts/acp/fit_body_spheres.py` -> `body_spheres.npz`, 375 spheres). Errors (mm): see the table in the report; the spheres never exceed the hull by more than 0.5 mm.
- **Terrain SDF**: analytic box / cylinder / capsule / sphere / plane of the primitive geoms of the compiled model's `terrain` body (world pose incl. the tile translation; any orientation), bucketed in a 2 m xy grid
  (each geom in the cells its bounding box + the reach of the body touches, K <= 44 geoms per cell, 1-3 usually); a body looks up the cell of its link origin. Terrains made of meshes (datasets without `geoms.yaml`) are not supported (use "force_head").
  Gradient: the SDF gradient (unit normal) via a custom autograd Function (no (batch x spheres x geoms) tensors kept for the backward pass).
- **Thresholds** `D_ON` = 3 mm for all nine bodies (`sdf_contact.D_ON`). Basis: d_b of the simulated robot (checkpoint `/tmp/acp_eval_v1/new10k`, 256 envs x 600 steps, v3_train10, simplified collision) at the steps where the sensor measures |F_z| > 1 N
  (max over the 4 substeps): median -0.6 .. -0.0 mm, p90 <= 0.7 mm (elbow 0.5, pelvis n/a), p95 <= 1.4 mm, p99 3.6-5.1 mm; accuracy-maximal threshold per body 0.5-1.5 mm; with 3 mm the agreement with the measured 1 N flag is 0.984 over all bodies / 0.955-0.992 per body (feet 0.955-0.959,
  F1 0.97). The reference labels of the data to be trained on have the body within <= 1 mm of the surface in contact frames (v3_train11), so d_on is not widened for old data whose labelled contacts hover 1-2 cm (on v3_train10 the best threshold would be 15-23 mm; agreement of the labels with 3 mm: see the report). Statistics on v3_train11: rerun `calibrate_contact_sdf reference` once it exists.
- **Where used**: env `measured_contact_flags` (`track_contact_label`, Stage 3 foot / slippage terms, `ACPSimReader.contact()` -> buffer / eval), imagination `step_wm` (the logits of the contact probabilities are `-(d_b - d_on)/tau` of the predicted state
  -> `reward` / Stage 3 / wmres cut as before), Stage 2 hindsight labels (`contact_measurement`: indicator x 2 > the 1 N threshold of `label_episode`; the contact position is the terrain point closest to the nearest sphere of a body in contact).
  Deviation from the paper (Fz > 1 N) for the labels: the planner must predict the contact the tracker is rewarded for; with the distance criterion label, reward and reference sidecar measure the same thing. The stored `contact` of the buffer is the hard flag.
- **World model**: still predicts the state only; the contact head runs but its bce loss is off (`wm_contact_loss` None -> False in "sdf", log `bce` = 0); `wm_contact_feedback` must be False.
- `StumbleHead` (Stage 3) and the observations keep their inputs; `geo_contact.py` (ray clearance) is unchanged and must be off with "sdf".

### P1-7: exact capsule distance, recalibrated d_on / tau, one-sided soft contact (2026-11-08, Codex review 2)

- **Robot side exact.** The simplified v3 model has only capsules (39) and spheres (6) on the contact links. `body_spheres.segments()` returns them as segments (A, B, r) in the link frame (a sphere = degenerate segment); `SdfContact.distance` evaluates, per terrain primitive of the cell,
  the minimum over t in [0, 1] of the signed distance of the primitive along the segment. The SDF of every convex primitive (box, cylinder, capsule, sphere, plane) is convex, hence so is its restriction to the segment: golden-section search, 24 iterations
  (<= 5 um of position), then the min over the geoms of the cell. Gradient: the unit normal at the minimizer, split between the end points as (1-t), t (envelope theorem; custom autograd Function, nothing stored). The 64-sphere foot cover with its 5.0 / 5.8 mm error is gone: the robot-side error is 0
  (`tests/test_acp_sdf_contact.py` compares against a 4001-point dense scan and checks the gradient). Meshes (non-simplified model only) still use the fitted spheres, as degenerate segments. The same function serves env, imagination and hindsight (`with_point`: the minimizer's terrain point).
- **Warp kernel.** The PyTorch golden-section search was bound by kernel launches (28.4 ms forward / 29.0 ms forward+backward at 2048 envs, 9 bodies, 45 capsules; 32 calls per policy update = ~0.9 s of the 1.86 s iteration). It now runs in NVIDIA Warp (`sdf_contact.py`): one thread per (capsule segment, geom of the cell) pair with the 24 golden-section iterations in registers (the same bracket / probe scheme and SDF formulas as before), a torch `min` over the cell's geoms, and a second per-segment kernel with the analytic unit normal (SDF gradient) of the chosen geom at the minimizer; zero-copy `wp.from_torch`, launched on torch's current stream, no host sync, the launching function is behind `torch.compiler.disable` (a clean graph break inside the compiled `_step_wm`). The envelope-theorem autograd Function is unchanged. Agreement with the replaced torch version (kept as the reference in `tests/test_acp_sdf_warp.py`): d within 1.2e-6 m, x* within ~1e-4 m, normals within 1e-6. Timings (RTX 3090, 2048 envs, Stage 1 launcher config, `/tmp/acp_prof/prof_iter.py`): `SdfContact.distance` 28.4 / 29.0 ms (forward / forward+backward) -> 0.6 / 1.0 ms (the pair kernel itself ~0.6 ms even with 41 geoms in the cell; the rest is torch glue); one training iteration 1862 ms (0.54 it/s) -> 912 ms (1.10 it/s), against 845 ms (1.18 it/s) with a trivial distance (measured while another job shared the GPU).
- **Calibration** (checkpoint `/tmp/acp_eval_v1/new10k`, 256 envs x 500 steps on `v3_final`, simplified collision, exact distance; label = sensor |F_z| > 1 N over the 4 substeps; `calibrate_contact_sdf rollout`). Hard threshold, F1 / precision / recall (acc over all bodies 0.988 / 0.988 / 0.987 / 0.986 at 1 / 1.5 / 2 / 3 mm):

  | body | 1 mm | 2 mm (chosen) | 3 mm (old) |
  |---|---|---|---|
  | knee L/R | .922/.923 (P .90 R .95) | .915/.916 (P .87 R .97) | .904/.907 (P .84 R .98) |
  | foot L/R | .970/.968 (P .97 R .97) | .970/.967 (P .96 R .98) | .966/.963 (P .94 R .99) |
  | hand L/R | .947/.941 | .943/.937 (R .98) | .937/.930 |
  | elbow L/R | .870/.929 | .841/.903 (R .985) | .826/.877 |
  | pelvis (9 positives only) | .64 | .67 | .64 |

  Foot d_b at force steps: q50 -0.6 mm, q95 0.3-0.4 mm, q99 3.3 mm. **d_on = 2 mm for all nine bodies** (was 3): within 0.01 F1 of the 1 mm optimum for every body, recall >= 0.97 (the force sensor also fires on grazing), and still 1 mm above the <= 1 mm gap of the labelled contacts of the repaired
  data (v3_train11). The old 3 mm was chosen with the 5-6 mm foot sphere error in mind; with the exact distance it only adds false positives.
- **Soft contact, one-sided.** `p = sigmoid(-(max(d, 0) - d_on) / tau)`: below the surface (d < 0) p is flat, so the gradient is exactly 0 and the Table 3 contact terms (ref = 1: reward grows with p) no longer pay for penetration (the old `sigmoid((3 mm - d) / 5 mm)` was .65 at the surface and
  reached .95 only 12 mm inside). tau 5 -> 1 mm: p = .88 at the surface, .73 at 1 mm, .5 at d_on, .12 at 4 mm, .02 at 6 mm, so the gradient lives in the 0-6 mm band above the surface and the saturation (the allowed gap of the data, <= 1 mm) is reached; Stage 3 `contact * contact_prev`
  is .77 instead of .38 at the surface. Cost: a body already inside the terrain gets no gradient to lift it (ref = 0) or to sink it (ref = 1); the simulator's penalty depth is sub-mm (q50 -0.6 mm at force steps) and the reset rejects > 1 cm penetration, so this is accepted. Test: `test_soft_contact_is_one_sided`.
  The hard flag / soft value are the same function in env and imagination; the (since removed) `check_mdp_parity --dr` found 0 flag mismatches on the real states.

### P1-10: Stage 2 / 3 env reward follows eta_rew and the planner (2026-11-08)

`ACPRunner.sync_env_reward` (called in `collect` when a planner is attached, i.e. Stages 2 / 3) replaces the raw env reward in `r_sim` (replay, bins return, critic data) by the imagination's objective: each env per-term reward (`reward_manager._termwise_reward_buf`) is multiplied by
`weights(eta_rew)[n] / env weight[n]` (the three reference-contact terms and `track_body_pos_anchor` fade with `1 - eta_rew`; so Stage 3's body-pos term is retired in `r_sim` as in the imagination), the env's own `track_planned_contact` (reference targets; weight 0 in Stage 2, 1.5 in Stage 3) is dropped, and
`1.5 eta_rew * dt * exp(-err / (sigma^2 max(1, m)))` with the PLANNER's targets of the transition's end state (planner on the pre-reset observation, the estimated pelvis pose of that state, `planned_contact_error` like the imagination) is added, on all rows (reference and goal, as `reward_mixed`).
The bins' `G_max` follows the current weights (`sum max(w, 0) dt / (1 - gamma)`). The raw env terms stay in the `rt/*` logs and the env keeps its own reward manager (only `r_sim` changes). Terms with env weight 0 but non-zero imagined weight cannot be recovered (Stage 3 reference-contact terms at eta_rew < 1): the runner prints a warning once;
Stage 3 runs at eta = 1. Open: the env-side reward manager itself still uses the Stage 1 weights and reference targets (diagnostics only); a parity check of the synced `r_sim` against `ACPMDP.reward(planned=..., eta_rew=...)` on real states for eta = 0 / .5 / 1 and reference / goal rows is not written yet (do it before Stage 2).

## Collision geometry default (2026-11-08)

`simplified_collision=True` is the default of the v1 env cfgs (`G1AcpTrackV1EnvCfg`, Stage 2 and Stage 3, Play): analytic capsules / spheres for the upper body, arms, pelvis, wrists and the hand (r = 0.04 sphere), the authors' leg meshes / foot capsules
unchanged (dynamics and mass properties unchanged; `assets/acp_g1.py`). The contact sidecars stay as labelled (on the authors' meshes, not relabelled): the strict check of `AcpReferenceContacts` (sidecar `simplified_collision` flag vs the env) is kept and can be waived
with the explicit `AcpReferenceContactsCfg.allow_label_collision_mismatch`, which v1 sets together with `simplified_collision=True` (prints a note once). The reset penetration rejection and the legal-start table call the simulator's collision detection, so they follow the simulated geometry.
The sphere approximation is built from the compiled model's collision geoms (nothing hard-coded): capsules = exact spheres of their own radius along the axis (zero outward error), spheres = themselves, meshes (knee / foot of the meshes model) = greedy inscribed spheres;
`body_spheres.load(simplified)` refits automatically when the collision geoms of the compiled model changed (signature hash in the cache). `scripts/acp/fit_body_spheres.py [--simplified]` prints the errors.
`spec_fn` of the simplified robot is a named function (`get_acp_g1_spec_simplified`): a `functools.partial` broke the yaml dump of the env cfg at training start.
`robot_limits.py`: joint velocity limits of the ACP G1 defined in the task package (the authors' MJCF has none; values identical to the earlier Unitree-table ones, no numeric change).

## CSI (Stage 1 start curriculum, restored 2026-11-08)

**Success and legal imagination starts (Codex review 2, P1-9).** A reset counts for the CSI of its clip only if it is a *success* (reached phase .99, reached the clip end (`dataset_exhausted`), or ended by the 10 s cap (`time_out`) with no failure and no `illegal_reset_contact`)
or a *failure* (a real failure termination `reset_terminated`); `illegal_reset_contact` (a time-out flag) and manual / external resets are neither and do not move the phase (previously every non-failure reset was a success: a two-step illegal start moved .9 -> .85).
With `csi_success_on_timeout=False` the capped episodes count as failures. **Legal start for every clip.** With `imagine_past_clip_end=False` an imagination start needs f + H <= clip end (H = 32); at the initial phase .9 130 of the 343 clips of `v3_final` (T >= 112 frames, 0 clips with T <= 32) have no such frame. Chosen: a per-clip
cap of the CSI start phase at (T - H - 1) / T (`EpisodeStart.set_phase_cap`, applied to the initial phase, every update and resumed phases; runner prints the number of capped clips): the smallest change (success rule, step and range of the paper unchanged, the four clip-end switches stay False; the cap only binds for clips shorter than H / (1 - .9) = 320 frames)
and the start state of the very first episode is a legal imagination start. Alternatives rejected: starts before the CSI start (needs the actor to be trained on states the CSI does not visit), shorter horizons H_eff (changes the SM horizon / value-bootstrap treatment). Check: after the cap all 343 clips have a start frame <= T - 1 - H (before: 213). The non-finite-state `bad` of the runner is applied after the env's reset and does not reach the CSI (rare, accepted).

**Success needs tracking (2026-11-08, ACP-unspecified).** A success (reached phase .99 / the clip end / the cap without a failure) also needs the pelvis within 0.5 m of the reference pelvis horizontally at the episode end
(`curriculums.CSI_SUCCESS_XY`, the `--xy-end` of `eval_tracker`); a clean end with a larger error is a failure (phase += 0.05). Reason: the v1 terminations have no horizontal term, so an episode that stays alive but
stalls counted as a success and moved the phase back (eval of `model_22000`, v3_train9: vault survive .78, success 0, final horizontal error median 2.1 m). Not a switch.

`reset_sampling="csi"` (`RESET_SAMPLING=csi`; `run_stage1.sh` now defaults to bins): `EpisodeStart(mode="csi")` keeps a phase phi_m per clip (init 0.9); an episode of m starts at phi_m of the clip (the 1 cm legal-start table still applies to CSI);
after the episodes of m that end in a reset call, phi_m -= 0.05 if they reached phase 0.99, else += 0.05, clipped to [0, 0.99] (paper Table S4: initial 0.9, completion 0.99, adjustment 0.05, range [0, 0.99]; checked against the
text of the supplementary methods). ACP-unspecified (marked in code): the updates of one clip within a call are averaged; `csi_success_on_timeout` (True): an episode that ends by the 10 s cap without a failure termination counts as a
success (a clip longer than the cap otherwise could not be started earlier than 10 s before its end: phase 0.99 is unreachable inside the cap). Sidecar: `csi_phase` per clip (restored by clip name), log `csi/phase_{mean,min,max}`, `csi/success_rate`,
`csi_by_motion/<clip>` (csi mode only). Stage 2 (reset bins / uniform) and Stage 3 (phase 0) are unchanged.

## World-model / critic scan (2026-11-08)

**Reverted to `wm_scan="planner"` (default and all three launchers, 2026-11-08, user decision):** the world model context and the critic / termination-head / Vfs inputs use the
17 x 11 planner scan (x +-0.8 m, y +-0.5 m, 0.1 m, centred on the pelvis, 187 values) from the true pelvis pose. Reason: the world model has no root height in its state (`invariant_features`), the scan is
its only terrain input, and the forward-only "wide" scan does not see behind the pelvis. Measured on the labelled contacts of `v3_final` (contact point in the reference pelvis yaw frame, pelvis contacts
excluded, 267,540 body-frames): x q01 / q05 / q50 / q95 / q99 = -0.30 / -0.20 / 0.04 / 0.39 / 0.53 m; covered by 17 x 11: 99.8 % of the contacts, 99.7 % of the contact frames have all supports
inside; by "wide": 62.3 % / 42.0 % (37 % of the contacts are behind the pelvis: vault 55 %, stepping 53 %, gap 55 %, walk 50 %, stairs 44 %; all supports visible in 24 % of the vault frames).
A symmetric extension would be x -0.5 .. 1.5, |y| <= 0.4 at 0.05 m (697 values, 99.3 % / 98.8 %), not implemented.

`wm_scan="wide"` (option): a 31 x 15 forward scan (x 0 .. 1.5 m, y +-0.35 m, 0.05 m, y-major, 465 values; the open-source reproduction's `CRITIC_SCAN`), the same exact ray cast as the actor scan
(`ExactHeightScanner.height_scan(..., offsets, shape)`). The actor (9 x 7) and the planner (17 x 11) keep the paper's scans, `StumbleHead` keeps 17 x 11. The env sensor of the wide scan
(`wm_height_scanner`) is removed (it cast 465 extra rays per env and step for the parity check only). Checkpoints do not load across the two settings.

`scripts/acp/check_mdp_parity.py` (env terms vs the analytic imagination terms on real states) is removed (2026-11-08, user decision); its last run (v3_final, 32 envs x 200 steps, DR): rewards
<= 3e-2 relative, SDF contact flags 0 mismatches, terminations 0, the 17 x 11 scan 0.24 % of the cells > 5 cm off (max 6.4 m, as the tracker scan). Stage 3 keeps `check_stage3.py`.

## Aligned with zita-ch/supermimic_vibed (2026-11-09)

The paper's PPO learns the 5 author vault rollouts from scratch (100 % at 2500 iterations, `/tmp/acp_ppo/REPORT.md`); every SuperMimic variant
stalled at the wall. The launchers (`run_stage{1,2,3}.sh`) now follow supermimic_vibed (commit 0525a23; comparison `/tmp/codex_smdiff/REPORT.md`):
- **Imagined termination**: `--agent.term-source analytic`, `soft_term_tau_dist 0.02` / `soft_term_tau_tilt 0.04` (`mdp.soft_termination` there:
  sigmoid of the env thresholds, pelvis height 0.25 m, tilt 0.8). The learned termination head was used before; on the vault rollouts it gave a
  survival of ~1e-12 for 32-step author-action rollouts that the simulator survives (`/tmp/codex_wmdiag/REPORT.md` C).
- **Collection noise**: `action_noise` 0.1 x the per-joint SuperMimic action scale alpha_j = 0.25 effort / (armature (20 pi)^2) on the joint
  target (`runner.sm_action_scale`; hip pitch / waist yaw 0.055 rad, hip roll / knee 0.035, ankle / waist roll-pitch / arms 0.044), instead of a
  uniform 0.05 rad.
- **World-model windows uniform over the buffer**: no failure-window oversampling and no held-out envs (every env trains the world model).
- `scan_slope_gradient False`: the scan has no slope gradient (constant terrain height), as there.
- **No input clipping**: the world-model / critic / head input normalizers (`rays.ScanFloorRunningNorm`, formerly `ClippedRunningNorm`) keep the
  0.05 m std floor of the scan entries but no longer clip the normalized values to +-8 (every entry, as `clip_all`). There it is a plain
  `RunningNorm`; the clip saturated ~26 % of the 32-step author vault rollouts (max 23 sigma) and zeroed their gradients. Older checkpoints load
  (the `clip_mask` / `clip` buffers are dropped).
- **Contact predicted by the world model** (`contact_detect force_head`, the default of the env cfg and the runner; `wm_contact_loss True`): c^meas is the contact
  sensor's |F_z| > 1 N (env, hindsight labels, evaluation), the imagination's contact is the world model's contact head (BCE against c^meas), as the ACP
  authors' SuperMimic. The "sdf" mode stays available (soft contact sigmoid(-(max(d, 0) - d_on) / 0.001), one-sided, as before).
- World model: the pelvis height is the first context input (both reference implementations keep the root height); buffer 1000 steps; the imagination
  steps the world model on the raw policy action (no +-20 clamp) and keeps the predicted state (no joint projection).
- Regularizers stay as Table 3: `contact_acc` = min(1e4, sum_b |p''_b|), `dof_pos_limits` = min(30, sum_j excess) (the paper gives these caps).
- **Stage 1 reset bins (2026-11-09)**: the start distribution follows `supermimic_vibed` instead of ACP CSI (`run_stage1.sh`: `RESET_SAMPLING=bins`; `RESET_SAMPLING=csi` keeps CSI available).
  One-second bins per clip (`ceil(T / 50)` at 50 Hz, including the upstream zero-width tail for `T % 50 == 1`), global bin probability
  `0.5 / B + 0.4 softmax(-25 G / G_max) + 0.1 EMA / sum(EMA)`, EMA `0.999 old + 0.001 softmax`, initialized uniformly; `G` starts at zero.
  Returns go to the episode's start bin: `0.8 old + 0.2 mean(new returns)` for episodes ending in the same simulator step. The real ACP reward (unchanged, already multiplied by dt)
  is discounted from exponent zero; failure has no bootstrap, clip end / time cap add `gamma^T mean_reward / (1 - gamma)`. `G_max = sum(positive ACP weights) dt / (1 - gamma)`
  (38 with the current Stage 1 weights and gamma 0.99), not the upstream 10 for its different reward. The upstream values `bins_v2=False` / `bins_difficulty_k=0` stay explicit;
  Stage 2 / 3 settings are unchanged. Stage 1 draws a fresh reset after attaching / restoring bins, before warm-up, so the first collected episodes also update their start bins.
  Stage 1 bins truncate on reaching the last valid clip frame (as upstream), rather than waiting one more step for the generic reference-validity timeout.
  Stage 1 bins no longer snap the sampled frame to a nearby legal reference frame (which changed the uniform-within-bin distribution and could cross bin boundaries).
  The reset perturbations, penetration rejection / lift, and ACP rewards are unchanged; root pose / velocity noise still differs from upstream. Verification: `/tmp/codex_bins/REPORT.md`.
- **Terminations: the robot fell or lost the reference** (the ACP authors; the paper gives no thresholds): besides the pelvis height (0.25 m) and tilt
  (0.8) of SuperMimic, `base_xy_too_far` (pelvis horizontally > 0.5 m from the reference pelvis) and `foot_pos_too_far` (a foot > 0.5 m, 3-D, from the
  reference foot) (# ACP-unspecified: 0.5 m, the success radius of the evaluation). Mirrored in the imagination (`mdp.termination` /
  `soft_termination`: sigmoid((e - 0.5) / soft_term_tau_dist)); env and imagination agree on 19200 / 19200 real env-steps. Stage 3: reference envs only.
  A robot that stalls in front of an obstacle now fails (before: it survived 2.7 m behind the reference).
  `contact_violation`: the torso (it carries the head; the G1 has no head link) touches the terrain with > 1 N (``contact_forces`` sensor, max over the
  policy step's substeps); the imagination ends where the exact SDF distance of the torso's collision primitives is below 2 mm, hazard
  sigmoid((0.002 - d) / 0.002) (`sdf_contact.VIOLATION_*`, `get_violation_sdf`); 31997 / 32000 real env-steps agree under heavy action noise.
  Only a contact the reference does not have counts: the term (env and imagination) applies where the reference torso is > 5 cm from the terrain
  (`VIOLATION_REF_CLEARANCE`; get-up / crawl frames lie on it). Full v3_train12, random policy: 127903 / 128000 env-steps agree (env 259, imagination 354 fired).
- Result (2026-11-09, `exp_sm_v7`, 5 author vault rollouts from scratch, 4090): vault 0 % at 2k, **100 % at 4k** (~26 min, ~20 M transitions; the paper's PPO
  needs ~100-120 M / ~45 min), 87.5-100 % at 6k-12k and 100 % at 14k-20k.
- **No self-collision** (as supermimic_vibed, `enabled_self_collisions=False`): the robot's collision geoms are contype 2 / conaffinity 1
  (`assets/acp_g1.get_acp_g1_spec`), so no robot-robot pair collides (906 pairs before) and every robot geom still collides with the terrain.
- Reset penetration check (`terrain_penetration_depth`): kinematics + collision detection as a captured CUDA graph instead of the full `sim.forward()` (the
  constraint solver is not needed for the contacts): 3.13 -> 0.66 ms per call at 2048 envs, depths identical (max difference 0 on 40960 states).
- CSI: a real failure / illegal reset / manual reset at phase >= `csi_success_phase` no longer counts as a success (`counted` = failure, clip end
  or cap without illegal_reset_contact; success needs no failure and the pelvis on the reference).

## Official supermimic is the only path (2026-11-11)

The ACP runner follows the official HybridRobotics/supermimic with no switches for it; the supermimic_vibed-era alternatives are removed (old checkpoints / old option values are not
supported; `rl/supermimic/` keeps its defaults for the vision / parkour tasks, the ACP runner configures the shared classes). What the runner does now:
- World model: semi-implicit gencoord integration (`WorldModel.set_integrator("semi_implicit")`), no output de-normalization, GELU (`act=nn.GELU`), complete k-step windows (`sample_complete_windows`;
  the termination head, when on, keeps its own masked-window batch, uniform over the buffer or clip-balanced by `wm_balance_frac`), loss = official `WorldModelLearner._compute_losses`
  on the gencoord loss blocks (`runner.wm_state_error`: root position, root rotation matrix x sqrt(1/2), joint position, joint velocity; weights 1 / nominal^2 with 0.01 / 0.02 / 0.1 / 3.14; mean over the
  valid steps; no root velocity supervision) + the ACP contact BCE.
- Optimizer: `MuonAdam` (Muon on the hidden Linear weights, fused capturable Adam / AdamW on the rest) for the world model, the actor (AdamW, betas (0.9, 0.95)) and every critic (Vfs too).
- Critic: LN + GELU MLP on the raw features (no input normalization), raw values inside the lambda return, targets clamped at the analytic v_max (sum of the positive reward weights x dt / (1 - gamma));
  actor bootstraps clamped at v_max; one permutation for all critic epochs. Stage 3: v_max per critic from its own objective (`ACPStage3MDP.max_step_reward`: Vm = the tracking terms, Vfs = goal terms +
  regularizers + planned contact).
- Imagined termination: one sigmoid of the largest threshold-normalized error, softness 0.05 (`mdp.soft_termination`), plus the contact-violation hazard.
- Reset sampling: `ReturnBins` (official `ReturnSampler`); a step that times out (cap, clip end) and terminates counts as the time-out (`runner.episode_end_flags`), a non-finite simulator step still
  counts as a failure.
- Context xi: DR-range encoding (`sim.dr_range_xi_stats`), passed through the world-model input normalization.
- Precision: TF32 matmuls (`allow_tf32`), no bf16 autocast (`amp` and `policy_autocast` removed; `networks.AMP_DTYPE` stays None for ACP).
- Schedule: `replay_ratio` 16 (utd = num_envs x 16 / (wm_batches x wm_batch x wm_window)).
- Clip end: the reference stays on the last frame of its clip (`runner.advance_reference_frame`), no masking, no terminal; imagination starts from every buffered state (`f <= clip end`).
  The real collection still truncates at the clip end (Stage 1 `done |= f1 >= clip_end`), recorded by `ReturnBins` as the clip end.
Removed config fields: `utd`, `wm_integrator`, `wm_loss` ("std" loss, `ERR_FLOOR`), `wm_output_denorm`, `wm_gelu`, `wm_complete_windows`, `wm_bryson_mean_loss`, `optimizer`, `critic_ln_gelu`,
`critic_running_norm`, `critic_target_raw_values`, `critic_reuse_permutation`, `value_max`, `value_max_auto`, `soft_term_tau_dist`, `soft_term_tau_tilt`, `reset_sampling`, `csi_*`, `bins_v2`,
`bins_difficulty_k`, `amp`, `policy_autocast`, `xi_normalization` (and `set_xi_stats` / `XI_FLOOR`), `imagine_past_clip_end`, `clip_end_bootstrap`, `clip_end_v2`, `clip_end_exact_last`, the `step_valid` / `step_last` masks, the
CSI phase cap (`EpisodeStart.set_phase_cap`; the env-side CSI mode stays for PPO), the runner's per-episode return bookkeeping (`ep_disc_ret`, `episode_return`). The launchers spell out the new field set.

## Speed (2026-11-08; results unchanged)

RTX 3090, 2048 envs, Stage 1 launcher config, `/tmp/acp_prof/prof_iter.py` (collect / world model / policy + critic per iteration):

| state | collect | world model | policy + critic | iteration |
|---|---|---|---|---|
| start of the day (torch SDF) | 361 ms | 302 ms | 1199 ms | 1862 ms (0.54 it/s) |
| + Warp SDF (see "P1-7") | 318 ms | 276 ms | 318 ms | 912 ms (1.10 it/s) |
| + the three items below | 292 ms | 271 ms | 214 ms | 776 ms (1.29 it/s) |

- **Forward kinematics once per imagined state** (`runner.py`): `_step_wm` returns the FK of the predicted state, which the next step's `_push` / `_observe` / world-model context and the SDF contact use; `mdp.reward` / `soft_termination` / `critic_features` / `link_deviation` take it (`fk=`), the collection and world-model paths are unchanged.
- **Critic features of the imagined states 1..H computed once** (`X1`, with grad when a learned termination head needs them) and reused, detached, by the critic fit and the bootstraps; only state 0 is computed separately.
  Check of both (old vs new bound onto one runner, same seed and restored state; 17 config variants incl. replay, past-clip-end, wmres, no-critic, Stage 3): eager 0.0 difference in every loss / parameter; with compile the actor update is identical and the critic fit differs by fusion noise (parameters <= 1.6e-5 abs).
- **Reset: the lift reuses the depth of the rejection's last `sim.forward()`** (`events/motion_reference.py`): one forward of all worlds less per reset call; bit-identical written states and `reset_penetration_stats` over 120 compared resets.
- **Not adopted: the reset penetration depth from the SDF instead of `sim.forward()`** (would save ~25 ms per reset): agreement with MuJoCo's depth 99.84 % (needed >= 99.9 %). Every disagreement is a state with a capsule axis inside a terrain primitive (5.5 % of the noisy reset samples), where MuJoCo's contact reports ~|sdf| - r, i.e. 2 r less than the penetration, or no contact (e.g. a foot capsule 29.7 mm inside reads 9.7 mm); there the SDF is the more correct one, so the current rejection accepts / under-lifts some deeply embedded starts. Kept as it is (results unchanged); code in `/tmp/rb/`.

## Solver iterations

`sim.mujoco.iterations / ls_iterations` are 10 / 20 (the mjlab scene defaults that were in effect; the authors' MJCF has 50), set in `acp_env_cfg.make_sim()` and spelled out in
the launchers (`--env.sim.mujoco.iterations 10 --env.sim.mujoco.ls-iterations 20`; the former duplicate fields `solver_iterations` / `solver_ls_iterations` are gone).
Measurement: `scripts/acp/measure_solver.py`, report of 2026-11-08.

## Standalone configuration (2026-11-08)

The ACP tasks inherit from no other task: no env / scene / terrain factory of the shadowing / perceptive / locomotion / parkour tasks and no import of their modules; the
only dependencies are the generic base types (mjlab `SceneCfg` / `SimulationCfg`, `InstinctRlEnvCfg`, the `*Cfg` dataclasses, generic mdp functions, `TerrainMotion`,
`FiledTerrainGenerator`). Everything the physics, the scene and the terrain depend on is written out in `tasks/acp/`:

- `scene_cfg.py`: `make_acp_terrain()` (terrain importer + generator, tile size, CoACD options), the contact / height-scan sensors (`make_acp_scene_sensors`), the spec edit
  (sky, ground material, headlight, terrain friction and geom group 0), `AcpSceneCfg(SceneCfg)` (env spacing 4, sensors and robot passed in by the G1 cfg).
- `acp_env_cfg.py`: `make_sim()` (200 Hz, implicitfast, newton, pyramidal cone, sparse Jacobian, 10 / 20 iterations, ccd 128, nconmax 128, njmax 700, maxmatch 128), field
  defaults `decimation = 4`, `episode_length_s` (120 base, 10 Stage 1, 20 Stage 2, 120 Play), `run_name`.
- `__post_init__` only derives (collision model of the robot and the kinematics from `simplified_collision`, dataset path / tile count of the terrain, the Play switches);
  it no longer assigns `decimation`, `episode_length_s`, `sim.*`, so `--env.episode-length-s`, `--env.decimation`, `--env.sim.mujoco.*`, `--env.sim.njmax`, `--env.run-name`,
  `--env.scene.env-spacing` take effect (checked with the training CLI parser).
- `terrain_layout.py`: `TILE_SIZE`, `grid_shape` / `tile_corner` / `grid_bounds`, `CompactFiledTerrainGenerator`.

### Tile size (from our data, `scripts/acp/check_tile_fit.py`)

Measured on `v3_train10` (334 terrains, 345 motions; `v3_train11` has the same terrains) in the `use_input_origin_frame` coordinates (tile centre = env origin = motion origin):
horizontal reach of the robot (root 13.05 x 7.40 m, body origins 13.19 x 7.63, bounding spheres of all geoms 13.38 x 7.79, with reset noise U(-0.2, 0.2) rad on all joints
13.43 x 7.82), terrain primitives (`geoms.yaml`) up to 14.82 x 10.40 m, of which only axis-aligned boxes (ground slabs) exceed 14.5 x 9; every rotated box / cylinder / capsule
lies within 8.29 x 4.53 m. Tile = reach + ~1 m margin, rounded up to 0.5 m: half extents 14.5 x 9.0, **29 x 18 m** (margin 1.07 / 1.18 m; the old 30 x 16 m cut the widest turning
clips at y = +-8 m, 0.13 m beyond the farthest reach). The 10 terrains with ground boxes beyond the tile get them clipped to it (as the STL crop does), the robot never gets within
1 m of that edge. The script re-checks everything after a dataset change (margin, emitted geometry inside the tile, robot AABB of every motion inside its own tile and >= 1 m away
from every other tile rectangle); a dataset with larger reach needs a larger `TILE_SIZE`. Layout: the same near-square grid, border 0; 334 tiles -> 15 x 23 grid, bbox x +-217.5,
y +-207 m, farthest tile centre 283.6 m from the origin (the old tiles: 268.1 m).

### Stage 3 arena layout

`GoalArenaCfg.origin = None`: the arena is placed from the reference grid actually in use (`terrain_layout.grid_bounds(num_tiles, tile size)`, read lazily from the scene's terrain
generator when the spec is built / the tables are made, i.e. after the dataset and the command line have fixed tile count and size): +y of the grid rectangle, `reference_gap` = 10 m
away, centred on x = 0 (levels along x, types along y). For 334 tiles the arena is x [-70, 70], y [217, 319] against the grid y <= 207 (it overlapped 42 reference tiles at the old
fixed origin (-63, 40)); farthest arena corner 326.6 m from the origin. One layout feeds the simulated geometry (`GoalArena.add_to_spec`), the arena tables (spawn / goal),
and the analytic height maps (built from the compiled model). Checks: `GoalArena.check_disjoint` (AABB of the arena vs the grid rectangle and vs every reference primitive already
in the spec, asserted at spec build), and in `scripts/acp/check_stage3.py` the same on the compiled model plus mujoco ray heights at 64 sampled spawn and 64 goal points of
every arena tile against the analytic arena surface. Stage 3 Play has one reference tile, so its arena sits right beside that tile.

## Phases

1. **Data**: contact labeler -> per-clip `contact_flag (T,9)`, `contact_pos_w (T,9,3)` next to each motion npz.
2. **Env** `Instinct-ACP-Track-G1-v1` (mjlab, dict managers): obs / rewards / reset-bin episode starts (the paper's CSI is available via `RESET_SAMPLING=csi`) / DR above; reference contact targets command.
3. **Stage 1 runner**: `rl/acp/runner.py` on top of `rl/supermimic` (world model, buffer, critic reused).
4. **Planner + Stage 2**: `rl/acp/planner.py`, `hindsight.py`, eta schedules, planner-in-the-loop.
5. **Stage 3**: procedural goal-reaching terrains (platforms, stones, gaps, beams, pallets, ladders), goal rewards, Vfs, fine-tune buffer.
6. **Eval / sim2sim**: success / contact error per terrain group; run in the playground MJCF (`~/Datasets/acp/playground`).

## Stage 3 (implemented: `Instinct-ACP-Stage3-G1-v1`)

Mixed env, half reference envs (Stage 2 setup, initial phase 0, mimic terms masked to them) and half goal-reaching envs on the procedural arena
of `tasks/acp/goal_terrains.py` (17 terrain types x 10 levels, tile = (type, level); the arena is added to the `terrain` body of the scene, beside
the motion-matched tiles, placed from their actual grid, see Standalone configuration). Goal command, curriculum, resets, goal rewards / terminations: `tasks/acp/mdp/goal.py`; env cfg:
`tasks/acp/acp_stage3_env_cfg.py`; runner: `rl/acp/stage3_runner.py` (`ACPStage3Runner(ACPRunner)`), analytic goal MDP and two-region heightmap:
`rl/acp/stage3_mdp.py`; hyperparameters `rl/acp/stage3_config.py`. Checks: `scripts/acp/check_goal_terrains.py` (no simulator),
`scripts/acp/check_stage3.py` (env, heightmap vs ray casts, reward parity, curriculum).

Runner = Stage 2 + (a) dual critic: Vm (reference, as before) and Vfs (goal command, d_g, dz, elapsed time / 15 s, planner contacts, scan, xi),
Vfs fixed lr 1e-3, trained but excluded from the actor loss for the first 500 iterations; (b) goal-env imagination `_imagine_goal`: fixed goal,
planner targets (frozen), analytic goal rewards with sigmoid-relaxed indicator gates, soft fall termination; (c) separate return normalization
(running std of each regime's discounted return); (d) goal-reaching planner replay: admitted if within 0.5 m (horizontal) at |dz| <= 0.1 m,
truncated at 0.2 m, partition keyed by (terrain type, level) with eviction of the highest work / distance episode; batches proportional to the
stored counts; (e) the world model trains on all windows, the termination head on reference windows only.

Not implemented / simplified (see the `ACP-unspecified` comments): `stumbling` is evaluated on real transitions only (the world model has no force
output); no learned termination head for goal envs (analytic soft fall only); the first goal-env transition of an imagined rollout assumes the
previous-step contact equals the current one (slippage); ladder rungs / rails are 4 cm boxes, thinner than the 5 cm heightmap resolution can resolve
exactly; single process (no multi-GPU).

## Which machine does what

- `white3090` (this checkout, RTX 3090) and the 4090 boxes (`ssh 4090-white` / `4090-black`): training, dataset builds, `acp-webviewer` (:8188).
- `xiangyk-173` (`ssh xiangyk-173`, host `skd`, RTX 3060 12 GB): runs the policy (`~/acp-viewer/run_viewer.sh`, viser :8192), records the video and
  sends it to Feishu. Not a full-power card — low fps / poor quality, so check the motion here and render final clips on the 3090 / 4090.
