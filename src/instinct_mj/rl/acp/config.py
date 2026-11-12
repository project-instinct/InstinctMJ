"""Runner config of the SuperMimic-style ACP trainer (:class:`~instinct_mj.rl.acp.runner.ACPRunner`).

The loop (collect -> k-step world-model fit -> H-step BPTT policy + GAE critic) is the one of the official HybridRobotics/supermimic and is the ONLY
path of the runner (no switches for it): semi-implicit gencoord world model without output de-normalization (GELU, Bryson loss on the gencoord loss blocks, complete
windows), Muon + fused Adam, the LN + GELU critic on raw features with targets clamped at the analytic v_max, the normalized single-sigmoid soft termination, return-based
reset sampling (``ReturnSampler``), DR-range encoding of the context, a replay ratio of ``replay_ratio`` and a reference that stays on the last frame of its clip.
The fields below are the orthogonal ACP research options; the ``contact_source`` / ``eta_*`` fields are the Stage 2 hooks (paper Eq. 3, Table S5).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(kw_only=True)
class ACPRunnerCfg:
    """Hyperparameters of :class:`ACPRunner`, mixed into the task's agent cfg (the runner receives its flat ``to_dict()``).
    ``seed``, ``max_iterations``, ``save_interval`` and ``log_interval`` are the ones of ``InstinctRlOnPolicyRunnerCfg``."""

    # ---- simulation / collection (collect -> k-step world-model fit -> H-step BPTT policy + GAE critic) ----
    buffer_steps: int = 1000
    """Replay buffer length in vectorized env steps, (T, N) layout (default 1000)."""
    replay_ratio: float = 16.0
    """Official supermimic ``Ratio``: the training rate follows the collected data, updates per vectorized simulator step
    utd = num_envs * replay_ratio / (wm_batches * wm_batch * wm_window) (16: utd 0.4 at 4096 envs, 0.2 at 2048; default 16)."""
    warmup_steps: int = 64
    """Simulator steps collected before any update (default 64)."""
    hist: int = 5
    """Proprio observation history length; task: ``PROPRIO_HISTORY_LENGTH`` (default 5)."""

    # ---- world model ----
    wm_window: int = 8
    """Window length k of the multi-step world-model fit (default 8)."""
    wm_batches: int = 10
    """World-model gradient steps per update (default 10)."""
    wm_batch: int = 2048
    """Windows per world-model batch (default 2048)."""
    wm_balance_frac: float = 0.0
    """Termination head only (``term_source`` learned / union): fraction of the head's window batch drawn clip-balanced (every clip equally likely) instead of uniformly over the buffer
    (default 0). Track only (the Stage 3 buffer has no reference clip for the goal envs)."""
    term_windows: int = 512
    """Termination head only: windows (of its own ``wm_batch`` window batch) the head trains on per step; the feature computation runs the FK (default 512)."""
    wm_only_iters: int = 500
    """World-model-only iterations at the start, no policy update (default 500)."""
    wm_physics_inputs: str = ""
    """Comma-separated extra inputs of the world model, appended after the ``(scan, previous action)`` context (default "":
    the plain world model, unchanged). ``torque``: the PD torque the actuators deliver at the current state (23, the env's PD law with the
    action delay, effort limits and encoder bias); ``clearance`` / ``clearance_vel``: height above the terrain and world vertical velocity
    of the nine contact links (9 each). All are known at training time and are recomputed from the imagined state, so this is an
    input-representation change only: same network, same losses, same training target. See ``research/ds_proposal.md``."""
    wm_ref_contacts: bool = False
    """WM-only contact command input (36: the 27 target values and their validity): the command the tracker's observation actually carries at that step (Stage 1: the reference targets with the observation noise; Stage 2: the per-env reference / planner mix; Stage 3: the planner's), stored in the buffer at collection and rebuilt from the observation in the imagination. Default off."""
    wm_contact_feedback: bool = False
    """WM-only contact probability feedback: measured at the start, then autoregressive with full gradients. Default off."""
    profile_phases: bool = False
    """Log synchronized collection / WM(+termination head) / policy+critic wall times. Default off."""
    lazy_exact_heightmap: bool = True
    """With exact_scan, build the auxiliary bilinear map only if ground_height is used. Exact rays stay unchanged. Default on (Stage 3 sets it to False)."""

    # ---- policy / critic ----
    horizon: int = 32
    """Imagination horizon H of the BPTT policy update (default 32)."""
    pi_batch: int = 2048
    """Imagination starts per policy update (default 1024)."""
    scan_slope_gradient: bool = False
    """The exact height scan has the slope gradient of the terrain: d(scan)/d(xy, yaw) from the hit normal on the faces (dz/dx = -n_x / n_z, zero for near-vertical faces and misses); default False: the terrain height
    is a constant for the gradient (0 w.r.t. the horizontal position and heading)."""
    soft_term_softness: float = 0.05
    """Softness of the imagined termination: one sigmoid of the largest threshold-normalized error, sigmoid((max_i e_i / theta_i - 1) / softness) (official
    ``WorldModelEnv._compute_term_prob``; the contact-violation hazard is kept; default 0.05)."""
    term_source: str = "analytic"
    """Termination of the imagined rollouts: "analytic" (default: the env's thresholds as one normalized sigmoid, ``soft_term_softness``), "learned" (SuperMimic paper, Eq. 1-2: d = sigmoid of a termination head
    ``dω(ŝ_t, g_t)`` trained with BCE on the autoregressive k-step windows against the REAL termination labels; the analytic term is not used) or "union" (1 - (1 - d_analytic)(1 - d_learned)). The head
    (``term_head.py``) is built and trained only for "learned" / "union". Goal-reaching starts of Stage 3 keep their analytic fall."""
    start_balance_frac: float = 0.0
    """Fraction of the imagination starts drawn clip-balanced (as ``start_sampling="per_motion"``), the rest uniformly (default 0)."""
    start_sampling: str = "uniform"
    """Imagination starts: "uniform" over the buffered states (default, official: every buffered state with a frame ``f <= clip end``) or "per_motion": every clip with an eligible
    state has the same probability, then a uniform eligible state of that clip (long successful episodes no longer dominate). Track only."""
    gamma: float = 0.99
    """Discount factor (default 0.99)."""
    lam: float = 0.95
    """GAE lambda of the critic targets (default 0.95)."""
    action_noise: float = 0.1
    """Gaussian exploration noise of the real-env collection, as zita-ch/supermimic_vibed: std ``action_noise`` x alpha_j [rad] on the joint target, alpha_j the per-joint
    SuperMimic action scale 0.25 * effort / (armature * (20 pi)^2) (``runner.sm_action_scale``: 0.35-0.55 rad; 0.1 -> 0.035-0.055 rad) (default 0.1)."""
    contact_detect: str = "force_head"
    """Contact decision of the nine contact bodies, shared by the environment and the imagination (must equal the env cfg's ``contact_detect``). "force_head" (default, the ACP
    authors' SuperMimic): the env's |F_z| > 1 N and the world model's learned contact head (BCE against it, ``wm_contact_loss``). "sdf": the signed
    distance d_b of the (exact capsule) collision geometry of each body to the terrain primitives (``sdf_contact.py``); the env decides ``d_b < d_on_b``, the imagination computes d_b from the
    predicted state by forward kinematics and uses ``sigmoid(-(max(d_b, 0) - d_on_b) / sdf_tau)``; the Stage 2 hindsight labels use the same decision."""
    wm_scan: str = "planner"
    """Height scan of the world model and the critics (privileged; the actor and the planner keep the paper's 9 x 7 / 17 x 11): "planner" (default) = the 17 x 11 scan of the planner
    (x +-0.8 m, y +-0.5 m, 0.1 m, centred on the pelvis, true pose; 187 values), "wide" = 31 x 15 points at 0.05 m, 0 .. 1.5 m ahead of the pelvis and +-0.35 m (the open-source
    reproduction's CRITIC_SCAN; 465 values). "wide" sees nothing behind the pelvis, where 37 % of the labelled contacts of v3_final are (see PLAN.md "World-model / critic scan")."""
    sdf_tau: float = 0.001
    """Temperature [m] of the imagined soft contact in the "sdf" mode (0.001: p = .88 at the surface, .5 at d_on = 2 mm; the soft contact is one-sided, flat below the surface)."""
    wm_contact_loss: bool | None = None
    """Train the world model's contact head (BCE against the stored c^meas, log ``wm/contact_bce``). None (default): on for contact_detect "force_head", off for "sdf" (the world model
    predicts states only; the contact is computed from the predicted state)."""
    geo_contact: bool = False
    """Imagination: the contact probabilities of the nine links in the rewards (track_contact_label, ...) come from the geometry of the robot state and the terrain
    (``geo_contact.py``: exact ray-cast clearance of the link surface points, ``sigmoid(-(clearance - d0)/tau)``) instead of the world model's contact head, which
    keeps training as an auxiliary output. Agreement with the measured 1 N contact label on real states: accuracy 0.97, F1 0.8-0.95 per link (SONNET_NOTES.md)."""
    geo_contact_d0: float = 0.008
    """Clearance [m] at which the geometric contact probability is 0.5."""
    geo_contact_tau: float = 0.008
    """Temperature [m] of the geometric contact probability."""
    clip_aliases: str = ""
    """Path of a YAML file {old clip name: new clip name} used when the bins of a sidecar are restored by clip name (a renamed clip keeps its bins; default: none)."""
    imagine_ref_replay_eps: float = 0.0
    """Demonstration replay in the imagination (DiffMimic, threshold version; default 0 = off): when the largest height deviation of a contact link from the
    reference exceeds this [m] after an imagined step, the NEXT step starts from the reference state of that frame (no gradient through the replacement) instead
    of the deviated state. The reward of the deviating step is computed on the deviated state; terminations, the termination head and the critic features use the
    states that are actually continued from. Track only."""
    imagine_link_hazard_eps: float = 0.0
    """Soft link-deviation hazard in the imagination (default 0 = off): the continuation of every imagined step is combined with
    h = sigmoid((link_deviation - eps) / tau) of the predicted (not replaced) state, d = 1 - (1 - d)(1 - h), gradient kept. Leaving the reference by more than
    eps [m] (the ``real_ref_inject_eps`` of the real collection) counts as a failure in the imagination too, consistent with the real injection. Track only."""
    imagine_link_hazard_tau: float = 0.02
    """Temperature [m] of the soft link-deviation hazard."""
    real_ref_inject_eps: float = 0.0
    """Reference injection in the real collection (default 0 = off): when the largest height deviation of a contact link from the reference of a real env exceeds
    this [m], the env is NOT terminated but moved (teleported) to the reference state of the current frame; the transition is stored as the end of an episode
    (``done``, not ``terminated``) and the history restarts; the discounted return of the episode (for its start bin) keeps accumulating until the episode really ends. Track only."""
    abpt_one_step_weight: float = 0.0
    """Weight of the one-step (0-step critic) objective in the actor loss: loss = -((1 - w) J_H + w (r_0 + gamma V(s_1))) (ABPT-style value gradient: the critic
    that learned the real reward values gives the gradient for what the H-step return cannot differentiate; default 0)."""
    bptt_stopgrad_terms: str = ""
    """Comma-separated reward terms whose VALUE stays in the imagined return (and in the critic targets) but whose gradient w.r.t. the actions is cut
    (stop-gradient), e.g. "track_contact_label,contact_acc,joint_torque" (default "": the Stage 1 run). The gradient decomposition of the stuck clips showed
    these terms carrying as much gradient norm as the whole kinematic tracking sum, pointing elsewhere (cosine 0.1-0.2 to it). Track only."""
    pi_lr_scale: float = 1.0
    """Multiplier of the actor learning rate only (``lr`` is shared by the world model, critic and termination head; default 1)."""
    policy_updates_per_iter: int = 1
    """Policy (BPTT) updates per iteration, each on fresh imagination starts (default 1)."""
    imagine_action_noise: float = 0.0
    """Std of the Gaussian noise added to the actions inside the imagination of the policy update (default 0: the Stage 1 run). > 0 trains the actor on
    the expectation over perturbed rollouts (randomized smoothing): the gradient through the contact-rich dynamics (grasps, foot placement on edges) is
    then an average over nearby contact outcomes instead of the derivative at one chaotic point. Track only."""
    critic_epochs: int = 5
    """Critic fitting epochs per policy update (default 5)."""
    critic_minibatches: int = 4
    """Critic minibatches per epoch (default 4)."""
    critic_hidden: str = "256,128"
    """Hidden widths of the critic MLP(s) (Linear -> LayerNorm -> GELU, raw input features, official ``build_mlp``), comma separated (default "256,128": SuperMimic Table A.4).
    Stage 3 uses it for both Vm and Vfs."""
    no_critic: bool = False
    """Ablation: no critic, i.e. no bootstrap value term in the return and no critic fit (default False)."""
    # ---- wmres: contact-triggered truncation of the imagination BPTT (research/glm_proposal.md, AHAC-style; default off) ----
    wmres_cut: bool = False
    """Truncate a start's imagined return at its first world-model contact event (a link's contact probability crossing ``wmres_cut_theta`` between
    consecutive steps, optionally requiring a move of ``wmres_cut_flux``): the BPTT gradient does not flow through the event transition; the return continues
    with the critic value of the last pre-event state, and the critic fit / GAE targets are NOT truncated (default False: the untruncated H-step objective)."""
    wmres_cut_theta: float = 0.5
    """Contact-probability threshold of the event detector (default 0.5)."""
    wmres_cut_flux: float = 0.0
    """Minimal probability change of the crossing link for an event (anti-dither; the analog of AHAC's stiffness threshold C; default 0: any crossing)."""
    wmres_cut_alpha: float = 1.0
    """Weight of the event step's reward in the truncated return: r_t* is scaled by 1 - alpha (default 1: the event transition is handed to the critic entirely)."""
    wmres_cut_warmup_iters: int = 500
    """Iterations before the truncation can fire (the contact head is noise before it is trained; default 500 = ``wm_only_iters``)."""
    wmres_cut_max_frac: float = 0.9
    """Safety valve: if the cut steps exceed this fraction of the batch's (H x B) steps, the batch falls back to the untruncated objective (default 0.9)."""
    wmres_cap_clips: str = ""
    """Per-clip horizon cap of the imagination BPTT: "prefix=H[,prefix=H...]" -- starts whose clip name starts with ``prefix`` trust only their first H
    transitions and continue the return with the critic value of the state at H (the same truncation as ``wmres_cut``, at the fixed step H; default "":
    no cap). Evidence (research/codex_diagnosis.md): the H32 BPTT direction of the ladder clips against the true physics gradient collapses (cosine .068)
    while H8 / H16 stay usable (.695 / .493), and the world model is only supervised on 16-step windows -- e.g. "ladder=16" keeps the policy inside the
    supervised horizon on the ladder and hands the rest to the critic. The critic fit / GAE targets are not capped."""

    # ---- schedule ----
    lr: float = 2e-4
    """Learning rate, CONSTANT (the official supermimic has no schedule; default 2e-4). The Adam(W) part of every :class:`MuonAdam` holds it as a device tensor (fused, capturable), the Muon part
    as a float, as the official ``build_optimizers``: a CUDA graph applies the same learning rate as the eager update."""

    # ---- performance (precision: TF32 matmuls, no autocast) ----
    compile: bool = True
    """Use ``torch.compile`` on the world-model functions (default True)."""
    compile_policy: bool = True
    """``torch.compile`` the imagination step, the reward / termination and the critic-feature functions (default True)."""
    planner_imagination_subset: bool = True
    """Only infer observation-selected planner rows when eta_rew is zero; pad small batches to preserve numerical agreement."""
    checkpoint_imagination: bool = False
    """Gradient checkpointing of every imagination step (recompute in backward): less memory, more compute (default False)."""
    heightmap_resolution: float = 0.05
    """Resolution [m] of the global terrain heightmap used by the analytic height scan (default 0.05)."""
    exact_scan: bool = True
    """The imagined height scans (actor / planner scan, world-model context, critic) are exact mujoco_warp ray casts against the terrain geoms with the
    ray origins of the env's ray-cast sensor; the gradient is the exact derivative (1 w.r.t. the root height, 0 w.r.t. its position and heading). False: the 0.05 m height map only, which
    misses thin structures (ladder rungs) and overhangs (default True)."""
    heightmap_cache_dir: str = "~/.cache/instinct_mj/supermimic_heightmaps"
    """Directory caching the terrain heightmaps (default "~/.cache/instinct_mj/supermimic_heightmaps")."""

    # ---- Stage 2 hooks (default off): contact commands of the imagined policy observation / reward ----
    contact_source: str = "reference"
    """"reference": the contact targets of the policy observation are the reference next-onset targets (Stage 1, eta = 0).
    "mixed": per env and step (real collection AND imagination) the whole contact command of the observation is the planner's with
    probability ``eta_obs(u)`` and the reference's otherwise, and the reward weights follow ``eta_rew(u)`` (Stage 2, paper Eq. 3).
    "planner": ``eta_obs = eta_rew = 1`` (Stage 3 style). The planner is frozen in imagination (default "reference")."""
    eta_obs_start_iter: int = 0
    """u_obs of Eq. (3): iteration at which the observation schedule leaves 0 (ACP-unspecified; default 0, Stage 2 sets it)."""
    eta_rew_start_iter: int = 0
    """u_rew of Eq. (3): iteration at which the reward schedule leaves 0; must be > ``eta_obs_start_iter`` for "mixed" (ACP-unspecified;
    default 0, Stage 2 sets it)."""
    eta_ann_iters: int = 1
    """N_ann of Eq. (3): shared transition duration in iterations (ACP-unspecified; default 1, Stage 2 sets it)."""
    planner_checkpoint: str = ""
    """Initial weights (``AttentionContactPlanner`` state dict) of the planner when the runner has one ("" = random init, default "")."""
    train_planner: bool = False
    """Record the successful episodes, label them in hindsight and train the planner once per iteration (``stage2.py``); the planner is
    also built (and trained) for ``contact_source`` != "reference" (Stage 2 sets it; Stage 1 default False)."""
    planner_lr: float = 1e-4
    """Adam learning rate of the planner (Table S5; default 1e-4)."""
    planner_updates_per_iter: int = 10
    """Planner optimizer steps per iteration (Table S5; default 10)."""
    planner_batch: int = 4096
    """Samples per planner optimizer step (Table S5; default 4096)."""
    planner_sub_batches: int = 8
    """Sub-batches a planner step is accumulated over (Table S5; default 8)."""
    planner_min_entries: int = 1000
    """Planner training starts when the replay holds this many samples (Table S5; default 1000)."""
    planner_replay_capacity: int = 200_000
    """Samples of the reference partition of the planner replay (ACP-unspecified; default 200000)."""
    planner_ref_tile_capacity: int = 5000
    """ACP-unspecified: samples per (terrain type, tile) ring of the reference partition of the planner replay (~ ``planner_replay_capacity`` / 40 tiles)."""
    planner_success_only: bool = True
    """Keep only successful episodes for the planner (paper filter). False: label every episode, a test switch for runs whose tracker
    does not finish clips yet (default True)."""
    # ---- official HybridRobotics/supermimic: the remaining parameters ----
    wm_nominal: str = "0.01,0.02,0.1,3.14"
    """Nominal errors of the Bryson loss on the official gencoord loss blocks: root position [m], root rotation [rad] (rotation matrix x sqrt(1/2)), joint position [rad], joint
    velocity [rad/s] (official ``WorldModelLearnerCfg``: 0.01 / 0.02 / 0.1 / 3.14; weights 1 / nominal^2; the root velocities are not scored)."""
    return_horizon: int = 20
    """Official ``ReturnSampler``: env steps per update of the per-bin values V_b (the per-step discounted returns of every visited bin, bootstrapped from V)."""
    return_memory: int = 256
    """``ReturnSampler``: most samples a bin's V remembers."""
    return_steepness: float = 25.0
    """``ReturnSampler``: softmax gain on V_b / v_max."""
    return_explore: float = 0.5
    """``ReturnSampler``: uniform share of the resets."""
    return_keep: float = 0.1
    """``ReturnSampler``: share on the bins in proportion to their summed squared focus."""
    cuda_graph_wm: bool = True
    """Capture the world-model gradient step (rollout, loss, backward, clip, optimizer step) as one CUDA graph after a few eager steps (official ``CapturedUpdate``).
    With ``compile`` the rollout and the loss are compiled in the default mode first and then captured (official: compile, then capture; never "reduce-overhead" inside the
    capture). Needs no contact-input world model and ``term_source`` analytic."""
    cuda_graph_policy: bool = True
    """Capture the policy update (H-step imagination, return, BPTT backward, gradient clipping, actor optimizer step, critic targets and the critic fit) as one
    CUDA graph after a few eager updates (:class:`CapturedUpdate`), replacing ~2000 kernel launches per update by one replay. The sampling of the imagination starts stays outside
    (its outputs are the graph inputs). The captured update is numerically the eager one, except that a non-finite actor gradient norm skips the actor step on the device
    (restoring parameters and optimizer state) instead of by a host check. Stage 1 only: needs ``cuda_graph_wm``, ``contact_source`` reference without a planner,
    ``term_source`` analytic, no wmres / ``imagine_ref_replay_eps`` / ``abpt_one_step_weight`` / ``imagine_link_hazard_eps`` /
    ``checkpoint_imagination``. ``step_wm`` is compiled in the default mode (the outer graph replaces "reduce-overhead"). Re-recorded when the learning rate changes by more than 0.5 %."""
    norm_after_collect: bool = False
    """Update the policy / WM normalizer statistics after the collection of an iteration from the queued real data (official ``_flush_normalization``)."""
    skip_env_critic_obs: bool = True
    """Training: the env's "critic" observation group (reference kinematics / contact positions, never read by this runner: the critic features are analytic) is not
    computed, a zero tensor of its shape is returned (~20 % of a collection step at 2048 envs; no effect on learning)."""
    return_resume_stats: bool = True
    """Save / restore the ReturnBins sample counts and effort with the sidecar (the probabilities are rebuilt without a new effort term)."""
    restart_iteration: bool = False
    """Start the iteration count (lr schedule, eta schedules) at 0 after loading a checkpoint: a new stage continues from the weights of the
    previous one (default False)."""

    # ---- export ----
    onnx_deploy_joint_order: bool = False
    """``export_as_onnx``: write a self-contained actor.onnx (raw observation in, actions out) with the joint blocks in the deploy joint order
    ``DEPLOY_POLICY_JOINT_ORDER`` instead of the env's joint order (default False)."""

    def to_dict(self) -> dict[str, Any]:
        """Plain dict of all fields (what the runner receives)."""
        return asdict(self)
