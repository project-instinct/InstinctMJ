"""Runner config of the SuperMimic-style ACP trainer (:class:`~instinct_mj.rl.acp.runner.ACPRunner`).

The hyperparameters of the loop (collect -> k-step world-model fit -> H-step BPTT policy + GAE critic) are those of
``SuperMimicRunnerCfg`` (no vision: the scan is analytic); the ``contact_source`` / ``eta_*`` fields are the Stage 2 hooks
(paper Eq. 3, Table S5).
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
    utd: float = 0.4
    """Updates per vectorized simulator step (default 0.4)."""
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
    wm_failure_frac: float = 0.25
    """Fraction of each world-model batch drawn from windows that start within ``wm_failure_horizon`` steps before a real
    termination; the failure head importance-weights them (default 0.25)."""
    wm_failure_horizon: int = 16
    """Steps before a real termination in which a window counts as a failure window (default 16)."""
    wm_balance_frac: float = 0.0
    """Fraction of every world-model window batch drawn clip-balanced (every clip equally likely) instead of uniformly over the buffer, whose rows are ~70 %
    the long episodes of the clips that are already solved (default 0: the Stage 1 run). Track only (the Stage 3 buffer has no reference clip for the goal envs)."""
    wm_holdout_envs: float = 0.1
    """Fraction of envs never used for world-model / head training, only for held-out calibration diagnostics (default 0.1)."""
    wm_only_iters: int = 500
    """World-model-only iterations at the start, no policy update (default 500)."""

    # ---- failure head ----
    # ---- failure head ----
    fail_head: bool = True
    """Use the learned failure ("continue") head in imagination, ``d = 1 - (1 - d_analytic) (1 - p_fail)``; the head is trained
    either way and the gradient through ``p_fail`` is kept (default True)."""
    fail_head_warmup_iters: int = 300
    """Iterations after the head was created before it affects the policy; it is trained from the start (default 300)."""
    fail_windows: int = 512
    """Windows (of ``wm_batch``) that train the head, the feature computation runs the FK (default 512)."""
    fail_clip: float = 0.95
    """Upper clip of ``p_fail`` in imagination (default 0.95)."""
    fail_pos_weight: float = 1.0
    """BCE positive weight; 1 because the failure oversampling is importance-weighted to keep the head calibrated (default 1.0)."""

    # ---- policy / critic ----
    horizon: int = 32
    """Imagination horizon H of the BPTT policy update (default 32)."""
    pi_batch: int = 1024
    """Imagination starts per policy update (default 1024)."""
    imagine_past_clip_end: bool = False
    """False (default, the Stage 1 run): imagination starts only at buffered states with ``f + H <= clip end``. True: starts with at least one
    frame left in the clip are allowed; the imagined steps beyond the clip end are masked (reward 0, no critic loss) and the last step inside the
    clip is terminal with zero continuation (the env time-outs there, ``dataset_exhausted``). Track only."""
    clip_end_bootstrap: bool = False
    """With ``imagine_past_clip_end``: the last imagined step inside the clip is a time-out, not a termination (SuperMimic: time-outs bootstrap): its return
    bootstraps with the critic value of the state at the clip end instead of zero (default False: zero continuation, the old runs). Track only."""
    reset_sampling: str = "csi"
    """Start of an episode: "csi" (default: the clip by ``clip_sampling``, the start phase by the CSI curriculum) or "bins" (SuperMimic App. D.1 as it is: 1 s bins of
    all clips, running return per bin attributed to the start bin of the episode, p_b = 0.5 uniform + 0.4 softmax(-25 G_b / G_max) + 0.1 past softmax weights, start
    frame uniform in the bin; the CSI phase updates are off; incompatible with ``clip_sampling="return"`` and ``csi_strict_link_eps``). The reset noise of the env
    (Table S2) is kept, the upstream reset noise is not added. Track only."""
    bins_v2: bool = False
    """``reset_sampling="bins"``: bins over the valid START frames of a clip (frames 0 .. T-2), a tail bin of less than 25 frames merged into the previous one (no zero-width and no very
    short bins); the return of an episode that ends at the clip end gets no mean-reward bootstrap (the finite task end), only the episodes that end by the episode cap keep the upstream
    bootstrap, and episodes shorter than 10 steps do not update G_b. The stored G_b / bin weights of a sidecar without v2 bins are not restored (default False: the old bins)."""
    bins_difficulty_k: float = 0.0
    """``reset_sampling="bins"``: the uniform half of the bin distribution is weighted by the clip difficulty d_c = 1 + k f_c (f_c: running failure rate of the clip, EMA 0.02 per update, initialized
    0.5, of its episodes that ended by a non-time-out termination), normalized to 0.5 again; the softmax and history parts are unchanged (default 0: the plain uniform half)."""
    clip_end_exact_last: bool = False
    """The last valid imagined transition of a start is found from the absolute frames (the transition into the clip end), so a clip end exactly at imagined step H is the last step too (default False: that step never
    was "last" and kept the critic bootstrap of the state after it)."""
    failure_window_fix: bool = False
    """World-model failure windows (``wm_failure_frac``): a window counts as a failure window only if the FIRST episode end at or after its start within the window is a real termination (default False: any
    termination in the next rows counts, also one of the NEXT episode, which the loss mask removes). The log has the sampled and the valid failure windows."""
    scan_slope_gradient: bool = False
    """The exact height scan has the slope gradient of the terrain: d(scan)/d(xy, yaw) from the hit normal on the faces (dz/dx = -n_x / n_z, zero for near-vertical faces and misses); default False: the terrain height
    is a constant for the gradient (0 w.r.t. the horizontal position and heading)."""
    soft_term_tau_dist: float = 0.02
    """Temperature [m] of the sigmoid of the soft distance terminations (height, horizontal position, contact, link) in the imagination: the per-step hazard is sigmoid((error - threshold) / tau) (default 0.02)."""
    soft_term_tau_tilt: float = 0.04
    """Temperature of the soft tilt termination (default 0.04)."""
    imagine_clamp_joints: bool = True
    """True (default, the old runs): the imagined joint positions are projected onto the hard joint range and the velocities pushing further out are zeroed (``clamp_joints``), which the
    world-model training rollout does not do and which removes the joint-limit reward and its gradient in the imagination. False: the same transition as the world-model rollout,
    the joint-limit reward sees the unclamped q; only a wide NaN guard (hard range +- 0.5 rad) remains."""
    clip_end_v2: bool = False
    """The clip end of the imagination as in the real env: the transition into the frame T (one after the last valid frame) is the last real one (``dataset_exhausted`` fires there, the reference
    stays on its last frame) and a start at the last valid frame is allowed (False: the imagined episode ends one transition early); the critic normalizer only sees the states reached
    through valid transitions (default False)."""
    clip_sampling: str = "uniform"
    """Clip choice at the reset of an env: "uniform" (the motion weights of the dataset, default) or "return" (SuperMimic App. D.1: a running estimate G_c of the
    discounted return of the episodes of clip c, p_c = 0.5 u_c + 0.4 softmax(-25 G_c / G_max) + 0.1 h_c with u_c proportional to the clip duration and h_c the
    EMA (0.999) of the past softmax weights; the CSI still chooses the start phase within the clip). Track only."""
    csi_strict_link_eps: float = 0.0
    """> 0: an episode counts as a CSI success only if it also never had a contact link further than this [m] from the reference (``mdp.link_deviation``, the
    quantity of the injection threshold); bookkeeping only, nothing terminates (default 0 = off). Track only."""
    start_balance_frac: float = 0.0
    """Fraction of the imagination starts drawn clip-balanced (as ``start_sampling="per_motion"``), the rest uniformly (default 0)."""
    start_sampling: str = "uniform"
    """Imagination starts: "uniform" over the eligible buffered states (default, the Stage 1 run) or "per_motion": every clip with an eligible
    state has the same probability, then a uniform eligible state of that clip (long successful episodes no longer dominate). Track only."""
    gamma: float = 0.99
    """Discount factor (default 0.99)."""
    lam: float = 0.95
    """GAE lambda of the critic targets (default 0.95)."""
    action_noise: float = 0.1
    """Std of the Gaussian exploration noise added to the actions during real-env collection (default 0.1)."""
    geo_contact: bool = False
    """Imagination: the contact probabilities of the nine links in the rewards (track_contact_label, ...) come from the geometry of the robot state and the terrain
    (``geo_contact.py``: exact ray-cast clearance of the link surface points, ``sigmoid(-(clearance - d0)/tau)``) instead of the world model's contact head, which
    keeps training as an auxiliary output. Agreement with the measured 1 N contact label on real states: accuracy 0.97, F1 0.8-0.95 per link (SONNET_NOTES.md)."""
    geo_contact_d0: float = 0.008
    """Clearance [m] at which the geometric contact probability is 0.5."""
    geo_contact_tau: float = 0.008
    """Temperature [m] of the geometric contact probability."""
    clip_aliases: str = ""
    """Path of a YAML file {old clip name: new clip name} used when the CSI phases are restored by name (a renamed clip keeps its phase; default: none)."""
    imagine_ref_replay_eps: float = 0.0
    """Demonstration replay in the imagination (DiffMimic, threshold version; default 0 = off): when the largest height deviation of a contact link from the
    reference exceeds this [m] after an imagined step, the NEXT step starts from the reference state of that frame (no gradient through the replacement) instead
    of the deviated state. The reward of the deviating step is computed on the deviated state; terminations, the failure head and the critic features use the
    states that are actually continued from. Track only."""
    imagine_link_hazard_eps: float = 0.0
    """Soft link-deviation hazard in the imagination (default 0 = off): the continuation of every imagined step is combined with
    h = sigmoid((link_deviation - eps) / tau) of the predicted (not replaced) state, d = 1 - (1 - d)(1 - h), gradient kept. Leaving the reference by more than
    eps [m] (the ``real_ref_inject_eps`` of the real collection) counts as a failure in the imagination too, consistent with the CSI. Track only."""
    imagine_link_hazard_tau: float = 0.02
    """Temperature [m] of the soft link-deviation hazard."""
    real_ref_inject_eps: float = 0.0
    """Reference injection in the real collection (default 0 = off): when the largest height deviation of a contact link from the reference of a real env exceeds
    this [m], the env is NOT terminated but moved (teleported) to the reference state of the current frame; the transition is stored as the end of an episode
    (``done``, not ``terminated``), the history restarts, and the CSI counts it as one failure of that clip (the episode does not count as a success later). Use
    with the task ``Instinct-ACP-Track-NoLinkTerm-G1-v0``. Track only."""
    abpt_one_step_weight: float = 0.0
    """Weight of the one-step (0-step critic) objective in the actor loss: loss = -((1 - w) J_H + w (r_0 + gamma V(s_1))) (ABPT-style value gradient: the critic
    that learned the real reward values gives the gradient for what the H-step return cannot differentiate; default 0)."""
    bptt_stopgrad_terms: str = ""
    """Comma-separated reward terms whose VALUE stays in the imagined return (and in the critic targets) but whose gradient w.r.t. the actions is cut
    (stop-gradient), e.g. "track_contact_label,contact_acc,joint_torque" (default "": the Stage 1 run). The gradient decomposition of the stuck clips showed
    these terms carrying as much gradient norm as the whole kinematic tracking sum, pointing elsewhere (cosine 0.1-0.2 to it). Track only."""
    pi_lr_scale: float = 1.0
    """Multiplier of the actor learning rate only (``lr`` is shared by the world model, critic and failure head; default 1)."""
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
    no_critic: bool = False
    """Ablation: no critic, i.e. no bootstrap value term in the return and no critic fit (default False)."""
    value_max: float = 0.0
    """Upper bound of the critic value wherever it is used (actor bootstrap at H and at the clip end, GAE targets; default 0 = off). The rewards are bounded
    (largest per-step reward = sum of the positive weights x dt), so no true value exceeds that / (1 - gamma); the bound only cuts off an overestimation feedback
    of the critic, e.g. through the never-regressed state after the clip end that the clip-end bootstrap queries."""

    # ---- schedule ----
    lr: float = 2e-4
    """Initial learning rate (default 2e-4)."""
    lr_decay_iters: int = 250_000
    """Iterations of the cosine decay from ``lr`` to ``lr * lr_final_ratio``, constant afterwards (default 250000)."""
    lr_final_ratio: float = 0.03
    """Final learning rate as a fraction of ``lr`` (default 0.03)."""

    # ---- performance ----
    amp: str = "bf16"
    """Autocast dtype of the MLP matmuls, "bf16" or "none" (default "bf16")."""
    compile: bool = True
    """Use ``torch.compile`` on the world-model functions (default True)."""
    compile_policy: bool = True
    """``torch.compile`` the imagination step, the reward / termination and the critic-feature functions (default True)."""
    planner_imagination_subset: bool = True
    """Only infer observation-selected planner rows when eta_rew is zero; pad small batches to preserve numerical agreement."""
    policy_autocast: bool = False
    """bf16 autocast of the actor MLP in imagination only; collection / play stay fp32 (default False)."""
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
