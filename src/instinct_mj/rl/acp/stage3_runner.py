"""SuperMimic training of ACP Stage 3: reference tracking mixed with goal reaching on procedural terrains (paper Fig. 3, Table S6).

``ACPStage3Runner`` is the Stage 2 runner (``ACPRunner``: collect -> k-step world-model fit -> H-step BPTT policy + GAE critic, the contact
planner trained in parallel on hindsight labels) for the mixed env ``Instinct-ACP-Stage3-G1-v1``. What changes with the goal-reaching envs:

* Two critics routed per env type: the reference critic Vm (features of ``ACPMDP.critic_features``) and the goal-reaching critic Vfs
  (``ACPStage3MDP.goal_critic_features``: goal command, distance, vertical offset, elapsed time, planned contacts, ...), Muon + fused Adam with the fixed
  learning rate 1e-3 (Table S6). Both critics are LN + GELU MLPs on raw features whose lambda-return targets are clamped at the analytic bound of their own objective (``ACPStage3MDP.max_step_reward``). During the first ``vfs_warmup_iters`` iterations Vfs trains on the goal-reaching imagination, which is
  excluded from the actor loss. The returns of the two regimes are normalized separately (BPTT counterpart of the separate advantage
  normalization of Table S6): each regime's discounted return is divided by the running standard deviation of its own batch.
* Imagination from reference starts is the Stage 2 one; imagination from goal-reaching starts (``_imagine_goal``) has no reference: the goal
  is fixed (position, heading, target base height), the contact targets are the planner's, the rewards are the analytic goal rewards
  (``ACPStage3MDP.reward_mixed``: indicator gates relaxed to sigmoids, ``stumbling`` from the learned ``StumbleHead``: the world model predicts
  no forces, so a head trained on the real stumbling indicators of the goal envs predicts them from the state and the action) and
  the termination is the soft fall (``goal_termination``). The planner is frozen in imagination (``eta_obs = eta_rew = 1``).
* The world model trains on the windows of all envs; the termination head (a function of reference features) only on the reference envs.
* Goal-reaching episodes are recorded for the planner: a goal episode is admitted when it got within 0.5 m horizontally at a base height
  within 0.1 m of the goal, truncated at the first step within 0.2 m, and stored in the goal partition of the replay keyed by (terrain type,
  difficulty level) with quality-based eviction (positive mechanical work per distance); the batches mix both partitions proportionally.

Collection, recording and the termination-head windows use the extension points of ``ACPRunner`` (``build_analytic_mdp``, ``make_buffer``,
``make_replay``, ``step_context``, ``buffer_fields``, ``before_reset``, ``recorder_fields``, ``term_head_windows``).
"""

from __future__ import annotations

import itertools
import math

import torch
import torch.nn as nn
import torch.utils.checkpoint
import torch.nn.functional as F

from instinct_mj.rl.supermimic.mdp import invariant_features, unflatten_state
from instinct_mj.rl.supermimic.networks import Critic, RunningNorm, mlp, run_mlp
from instinct_mj.tasks.acp.mdp import goal as goal_mdp

from .buffer import ACPReplayBuffer
from .mdp import ACT_DIM, INV_DIM, ODO_HIST, SCAN_NX, SCAN_NY, XI_DIM, targets_to_world
from .motion import NUM_CONTACT, PELVIS
from .optim import MuonAdam, graph_state, step_if_finite
from .rays import ScanFloorRunningNorm
from .runner import TARGETS_SLICE, ACPRunner, lambda_return_targets
from .stage2 import PartitionedReplay
from .stage3_mdp import GOAL_TERMS, build_stage3_mdp


def ranking_metrics(p: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
    """Tie-aware AUROC (mid-ranks) and average precision (precision at each DISTINCT score threshold, weighted by the recall increment) of the scores
    ``p`` (M,) against the boolean labels ``y`` (M,), plus the counts. The ranking metrics are omitted when there is no positive or no negative."""
    n, npos = len(y), int(y.sum())
    out = {"n_samples": float(n), "n_pos": float(npos)}
    if npos == 0:
        return out
    out["pred_on_pos"] = float(p[y].mean())
    if npos == n:
        return out
    out["pred_on_neg"] = float(p[~y].mean())
    uniq, inverse, counts = torch.unique(p, return_inverse=True, return_counts=True)  # ascending scores
    mid_rank = counts.cumsum(0) - (counts - 1) / 2.0  # 1-based mid-rank of every distinct score
    out["auroc"] = float((mid_rank[inverse][y].sum() - npos * (npos + 1) / 2) / (npos * (n - npos)))
    pos_g = torch.zeros(len(uniq), device=p.device).index_add_(0, inverse, y.float()).flip(0)  # descending score groups
    cum_tp, cum_n = pos_g.cumsum(0), counts.flip(0).cumsum(0).float()
    out["ap"] = float((pos_g / npos * cum_tp / cum_n).sum())
    return out


class StumbleHead(nn.Module):
    """P(link f stumbles during the control step) for the nine contact links, the learned counterpart of the stumbling penalty (Table 3), whose
    force condition (F_xy > 1.1 F_z + 1 N) the world model cannot evaluate. Input: the invariant state s_t and the action a_t of the step, the
    17 x 11 height scan of the true pelvis pose at s_t (whether the foot lands on a ledge or a rung), the normalized DR context xi (action delay,
    friction, ...) and the contact of the nine links at s_t (the contact history). MLP 256-128 on a ``ScanFloorRunningNorm`` input."""

    SCAN = slice(INV_DIM + ACT_DIM, INV_DIM + ACT_DIM + SCAN_NY * SCAN_NX)

    def __init__(self):
        super().__init__()
        self.in_dim = INV_DIM + ACT_DIM + SCAN_NY * SCAN_NX + XI_DIM + NUM_CONTACT
        self.norm = ScanFloorRunningNorm(self.in_dim, self.SCAN)
        self.net = mlp(self.in_dim, [256, 128], NUM_CONTACT)

    @staticmethod
    def features(S, a: torch.Tensor, scan: torch.Tensor, xi_n: torch.Tensor, contact_prev: torch.Tensor) -> torch.Tensor:
        return torch.cat((invariant_features(S), a, scan, xi_n, contact_prev), -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Logits (B, 9) of the features of :meth:`features`."""
        return run_mlp(self.net, self.norm(x))


class ACPStage3Buffer(ACPReplayBuffer):
    """``ACPReplayBuffer`` with the arena tile of every step (goal-reaching envs), the contact of the nine links at the start state ``s`` (the
    previous control step's contact of the slippage term), the measured stumbling indicators of the step and the env type:
    ``eligible_starts`` selects the reference envs (default) or the goal-reaching envs."""

    start_keys = ACPReplayBuffer.start_keys + ("tile", "contact_prev")

    def __init__(self, T: int, N: int, device: str, is_goal: torch.Tensor):
        super().__init__(T, N, device)
        self.data["tile"] = torch.zeros(T, N, dtype=torch.long, device=device)
        self.data["contact_prev"] = torch.zeros(T, N, NUM_CONTACT, device=device)
        self.data["stumble"] = torch.zeros(T, N, NUM_CONTACT, device=device)
        self.is_goal = is_goal

    def eligible_starts(self, ok_fn, env_range=None, max_age=None, key: str = "f", goal: bool = False) -> torch.Tensor:
        idx = super().eligible_starts(ok_fn, env_range, max_age, key)
        return idx[self.is_goal[idx[:, 1]] == goal]


class ACPStage3Runner(ACPRunner):
    """Trainer of the mixed Stage 3 env; see the module docstring. Public API as :class:`ACPRunner`."""

    policy_graph_capable = True  # ``cuda_graph_policy``: ``policy_update`` is the captured ``_policy_graph_step`` (see ``check_policy_graph_options``)
    needs_contact = True  # the measured contact of the step is the next step's ``contact_prev`` (slippage, StumbleHead inputs)

    def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device: str = "cuda:0"):
        super().__init__(env, train_cfg, log_dir, device)
        if self.cfg["contact_source"] != "planner" or not self.cfg["train_planner"]:
            raise ValueError("Stage 3 uses the planner throughout (contact_source='planner') and keeps training it (train_planner=True).")
        if self.cfg["start_sampling"] != "uniform" or self.cfg["wm_balance_frac"] > 0 or self.cfg["start_balance_frac"] > 0 or self.cfg["imagine_action_noise"] > 0 or self.cfg["bptt_stopgrad_terms"] or self.cfg["imagine_ref_replay_eps"] > 0 or self.cfg["real_ref_inject_eps"] > 0 or self.cfg["imagine_link_hazard_eps"] > 0 or self.cfg["abpt_one_step_weight"] > 0 or self.cfg["geo_contact"]:
            raise ValueError("start_sampling / wm_balance_frac / start_balance_frac / imagine_action_noise / bptt_stopgrad_terms / imagine_ref_replay_eps / real_ref_inject_eps / imagine_link_hazard_eps / abpt_one_step_weight / geo_contact are Track (Stage 1 / 2) options: the Stage 3 returns do not implement them.")
        if self.cfg["no_critic"]:
            raise ValueError("Stage 3 bootstraps with the two critics (no_critic=False).")

    # ------------------------------------------------------------------------------------------------
    def setup_training(self) -> None:
        """``ACPRunner.setup_training`` (with the Stage 3 MDP, buffer and replay of the hooks below) plus the goal-reaching critic Vfs."""
        c, device = self.cfg, self.device
        super().setup_training()
        self.critic_goal = Critic(self.mdp.goal_critic_dim(), hidden=[int(h) for h in c["critic_hidden"].split(",")], act=nn.GELU, layer_norm=True, normalize_input=False).to(device)  # like Vm: LN + GELU on the raw features
        self.opt_vg = MuonAdam([self.critic_goal], c["vfs_lr"])  # Muon + fused Adam, fixed learning rate (Table S6), not scheduled in learn()
        self.v_max_goal = None  # analytic value bound of the goal-reaching objective (Vfs); ``self.v_max`` is the one of the reference objective (Vm)
        self.stumble_head = StumbleHead().to(device)
        self.opt_stumble = torch.optim.Adam(self.stumble_head.parameters(), lr=c["stumble_lr"])
        self.stumble_envs = self.stumble_eval_envs = self.is_goal.nonzero(as_tuple=False).squeeze(-1)  # all goal envs train the head and are evaluated
        self.ret_std: dict[str, torch.Tensor | None] = {"ref": None, "goal": None}
        self.contact_prev = torch.zeros(self.N, NUM_CONTACT, device=device)
        self.robot = self.unwrapped.scene["robot"]
        self.action_joint_ids = self.unwrapped.action_manager.get_term("joint_pos").target_ids
        self.s3_acc: dict[str, list] = {}
        if self.resume_path is not None:
            self._load_stage3_sidecar(self.resume_path)

    # ------------------------------------------------------------------------------------------------
    # extension points of ``ACPRunner`` (collection, recording, world-model / termination-head windows)
    def uses_env_reward(self) -> bool:
        """Stage 3 starts the reference envs at phase 0 (``episode_start`` mode "start", no reset bins): the env reward (and its planner-term sync) only fills the buffer's ``r``
        field, which nothing reads, so the reward manager is not evaluated (~17 % of an env step at 2048 envs)."""
        return self.reset_bins is not None

    def build_analytic_mdp(self):
        c = self.cfg
        self.goal_term = self.unwrapped.command_manager.get_term("acp_goal")
        self.is_goal = self.goal_term.is_goal
        self.tables = self.goal_term.tables
        self.num_goal_envs = int(self.is_goal.sum())
        if self.num_goal_envs == 0 or self.num_goal_envs == self.N:
            raise ValueError("Stage 3 trains on a mix of reference and goal-reaching envs (goal_env_fraction in (0, 1)).")
        return build_stage3_mdp(self.unwrapped, c["heightmap_resolution"], c["heightmap_cache_dir"], self.device, c, exact_scan=c["exact_scan"])

    def make_buffer(self) -> ACPStage3Buffer:
        c = self.cfg
        return ACPStage3Buffer(c["buffer_steps"], self.N, self.device, self.is_goal)

    def make_replay(self) -> PartitionedReplay:
        c = self.cfg
        return PartitionedReplay(c["planner_replay_capacity"], c["planner_tile_capacity"], self.device, c["planner_ref_tile_capacity"])

    def step_context(self) -> dict:
        """The arena tile, the goal distance / vertical offset (state s_t) and the contact of the previous step (slippage) of every env."""
        gt = self.goal_term
        dist, _, dz, _ = gt.goal_geometry()
        return {"tile": gt.tile_id.clone(), "goal_dist": dist, "goal_dz": dz, "contact_prev": self.contact_prev}

    def buffer_fields(self, ctx: dict) -> dict:
        return {"tile": ctx["tile"], "contact_prev": ctx["contact_prev"], "stumble": ctx["stumble"]}

    def before_reset(self, ctx: dict, done: torch.Tensor, terminated: torch.Tensor, contact: torch.Tensor) -> None:
        ctx["stumble"] = goal_mdp.stumbling_links(self.unwrapped)  # measured during the step, read before the reset clears the sensor history
        gt = self.goal_term
        ended = done & self.is_goal
        if ended.any():
            self.s3_acc.setdefault("goal/episode_success", []).append(((gt.min_xy_ok <= 0.5) & ~terminated)[ended].float().mean())
        self.contact_prev = torch.where(done[:, None], torch.zeros_like(contact), contact)  # the env's slippage timer restarts with the episode

    def recorder_fields(self, progress: torch.Tensor, ctx: dict) -> dict:
        """Goal envs report the goal distance / offset, terrain type and level and the positive mechanical work of the step (power at the end of
        the step x dt) to the recorder; reference envs report their clip's terrain type and tile (the strata of the reference partition)."""
        gt, robot, ids = self.goal_term, self.robot, self.action_joint_ids
        work = (robot.data.qfrc_actuator[:, ids] * robot.data.joint_vel[:, ids]).clamp(min=0.0).sum(-1) * self.unwrapped.step_dt
        return dict(is_goal=self.is_goal, phase=torch.where(self.is_goal, torch.zeros_like(progress), progress.clamp(0.0, 1.0)),
                    goal_xy=ctx["goal_dist"], goal_dz=ctx["goal_dz"], work_power=work,
                    terrain_type=torch.where(self.is_goal, gt.terrain_type, ctx["ref_type"]),
                    tile=torch.where(self.is_goal, gt.level, ctx["ref_tile"]))  # fmt: skip

    def wm_update(self):
        """``ACPRunner.wm_update`` plus one step of the stumbling head on single real transitions of the goal-reaching envs."""
        m = super().wm_update()
        if m is None:  # no complete world-model window: the iteration is skipped
            return None
        self.stumble_update()
        return m

    def stumble_update(self) -> None:
        """One Adam step of :class:`StumbleHead` on ``wm_batch`` random buffered transitions of the goal envs: (state, action, scan, context,
        contact) at s_t -> the stumbling indicators measured during the step."""
        c, buf = self.cfg, self.buffer
        B = c["wm_batch"]
        env = self.stumble_envs[torch.randint(0, len(self.stumble_envs), (B,), device=self.device)]
        rows = buf._row(torch.randint(0, buf.size, (B,), device=self.device))
        S = unflatten_state(buf.data["s"][rows, env])
        with torch.no_grad():
            bp, bR, _, _ = self.mdp.base_of(self.kin.forward(*S))
            xi_n = self.mdp.normalize_xi(self.xi_of(env, buf.data["xi_dyn"][rows, env]))
            X = StumbleHead.features(S, buf.data["a"][rows, env], self.mdp.scan(bp, bR), xi_n, buf.data["contact_prev"][rows, env])
        y = buf.data["stumble"][rows, env]
        self.stumble_head.norm.update(X)
        logits = self.stumble_head(X)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        self.opt_stumble.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.stumble_head.parameters(), 1.0)
        self.opt_stumble.step()
        acc = self.s3_acc
        acc.setdefault("goal/stumble_loss", []).append(loss.detach())
        acc.setdefault("goal/stumble_true_rate", []).append(y.mean())  # per (step, link)
        acc.setdefault("goal/stumble_pred_rate", []).append(torch.sigmoid(logits.detach()).mean())

    @torch.no_grad()
    def stumble_eval(self) -> dict:
        """Validity of the stumbling head on the newest ``stumble_eval_states`` buffered transitions of the goal envs (keys ``goal/stumble_train_*``), evaluated in chunks: tie-aware AUROC and average precision of the predicted link
        probabilities against the measured indicators (and the numbers of positives / samples), the mean prediction on positives / negatives, the
        predicted vs true rate per terrain type. A constant output would match the overall rate (the base rate is ~0.4 %), so the ranking metrics
        are the evidence, not the rate."""
        c, buf, env = self.cfg, self.buffer, self.stumble_eval_envs
        n_ages = min(buf.size, max(1, c["stumble_eval_states"] // len(env)))
        chunk = max(1, c["stumble_eval_chunk"] // len(env))  # ages per chunk
        ps, ys, terrain = [], [], []
        for a0 in range(0, n_ages, chunk):
            rows = buf._row(buf.size - 1 - torch.arange(a0, min(a0 + chunk, n_ages), device=self.device))[:, None]  # (ages, 1) x (1, envs): no copy of the other envs
            pick = lambda key: buf.data[key][rows, env[None, :]].flatten(0, 1)  # noqa: E731
            S = unflatten_state(pick("s"))
            bp, bR, _, _ = self.mdp.base_of(self.kin.forward(*S))
            e = env.repeat(rows.shape[0])
            xi_n = self.mdp.normalize_xi(self.xi_of(e, pick("xi_dyn")))
            ps.append(torch.sigmoid(self.stumble_head(StumbleHead.features(S, pick("a"), self.mdp.scan(bp, bR), xi_n, pick("contact_prev")))))
            ys.append(pick("stumble")), terrain.append(self.goal_term.terrain_type[e])
        p, y, terrain = torch.cat(ps), torch.cat(ys), torch.cat(terrain)
        name = "train"
        out = {f"goal/stumble_{name}_pred_rate": float(p.mean()), f"goal/stumble_{name}_true_rate": float(y.mean())}
        out.update({f"goal/stumble_{name}_{k}": v for k, v in ranking_metrics(p.flatten(), y.flatten() > 0.5).items()})
        for t, tname in enumerate(self.unwrapped.cfg.scene.arena.types):
            sel = terrain == t
            if sel.any():
                out[f"goal/stumble_{name}_pred_rate/{tname}"], out[f"goal/stumble_{name}_true_rate/{tname}"] = float(p[sel].mean()), float(y[sel].mean())
        return out

    def term_head_windows(self, w: dict, limit: int) -> torch.Tensor:
        """The termination head is a function of the reference critic features: it trains on the windows of the reference envs only (the world
        model trains on all)."""
        return (~self.is_goal[w["env"]]).nonzero(as_tuple=False).squeeze(-1)[:limit]

    # ------------------------------------------------------------------------------------------------
    def sample_goal_starts(self, B: int):
        """Uniform over all buffered states of the goal-reaching envs (the imagination has no reference clip to stay inside)."""
        eligible = self.buffer.eligible_starts(lambda done: torch.ones_like(done, dtype=torch.bool), key="done", goal=True)
        if len(eligible) == 0:
            raise RuntimeError("No buffered state of a goal-reaching env.")
        pick = eligible[torch.randint(0, len(eligible), (B,), device=self.device)]
        return self.buffer.starts_at(pick[:, 0], pick[:, 1], ODO_HIST)

    def _imagine_goal(self, s: dict, h: dict) -> dict:
        """H-step imagination of the policy on goal-reaching envs from buffer start states: no reference, a fixed goal, the planner's contact
        targets (frozen planner) in the observation. Gradients flow as in :meth:`ACPRunner._imagine`."""
        c, mdp, kin = self.cfg, self.mdp, self.kin
        HZ = self.horizon
        S = unflatten_state(s["s"])
        a_prev, a_prev2, env_idx = s["a_prev"], s["a_prev2"], s["env"]
        xi = self.xi_of(env_idx, s["xi_dyn"])
        xi_n = mdp.normalize_xi(xi)
        gpos, gyaw = s["goal"][:, :3], s["goal"][:, 3]
        with torch.no_grad():
            B = S[0].shape[0]
            hS = tuple(x.flatten(0, 1) for x in unflatten_state(h["s"]))
            fk = kin.forward(*hS)
            xi_h = xi.unsqueeze(1).expand(-1, ODO_HIST, -1).flatten(0, 1)
            o_hist = mdp.proprio(fk[1][:, PELVIS], fk[3][:, PELVIS], hS[4], hS[5], h["a_prev"].flatten(0, 1), xi_h).view(B, ODO_HIST, -1)
            noise = (2 * torch.rand_like(o_hist) - 1) * mdp.proprio_noise
            o_hist = (o_hist + noise.gather(1, h["slot"].unsqueeze(-1).expand_as(noise)))[:, ODO_HIST - self.hist :]
            lp_hist = fk[2 if mdp.contact_acc_vel else 0].view(B, ODO_HIST, NUM_CONTACT, 3)[:, -2:].transpose(0, 1).contiguous()
            pelvis_hist = mdp.pose_history(fk[0][:, PELVIS].view(B, ODO_HIST, 3), fk[1][:, PELVIS].view(B, ODO_HIST, 3, 3))
            odo_delay, odo_drift = s["odo_delay"], s["odo_drift"]
            no_ref = (torch.zeros(B, NUM_CONTACT, 3, device=self.device), torch.zeros(B, NUM_CONTACT, dtype=torch.bool, device=self.device))
            use_all = torch.ones(B, dtype=torch.bool, device=self.device)
        Ss, As, Aps, Aps2, Ls, Pw, Pb, Hard, Fks = [], [], [], [], [], [], [], [], []  # Fks: ``kin.forward_all`` of the states S_0 .. S_H
        wm_contact0 = s["wm_contact0"] if self.wm_contact_feedback else None

        def observe(fk, hist, push):
            bp, bR, _, _ = mdp.base_of(fk)
            if push:
                hist = mdp.push_pose(hist, bp, bR)
            ep_, eyaw = mdp.estimated_pose(hist, odo_delay, odo_drift)
            eR = mdp.yaw_rotation(eyaw)
            return mdp.scan(ep_, eR), mdp.goal_command(ep_, eR, gpos, gyaw), hist

        fk = self.fk_all(S)  # forward kinematics of the current state, computed once (the next one comes out of ``step_wm``)
        for t in range(HZ + 1):
            if t > 0:
                o_hist = self.push(o_hist, S, fk, a_prev, xi)
            Fks.append(fk)
            scan, goal, pelvis_hist = observe(fk, pelvis_hist, t > 0)
            goal_noisy = mdp.noisy_goal(goal)  # one noise sample per imagined state: the observation and the planner see the same goal
            ptgt, pvalid, _ = self._plan(o_hist, scan, goal_noisy, 1.0)
            Pw.append((targets_to_world(*mdp.estimated_pose(pelvis_hist, odo_delay, odo_drift), ptgt), pvalid)), Pb.append((ptgt, pvalid))
            if t == HZ:  # the observation of the last state is not built: only its planned contacts (reward / critic of that state)
                break
            Aps.append(a_prev), Aps2.append(a_prev2)
            obs = self.assemble(o_hist, scan, *no_ref, goal_noisy, (use_all, ptgt))
            a = self.ac.actor(self.pol_norm(obs))
            step_args = (S, fk, a, a_prev, xi_n)
            if self.wm_contact_spec:
                step_args += (obs[:, TARGETS_SLICE].detach() if self.wm_ref_contacts else None, wm_contact0)
            if c["checkpoint_imagination"] and torch.is_grad_enabled():
                a, S, fk, logits, hard = torch.utils.checkpoint.checkpoint(self.step_wm, *step_args, use_reentrant=False)
            else:
                a, S, fk, logits, hard = self.step_wm(*step_args)
            if self.wm_contact_feedback:
                wm_contact0 = torch.sigmoid(logits)
            a_prev2, a_prev = a_prev, a
            Ss.append(S), As.append(a), Ls.append(logits), Hard.append(hard.detach())
        return {"S0": unflatten_state(s["s"]), "S": S, "Ss": Ss, "As": As, "Aps": Aps, "Aps2": Aps2, "Ls": Ls, "Pw": Pw, "Pb": Pb, "Hard": Hard, "Fks": Fks,
                "lp_hist": lp_hist, "ep0": s["ep_step"].float(), "xi": xi, "xi_n": xi_n, "gpos": gpos, "gyaw": gyaw, "tile": s["tile"],
                "f0": s["f"], "phase": s["phase"], "origin": s["origin"], "contact_prev0": s["contact_prev"]}  # fmt: skip

    # ------------------------------------------------------------------------------------------------
    def _fit_critic(self, critic, opt, X: torch.Tensor, targets: torch.Tensor, graph: bool = False) -> torch.Tensor:
        """Official critic fit on the raw features: ``critic_epochs`` epochs of ``critic_minibatches`` minibatches of ONE permutation."""
        c, dev = self.cfg, self.device
        Xf, Yf = X[:-1].reshape(-1, X.shape[-1]), targets.reshape(-1)
        n = Xf.shape[0]
        mb = n // c["critic_minibatches"]
        v_loss = torch.zeros((), device=dev)
        perm = torch.randperm(n, device=dev)
        for _ in range(c["critic_epochs"]):
            for i in range(c["critic_minibatches"]):
                idx = perm[i * mb : (i + 1) * mb]
                v_loss = ((critic(Xf[idx]) - Yf[idx]) ** 2).mean()
                opt.zero_grad(set_to_none=not graph)  # (a captured update zeroes the gradients in place)
                v_loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                opt.step()
        return v_loss.detach()

    def _gae_targets(self, critic, X: torch.Tensor, rews: torch.Tensor, dones: torch.Tensor, v_max) -> torch.Tensor:
        """GAE(lambda) targets of one critic: raw values inside the lambda return, the targets clamped at the analytic bound ``v_max`` of that critic's objective (official)."""
        c = self.cfg
        with torch.no_grad():
            return lambda_return_targets(rews, dones, critic(X), c["gamma"], c["lam"]).clamp(max=v_max)

    def _survival(self, d: torch.Tensor, graph: bool) -> torch.Tensor:
        """``cumprod(1 - d)`` over the steps (the backward of ``torch.cumprod`` synchronizes with the host: a captured update multiplies step by step)."""
        if graph:
            return torch.stack(list(itertools.accumulate((1 - d).unbind(0), torch.mul)))
        return torch.cumprod(1 - d, 0)

    def _return_ref(self, im: dict, B: int, v_max, graph: bool = False) -> dict:
        """Discounted imagination return of the reference starts (Vm bootstrap, termination head) as in ``ACPRunner.policy_update``. The forward kinematics of the imagined states are
        the ones the imagination computed (``im["Fk"]``, ``im["fk0"]``): nothing is recomputed here."""
        c, mdp, dev, HZ = self.cfg, self.mdp, self.device, self.horizon
        S0, f0, S, f, phase, origin, xi, xi_n = im["S0"], im["f0"], im["S"], im["f"], im["phase"], im["origin"], im["xi"], im["xi_n"]
        Ss, As, Aps, Aps2, Fs, Ls, Gs = im["Ss"], im["As"], im["Aps"], im["Aps2"], im["Fs"], im["Ls"], im["Gs"]
        flat = lambda xs: torch.stack(xs).flatten(0, 1)  # noqa: E731  (H*B, ...)
        rep = lambda x: x.unsqueeze(0).expand(HZ, *x.shape).flatten(0, 1)  # noqa: E731
        S_all = tuple(flat([St[i] for St in Ss]) for i in range(6))
        fk_all = tuple(torch.stack([F_[i] for F_ in im["Fk"]]).flatten(0, 1) for i in range(4))  # kin.forward_all of S_all
        planned = (torch.stack([p[0] for p in im["Ptgt"][1:]]).flatten(0, 1), torch.stack([p[1] for p in im["Ptgt"][1:]]).flatten(0, 1))
        eta_rew = im["eta_rew"] if torch.is_tensor(im["eta_rew"]) else torch.tensor(im["eta_rew"], device=dev)
        qd_start = torch.cat((S0[5].unsqueeze(0), torch.stack([St[5] for St in Ss[:-1]])), 0).flatten(0, 1)
        r, d, terms = self.reward_term(S_all, qd_start, flat(Fs), rep(phase), rep(origin), flat(As), flat(Aps), flat(Aps2), rep(xi), flat(Ls),
                                       im["lp_hist"], im["ep0"], planned, eta_rew, HZ, fk_all)  # fmt: skip
        G1 = torch.stack(Gs[1:]).flatten(0, 1)
        X1 = self.critic_input(S_all, flat(Fs), rep(phase), rep(origin), rep(xi_n), G1[:, :3], G1[:, 3], fk_all) if c["term_source"] != "analytic" else None
        d, term_stats = self.apply_term_source(d, X1)  # reference starts only; the goal starts keep the analytic fall
        r, d = r.view(HZ, B), d.view(HZ, B)
        alive = self._survival(d, graph)
        c_t = torch.cat((torch.ones(1, B, device=dev), alive[:-1]), 0)
        disc = c["gamma"] ** torch.arange(HZ, device=dev, dtype=torch.float32).unsqueeze(-1)
        J = (disc * c_t * r).sum(0) + (c["gamma"] ** HZ) * alive[-1] * self.value_bound(self.critic(self.critic_input(S, f, phase, origin, xi_n, Gs[-1][:, :3], Gs[-1][:, 3], im["Fk"][-1])), v_max)
        with torch.no_grad():
            S_c = tuple(torch.cat((S0[i].unsqueeze(0), torch.stack([St[i].detach() for St in Ss])), 0).flatten(0, 1) for i in range(6))
            fk_c = tuple(torch.cat((im["fk0"][i].unsqueeze(0), torch.stack([F_[i].detach() for F_ in im["Fk"]])), 0).flatten(0, 1) for i in range(4))  # kin.forward_all of S_c
            f_c = torch.cat((f0.unsqueeze(0), torch.stack(Fs)), 0).flatten()
            rep1 = lambda x: x.unsqueeze(0).expand(HZ + 1, *x.shape).flatten(0, 1)  # noqa: E731
            Gc = torch.stack(Gs).flatten(0, 1)
            X = self.critic_input(S_c, f_c, rep1(phase), rep1(origin), rep1(xi_n), Gc[:, :3], Gc[:, 3], fk_c).view(HZ + 1, B, -1)
        terms = {k: v.detach().view(HZ, B).mean() for k, v in terms.items()}
        terms.update(term_stats)
        return {"J": J, "X": X, "rews": r.detach(), "dones": d.detach(), "survival": alive[-1].mean().detach(), "terms": terms,
                "reward": (c_t * r).detach().sum(0).mean() / HZ / mdp.dt, "S_all": S_all, "hard": torch.stack(im["Hard"]).mean()}  # fmt: skip

    def _return_goal(self, im: dict, B: int, v_max_goal, graph: bool = False) -> dict:
        """Discounted imagination return of the goal-reaching starts: analytic goal rewards, soft fall termination, Vfs bootstrap. The forward kinematics of the imagined
        states are the ones the imagination computed (``im["Fks"]``, states S_0 .. S_H): nothing is recomputed here."""
        c, mdp, dev, HZ = self.cfg, self.mdp, self.device, self.horizon
        S0, S, Ss = im["S0"], im["S"], im["Ss"]
        flat = lambda xs: torch.stack(xs).flatten(0, 1)  # noqa: E731
        rep = lambda x: x.unsqueeze(0).expand(HZ, *x.shape).flatten(0, 1)  # noqa: E731
        S_all = tuple(flat([St[i] for St in Ss]) for i in range(6))
        stack_fk = lambda fks: tuple(torch.stack([F_[i] for F_ in fks]).flatten(0, 1) for i in range(4))  # noqa: E731  (kin.forward_all of states, step-major)
        fk_all, fk_start = stack_fk(im["Fks"][1:]), stack_fk(im["Fks"][:-1])  # of S_all, and of S_start (S_0 .. S_{H-1})
        planned = (torch.stack([p[0] for p in im["Pw"][1:]]).flatten(0, 1), torch.stack([p[1] for p in im["Pw"][1:]]).flatten(0, 1))
        qd_start = torch.cat((S0[5].unsqueeze(0), torch.stack([St[5] for St in Ss[:-1]])), 0).flatten(0, 1)
        C = torch.sigmoid(torch.stack(im["Ls"]))  # (H, B, 9) contact probabilities of the transitions' end states
        C_prev = torch.cat((im["contact_prev0"].unsqueeze(0), C[:-1]), 0)  # the first transition: the measured contact of its start state
        S_start = tuple(torch.cat((S0[i].unsqueeze(0), torch.stack([St[i] for St in Ss[:-1]])), 0).flatten(0, 1) for i in range(6))
        bp_s, bR_s, _, _ = mdp.base_of(fk_start)
        x_st = StumbleHead.features(S_start, flat(im["As"]), mdp.scan(bp_s, bR_s), rep(im["xi_n"]), C_prev.flatten(0, 1))
        p_stumble = torch.sigmoid(self.stumble_head(x_st)).sum(-1)  # expected number of stumbling links
        tile, gpos, gyaw = rep(im["tile"]), rep(im["gpos"]), rep(im["gyaw"])
        r, terms = mdp.reward_mixed(S_all, qd_start, rep(im["f0"]), rep(im["phase"]), rep(im["origin"]), flat(im["As"]), flat(im["Aps"]), flat(im["Aps2"]),
                                    rep(im["xi"]), C.flatten(0, 1), C_prev.flatten(0, 1), im["lp_hist"], im["ep0"], HZ, torch.ones(HZ * B, dtype=torch.bool, device=dev),
                                    gpos, gyaw, tile, planned, torch.ones((), device=dev), hard=False, stumbling=p_stumble, fk=fk_all)  # fmt: skip
        d = mdp.goal_termination(S_all, tile, fk=fk_all)
        r, d = r.view(HZ, B), d.view(HZ, B)
        alive = self._survival(d, graph)
        c_t = torch.cat((torch.ones(1, B, device=dev), alive[:-1]), 0)
        disc = c["gamma"] ** torch.arange(HZ, device=dev, dtype=torch.float32).unsqueeze(-1)
        t_steps = (im["ep0"].unsqueeze(0) + torch.arange(HZ + 1, device=dev).unsqueeze(1)) * mdp.dt / mdp.goal_episode_s  # (H+1, B)

        def features(Sx, gp, gy, t_norm, Pb, xin, fk):
            return mdp.goal_critic_features(Sx, gp, gy, t_norm, Pb[0], Pb[1], xin, fk)

        X_T = features(S, im["gpos"], im["gyaw"], t_steps[-1], im["Pb"][-1], im["xi_n"], im["Fks"][-1])
        J = (disc * c_t * r).sum(0) + (c["gamma"] ** HZ) * alive[-1] * self.value_bound(self.critic_goal(X_T), v_max_goal)
        with torch.no_grad():
            S_c = tuple(torch.cat((S0[i].unsqueeze(0), torch.stack([St[i].detach() for St in Ss])), 0).flatten(0, 1) for i in range(6))
            fk_c = stack_fk([tuple(x.detach() for x in F_) for F_ in im["Fks"]])  # of S_c
            Pb_t = torch.stack([p[0] for p in im["Pb"]]).flatten(0, 1), torch.stack([p[1] for p in im["Pb"]]).flatten(0, 1)
            rep1 = lambda x: x.unsqueeze(0).expand(HZ + 1, *x.shape).flatten(0, 1)  # noqa: E731
            X = features(S_c, rep1(im["gpos"]), rep1(im["gyaw"]), t_steps.flatten(), Pb_t, rep1(im["xi_n"]), fk_c).view(HZ + 1, B, -1)
        terms = {f"goal/{k}": v.detach().view(HZ, B).mean() for k, v in terms.items() if k in (*GOAL_TERMS, "track_planned_contact")}
        return {"J": J, "X": X, "rews": r.detach(), "dones": d.detach(), "survival": alive[-1].mean().detach(), "terms": terms,
                "reward": (c_t * r).detach().sum(0).mean() / HZ / mdp.dt, "hard": torch.stack(im["Hard"]).mean()}  # fmt: skip

    def _scale(self, J: torch.Tensor, prev: torch.Tensor, has_prev: torch.Tensor) -> torch.Tensor:
        """Running standard deviation of the regime's discounted return (separate normalization of the two regimes): ``prev`` is the running value (``has_prev``: it exists, else
        the batch's standard deviation starts it)."""
        std = J.detach().std().clamp(min=1e-3)
        m = self.cfg["return_norm_momentum"]
        return torch.where(has_prev, m * prev + (1 - m) * std, std)

    # ------------------------------------------------------------------------------------------------
    def _policy_step(self, x: dict, warm: bool, graph: bool = False) -> dict:
        """One policy step on both regimes from the imagination starts ``x`` (``sm:`` / ``hm:`` reference starts / histories, ``sg:`` / ``hg:`` goal-reaching ones, the value bounds
        ``e:vmax`` / ``e:vmax_goal``, the etas ``e:obs`` / ``e:rew`` and the running return scales ``rs:prev`` / ``rs:has`` of (reference, goal), all tensors): H-step imagination from
        both kinds of starts, BPTT of the separately normalized discounted returns (the goal-reaching part is excluded from the actor loss during the Vfs warmup), then the GAE fits of Vm and
        Vfs. ``graph``: the capture-safe variants of the host-synchronizing steps (``policy_graph_mode`` = ``warm``). Returns tensors only."""
        c, dev = self.cfg, self.device
        B_g = int(c["pi_batch"] * c["pi_goal_fraction"])
        B_m = c["pi_batch"] - B_g
        pick = lambda prefix: {k[len(prefix):]: v for k, v in x.items() if k.startswith(prefix)}  # noqa: E731
        s_m, h_m, s_g, h_g = pick("sm:"), pick("hm:"), pick("sg:"), pick("hg:")
        v_max, v_max_goal = x["e:vmax"], x["e:vmax_goal"]
        frozen = (self.wm, self.critic, self.critic_goal, self.stumble_head, *([self.term_head] if self.term_head is not None else []))
        for m_ in frozen:
            for p_ in m_.parameters():
                p_.requires_grad_(False)
        im_m = self._imagine(s_m, h_m, (x["e:obs"], x["e:rew"], 2))  # (Stage 3: the planner throughout, the etas are 1)
        with torch.set_grad_enabled(not warm):  # warmup: the goal-reaching rollout only supplies the training data of Vfs
            im_g = self._imagine_goal(s_g, h_g)
            ret_g = self._return_goal(im_g, B_g, v_max_goal, graph)
        ret_m = self._return_ref(im_m, B_m, v_max, graph)
        std_m, std_g = self._scale(ret_m["J"], x["rs:prev"][0], x["rs:has"][0]), self._scale(ret_g["J"], x["rs:prev"][1], x["rs:has"][1])
        J_m = ret_m["J"] / std_m
        J_g = ret_g["J"] / std_g
        loss = -(J_m.sum() / B_m) if warm else -(J_m.sum() + J_g.sum()) / (B_m + B_g)
        self.opt_pi.zero_grad(set_to_none=not graph)
        loss.backward()
        gn = nn.utils.clip_grad_norm_(self.ac.actor.parameters(), 1.0)
        nn.utils.clip_grad_value_(self.ac.actor.parameters(), 0.1)
        if graph and torch.cuda.is_current_stream_capturing():
            step_if_finite(self.opt_pi, torch.isfinite(gn))
        elif torch.isfinite(gn):
            self.opt_pi.step()
        for m_ in frozen:
            for p_ in m_.parameters():
                p_.requires_grad_(True)
        v_loss = vg_loss = torch.zeros((), device=dev)
        if not c["no_critic"]:
            v_loss = self._fit_critic(self.critic, self.opt_v, ret_m["X"], self._gae_targets(self.critic, ret_m["X"], ret_m["rews"], ret_m["dones"], v_max), graph)
            vg_loss = self._fit_critic(self.critic_goal, self.opt_vg, ret_g["X"], self._gae_targets(self.critic_goal, ret_g["X"], ret_g["rews"], ret_g["dones"], v_max_goal), graph)
        term_means = dict(ret_m["terms"])
        with torch.no_grad():
            term_means["jl_soft"] = self.mdp.joint_violation(ret_m["S_all"][4])[0]
            term_means["jl_hard"] = ret_m["hard"]
        r_comb = (ret_m["reward"] * B_m + ret_g["reward"] * B_g) / (B_m + B_g)
        log = {"goal/imagination_reward": ret_g["reward"], "goal/imagination_survival": ret_g["survival"], "goal/vfs_loss": vg_loss, "goal/return_std": std_g, "ref/return_std": std_m,
               "goal/return_mean": J_g.detach().mean(), "ref/return_mean": J_m.detach().mean(), **ret_g["terms"]}  # fmt: skip
        return {"pi": loss.detach(), "v": v_loss, "ir": r_comb, "gn": gn.detach(), "surv": ret_m["survival"], "rs": torch.stack((std_m, std_g)).detach(),
                **{"t:" + k: t for k, t in term_means.items()}, **{"l:" + k: t for k, t in log.items()}}  # fmt: skip

    def _policy_graph_step(self, x: dict) -> dict:
        """The body of the captured policy update (``cuda_graph_policy``); ``policy_graph_mode`` (``warm``) is fixed at the capture."""
        return self._policy_step(x, bool(self._policy_mode), graph=True)

    def policy_graph_mode(self, warm: bool) -> int:
        """The structure of the captured update: 1 while Vfs warms up (the goal-reaching rollout is outside the actor's graph of gradients), else 0. The graph is re-recorded when it
        changes (once, at ``vfs_warmup_iters``). The etas are 1 (``contact_source`` "planner"), the value bounds and the running return scales are graph inputs."""
        return int(warm)

    def policy_graph_state(self) -> dict:
        """The tensors the captured update reads through Python objects (their addresses are checked at every replay): the base runner's plus the goal critic and the stumbling head."""
        planner = [self.planner] if self.planner is not None else []
        return graph_state([self.wm, self.ac, self.critic, self.critic_goal, self.stumble_head, self.pol_norm, *planner], [self.opt_pi, self.opt_v, self.opt_vg],
                           grad_modules=[self.ac.actor, self.critic, self.critic_goal])  # (the stumbling head's own gradients belong to its eager trainer)

    def policy_update(self):
        """One policy step on both regimes (see :meth:`_policy_step`; one CUDA graph replay with ``cuda_graph_policy``). The sampling of the starts stays outside the graph (data-dependent
        shapes)."""
        c, dev = self.cfg, self.device
        warm = self.it < c["vfs_warmup_iters"]
        eta_obs, eta_rew = self.eta()  # 1 / 1 (contact_source "planner")
        self.update_value_bound(eta_rew)  # Vm: the tracking terms of the reference starts
        self.v_max_goal = self.mdp.max_step_reward(eta_rew, goal=True) / (1 - c["gamma"])  # Vfs: the goal terms, the regularizers and the planned-contact term of the goal starts
        B_g = int(c["pi_batch"] * c["pi_goal_fraction"])
        B_m = c["pi_batch"] - B_g
        s_m, h_m = self.sample_valid_starts(B_m)
        s_g, h_g = self.sample_goal_starts(B_g)
        has = [self.ret_std["ref"] is not None, self.ret_std["goal"] is not None]
        x = {**{"sm:" + k: v for k, v in s_m.items()}, **{"hm:" + k: v for k, v in h_m.items()}, **{"sg:" + k: v for k, v in s_g.items()}, **{"hg:" + k: v for k, v in h_g.items()},
             "e:obs": torch.tensor(eta_obs, device=dev), "e:rew": torch.tensor(eta_rew, device=dev), "e:vmax": torch.tensor(self.v_max, device=dev),
             "e:vmax_goal": torch.tensor(self.v_max_goal, device=dev), "rs:has": torch.tensor(has, device=dev),
             "rs:prev": torch.stack([r if r is not None else torch.zeros((), device=dev) for r in (self.ret_std["ref"], self.ret_std["goal"])])}  # fmt: skip
        captured = getattr(self, "_captured_policy", None)
        if captured is None:
            out = self._policy_step(x, warm)
        else:
            mode = self.policy_graph_mode(warm)
            if mode != getattr(self, "_policy_mode", None):  # the structure of the update changes (the Vfs warmup ends): re-record
                if getattr(self, "_policy_mode", None) is not None:
                    print(f"[policy graph] mode {self._policy_mode} -> {mode} at iteration {self.it}: re-recording", flush=True)
                    captured.invalidate()
                self._policy_mode = mode
            out = {k: v.clone() for k, v in captured(x).items()}  # (the next replay overwrites the graph's outputs)
        self.ret_std = {"ref": out["rs"][0], "goal": out["rs"][1]}
        term_means = {k[2:]: v for k, v in out.items() if k.startswith("t:")}
        self._log_stage3(warm, {k[2:]: v for k, v in out.items() if k.startswith("l:")})
        return out["pi"], out["v"], out["ir"], out["gn"], out["surv"], term_means

    def _log_stage3(self, warm: bool, log: dict) -> None:
        """Goal-reaching diagnostics, flushed to the writer / stdout with the log line of the base ``learn`` loop."""
        acc = self.s3_acc
        for k, v in log.items():
            acc.setdefault(k, []).append(v)
        if (self.it + 1) % self.cfg["log_interval"] != 0:
            return
        row = {k: sum(float(x) for x in v) / len(v) for k, v in acc.items() if len(v)}
        for k in ("goal/episode_success", "goal/imagination_reward", "goal/imagination_survival", "goal/vfs_loss", "goal/stumble_loss"):
            row.setdefault(k, float("nan"))
        gt = self.goal_term
        goal = self.is_goal
        row["stage3/vfs_warmup"] = float(warm)
        row.update(self.stumble_eval())
        for t, name in enumerate(self.unwrapped.cfg.scene.arena.types):
            sel = goal & (gt.terrain_type == t)
            if sel.any():
                row[f"terrain_level/{name}"] = float(gt.level[sel].float().mean())
                row[f"terrain_success_ema/{name}"] = float(gt.success_ema[sel].mean())
        row["terrain_level/mean"] = float(gt.level[goal].float().mean())
        if self.writer is not None:
            for k, v in row.items():
                if not math.isnan(v):
                    self.writer.add_scalar(f"stage3/{k}" if not k.startswith(("terrain", "stage3")) else k, v, self.it + 1)
        print(f"  stage3 | warmup {int(warm)} | goal imag_r {row['goal/imagination_reward']:.3f} surv {row['goal/imagination_survival']:.3f} vfs {row['goal/vfs_loss']:.4f} "
              f"| level {row['terrain_level/mean']:.2f} | replay goal {self.replay.goal_size} ref {self.replay.ref_size}", flush=True)
        self.s3_acc = {}

    # ------------------------------------------------------------------------------------------------
    def save(self, path: str, infos=None) -> None:
        """``ACPRunner.save`` plus the Stage 3 state in the sidecar (Vfs, its optimizer, the return scales, the terrain curriculum)."""
        super().save(path, infos={"stage": 3, **(infos or {})})
        sidecar = self._sidecar(path)
        ck = torch.load(sidecar, map_location="cpu", weights_only=False)
        gt = self.goal_term
        ck.update({
            "critic_goal": self.critic_goal.state_dict(), "opt_vg": self.opt_vg.state_dict(), "ret_std": self.ret_std,
            "stumble_head": self.stumble_head.state_dict(), "opt_stumble": self.opt_stumble.state_dict(),
            "goal_level": gt.level.cpu(), "goal_success_ema": gt.success_ema.cpu(), "goal_terrain_type": gt.terrain_type.cpu(),
        })  # fmt: skip
        torch.save(ck, sidecar)

    def load(self, path: str):
        """``ACPRunner.load``; a Stage 2 checkpoint starts Stage 3 at iteration 0 when ``restart_iteration``."""
        infos = super().load(path)
        if self.cfg["restart_iteration"] and not (isinstance(infos, dict) and infos.get("stage") == 3):
            self.it = 0
        return infos

    def _load_stage3_sidecar(self, path: str) -> None:
        ck = torch.load(self._sidecar(path), map_location=self.device, weights_only=False)
        if "critic_goal" not in ck:  # Stage 2 sidecar: Vfs starts from scratch
            return
        self.critic_goal.load_state_dict(ck["critic_goal"]), self.opt_vg.load_state_dict(ck["opt_vg"])
        if "stumble_head" in ck:  # a Stage 3 checkpoint saved before the stumbling head existed: the head starts from scratch
            self.stumble_head.load_state_dict(ck["stumble_head"]), self.opt_stumble.load_state_dict(ck["opt_stumble"])
        self.ret_std = {k: (v.to(self.device) if v is not None else None) for k, v in ck["ret_std"].items()}
        gt = self.goal_term
        if torch.equal(ck["goal_terrain_type"].to(self.device), gt.terrain_type):  # same env layout: the curriculum continues
            gt.level, gt.success_ema = ck["goal_level"].to(self.device), ck["goal_success_ema"].to(self.device)
