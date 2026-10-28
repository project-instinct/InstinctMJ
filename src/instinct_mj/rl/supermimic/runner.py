"""SuperMimic training (Algorithm A.1): collect -> fit world model -> BPTT policy + critic update, single stage.

Adapted from the SuperMimic reference implementation (MIT License), Chong Zhang / ETH Zurich (``train.py``).

The simulator is the unchanged mjlab perceptive shadowing env (including the depth camera) wrapped by
``InstinctRlVecEnvWrapper`` (DR, resets, the task's BeyondMimic adaptive bin sampling, terminations and timeouts stay the
task's; the reference's return-based reset-bin sampler is therefore NOT used). Rewards / terminations in imagination are
the analytic heads of ``mdp.py``. The policy IS the task's own actor (``EncoderActorCritic`` with the Conv2d depth
encoder, built by ``instinct_rl.modules.build_actor_critic`` from the task's policy cfg) and acts on the task's ``policy`` observation group.
Vision is a function of the state, not part of it: the world model predicts only the robot state (and contacts). In
imagination the depth image of every predicted state is rendered differentiably against the terrain and robot meshes
(``render.py``: the camera pose from the predicted torso pose, ray casting with the implicit-function gradient), passed
through the camera's own noise pipeline (``DepthPipeline.noised``; its random stages are fresh samples) and fed to the
policy as in the env; the policy's depth encoder ``E`` is trained end-to-end by the BPTT policy loss (no latent, no
reconstruction loss, no stop-gradient on the visual path).

The runner replaces ``OnPolicyRunner`` as the task's ``runner_entry_point`` (``instinct-train`` / ``instinct-play`` use
``__init__``, ``learn``, ``save`` / ``load``, ``get_inference_policy``, ``export_as_onnx``, ``add_git_repo_to_log``); there
is no PPO anywhere.
"""

from __future__ import annotations

import copy
import math
import os
import time

import instinct_rl
import instinct_rl.modules as modules
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from instinct_rl.utils.utils import get_subobs_size, store_code_state
from tensorboardX import SummaryWriter

from . import networks
from .buffer import ReplayBuffer
from .depth import DepthPipeline
from .mdp import DEPTH_DIM, flatten_state, unflatten_state
from .networks import Critic, FailureHead, WorldModel
from .render import DepthRenderer
from .sim import SimReader, build_mdp

# per-component floors on the error normalization (pos, rot, lin vel, ang vel, q, qd increments)
ERR_FLOOR = torch.cat((torch.full((3,), 3e-3), torch.full((3,), 5e-3), torch.full((3,), 2e-2), torch.full((3,), 5e-2),
                       torch.full((29,), 3e-3), torch.full((29,), 5e-2)))  # fmt: skip
CONTACT_LOSS_WEIGHT = 0.1


class SuperMimicRunner:
    """Trainer / play-time policy holder of the task (registered as the task's ``runner_entry_point``).

    Construction is light and is all that ``instinct-play`` needs: the task's policy (``EncoderActorCritic`` built by
    ``instinct_rl.modules.build_actor_critic``), its optimizer and the observation normalizers, plus ``load`` /
    ``get_inference_policy`` / ``export_as_onnx``. The training-only components (analytic MDP heads, heightmap, world model,
    depth renderer, failure head, critic, replay buffer, ``torch.compile``) are built by :meth:`setup_training`, which ``learn``
    calls first; the sidecar of a resumed run is read there as well."""

    def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device: str = "cuda:0"):
        """``train_cfg`` = the task agent cfg ``to_dict()``: ``policy`` / ``normalizers`` (as for ``OnPolicyRunner``) plus the
        flat ``SuperMimicRunnerCfg`` hyperparameters."""
        self.cfg, self.env, self.log_dir, self.device = copy.deepcopy(train_cfg), env, log_dir, device
        c = self.cfg
        self.unwrapped = env.unwrapped
        policy_cfg = c["policy"]
        obs_format = env.get_obs_format()

        # ---- the task's policy, optimizer and observation normalizers (same construction as ``OnPolicyRunner``) ----
        self.ac = modules.build_actor_critic(
            policy_cfg.pop("class_name"), policy_cfg, obs_format, num_actions=env.num_actions, num_rewards=env.num_rewards
        ).to(device)
        self.opt_pi = torch.optim.AdamW(self.ac.parameters(), lr=c["lr"], betas=(0.9, 0.95), weight_decay=0.0)
        self.normalizers = {}
        for obs_group_name, config in c["normalizers"].items():
            config = config.copy()
            normalizer = modules.build_normalizer(
                input_shape=get_subobs_size(obs_format[obs_group_name]),
                normalizer_class_name=config.pop("class_name"),
                normalizer_kwargs=config,
            )
            normalizer.to(device)
            self.normalizers[obs_group_name] = normalizer
        self.pol_norm = self.normalizers["policy"]
        assert math.prod(obs_format["policy"]["depth_image"]) == DEPTH_DIM

        self.N = env.num_envs
        self.it = 0
        self.git_status_repos = [instinct_rl.__file__]  # files whose repo status is stored with the run
        self.resume_path: str | None = None  # set by ``load``: the sidecar is read by ``setup_training``
        self.training_ready = False
        self.writer = None
        _, _ = env.reset()

    def setup_training(self) -> None:
        """Build everything that only training needs (called by ``learn``). The simulator runs with ``auto_reset=False``
        from here on: finished envs are reset explicitly after their pre-reset data was read (see ``collect``)."""
        if dist.is_initialized():
            raise NotImplementedError("SuperMimicRunner trains in a single process (no distributed data parallel).")
        c, env, device = self.cfg, self.env, self.device
        self.unwrapped.cfg.auto_reset = False
        torch.manual_seed(c["seed"])
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        networks.AMP_DTYPE = torch.bfloat16 if c["amp"] == "bf16" else None
        self.pol_norm.eval()  # statistics are updated explicitly from collected real observations
        self.motion, self.kin, self.scanner, self.mdp = build_mdp(
            self.unwrapped, c["heightmap_resolution"], c["heightmap_cache_dir"], device
        )
        self.reader = SimReader(self.unwrapped)
        self.xi = self.reader.context(self.mdp)  # (N, xi_dim) episode-constant context of every env
        self.mdp.set_xi_stats(self.xi)
        self.xi_n = self.mdp.normalize_xi(self.xi)
        self.depth_pipe = DepthPipeline(self.unwrapped)
        self.renderer = DepthRenderer(self.unwrapped, c["pi_batch"], c["render_robot"], device)  # the camera's parent body is the root (torso_link)
        self.K = len(self.reader.contact_body_ids)
        self.hist, self.horizon, self.goal_dim = c["hist"], c["horizon"], self.mdp.goal_dim
        assert c["goal_frames"] == self.motion.goal_frames
        if not bool((self.motion.clip_len > self.horizon).any()):
            raise ValueError(f"No reference clip is longer than the imagination horizon ({self.horizon} frames).")
        assert c["warmup_steps"] >= max(self.horizon, c["wm_window"])

        self.wm = WorldModel(self.scanner.num_rays, self.K, self.mdp.xi_dim).to(device)  # state only, no visual latent
        self.critic = Critic(self.mdp.critic_dim()).to(device)
        self.wm_params = list(self.wm.parameters())
        self.opt_wm = torch.optim.Adam(self.wm_params, lr=c["lr"])
        self.opt_v = torch.optim.Adam(self.critic.parameters(), lr=c["lr"])
        self.fail_head = FailureHead(self.mdp.critic_dim()).to(device)
        self.opt_fail = torch.optim.Adam(self.fail_head.parameters(), lr=c["lr"])
        self.fail_head_start_it = 0  # iteration at which the head was created (a resume may create it later)
        self.err_floor = ERR_FLOOR.to(device)
        self.buffer = ReplayBuffer(c["buffer_steps"], self.N, self.K, device, c["wm_holdout_envs"])
        self.obs, _ = env.get_observations()  # real flat policy observation (post-reset)
        self.sim_steps, self.wall_offset = 0, 0.0
        self.writer = SummaryWriter(log_dir=self.log_dir) if self.log_dir is not None else None
        self._build_fns()
        if self.resume_path is not None:
            self._load_sidecar(self.resume_path)
        self.training_ready = True

    # ------------------------------------------------------------------------------------------------
    def _build_fns(self):
        mdp, wm = self.mdp, self.wm
        k = self.cfg["wm_window"]

        def _wm_rollout(s, a, xi_n):
            """k-step rollout from the real state: states and contact logits (B, k, ...)."""
            S = unflatten_state(s[:, 0])
            preds, logits = [], []
            for j in range(k):
                S, lg, _ = wm.step(S, a[:, j], xi_n, mdp.scan(S, noisy=False))
                preds.append(S), logits.append(lg)
            return tuple(torch.stack([P[i] for P in preds], 1) for i in range(6)), torch.stack(logits, 1)

        def _wm_err(S_hat, logits, s1, contact, mask):
            std = torch.maximum(wm.out_norm.std(), self.err_floor)
            e = ((wm.state_error(S_hat, unflatten_state(s1)) / std) ** 2).mean(-1)  # (B, k)
            bce = F.binary_cross_entropy_with_logits(logits, contact, reduction="none").sum(-1)
            B = s1.shape[0]
            return ((e + CONTACT_LOSS_WEIGHT * bce) * mask).sum() / B, (e[:, -1], (bce * mask).sum() / B)

        def _reward_term(S, f, origin, a, a_prev, xi, logits):
            r, terms = mdp.reward(S, f, origin, a, a_prev, xi, torch.sigmoid(logits), terms=True)
            return r, mdp.soft_termination(S, f, origin), terms

        def _critic_input(S, f, phase, origin, xi_n):
            return mdp.critic_features(S, f, origin, xi_n, mdp.goal(S, f, phase, noisy=False), mdp.scan(S, noisy=False))

        def _sim_stats(S, S1, a, a_prev, f1, origin1, xi, xi_n, contact):
            """Per collected transition: world-model input / target features (for the running normalizers) and the analytic
            reward of the real transition (``sim/analytic_reward_abs_gap``)."""
            return wm.inputs(S, a, xi_n, mdp.scan(S, noisy=False)), wm.targets(S, S1), mdp.reward(S1, f1, origin1, a, a_prev, xi, contact)

        ac, pol_norm, policy_autocast = self.ac, self.pol_norm, self.cfg["policy_autocast"]

        def _imag_obs(S, depth, o_hist, f, phase):
            """Normalized policy observation in the env's format with the rendered ``depth`` (B, DEPTH_DIM) of the state."""
            return pol_norm(mdp.policy_obs(o_hist, mdp.goal(S, f, phase), depth))

        def _imag_next(S, a, o_hist, xi, xi_n, scan):
            """World-model step, terrain scan and proprio history of the next state."""
            a = torch.clamp(a, -20, 20)
            S1, logits, _ = wm.step(S, a, xi_n, scan)
            o_hist1 = torch.cat((o_hist[:, 1:], mdp.add_noise(mdp.obs(S1, a, xi)).unsqueeze(1)), 1)
            return a, S1, logits, mdp.scan(S1, noisy=False), o_hist1

        def _imag_core(S, depth, o_hist, f, phase, xi, xi_n, scan):
            """One imagination step: the policy's own network (depth encoder E + actor) on the policy observation, then the
            world-model step. The gradient flows into E and into the state through the rendered image. The network runs
            eagerly (``instinct_rl``'s observation slicing graph-breaks under ``torch.compile`` into many fragments), the
            elementwise work before and after it is compiled."""
            obs = _imag_obs(S, depth, o_hist, f, phase)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=policy_autocast):  # training imagination only
                a = ac.act_inference(obs).float()
            return _imag_next(S, a, o_hist, xi, xi_n, scan)

        def _rays(p, R, env):
            return self.renderer.rays(p, R, env)

        def _depth(o, d, forward, dist, n, geomid):
            """Raw measurement from the hits, the sensor's noise pipeline, normalize + crop / resize: (B, DEPTH_DIM)."""
            return self.depth_pipe.noised(self.renderer.measure(o, d, forward, dist, n, geomid)).flatten(1)

        if self.cfg["compile"]:
            _wm_rollout, _wm_err = (torch.compile(fn) for fn in (_wm_rollout, _wm_err))
        if self.cfg["compile_policy"]:
            # level-batched FK keeps the graph small enough to compile (the former unrolled 30-body loop did not finish)
            _reward_term, _critic_input = (torch.compile(fn, dynamic=True) for fn in (_reward_term, _critic_input))
            _imag_obs, _imag_next, _sim_stats = torch.compile(_imag_obs), torch.compile(_imag_next), torch.compile(_sim_stats)
            _rays, _depth = torch.compile(_rays), torch.compile(_depth)  # around the (uncompiled) warp ray cast
        self.wm_rollout, self.wm_err, self.rays_fn, self.depth_fn = _wm_rollout, _wm_err, _rays, _depth
        self.reward_term, self.critic_input, self.imag_core, self.sim_stats = _reward_term, _critic_input, _imag_core, _sim_stats

    def rollout(self, w: dict, xi_n):
        """k-step world-model rollout of a window batch from the real states: predicted states and contact logits."""
        return self.wm_rollout(w["s"], w["a"], xi_n)

    def render_depth(self, S, env_idx) -> torch.Tensor:
        """The policy's depth observation (B, DEPTH_DIM) of the states ``S`` of envs ``env_idx``: the raw measurement of the
        camera rendered against the terrain meshes (differentiable w.r.t. the root pose), then the sensor's noise pipeline,
        normalization and crop / resize."""
        o, d, forward = self.rays_fn(S[0], S[1], env_idx)
        dist, n, geomid = self.renderer.scene.cast(S[0], S[1], S[4], o.detach(), d.detach())
        return self.depth_fn(o, d, forward, dist, n, geomid)

    def lr_at(self, it: int) -> float:
        c = self.cfg
        if it >= c["lr_decay_iters"]:
            return c["lr"] * c["lr_final_ratio"]
        return c["lr"] * (c["lr_final_ratio"] + (1 - c["lr_final_ratio"]) * 0.5 * (1 + math.cos(math.pi * it / c["lr_decay_iters"])))

    # ------------------------------------------------------------------------------------------------
    @torch.no_grad()
    def collect(self, n_steps: int, stats: dict):
        """Step the task env with the CURRENT policy on the real policy observation (real depth) + action noise.

        The env runs with ``auto_reset=False`` and finished envs are reset explicitly right after the pre-reset data
        (``s1``, the applied action, contacts, rewards, flags) has been read. This is the same sequence
        ``ManagerBasedRlEnv.step`` runs for ``auto_reset=True`` (rewards -> terminations -> ``_reset_idx`` ->
        ``write_data_to_sim`` -> ``forward`` -> commands with dt=0 for reset envs -> ``sense`` -> observations of the
        reset envs), because ``reset(env_ids)`` performs exactly those steps for the given envs; the only difference is
        that the finished envs' commands also advance once with ``dt`` before being reset, which the reset overwrites.
        """
        mdp, reader, env = self.mdp, self.reader, self.env
        for _ in range(n_steps):
            S = reader.state()
            f, phase, origin = self.motion.current_frame()
            a_prev = reader.last_action()
            ep_step = self.unwrapped.episode_length_buf.clone()
            self.pol_norm.update(self.obs)
            a = self.ac.act_inference(self.pol_norm(self.obs))
            a = torch.clamp(a + self.cfg["action_noise"] * torch.randn_like(a), -20, 20)

            obs1, rew, dones, _ = env.step(a)
            # ---- pre-reset data of this transition ----
            a = self.unwrapped.action_manager.action.clone()
            done = dones.bool()
            terminated = self.unwrapped.reset_terminated.clone()
            S1 = reader.state()
            contact = reader.contact()
            r_sim = rew.sum(-1).clone()
            done_ids = done.nonzero(as_tuple=False).squeeze(-1)
            f1, phase1, origin1 = self.motion.current_frame()
            self.obs = obs1
            if len(done_ids) > 0:
                self.unwrapped.reset(env_ids=done_ids)
                self.obs = torch.where(done[:, None], env.get_observations()[0], obs1)  # post-reset obs of reset envs
            self.buffer.add({
                "s": flatten_state(S), "s1": flatten_state(S1), "a": a, "a_prev": a_prev, "contact": contact,
                "origin": origin, "r": r_sim, "f": f, "phase": phase, "ep_step": ep_step.int(), "done": done,
                "f1": f1, "phase1": phase1, "origin1": origin1, "terminated": terminated,
            })  # fmt: skip
            self.sim_steps += self.N
            wm_in, wm_tgt, r_an = self.sim_stats(S, S1, a, a_prev, f1, origin1, self.xi, self.xi_n, contact)
            self.wm.in_norm.update(wm_in)
            self.wm.out_norm.update(wm_tgt)
            stats["r_gap"].append((r_an - r_sim).abs().mean() / mdp.dt)
            stats["r"].append(r_sim.mean() / mdp.dt)
            if len(done_ids) > 0:
                stats["ep_len"].append((ep_step[done] + 1).float().mean())
                stats["term"].append(terminated[done].float().sum())
                stats["done"].append(done.float().sum())

    def wm_update(self):
        """One world-model gradient step on a k-step window batch (dynamics, contact and failure losses)."""
        c = self.cfg
        w = self.buffer.sample_windows(c["wm_batch"], c["wm_window"], self.buffer.train_envs, c["wm_failure_frac"], c["wm_failure_horizon"])
        env_idx = w["env"]
        k = w["mask"].shape[1]
        S_hat, logits = self.rollout(w, self.xi_n[env_idx])
        loss, (e_last, bce) = self.wm_err(S_hat, logits, w["s1"], w["contact"], w["mask"])
        self.opt_wm.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.wm_params, 1.0)
        self.opt_wm.step()
        # failure ("continue") head on TRUE and (detached) WM-PREDICTED states of the windows, importance-weighted
        Bf = min(c["fail_windows"], w["mask"].shape[0])
        with torch.no_grad():
            flat_f = lambda x: x[:Bf].flatten(0, 1)  # noqa: E731
            xn = self.xi_n[env_idx[:Bf]].unsqueeze(1).expand(Bf, k, -1).flatten(0, 1)
            args_f = (flat_f(w["f1"]), flat_f(w["phase1"]), flat_f(w["origin1"]), xn)
            X_true = self.critic_input(unflatten_state(flat_f(w["s1"])), *args_f)
            X_pred = self.critic_input(tuple(flat_f(x) for x in S_hat), *args_f)
            self.fail_head.norm.update(X_true)
        label = flat_f(w["terminated"]).float()
        weight = (w["mask"][:Bf] * w["weight"][:Bf, None]).flatten()
        pos_weight = torch.where(label > 0.5, c["fail_pos_weight"], 1.0)
        head_loss = lambda X: (F.binary_cross_entropy_with_logits(self.fail_head(X), label, reduction="none") * pos_weight * weight).sum() / weight.sum().clamp(min=1e-6)  # noqa: E731
        fail_loss = head_loss(X_true) + head_loss(X_pred)
        self.opt_fail.zero_grad(set_to_none=True)
        fail_loss.backward()
        nn.utils.clip_grad_norm_(self.fail_head.parameters(), 1.0)
        self.opt_fail.step()
        m = w["mask"][:, -1]
        return {
            "fail_loss": fail_loss.detach() / 2, "fail_batch_pos": (label * w["mask"][:Bf].flatten()).sum() / w["mask"][:Bf].sum(),
            "wm": loss.detach() / k, "wm8": (e_last.detach() * m).sum() / m.sum().clamp(min=1), "bce": bce.detach(),
        }  # fmt: skip

    @torch.no_grad()
    def render_error(self) -> dict:
        """Renderer vs the live sensor on the current real states of all envs: mean absolute error of the clean processed
        image (normalized units) and the fraction of pixels off by more than 0.05."""
        S, ids = self.reader.state(), torch.arange(self.N, device=self.device)
        err = (self.depth_pipe.clean(self.renderer.render(S[0], S[1], S[4], ids)) - self.depth_pipe.clean_from_sensor()).abs()
        scene = self.renderer.scene  # in-place: the counters live in the CUDA graph of the cast
        counts = torch.stack((scene.dropped, scene.unresolved)).float()
        scene.dropped.zero_(), scene.unresolved.zero_()
        return {"render/clean_mae": err.mean(), "render/frac_err_gt_0.05": (err > 0.05).float().mean(),
                "render/skip_dropped_samples": counts[0], "render/skip_unresolved_rays": counts[1]}  # counts since the last log

    @torch.no_grad()
    def fail_calibration(self, num: int = 2048) -> dict:
        """Failure head vs actual REAL terminations on held-out windows (envs never used for training), on true states and
        on open-loop WM-predicted states: mean predicted vs actual rate, reliability bins, AUROC, per motion."""
        k = self.cfg["wm_window"]
        w = self.buffer.sample_windows(num, k, self.buffer.holdout_envs)
        env = w["env"]
        xn = self.xi_n[env].unsqueeze(1).expand(num, k, -1).flatten(0, 1)
        args_f = (w["f1"].flatten(), w["phase1"].flatten(), w["origin1"].flatten(0, 1), xn)
        S_hat, _ = self.rollout(w, self.xi_n[env])
        mask = w["mask"].flatten() > 0.5
        label = w["terminated"].flatten()[mask].float()
        out = {}
        motion = self.motion.clip_id[w["f1"].flatten()][mask]
        for name, S in (("true_states", unflatten_state(w["s1"].flatten(0, 1))), ("wm_states", tuple(x.flatten(0, 1) for x in S_hat))):
            p = torch.sigmoid(self.fail_head(self.critic_input(S, *args_f)))[mask]
            out[f"fail/{name}/pred_rate"], out[f"fail/{name}/true_rate"] = float(p.mean()), float(label.mean())
            for lo, hi in ((0.0, 0.02), (0.02, 0.1), (0.1, 0.5), (0.5, 1.01)):
                m = (p >= lo) & (p < hi)
                if m.any():
                    out[f"fail/{name}/bin_{lo}_{hi}/pred"], out[f"fail/{name}/bin_{lo}_{hi}/actual"] = float(p[m].mean()), float(label[m].mean())
            if 0 < label.sum() < len(label):
                order = p.argsort()
                ranks = torch.empty_like(p)
                ranks[order] = torch.arange(1, len(p) + 1, device=self.device, dtype=p.dtype)
                npos = label.sum()
                out[f"fail/{name}/auroc"] = float((ranks[label > 0.5].sum() - npos * (npos + 1) / 2) / (npos * (len(p) - npos)))
            if name == "true_states":
                for m_id in motion.unique().tolist():
                    sel = motion == m_id
                    out[f"fail/motion_{m_id}/pred_rate"], out[f"fail/motion_{m_id}/true_rate"] = float(p[sel].mean()), float(label[sel].mean())
        return out

    @torch.no_grad()
    def termination_check(self, num: int = 256) -> dict:
        """Imagination vs reality from the same held-out real start states: fraction of the starts that terminate within the
        horizon in imagination (analytic hard termination; and 1 - survival with the failure head) vs in the buffer."""
        H, c = self.horizon, self.cfg
        eligible = self.buffer.eligible_starts(
            lambda f: f + H <= self.motion.clip_end[self.motion.clip_id[f]], self.buffer.holdout_envs, max_age=self.buffer.size - H - 1
        )
        if len(eligible) < 8:
            return {}
        pick = eligible[torch.randint(0, len(eligible), (min(num, len(eligible)),), device=self.device)]
        s, h = self.buffer.starts_at(pick[:, 0], pick[:, 1], self.hist)
        im = self._imagine(s, h)
        B = len(pick)
        flat = lambda xs: torch.stack(xs).flatten(0, 1)  # noqa: E731
        rep = lambda x: x.unsqueeze(0).expand(H, *x.shape).flatten(0, 1)  # noqa: E731
        S_all = tuple(flat([St[i] for St in im["Ss"]]) for i in range(6))
        f_all = flat(im["Fs"])
        hard = self.mdp.hard_termination(S_all, f_all, rep(im["origin"])).view(H, B).any(0)
        soft = self.mdp.soft_termination(S_all, f_all, rep(im["origin"])).view(H, B)
        p_fail = torch.sigmoid(self.fail_head(self.critic_input(S_all, f_all, rep(im["phase"]), rep(im["origin"]), rep(im["xi_n"])))).view(H, B).clamp(max=c["fail_clip"])
        survive_with_head = ((1 - soft) * (1 - p_fail)).prod(0)
        # real outcome: a REAL termination within the next H transitions of the same episode
        rows = self.buffer._row(pick[:, 0].unsqueeze(-1) + torch.arange(H, device=self.device))
        env = pick[:, 1].unsqueeze(-1).expand_as(rows)
        done, term = self.buffer.data["done"][rows, env], self.buffer.data["terminated"][rows, env]
        same_episode = torch.cumsum(done.int(), 1) - done.int() == 0
        real = (term & same_episode).any(1)
        out = {"termcheck/imagined_hard": float(hard.float().mean()), "termcheck/imagined_soft_with_head": float((1 - survive_with_head).mean()),
               "termcheck/real": float(real.float().mean())}
        motion = self.motion.clip_id[s["f"]]
        for m_id in motion.unique().tolist():
            sel = motion == m_id
            out[f"termcheck/motion_{m_id}/imagined_hard"], out[f"termcheck/motion_{m_id}/real"] = float(hard[sel].float().mean()), float(real[sel].float().mean())
        return out

    def sample_valid_starts(self, B: int):
        """Uniform over all buffered states whose imagination stays inside the reference clip (``f + H <= clip end``)."""
        eligible = self.buffer.eligible_starts(lambda f: f + self.horizon <= self.motion.clip_end[self.motion.clip_id[f]])
        if len(eligible) == 0:
            raise RuntimeError("No buffered state has a full imagination horizon left inside its reference clip.")
        pick = eligible[torch.randint(0, len(eligible), (B,), device=self.device)]
        return self.buffer.starts_at(pick[:, 0], pick[:, 1], self.hist)

    def _imagine(self, s: dict, h: dict) -> dict:
        """H-step imagination of the policy from buffer start states ``s`` / histories ``h``: the world model rolls out the
        state, the depth image of every state is rendered. Gradients flow as in ``policy_update`` (call under ``no_grad``
        for diagnostics)."""
        c, mdp = self.cfg, self.mdp
        HZ = self.horizon
        S = unflatten_state(s["s"])
        f, phase, origin, a_prev, env_idx = s["f"], s["phase"], s["origin"], s["a_prev"], s["env"]
        xi, xi_n = self.xi[env_idx], self.xi_n[env_idx]
        with torch.no_grad():
            o_hist = mdp.obs(unflatten_state(h["s"]), h["a_prev"], self.xi[h["env"]].unsqueeze(1))
            # one noise sample per distinct historical observation (backfilled slots duplicate it, as in collection)
            noise = (2 * torch.rand_like(o_hist) - 1) * mdp.proprio_noise
            o_hist = o_hist + noise.gather(1, h["slot"].unsqueeze(-1).expand_as(noise))
        S0, f0, scan = S, f, mdp.scan(S, noisy=False)
        Ss, As, Aps, Fs, Ls = [], [], [], [], []
        for _ in range(HZ):
            Aps.append(a_prev)
            # policy observation in EXACTLY the env's policy-group format, with the rendered (noised) depth image of S
            step_args = (S, self.render_depth(S, env_idx), o_hist, f, phase, xi, xi_n, scan)
            if c["checkpoint_imagination"] and torch.is_grad_enabled():
                # recompute the step in the backward pass instead of keeping its activations (same RNG state is replayed)
                a_prev, S, logits, scan, o_hist = torch.utils.checkpoint.checkpoint(self.imag_core, *step_args, use_reentrant=False)
            else:
                a_prev, S, logits, scan, o_hist = self.imag_core(*step_args)
            f = f + 1
            Ss.append(S), As.append(a_prev), Fs.append(f), Ls.append(logits)
        return {"S0": S0, "f0": f0, "S": S, "f": f, "phase": phase, "origin": origin, "xi": xi, "xi_n": xi_n, "Ss": Ss, "As": As, "Aps": Aps, "Fs": Fs, "Ls": Ls}

    def policy_update(self):
        """One policy step: H-step imagination from buffer states, BPTT of the discounted return, then GAE critic fit."""
        c, mdp, dev = self.cfg, self.mdp, self.device
        B, HZ = c["pi_batch"], self.horizon
        s, h = self.sample_valid_starts(B)
        frozen = (self.wm, self.critic, self.fail_head)
        for m_ in frozen:
            for p_ in m_.parameters():
                p_.requires_grad_(False)
        im = self._imagine(s, h)
        S0, f0, S, f, phase, origin, xi, xi_n = im["S0"], im["f0"], im["S"], im["f"], im["phase"], im["origin"], im["xi"], im["xi_n"]
        Ss, As, Aps, Fs, Ls = im["Ss"], im["As"], im["Aps"], im["Fs"], im["Ls"]
        flat = lambda xs: torch.stack(xs).flatten(0, 1)  # noqa: E731  (H*B, ...)
        rep = lambda x: x.unsqueeze(0).expand(HZ, *x.shape).flatten(0, 1)  # noqa: E731
        S_all = tuple(flat([St[i] for St in Ss]) for i in range(6))
        r, d, terms = self.reward_term(S_all, flat(Fs), rep(origin), flat(As), flat(Aps), rep(xi), flat(Ls))
        p_fail_mean = torch.zeros((), device=dev)
        if c["fail_head"] and self.it >= self.fail_head_start_it + c["fail_head_warmup_iters"]:
            # learned failure ("continue") head, combined with the analytic soft termination; the gradient is kept
            p_fail = torch.sigmoid(self.fail_head(self.critic_input(S_all, flat(Fs), rep(phase), rep(origin), rep(xi_n)))).clamp(max=c["fail_clip"])
            d = 1 - (1 - d) * (1 - p_fail)
            p_fail_mean = p_fail.detach().mean()
        r, d = r.view(HZ, B), d.view(HZ, B)
        alive = torch.cumprod(1 - d, 0)  # c_{t+1}
        c_t = torch.cat((torch.ones(1, B, device=dev), alive[:-1]), 0)
        disc = c["gamma"] ** torch.arange(HZ, device=dev, dtype=torch.float32).unsqueeze(-1)
        J = (disc * c_t * r).sum(0)
        cT = alive[-1]
        r_sum = (c_t * r).detach().sum(0)
        rews, dones = r.detach(), d.detach()
        if not c["no_critic"]:
            J = J + (c["gamma"] ** HZ) * cT * self.critic(self.critic_input(S, f, phase, origin, xi_n))
        loss = -J.mean()
        self.opt_pi.zero_grad(set_to_none=True)
        loss.backward()
        gn = nn.utils.clip_grad_norm_(self.ac.parameters(), 1.0)
        nn.utils.clip_grad_value_(self.ac.parameters(), 0.1)
        if torch.isfinite(gn):
            self.opt_pi.step()
        for m_ in frozen:
            for p_ in m_.parameters():
                p_.requires_grad_(True)
        # ---------------- critic: GAE(lambda) targets over the imagined transitions ----------------
        v_loss = torch.zeros((), device=dev)
        if not c["no_critic"]:
            with torch.no_grad():
                S_c = tuple(torch.cat((S0[i].unsqueeze(0), torch.stack([St[i].detach() for St in Ss])), 0).flatten(0, 1) for i in range(6))
                f_c = torch.cat((f0.unsqueeze(0), torch.stack(Fs)), 0).flatten()
                rep1 = lambda x: x.unsqueeze(0).expand(HZ + 1, *x.shape).flatten(0, 1)  # noqa: E731
                X = self.critic_input(S_c, f_c, rep1(phase), rep1(origin), rep1(xi_n)).view(HZ + 1, B, -1)
            self.critic.norm.update(X)
            with torch.no_grad():
                V = self.critic(X)
                R = V[-1]
                targets = []
                for t in range(HZ - 1, -1, -1):
                    R = rews[t] + c["gamma"] * (1 - dones[t]) * ((1 - c["lam"]) * V[t + 1] + c["lam"] * R)
                    targets.append(R)
                targets = torch.stack(targets[::-1])  # (H, B)
            Xf, Yf = X[:-1].reshape(-1, X.shape[-1]), targets.reshape(-1)
            n = Xf.shape[0]
            mb = n // c["critic_minibatches"]
            for _ in range(c["critic_epochs"]):
                perm = torch.randperm(n, device=dev)
                for i in range(c["critic_minibatches"]):
                    idx = perm[i * mb : (i + 1) * mb]
                    v_loss = ((self.critic(Xf[idx]) - Yf[idx]) ** 2).mean()
                    self.opt_v.zero_grad(set_to_none=True)
                    v_loss.backward()
                    nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
                    self.opt_v.step()
        term_means = {k: v.detach().view(HZ, B).mean() for k, v in terms.items()}
        term_means["p_fail_mean"] = p_fail_mean
        return loss.detach(), v_loss.detach(), (r_sum / HZ / mdp.dt).mean(), gn.detach(), cT.mean().detach(), term_means

    # ------------------------------------------------------------------------------------------------
    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        """Train until iteration ``num_learning_iterations`` (total, so a resumed run continues). The env's random
        initial episode lengths are not used: resets come from the task's reference sampling."""
        if not self.training_ready:
            self.setup_training()
        if self.log_dir is not None:
            store_code_state(self.log_dir, self.git_status_repos)
        c = self.cfg
        stats = {"r": [], "r_gap": [], "ep_len": [], "term": [], "done": []}
        self.collect(c["warmup_steps"], stats)  # also refills the (unsaved) replay buffer after a resume
        t_start = time.time() - self.wall_offset
        t_log = time.time()
        step_acc = 0.0
        agg: dict[str, list] = {k: [] for k in ("wm", "wm8", "bce", "fail_loss", "fail_batch_pos", "pi", "v", "ir", "gn", "surv")}
        terms_acc: dict[str, list] = {}
        while self.it < num_learning_iterations:
            it = self.it
            lr = self.lr_at(it)
            for o in (self.opt_wm, self.opt_pi, self.opt_v, self.opt_fail):
                for gparam in o.param_groups:
                    gparam["lr"] = lr
            step_acc += 1.0 / c["utd"]
            n = int(step_acc)
            step_acc -= n
            self.collect(n, stats)
            for _ in range(c["wm_batches"]):
                m = self.wm_update()
            for k_ in ("wm", "wm8", "bce", "fail_loss", "fail_batch_pos"):
                agg[k_].append(m[k_])
            if it >= c["wm_only_iters"]:
                pl, vl, ir, gn, surv, terms = self.policy_update()
                agg["pi"].append(pl), agg["v"].append(vl), agg["ir"].append(ir), agg["gn"].append(gn), agg["surv"].append(surv)
                for k, v in terms.items():
                    terms_acc.setdefault(k, []).append(v)
            self.it += 1
            if self.it % c["log_interval"] == 0:
                mean = lambda x: float(sum(float(v) for v in x) / len(x)) if len(x) else float("nan")  # noqa: E731
                ips = c["log_interval"] / (time.time() - t_log)
                wall_h = (time.time() - t_start) / 3600
                term_frac = float(sum(float(v) for v in stats["term"]) / max(sum(float(v) for v in stats["done"]), 1))
                row = {
                    "wm/loss": mean(agg["wm"]), "wm/err_8step": mean(agg["wm8"]), "wm/contact_bce": mean(agg["bce"]),
                    "fail/bce_train": mean(agg["fail_loss"]), "fail/positive_share_in_batch": mean(agg["fail_batch_pos"]),
                    "policy/loss": mean(agg["pi"]), "policy/grad_norm": mean(agg["gn"]), "critic/value_loss": mean(agg["v"]),
                    "imagination/reward": mean(agg["ir"]), "imagination/survival": mean(agg["surv"]),
                    "sim/reward": mean(stats["r"]), "sim/analytic_reward_abs_gap": mean(stats["r_gap"]),
                    "sim/episode_length": mean(stats["ep_len"]), "sim/terminated_fraction": term_frac,
                    "perf/it_per_s": ips, "perf/lr": lr, "perf/sim_steps": float(self.sim_steps), "perf/wall_h": wall_h,
                    "perf/max_mem_gb": torch.cuda.max_memory_allocated() / 2**30,
                }  # fmt: skip
                row.update({k: float(v) for k, v in self.render_error().items()})
                if self.buffer.size >= self.horizon:
                    row.update(self.fail_calibration())
                    if self.it % (5 * c["log_interval"]) == 0:
                        row.update(self.termination_check())
                for k, v in terms_acc.items():
                    row[f"imagination/reward_term/{k}"] = mean(v)
                if self.writer is not None:
                    for k, v in row.items():
                        if not math.isnan(v):
                            self.writer.add_scalar(k, v, self.it)
                    self.writer.flush()
                print(f"it {self.it} | {wall_h:.3f} h | wm {row['wm/loss']:.4f} e8 {row['wm/err_8step']:.4f} bce {row['wm/contact_bce']:.4f} "
                      f"| render mae {row['render/clean_mae']:.4f} | pi {row['policy/loss']:.3f} v {row['critic/value_loss']:.4f} "
                      f"imag_r {row['imagination/reward']:.3f} surv {row['imagination/survival']:.3f} | sim_r {row['sim/reward']:.3f} "
                      f"gap {row['sim/analytic_reward_abs_gap']:.4f} ep_len {row['sim/episode_length']:.1f} term {term_frac:.2f} "
                      f"| {ips:.2f} it/s mem {row['perf/max_mem_gb']:.1f}G", flush=True)  # fmt: skip
                t_log = time.time()
                stats = {k: [] for k in stats}
                agg = {k: [] for k in agg}
                terms_acc = {}
            if self.it % c["save_interval"] == 0 or self.it == num_learning_iterations:
                self.wall_offset = time.time() - t_start
                self.save(os.path.join(self.log_dir, f"model_{self.it}.pt"))

    # ------------------------------------------------------------------------------------------------
    @staticmethod
    def _sidecar(path: str) -> str:
        """``.../model_<it>.pt`` -> ``.../supermimic_state_<it>.pt`` (does not match play's ``model_.*.pt`` lookup)."""
        d, name = os.path.split(path)
        return os.path.join(d, "supermimic_state_" + name[len("model_"):])

    # keys of the sidecar that belong to modules added after the first release (see ``_load_sidecar``)
    NEW_HEAD_KEYS = ("fail_head", "opt_fail", "fail_head_start_it")

    def add_git_repo_to_log(self, repo_path: str) -> None:
        """Also store the git status / diff of the repo containing ``repo_path`` with the run (``learn`` writes them)."""
        self.git_status_repos.append(repo_path)

    def save(self, path: str, infos=None) -> None:
        """``path``: ``OnPolicyRunner`` checkpoint format (``model_state_dict``, ``optimizer_state_dict``, the
        ``<group>_normalizer_state_dict`` of every normalizer, ``iter``, ``infos``): directly playable. Sidecar: world model,
        failure head, critic, their optimizers and the training state (resume)."""
        run_state_dict = {"model_state_dict": self.ac.state_dict(), "optimizer_state_dict": self.opt_pi.state_dict()}
        run_state_dict.update({f"{group}_normalizer_state_dict": n.state_dict() for group, n in self.normalizers.items()})
        run_state_dict.update({"iter": self.it, "infos": infos})
        torch.save(run_state_dict, path)
        torch.save({
            "wm": self.wm.state_dict(), "critic": self.critic.state_dict(),
            "opt_wm": self.opt_wm.state_dict(), "opt_v": self.opt_v.state_dict(), "iteration": self.it,
            "wall_time": self.wall_offset, "sim_steps": self.sim_steps, "cfg": self.cfg,
            "xi_mean": self.mdp.xi_mean, "xi_scale": self.mdp.xi_scale,
            "fail_head": self.fail_head.state_dict(), "opt_fail": self.opt_fail.state_dict(), "fail_head_start_it": self.fail_head_start_it,
        }, self._sidecar(path))  # fmt: skip

    def load(self, path: str):
        """Load the policy (actor-critic, its optimizer, the normalizers, the iteration) from an ``OnPolicyRunner``-format
        checkpoint: all that play needs. The sidecar of a SuperMimic run is NOT read here but by ``setup_training``
        (``learn``), so loading for play stays light."""
        loaded_dict = torch.load(path, weights_only=True)
        self.ac.load_state_dict(loaded_dict["model_state_dict"])
        self.opt_pi.load_state_dict(loaded_dict["optimizer_state_dict"])
        for group_name, normalizer in self.normalizers.items():
            normalizer.load_state_dict(loaded_dict[f"{group_name}_normalizer_state_dict"])
        self.it = loaded_dict["iter"]
        self.resume_path = path
        return loaded_dict["infos"]

    def _load_sidecar(self, path: str) -> None:
        """Resume the training state from the sidecar of the checkpoint loaded by ``load``. Existing modules must load
        strictly (missing keys error). A sidecar written before the failure head existed has exactly ``NEW_HEAD_KEYS``
        missing: that is only accepted with ``resume_init_new_heads`` (``--agent.resume-init-new-heads``), in which case the
        head is initialized fresh (and warmed up for ``fail_head_warmup_iters`` iterations from the loaded iteration before it
        affects the policy)."""
        ck = torch.load(self._sidecar(path), map_location=self.device, weights_only=False)
        self.wm.load_state_dict(ck["wm"]), self.critic.load_state_dict(ck["critic"])
        self.opt_wm.load_state_dict(ck["opt_wm"]), self.opt_v.load_state_dict(ck["opt_v"])
        self.wall_offset, self.sim_steps = ck["wall_time"], ck["sim_steps"]
        assert self.it == ck["iteration"]
        self.mdp.xi_mean, self.mdp.xi_scale = ck["xi_mean"].to(self.device), ck["xi_scale"].to(self.device)
        self.xi_n = self.mdp.normalize_xi(self.xi)
        missing = [k for k in self.NEW_HEAD_KEYS if k not in ck]
        if missing:
            if len(missing) != len(self.NEW_HEAD_KEYS):
                raise KeyError(f"sidecar {self._sidecar(path)} has only part of the failure-head keys; missing {missing}")
            if not self.cfg["resume_init_new_heads"]:
                raise KeyError(f"sidecar {self._sidecar(path)} has no failure head ({list(self.NEW_HEAD_KEYS)}); set --agent.resume-init-new-heads to create it fresh")
            self.fail_head_start_it = self.it
            print(f"[INFO] sidecar has no failure head: initialized it fresh at iteration {self.it} "
                  f"(warmup {self.cfg['fail_head_warmup_iters']} iterations before it affects the policy).")
        else:
            self.fail_head.load_state_dict(ck["fail_head"]), self.opt_fail.load_state_dict(ck["opt_fail"])
            self.fail_head_start_it = ck["fail_head_start_it"]

    def train_mode(self) -> None:
        """Training mode of the policy and the normalizers (``OnPolicyRunner.train_mode``)."""
        self.ac.train()
        for normalizer in self.normalizers.values():
            normalizer.train()

    def eval_mode(self) -> None:
        """Evaluation mode of the policy and the normalizers (``OnPolicyRunner.eval_mode``)."""
        self.ac.eval()
        for normalizer in self.normalizers.values():
            normalizer.eval()

    def get_inference_policy(self, device=None):
        """Observation (flat ``policy`` group) -> action, as ``OnPolicyRunner.get_inference_policy``."""
        self.eval_mode()
        if device is not None:
            self.ac.to(device)
        self.normalizers["policy"].to(device)
        return lambda x: self.ac.act_inference(self.normalizers["policy"](x))

    def export_as_onnx(self, obs, export_model_dir: str) -> None:
        """``0-depth_image.onnx`` + ``actor.onnx`` (``ac.export_as_onnx``) and ``policy_normalizer.npz`` in ``export_model_dir``."""
        self.eval_mode()
        obs = self.normalizers["policy"](obs)
        self.normalizers["policy"].export(os.path.join(export_model_dir, "policy_normalizer.npz"))
        self.ac.export_as_onnx(obs, export_model_dir)
