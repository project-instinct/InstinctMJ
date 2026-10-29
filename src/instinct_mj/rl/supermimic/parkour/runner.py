"""SuperMimic training of the parkour task (``Instinct-Parkour-Target-Amp-SuperMimic-G1-v0``): collect -> fit world model
(+ discriminator) -> BPTT policy + critic update, single stage, no PPO.

The structure follows ``..runner.SuperMimicRunner`` (perceptive shadowing). Differences of the task:

- The policy (the task's ``EncoderMoEActorCritic``) observes 8-frame histories of proprioception and of the velocity command,
  and a stack of 8 depth frames. The depth stack is encoded by the policy's encoder E into the visual latent z, which is a
  state of the world model (as for perceptive shadowing); E is trained by the reconstruction of the stack.
- The velocity command is a function of the state and of the command parameters (target point, speed limits), recomputed
  at every imagined state by ``ParkourMDP.velocity_command``.
- The world model also takes the previous action and the per-joint actuation delay (sampled per episode) as context.
- The feet air / contact timers of the ``feet_air_time`` reward are carried through the imagination with the predicted
  feet contacts.
- The adversarial style reward of the task's WASABI (``StyleReward``, trained on real windows only) is added to the analytic
  task reward and maximized by the same BPTT, on the imagined ``amp_policy`` windows.
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

from .. import networks
from ..buffer import ReplayBuffer
from ..mdp import STATE_DIM, flatten_state, unflatten_state
from ..networks import Critic, FailureHead, LatentDecoder, WorldModel
from ..style import StyleReward
from .mdp import AMP_HIST, DEPTH_SHAPE, HIST, NUM_J, XI_DIM, build_parkour_mdp
from .sim import CMD_DIM, ParkourSimReader

# per-component floors on the error normalization (pos, rot, lin vel, ang vel, q, qd increments)
ERR_FLOOR = torch.cat((torch.full((3,), 3e-3), torch.full((3,), 5e-3), torch.full((3,), 2e-2), torch.full((3,), 5e-2),
                       torch.full((29,), 3e-3), torch.full((29,), 5e-2)))  # fmt: skip
CONTACT_LOSS_WEIGHT = 0.1
LATENT_GAP_STEPS = (1, 8, 16, 32)
DEPTH_SCALE = 255.0  # the depth stacks are buffered as uint8 (the processed image is in [0, 1])


class ParkourSuperMimicRunner:
    """Trainer / play-time policy holder of the parkour SuperMimic task (the task's ``runner_entry_point``). Construction is
    light (play needs only the policy, its optimizer, the normalizers, ``load`` / ``get_inference_policy`` /
    ``export_as_onnx``); the training components are built by :meth:`setup_training`."""

    def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device: str = "cuda:0"):
        self.cfg, self.env, self.log_dir, self.device = copy.deepcopy(train_cfg), env, log_dir, device
        c = self.cfg
        self.unwrapped = env.unwrapped
        policy_cfg = c["policy"]
        self.obs_format = obs_format = env.get_obs_format()
        self.ac = modules.build_actor_critic(
            policy_cfg.pop("class_name"), policy_cfg, obs_format, num_actions=env.num_actions, num_rewards=env.num_rewards
        ).to(device)
        self.opt_pi = torch.optim.AdamW(self.ac.parameters(), lr=c["lr"], betas=(0.9, 0.95), weight_decay=0.0)
        self.normalizers = {}
        for obs_group_name, config in (c.get("normalizers") or {}).items():
            config = config.copy()
            normalizer = modules.build_normalizer(
                input_shape=get_subobs_size(obs_format[obs_group_name]),
                normalizer_class_name=config.pop("class_name"),
                normalizer_kwargs=config,
            )
            normalizer.to(device)
            self.normalizers[obs_group_name] = normalizer
        self.pol_norm = self.normalizers.get("policy")  # the task trains without observation normalization
        offset = 0
        for name, shape in obs_format["policy"].items():
            if name == "depth_image":
                break
            offset += math.prod(shape)
        self.depth_dim = math.prod(DEPTH_SHAPE)
        self.depth_slice = slice(offset, offset + self.depth_dim)
        assert tuple(obs_format["policy"]["depth_image"]) == DEPTH_SHAPE
        # the policy's depth encoder E: the encoder block of the depth_image term
        blocks = self.ac.encoders._parallel_blocks
        cfgs = self.ac.encoders.block_configs
        name = next(n for n in blocks if "depth_image" in cfgs[n]["component_names"])
        self.enc, self.z_dim = blocks[name], blocks[name].output_size
        # the actor MLP input is the encoder output segment: every non-image term in order, then the depth latent(s)
        segments = list(self.ac.encoders.output_segment)
        assert segments[: len(obs_format["policy"]) - 1] == [n for n in obs_format["policy"] if n != "depth_image"]
        assert len(segments) == len(obs_format["policy"]) and segments[-1].endswith(name)
        self.N = env.num_envs
        self.it = 0
        self.git_status_repos = [instinct_rl.__file__]
        self.resume_path: str | None = None
        self.training_ready = False
        self.writer = None
        _, _ = env.reset()

    def normalize_policy_obs(self, obs: torch.Tensor) -> torch.Tensor:
        return self.pol_norm(obs) if self.pol_norm is not None else obs

    def start_latent(self, depth: torch.Tensor):
        """Latent of the start depth stack for the world model (detached) and for the actor at the first imagination step
        (with the gradient into E scaled by ``latent_policy_grad``, same value)."""
        coef = self.cfg["latent_policy_grad"]
        with torch.set_grad_enabled(coef > 0 and torch.is_grad_enabled()):
            z0 = self.encode_depth(depth)
        z = z0.detach()
        return z, (z + coef * (z0 - z) if coef > 0 else z)

    def encode_depth(self, depth: torch.Tensor) -> torch.Tensor:
        """Latent of the policy's depth encoder for depth stacks (B, *DEPTH_SHAPE) in [0, 1], normalized as in the policy
        observation (the encoder sees exactly what it sees in ``act_inference``)."""
        x = depth.flatten(1)
        if self.pol_norm is not None:
            n = self.pol_norm
            x = (x - n._mean[0, self.depth_slice]) / (n._std[0, self.depth_slice] + n.eps)
        return self.enc(x.view(-1, *DEPTH_SHAPE))

    # ------------------------------------------------------------------------------------------------
    def setup_training(self) -> None:
        if dist.is_initialized():
            raise NotImplementedError("ParkourSuperMimicRunner trains in a single process.")
        c, env, device = self.cfg, self.env, self.device
        self.unwrapped.cfg.auto_reset = False
        torch.manual_seed(c["seed"])
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        networks.AMP_DTYPE = torch.bfloat16 if c["amp"] == "bf16" else None
        if self.pol_norm is not None:
            self.pol_norm.eval()
        self.kin, self.scanner, self.mdp = build_parkour_mdp(self.unwrapped, c["heightmap_resolution"], c["heightmap_cache_dir"], device)
        self.reader = ParkourSimReader(self.unwrapped)
        self.K = self.reader.num_contacts
        self.horizon, self.hist = c["horizon"], max(HIST, AMP_HIST)
        friction = self.reader.friction()
        self.fr_mean, self.fr_std = friction.mean(0), friction.std(0).clamp(min=1e-3)
        self.fr_n = (friction - self.fr_mean) / self.fr_std  # (N, 1) normalized friction (startup randomization)
        ctx_dim = XI_DIM + NUM_J  # world-model context: [episode context xi (lags, friction), previous action]
        self.wm = WorldModel(self.scanner.num_rays, self.K, ctx_dim, self.z_dim).to(device)
        self.latent_dec = LatentDecoder(self.z_dim, channels=DEPTH_SHAPE[0]).to(device)
        self.critic = Critic(self.mdp.critic_dim).to(device)
        self.wm_params = [*self.wm.parameters(), *self.enc.parameters(), *self.latent_dec.parameters()]
        self.opt_wm = torch.optim.Adam(self.wm_params, lr=c["lr"])
        self.opt_v = torch.optim.Adam(self.critic.parameters(), lr=c["lr"])
        self.fail_head = FailureHead(self.mdp.critic_dim).to(device)
        self.opt_fail = torch.optim.Adam(self.fail_head.parameters(), lr=c["lr"])
        self.fail_head_start_it = 0
        self.style = StyleReward(
            self.obs_format["amp_policy"], device, c["style_discriminator_kwargs"], "AdamW", c["style_optimizer_kwargs"],
            c["style_reward_coef"], c["style_reward_type"], c["style_loss_func"], 1.0, c["style_gradient_penalty_coef"],
            c["style_weight_decay_coef"], c["style_logit_weight_decay_coef"],
        )  # fmt: skip
        self.err_floor = ERR_FLOOR.to(device)
        fields = {
            "s": (STATE_DIM, torch.float32), "s1": (STATE_DIM, torch.float32),
            "a": (NUM_J, torch.float32), "a_prev": (NUM_J, torch.float32),
            "contact": (self.K, torch.float32),  # contact indicators at s1 (see sim.py)
            "cmd": (CMD_DIM, torch.float32),  # command parameters at s
            "time_left": ((), torch.float32),  # time until the command is resampled, at s
            "lags": (NUM_J, torch.float32),  # per-joint actuation delay of the episode
            "timers": (4, torch.float32), "timers1": (4, torch.float32),  # feet air / contact timers at s / s1
            "depth": (DEPTH_SHAPE, torch.uint8),  # the env's policy depth stack at s
            "depth1": (DEPTH_SHAPE, torch.uint8),  # ... at s1 (pre-reset): latent target / reconstruction input
            "r": ((), torch.float32), "ep_step": ((), torch.int32), "done": ((), torch.bool), "terminated": ((), torch.bool),
        }  # fmt: skip
        self.buffer = ReplayBuffer(c["buffer_steps"], self.N, self.K, device, c["wm_holdout_envs"], fields=fields)
        self.buffer.window_keys = ("s", "s1", "a", "a_prev", "contact", "cmd", "lags", "timers", "timers1", "depth", "depth1", "done", "terminated")
        self.buffer.start_keys = ("s", "a_prev", "cmd", "lags", "timers", "depth")
        self.buffer.hist_keys = ("s", "a_prev", "cmd")
        amp_dim = get_subobs_size(self.obs_format["amp_reference"])
        self.amp_ref = torch.zeros(c["style_reference_steps"], self.N, amp_dim, dtype=torch.float16, device=device)
        self.amp_ref_ptr, self.amp_ref_size = 0, 0
        self.obs, extras = env.get_observations()
        self.sim_steps, self.wall_offset = 0, 0.0
        self.writer = SummaryWriter(log_dir=self.log_dir) if self.log_dir is not None else None
        self._build_fns()
        if self.resume_path is not None:
            self._load_sidecar(self.resume_path)
        self.training_ready = True

    @staticmethod
    def episode_context(lags, fr_n):
        """Episode context xi (``mdp.XI_DIM``): per-joint actuation lag (sampled per episode) and normalized friction."""
        return torch.cat((lags, fr_n), -1)

    @staticmethod
    def wm_context(xi, a_prev):
        """World-model context: episode context xi and the previous action (actuation delay)."""
        return torch.cat((xi, a_prev), -1)

    def _build_fns(self):
        mdp, wm, k = self.mdp, self.wm, self.cfg["wm_window"]

        def _wm_rollout(s, a, ctx, z0):
            """k-step rollout from the real state and the initial latent: states, contact logits, latents (B, k, ...)."""
            S, z = unflatten_state(s[:, 0]), z0
            preds, logits, zs = [], [], []
            for j in range(k):
                S, lg, z = wm.step(S, a[:, j], ctx[:, j], mdp.scan(S), z)
                preds.append(S), logits.append(lg), zs.append(z)
            return tuple(torch.stack([P[i] for P in preds], 1) for i in range(6)), torch.stack(logits, 1), torch.stack(zs, 1)

        def _wm_err(S_hat, logits, s1, contact, mask):
            std = torch.maximum(wm.out_norm.std(), self.err_floor)
            e = ((wm.state_error(S_hat, unflatten_state(s1)) / std) ** 2).mean(-1)  # (B, k)
            bce = F.binary_cross_entropy_with_logits(logits, contact, reduction="none").sum(-1)
            B = s1.shape[0]
            return ((e + CONTACT_LOSS_WEIGHT * bce) * mask).sum() / B, (e[:, -1], (bce * mask).sum() / B)

        def _reward_term(S0, S1, a, a_prev, cmd, timers1, logits):
            contact = torch.sigmoid(logits)
            r, terms = mdp.reward(S0, S1, a, a_prev, cmd, timers1, contact, terms=True)
            d = mdp.soft_termination(S1, cmd, contact)
            # ``is_alive`` is paid only if the step does not terminate
            r = r - terms["is_alive"] * d
            return r, d, terms

        def _critic_input(S, cmd, timers, xi_n):
            return mdp.critic_features(S, cmd, timers, xi_n)

        def _sim_stats(S, S1, a, a_prev, ctx, cmd, timers1, contact):
            return wm.inputs(S, a, ctx, mdp.scan(S)), wm.targets(S, S1), mdp.reward(S, S1, a, a_prev, cmd, timers1, contact)

        ac, policy_autocast, d0 = self.ac, self.cfg["policy_autocast"], self.depth_slice.start
        normalize = self.normalize_policy_obs

        def _imag_core(S, z, z_pi, o_hist, amp_hist, timers, a_prev, cmd, xi, scan):
            """One imagination step: actor on the policy observation with the latent in place of the encoder output
            (``z_pi``: detached, except at the first step with ``latent_policy_grad``), world-model step (z is a world-model
            state), feet timers, proprio and amp histories."""
            obs = normalize(mdp.policy_obs(o_hist, z.new_zeros(z.shape[0], *DEPTH_SHAPE)))
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=policy_autocast):
                a = ac.actor(torch.cat((obs[:, :d0], obs[:, d0 + self.depth_dim :], z_pi), -1)).float()
            a = torch.clamp(a, -20, 20)
            S1, logits, z1 = wm.step(S, a, torch.cat((xi, a_prev), -1), scan, z)
            timers1 = mdp.advance_feet_timers(timers, torch.sigmoid(logits[:, :2]))
            o_hist1 = torch.cat((o_hist[:, 1:], mdp.add_proprio_noise(mdp.proprio(S1, a, cmd)).unsqueeze(1)), 1)
            amp_hist1 = torch.cat((amp_hist[:, 1:], mdp.amp_frame(S1).unsqueeze(1)), 1)
            return a, S1, z1, logits, timers1, o_hist1, amp_hist1, mdp.scan(S1)

        if self.cfg["compile"]:
            _wm_rollout, _wm_err = (torch.compile(fn) for fn in (_wm_rollout, _wm_err))
        if self.cfg["compile_policy"]:
            _reward_term, _critic_input = (torch.compile(fn, dynamic=True) for fn in (_reward_term, _critic_input))
            _imag_core, _sim_stats = torch.compile(_imag_core), torch.compile(_sim_stats)
        self.wm_rollout, self.wm_err = _wm_rollout, _wm_err
        self.reward_term, self.critic_input, self.imag_core, self.sim_stats = _reward_term, _critic_input, _imag_core, _sim_stats

    def lr_at(self, it: int) -> float:
        c = self.cfg
        if it >= c["lr_decay_iters"]:
            return c["lr"] * c["lr_final_ratio"]
        return c["lr"] * (c["lr_final_ratio"] + (1 - c["lr_final_ratio"]) * 0.5 * (1 + math.cos(math.pi * it / c["lr_decay_iters"])))

    # ------------------------------------------------------------------------------------------------
    @torch.no_grad()
    def collect(self, n_steps: int, stats: dict):
        """Step the env with the current policy (+ action noise); the env runs with ``auto_reset=False`` and finished envs
        are reset explicitly after the pre-reset data was read (see ``SuperMimicRunner.collect``)."""
        mdp, reader, env = self.mdp, self.reader, self.env
        for _ in range(n_steps):
            S = reader.state()
            a_prev = reader.last_action()
            cmd, time_left = reader.command_params(), reader.command_time_left()
            lags, timers = reader.actuator_lags(), reader.feet_timers()
            ep_step = self.unwrapped.episode_length_buf.clone()
            depth = self.obs[:, self.depth_slice].view(self.N, *DEPTH_SHAPE)
            if self.pol_norm is not None:
                self.pol_norm.update(self.obs)
            a = self.ac.act_inference(self.normalize_policy_obs(self.obs))
            a = torch.clamp(a + self.cfg["action_noise"] * torch.randn_like(a), -20, 20)

            obs1, rew, dones, infos = env.step(a)
            # ---- pre-reset data of this transition ----
            a = self.unwrapped.action_manager.action.clone()
            done = dones.bool()
            terminated = self.unwrapped.reset_terminated.clone()
            S1, contact, timers1 = reader.state(), reader.contact(), reader.feet_timers()
            r_sim = rew.sum(-1).clone() if rew.dim() > 1 else rew.clone()
            depth1 = (obs1[:, self.depth_slice].view(self.N, *DEPTH_SHAPE) * DEPTH_SCALE).round().to(torch.uint8)
            self.amp_ref[self.amp_ref_ptr] = infos["observations"]["amp_reference"].half()
            self.amp_ref_ptr = (self.amp_ref_ptr + 1) % self.amp_ref.shape[0]
            self.amp_ref_size = min(self.amp_ref_size + 1, self.amp_ref.shape[0])
            done_ids = done.nonzero(as_tuple=False).squeeze(-1)
            self.obs = obs1
            if len(done_ids) > 0:
                self.unwrapped.reset(env_ids=done_ids)
                self.obs = torch.where(done[:, None], env.get_observations()[0], obs1)
            self.buffer.add({
                "s": flatten_state(S), "s1": flatten_state(S1), "a": a, "a_prev": a_prev, "contact": contact, "cmd": cmd,
                "time_left": time_left, "lags": lags, "timers": timers, "timers1": timers1,
                "depth": (depth * DEPTH_SCALE).round().to(torch.uint8), "depth1": depth1,
                "r": r_sim, "ep_step": ep_step.int(), "done": done, "terminated": terminated,
            })  # fmt: skip
            self.sim_steps += self.N
            ctx = self.wm_context(self.episode_context(lags, self.fr_n), a_prev)
            wm_in, wm_tgt, r_an = self.sim_stats(S, S1, a, a_prev, ctx, cmd, timers1, contact)
            self.wm.in_norm.update(wm_in)
            self.wm.out_norm.update(wm_tgt)
            r_an = r_an + mdp.volume_points_penetration(S1)
            stats["r_gap"].append((r_an - r_sim).abs().mean() / mdp.dt)
            stats["r"].append(r_sim.mean() / mdp.dt)
            if len(done_ids) > 0:
                stats["ep_len"].append((ep_step[done] + 1).float().mean())
                stats["term"].append(terminated[done].float().sum())
                stats["done"].append(done.float().sum())

    def wm_update(self):
        """One world-model step on a k-step window batch (dynamics, contact, latent, reconstruction; failure head)."""
        c = self.cfg
        w = self.buffer.sample_windows(c["wm_batch"], c["wm_window"], self.buffer.train_envs, c["wm_failure_frac"], c["wm_failure_horizon"])
        env_idx = w["env"]
        Bz, k = min(c["latent_windows"], w["mask"].shape[0]), w["mask"].shape[1]
        xi_w = self.episode_context(w["lags"], self.fr_n[env_idx].unsqueeze(1).expand(-1, k, -1))  # (B, k, XI_DIM)
        ctx = self.wm_context(xi_w, w["a_prev"])
        with torch.no_grad():
            z0 = self.encode_depth(w["depth"][:, 0].float() / DEPTH_SCALE)
            z_tgt = self.encode_depth(w["depth1"][:Bz].flatten(0, 1).float() / DEPTH_SCALE).view(Bz, k, -1)
            # before the rollout: the latent statistics are used (in place updated) inside its graph
            self.wm.update_z_stats(z_tgt, z_tgt - torch.cat((z0[:Bz].unsqueeze(1), z_tgt[:, :-1]), 1), w["mask"][:Bz])
        S_hat, logits, Z_hat = self.wm_rollout(w["s"], w["a"], ctx, z0)
        loss, (e_last, bce) = self.wm_err(S_hat, logits, w["s1"], w["contact"], w["mask"])
        mask = w["mask"][:Bz]
        z_scale = z_tgt.var((0, 1)).mean().clamp(min=1e-6)
        lz = ((Z_hat[:Bz] - z_tgt) ** 2).mean(-1) / z_scale
        latent_loss = (lz * mask).sum() / Bz
        Br = min(c["latent_recon_images"], Bz * k)
        stack = w["depth1"][:Bz].flatten(0, 1)[:Br].float() / DEPTH_SCALE
        recon = (self.latent_dec(self.encode_depth(stack)) - stack).abs().mean()
        total = loss + c["latent_weight"] * latent_loss + c["latent_recon_weight"] * recon
        self.opt_wm.zero_grad(set_to_none=True)
        total.backward()
        nn.utils.clip_grad_norm_(self.wm_params, 1.0)
        self.opt_wm.step()
        # failure ("continue") head on true and (detached) predicted states, importance weighted
        Bf = min(c["fail_windows"], w["mask"].shape[0])
        flat_f = lambda x: x[:Bf].flatten(0, 1)  # noqa: E731
        with torch.no_grad():
            xn = flat_f(xi_w)
            cmd_f, timers_f = flat_f(w["cmd"]), flat_f(w["timers1"])
            X_true = self.critic_input(unflatten_state(flat_f(w["s1"])), cmd_f, timers_f, xn)
            X_pred = self.critic_input(tuple(flat_f(x) for x in S_hat), cmd_f, timers_f, xn)
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
            "wm": loss.detach() / k, "wm8": (e_last.detach() * m).sum() / m.sum().clamp(min=1), "bce": bce.detach(),
            "latent": (lz.detach() * mask).sum() / mask.sum().clamp(min=1),
            "latent_k": (lz.detach()[:, -1] * mask[:, -1]).sum() / mask[:, -1].sum().clamp(min=1), "recon": recon.detach(),
            "fail_loss": fail_loss.detach() / 2,
        }  # fmt: skip

    def history(self, h: dict):
        """Proprio (HIST, noised like the env, one sample per distinct backfilled observation) and amp (AMP_HIST) histories
        of buffered starts, from the buffered states / actions / commands (oldest first)."""
        mdp = self.mdp
        S_h = unflatten_state(h["s"])
        o = mdp.proprio(S_h, h["a_prev"], h["cmd"])
        noise = mdp.add_proprio_noise(torch.zeros_like(o))
        o = o + noise.gather(1, h["slot"].unsqueeze(-1).expand_as(noise))
        return o[:, -HIST:], mdp.amp_frame(S_h)[:, -AMP_HIST:]

    def style_update(self) -> dict:
        """Discriminator steps: recent real policy windows (rebuilt from the buffered states) vs the env's reference
        windows."""
        c, buf = self.cfg, self.buffer
        out = {}
        if self.amp_ref_size == 0:
            return out
        recent = min(c["style_recent_steps"], buf.size)
        for _ in range(c["style_minibatches"]):
            B = c["style_batch"]
            age = buf.size - 1 - torch.randint(0, recent, (B,), device=self.device)
            env = torch.randint(0, self.N, (B,), device=self.device)
            _, h = buf.starts_at(age, env, AMP_HIST)
            with torch.no_grad():
                policy_windows = self.mdp.amp_window(self.mdp.amp_frame(unflatten_state(h["s"])))
            rows = (self.amp_ref_ptr - 1 - torch.randint(0, self.amp_ref_size, (B,), device=self.device)) % self.amp_ref.shape[0]
            reference_windows = self.amp_ref[rows, torch.randint(0, self.N, (B,), device=self.device)].float()
            out = self.style.update(policy_windows, reference_windows)
        return out

    def sample_valid_starts(self, B: int):
        """Buffered states whose command stays for the whole imagination (no target resampling inside)."""
        need = (self.horizon + self.cfg["command_margin_steps"]) * self.mdp.dt
        eligible = self.buffer.eligible_starts(lambda t: t > need, key="time_left", max_age=self.buffer.size - 1)
        if len(eligible) == 0:
            raise RuntimeError("No buffered state has a command that stays for the imagination horizon.")
        pick = eligible[torch.randint(0, len(eligible), (B,), device=self.device)]
        return self.buffer.starts_at(pick[:, 0], pick[:, 1], self.hist)

    def _imagine(self, s: dict, h: dict) -> dict:
        c = self.cfg
        HZ = self.horizon
        S = unflatten_state(s["s"])
        cmd, env_idx = s["cmd"], s["env"]
        xi_n = self.episode_context(s["lags"], self.fr_n[env_idx])
        z, z_pi = self.start_latent(s["depth"].float() / DEPTH_SCALE)
        with torch.no_grad():
            o_hist, amp_hist = self.history(h)
        timers, a_prev = s["timers"], s["a_prev"]
        S0, scan = S, self.mdp.scan(S)
        Ss, S_prev, As, Aps, Ls, Ts, Amps = [], [], [], [], [], [], []
        for _ in range(HZ):
            S_prev.append(S), Aps.append(a_prev)
            args = (S, z, z_pi, o_hist, amp_hist, timers, a_prev, cmd, xi_n, scan)
            if c["checkpoint_imagination"] and torch.is_grad_enabled():
                a_prev, S, z, logits, timers, o_hist, amp_hist, scan = torch.utils.checkpoint.checkpoint(self.imag_core, *args, use_reentrant=False)
            else:
                a_prev, S, z, logits, timers, o_hist, amp_hist, scan = self.imag_core(*args)
            z_pi = z.detach()
            Ss.append(S), As.append(a_prev), Ls.append(logits), Ts.append(timers), Amps.append(amp_hist)
        return {"S0": S0, "S": S, "cmd": cmd, "xi_n": xi_n, "timers": timers, "Ss": Ss, "S_prev": S_prev, "As": As, "Aps": Aps,
                "Ls": Ls, "Ts": Ts, "Amps": Amps, "timers0": s["timers"]}  # fmt: skip

    def policy_update(self):
        """H-step imagination from buffered states, BPTT of the discounted return (task + style reward), GAE critic fit."""
        c, mdp, dev = self.cfg, self.mdp, self.device
        B, HZ = c["pi_batch"], self.horizon
        s, h = self.sample_valid_starts(B)
        frozen = (self.wm, self.critic, self.fail_head, self.style.discriminator)
        for m_ in frozen:
            for p_ in m_.parameters():
                p_.requires_grad_(False)
        im = self._imagine(s, h)
        cmd, xi_n = im["cmd"], im["xi_n"]
        flat = lambda xs: torch.stack(xs).flatten(0, 1)  # noqa: E731
        rep = lambda x: x.unsqueeze(0).expand(HZ, *x.shape).flatten(0, 1)  # noqa: E731
        S_all = tuple(flat([St[i] for St in im["Ss"]]) for i in range(6))
        S_prev_all = tuple(flat([St[i] for St in im["S_prev"]]) for i in range(6))
        r, d, terms = self.reward_term(S_prev_all, S_all, flat(im["As"]), flat(im["Aps"]), rep(cmd), flat(im["Ts"]), flat(im["Ls"]))
        r = r + mdp.volume_points_penetration(S_all)
        r_style = self.style.reward(mdp.amp_window(flat(im["Amps"])))
        r = r + r_style
        p_fail_mean = torch.zeros((), device=dev)
        if c["fail_head"] and self.it >= self.fail_head_start_it + c["fail_head_warmup_iters"]:
            p_fail = torch.sigmoid(self.fail_head(self.critic_input(S_all, rep(cmd), flat(im["Ts"]), rep(xi_n)))).clamp(max=c["fail_clip"])
            d = 1 - (1 - d) * (1 - p_fail)
            p_fail_mean = p_fail.detach().mean()
        r, d = r.view(HZ, B), d.view(HZ, B)
        alive = torch.cumprod(1 - d, 0)
        c_t = torch.cat((torch.ones(1, B, device=dev), alive[:-1]), 0)
        disc = c["gamma"] ** torch.arange(HZ, device=dev, dtype=torch.float32).unsqueeze(-1)
        J = (disc * c_t * r).sum(0)
        cT = alive[-1]
        r_sum = (c_t * r).detach().sum(0)
        rews, dones = r.detach(), d.detach()
        if not c["no_critic"]:
            J = J + (c["gamma"] ** HZ) * cT * self.critic(self.critic_input(im["S"], cmd, im["timers"], xi_n))
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
        v_loss = torch.zeros((), device=dev)
        if not c["no_critic"]:
            with torch.no_grad():
                S_c = tuple(torch.cat((im["S0"][i].unsqueeze(0), torch.stack([St[i].detach() for St in im["Ss"]])), 0).flatten(0, 1) for i in range(6))
                T_c = torch.cat((im["timers0"].unsqueeze(0), torch.stack([t.detach() for t in im["Ts"]])), 0).flatten(0, 1)
                rep1 = lambda x: x.unsqueeze(0).expand(HZ + 1, *x.shape).flatten(0, 1)  # noqa: E731
                X = self.critic_input(S_c, rep1(cmd), T_c, rep1(xi_n)).view(HZ + 1, B, -1)
            self.critic.norm.update(X)
            with torch.no_grad():
                V = self.critic(X)
                R = V[-1]
                targets = []
                for t in range(HZ - 1, -1, -1):
                    R = rews[t] + c["gamma"] * (1 - dones[t]) * ((1 - c["lam"]) * V[t + 1] + c["lam"] * R)
                    targets.append(R)
                targets = torch.stack(targets[::-1])
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
        term_means["style"] = r_style.detach().mean()
        term_means["p_fail_mean"] = p_fail_mean
        return loss.detach(), v_loss.detach(), (r_sum / HZ / mdp.dt).mean(), gn.detach(), cT.mean().detach(), term_means

    @torch.no_grad()
    def latent_gap(self, num: int = 256) -> dict:
        """Latent error vs open-loop horizon step on held-out real windows (units of the target latent variance)."""
        steps = [n for n in LATENT_GAP_STEPS if n <= self.horizon]
        w = self.buffer.sample_windows(num, self.horizon, self.buffer.holdout_envs)
        z0 = self.encode_depth(w["depth"][:, 0].float() / DEPTH_SCALE)
        xi_w = self.episode_context(w["lags"], self.fr_n[w["env"]].unsqueeze(1).expand(-1, self.horizon, -1))
        ctx = self.wm_context(xi_w, w["a_prev"])
        S, z, out = unflatten_state(w["s"][:, 0]), z0, {}
        for j in range(self.horizon):
            S, _, z = self.wm.step(S, w["a"][:, j], ctx[:, j], self.mdp.scan(S), z)
            if j + 1 in steps:
                m = w["mask"][:, j]
                z_tgt = self.encode_depth(w["depth1"][:, j].float() / DEPTH_SCALE)
                z_scale = z_tgt.var(0).mean().clamp(min=1e-6)
                for name, zz in (("pred", z), ("copy", z0)):
                    out[f"latent_gap/step_{j + 1}_{name}"] = float(((((zz - z_tgt) ** 2).mean(-1) / z_scale) * m).sum() / m.sum().clamp(min=1))
        return out

    # ------------------------------------------------------------------------------------------------
    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        if not self.training_ready:
            self.setup_training()
        if self.log_dir is not None:
            store_code_state(self.log_dir, self.git_status_repos)
        c = self.cfg
        stats = {"r": [], "r_gap": [], "ep_len": [], "term": [], "done": []}
        self.collect(c["warmup_steps"], stats)
        t_start = time.time() - self.wall_offset
        t_log = time.time()
        step_acc = 0.0
        keys = ("wm", "wm8", "bce", "latent", "latent_k", "recon", "fail_loss", "pi", "v", "ir", "gn", "surv", "d_policy", "d_reference")
        agg: dict[str, list] = {k: [] for k in keys}
        terms_acc: dict[str, list] = {}
        while self.it < num_learning_iterations:
            lr = self.lr_at(self.it)
            for o in (self.opt_wm, self.opt_pi, self.opt_v, self.opt_fail):
                for gparam in o.param_groups:
                    gparam["lr"] = lr
            step_acc += 1.0 / c["utd"]
            n = int(step_acc)
            step_acc -= n
            self.collect(n, stats)
            for _ in range(c["wm_batches"]):
                m = self.wm_update()
            for k_ in ("wm", "wm8", "bce", "latent", "latent_k", "recon", "fail_loss"):
                agg[k_].append(m[k_])
            sm = self.style_update()
            if sm:
                agg["d_policy"].append(sm["discriminator_policy"]), agg["d_reference"].append(sm["discriminator_reference"])
            if self.it >= c["wm_only_iters"]:
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
                    "latent/err_window": mean(agg["latent"]), "latent/err_8step": mean(agg["latent_k"]), "latent/recon_l1": mean(agg["recon"]),
                    "fail/bce_train": mean(agg["fail_loss"]), "style/d_policy": mean(agg["d_policy"]), "style/d_reference": mean(agg["d_reference"]),
                    "policy/loss": mean(agg["pi"]), "policy/grad_norm": mean(agg["gn"]), "critic/value_loss": mean(agg["v"]),
                    "imagination/reward": mean(agg["ir"]), "imagination/survival": mean(agg["surv"]),
                    "sim/reward": mean(stats["r"]), "sim/analytic_reward_abs_gap": mean(stats["r_gap"]),
                    "sim/episode_length": mean(stats["ep_len"]), "sim/terminated_fraction": term_frac,
                    "sim/terrain_level": float(self.unwrapped.scene.terrain.terrain_levels.float().mean()),
                    "perf/it_per_s": ips, "perf/lr": lr, "perf/sim_steps": float(self.sim_steps), "perf/wall_h": wall_h,
                    "perf/max_mem_gb": torch.cuda.max_memory_allocated() / 2**30,
                }  # fmt: skip
                if self.buffer.size >= self.horizon:
                    row.update(self.latent_gap())
                for k, v in terms_acc.items():
                    row[f"imagination/reward_term/{k}"] = mean(v)
                if self.writer is not None:
                    for k, v in row.items():
                        if not math.isnan(v):
                            self.writer.add_scalar(k, v, self.it)
                    self.writer.flush()
                g = lambda k: row.get(f"latent_gap/step_{k}_pred", float("nan"))  # noqa: E731
                print(f"it {self.it} | {wall_h:.3f} h | wm {row['wm/loss']:.4f} e8 {row['wm/err_8step']:.4f} bce {row['wm/contact_bce']:.4f} "
                      f"| z {row['latent/err_window']:.4f} rec {row['latent/recon_l1']:.4f} gap@32 {g(32):.4f} "
                      f"| D pol {row['style/d_policy']:.2f} ref {row['style/d_reference']:.2f} style {row.get('imagination/reward_term/style', float('nan')):.3f} "
                      f"| pi {row['policy/loss']:.3f} v {row['critic/value_loss']:.4f} imag_r {row['imagination/reward']:.3f} surv {row['imagination/survival']:.3f} "
                      f"| sim_r {row['sim/reward']:.3f} gap {row['sim/analytic_reward_abs_gap']:.4f} ep_len {row['sim/episode_length']:.1f} "
                      f"term {term_frac:.2f} level {row['sim/terrain_level']:.2f} | {ips:.2f} it/s mem {row['perf/max_mem_gb']:.1f}G", flush=True)  # fmt: skip
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
        d, name = os.path.split(path)
        return os.path.join(d, "supermimic_state_" + name[len("model_"):])

    def add_git_repo_to_log(self, repo_path: str) -> None:
        self.git_status_repos.append(repo_path)

    def save(self, path: str, infos=None) -> None:
        """``path``: the ``OnPolicyRunner`` checkpoint format (directly playable); sidecar: the training state."""
        run_state_dict = {"model_state_dict": self.ac.state_dict(), "optimizer_state_dict": self.opt_pi.state_dict()}
        run_state_dict.update({f"{group}_normalizer_state_dict": n.state_dict() for group, n in self.normalizers.items()})
        run_state_dict.update({"iter": self.it, "infos": infos})
        torch.save(run_state_dict, path)
        torch.save({
            "wm": self.wm.state_dict(), "latent_dec": self.latent_dec.state_dict(), "critic": self.critic.state_dict(),
            "opt_wm": self.opt_wm.state_dict(), "opt_v": self.opt_v.state_dict(), "iteration": self.it,
            "wall_time": self.wall_offset, "sim_steps": self.sim_steps, "cfg": self.cfg,
            "fr_mean": self.fr_mean, "fr_std": self.fr_std, "style": self.style.state_dict(),
            "fail_head": self.fail_head.state_dict(), "opt_fail": self.opt_fail.state_dict(), "fail_head_start_it": self.fail_head_start_it,
        }, self._sidecar(path))  # fmt: skip

    def load(self, path: str):
        loaded_dict = torch.load(path, weights_only=True)
        self.ac.load_state_dict(loaded_dict["model_state_dict"])
        self.opt_pi.load_state_dict(loaded_dict["optimizer_state_dict"])
        for group_name, normalizer in self.normalizers.items():
            normalizer.load_state_dict(loaded_dict[f"{group_name}_normalizer_state_dict"])
        self.it = loaded_dict["iter"]
        self.resume_path = path
        return loaded_dict["infos"]

    def _load_sidecar(self, path: str) -> None:
        ck = torch.load(self._sidecar(path), map_location=self.device, weights_only=False)
        self.wm.load_state_dict(ck["wm"]), self.latent_dec.load_state_dict(ck["latent_dec"]), self.critic.load_state_dict(ck["critic"])
        self.opt_wm.load_state_dict(ck["opt_wm"]), self.opt_v.load_state_dict(ck["opt_v"])
        self.style.load_state_dict(ck["style"])
        self.fail_head.load_state_dict(ck["fail_head"]), self.opt_fail.load_state_dict(ck["opt_fail"])
        self.fail_head_start_it = ck["fail_head_start_it"]
        self.wall_offset, self.sim_steps = ck["wall_time"], ck["sim_steps"]
        assert self.it == ck["iteration"]
        self.fr_mean, self.fr_std = ck["fr_mean"].to(self.device), ck["fr_std"].to(self.device)
        self.fr_n = (self.reader.friction() - self.fr_mean) / self.fr_std

    def train_mode(self) -> None:
        self.ac.train()
        for normalizer in self.normalizers.values():
            normalizer.train()

    def eval_mode(self) -> None:
        self.ac.eval()
        for normalizer in self.normalizers.values():
            normalizer.eval()

    def get_inference_policy(self, device=None):
        self.eval_mode()
        if device is not None:
            self.ac.to(device)
            for normalizer in self.normalizers.values():
                normalizer.to(device)
        return lambda x: self.ac.act_inference(self.normalize_policy_obs(x))

    def export_as_onnx(self, obs, export_model_dir: str) -> None:
        self.eval_mode()
        obs = self.normalize_policy_obs(obs)
        if self.pol_norm is not None:
            self.pol_norm.export(os.path.join(export_model_dir, "policy_normalizer.npz"))
        self.ac.export_as_onnx(obs, export_model_dir)
