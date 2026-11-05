"""SuperMimic training of ACP Stage 1: collect -> fit world model -> BPTT policy + critic update.

Adapted from ``instinct_mj.rl.supermimic.runner`` (SuperMimic reference implementation, MIT License, Chong Zhang / ETH Zurich).

The simulator is the unchanged mjlab ``Instinct-ACP-Track-G1-v0`` env wrapped by ``InstinctRlVecEnvWrapper`` (DR, resets, CSI start
phases, terminations and timeouts stay the task's). Rewards / terminations / observations in imagination are the analytic
heads of ``mdp.py`` (Table 3 rewards on imagined states by FK). The policy IS the task's ``ActorCritic`` actor (MLP on the 468-dim
``policy`` group); the runner's own critic lives on privileged features. The world model predicts the pelvis-rooted robot state and the
logits of the 9 contact links (label: the env's c_meas); its context is the 17 x 11 pelvis scan and the previous action (the action
delay of the DR changes the effective target within a step).

Stage 2 hooks (``contact_source``): per env and imagined step the contact targets of the observation are the frozen planner's with
probability ``eta_obs`` (paper Eq. 3); the reward weights follow ``eta_rew``. ``planner_update`` (hindsight training) is a stub.

The runner replaces ``OnPolicyRunner`` as the task's ``runner_entry_point`` (same public API as ``SuperMimicRunner``).
"""

from __future__ import annotations

import copy
import math
import os
import time

import instinct_rl
import instinct_rl.modules as modules
import torch
import yaml
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from instinct_rl.utils.utils import get_subobs_size, store_code_state
from tensorboardX import SummaryWriter

from instinct_mj.rl.supermimic import networks
from instinct_mj.rl.supermimic.mdp import flatten_state, unflatten_state
from instinct_mj.rl.supermimic.rotations import mat_to_quat
from instinct_mj.rl.supermimic.networks import Critic, FailureHead, WorldModel
from instinct_mj.rl.supermimic.runner import CONTACT_LOSS_WEIGHT

from instinct_mj.tasks.acp.mdp.commands import yaw_of_quat

from .buffer import ACPReplayBuffer
from .geo_contact import GeometricContact
from .mdp import ACT_DIM, ERR_FLOOR, NUM_J, ODO_HIST, POLICY_OBS_DIM, PROPRIO_DIM, SCAN_NX, SCAN_NY, TRACKER_X, TRACKER_Y, make_xi, split_xi_dyn, targets_to_world
from .motion import NUM_CONTACT, PELVIS
from .rays import ClippedRunningNorm
from .planner import CONTACT_LINKS, AttentionContactPlanner, points_physical
from .hindsight import HindsightReplay
from .stage2 import EpisodeRecorder, PartitionedReplay, PlannerTrainer, add_episodes, eta as eta_schedule, label_episode_samples, mix_contacts
from .sim import ACPSimReader, build_mdp

SCAN_DIM = SCAN_NX * SCAN_NY
PLANNER_PROPRIO = 5 * PROPRIO_DIM  # the first 375 entries of the policy observation
TARGETS_SLICE = slice(PLANNER_PROPRIO + 63, PLANNER_PROPRIO + 63 + 27)  # contact targets of the policy observation (after the tracker scan)
GOAL_SLICE = slice(PLANNER_PROPRIO + 63 + 27, PLANNER_PROPRIO + 63 + 30)


def clip_end_validity(f0: torch.Tensor, clip_end: torch.Tensor, horizon: int, end_offset: int = 0, exact_last: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Per imagined transition j (0..H-1, shape (H, B)) of starts at the reference frames ``f0``: ``valid`` = the transition ends inside the clip
    (``f0 + j + 1 <= clip_end + end_offset``) and ``last`` = it is the last valid one although the horizon goes on (the env time-outs there). Both are all-valid /
    all-zero for starts whose horizon fits in the clip, so those behave as without the masks. ``clip_end`` is the last valid frame; the real env ends the episode
    (``dataset_exhausted``) with the transition into the frame after it, so ``end_offset = 1`` is the real boundary (``clip_end_v2``). ``exact_last``: ``last`` from the absolute frames, so
    the transition that ends the clip at exactly step ``horizon`` is the last one too."""
    j = torch.arange(horizon, device=f0.device).unsqueeze(1)
    valid = (f0.unsqueeze(0) + j + 1 <= clip_end.unsqueeze(0) + end_offset).float()
    if exact_last:
        return valid, valid * (f0.unsqueeze(0) + j + 1 == clip_end.unsqueeze(0) + end_offset).float()
    following = torch.cat((valid[1:], torch.ones_like(valid[:1])), 0)
    return valid, valid * (1.0 - following)


CLIP_FAMILIES = ("walk", "get_up", "stairs", "box", "climb", "ladder", "vault", "gap", "sparse")


def clip_family(group: str) -> str:
    """Clip family of a dataset group ``<skill>__<source>`` (kimodo / numbered / new variants collapse into the skill: walk*, stairs*, box*, ladder*, vault*,
    gap*, sparse*, ...); stones / beam / stakes / pallets are the "stepping" family."""
    skill = group.split("__")[0]
    return next((f for f in CLIP_FAMILIES if skill.startswith(f)), "stepping")


def episode_return(disc_ret: torch.Tensor, ret: torch.Tensor, T: torch.Tensor, terminated: torch.Tensor, gamma: float) -> torch.Tensor:
    """Discounted return estimate of finished episodes (upstream ``env.py``): ``disc_ret`` + (not terminated) gamma^T mean_reward / (1 - gamma), the truncated
    episodes bootstrap with their mean reward (``ret``: undiscounted sum, ``T``: episode length)."""
    return disc_ret + torch.where(terminated, 0.0, (gamma**T) * (ret / T) / (1 - gamma))


def clip_end_bootstrap_weights(c_t: torch.Tensor, d_real: torch.Tensor, step_last: torch.Tensor, gamma: float) -> torch.Tensor:
    """(H, B) weight of the critic value of the state after imagined step t in the return when the last step inside the clip is a time-out that bootstraps
    (``clip_end_bootstrap``): ``step_last_t gamma^{t+1} c_t (1 - d_real_t)`` (``c_t``: probability to be alive at the start of step t, ``d_real``: the real
    termination probability of the step, without the forced time-out). Zero except at the last valid step of a start."""
    disc = gamma ** torch.arange(1, c_t.shape[0] + 1, device=c_t.device, dtype=torch.float32).unsqueeze(-1)
    return step_last * disc * c_t * (1 - d_real)


def lambda_return_targets(rews: torch.Tensor, dones: torch.Tensor, V: torch.Tensor, gamma: float, lam: float, last: torch.Tensor | None = None,
                          d_real: torch.Tensor | None = None) -> torch.Tensor:
    """GAE(lambda) critic targets (H, B) of the rewards ``rews`` and continuation ``1 - dones``, bootstrapped with ``V`` (H + 1, B). With ``last`` (the last
    valid step inside the clip, ``clip_end_bootstrap``) that step's target is ``r + gamma (1 - d_real) V_{t+1}``: it bootstraps with the value of the state
    at the clip end instead of ending with zero and does not use the (masked) steps after it."""
    R = V[-1]
    targets = []
    for t in range(rews.shape[0] - 1, -1, -1):
        nxt, cont = (1 - lam) * V[t + 1] + lam * R, 1 - dones[t]
        if last is not None:
            nxt, cont = torch.where(last[t] > 0.5, V[t + 1], nxt), torch.where(last[t] > 0.5, 1 - d_real[t], cont)
        R = rews[t] + gamma * cont * nxt
        targets.append(R)
    return torch.stack(targets[::-1])


def to_host(groups: dict[str, dict[str, list]]) -> dict[str, dict[str, list[float]]]:
    """The values of the logged lists (0-dim tensors or floats) as Python floats with ONE device-to-host copy for all of them (one sync per element
    otherwise): ``float(v)`` of a float32 tensor is exactly the double of ``.double().tolist()``, so the logged means are the same numbers."""
    flat = [v for lists in groups.values() for xs in lists.values() for v in xs if torch.is_tensor(v)]
    host = iter(torch.stack([v.detach().double().reshape(()) for v in flat]).tolist() if flat else [])
    return {g: {k: [next(host) if torch.is_tensor(v) else float(v) for v in xs] for k, xs in lists.items()} for g, lists in groups.items()}


class ACPRunner:
    """Trainer / play-time policy holder of the task (registered as the task's ``runner_entry_point``).

    Construction is light and is all that ``instinct-play`` needs: the task's policy (``ActorCritic`` built by
    ``instinct_rl.modules.build_actor_critic``), its optimizer and the observation normalizers, plus ``load`` /
    ``get_inference_policy`` / ``export_as_onnx``. The training-only components are built by :meth:`setup_training`, which ``learn`` calls first."""

    def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device: str = "cuda:0"):
        """``train_cfg`` = the task agent cfg ``to_dict()``: ``policy`` / ``normalizers`` plus the flat ``ACPRunnerCfg`` hyperparameters."""
        self.cfg, self.env, self.log_dir, self.device = copy.deepcopy(train_cfg), env, log_dir, device
        c = self.cfg
        self.unwrapped = env.unwrapped
        policy_cfg = c["policy"]
        obs_format = env.get_obs_format()
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
        self.planner = None  # the contact planner of Stage 2 (inference, collection, imagination and the trainer share this module)
        if c["contact_source"] != "reference" or c["train_planner"]:
            self.planner = AttentionContactPlanner().to(device).eval()
            if c["planner_checkpoint"]:
                self.planner.load_state_dict(torch.load(c["planner_checkpoint"], map_location=device, weights_only=True))
        assert get_subobs_size(obs_format["policy"]) == POLICY_OBS_DIM and env.num_actions == ACT_DIM
        self.N = env.num_envs
        self.it = 0
        self.git_status_repos = [instinct_rl.__file__]
        self.resume_path: str | None = None
        self.checkpoint_it = 0
        self.training_ready = False
        self.writer = None
        _, _ = env.reset()

    def setup_training(self) -> None:
        """Build everything that only training needs (called by ``learn``). The simulator runs with ``auto_reset=False`` from here on."""
        if dist.is_initialized():
            raise NotImplementedError("ACPRunner trains in a single process (no distributed data parallel).")
        c, env, device = self.cfg, self.env, self.device
        if c["contact_source"] not in ("reference", "planner", "mixed"):
            raise ValueError(f"contact_source must be 'reference', 'planner' or 'mixed', got {c['contact_source']!r}")
        if c["contact_source"] == "mixed" and c["eta_rew_start_iter"] <= c["eta_obs_start_iter"]:
            raise ValueError("contact_source='mixed': eta_rew_start_iter must be > eta_obs_start_iter (u_rew > u_obs)")
        if c["start_sampling"] not in ("uniform", "per_motion"):
            raise ValueError(f"start_sampling must be 'uniform' or 'per_motion', got {c['start_sampling']!r}")
        if c["clip_sampling"] not in ("uniform", "return"):
            raise ValueError(f"clip_sampling must be 'uniform' or 'return', got {c['clip_sampling']!r}")
        if c["reset_sampling"] not in ("csi", "bins"):
            raise ValueError(f"reset_sampling must be 'csi' or 'bins', got {c['reset_sampling']!r}")
        if c["reset_sampling"] == "bins" and (c["clip_sampling"] != "uniform" or c["csi_strict_link_eps"] > 0 or c["real_ref_inject_eps"] > 0):
            raise ValueError("reset_sampling='bins' replaces the CSI: clip_sampling='return', csi_strict_link_eps and real_ref_inject_eps (a CSI failure count) must be off")
        if c["clip_end_bootstrap"] and not c["imagine_past_clip_end"]:
            raise ValueError("clip_end_bootstrap needs imagine_past_clip_end (the clip end inside the horizon)")
        self.unwrapped.cfg.auto_reset = False
        torch.manual_seed(c["seed"])
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        networks.AMP_DTYPE = torch.bfloat16 if c["amp"] == "bf16" else None
        self.pol_norm.eval()  # statistics are updated explicitly from collected real observations
        self.motion, self.kin, self.scanner, self.mdp = self.build_analytic_mdp()
        if c["geo_contact"]:
            self.mdp.geo = GeometricContact(self.scanner, self.unwrapped.scene["motion_reference"].cfg.robot_model_path, list(CONTACT_LINKS), device,
                                            d0=c["geo_contact_d0"], tau=c["geo_contact_tau"])
        self.mdp.soft_tz, self.mdp.soft_tt = c["soft_term_tau_dist"], c["soft_term_tau_tilt"]
        self.scanner.slope_gradient = c["scan_slope_gradient"]
        self.mdp.stopgrad = tuple(n for n in c["bptt_stopgrad_terms"].split(",") if n)
        if unknown := set(self.mdp.stopgrad) - set(self.mdp.w):
            raise ValueError(f"bptt_stopgrad_terms: unknown reward terms {sorted(unknown)}")
        self.reader = ACPSimReader(self.unwrapped)
        self.motion_names = [os.path.splitext(os.path.basename(f))[0] for f in next(iter(self.unwrapped.scene["motion_reference"].motion_buffers.values()))._all_motion_files]
        yaml_groups = [e.get("group", "") for e in next(iter(self.unwrapped.scene["motion_reference"].motion_buffers.values())).yaml_data["motion_files"]]
        self.family_names = sorted({clip_family(g) for g in yaml_groups})
        self.clip_family = torch.tensor([self.family_names.index(clip_family(g)) for g in yaml_groups], device=device)  # family of every clip (monitors by clip group)
        self.static = self.reader.static_context(self.mdp)  # (N, XI_STATIC_DIM) startup DR context of every env
        self.mdp.set_xi_stats(make_xi(self.static, self.reader.dynamic_context()))
        self.hist, self.horizon = c["hist"], c["horizon"]
        assert self.hist == 5, "the policy observation has 5-step proprio histories"
        assert abs(self.motion.framerate * self.unwrapped.step_dt - 1.0) < 1e-6, "imagination advances the reference by one frame per step"
        if not bool((self.motion.clip_len > self.horizon).any()):
            raise ValueError(f"No reference clip is longer than the imagination horizon ({self.horizon} frames).")
        assert c["warmup_steps"] >= max(self.horizon, c["wm_window"])
        self.wm = WorldModel(SCAN_DIM + ACT_DIM, NUM_CONTACT, self.mdp.xi_dim, 0, num_joints=NUM_J).to(device)  # context: scan + previous action
        self.critic = Critic(self.mdp.critic_dim()).to(device)
        self.opt_wm = torch.optim.Adam(self.wm.parameters(), lr=c["lr"])
        self.opt_v = torch.optim.Adam(self.critic.parameters(), lr=c["lr"])
        self.fail_head = FailureHead(self.mdp.critic_dim()).to(device)
        self.opt_fail = torch.optim.Adam(self.fail_head.parameters(), lr=c["lr"])
        self.fail_head_start_it = 0
        # the inputs contain exact height scans: floor the scan std of the input normalizers and clip the normalized values (see ClippedRunningNorm)
        wm_scan = slice(self.wm.in_dim - SCAN_DIM - ACT_DIM, self.wm.in_dim - ACT_DIM)
        critic_scan = slice(self.mdp.critic_dim() - self.mdp.xi_dim - SCAN_DIM, self.mdp.critic_dim() - self.mdp.xi_dim)
        self.wm.in_norm = ClippedRunningNorm(self.wm.in_dim, wm_scan).to(device)
        self.critic.norm = ClippedRunningNorm(self.mdp.critic_dim(), critic_scan).to(device)
        self.fail_head.norm = ClippedRunningNorm(self.mdp.critic_dim(), critic_scan).to(device)
        # planner (Stage 2): one module for the trainer, the real collection and the (no-grad) imagination
        self.planner_trainer = self.recorder = self.replay = None
        if c["train_planner"]:
            self.planner_trainer = PlannerTrainer(
                device, lr=c["planner_lr"], planner=self.planner, updates_per_iter=c["planner_updates_per_iter"],
                batch_size=c["planner_batch"], sub_batches=c["planner_sub_batches"], min_entries=c["planner_min_entries"],
            )  # fmt: skip
            self.recorder = EpisodeRecorder(self.N, device)
            points_sensor = self.unwrapped.scene["acp_contact_points"]
            self.contact_point_ids = [points_sensor.primary_names.index(name) for name in CONTACT_LINKS]
            self.last_contact_pos = torch.zeros(self.N, NUM_CONTACT, 3, device=device)
            groups = [e.get("group", "") for e in next(iter(self.unwrapped.scene["motion_reference"].motion_buffers.values())).yaml_data["motion_files"]]
            self.clip_group = torch.tensor([sorted(set(groups)).index(g) for g in groups], device=device)  # terrain type of a clip (no groups: one type)
            self.replay = self.make_replay()
            assert self.replay.ref_tile_capacity == c["planner_ref_tile_capacity"], "make_replay must pass planner_ref_tile_capacity"
            ref_tiles_max = len(self.clip_group)  # one reference tile per clip
            sample_bytes = sum(math.prod(shape) * (1 if k == "target_flag" else 4) for k, shape in HindsightReplay.shapes.items())
            print(f"[planner replay] reference partition: <= {ref_tiles_max} (type, tile) rings x {self.replay.ref_tile_capacity} samples x {sample_bytes} B = "
                  f"{ref_tiles_max * self.replay.ref_tile_capacity * sample_bytes / 2**30:.2f} GiB (plain ring: {self.replay.ref_capacity} samples, allocated on first use)", flush=True)
        elif self.planner is not None:
            for p_ in self.planner.parameters():
                p_.requires_grad_(False)
        self.planner_obs = None  # the env's noisy planner group at the current observation (the planner's 17 x 11 points)
        self.err_floor = ERR_FLOOR.to(device)
        self.buffer = self.make_buffer()
        self.buffer.clip_table = self.motion.clip_id  # global frame -> clip (clip-balanced window sampling)
        self.obs, extras = env.get_observations()
        self.planner_obs = extras["observations"]["planner"]
        self.sim_steps, self.wall_offset = 0, 0.0
        self.ep_disc_ret = None
        if c["clip_sampling"] == "return" or c["reset_sampling"] == "bins":  # SuperMimic App. D.1: per-episode discounted return of the env reward, G_max = the largest attainable discounted return
            g_max = sum(max(w_, 0.0) for w_ in self.mdp.w.values()) * self.mdp.dt / (1 - c["gamma"])
            self.csi().enable_bin_sampling(c["gamma"], g_max, c["bins_v2"], c["bins_difficulty_k"]) if c["reset_sampling"] == "bins" else self.csi().enable_return_sampling(c["gamma"], g_max)
            self.ep_disc_ret, self.ep_ret, self.ep_T = (torch.zeros(self.N, device=device) for _ in range(3))
        self.hist_age = self.unwrapped.episode_length_buf.clone() if c["real_ref_inject_eps"] > 0 else None  # steps since the last reset OR injection
        self.writer = SummaryWriter(log_dir=self.log_dir) if self.log_dir is not None else None
        self._build_fns()
        if self.resume_path is not None:
            self._load_sidecar(self.resume_path)
            planner_opt = [self.planner_trainer.opt] if self.planner_trainer is not None else []  # (its counters too: 70 syncs per optimizer step otherwise)
            for opt in (self.opt_wm, self.opt_pi, self.opt_v, self.opt_fail, *planner_opt):  # the Adam step counters were saved as CUDA tensors: back on the CPU (a CUDA counter makes every step sync)
                for state in opt.state.values():
                    state["step"] = state["step"].cpu()
        self.training_ready = True

    # ------------------------------------------------------------------------------------------------
    # Extension points (overridden by the Stage 3 runner for the mixed reference / goal-reaching env; the defaults are the Stage 1 / 2 behaviour)
    def build_analytic_mdp(self):
        """(motion library, kinematics, height scanner, analytic MDP) of the live env."""
        c = self.cfg
        return build_mdp(self.unwrapped, c["heightmap_resolution"], c["heightmap_cache_dir"], self.device, c["exact_scan"])

    def make_buffer(self) -> ACPReplayBuffer:
        c = self.cfg
        return ACPReplayBuffer(c["buffer_steps"], self.N, self.device, c["wm_holdout_envs"])

    def make_replay(self) -> PartitionedReplay:
        c = self.cfg
        return PartitionedReplay(c["planner_replay_capacity"], c["planner_replay_capacity"], self.device, c["planner_ref_tile_capacity"])

    def step_context(self) -> dict:
        """Per-env data of the state s_t that ``collect`` needs after the step (before the envs are reset)."""
        return {}

    def buffer_fields(self, ctx: dict) -> dict:
        """Extra fields of the transition added to the replay buffer (``step_context`` data the buffer stores)."""
        return {}

    def before_reset(self, ctx: dict, done: torch.Tensor, terminated: torch.Tensor, contact: torch.Tensor) -> None:
        """Called with the pre-reset data of a transition, before the done envs are reset."""

    def recorder_fields(self, progress: torch.Tensor, ctx: dict) -> dict:
        """Env-type dependent arguments of ``EpisodeRecorder.step``: reference progress ``progress`` (N,); goal-reaching envs report the goal
        distance / offset instead (``is_goal`` flags them). ``ctx``: ``step_context`` of the step plus ``ref_type`` / ``ref_tile``, the terrain
        type (group of the clip in the dataset metadata) and the tile (clip index) of the reference clip of every env."""
        zeros = torch.zeros(self.N, device=self.device)
        return dict(is_goal=torch.zeros(self.N, dtype=torch.bool, device=self.device), phase=progress.clamp(0.0, 1.0), goal_xy=zeros, goal_dz=zeros,
                    terrain_type=ctx["ref_type"], tile=ctx["ref_tile"])  # fmt: skip

    def fail_head_windows(self, w: dict, limit: int) -> torch.Tensor:
        """Indices of the (at most ``limit``) sampled windows the failure head trains on / is calibrated with."""
        return torch.arange(min(limit, w["mask"].shape[0]), device=self.device)

    # ------------------------------------------------------------------------------------------------
    def eta(self) -> tuple[float, float]:
        """(eta_obs, eta_rew) at the current iteration (paper Eq. 3; 0 / 0 for "reference", 1 / 1 for "planner")."""
        c = self.cfg
        if c["contact_source"] == "reference":
            return 0.0, 0.0
        if c["contact_source"] == "planner":
            return 1.0, 1.0
        return eta_schedule(self.it, c["eta_obs_start_iter"], c["eta_ann_iters"]), eta_schedule(self.it, c["eta_rew_start_iter"], c["eta_ann_iters"])

    @torch.no_grad()
    def plan_contacts(self, obs: torch.Tensor, planner_obs: torch.Tensor, eta_obs: float) -> tuple[torch.Tensor, torch.Tensor]:
        """The policy observation with its contact-target part replaced, per env with probability ``eta_obs``, by the whole command of the
        planner (occupancy > 0.5, k = 1), computed from the planner inputs of the same observation (the first 375 entries, the goal, the
        env's planner points). Returns the observation and the (N,) mask of the envs that got the planner command."""
        points = planner_obs.view(-1, SCAN_DIM, 3)
        tgt, valid = AttentionContactPlanner.decode(self.planner(obs[:, :PLANNER_PROPRIO], obs[:, GOAL_SLICE], points), points)
        mixed, use = mix_contacts(obs[:, TARGETS_SLICE], tgt.flatten(1), eta_obs)
        obs = obs.clone()
        obs[:, TARGETS_SLICE] = mixed
        return obs, use

    def record_episodes(self, done_ids: torch.Tensor, stats: dict) -> None:
        """Hindsight-label the episodes that just ended (successful ones, paper filter) into the planner replay."""
        c = self.cfg
        episodes = self.recorder.finish(done_ids, only_successful=c["planner_success_only"])
        if c["planner_success_only"]:
            added = add_episodes(self.replay, episodes)
        else:  # test switch: every episode is labeled
            added = dict(ref_episodes=0, goal_episodes=0, samples=0)
            for ep in episodes:
                batch = label_episode_samples(ep)
                if ep.is_goal:
                    added["goal_episodes"] += int(self.replay.add_goal(batch, ep.terrain_type, ep.tile, ep.quality))
                else:
                    self.replay.add_reference(batch, ep.terrain_type, ep.tile)
                    added["ref_episodes"] += 1
                added["samples"] += ep.proprio.shape[0]
        stats["rec_episodes"].append(float(added["ref_episodes"] + added["goal_episodes"])), stats["rec_samples"].append(float(added["samples"]))

    def planner_update(self) -> dict | None:
        """One planner iteration on the hindsight replay (ten optimizer steps once the replay holds enough samples), see ``stage2.py``."""
        losses = self.planner_trainer.update(self.replay)
        self.planner.eval()
        return losses

    def _build_fns(self):
        mdp, wm, kin = self.mdp, self.wm, self.kin
        k = self.cfg["wm_window"]

        def _ctx(S, a_prev):
            """World-model context: the true pelvis scan and the previous action."""
            bp, bR = kin.forward(*S)[:2]
            return torch.cat((mdp.scan(bp[:, PELVIS], bR[:, PELVIS]), a_prev), -1)

        def _wm_rollout(s, a, a_prev, xi_n):
            """k-step rollout from the real state: states and contact logits (B, k, ...)."""
            S = unflatten_state(s[:, 0])
            preds, logits = [], []
            for j in range(a.shape[1]):
                S, lg, _ = wm.step(S, mdp.pad(a[:, j]), xi_n, _ctx(S, a_prev[:, j]))
                preds.append(S), logits.append(lg)
            return tuple(torch.stack([P[i] for P in preds], 1) for i in range(6)), torch.stack(logits, 1)

        def _wm_err(S_hat, logits, s1, contact, mask):
            std = torch.maximum(wm.out_norm.std(), self.err_floor)
            e = ((wm.state_error(S_hat, unflatten_state(s1)) / std) ** 2).mean(-1)  # (B, k)
            bce = F.binary_cross_entropy_with_logits(logits, contact, reduction="none").sum(-1)
            B = s1.shape[0]
            return ((e + CONTACT_LOSS_WEIGHT * bce) * mask).sum() / B, (e[:, -1], (bce * mask).sum() / B)

        def _reward_term(S, qd0, f, phase, origin, a, a_prev, a_prev2, xi, logits, lp_hist, ep0, planned, eta_rew, H):
            r, terms = mdp.reward(S, qd0, f, phase, origin, a, a_prev, a_prev2, xi, torch.sigmoid(logits), lp_hist, ep0, H, planned, eta_rew, terms=True,
                                  use_geo_contact=mdp.geo is not None)
            return r, mdp.soft_termination(S, f, origin, phase), terms

        def _critic_input(S, f, phase, origin, xi_n, gpos, gyaw):
            return mdp.critic_features(S, f, phase, origin, xi_n, gpos, gyaw)

        def _observe(S, f, phase, origin, gpos, gyaw, gt, hist, delay, drift, push: bool):
            """Policy-group observation pieces of the state ``S`` (all but the proprio history): scan, reference contact targets and
            goal command (after the env's goal resampling) from the ESTIMATED pelvis pose (odometry delay and drift of the episode,
            built from the pelvis pose history ``hist`` as the sensor; ``push``: ``S`` is a new state of the history)."""
            bp, bR, _, bw = mdp.base_of(kin.forward(*S))
            if push:
                hist = mdp.push_pose(hist, bp, bR)
            ep_, eyaw = mdp.estimated_pose(hist, delay, drift)
            eR = mdp.yaw_rotation(eyaw)
            gpos, gyaw, gt = mdp.goal_update(f, phase, origin, gpos, gyaw, gt)
            scan = mdp.scan(ep_, eR)
            tgt, valid = mdp.targets(ep_, eR, f, phase, origin)
            goal = mdp.goal(ep_, eR, gpos, gyaw)
            return bp, bR, bw, scan, tgt, valid, goal, gpos, gyaw, gt, hist

        def _push(o_hist, S, a_prev, xi):
            bp, bR, _, bw = mdp.base_of(kin.forward(*S))
            return torch.cat((o_hist[:, 1:], mdp.add_noise(mdp.proprio(bR, bw, S[4], S[5], a_prev, xi)).unsqueeze(1)), 1)

        def _assemble(o_hist, scan, tgt, valid, goal_noisy, use_planner_tgt):
            t27 = mdp.noisy_targets(tgt, valid)
            if use_planner_tgt is not None:  # per env: the planner's targets (no noise) or the reference ones
                t27 = torch.where(use_planner_tgt[0].unsqueeze(-1), use_planner_tgt[1].flatten(1), t27)
            return mdp.policy_obs(o_hist, mdp.noisy_tracker_scan(mdp.tracker_scan(scan)), t27, goal_noisy)

        clamp_joints = self.cfg["imagine_clamp_joints"]

        def _step_wm(S, a, a_prev, xi_n):
            a = torch.clamp(a, -20, 20)
            S1, logits, _ = wm.step(S, mdp.pad(a), xi_n, _ctx(S, a_prev))
            hard = mdp.joint_violation(S1[4])[1]  # fraction of the predicted joint positions beyond the hard range, before the clamp
            return a, mdp.clamp_joints(S1, 0.0 if clamp_joints else 0.5), logits, hard

        def _sim_stats(S, S1, a, a_prev, xi_n):
            return wm.inputs(S, mdp.pad(a), xi_n, _ctx(S, a_prev)), wm.targets(S, S1)

        if self.cfg["compile"]:
            _wm_rollout, _wm_err = (torch.compile(fn) for fn in (_wm_rollout, _wm_err))
        if self.cfg["compile_policy"]:
            _reward_term = torch.compile(_reward_term, dynamic=True)
            _critic_input = torch.compile(_critic_input, dynamic=True)
            _observe, _push, _step_wm = torch.compile(_observe), torch.compile(_push), torch.compile(_step_wm)
            _sim_stats = torch.compile(_sim_stats)  # collection: world-model normalizer statistics of every real transition
            self.kin.forward = torch.compile(self.kin.forward, dynamic=True)  # forward kinematics: also called uncompiled by every collection step
        self.wm_rollout, self.wm_err, self.reward_term, self.critic_input = _wm_rollout, _wm_err, _reward_term, _critic_input
        self.observe, self.push, self.assemble, self.step_wm, self.sim_stats = _observe, _push, _assemble, _step_wm, _sim_stats

    def value_bound(self, V: torch.Tensor) -> torch.Tensor:
        """Critic value clipped from above at ``value_max`` (no-op when 0)."""
        return V.clamp(max=self.cfg["value_max"]) if self.cfg["value_max"] > 0 else V

    def csi(self):
        """The ``CurriculumStateInit`` term of the env (per-motion initialization phases)."""
        return self.unwrapped.curriculum_manager.get_term_cfg("curriculum_state_init").func

    def xi_of(self, env_idx: torch.Tensor, dyn: torch.Tensor) -> torch.Tensor:
        return make_xi(self.static[env_idx], dyn)

    def lr_at(self, it: int) -> float:
        c = self.cfg
        if it >= c["lr_decay_iters"]:
            return c["lr"] * c["lr_final_ratio"]
        return c["lr"] * (c["lr_final_ratio"] + (1 - c["lr_final_ratio"]) * 0.5 * (1 + math.cos(math.pi * it / c["lr_decay_iters"])))

    # ------------------------------------------------------------------------------------------------
    @torch.no_grad()
    def collect(self, n_steps: int, stats: dict):
        """Step the task env with the CURRENT policy on the real policy observation + action noise. The env runs with
        ``auto_reset=False`` and finished envs are reset explicitly right after the pre-reset data has been read (see
        ``SuperMimicRunner.collect``). With ``eta_obs`` > 0 the contact targets of the observation are, per env, the whole command of the
        planner (as in imagination). With ``train_planner`` the episodes are recorded for the hindsight planner replay."""
        mdp, reader, env, am = self.mdp, self.reader, self.env, self.unwrapped.action_manager
        fps = self.motion.framerate
        c_inject, c_strict = self.cfg["real_ref_inject_eps"], self.cfg["csi_strict_link_eps"]
        for _ in range(n_steps):
            S = reader.state()
            f, phase, origin = self.motion.current_frame()
            a_prev, a_prev2 = reader.last_action(), am.prev_action.clone()
            ep_step = self.unwrapped.episode_length_buf.clone() if self.hist_age is None else self.hist_age.clone()
            xi_dyn, goal = reader.dynamic_context(), reader.goal(fps)
            odo_drift, odo_delay = reader.odometry()
            ctx = self.step_context()
            fk = self.kin.forward(*S)
            lp0 = fk[0]
            eta_obs = self.eta()[0]
            obs_in, planner_obs_in = self.obs, self.planner_obs
            if eta_obs > 0:
                obs_in, _ = self.plan_contacts(self.obs, planner_obs_in, eta_obs)
            if self.recorder is not None:  # everything the recorder stores for step t describes the state s_t the action is computed from
                sensor = self.unwrapped.scene["height_scanner"]  # the estimated pelvis pose of the observation (frame of the contact targets)
                est_pos, est_yaw = sensor.estimated_pos_w.clone(), yaw_of_quat(sensor.estimated_quat_w)
                force_z, contact_pos = self.contact_measurement(lp0, ep_step == 0)
            self.pol_norm.update(obs_in)
            a = self.ac.act_inference(self.pol_norm(obs_in))
            a = torch.clamp(a + self.cfg["action_noise"] * torch.randn_like(a), -20, 20)

            obs1, rew, dones, extras1 = env.step(a)
            # ---- pre-reset data of this transition ----
            a = am.action.clone()
            done = dones.bool()
            terminated = self.unwrapped.reset_terminated.clone()
            S1 = reader.state()
            bad = ~torch.stack([torch.isfinite(x).flatten(1).all(-1) for x in S1]).all(0)
            any_bad = bool(bad.any())  # (read once: the branch below needs the same answer)
            if any_bad:
                # a blown-up simulation (non-finite state, seen on the ladder clips) is a failure of that episode: the transition is kept as a terminated one
                # from s to s, so that no non-finite value reaches the running normalizers, the replay buffer or the world model
                print(f"[collect] non-finite simulator state in envs {bad.nonzero().squeeze(-1).tolist()[:8]} (clips {self.motion.clip_id[f][bad][:8].tolist()}, "
                      f"episode steps {ep_step[bad][:8].tolist()}): episode terminated", flush=True)
                S1 = tuple(torch.where(bad.view(-1, *[1] * (x0.dim() - 1)), x0, x1) for x0, x1 in zip(S, S1))
                done, terminated = done | bad, terminated | bad
            contact = reader.contact()
            goal1 = reader.goal(fps)
            r_sim = rew.sum(-1).clone()
            if any_bad:
                contact, goal1, r_sim = torch.where(bad[:, None], torch.zeros_like(contact), contact), torch.where(bad[:, None], goal, goal1), torch.where(bad, torch.zeros_like(r_sim), r_sim)
            done_ids = done.nonzero(as_tuple=False).squeeze(-1)
            f1, phase1, origin1 = self.motion.current_frame()
            if c_strict > 0:  # strict CSI success: the episode never left the reference by more than the threshold (bookkeeping only, nothing terminates)
                self.csi().drifted |= mdp.link_deviation(S1, f1, origin1) > c_strict
            if self.ep_disc_ret is not None:  # return of the episodes that end now (upstream env.py): time-outs bootstrap with their mean reward
                self.ep_disc_ret += (self.cfg["gamma"] ** self.ep_T) * r_sim
                self.ep_ret += r_sim
                self.ep_T += 1
                if len(done_ids) > 0:
                    T = self.ep_T[done_ids]
                    csi = self.csi()
                    no_bootstrap = terminated[done_ids]
                    if self.cfg["bins_v2"]:  # an episode that ends at the clip end is a finite task end: no mean-reward bootstrap
                        no_bootstrap = no_bootstrap | self.unwrapped.termination_manager.get_term("dataset_exhausted")[done_ids]
                    G = episode_return(self.ep_disc_ret[done_ids], self.ep_ret[done_ids], T, no_bootstrap, self.cfg["gamma"])
                    finite = torch.isfinite(G)  # a non-finite env reward (blown-up simulation, see "sim_r nan" of the earlier runs) must not enter the clip probabilities
                    if self.cfg["bins_v2"]:
                        finite &= T >= 10  # an episode of less than 10 steps (a reset transient) says nothing about the start bin
                    if csi.bins and csi.difficulty_k > 0:  # the failure rate of the clip (terminations that are no time-outs)
                        csi.update_clip_failures(csi.motion_buffer.assigned_env_motion_selection[done_ids], terminated[done_ids].float())
                    if csi.bins:  # attributed to the START bin of the episode
                        finite &= csi.start_bin[done_ids] >= 0
                        csi.update_bin_returns(csi.start_bin[done_ids][finite], G[finite])
                    else:
                        csi.update_clip_returns(csi.motion_buffer.assigned_env_motion_selection[done_ids][finite], G[finite])
                    self.ep_disc_ret[done_ids], self.ep_ret[done_ids], self.ep_T[done_ids] = 0.0, 0.0, 0.0
            if self.recorder is not None:
                self.record_step(self.obs, planner_obs_in, est_pos, est_yaw, force_z, contact_pos, f, phase, ctx)
                if len(done_ids) > 0:
                    self.record_episodes(done_ids, stats)
            self.before_reset(ctx, done, terminated, contact)
            self.obs, self.planner_obs = obs1, extras1["observations"]["planner"]
            if len(done_ids) > 0:
                self.unwrapped.reset(env_ids=done_ids)
                obs_reset, extras_reset = env.get_observations()
                self.obs = torch.where(done[:, None], obs_reset, obs1)
                self.planner_obs = torch.where(done[:, None], extras_reset["observations"]["planner"], self.planner_obs)
            inj = torch.zeros_like(done)
            if c_inject > 0:  # reference injection: an env that left the reference moves to the reference state instead of ending its episode
                inj = (mdp.link_deviation(S1, f1, origin1) > c_inject) & ~done
                if inj.any():
                    self.inject_reference(inj.nonzero(as_tuple=False).squeeze(-1), f1, origin1)
                    obs_new, extras_new = env.get_observations()
                    self.obs = torch.where(inj[:, None], obs_new, self.obs)
                    self.planner_obs = torch.where(inj[:, None], extras_new["observations"]["planner"], self.planner_obs)
                stats.setdefault("inject", []).append(inj.float().mean())
                self.hist_age = torch.where(done | inj, torch.zeros_like(ep_step), ep_step + 1)  # steps of real history since the episode start / injection
            self.buffer.add({
                "s": flatten_state(S), "s1": flatten_state(S1), "a": a, "a_prev": a_prev, "a_prev2": a_prev2, "contact": contact,
                "origin": origin, "r": r_sim, "f": f, "phase": phase, "ep_step": ep_step.int(), "done": done | inj,
                "f1": f1, "phase1": phase1, "origin1": origin1, "terminated": terminated,
                "xi_dyn": xi_dyn, "goal": goal, "goal1": goal1, "odo_drift": odo_drift, "odo_delay": odo_delay,
                **self.buffer_fields(ctx),
            })  # fmt: skip
            self.sim_steps += self.N
            xi = self.xi_of(torch.arange(self.N, device=self.device), xi_dyn)
            xi_n = mdp.normalize_xi(xi)
            wm_in, wm_tgt = self.sim_stats(S, S1, a, a_prev, xi_n)
            self.wm.in_norm.update(wm_in)
            self.wm.out_norm.update(wm_tgt)
            stats["jl_soft"].append(mdp.joint_violation(S1[4])[0])
            stats["r"].append(r_sim.mean() / mdp.dt)
            for name, v in self.unwrapped.reward_manager._termwise_reward_buf["rewards"].items():  # the env's weighted step reward of every term (already computed)
                stats.setdefault(f"rt/{name}", []).append(torch.where(bad, torch.zeros_like(v), v).mean())
            if len(done_ids) > 0:
                stats["ep_len"].append((ep_step[done] + 1).float().mean())
                stats["term"].append(terminated[done].float().sum())
                stats["done"].append(done.float().sum())

    @torch.no_grad()
    def inject_reference(self, ids: torch.Tensor, f: torch.Tensor, origin: torch.Tensor) -> None:
        """Move the envs ``ids`` to the reference state of the frames ``f`` (root pose and velocity, joint positions and velocities) without ending their
        episode: the odometry and observation histories restart at the new pose and the CSI counts one failure of the clip."""
        p, R, v, w, q, qd = self.mdp.reference_state(f[ids], origin[ids])
        robot = self.unwrapped.scene["robot"]
        robot.write_root_link_pose_to_sim(torch.cat((p, mat_to_quat(R)), -1), env_ids=ids)
        robot.write_root_link_velocity_to_sim(torch.cat((v, w), -1), env_ids=ids)
        robot.write_joint_state_to_sim(q, qd, env_ids=ids)
        self.unwrapped.sim.forward()
        self.unwrapped.scene["height_scanner"].reset(ids)
        self.unwrapped.sim.sense()
        self.csi().register_injection(ids)
        # the teleport is no step of the episode: like the env's reset path, the observation history / delay buffers of these envs restart with the
        # post-injection frame (the other envs keep their timelines) and the finite-difference state of ``contact_acc`` restarts. The commands
        # (contact targets, goal) are functions of the reference time and the current pose, so there is nothing to resample.
        self.unwrapped.reward_manager.get_term_cfg("contact_acc").func.reset(ids)
        self.unwrapped.observation_manager.reset(ids)
        self.unwrapped.observation_manager.compute(update_history=True, env_ids=ids)  # fills the cache ``env.get_observations()`` returns

    @torch.no_grad()
    def contact_measurement(self, link_pos: torch.Tensor, fresh: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Contact of the 9 links at the current state s_t (the window of substeps that led to it): the |F_z| maximum over the window
        (``c_meas``) and the world position of the contact (the contact point of the strongest contact, measured by the solver; a body
        without a contact point at the last substep keeps its last one, and a reset env starts with the link origins ``link_pos``)."""
        term = self.unwrapped.command_manager.get_term("acp_reference_contacts")
        force_z = term.contact_sensor.data.force_history[:, term.contact_sensor_ids, :, 2].abs().amax(-1)
        points = self.unwrapped.scene["acp_contact_points"].data
        found = points.found.view(self.N, -1)[:, self.contact_point_ids] > 0
        self.last_contact_pos = torch.where(fresh.view(-1, 1, 1), link_pos, self.last_contact_pos)
        self.last_contact_pos = torch.where(found.unsqueeze(-1), points.pos.view(self.N, -1, 3)[:, self.contact_point_ids], self.last_contact_pos)
        return force_z, self.last_contact_pos.clone()

    @torch.no_grad()
    def record_step(self, obs_pre, planner_obs_pre, est_pos, est_yaw, force_z, contact_pos, f, phase, ctx: dict | None = None) -> None:
        """Feed the episode recorder one control step (reference envs). Everything describes the state s_t the action is computed from:
        the planner inputs (the observation), the contact force and position of the 9 links, the estimated pelvis pose (the frame of the
        contact targets) and the reference progress, so that the hindsight label of step t is the next NEW contact after s_t."""
        motion = self.motion
        clip = motion.clip_id[f]
        progress = ((f - motion.clip_start[clip]).float() + phase) / (motion.clip_len[clip] - 1).float()
        ctx = {**(ctx or {}), "ref_type": self.clip_group[clip], "ref_tile": clip}
        self.recorder.step(
            proprio=obs_pre[:, :PLANNER_PROPRIO], goal=obs_pre[:, GOAL_SLICE], points=planner_obs_pre.view(-1, SCAN_DIM, 3),
            contact_force_z=force_z, body_pos_w=contact_pos, pelvis_pos_w=est_pos, pelvis_yaw=est_yaw, **self.recorder_fields(progress, ctx),
        )  # fmt: skip

    def wm_update(self):
        """One world-model gradient step on a k-step window batch (dynamics, contact and failure-head losses)."""
        c = self.cfg
        w = self.buffer.sample_windows(c["wm_batch"], c["wm_window"], self.buffer.train_envs, c["wm_failure_frac"], c["wm_failure_horizon"], c["wm_balance_frac"], c["failure_window_fix"])
        env_idx = w["env"]
        k = w["mask"].shape[1]
        xi_n = self.mdp.normalize_xi(self.xi_of(env_idx, w["xi_dyn"][:, 0]))
        S_hat, logits = self.wm_rollout(w["s"], w["a"], w["a_prev"], xi_n)
        loss, (e_last, bce) = self.wm_err(S_hat, logits, w["s1"], w["contact"], w["mask"])
        self.opt_wm.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.wm.parameters(), 1.0)
        self.opt_wm.step()
        m = w["mask"][:, -1]
        out = {"wm": loss.detach() / k, "wm8": (e_last.detach() * m).sum() / m.sum().clamp(min=1), "bce": bce.detach(), "fail_win_sampled": w["failure_sampled"], "fail_win_valid": w["failure_valid"]}
        if not c["fail_head"]:  # the head is not used: no training step
            return out
        # failure ("continue") head on TRUE and (detached) WM-PREDICTED states of the windows, importance-weighted
        sel = self.fail_head_windows(w, c["fail_windows"])
        Bf = len(sel)
        with torch.no_grad():
            flat_f = lambda x: x[sel].flatten(0, 1)  # noqa: E731
            xn = xi_n[sel].unsqueeze(1).expand(Bf, k, -1).flatten(0, 1)
            g1 = flat_f(w["goal1"])
            args_f = (flat_f(w["f1"]), flat_f(w["phase1"]), flat_f(w["origin1"]), xn, g1[:, :3], g1[:, 3])
            X_true = self.critic_input(unflatten_state(flat_f(w["s1"])), *args_f)
            X_pred = self.critic_input(tuple(flat_f(x) for x in S_hat), *args_f)
            self.fail_head.norm.update(X_true)
        label = flat_f(w["terminated"]).float()
        weight = (w["mask"][sel] * w["weight"][sel, None]).flatten()
        pos_weight = torch.where(label > 0.5, c["fail_pos_weight"], 1.0)
        head_loss = lambda X: (F.binary_cross_entropy_with_logits(self.fail_head(X), label, reduction="none") * pos_weight * weight).sum() / weight.sum().clamp(min=1e-6)  # noqa: E731
        fail_loss = head_loss(X_true) + head_loss(X_pred)
        self.opt_fail.zero_grad(set_to_none=True)
        fail_loss.backward()
        nn.utils.clip_grad_norm_(self.fail_head.parameters(), 1.0)
        self.opt_fail.step()
        out["fail_loss"], out["fail_batch_pos"] = fail_loss.detach() / 2, (label * w["mask"][sel].flatten()).sum() / w["mask"][sel].sum().clamp(min=1)
        return out

    def sample_valid_starts(self, B: int):
        """Uniform over all buffered states whose imagination stays inside the reference clip (``f + H <= clip end``). With
        ``imagine_past_clip_end`` every state with a frame left in its clip (``f + 1 <= clip end``); with ``start_sampling="per_motion"`` the clips
        are drawn with equal probability instead of the states."""
        c = self.cfg
        room = int(not c["clip_end_v2"]) if c["imagine_past_clip_end"] else self.horizon
        eligible = self.buffer.eligible_starts(lambda f: f + room <= self.motion.clip_end[self.motion.clip_id[f]])
        if len(eligible) == 0:
            raise RuntimeError("No buffered state has a full imagination horizon left inside its reference clip.")
        if c["start_sampling"] == "per_motion" or c["start_balance_frac"] > 0:
            clip = self.motion.clip_id[self.buffer.data["f"][self.buffer._row(eligible[:, 0]), eligible[:, 1]]]
            weight = 1.0 / torch.bincount(clip, minlength=len(self.motion.clip_len)).float()[clip]
            n_bal = B if c["start_sampling"] == "per_motion" else int(B * c["start_balance_frac"])
            pick = torch.cat((eligible[torch.multinomial(weight, n_bal, replacement=True)],
                              eligible[torch.randint(0, len(eligible), (B - n_bal,), device=self.device)]))
        else:
            pick = eligible[torch.randint(0, len(eligible), (B,), device=self.device)]
        return self.buffer.starts_at(pick[:, 0], pick[:, 1], ODO_HIST)

    def _plan(self, o_hist, scan, goal_noisy, eta_obs, *, observation_only=False):
        """Frozen planner on the imagined planner inputs (375 proprio + 3 goal + encoded 17 x 11 points): physical targets (B, 9, 3),
        validity (B, 9) and the per-env choice of the observation (planner with probability ``eta_obs``)."""
        with torch.no_grad():
            pts = self.mdp.planner_points(scan)
            proprio = torch.cat([o_hist[..., a:b].flatten(1) for a, b in ((0, 3), (3, 6), (6, 29), (29, 52), (52, 75))], -1)  # = obs[:, :375]
            use = torch.rand(scan.shape[0], device=self.device) < eta_obs
            # The planner has no random operations. Keep full-batch point noise before this
            # draw, then omit unused rows only if the reward does not need their targets.
            if observation_only and self.cfg.get("planner_imagination_subset", True) and scan.shape[0] > 1024 and eta_obs < 1:
                tgt, valid = self.planner.selected_targets(proprio, goal_noisy, pts, use)
            else:
                tgt, valid = AttentionContactPlanner.decode(self.planner(proprio, goal_noisy, pts), pts)
        return tgt, valid, use

    def _imagine(self, s: dict, h: dict) -> dict:
        """H-step imagination of the policy from buffer start states ``s`` / histories ``h``. Gradients flow as in ``policy_update``
        (call under ``no_grad`` for diagnostics)."""
        c, mdp, kin = self.cfg, self.mdp, self.kin
        HZ = self.horizon
        S = unflatten_state(s["s"])
        f, phase, origin, a_prev, a_prev2, env_idx = s["f"], s["phase"], s["origin"], s["a_prev"], s["a_prev2"], s["env"]
        xi = self.xi_of(env_idx, s["xi_dyn"])
        xi_n = mdp.normalize_xi(xi)
        gpos, gyaw, gt = s["goal"][:, :3], s["goal"][:, 3], s["goal"][:, 4]
        eta_obs, eta_rew = self.eta()
        use_planner = self.planner is not None and (eta_obs > 0 or eta_rew > 0)  # Stage 1 imagination reads the reference targets only
        with torch.no_grad():
            B = S[0].shape[0]
            hS = tuple(x.flatten(0, 1) for x in unflatten_state(h["s"]))  # ODO_HIST real states, oldest first (the episode start back-fills)
            fk = kin.forward(*hS)
            xi_h = xi.unsqueeze(1).expand(-1, ODO_HIST, -1).flatten(0, 1)
            o_hist = mdp.proprio(fk[1][:, PELVIS], fk[3][:, PELVIS], hS[4], hS[5], h["a_prev"].flatten(0, 1), xi_h).view(B, ODO_HIST, -1)
            # one noise sample per distinct historical observation (backfilled slots duplicate it, as in collection)
            noise = (2 * torch.rand_like(o_hist) - 1) * mdp.proprio_noise
            o_hist = (o_hist + noise.gather(1, h["slot"].unsqueeze(-1).expand_as(noise)))[:, ODO_HIST - self.hist :]
            lp_hist = fk[2 if mdp.contact_acc_vel else 0].view(B, ODO_HIST, NUM_CONTACT, 3)[:, -2:].transpose(0, 1).contiguous()  # states at -1 and 0 (positions, velocities with contact_acc vel1)
            pelvis_hist = mdp.pose_history(fk[0][:, PELVIS].view(B, ODO_HIST, 3), fk[1][:, PELVIS].view(B, ODO_HIST, 3, 3))
            odo_delay, odo_drift = s["odo_delay"], s["odo_drift"]
        ep0 = s["ep_step"].float()
        S0, f0 = S, f
        eps_rep = c["imagine_ref_replay_eps"]
        Su, Reps = [], []  # the states continued from (the reference where the deviation exceeded ``eps_rep``) and where that happened
        clip_end = self.motion.clip_end[self.motion.clip_id[f0]] if c["imagine_past_clip_end"] else None
        step_valid, step_last = clip_end_validity(f0, clip_end, HZ, int(c["clip_end_v2"]), c["clip_end_exact_last"]) if c["imagine_past_clip_end"] else (None, None)
        Ss, As, Aps, Aps2, Fs, Ls, Gs, Ptgt, Hard = [], [], [], [], [], [], [], [], []
        for t in range(HZ):
            if t > 0:
                o_hist = self.push(o_hist, S, a_prev, xi)
            Aps.append(a_prev), Aps2.append(a_prev2)
            bp, bR, bw, scan, tgt, valid, goal, gpos, gyaw, gt, pelvis_hist = self.observe(
                S, f, phase, origin, gpos, gyaw, gt, pelvis_hist, odo_delay, odo_drift, t > 0
            )
            Gs.append(torch.cat((gpos, gyaw.unsqueeze(-1), gt.unsqueeze(-1)), -1))
            choice = None
            goal_noisy = mdp.noisy_goal(goal)  # one noise sample per imagined state: the observation and the planner see the same goal
            if use_planner:
                ptgt, pvalid, use = self._plan(o_hist, scan, goal_noisy, eta_obs, observation_only=eta_rew == 0)
                Ptgt.append((targets_to_world(*mdp.estimated_pose(pelvis_hist, odo_delay, odo_drift), ptgt), pvalid))
                choice = (use, ptgt)
            obs = self.assemble(o_hist, scan, tgt, valid, goal_noisy, choice)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=c["policy_autocast"]):
                a = self.ac.actor(self.pol_norm(obs)).float()
            if c["imagine_action_noise"] > 0:
                a = a + c["imagine_action_noise"] * torch.randn_like(a)
            step_args = (S, a, a_prev, xi_n)
            if c["checkpoint_imagination"] and torch.is_grad_enabled():
                a, S, logits, hard = torch.utils.checkpoint.checkpoint(self.step_wm, *step_args, use_reentrant=False)
            else:
                a, S, logits, hard = self.step_wm(*step_args)
            a_prev2, a_prev = a_prev, a
            f = f + 1 if clip_end is None else torch.minimum(f + 1, clip_end)  # beyond the clip end the reference stays at its last frame (masked)
            Ss.append(S), As.append(a), Fs.append(f), Ls.append(logits), Hard.append(hard.detach())
            if eps_rep > 0:  # demonstration replay: continue from the reference where the imagined state has left it
                replaced = mdp.link_deviation(S, f, origin) > eps_rep
                ref = mdp.reference_state(f, origin)
                S = tuple(torch.where(replaced.view(-1, *[1] * (x.dim() - 1)), r_.detach(), x) for x, r_ in zip(S, ref))
                Reps.append(replaced)
            Su.append(S)
        # goal at the last imagined state (its observation is not built); planner targets of that state for the planned-contact reward
        gpos, gyaw, gt = mdp.goal_update(f, phase, origin, gpos, gyaw, gt)
        Gs.append(torch.cat((gpos, gyaw.unsqueeze(-1), gt.unsqueeze(-1)), -1))
        if use_planner and eta_rew > 0:
            o_hist = self.push(o_hist, S, a_prev, xi)
            bp, bR, bw, scan, tgt, valid, goal, _, _, _, pelvis_hist = self.observe(S, f, phase, origin, gpos, gyaw, gt, pelvis_hist, odo_delay, odo_drift, True)
            ptgt, pvalid, _ = self._plan(o_hist, scan, mdp.noisy_goal(goal), eta_obs)
            Ptgt.append((targets_to_world(*mdp.estimated_pose(pelvis_hist, odo_delay, odo_drift), ptgt), pvalid))
        return {"S0": S0, "f0": f0, "S": S, "f": f, "phase": phase, "origin": origin, "xi": xi, "xi_n": xi_n, "Ss": Ss, "As": As,
                "Aps": Aps, "Aps2": Aps2, "Fs": Fs, "Ls": Ls, "Gs": Gs, "Ptgt": Ptgt, "Hard": Hard, "lp_hist": lp_hist, "ep0": ep0, "eta_rew": eta_rew,
                "step_valid": step_valid, "step_last": step_last, "Su": Su, "Reps": Reps}  # fmt: skip

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
        Ss, As, Aps, Aps2, Fs, Ls, Gs = im["Ss"], im["As"], im["Aps"], im["Aps2"], im["Fs"], im["Ls"], im["Gs"]
        flat = lambda xs: torch.stack(xs).flatten(0, 1)  # noqa: E731  (H*B, ...)
        rep = lambda x: x.unsqueeze(0).expand(HZ, *x.shape).flatten(0, 1)  # noqa: E731
        S_all = tuple(flat([St[i] for St in Ss]) for i in range(6))
        replay = c["imagine_ref_replay_eps"] > 0
        Sc = im["Su"] if replay else Ss  # the states the rollout continued from (the reference where it was replaced): terminations, failure head, critic
        S_cont = tuple(flat([St[i] for St in Sc]) for i in range(6))
        planned = None
        if im["Ptgt"] and im["eta_rew"] > 0:  # the planned-contact reward of transition j uses the planner targets of state j + 1
            planned = (torch.stack([p[0] for p in im["Ptgt"][1:]]).flatten(0, 1), torch.stack([p[1] for p in im["Ptgt"][1:]]).flatten(0, 1))
        eta_rew = torch.tensor(im["eta_rew"], device=dev) if im["eta_rew"] > 0 else None
        qd_start = torch.cat((S0[5].unsqueeze(0), torch.stack([St[5] for St in Sc[:-1]])), 0).flatten(0, 1)  # joint velocities at the transition starts
        r, d, terms = self.reward_term(S_all, qd_start, flat(Fs), rep(phase), rep(origin), flat(As), flat(Aps), flat(Aps2), rep(xi), flat(Ls),
                                       im["lp_hist"], im["ep0"], planned, eta_rew, HZ)  # fmt: skip
        if replay:
            d = mdp.soft_termination(S_cont, flat(Fs), rep(origin), rep(phase))
        if c["imagine_link_hazard_eps"] > 0:  # leaving the reference by more than eps is a failure in the imagination too (as the injection of the real env)
            d = 1 - (1 - d) * (1 - torch.sigmoid((mdp.link_deviation(S_all, flat(Fs), rep(origin)) - c["imagine_link_hazard_eps"]) / c["imagine_link_hazard_tau"]))
        G1 = torch.stack(Gs[1:]).flatten(0, 1)
        p_fail_mean = torch.zeros((), device=dev)
        if c["fail_head"] and self.it >= self.fail_head_start_it + c["fail_head_warmup_iters"]:
            p_fail = torch.sigmoid(self.fail_head(self.critic_input(S_cont, flat(Fs), rep(phase), rep(origin), rep(xi_n), G1[:, :3], G1[:, 3]))).clamp(max=c["fail_clip"])
            d = 1 - (1 - d) * (1 - p_fail)
            p_fail_mean = p_fail.detach().mean()
        r, d = r.view(HZ, B), d.view(HZ, B)
        step_valid = im["step_valid"]
        d_real = d
        if step_valid is not None:  # steps beyond the clip end: no reward; the last step inside the clip ends the episode (time-out, zero continuation)
            r, d = r * step_valid, 1 - (1 - d) * (1 - im["step_last"])
        bootstrap = c["clip_end_bootstrap"] and step_valid is not None  # the time-out at the clip end bootstraps with the critic value of the last state
        alive = torch.cumprod(1 - d, 0)  # c_{t+1}
        c_t = torch.cat((torch.ones(1, B, device=dev), alive[:-1]), 0)
        disc = c["gamma"] ** torch.arange(HZ, device=dev, dtype=torch.float32).unsqueeze(-1)
        J = (disc * c_t * r).sum(0)
        cT = alive[-1]
        r_sum = (c_t * r).detach().sum(0)
        rews, dones = r.detach(), d.detach()
        if not c["no_critic"]:
            J = J + (c["gamma"] ** HZ) * cT * self.value_bound(self.critic(self.critic_input(S, f, phase, origin, xi_n, Gs[-1][:, :3], Gs[-1][:, 3])))
            if bootstrap:
                w_boot = clip_end_bootstrap_weights(c_t, d_real, im["step_last"], c["gamma"]).flatten()
                idx = im["step_last"].flatten().nonzero(as_tuple=False).squeeze(-1)  # the (step, start) rows whose clip ends there
                V_end = self.value_bound(self.critic(self.critic_input(tuple(x[idx] for x in S_cont), flat(Fs)[idx], rep(phase)[idx], rep(origin)[idx], rep(xi_n)[idx], G1[idx, :3], G1[idx, 3])))
                J = J + torch.zeros(HZ * B, device=dev).index_add(0, idx, w_boot[idx] * V_end).view(HZ, B).sum(0)
        if c["abpt_one_step_weight"] > 0 and not c["no_critic"]:  # ABPT-style value gradient: the one-step return with the critic bootstrap
            J1 = r[0] + c["gamma"] * (1 - (d_real[0] if bootstrap else d[0])) * self.critic(self.critic_input(Sc[0], Fs[0], phase, origin, xi_n, Gs[1][:, :3], Gs[1][:, 3]))
            J = (1 - c["abpt_one_step_weight"]) * J + c["abpt_one_step_weight"] * J1
        loss = -J.mean()
        self.opt_pi.zero_grad(set_to_none=True)
        loss.backward()
        gn = nn.utils.clip_grad_norm_(self.ac.actor.parameters(), 1.0)
        nn.utils.clip_grad_value_(self.ac.actor.parameters(), 0.1)
        if not c["no_critic"]:
            with torch.no_grad():
                S_c = tuple(torch.cat((S0[i].unsqueeze(0), torch.stack([St[i].detach() for St in Sc])), 0).flatten(0, 1) for i in range(6))
                f_c = torch.cat((f0.unsqueeze(0), torch.stack(Fs)), 0).flatten()
                rep1 = lambda x: x.unsqueeze(0).expand(HZ + 1, *x.shape).flatten(0, 1)  # noqa: E731
                Gc = torch.stack(Gs).flatten(0, 1)
                X = self.critic_input(S_c, f_c, rep1(phase), rep1(origin), rep1(xi_n), Gc[:, :3], Gc[:, 3]).view(HZ + 1, B, -1)
            if c["clip_end_v2"] and step_valid is not None:  # only the states reached through valid transitions (not the masked rollout after the clip end)
                self.critic.norm.update(X[torch.cat((torch.ones_like(step_valid[:1]), step_valid), 0) > 0.5])
            else:
                self.critic.norm.update(X)
            with torch.no_grad():
                V = self.value_bound(self.critic(X))
                targets = lambda_return_targets(rews, dones, V, c["gamma"], c["lam"], im["step_last"] if bootstrap else None, d_real.detach() if bootstrap else None)  # (H, B)
        if torch.isfinite(gn):
            self.opt_pi.step()
        for m_ in frozen:
            for p_ in m_.parameters():
                p_.requires_grad_(True)
        v_loss = torch.zeros((), device=dev)
        if not c["no_critic"]:
            Xf, Yf = X[:-1].reshape(-1, X.shape[-1]), targets.reshape(-1)
            n = Xf.shape[0]
            mb = n // c["critic_minibatches"]
            wf = None if step_valid is None else step_valid.reshape(-1)  # the states beyond the clip end are not regressed
            for _ in range(c["critic_epochs"]):
                perm = torch.randperm(n, device=dev)
                for i in range(c["critic_minibatches"]):
                    idx = perm[i * mb : (i + 1) * mb]
                    err = (self.critic(Xf[idx]) - Yf[idx]) ** 2
                    v_loss = err.mean() if wf is None else (err * wf[idx]).sum() / wf[idx].sum().clamp(min=1.0)
                    self.opt_v.zero_grad(set_to_none=True)
                    v_loss.backward()
                    nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
                    self.opt_v.step()
        term_means = {k: v.detach().view(HZ, B).mean() for k, v in terms.items()}
        term_means["p_fail_mean"] = p_fail_mean
        if replay:
            term_means["replaced_frac"] = torch.stack(im["Reps"]).float().mean()  # share of the imagined steps that were continued from the reference
        with torch.no_grad():
            term_means["jl_soft"] = mdp.joint_violation(S_all[4])[0]  # fraction of controlled joints beyond the soft limits (after the clamp)
            term_means["jl_hard_preclamp"] = torch.stack(im["Hard"]).mean()  # ... beyond the hard range BEFORE the clamp
        return loss.detach(), v_loss.detach(), (r_sum / HZ / mdp.dt).mean(), gn.detach(), cT.mean().detach(), term_means

    # ------------------------------------------------------------------------------------------------
    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        """Train until iteration ``num_learning_iterations`` (total, so a resumed run continues)."""
        if not self.training_ready:
            self.setup_training()
        if self.log_dir is not None:
            store_code_state(self.log_dir, self.git_status_repos)
        c = self.cfg
        stats = {"r": [], "ep_len": [], "term": [], "done": [], "jl_soft": [], "rec_episodes": [], "rec_samples": []}
        self.collect(c["warmup_steps"], stats)  # also refills the (unsaved) replay buffer after a resume
        t_start = time.time() - self.wall_offset
        t_log = time.time()
        step_acc = 0.0
        wm_keys = ("wm", "wm8", "bce", "fail_win_sampled", "fail_win_valid", *(("fail_loss", "fail_batch_pos") if c["fail_head"] else ()))
        agg: dict[str, list] = {k: [] for k in ("wm", "wm8", "bce", "fail_win_sampled", "fail_win_valid", "fail_loss", "fail_batch_pos", "pi", "v", "ir", "gn", "surv")}
        terms_acc: dict[str, list] = {}
        planner_acc: dict[str, list] = {}
        planner_acc: dict[str, list] = {}
        while self.it < num_learning_iterations:
            it = self.it
            lr = self.lr_at(it)
            for o in (self.opt_wm, self.opt_pi, self.opt_v, self.opt_fail):
                for gparam in o.param_groups:
                    gparam["lr"] = lr * (c["pi_lr_scale"] if o is self.opt_pi else 1.0)
            step_acc += 1.0 / c["utd"]
            n = int(step_acc)
            step_acc -= n
            self.collect(n, stats)
            for _ in range(c["wm_batches"]):
                m = self.wm_update()
            for k_ in wm_keys:
                agg[k_].append(m[k_])
            if self.planner_trainer is not None:
                planner_losses = self.planner_update()
                if planner_losses is not None:
                    for k_, v_ in planner_losses.items():
                        planner_acc.setdefault(k_, []).append(v_)
            if it >= c["wm_only_iters"]:
                for _ in range(c["policy_updates_per_iter"]):
                    pl, vl, ir, gn, surv, terms = self.policy_update()
                    agg["pi"].append(pl), agg["v"].append(vl), agg["ir"].append(ir), agg["gn"].append(gn), agg["surv"].append(surv)
                    for k, v in terms.items():
                        terms_acc.setdefault(k, []).append(v)
            self.it += 1
            if self.it % c["log_interval"] == 0:
                mean = lambda x: float(sum(x) / len(x)) if len(x) else float("nan")  # noqa: E731
                host = to_host({"agg": agg, "stats": stats, "terms": terms_acc, "planner": planner_acc})  # one device-to-host copy for every logged list
                agg_h, stats_h, terms_h, planner_h = host["agg"], host["stats"], host["terms"], host["planner"]
                ips = c["log_interval"] / (time.time() - t_log)
                wall_h = (time.time() - t_start) / 3600
                term_frac = float(sum(stats_h["term"]) / max(sum(stats_h["done"]), 1))
                row = {
                    "wm/loss": mean(agg_h["wm"]), "wm/err_8step": mean(agg_h["wm8"]), "wm/contact_bce": mean(agg_h["bce"]),
                    "wm/failure_windows_sampled": mean(agg_h["fail_win_sampled"]), "wm/failure_windows_valid": mean(agg_h["fail_win_valid"]),
                    "fail/bce_train": mean(agg_h["fail_loss"]), "fail/positive_share_in_batch": mean(agg_h["fail_batch_pos"]),
                    "policy/loss": mean(agg_h["pi"]), "policy/grad_norm": mean(agg_h["gn"]), "critic/value_loss": mean(agg_h["v"]),
                    "imagination/reward": mean(agg_h["ir"]), "imagination/survival": mean(agg_h["surv"]),
                    "sim/reward": mean(stats_h["r"]),
                    "sim/episode_length": mean(stats_h["ep_len"]), "sim/terminated_fraction": term_frac,
                    "perf/it_per_s": ips, "perf/lr": lr, "perf/sim_steps": float(self.sim_steps), "perf/wall_h": wall_h,
                    "perf/max_mem_gb": torch.cuda.max_memory_allocated() / 2**30,
                    "stage2/eta_obs": self.eta()[0], "stage2/eta_rew": self.eta()[1],
                    "csi/phase_mean": float(self.csi().phase.mean()), "csi/phase_min": float(self.csi().phase.min()),
                    "csi/phase_max": float(self.csi().phase.max()), "csi/success_rate": self.csi().success_rate,
                }  # fmt: skip
                if "inject" in stats:
                    row["sim/inject_rate"] = mean(stats_h["inject"])  # share of the envs moved to the reference per control step
                pen_msg = ""
                pen = getattr(self.unwrapped, "reset_penetration_stats", None)  # reset penetration rejection (task option): counts since the last log
                if pen is not None:
                    n_reset, n_resampled, n_fallback, n_ref_pen = (pen - getattr(self, "_pen_logged", torch.zeros_like(pen))).tolist()
                    self._pen_logged = pen.clone()
                    row["sim/reset_penetration/resample_rate"], row["sim/reset_penetration/fallback_rate"] = n_resampled / max(n_reset, 1), n_fallback / max(n_reset, 1)
                    row["sim/reset_penetration/reference_penetrating_rate"] = n_ref_pen / max(n_reset, 1)
                    pen_msg = f"pen {n_resampled / max(n_reset, 1):.3f}/{n_fallback / max(n_reset, 1):.4f} "  # share of resets that were resampled / fell back to the reference
                for stem, ph in zip(self.motion_names, self.csi().phase.tolist()):
                    row[f"csi_by_motion/{stem}"] = ph
                csi = self.csi()
                if csi.return_sampling:  # return-based clip sampling: how concentrated the clip distribution is and the return estimate per clip family
                    p_c = csi.clip_p
                    row["clip_sampling/p_entropy"], row["clip_sampling/p_entropy_uniform"] = float(-(p_c * p_c.clamp(min=1e-12).log()).sum()), math.log(len(p_c))
                    row["clip_sampling/p_max"], row["clip_sampling/G_mean"] = float(p_c.max()), float(csi.clip_G.mean())
                    for i, name in enumerate(self.family_names):
                        fam = self.clip_family == i
                        row[f"clip_sampling/G_mean/{name}"], row[f"clip_sampling/p_mass/{name}"] = float(csi.clip_G[fam].mean()), float(p_c[fam].sum())
                if csi.bins:  # SuperMimic reset bins: concentration of the bin distribution, mass and mean return per clip family
                    p_b = csi.bin_p
                    row["bins/p_entropy"], row["bins/p_entropy_uniform"] = float(-(p_b * p_b.clamp(min=1e-12).log()).sum()), math.log(len(p_b))
                    row["bins/p_max"], row["bins/G_mean"] = float(p_b.max()), float(csi.bin_G.mean())
                    if csi.difficulty_k > 0:
                        for i, name in enumerate(self.family_names):
                            row[f"bins/difficulty_mean/{name}"] = float((1.0 + csi.difficulty_k * csi.clip_fail)[self.clip_family == i].mean())
                    for i, name in enumerate(self.family_names):
                        fam = self.clip_family[csi.bin_clip] == i
                        row[f"bins/G_mean/{name}"], row[f"bins/p_mass/{name}"] = float(csi.bin_G[fam].mean()), float(p_b[fam].sum())
                for k, v in terms_h.items():
                    row[f"imagination/{'joint_limit/' + k if k.startswith('jl_') else 'reward_term/' + k}"] = mean(v)
                row["sim/joint_limit/jl_soft"] = mean(stats_h["jl_soft"])
                for k, v in stats_h.items():
                    if k.startswith("rt/"):
                        row[f"sim/reward_term/{k[3:]}"] = mean(v)  # the env's weighted per-term step reward of the collected transitions
                planner_msg = ""
                if self.planner_trainer is not None:
                    row.update({f"planner/{k}": mean(v) for k, v in planner_h.items()})
                    row["planner/replay_size"] = float(len(self.replay))
                    row["planner/optimizer_steps"] = float(self.planner_trainer.num_updates)
                    row["planner/recorded_episodes"] = float(sum(stats_h["rec_episodes"]))
                    row["planner/recorded_samples"] = float(sum(stats_h["rec_samples"]))
                    if len(self.replay) > 0:
                        row["planner/replay_positive_rate"] = float(self.replay.sample(4096)["target_flag"].float().mean())  # share of (step, body) labels with a target
                    planner_msg = (f"| planner replay {len(self.replay)} ep {int(sum(stats_h['rec_episodes']))} loss_aff {row.get('planner/affordance', float('nan')):.3f} "
                                   f"occ {row.get('planner/occupancy', float('nan')):.3f} acc {row.get('planner/occ_acc', float('nan')):.3f} ")
                if self.writer is not None:
                    for k, v in row.items():
                        if not math.isnan(v):
                            self.writer.add_scalar(k, v, self.it)
                    self.writer.flush()
                print(f"it {self.it} | {wall_h:.3f} h | wm {row['wm/loss']:.4f} e8 {row['wm/err_8step']:.4f} bce {row['wm/contact_bce']:.4f} "
                      f"| pi {row['policy/loss']:.3f} v {row['critic/value_loss']:.4f} "
                      f"imag_r {row['imagination/reward']:.3f} surv {row['imagination/survival']:.3f} | sim_r {row['sim/reward']:.3f} "
                      f"ep_len {row['sim/episode_length']:.1f} term {term_frac:.2f} "
                      f"csi {row['csi/phase_mean']:.3f} eta {row['stage2/eta_obs']:.2f}/{row['stage2/eta_rew']:.2f} {pen_msg}{planner_msg}| {ips:.2f} it/s mem {row['perf/max_mem_gb']:.1f}G", flush=True)  # fmt: skip
                t_log = time.time()
                stats = {k: [] for k in stats}  # (keeps the per-term keys)
                agg = {k: [] for k in agg}
                terms_acc = {}
                planner_acc = {}
            if self.it % c["save_interval"] == 0 or self.it == num_learning_iterations:
                self.wall_offset = time.time() - t_start
                self.save(os.path.join(self.log_dir, f"model_{self.it}.pt"))

    # ------------------------------------------------------------------------------------------------
    @staticmethod
    def _sidecar(path: str) -> str:
        """``.../model_<it>.pt`` -> ``.../acp_state_<it>.pt`` (does not match play's ``model_.*.pt`` lookup)."""
        d, name = os.path.split(path)
        return os.path.join(d, "acp_state_" + name[len("model_"):])

    def add_git_repo_to_log(self, repo_path: str) -> None:
        """Also store the git status / diff of the repo containing ``repo_path`` with the run (``learn`` writes them)."""
        self.git_status_repos.append(repo_path)

    def save(self, path: str, infos=None) -> None:
        """``path``: ``OnPolicyRunner`` checkpoint format (directly playable). Sidecar: world model, failure head, critic, their optimizers
        and the training state (resume)."""
        run_state_dict = {"model_state_dict": self.ac.state_dict(), "optimizer_state_dict": self.opt_pi.state_dict()}
        run_state_dict.update({f"{group}_normalizer_state_dict": n.state_dict() for group, n in self.normalizers.items()})
        run_state_dict.update({"iter": self.it, "infos": infos, "has_planner": self.planner_trainer is not None})
        if self.planner is not None:
            run_state_dict["planner_state_dict"] = self.planner.state_dict()  # play needs the planner of a Stage 2 policy
        torch.save(run_state_dict, path)
        torch.save({
            "wm": self.wm.state_dict(), "critic": self.critic.state_dict(),
            "opt_wm": self.opt_wm.state_dict(), "opt_v": self.opt_v.state_dict(), "iteration": self.it,
            "wall_time": self.wall_offset, "sim_steps": self.sim_steps, "cfg": self.cfg,
            "xi_mean": self.mdp.xi_mean, "xi_scale": self.mdp.xi_scale,
            "fail_head": self.fail_head.state_dict(), "opt_fail": self.opt_fail.state_dict(), "fail_head_start_it": self.fail_head_start_it,
            "csi_phase": self.csi().phase, "csi_success_rate": self.csi().success_rate, "motion_names": self.motion_names,
            **({"csi_clip_G": self.csi().clip_G, "csi_clip_h": self.csi().clip_h} if self.csi().return_sampling else {}),
            **({"bin_G": self.csi().bin_G, "bin_w_avg": self.csi().bin_w_avg, "bin_clip": self.csi().bin_clip, "bin_local": self.csi().bin_local, "bins_v2": self.csi().bins_v2, "clip_fail": self.csi().clip_fail} if self.csi().bins else {}),
            **({"planner_trainer": self.planner_trainer.state_dict()} if self.planner_trainer is not None else {}),
        }, self._sidecar(path))  # fmt: skip

    @staticmethod
    def iteration_after_load(restart_iteration: bool, checkpoint: dict) -> int:
        """Iteration a run continues at after loading ``checkpoint``: 0 on the explicit transition from a checkpoint without a trained planner
        (Stage 1) into a stage that restarts its counters (``restart_iteration``), else the checkpoint's own (same-stage resume)."""
        return 0 if restart_iteration and not checkpoint.get("has_planner", False) else checkpoint["iter"]

    def load(self, path: str):
        """Load the policy (actor-critic, its optimizer, the normalizers, the iteration): all that play needs. The sidecar is read by
        ``setup_training`` (``learn``)."""
        loaded_dict = torch.load(path, weights_only=True)
        self.ac.load_state_dict(loaded_dict["model_state_dict"])
        self.opt_pi.load_state_dict(loaded_dict["optimizer_state_dict"])
        for group_name, normalizer in self.normalizers.items():
            normalizer.load_state_dict(loaded_dict[f"{group_name}_normalizer_state_dict"])
        if self.planner is not None and "planner_state_dict" in loaded_dict:  # a Stage 1 checkpoint has no planner
            self.planner.load_state_dict(loaded_dict["planner_state_dict"])
        self.checkpoint_it = loaded_dict["iter"]
        # a new stage counts its iterations (lr, eta schedules) from 0: only when the checkpoint is of the previous stage (it has no trained
        # planner; a Stage 2 checkpoint resumes at its own iteration)
        self.it = self.iteration_after_load(self.cfg["restart_iteration"], loaded_dict)
        self.resume_path = path
        return loaded_dict["infos"]

    def _load_sidecar(self, path: str) -> None:
        """Resume the training state from the sidecar of the checkpoint loaded by ``load`` (strict)."""
        ck = torch.load(self._sidecar(path), map_location=self.device, weights_only=False)
        self.wm.load_state_dict(ck["wm"]), self.critic.load_state_dict(ck["critic"])
        self.opt_wm.load_state_dict(ck["opt_wm"]), self.opt_v.load_state_dict(ck["opt_v"])
        self.wall_offset, self.sim_steps = ck["wall_time"], ck["sim_steps"]
        assert self.checkpoint_it == ck["iteration"]
        self.mdp.xi_mean, self.mdp.xi_scale = ck["xi_mean"].to(self.device), ck["xi_scale"].to(self.device)
        self.fail_head.load_state_dict(ck["fail_head"]), self.opt_fail.load_state_dict(ck["opt_fail"])
        self.fail_head_start_it = ck["fail_head_start_it"]
        csi = self.csi()  # the per-motion initialization phases of the Curriculum State Initialization continue where they were
        saved = ck["csi_phase"].to(self.device)
        assert len(set(self.motion_names)) == len(self.motion_names), "clip file names (stems) must be unique: the CSI phases are matched by name"
        if "motion_names" in ck:  # clips are matched by name: a dataset with added (or removed) clips resumes with the phases of the clips it shares; new clips start at the maximum
            phase = torch.full((len(self.motion_names),), csi.max_phase, device=self.device)
            index = {n: i for i, n in enumerate(ck["motion_names"])}
            if self.cfg["clip_aliases"]:
                for old, new in yaml.safe_load(open(os.path.expanduser(self.cfg["clip_aliases"]))).items():
                    if old in index and new not in index:
                        index[new] = index[old]
            for j, n in enumerate(self.motion_names):
                if n in index:
                    phase[j] = saved[index[n]]
            print(f"[resume] CSI phases of {sum(n in index for n in self.motion_names)} of {len(self.motion_names)} clips restored; new clips: {[n for n in self.motion_names if n not in index]}", flush=True)
            csi.phase = phase
        else:
            assert len(saved) == len(self.motion_names), "the sidecar has no motion names and the dataset has a different number of clips (migrate_sidecar_names.py)"
            csi.phase = saved
        csi.success_rate = ck["csi_success_rate"]
        if csi.bins and "bin_G" in ck and ck.get("bins_v2", False) == self.cfg["bins_v2"]:  # (bins of another layout are not restored) the bins are matched by clip name and bin index (a sidecar of the CSI runs has none: the bins start at G_b = 0)
            saved_names = ck["motion_names"]
            current = {(self.motion_names[c_], b_): i for i, (c_, b_) in enumerate(zip(csi.bin_clip.tolist(), csi.bin_local.tolist()))}
            for j, (c_, b_) in enumerate(zip(ck["bin_clip"].tolist(), ck["bin_local"].tolist())):
                i = current.get((saved_names[c_], b_))
                if i is not None:
                    csi.bin_G[i], csi.bin_w_avg[i] = ck["bin_G"][j], ck["bin_w_avg"][j]
        if csi.bins and "clip_fail" in ck:  # the failure rates of the clips, by clip name
            index = {n: i for i, n in enumerate(ck["motion_names"])}
            for j, n in enumerate(self.motion_names):
                if n in index:
                    csi.clip_fail[j] = ck["clip_fail"][index[n]]
        if csi.return_sampling and "csi_clip_G" in ck:  # the return estimates of the clips (same dataset)
            assert len(ck["csi_clip_G"]) == len(self.motion_names), "the return estimates of the sidecar belong to another dataset"
            csi.clip_G, csi.clip_h = ck["csi_clip_G"].to(self.device), ck["csi_clip_h"].to(self.device)
        if self.planner_trainer is not None and "planner_trainer" in ck:  # a Stage 1 sidecar has no planner: it starts from its initial weights
            self.planner_trainer.load_state_dict(ck["planner_trainer"])

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
        """Observation (flat ``policy`` group) -> action, as ``OnPolicyRunner.get_inference_policy``. With ``contact_source`` != "reference"
        (Stage 2 / 3) the contact targets of the observation are replaced by the planner's from the env's current planner observation
        (full planner -> tracker; the Stage 1 policy reads the reference targets of the env)."""
        self.eval_mode()
        if device is not None:
            self.ac.to(device)
            if self.planner is not None:
                self.planner.to(device)
        self.normalizers["policy"].to(device)
        if self.planner is None:
            return lambda x: self.ac.act_inference(self.normalizers["policy"](x))

        def policy(x):
            x, _ = self.plan_contacts(x, self.env.get_observations()[1]["observations"]["planner"], 1.0)
            return self.ac.act_inference(self.normalizers["policy"](x))

        return policy

    def deploy_permutations(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Index tensors between the env joint order and ``DEPLOY_POLICY_JOINT_ORDER`` of the 23 controlled joints:
        ``obs_env = obs_deploy[:, idx_in]`` (every 23-block of the joint_pos / joint_vel / last_action histories) and
        ``action_deploy = action_env[:, idx_out]``."""
        from instinct_mj.tasks.acp.config.g1 import DEPLOY_POLICY_JOINT_ORDER

        robot_joints = list(self.unwrapped.scene["robot"].joint_names)
        env_names = [robot_joints[i] for i in self.unwrapped.action_manager.get_term("joint_pos").target_ids.tolist()]
        assert sorted(env_names) == sorted(DEPLOY_POLICY_JOINT_ORDER)
        env_of_deploy = [env_names.index(n) for n in DEPLOY_POLICY_JOINT_ORDER]  # action_deploy[k] = action_env[env_of_deploy[k]]
        deploy_of_env = [DEPLOY_POLICY_JOINT_ORDER.index(n) for n in env_names]  # obs_env[j] = obs_deploy[deploy_of_env[j]]
        idx_in = torch.arange(POLICY_OBS_DIM)
        for term in (2, 3, 4):  # joint_pos, joint_vel, last_action: 5 steps x 23, after base_ang_vel and projected_gravity (5 x 3 each)
            start = 5 * 3 * 2 + (term - 2) * 5 * ACT_DIM
            for step in range(5):
                b = start + step * ACT_DIM
                idx_in[b : b + ACT_DIM] = b + torch.tensor(deploy_of_env)
        return idx_in, torch.tensor(env_of_deploy)

    def export_as_onnx(self, obs, export_model_dir: str) -> None:
        """``actor.onnx`` (468 -> 23, ``ac.export_as_onnx``) and ``policy_normalizer.npz`` in ``export_model_dir``. With
        ``onnx_deploy_joint_order`` the actor.onnx instead takes the RAW policy observation with the joint blocks in the deploy joint
        order (``DEPLOY_POLICY_JOINT_ORDER``) and returns the actions in that order; the observation normalizer is part of the graph.

        With a planner (``contact_source`` != "reference") ``policy.onnx`` is written instead: the authors' interface, input ``planner_obs``
        (939 = 375 proprioception history | 561 planner points | 3 goal), outputs ``actions`` (23), ``contacts`` (9 x 3, base-yaw frame, zero
        where unoccupied) and ``occupancy`` (9 x 1 probability); the planner (top-1 decode, occupancy > 0.5), the tracker-scan crop of the
        points, the observation normalizer and the actor are one graph. ``onnx_deploy_joint_order`` permutes the proprioception input and
        the actions."""
        self.eval_mode()
        if self.planner is not None:
            self.export_policy_onnx(obs, export_model_dir)
            return
        if not self.cfg["onnx_deploy_joint_order"]:
            obs = self.normalizers["policy"](obs)
            self.normalizers["policy"].export(os.path.join(export_model_dir, "policy_normalizer.npz"))
            self.ac.export_as_onnx(obs, export_model_dir)
            return
        idx_in, idx_out = (i.to(obs.device) for i in self.deploy_permutations())

        class DeployActor(nn.Module):
            def __init__(self, normalizer, actor):
                super().__init__()
                self.normalizer, self.actor = normalizer, actor

            def forward(self, x):
                return self.actor(self.normalizer(x[:, idx_in]))[:, idx_out]

        # the observation is fed in the deploy joint order: permute the example so that the traced graph is exercised on it
        example = obs[:1][:, torch.argsort(idx_in)]
        with torch.no_grad():
            torch.onnx.export(
                DeployActor(self.normalizers["policy"], self.ac.actor), example, os.path.join(export_model_dir, "actor.onnx"),
                input_names=["input"], output_names=["output"], opset_version=18,
            )  # fmt: skip
        print(f"Exported the ACP actor (deploy joint order, normalizer inside) to {os.path.join(export_model_dir, 'actor.onnx')}")

    def policy_graph(self, deploy_order: bool) -> nn.Module:
        """Module ``planner_obs`` (N, 939) -> (actions, contacts, occupancy) with the planner in the loop (see :meth:`export_as_onnx`)."""
        idx_in, idx_out = self.deploy_permutations() if deploy_order else (torch.arange(POLICY_OBS_DIM), torch.arange(ACT_DIM))
        planner, normalizer, actor = self.planner, self.normalizers["policy"], self.ac.actor
        device = next(planner.parameters()).device
        idx_proprio, idx_out = idx_in[:PLANNER_PROPRIO].to(device), idx_out.to(device)  # the joint blocks all lie in the first 375 entries

        class PolicyGraph(nn.Module):
            def __init__(self):
                super().__init__()
                self.planner, self.normalizer, self.actor = planner, normalizer, actor

            def forward(self, planner_obs):
                proprio = planner_obs[:, :PLANNER_PROPRIO][:, idx_proprio]
                points = planner_obs[:, PLANNER_PROPRIO : PLANNER_PROPRIO + 3 * SCAN_DIM].reshape(-1, SCAN_DIM, 3)
                goal = planner_obs[:, PLANNER_PROPRIO + 3 * SCAN_DIM :]
                out = self.planner(proprio, goal, points)
                top = out["alpha"].argmax(-1)  # k = 1 decode: the point of the largest affordance weight
                physical = points_physical(points)
                target = torch.gather(physical.unsqueeze(1).expand(-1, out["alpha"].shape[1], -1, -1), 2, top[..., None, None].expand(-1, -1, 1, 3)).squeeze(2)
                occupancy = torch.sigmoid(out["occ_logit"])
                target = target * (occupancy > 0.5).unsqueeze(-1).to(target.dtype)
                scan = points[:, :, 2].reshape(-1, SCAN_NY, SCAN_NX)[:, TRACKER_Y, TRACKER_X].flatten(1)  # central 9 x 7 block (z channel)
                obs = torch.cat((proprio, scan, target.flatten(1), goal), -1)
                return self.actor(self.normalizer(obs))[:, idx_out], target, occupancy.unsqueeze(-1)

        return PolicyGraph()

    def export_policy_onnx(self, obs, export_model_dir: str) -> None:
        """Write ``policy.onnx`` (see :meth:`export_as_onnx`); ``obs``: the policy observation of the env (the example input is built from the
        env's planner observation)."""
        deploy_order = self.cfg["onnx_deploy_joint_order"]
        graph = self.policy_graph(deploy_order).eval()
        planner_obs = self.env.get_observations()[1]["observations"]["planner"][:1]
        example = torch.cat((obs[:1, :PLANNER_PROPRIO], planner_obs, obs[:1, GOAL_SLICE]), -1)
        if deploy_order:
            example = torch.cat((example[:, :PLANNER_PROPRIO][:, torch.argsort(self.deploy_permutations()[0][:PLANNER_PROPRIO])], example[:, PLANNER_PROPRIO:]), -1)
        with torch.no_grad():
            torch.onnx.export(
                graph, example, os.path.join(export_model_dir, "policy.onnx"),
                input_names=["planner_obs"], output_names=["actions", "contacts", "occupancy"], opset_version=18,
            )  # fmt: skip
        print(f"Exported the ACP policy with the planner (939 -> 23 + contacts + occupancy) to {os.path.join(export_model_dir, 'policy.onnx')}")
