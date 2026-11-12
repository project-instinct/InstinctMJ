"""SuperMimic training of ACP Stage 1: collect -> fit world model -> BPTT policy + critic update.

Adapted from ``instinct_mj.rl.supermimic.runner`` (SuperMimic reference implementation, MIT License, Chong Zhang / ETH Zurich).

The simulator is the unchanged mjlab ``Instinct-ACP-Track-G1-v1`` env wrapped by ``InstinctRlVecEnvWrapper`` (DR, resets, SuperMimic reset-bin
start frames, terminations and timeouts stay the task's). Rewards / terminations / observations in imagination are the analytic
heads of ``mdp.py`` (Table 3 rewards on imagined states by FK). The policy IS the task's ``ActorCritic`` actor (MLP on the 468-dim
``policy`` group); the runner's own critic lives on privileged features. The world model predicts the pelvis-rooted robot state and the
logits of the 9 contact links (label: the env's c_meas); its context is the 17 x 11 pelvis scan and the previous action (the action
delay of the DR changes the effective target within a step).

The loop follows the official HybridRobotics/supermimic and has no switches for it (see ``config.py`` and ``PLAN.md``, "official supermimic is the only path"): semi-implicit gencoord
world model (GELU, no output de-normalization, Bryson loss on the gencoord loss blocks, complete windows), Muon + fused Adam, LN + GELU critic on raw features with
lambda-return targets clamped at the analytic v_max, one normalized soft-termination sigmoid, ``ReturnSampler`` reset bins, DR-range context encoding, replay ratio, and a reference
that stays on the last frame of its clip in the imagination.

Stage 2 hooks (``contact_source``): per env and imagined step the contact targets of the observation are the frozen planner's with
probability ``eta_obs`` (paper Eq. 3); the reward weights follow ``eta_rew``. ``planner_update`` (hindsight training) is a stub.

The runner replaces ``OnPolicyRunner`` as the task's ``runner_entry_point`` (same public API as ``SuperMimicRunner``).
"""

from __future__ import annotations

import copy
import itertools
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
from instinct_mj.rl.supermimic.networks import Critic, WorldModel
from instinct_mj.rl.supermimic.runner import CONTACT_LOSS_WEIGHT

from instinct_mj.tasks.acp.mdp.commands import yaw_of_quat

from .buffer import ACPReplayBuffer
from .geo_contact import GeometricContact
from .sdf_contact import get_sdf_contact
from .mdp import ACT_DIM, NUM_J, ODO_HIST, POLICY_OBS_DIM, PROPRIO_DIM, SCAN_NX, SCAN_NY, TRACKER_X, TRACKER_Y, make_xi, planned_contact_error, split_xi_dyn, targets_to_world
from .motion import NUM_CONTACT, PELVIS
from .rays import ScanFloorRunningNorm
from .term_head import TerminationHead
from .planner import CONTACT_LINKS, AttentionContactPlanner, points_physical
from .hindsight import HindsightReplay
from .stage2 import EpisodeRecorder, PartitionedReplay, PlannerTrainer, add_episodes, eta as eta_schedule, label_episode_samples, mix_contacts
from .sim import ACPSimReader, build_mdp, dr_range_xi_stats
from . import wm_features
from . import wmres
from .wm_contact_inputs import ContactInputWorldModel, ContactInputs
from .optim import CapturedUpdate, MuonAdam, graph_state, set_lr, step_if_finite

SCAN_DIM = SCAN_NX * SCAN_NY
PLANNER_PROPRIO = 5 * PROPRIO_DIM  # the first 375 entries of the policy observation
TARGETS_SLICE = slice(PLANNER_PROPRIO + 63, PLANNER_PROPRIO + 63 + 27)  # contact targets of the policy observation (after the tracker scan)
GOAL_SLICE = slice(PLANNER_PROPRIO + 63 + 27, PLANNER_PROPRIO + 63 + 30)

# SuperMimic action scale alpha_j = 0.25 * effort_j / (armature_j * (2 pi 10 Hz)^2) of zita-ch/supermimic_vibed (robot.py:22-36, 60-90; BeyondMimic gains), per motor:
# (effort limit [Nm], armature [kg m^2]). The collection noise is ``action_noise`` x alpha_j on the joint target, as there (train.py:278: sigma on the raw action, then x alpha).
_ARM_5020, _ARM_7520_14, _ARM_7520_22 = 0.003609725, 0.010177520, 0.025101925
SM_MOTORS = {
    "hip_pitch": (88.0, _ARM_7520_14), "hip_roll": (139.0, _ARM_7520_22), "hip_yaw": (88.0, _ARM_7520_14), "knee": (139.0, _ARM_7520_22),
    "ankle_pitch": (50.0, 2 * _ARM_5020), "ankle_roll": (50.0, 2 * _ARM_5020), "waist_yaw": (88.0, _ARM_7520_14), "waist_roll": (50.0, 2 * _ARM_5020),
    "waist_pitch": (50.0, 2 * _ARM_5020), "shoulder_pitch": (25.0, _ARM_5020), "shoulder_roll": (25.0, _ARM_5020), "shoulder_yaw": (25.0, _ARM_5020),
    "elbow": (25.0, _ARM_5020),
}  # fmt: skip


def sm_action_scale(joint_name: str) -> float:
    effort, armature = SM_MOTORS[joint_name.replace("left_", "").replace("right_", "").replace("_joint", "")]
    return 0.25 * effort / (armature * (2.0 * math.pi * 10.0) ** 2)


ROOT_ORI_SCALE = math.sqrt(0.5)
"""Scale of the flattened root rotation matrix in the world-model loss: a plain squared difference is 0.5 ||dR||_F^2 = 2 (1 - cos theta) = theta^2 + O(theta^4), a nominal in radians
(official ``_root_loss_blocks``)."""


def wm_loss_blocks(S) -> tuple[torch.Tensor, ...]:
    """The squared-error channel groups of the official gencoord world-model loss (``GenCoordRepresentation.loss_blocks``) of a state (p, R, v, w, q, qd): the world root position
    (3), the root rotation matrix flattened x sqrt(1/2) (9, chordal) and the joint positions and velocities. The root velocities are not scored (they follow from the
    integrated pose and the velocity increments)."""
    p, R, _, _, q, qd = S
    return p, R.flatten(-2) * ROOT_ORI_SCALE, q, qd


def wm_state_error(S_hat, S1, nominal: tuple[float, float, float, float]) -> torch.Tensor:
    """(..., ) per-step Bryson loss of the official ``WorldModelLearner._compute_losses``: sum_g(mse_g w_g width_g) / sum_g width_g with w_g = 1 / nominal_g^2, the mse_g of a step
    being the mean of the squared errors over the channels of the group g (the batch / window mean is taken by the caller over the valid steps)."""
    pred, gt = wm_loss_blocks(S_hat), wm_loss_blocks(S1)
    widths = [x.shape[-1] for x in pred]
    total = 0.0
    for x, y, n in zip(pred, gt, nominal):
        total = total + ((x - y) ** 2).sum(-1) / n**2
    return total / sum(widths)


def advance_reference_frame(f: torch.Tensor, clip_end: torch.Tensor) -> torch.Tensor:
    """Reference frame of the next imagined step (official ``WorldModelEnv._compute_commands``): one frame ahead, and once the clip end (its last valid frame) is reached the reference stays
    on it: no masking of the later steps, no terminal at the clip end; rewards, terminations and observations keep tracking the frozen last frame."""
    return torch.minimum(f + 1, clip_end)


def episode_end_flags(terminated: torch.Tensor, time_out: torch.Tensor, clip_end: torch.Tensor, done: torch.Tensor, bad: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(fell, clip_end, timed_out) of the step for ``ReturnBins.record`` (official ``supermimic_runner.py``: ``terminated & ~truncated``, ``clip_end``, ``truncated & ~clip_end``). A step
    that both times out (episode cap or illegal-reset guard: ``time_out``; the clip end: ``clip_end``) and terminates counts as the time-out. ``done`` guards the clip end of a
    step that did not end the episode. A non-finite blown-up simulator step (``bad``) is a failure whatever else happened."""
    hit = clip_end & done
    truncated = time_out | hit
    fell = (terminated & ~truncated) | bad
    return fell, hit & ~bad, truncated & ~hit & ~bad


CLIP_FAMILIES = ("walk", "get_up", "stairs", "box", "climb", "ladder", "vault", "gap", "sparse")


def clip_family(group: str) -> str:
    """Clip family of a dataset group ``<skill>__<source>`` (kimodo / numbered / new variants collapse into the skill: walk*, stairs*, box*, ladder*, vault*,
    gap*, sparse*, ...); stones / beam / stakes / pallets are the "stepping" family."""
    skill = group.split("__")[0]
    return next((f for f in CLIP_FAMILIES if skill.startswith(f)), "stepping")


def lambda_return_targets(rews: torch.Tensor, dones: torch.Tensor, V: torch.Tensor, gamma: float, lam: float) -> torch.Tensor:
    """GAE(lambda) critic targets (H, B) of the rewards ``rews`` and continuation ``1 - dones``, bootstrapped with ``V`` (H + 1, B) (official ``gae_returns``; clamp the
    targets afterwards at v_max)."""
    R = V[-1]
    targets = []
    for t in range(rews.shape[0] - 1, -1, -1):
        R = rews[t] + gamma * (1 - dones[t]) * ((1 - lam) * V[t + 1] + lam * R)
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

    MAX_SKIPPED_STREAK = 1000
    """``learn`` raises after this many consecutive iterations without a complete world-model window (the episodes are shorter than ``wm_window`` and stay so)."""

    needs_contact = False
    """The measured contact of the transitions (``ACPSimReader.contact``) is read by a subclass (Stage 3: slippage / stumbling inputs). Otherwise it is read only for the
    world model's contact head (``wm_contact_loss``, the "force_head" mode); in the "sdf" mode of Stages 1 / 2 it is not computed (a second SDF evaluation of every step)."""

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
        self.opt_pi = MuonAdam([self.ac], c["lr"] * c["pi_lr_scale"], betas=(0.9, 0.95), adam_cls=torch.optim.AdamW, weight_decay=0.0)  # Muon on the hidden Linear weights, fused capturable AdamW on the rest
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
        self.num_skipped_iterations = 0  # iterations of ``learn`` without a complete world-model window: no WM / policy update, ``it`` not advanced (see ``learn``)
        self._skipped_streak = 0
        self.git_status_repos = [instinct_rl.__file__]
        self.resume_path: str | None = None
        self.checkpoint_it = 0
        self.training_ready = False
        self.writer = None
        _, _ = env.reset()

    def _skip_env_observation_group(self, group: str) -> None:
        """Training only: the env's observation ``group`` returns a constant zero tensor of its shape instead of being computed (no noise, no
        random numbers, nothing reads it; play / export build their own env)."""
        om = self.unwrapped.observation_manager
        if group not in om._group_obs_term_names:
            return
        zeros = torch.zeros_like(om.compute_group(group))
        compute_group = om.compute_group

        def skipped(group_name, update_history=False, env_ids=None):
            return zeros if group_name == group else compute_group(group_name, update_history, env_ids)

        om.compute_group = skipped

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
        if c["replay_ratio"] <= 0:
            raise ValueError(f"replay_ratio must be > 0, got {c['replay_ratio']}")
        # official supermimic Ratio: updates per vectorized step = N * replay_ratio / (wm_batches * wm_batch * wm_window)
        self.utd = self.N * c["replay_ratio"] / (c["wm_batches"] * c["wm_batch"] * c["wm_window"])
        print(f"[schedule] replay_ratio {c['replay_ratio']}: utd = {self.utd:.4f} updates per vectorized step ({self.N} envs)", flush=True)
        if c["soft_term_softness"] <= 0:
            raise ValueError(f"soft_term_softness must be > 0, got {c['soft_term_softness']}")
        if c["cuda_graph_wm"] and (c["wm_ref_contacts"] or c["wm_contact_feedback"] or c["term_source"] != "analytic"):
            raise ValueError("cuda_graph_wm needs no contact-input world model, term_source='analytic'")
        if c.get("cuda_graph_policy", False):
            self.check_policy_graph_options()
        if c["term_source"] not in ("analytic", "learned", "union"):
            raise ValueError(f"term_source must be 'analytic', 'learned' or 'union', got {c['term_source']!r}")
        self.unwrapped.cfg.auto_reset = False
        if self.planner is not None:  # training / imagination: the compiled (fused) planner attention, same math as nn.MultiheadAttention
            self.planner.compile_forward()
        if c.get("skip_env_critic_obs", True):  # the env's "critic" group is never read in training (the critic features are analytic), ~20 % of a collection step
            self._skip_env_observation_group("critic")
        torch.manual_seed(c["seed"])
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        networks.AMP_DTYPE = None  # TF32 matmuls (flags above), no bf16 autocast of the MLPs
        self.pol_norm.eval()  # statistics are updated explicitly from collected real observations
        self.motion, self.kin, self.scanner, self.mdp = self.build_analytic_mdp()
        action_term = self.unwrapped.action_manager.get_term("joint_pos")
        alpha = torch.tensor([sm_action_scale(n) for n in action_term.target_names], device=device)
        self.action_noise_std = c["action_noise"] * alpha / self.mdp.act_scale_j[self.mdp.act_ids]  # in our action units (target = default + act_scale * a)
        if c["geo_contact"]:
            self.mdp.geo = GeometricContact(self.scanner, self.unwrapped.scene["motion_reference"].cfg.robot_model_path, list(CONTACT_LINKS), device,
                                            d0=c["geo_contact_d0"], tau=c["geo_contact_tau"])
        if c["contact_detect"] not in ("sdf", "force_head"):
            raise ValueError(f"contact_detect must be 'sdf' or 'force_head', got {c['contact_detect']!r}")
        if c["contact_detect"] != getattr(self.unwrapped.cfg, "contact_detect", "force_head"):
            raise ValueError(f"agent contact_detect={c['contact_detect']!r} but the env cfg has contact_detect={getattr(self.unwrapped.cfg, 'contact_detect', 'force_head')!r}")
        self.sdf = None  # SdfContact: the contact decision of the env and the imagination in the "sdf" mode
        if c["contact_detect"] == "sdf":
            if c["geo_contact"] or c["wm_contact_feedback"]:
                raise ValueError("contact_detect='sdf' replaces geo_contact and the world model's contact head: geo_contact / wm_contact_feedback must be False")
            self.sdf = get_sdf_contact(self.unwrapped, c["sdf_tau"])
        self.wm_contact_loss = (c["contact_detect"] == "force_head") if c.get("wm_contact_loss") is None else bool(c["wm_contact_loss"])
        self.mdp.soft_softness = c["soft_term_softness"]
        self.scanner.slope_gradient = c["scan_slope_gradient"]
        if c["wm_scan"] not in ("wide", "planner"):
            raise ValueError(f"wm_scan must be 'wide' or 'planner', got {c['wm_scan']!r}")
        self.mdp.wide_scan = c["wm_scan"] == "wide"
        self.mdp.stopgrad = tuple(n for n in c["bptt_stopgrad_terms"].split(",") if n)
        if unknown := set(self.mdp.stopgrad) - set(self.mdp.w):
            raise ValueError(f"bptt_stopgrad_terms: unknown reward terms {sorted(unknown)}")
        self.reader = ACPSimReader(self.unwrapped)
        self.motion_names = [os.path.splitext(os.path.basename(f))[0] for f in next(iter(self.unwrapped.scene["motion_reference"].motion_buffers.values()))._all_motion_files]
        yaml_groups = [e.get("group", "") for e in next(iter(self.unwrapped.scene["motion_reference"].motion_buffers.values())).yaml_data["motion_files"]]
        self.family_names = sorted({clip_family(g) for g in yaml_groups})
        self.clip_family = torch.tensor([self.family_names.index(clip_family(g)) for g in yaml_groups], device=device)  # family of every clip (monitors by clip group)
        self.static = self.reader.static_context(self.mdp)  # (N, XI_STATIC_DIM) startup DR context of every env
        self.mdp.xi_mean, self.mdp.xi_scale = dr_range_xi_stats(self.reader)  # official supermimic SimContext: midpoint / half-width of the env's DR ranges
        xi_n0 = self.mdp.normalize_xi(make_xi(self.static, self.reader.dynamic_context()))
        print(f"[xi] DR-range normalization: |xi_n| max {float(xi_n0.abs().max()):.3f} (<= 1 inside the ranges), constant channels {int((self.mdp.xi_scale == 1).sum())}", flush=True)
        self.hist, self.horizon = c["hist"], c["horizon"]
        assert self.hist == 5, "the policy observation has 5-step proprio histories"
        assert abs(self.motion.framerate * self.unwrapped.step_dt - 1.0) < 1e-6, "imagination advances the reference by one frame per step"
        assert c["warmup_steps"] >= max(self.horizon, c["wm_window"])
        self.wm_physics_inputs = wm_features.parse(c["wm_physics_inputs"])
        self.wm_ref_contacts = c.get("wm_ref_contacts", False)
        self.wm_contact_feedback = c.get("wm_contact_feedback", False)
        self.wm_contact_spec = ",".join(x for x, enabled in (("targets", self.wm_ref_contacts), ("ar", self.wm_contact_feedback)) if enabled)
        wm_extra = wm_features.dim(self.wm_physics_inputs)
        if wm_extra:
            print(f"[world model] extra physics inputs {self.wm_physics_inputs} ({wm_extra} dims)", flush=True)
        wm_in = 1 + self.mdp.critic_scan_dim + ACT_DIM + wm_extra  # context: pelvis height + scan + previous action (+ physics inputs)
        if self.wm_contact_spec:
            self.wm = ContactInputWorldModel(wm_in, NUM_CONTACT, self.mdp.xi_dim, spec=self.wm_contact_spec, num_joints=NUM_J, act=nn.GELU).to(device)
        else:
            self.wm = WorldModel(wm_in, NUM_CONTACT, self.mdp.xi_dim, 0, num_joints=NUM_J, act=nn.GELU).to(device)  # official build_mlp: Linear -> LayerNorm -> GELU
        self.wm.denormalize_output = False  # official Dynamics: the network predicts the physical increments directly
        self.wm.set_integrator("semi_implicit", dt=self.mdp.dt, decimation=int(self.unwrapped.cfg.decimation))  # official GenCoordRepresentation.apply_prediction
        print(f"[world model] semi-implicit integration: dt {self.mdp.dt}, substep coefficient {self.wm.substep_c:.4f}", flush=True)
        self.wm_contact_inputs = ContactInputs(self.mdp, self.kin, self.wm_contact_spec) if self.wm_contact_spec else None
        self.critic = Critic(self.mdp.critic_dim(), hidden=[int(h) for h in c["critic_hidden"].split(",")], act=nn.GELU, layer_norm=True, normalize_input=False).to(device)  # official: LN + GELU on the raw features
        self.wm_nominal = tuple(float(x) for x in c["wm_nominal"].split(","))
        assert len(self.wm_nominal) == 4, "wm_nominal: root position, root rotation, joint position, joint velocity"
        self.opt_wm = MuonAdam([self.wm], c["lr"])
        self.opt_v = MuonAdam([self.critic], c["lr"])
        self.v_max = None  # the analytic value bound of the current objective (set by every policy update)
        # the inputs contain exact height scans: floor the scan std of the input normalizers (see ScanFloorRunningNorm; no clipping)
        wm_scan = slice(self.wm.in_dim - self.mdp.critic_scan_dim - ACT_DIM - wm_extra, self.wm.in_dim - ACT_DIM - wm_extra)
        critic_scan = slice(self.mdp.critic_dim() - self.mdp.xi_dim - self.mdp.critic_scan_dim, self.mdp.critic_dim() - self.mdp.xi_dim)
        if not self.wm_contact_spec:
            self.wm.in_norm = ScanFloorRunningNorm(self.wm.in_dim, wm_scan).to(device)
        xi0 = 12 + 3 * NUM_J  # WM inputs: invariant state (12 + 2 NUM_J), action (NUM_J), xi_n, context
        self.wm.in_norm.set_passthrough(slice(xi0, xi0 + self.mdp.xi_dim))  # the DR context enters the WM as encoded by its ranges (no second, running normalization)
        # learned termination head d_ω(ŝ, g) (SuperMimic Sec. 3.2) of the "learned" / "union" imagination: built (and trained) only then, so the default runs consume the same random numbers as before
        self.term_head = self.opt_term = None
        if c["term_source"] != "analytic":
            self.term_head = TerminationHead(self.mdp.critic_dim(), critic_scan).to(device)
            self.opt_term = torch.optim.Adam(self.term_head.parameters(), lr=c["lr"])
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
        self.buffer = self.make_buffer()
        if self.wm_ref_contacts:  # the contact command the policy observation actually carried at s_t (27: reference with noise / planner / mix)
            self.buffer.data["wm_cmd"] = torch.zeros(c["buffer_steps"], self.N, 3 * NUM_CONTACT, device=device)
            self.buffer.window_keys += ("wm_cmd",)
        if self.wm_contact_feedback:
            self.buffer.data["wm_contact0"] = torch.zeros(c["buffer_steps"], self.N, NUM_CONTACT, device=device)
            self.buffer.window_keys += ("wm_contact0",)
            self.buffer.start_keys += ("wm_contact0",)
        self.buffer.clip_table = self.motion.clip_id  # global frame -> clip (clip-balanced window sampling)
        self.obs, extras = env.get_observations()
        self.planner_obs = extras["observations"]["planner"]
        self.sim_steps, self.wall_offset = 0, 0.0
        self.reset_bins = None
        if self.episode_start().mode == "bins":  # official supermimic ReturnSampler: per-step returns of the visited 1 s bins (the "start" mode of evaluation / Stage 3 has no bins)
            g_max = sum(max(w_, 0.0) for w_ in self.mdp.w.values()) * self.mdp.dt / (1 - c["gamma"])
            self.reset_bins = self.episode_start().enable_return_bins(g_max, gamma=c["gamma"], horizon=c["return_horizon"], memory=c["return_memory"], steepness=c["return_steepness"],
                                                                      explore=c["return_explore"], keep=c["return_keep"])  # fmt: skip
        # The env reward feeds the reset bins (V_b) and the Stage 2 / 3 reward sync only; the actor and the critic train on the imagined rewards. A run without bins and without a planner
        # has neither, so the env's reward manager is not evaluated there (~18 % of a collection step at 2048 envs, mostly the SDF contact of track_contact_label). The env reward is not logged.
        self.env_reward = self.uses_env_reward()
        if not self.env_reward:
            rm = self.unwrapped.reward_manager
            zero_reward = {k: torch.zeros(self.N, device=device) for k in rm.compute(dt=self.unwrapped.step_dt)}
            rm.compute = lambda *args, **kwargs: zero_reward
        self.read_contact = self.needs_contact or self.wm_contact_loss or c["contact_detect"] != "sdf"
        self.no_contact = torch.zeros(self.N, NUM_CONTACT, device=device)
        self.hist_age =self.unwrapped.episode_length_buf.clone() if c["real_ref_inject_eps"] > 0 else None  # steps since the last reset OR injection
        self.writer = SummaryWriter(log_dir=self.log_dir) if self.log_dir is not None else None
        self._build_fns()
        # (the graphs record the addresses of the tensors they read through Python objects and refuse a replay after one moved: lazily evaluated, ``self.wm`` etc. exist by then)
        self._captured_wm = CapturedUpdate(self._wm_graph_step, state=lambda: graph_state([self.wm], [self.opt_wm], grad_modules=[self.wm])) if c["cuda_graph_wm"] else None
        self._captured_policy = None
        self._policy_mode = None  # structure of the captured imagination (see ``policy_graph_mode``)
        if c.get("cuda_graph_policy", False):
            self._captured_policy = CapturedUpdate(self._policy_graph_step, state=self.policy_graph_state)
            # The actor's input normalizer rebinds its ``_std`` buffer at every update (collection): the graph would keep reading the one it captured. Keep the buffer's address.
            std_buf, update = self.pol_norm._std, self.pol_norm.update

            def update_keep_std(x, update=update, std_buf=std_buf):
                update(x)
                std_buf.copy_(self.pol_norm._std)
                self.pol_norm._std = std_buf

            self.pol_norm.update = update_keep_std
        if self.resume_path is not None:
            self._load_sidecar(self.resume_path)
            planner_opt = [self.planner_trainer.opt] if self.planner_trainer is not None else []  # (its counters too: 70 syncs per optimizer step otherwise)
            for opt in ([self.opt_term] if self.opt_term is not None else []):
                set_lr(opt, c["lr"])  # (the learning rate is constant: not the one stored in the checkpoint, e.g. of a run with the former cosine decay)
            for opt in (*planner_opt, *([self.opt_term] if self.opt_term is not None else [])):  # the Adam step counters were saved as CUDA tensors: back on the CPU (a CUDA counter makes every step sync); the capturable MuonAdam of the WM / actor / critic keeps them on the device
                for state in opt.state.values():
                    if "step" in state:
                        state["step"] = state["step"].cpu()
        if self.reset_bins is not None and self.planner is None:  # Stage 1: draw the first collected episodes from the bins too (also after loading their sidecar).
            # Snapping to a legal reference frame changes the bin's uniform start distribution; keep the drawn frame, as supermimic_vibed does.
            # The robot's reset noise / penetration rejection is unchanged. Stage 2 keeps its existing start handling.
            self.unwrapped.event_manager.get_term_cfg("episode_start").params["legal_start_max_penetration"] = None
            self.unwrapped.reset()
            self.obs, extras = env.get_observations()
            self.planner_obs = extras["observations"]["planner"]
        self.training_ready = True

    # ------------------------------------------------------------------------------------------------
    # Extension points (overridden by the Stage 3 runner for the mixed reference / goal-reaching env; the defaults are the Stage 1 / 2 behaviour)
    def uses_env_reward(self) -> bool:
        """Whether anything consumes the env's reward (``setup_training``): the reset bins (Stage 1 / 2) and, with a planner, its sync into ``r_sim``. The reward manager is not
        evaluated otherwise. (``r_sim`` is stored in the buffer's ``r`` field, which no sampler returns: ``window_keys`` / ``start_keys``.)"""
        return self.reset_bins is not None or self.planner is not None

    def build_analytic_mdp(self):
        """(motion library, kinematics, height scanner, analytic MDP) of the live env."""
        c = self.cfg
        return build_mdp(self.unwrapped, c["heightmap_resolution"], c["heightmap_cache_dir"], self.device, c["exact_scan"], c.get("lazy_exact_heightmap", False))

    def make_buffer(self) -> ACPReplayBuffer:
        c = self.cfg
        return ACPReplayBuffer(c["buffer_steps"], self.N, self.device)

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

    def term_head_windows(self, w: dict, limit: int) -> torch.Tensor:
        """Indices of the (at most ``limit``) sampled windows the termination head trains on."""
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
        extra = wm_features.dim(self.wm_physics_inputs)  # 0 unless ``wm_physics_inputs`` is set

        def _ctx(S, a, a_prev, xi_n, cmd=None, contact0=None, fk=None):
            """World-model context: the pelvis height (the root height of supermimic_vibed's invariant features), the true pelvis scan, the previous action and the optional
            physics inputs (``wm_physics_inputs``).
            ``fk``: ``kin.forward_all(*S)`` when the caller has it already (else computed here)."""
            fk = kin.forward(*S) if fk is None else tuple(x[:, :NUM_CONTACT] for x in fk)
            ctx = [fk[0][:, PELVIS, 2:3], mdp.critic_scan(fk[0][:, PELVIS], fk[1][:, PELVIS]), a_prev]
            if extra:  # the torque of the command applied at this step (a / a_prev by the delay) and the clearance of the contact links
                xi = mdp.xi_mean + xi_n * mdp.xi_scale  # inverse of mdp.normalize_xi: the raw per-episode DR context (bias, kp / kd, lag)
                ctx.append(wm_features.features(mdp, S, a, a_prev, xi, fk, self.wm_physics_inputs))
            if self.wm_contact_spec:
                ctx.append(self.wm_contact_inputs.features(S, a, a_prev, xi_n, contact0, cmd))
            return torch.cat(ctx, -1)

        def _wm_rollout(s, a, a_prev, xi_n, cmd=None, contact0=None):
            """k-step rollout from the real state: states and contact logits (B, k, ...)."""
            S = unflatten_state(s[:, 0])
            preds, logits = [], []
            for j in range(a.shape[1]):
                if self.wm_contact_spec:
                    ctx = _ctx(S, a[:, j], a_prev[:, j], xi_n, cmd[:, j] if cmd is not None else None, contact0)
                else:
                    ctx = _ctx(S, a[:, j], a_prev[:, j], xi_n)
                S, lg, _ = wm.step(S, mdp.pad(a[:, j]), xi_n, ctx)
                if self.wm_contact_feedback:
                    contact0 = torch.sigmoid(lg)  # full BPTT through contact feedback, as in e5b
                preds.append(S), logits.append(lg)
            return tuple(torch.stack([P[i] for P in preds], 1) for i in range(6)), torch.stack(logits, 1)

        nominal = self.wm_nominal

        def _wm_err(S_hat, logits, s1, contact, mask):
            # official WorldModelLearner: Bryson-weighted squared errors of the gencoord loss blocks (root position, root rotation matrix, joint position / velocity), mean over the valid steps;
            # + the ACP contact BCE
            e = wm_state_error(S_hat, unflatten_state(s1), nominal)  # (B, k)
            bce = F.binary_cross_entropy_with_logits(logits, contact, reduction="none").sum(-1) if self.wm_contact_loss else torch.zeros_like(e)  # (sdf mode: the contact is computed from the predicted state, no head loss)
            return ((e + CONTACT_LOSS_WEIGHT * bce) * mask).sum() / mask.sum().clamp(min=1.0), (e[:, -1], (bce * mask).sum() / s1.shape[0])

        def _reward_term(S, qd0, f, phase, origin, a, a_prev, a_prev2, xi, logits, lp_hist, ep0, planned, eta_rew, H, fk=None):
            """``fk``: ``kin.forward_all(*S)`` when the caller has it already (the imagination does)."""
            r, terms = mdp.reward(S, qd0, f, phase, origin, a, a_prev, a_prev2, xi, torch.sigmoid(logits), lp_hist, ep0, H, planned, eta_rew, terms=True,
                                  use_geo_contact=mdp.geo is not None, fk=fk)
            return r, mdp.soft_termination(S, f, origin, phase, fk=fk), terms

        def _critic_input(S, f, phase, origin, xi_n, gpos, gyaw, fk=None):
            return mdp.critic_features(S, f, phase, origin, xi_n, gpos, gyaw, fk=fk)

        def _observe(fk, f, phase, origin, gpos, gyaw, gt, hist, delay, drift, push: bool):
            """Policy-group observation pieces of a state (all but the proprio history; ``fk``: its ``kin.forward_all``): scan, reference contact targets and
            goal command (after the env's goal resampling) from the ESTIMATED pelvis pose (odometry delay and drift of the episode,
            built from the pelvis pose history ``hist`` as the sensor; ``push``: the state is a new state of the history)."""
            bp, bR, _, bw = mdp.base_of(fk)
            if push:
                hist = mdp.push_pose(hist, bp, bR)
            ep_, eyaw = mdp.estimated_pose(hist, delay, drift)
            eR = mdp.yaw_rotation(eyaw)
            gpos, gyaw, gt = mdp.goal_update(f, phase, origin, gpos, gyaw, gt)
            scan = mdp.scan(ep_, eR)
            tgt, valid = mdp.targets(ep_, eR, f, phase, origin)
            goal = mdp.goal(ep_, eR, gpos, gyaw)
            return bp, bR, bw, scan, tgt, valid, goal, gpos, gyaw, gt, hist

        def _push(o_hist, S, fk, a_prev, xi):
            bp, bR, _, bw = mdp.base_of(fk)
            return torch.cat((o_hist[:, 1:], mdp.add_noise(mdp.proprio(bR, bw, S[4], S[5], a_prev, xi)).unsqueeze(1)), 1)

        def _assemble(o_hist, scan, tgt, valid, goal_noisy, use_planner_tgt):
            t27 = mdp.noisy_targets(tgt, valid)
            if use_planner_tgt is not None:  # per env: the planner's targets (no noise) or the reference ones
                t27 = torch.where(use_planner_tgt[0].unsqueeze(-1), use_planner_tgt[1].flatten(1), t27)
            return mdp.policy_obs(o_hist, mdp.noisy_tracker_scan(mdp.tracker_scan(scan)), t27, goal_noisy)

        def _step_wm(S, fk, a, a_prev, xi_n, cmd=None, contact0=None):
            """One world-model step from ``S`` (``fk``: its ``kin.forward_all``) on the raw policy action. Returns the action, the predicted state, ITS ``kin.forward_all``
            (the next step's ``fk``; the sdf contact needs it anyway), the contact logits and the hard joint-limit violation."""
            ctx = _ctx(S, a, a_prev, xi_n, cmd, contact0, fk) if self.wm_contact_spec else _ctx(S, a, a_prev, xi_n, fk=fk)
            S1, logits, _ = wm.step(S, mdp.pad(a), xi_n, ctx)
            hard = mdp.joint_violation(S1[4])[1]  # fraction of the predicted joint positions beyond the hard range
            fk1 = kin.forward_all(*S1)
            if self.sdf is not None:  # sdf mode: the logits of the contact probabilities are -(d_b - d_on_b) / tau of the predicted state (replaces the head's output)
                logits = self.sdf.logits(self.sdf.distance(fk1[0][:, :NUM_CONTACT], fk1[1][:, :NUM_CONTACT]))
            return a, S1, fk1, logits, hard

        def _fk_all(S):
            """``kin.forward_all`` of a state: the FK the imagination passes on to its consumers (the start state's; the later ones come from ``step_wm``)."""
            return kin.forward_all(*S)

        def _sim_stats(S, a, a_prev, xi_n, cmd=None, contact0=None):
            ctx = _ctx(S, a, a_prev, xi_n, cmd, contact0) if self.wm_contact_spec else _ctx(S, a, a_prev, xi_n)
            return wm.inputs(S, mdp.pad(a), xi_n, ctx)

        # The captured world-model update (``cuda_graph_wm``) records the rollout and the loss inside its own CUDA graph: with ``compile`` they are compiled in the default mode first
        # (official: compile, then capture; the kernels are fused, the outer graph replaces reduce-overhead). Their compilation happens in the eager warm-up calls of CapturedUpdate.
        # A reduce-overhead function (cudagraph trees) must never run inside an outer capture, so the graph runs never use the reduce-overhead ``_wm_rollout`` below.
        self._wm_rollout_eager, self._wm_err_eager = _wm_rollout, _wm_err
        if self.cfg["compile"] and self.cfg["cuda_graph_wm"]:
            self._wm_rollout_eager, self._wm_err_eager = torch.compile(_wm_rollout), torch.compile(_wm_err)
        if self.cfg["compile"] and not self.cfg["cuda_graph_wm"]:
            _wm_rollout = torch.compile(_wm_rollout, mode="reduce-overhead")
            _wm_err = torch.compile(_wm_err)
        if self.cfg["compile_policy"]:
            _reward_term = torch.compile(_reward_term, dynamic=True)
            _critic_input = torch.compile(_critic_input, dynamic=True)
            _observe, _push, _step_wm, _fk_all = torch.compile(_observe), torch.compile(_push), torch.compile(_step_wm, mode=None if self.cfg.get("cuda_graph_policy", False) else os.environ.get("ACP_STEP_WM_COMPILE_MODE", "reduce-overhead")), torch.compile(_fk_all, dynamic=True)
            _sim_stats = torch.compile(_sim_stats)  # collection: world-model input normalizer statistics of every real transition
            self.kin.forward = torch.compile(self.kin.forward, dynamic=True)  # forward kinematics: also called uncompiled by every collection step
        self.wm_rollout, self.wm_err, self.reward_term, self.critic_input = _wm_rollout, _wm_err, _reward_term, _critic_input
        self.observe, self.push, self.assemble, self.step_wm, self.sim_stats, self.fk_all = _observe, _push, _assemble, _step_wm, _sim_stats, _fk_all

    def value_bound(self, V: torch.Tensor, v_max: float | None = None) -> torch.Tensor:
        """Critic value clipped from above at the analytic bound ``v_max`` (default ``self.v_max``, the bound of the current objective, see :meth:`update_value_bound`)."""
        return V.clamp(max=self.v_max if v_max is None else v_max)

    def update_value_bound(self, eta_rew: float) -> None:
        """v_max = (sum of the positive reward weights of the objective at ``eta_rew``) x dt / (1 - gamma) (official ``ActorCriticPolicyLearner._v_max``)."""
        self.v_max = self.mdp.max_step_reward(eta_rew) / (1 - self.cfg["gamma"])

    def episode_start(self):
        """The ``EpisodeStart`` term of the env (start frame of the episodes; owns the :class:`ReturnBins`)."""
        return self.unwrapped.curriculum_manager.get_term_cfg("episode_start").func

    def xi_of(self, env_idx: torch.Tensor, dyn: torch.Tensor) -> torch.Tensor:
        return make_xi(self.static[env_idx], dyn)

    # ------------------------------------------------------------------------------------------------
    @torch.no_grad()
    def collect(self, n_steps: int, stats: dict):
        """Step the task env with the CURRENT policy on the real policy observation + action noise. The env runs with
        ``auto_reset=False`` and finished envs are reset explicitly right after the pre-reset data has been read (see
        ``SuperMimicRunner.collect``). With ``eta_obs`` > 0 the contact targets of the observation are, per env, the whole command of the
        planner (as in imagination). With ``train_planner`` the episodes are recorded for the hindsight planner replay."""
        mdp, reader, env, am = self.mdp, self.reader, self.env, self.unwrapped.action_manager
        fps = self.motion.framerate
        c_inject = self.cfg["real_ref_inject_eps"]
        defer_norm = self.cfg.get("norm_after_collect", False)
        pending_norm = []
        for _ in range(n_steps):
            if self.cfg.get("profile_phases", False):
                torch.cuda.synchronize()
                t_cs = time.perf_counter()
            S = reader.state()
            wm_contact0 = reader.contact().clone() if self.wm_contact_feedback else None
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
                force_z, contact_pos = self.contact_measurement(lp0, ep_step == 0, fk[1])
            wm_cmd = obs_in[:, TARGETS_SLICE].clone() if self.wm_ref_contacts else None  # the contact command that was actually delivered to the tracker
            if not defer_norm:
                self.pol_norm.update(obs_in)
            a = self.ac.act_inference(self.pol_norm(obs_in))
            a = torch.clamp(a + self.action_noise_std * torch.randn_like(a), -20, 20)

            prof = self.cfg.get("profile_phases", False)
            if prof:
                torch.cuda.synchronize()
                t_c0 = time.perf_counter()
            obs1, rew, dones, extras1 = env.step(a)
            if prof:
                torch.cuda.synchronize()
                t_c1 = time.perf_counter()
            # ---- pre-reset data of this transition ----
            a = am.action.clone()
            done = dones.bool()
            terminated = self.unwrapped.reset_terminated.clone()
            time_out = self.unwrapped.reset_time_outs.bool().clone()  # the episode cap, the clip end (dataset_exhausted), the reset guard
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
            contact = reader.contact() if self.read_contact else self.no_contact
            goal1 = reader.goal(fps)
            r_sim = rew.sum(-1).clone()
            if self.planner is not None and self.env_reward:  # Stages 2 / 3: the env reward of the stage's own weights (eta_rew) and of the planner's targets, as the imagination (bins)
                r_sim = self.sync_env_reward(obs1, extras1["observations"]["planner"], S1)
            if any_bad:
                contact, goal1, r_sim = torch.where(bad[:, None], torch.zeros_like(contact), contact), torch.where(bad[:, None], goal, goal1), torch.where(bad, torch.zeros_like(r_sim), r_sim)
            f1, phase1, origin1 = self.motion.current_frame()
            clip = self.motion.clip_id[f]
            at_clip_end = f1 >= self.motion.clip_end[clip]
            if self.reset_bins is not None and self.planner is None:  # Stage 1 bins: upstream truncates on reaching the last valid frame, not one step beyond it.
                done = done | at_clip_end
            done_ids = done.nonzero(as_tuple=False).squeeze(-1)
            if self.reset_bins is not None:  # official ReturnSampler: the frame before the step, the reward and how the episode ended; a step that times out AND terminates counts as the time-out
                fell, clip_end_hit, timed_out = episode_end_flags(terminated, time_out, at_clip_end | self.unwrapped.termination_manager.get_term("dataset_exhausted").bool(), done, bad)
                start = self.motion.clip_start[clip]
                self.reset_bins.record(clip, f - start, f1 - start, r_sim, fell, clip_end_hit, timed_out)
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
                **({"wm_contact0": wm_contact0} if self.wm_contact_feedback else {}),
                **({"wm_cmd": wm_cmd} if self.wm_ref_contacts else {}),
            })  # fmt: skip
            if prof:
                torch.cuda.synchronize()
                t_c2 = time.perf_counter()
            self.sim_steps += self.N
            xi = self.xi_of(torch.arange(self.N, device=self.device), xi_dyn)
            xi_n = mdp.normalize_xi(xi)
            if self.wm_contact_spec:
                wm_in = self.sim_stats(S, a, a_prev, xi_n, wm_cmd, wm_contact0)
            else:
                wm_in = self.sim_stats(S, a, a_prev, xi_n)
            if defer_norm:
                pending_norm.append((obs_in.detach().clone(), wm_in.detach().clone()))
            else:
                self.wm.in_norm.update(wm_in)
            stats["jl_soft"].append(mdp.joint_violation(S1[4])[0])
            if prof:
                torch.cuda.synchronize()
                t_c3 = time.perf_counter()
                self._collect_prof = [a_ + b_ for a_, b_ in zip(getattr(self, "_collect_prof", [0.0] * 4)[:4], (t_c0 - t_cs, t_c1 - t_c0, t_c2 - t_c1, t_c3 - t_c2))]
            if len(done_ids) > 0:
                stats["ep_len"].append((ep_step[done] + 1).float().mean())
                stats["term"].append(terminated[done].float().sum())
                stats["done"].append(done.float().sum())
        # Keep sequential updates to preserve existing max-count boundary behavior.
        for obs_real, wm_in_real in pending_norm:
            self.pol_norm.update(obs_real)
            self.wm.in_norm.update(wm_in_real)

    @torch.no_grad()
    def sync_env_reward(self, obs1: torch.Tensor, planner_obs1: torch.Tensor, S1) -> torch.Tensor:
        """Step reward of the real transitions with the SAME objective as the imagination (``ACPMDP.reward`` with ``eta_rew``): the env's weighted per-term rewards are rescaled by
        ``weights(eta_rew) / env weight`` (the reference-contact terms and ``track_body_pos_anchor`` fade with ``1 - eta_rew``), the env's own ``track_planned_contact`` (reference
        targets; weight 0 in Stage 2, 1.5 in Stage 3) is replaced by ``1.5 eta_rew`` x the term evaluated with the PLANNER's targets of the end state ``S1`` (planner on the
        pre-reset observation of the step, the estimated pelvis pose of the same state). Replaces the raw env reward in ``r_sim`` (replay, bins)."""
        mdp, eta = self.mdp, self.eta()[1]
        w_img = mdp.weights(torch.tensor(eta, device=self.device))
        terms = self.unwrapped.reward_manager._termwise_reward_buf["rewards"]
        r = torch.zeros(self.N, device=self.device)
        if self.reset_bins is not None:  # G_max of the bins: the largest attainable discounted return of the CURRENT objective
            self.reset_bins.max_return = sum(max(float(w_), 0.0) for w_ in w_img.values()) * mdp.dt / (1 - self.cfg["gamma"])
        for n, v in terms.items():
            if n == "track_planned_contact":
                continue
            w_env, w_new = float(mdp.w[n]), float(w_img[n])
            if w_env != 0.0:
                r = r + v * (mdp.dt * w_new / w_env)  # (the termwise buffer holds weight x value, without dt)
            elif w_new != 0.0 and not getattr(self, "_warned_sync", False):
                self._warned_sync = True
                print(f"[sync_env_reward] term {n}: env weight 0 but weight {w_new} at eta_rew={eta}: cannot be recovered from the env reward (missing in r_sim)", flush=True)
        if eta > 0:
            points = planner_obs1.view(-1, SCAN_DIM, 3)
            tgt, valid = AttentionContactPlanner.decode(self.planner(obs1[:, :PLANNER_PROPRIO], obs1[:, GOAL_SLICE], points), points)
            sensor = self.unwrapped.scene["height_scanner"]
            tgt_w = targets_to_world(sensor.estimated_pos_w.clone(), yaw_of_quat(sensor.estimated_quat_w), tgt)
            bp, bR = self.kin.forward(*S1)[:2]
            err = planned_contact_error(bp, bR[:, PELVIS], tgt_w, valid)
            sigma = mdp.p["track_planned_contact"]["sigma"]
            r = r + float(w_img["track_planned_contact"]) * mdp.dt * torch.exp(-err / (sigma**2 * valid.sum(-1).float().clamp(min=1.0)))
        return r

    @torch.no_grad()
    def inject_reference(self, ids: torch.Tensor, f: torch.Tensor, origin: torch.Tensor) -> None:
        """Move the envs ``ids`` to the reference state of the frames ``f`` (root pose and velocity, joint positions and velocities) without ending their
        episode: the odometry and observation histories restart at the new pose."""
        p, R, v, w, q, qd = self.mdp.reference_state(f[ids], origin[ids])
        robot = self.unwrapped.scene["robot"]
        robot.write_root_link_pose_to_sim(torch.cat((p, mat_to_quat(R)), -1), env_ids=ids)
        robot.write_root_link_velocity_to_sim(torch.cat((v, w), -1), env_ids=ids)
        robot.write_joint_state_to_sim(q, qd, env_ids=ids)
        self.unwrapped.sim.forward()
        self.unwrapped.scene["height_scanner"].reset(ids)
        self.unwrapped.sim.sense()
        # the teleport is no step of the episode: like the env's reset path, the observation history / delay buffers of these envs restart with the
        # post-injection frame (the other envs keep their timelines) and the finite-difference state of ``contact_acc`` restarts. The commands
        # (contact targets, goal) are functions of the reference time and the current pose, so there is nothing to resample.
        self.unwrapped.reward_manager.get_term_cfg("contact_acc").func.reset(ids)
        self.unwrapped.observation_manager.reset(ids)
        self.unwrapped.observation_manager.compute(update_history=True, env_ids=ids)  # fills the cache ``env.get_observations()`` returns

    @torch.no_grad()
    def contact_measurement(self, link_pos: torch.Tensor, fresh: torch.Tensor, link_rot: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Contact of the 9 links at the current state s_t (the window of substeps that led to it): the |F_z| maximum over the window
        (``c_meas``) and the world position of the contact (the contact point of the strongest contact, measured by the solver; a body
        without a contact point at the last substep keeps its last one, and a reset env starts with the link origins ``link_pos``).
        In the "sdf" mode (needs ``link_rot``) the first return value is the contact indicator ``d_b < d_on_b`` of the state times 2 (it exceeds the 1 N threshold of
        ``label_episode`` exactly where the body is in contact) and the position is the terrain point closest to the nearest sphere of a body in contact."""
        if self.sdf is not None:
            d, pt = self.sdf.distance(link_pos, link_rot, with_point=True)
            flag = self.sdf.flags(d)
            self.last_contact_pos = torch.where(fresh.view(-1, 1, 1), link_pos, self.last_contact_pos)
            self.last_contact_pos = torch.where(flag.unsqueeze(-1), pt, self.last_contact_pos)
            return 2.0 * flag.float(), self.last_contact_pos.clone()
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
        """One world-model gradient step (official ``WorldModelLearner``) on a batch of complete k-step windows (no episode end inside; the last transition may end), plus the
        contact-head BCE. With a termination head, a second batch of masked windows (uniform over the buffer, clip balancing ``wm_balance_frac``) is
        rolled out without gradient and trains the head on the predicted states."""
        c = self.cfg
        w = self.buffer.sample_complete_windows(c["wm_batch"], c["wm_window"])
        if w is None:  # no env has a complete window yet (episodes shorter than the window): no update; ``learn`` keeps collecting (the head windows below need no extra check)
            self._report_no_complete_window()
            return None
        xi_n = self.mdp.normalize_xi(self.xi_of(w["env"], w["xi_dyn"][:, 0]))
        if c["cuda_graph_wm"]:
            out_g = self._captured_wm({"s": w["s"], "a": w["a"], "a_prev": w["a_prev"], "xi_n": xi_n, "s1": w["s1"], "contact": w["contact"], "mask": w["mask"]})
            return {"wm": out_g["loss"].clone(), "wm8": out_g["e_last"].mean(), "bce": out_g["bce"].detach().clone()}
        S_hat, logits = self._rollout_windows(w, xi_n)
        loss, (e_last, bce) = self.wm_err(S_hat, logits, w["s1"], w["contact"], w["mask"])
        self.opt_wm.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.wm.parameters(), 1.0)
        self.opt_wm.step()
        out = {"wm": loss.detach(), "wm8": e_last.detach().mean(), "bce": bce.detach()}
        if self.term_head is None:  # no head: no extra windows
            return out
        w = self.buffer.sample_windows(c["wm_batch"], c["wm_window"], None, c["wm_balance_frac"])
        xi_n = self.mdp.normalize_xi(self.xi_of(w["env"], w["xi_dyn"][:, 0]))
        with torch.no_grad():
            S_hat, _ = self._rollout_windows(w, xi_n)
        out.update(self.term_update(w, S_hat, xi_n))
        return out

    def _report_no_complete_window(self) -> None:
        """Rate-limited diagnostic (the first skipped iteration, then every 50th) of an iteration without a complete world-model window."""
        n = self.num_skipped_iterations = getattr(self, "num_skipped_iterations", 0) + 1
        if n == 1 or n % 50 == 0:
            print(f"[wm] it {self.it}: no complete {self.cfg['wm_window']}-transition window in the replay buffer: world-model and policy updates skipped (the iteration does not count), "
                  f"collecting on ({n} skipped so far; {self.buffer.episode_length_stats()})", flush=True)

    def _rollout_windows(self, w: dict, xi_n: torch.Tensor):
        """Autoregressive k-step rollout of the world model from the first state of every window: predicted states and contact logits (B, k, ...)."""
        if self.wm_contact_spec:
            return self.wm_rollout(w["s"], w["a"], w["a_prev"], xi_n, w.get("wm_cmd"), w["wm_contact0"][:, 0] if self.wm_contact_feedback else None)
        return self.wm_rollout(w["s"], w["a"], w["a_prev"], xi_n)

    def _wm_graph_step(self, x: dict) -> dict:
        """The capture-safe world-model gradient step of ``cuda_graph_wm`` (official ``WorldModelLearner._update_from``): eager rollout, loss, backward, clip, step."""
        S_hat, logits = self._wm_rollout_eager(x["s"], x["a"], x["a_prev"], x["xi_n"])
        loss, (e_last, bce) = self._wm_err_eager(S_hat, logits, x["s1"], x["contact"], x["mask"])
        self.opt_wm.zero_grad(set_to_none=False)
        loss.backward()
        nn.utils.clip_grad_norm_(self.wm.parameters(), 1.0)
        self.opt_wm.step()
        return {"loss": loss.detach(), "e_last": e_last.detach(), "bce": bce.detach()}

    def term_update(self, w: dict, S_hat, xi_n: torch.Tensor) -> dict:
        """One Adam step of the termination head on the autoregressively PREDICTED states of the world-model windows (SuperMimic Eq. 1: BCE(d̂_j, d_j) on the rollout states; the paper's
        loss is unweighted). Labels: the REAL termination of the transition (``terminated``: no time-out, clip end or episode cap), the features those of ``critic_input`` at the predicted next state
        with the reference frame / goal of that step. ACP-unspecified: (i) the predicted states are detached (the head does not reshape the dynamics; its gradient reaches the policy through
        the imagination only), (ii) at most ``term_windows`` windows per step (the feature computation runs the forward kinematics and the scan), (iii) the mean over the weighted steps instead of the sum (Adam is scale-invariant)."""
        c, k = self.cfg, w["mask"].shape[1]
        sel = self.term_head_windows(w, c["term_windows"])
        Bf = len(sel)
        with torch.no_grad():
            flat_f = lambda x: x[sel].flatten(0, 1)  # noqa: E731
            xn = xi_n[sel].unsqueeze(1).expand(Bf, k, -1).flatten(0, 1)
            g1 = flat_f(w["goal1"])
            X = self.critic_input(tuple(flat_f(x) for x in S_hat), flat_f(w["f1"]), flat_f(w["phase1"]), flat_f(w["origin1"]), xn, g1[:, :3], g1[:, 3])
            self.term_head.norm.update(X)
            label = flat_f(w["terminated"]).float()
            weight = (w["mask"][sel] * w["weight"][sel, None]).flatten()
        logits = self.term_head(X)
        loss = (F.binary_cross_entropy_with_logits(logits, label, reduction="none") * weight).sum() / weight.sum().clamp(min=1e-6)
        self.opt_term.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.term_head.parameters(), 1.0)
        self.opt_term.step()
        valid = w["mask"][sel].flatten()
        return {"term_bce": loss.detach(), "term_pos": (label * valid).sum() / valid.sum().clamp(min=1), "term_pred": (torch.sigmoid(logits.detach()) * valid).sum() / valid.sum().clamp(min=1)}

    def apply_term_source(self, d: torch.Tensor, X: torch.Tensor | None) -> tuple[torch.Tensor, dict]:
        """Imagined termination probability ``d`` (analytic, (H*B,)) -> the one of ``term_source``: "learned" the head's sigmoid (the paper's ``d̂``; the analytic one is dropped), "union"
        ``1 - (1 - d)(1 - d̂)``; "analytic" returns ``d`` untouched. ``X``: ``critic_input`` of the states the transitions end in, flattened like ``d`` (None for "analytic"). Gradients flow through both."""
        src = self.cfg["term_source"]
        if src == "analytic":
            return d, {}
        p = torch.sigmoid(self.term_head(X))
        stats = {"term_analytic_mean": d.detach().mean(), "term_learned_mean": p.detach().mean()}
        return (p if src == "learned" else 1 - (1 - d) * (1 - p)), stats

    def sample_valid_starts(self, B: int):
        """Imagination starts, official ``WorldModelEnv.sample_initial_states``: uniform over ALL buffered states whose reference frame lies in its clip (``f <= clip end``; the
        reference then stays on the last frame of the clip, see :meth:`_imagine`). With ``start_sampling="per_motion"`` the clips are drawn with equal probability instead of the states."""
        c = self.cfg
        eligible = self.buffer.eligible_starts(lambda f: f <= self.motion.clip_end[self.motion.clip_id[f]])
        if len(eligible) == 0:
            raise RuntimeError("No buffered state is an imagination start.")
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

    def _imagine(self, s: dict, h: dict, graph_eta: tuple | None = None) -> dict:
        """H-step imagination of the policy from buffer start states ``s`` / histories ``h``. Gradients flow as in ``policy_update``
        (call under ``no_grad`` for diagnostics).

        ``graph_eta`` (the captured policy update, Stage 2): ``(eta_obs, eta_rew, mode)`` with ``eta_obs`` / ``eta_rew`` 0-dim device tensors (graph inputs: they change
        every iteration during the annealing) and the Python ``mode`` of :meth:`policy_graph_mode` fixed at the capture. Branch-free in the tensors: the planner always runs on the full batch."""
        c, mdp, kin = self.cfg, self.mdp, self.kin
        HZ = self.horizon
        S = unflatten_state(s["s"])
        f, phase, origin, a_prev, a_prev2, env_idx = s["f"], s["phase"], s["origin"], s["a_prev"], s["a_prev2"], s["env"]
        xi = self.xi_of(env_idx, s["xi_dyn"])
        xi_n = mdp.normalize_xi(xi)
        gpos, gyaw, gt = s["goal"][:, :3], s["goal"][:, 3], s["goal"][:, 4]
        if graph_eta is None:
            eta_obs, eta_rew = self.eta()
            use_planner = self.planner is not None and (eta_obs > 0 or eta_rew > 0)  # Stage 1 imagination reads the reference targets only
            plan_rew = eta_rew > 0
        else:
            eta_obs, eta_rew, mode = graph_eta
            use_planner, plan_rew = mode >= 1, mode >= 2
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
        wm_contact0 = s["wm_contact0"] if self.wm_contact_feedback else None
        eps_rep = c["imagine_ref_replay_eps"]
        Su, Reps = [], []  # the states continued from (the reference where the deviation exceeded ``eps_rep``) and where that happened
        clip_end = self.motion.clip_end[self.motion.clip_id[f0]]  # last frame of the clip of every start
        Ss, As, Aps, Aps2, Fs, Ls, Gs, Ptgt, Hard = [], [], [], [], [], [], [], [], []
        fk = fk0 = self.fk_all(S)  # forward kinematics of the current state, computed once: the new state's comes out of ``step_wm`` (or ``fk_all`` of the replaced state)
        Fk, Fkc = [], []  # ``fk`` of the predicted states (Ss) and of the states continued from (Su)
        for t in range(HZ):
            if t > 0:
                o_hist = self.push(o_hist, S, fk, a_prev, xi)
            Aps.append(a_prev), Aps2.append(a_prev2)
            bp, bR, bw, scan, tgt, valid, goal, gpos, gyaw, gt, pelvis_hist = self.observe(
                fk, f, phase, origin, gpos, gyaw, gt, pelvis_hist, odo_delay, odo_drift, t > 0
            )
            Gs.append(torch.cat((gpos, gyaw.unsqueeze(-1), gt.unsqueeze(-1)), -1))
            choice = None
            goal_noisy = mdp.noisy_goal(goal)  # one noise sample per imagined state: the observation and the planner see the same goal
            if use_planner:
                ptgt, pvalid, use = self._plan(o_hist, scan, goal_noisy, eta_obs, observation_only=not plan_rew and graph_eta is None)  # (the captured path: always the full batch)
                Ptgt.append((targets_to_world(*mdp.estimated_pose(pelvis_hist, odo_delay, odo_drift), ptgt), pvalid))
                choice = (use, ptgt)
            obs = self.assemble(o_hist, scan, tgt, valid, goal_noisy, choice)
            a = self.ac.actor(self.pol_norm(obs))
            if c["imagine_action_noise"] > 0:
                a = a + c["imagine_action_noise"] * torch.randn_like(a)
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
            f = advance_reference_frame(f, clip_end)  # once the clip end is reached the reference stays on its last frame (no mask, no terminal)
            Ss.append(S), As.append(a), Fs.append(f), Ls.append(logits), Hard.append(hard.detach()), Fk.append(fk)
            if eps_rep > 0:  # demonstration replay: continue from the reference where the imagined state has left it
                replaced = mdp.link_deviation(S, f, origin, fk) > eps_rep
                ref = mdp.reference_state(f, origin)
                S = tuple(torch.where(replaced.view(-1, *[1] * (x.dim() - 1)), r_.detach(), x) for x, r_ in zip(S, ref))
                fk = self.fk_all(S)  # the continued state differs from the predicted one
                Reps.append(replaced)
            Su.append(S), Fkc.append(fk)
        # goal at the last imagined state (its observation is not built); planner targets of that state for the planned-contact reward
        gpos, gyaw, gt = mdp.goal_update(f, phase, origin, gpos, gyaw, gt)
        Gs.append(torch.cat((gpos, gyaw.unsqueeze(-1), gt.unsqueeze(-1)), -1))
        if use_planner and plan_rew:
            o_hist = self.push(o_hist, S, fk, a_prev, xi)
            bp, bR, bw, scan, tgt, valid, goal, _, _, _, pelvis_hist = self.observe(fk, f, phase, origin, gpos, gyaw, gt, pelvis_hist, odo_delay, odo_drift, True)
            ptgt, pvalid, _ = self._plan(o_hist, scan, mdp.noisy_goal(goal), eta_obs)
            Ptgt.append((targets_to_world(*mdp.estimated_pose(pelvis_hist, odo_delay, odo_drift), ptgt), pvalid))
        return {"S0": S0, "f0": f0, "S": S, "f": f, "phase": phase, "origin": origin, "xi": xi, "xi_n": xi_n, "Ss": Ss, "As": As,
                "Aps": Aps, "Aps2": Aps2, "Fs": Fs, "Ls": Ls, "Gs": Gs, "Ptgt": Ptgt, "Hard": Hard, "lp_hist": lp_hist, "ep0": ep0, "eta_rew": eta_rew, "plan_rew": plan_rew,
                "Su": Su, "Reps": Reps, "fk0": fk0, "Fk": Fk, "Fkc": Fkc}  # fmt: skip

    def _wmres_cut(self, im: dict):
        """wmres: the truncation of the imagination BPTT (``research/glm_proposal.md`` §3 / §5, AHAC-style).

        Two sources, combined per start to the EARLIER step: (a) ``wmres_cut``: the first world-model contact event (a threshold crossing of the
        WM contact-head probabilities, ``im["Ls"]``; after ``wmres_cut_warmup_iters``), (b) ``wmres_cap_clips``: a per-clip fixed horizon cap (from
        iteration 0; the config is a measured constant, not a function of the WM). Returns ``None`` when both are off, else ``(d_cut, stats)`` where
        ``d_cut`` (H, B) is 1 at the truncation step of each fired start (``None`` if nothing fired or the safety valve tripped) and ``stats`` are
        log scalars."""
        c = self.cfg
        events = c["wmres_cut"] and self.it >= c["wmres_cut_warmup_iters"]
        caps = c["wmres_cap_clips"] != ""
        if not events and not caps:
            return None
        HZ, dev = self.horizon, self.device
        B = im["f0"].shape[0]
        t_eff = torch.full((B,), HZ, dtype=torch.long, device=dev)
        stats: dict[str, torch.Tensor] = {}
        if events:  # (a) the contact-event source
            t_ev, fired_ev = wmres.cut_step_from_logits(torch.stack(im["Ls"]), c["wmres_cut_theta"], c["wmres_cut_flux"])
            t_eff = torch.where(fired_ev, torch.minimum(t_eff, t_ev), t_eff)
            stats["cut_ev_frac_starts"] = fired_ev.float().mean()
        if caps:  # (b) the per-clip horizon cap (cached per-clip tensor; a start's cap is its clip's entry)
            if getattr(self, "_wmres_caps", None) is None:
                self._wmres_caps = wmres.parse_caps(c["wmres_cap_clips"], self.motion_names, HZ, dev)
                capped = [n for n, h in zip(self.motion_names, self._wmres_caps.tolist()) if h < HZ]
                print(f"[wmres] wmres_cap_clips={c['wmres_cap_clips']!r}: {len(capped)}/{len(self.motion_names)} "
                      f"clips capped below H={HZ}: {capped[:6]}{' ...' if len(capped) > 6 else ''}", flush=True)
            t_cap = self._wmres_caps[self.motion.clip_id[im["f0"]]]
            fired_cap = t_cap < HZ
            t_eff = torch.where(fired_cap, torch.minimum(t_eff, t_cap), t_eff)
            stats["cut_cap_frac_starts"] = fired_cap.float().mean()
        fired = t_eff < HZ
        stats["cut_frac_starts"] = fired.float().mean()
        if not fired.any():
            return None, stats
        d_cut = wmres.cut_termination(t_eff, fired, HZ, B)
        removed = (HZ - t_eff[fired]).sum() / (HZ * B)  # fraction of the batch's imagined steps masked out
        stats.update(cut_frac_steps=d_cut.mean(), cut_step_mean=t_eff[fired].float().mean(), cut_frac_horizon=removed)
        if removed > c["wmres_cut_max_frac"]:  # safety valve: this batch keeps the untruncated objective
            return None, stats | {"cut_skipped": d_cut.new_tensor(1.0)}
        return d_cut, stats

    def check_policy_graph_options(self) -> None:
        """``cuda_graph_policy``: refuse the options whose policy update is not capturable (host syncs, data-dependent shapes, per-iteration Python values, learned heads).
        Stage 2 (frozen planner in the imagination, ``eta_obs`` / ``eta_rew`` schedules) is capturable, see :meth:`policy_graph_mode`; its planner training stays eager, outside the graph."""
        c = self.cfg
        bad = []
        if type(self).policy_update is not ACPRunner.policy_update and not getattr(type(self), "policy_graph_capable", False):
            bad.append(f"{type(self).__name__} overrides policy_update without a capturable implementation (Stage 3 style)")
        if not c["cuda_graph_wm"]:
            bad.append("cuda_graph_wm must be True (the eager world-model update would run reduce-overhead cudagraph trees next to the policy graph)")
        if c["contact_source"] != "reference" and getattr(self, "planner", None) is None:
            bad.append("contact_source other than 'reference' needs the planner")
        if c["term_source"] != "analytic":
            bad.append("term_source must be 'analytic'")
        if c["wmres_cut"] or c["wmres_cap_clips"]:
            bad.append("wmres_cut / wmres_cap_clips must be off (data-dependent shapes)")
        if c["imagine_ref_replay_eps"] > 0 or c["imagine_link_hazard_eps"] > 0 or c["abpt_one_step_weight"] > 0:
            bad.append("imagine_ref_replay_eps / imagine_link_hazard_eps / abpt_one_step_weight must be 0")
        if c["checkpoint_imagination"] or c["wm_ref_contacts"] or c["wm_contact_feedback"]:
            bad.append("checkpoint_imagination / wm_ref_contacts / wm_contact_feedback must be False")
        if c["no_critic"]:
            bad.append("no_critic must be False")
        if bad:
            raise ValueError("cuda_graph_policy is not compatible with this configuration: " + "; ".join(bad))

    def policy_graph_state(self) -> dict:
        """The tensors the captured policy update reads through Python objects, address-checked at every replay (see :func:`graph_state`); a subclass with more modules extends it.
        (Stage 2: the frozen planner is trained eagerly, in place, outside the graph: its parameters are read, its gradients are not the graph's.)"""
        return graph_state([self.wm, self.ac, self.critic, self.pol_norm, *([self.planner] if self.planner is not None else [])], [self.opt_pi, self.opt_v], grad_modules=[self.ac.actor, self.critic])

    def _policy_graph_step(self, x: dict) -> dict:
        """The body of the captured policy update (``cuda_graph_policy``): the inputs are the imagination starts ``s`` / histories ``h`` as ``"s:key"`` / ``"h:key"``."""
        s = {k[2:]: v for k, v in x.items() if k.startswith("s:")}
        h = {k[2:]: v for k, v in x.items() if k.startswith("h:")}
        mode = self._policy_mode
        pi, v, ir, gn, surv, terms = self._policy_step(s, h, (x["e:obs"], x["e:rew"], mode, x["e:vmax"]) if mode > 0 else None)
        return {"pi": pi, "v": v, "ir": ir, "gn": gn, "surv": surv, **{"t:" + k: t for k, t in terms.items()}}

    def policy_graph_mode(self, eta_obs: float, eta_rew: float) -> int:
        """The structure of the captured imagination: 0 = the reference targets only (Stage 1, or Stage 2 while both etas are 0: no planner in the graph), 1 = the planner chooses
        the observation targets (``eta_obs`` > 0, ``eta_rew`` == 0: env reward weights, no planned-contact term), 2 = also the reward uses the planner's targets (``eta_rew`` > 0).
        These are the branches the eager update takes on the Python values of the etas; the graph is re-recorded when the mode changes (at most twice in a Stage 2 run: the
        schedules are monotone), the etas themselves (and the value bound of mode 2) are graph inputs."""
        if self.planner is None or (eta_obs <= 0 and eta_rew <= 0):
            return 0
        return 2 if eta_rew > 0 else 1

    def policy_update(self):
        """One policy step: H-step imagination from buffer states, BPTT of the discounted return, then GAE critic fit (one CUDA graph replay with ``cuda_graph_policy``)."""
        s, h = self.sample_valid_starts(self.cfg["pi_batch"])
        captured = getattr(self, "_captured_policy", None)
        if captured is None:
            return self._policy_step(s, h)
        eta_obs, eta_rew = self.eta()
        mode = self.policy_graph_mode(eta_obs, eta_rew)
        if mode != getattr(self, "_policy_mode", None):  # the structure of the update changes (planner enters the imagination / the reward): re-record
            if getattr(self, "_policy_mode", None) is not None:
                print(f"[policy graph] mode {self._policy_mode} -> {mode} at iteration {self.it}: re-recording", flush=True)
                captured.invalidate()
            self._policy_mode = mode
        self.update_value_bound(eta_rew)  # host float: baked into the capture in modes 0 / 1 (constant there: eta_rew == 0), the graph input "e:vmax" in mode 2
        dev = self.device
        etas = {"e:obs": torch.tensor(eta_obs, device=dev), "e:rew": torch.tensor(eta_rew, device=dev), "e:vmax": torch.tensor(self.v_max, device=dev)}
        out = captured({**{"s:" + k: v for k, v in s.items()}, **{"h:" + k: v for k, v in h.items()}, **etas})
        out = {k: v.clone() for k, v in out.items()}  # (the next replay overwrites the graph's outputs)
        return out["pi"], out["v"], out["ir"], out["gn"], out["surv"], {k[2:]: v for k, v in out.items() if k.startswith("t:")}

    def _policy_step(self, s: dict, h: dict, graph_eta: tuple | None = None):
        """The policy update from the sampled imagination starts (see :meth:`policy_update`); capture-safe when ``_captured_policy`` is set.
        ``graph_eta``: ``(eta_obs, eta_rew, mode, v_max)`` of a Stage 2 capture (see :meth:`_imagine`); ``v_max`` is the value bound as a device tensor."""
        c, mdp, dev = self.cfg, self.mdp, self.device
        B, HZ = c["pi_batch"], self.horizon
        graph = getattr(self, "_captured_policy", None) is not None  # the capture-safe variants of the host-synchronizing steps below
        v_max = self.v_max if graph_eta is None or graph_eta[2] < 2 else graph_eta[3]  # (mode 2: eta_rew changes every iteration of the annealing, so does the bound)
        frozen = (self.wm, self.critic, *([self.term_head] if self.term_head is not None else []))
        for m_ in frozen:
            for p_ in m_.parameters():
                p_.requires_grad_(False)
        im = self._imagine(s, h, None if graph_eta is None else graph_eta[:3])
        if not graph:
            self.update_value_bound(float(im["eta_rew"]))
            v_max = self.v_max
        S0, f0, S, f, phase, origin, xi, xi_n = im["S0"], im["f0"], im["S"], im["f"], im["phase"], im["origin"], im["xi"], im["xi_n"]
        Ss, As, Aps, Aps2, Fs, Ls, Gs = im["Ss"], im["As"], im["Aps"], im["Aps2"], im["Fs"], im["Ls"], im["Gs"]
        flat = lambda xs: torch.stack(xs).flatten(0, 1)  # noqa: E731  (H*B, ...)
        rep = lambda x: x.unsqueeze(0).expand(HZ, *x.shape).flatten(0, 1)  # noqa: E731
        S_all = tuple(flat([St[i] for St in Ss]) for i in range(6))
        replay = c["imagine_ref_replay_eps"] > 0
        Sc = im["Su"] if replay else Ss  # the states the rollout continued from (the reference where it was replaced): terminations, termination head, critic
        S_cont = tuple(flat([St[i] for St in Sc]) for i in range(6))
        flat_fk = lambda fks: tuple(torch.stack([F[i] for F in fks]).flatten(0, 1) for i in range(4))  # noqa: E731  (kin.forward_all of the imagined states, flattened like S_all)
        fk_all, fk_cont = flat_fk(im["Fk"]), flat_fk(im["Fkc"])  # the FK of S_all / S_cont, computed once by the imagination
        planned = None
        if im["Ptgt"] and im["plan_rew"]:  # the planned-contact reward of transition j uses the planner targets of state j + 1
            planned = (torch.stack([p[0] for p in im["Ptgt"][1:]]).flatten(0, 1), torch.stack([p[1] for p in im["Ptgt"][1:]]).flatten(0, 1))
        eta_rew = (im["eta_rew"] if torch.is_tensor(im["eta_rew"]) else torch.tensor(im["eta_rew"], device=dev)) if im["plan_rew"] else None
        qd_start = torch.cat((S0[5].unsqueeze(0), torch.stack([St[5] for St in Sc[:-1]])), 0).flatten(0, 1)  # joint velocities at the transition starts
        r, d, terms = self.reward_term(S_all, qd_start, flat(Fs), rep(phase), rep(origin), flat(As), flat(Aps), flat(Aps2), rep(xi), flat(Ls),
                                       im["lp_hist"], im["ep0"], planned, eta_rew, HZ, fk_all)  # fmt: skip
        if replay:
            d = mdp.soft_termination(S_cont, flat(Fs), rep(origin), rep(phase), fk=fk_cont)
        if c["imagine_link_hazard_eps"] > 0:  # leaving the reference by more than eps is a failure in the imagination too (as the injection of the real env)
            d = 1 - (1 - d) * (1 - torch.sigmoid((mdp.link_deviation(S_all, flat(Fs), rep(origin), fk_all) - c["imagine_link_hazard_eps"]) / c["imagine_link_hazard_tau"]))
        G1 = torch.stack(Gs[1:]).flatten(0, 1)
        # critic features of the states 1..H (S_cont, frames Fs, goals Gs[1:]): one differentiable evaluation serves the termination head, the critic
        # bootstraps and (detached) the critic fit. Only when the head reads them: otherwise the bootstraps evaluate their few rows, the fit all states, directly.
        X1 = self.critic_input(S_cont, flat(Fs), rep(phase), rep(origin), rep(xi_n), G1[:, :3], G1[:, 3], fk_cont) if c["term_source"] != "analytic" else None

        def feats_at(rows):  # critic_input of rows of the flattened states 1..H (rows: slice or index tensor)
            if X1 is not None:
                return X1[rows]
            return self.critic_input(tuple(x[rows] for x in S_cont), flat(Fs)[rows], rep(phase)[rows], rep(origin)[rows], rep(xi_n)[rows], G1[rows, :3], G1[rows, 3],
                                     tuple(x[rows] for x in fk_cont))  # fmt: skip

        d, term_stats = self.apply_term_source(d, X1)
        r, d = r.view(HZ, B), d.view(HZ, B)
        if graph:  # (the backward of ``torch.cumprod`` synchronizes with the host: .any().item() on the zeros of the input)
            alive = torch.stack(list(itertools.accumulate((1 - d).unbind(0), torch.mul)))
        else:
            alive = torch.cumprod(1 - d, 0)  # c_{t+1}
        c_t = torch.cat((torch.ones(1, B, device=dev), alive[:-1]), 0)
        disc = c["gamma"] ** torch.arange(HZ, device=dev, dtype=torch.float32).unsqueeze(-1)
        r_obj, alive_obj, c_obj = r, alive, c_t  # wmres: the objective terms; identical to r / alive / c_t unless the cut fires
        cut = self._wmres_cut(im)  # wmres: AHAC-style contact-event truncation (research/glm_proposal.md); None unless enabled
        d_cut, cut_stats = (None, None) if cut is None else cut
        if d_cut is not None:
            alive_obj = torch.cumprod((1 - d) * (1 - d_cut), 0)
            c_obj = torch.cat((torch.ones(1, B, device=dev), alive_obj[:-1]), 0)
            r_obj = r * (1 - c["wmres_cut_alpha"] * d_cut)
        J = (disc * c_obj * r_obj).sum(0)
        cT = alive_obj[-1]
        r_sum = (c_t * r).detach().sum(0)
        rews, dones = r.detach(), d.detach()
        if not c["no_critic"]:
            J = J + (c["gamma"] ** HZ) * cT * self.value_bound(self.critic(feats_at(slice(-B, None))), v_max)  # the last state S_H
            if d_cut is not None:  # wmres: the truncated returns continue with the critic value of the last pre-event state (S_{t*}, t* >= 1)
                rows = d_cut.flatten().nonzero(as_tuple=False).squeeze(-1)  # row = t* * B + b in the flattened (H, B) objective arrays
                pre = rows - B  # the pre-event state S_{t*} and its frame / goal sit one step earlier in the flattened transition arrays
                V_cut = self.value_bound(self.critic(feats_at(pre)), v_max)
                J = J + torch.zeros(B, device=dev).index_add(0, rows % B, (disc[rows // B].squeeze(-1) * c_obj.flatten()[rows]) * V_cut)
        if c["abpt_one_step_weight"] > 0 and not c["no_critic"]:  # ABPT-style value gradient: the one-step return with the critic bootstrap
            J1 = r[0] + c["gamma"] * (1 - d[0]) * self.value_bound(self.critic(feats_at(slice(0, B))), v_max)
            J = (1 - c["abpt_one_step_weight"]) * J + c["abpt_one_step_weight"] * J1
        loss = -J.mean()
        self.opt_pi.zero_grad(set_to_none=not graph)
        loss.backward()
        gn = nn.utils.clip_grad_norm_(self.ac.actor.parameters(), 1.0)
        nn.utils.clip_grad_value_(self.ac.actor.parameters(), 0.1)
        if not c["no_critic"]:
            with torch.no_grad():  # states 0..H: the features of 1..H are the (detached) ones above, only the start state is new
                X0 = self.critic_input(S0, f0, phase, origin, xi_n, Gs[0][:, :3], Gs[0][:, 3], im["fk0"])
                X = torch.cat((X0.unsqueeze(0), (feats_at(slice(None)) if X1 is None else X1.detach()).view(HZ, B, -1)), 0)
                V = self.critic(X)  # raw values inside the lambda return (official), the targets are clamped at v_max
                targets = lambda_return_targets(rews, dones, V, c["gamma"], c["lam"]).clamp(max=v_max)  # (H, B)
        if graph and torch.cuda.is_current_stream_capturing():
            step_if_finite(self.opt_pi, torch.isfinite(gn))
        elif torch.isfinite(gn):
            self.opt_pi.step()
        for m_ in frozen:
            for p_ in m_.parameters():
                p_.requires_grad_(True)
        v_loss = torch.zeros((), device=dev)
        if not c["no_critic"]:
            Xf, Yf = X[:-1].reshape(-1, X.shape[-1]), targets.reshape(-1)
            n = Xf.shape[0]
            mb = n // c["critic_minibatches"]
            perm = torch.randperm(n, device=dev)  # one permutation for all epochs (official)
            for _ in range(c["critic_epochs"]):
                for i in range(c["critic_minibatches"]):
                    idx = perm[i * mb : (i + 1) * mb]
                    v_loss = ((self.critic(Xf[idx]) - Yf[idx]) ** 2).mean()
                    self.opt_v.zero_grad(set_to_none=not graph)
                    v_loss.backward()
                    nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
                    self.opt_v.step()
        term_means = {k: v.detach().view(HZ, B).mean() for k, v in terms.items()}
        term_means.update(term_stats)
        if cut_stats is not None:  # wmres: cut_frac_starts / cut_frac_steps / cut_step_mean / cut_skipped of this update
            term_means.update(cut_stats)
        if replay:
            term_means["replaced_frac"] = torch.stack(im["Reps"]).float().mean()  # share of the imagined steps that were continued from the reference
        with torch.no_grad():
            term_means["jl_soft"] = mdp.joint_violation(S_all[4])[0]  # fraction of controlled joints beyond the soft limits
            term_means["jl_hard"] = torch.stack(im["Hard"]).mean()  # ... beyond the hard range
        return loss.detach(), v_loss.detach(), (r_sum / HZ / mdp.dt).mean(), gn.detach(), cT.mean().detach(), term_means

    # ------------------------------------------------------------------------------------------------
    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        """Train until iteration ``num_learning_iterations`` (total, so a resumed run continues)."""
        if not self.training_ready:
            self.setup_training()
        if self.log_dir is not None:
            store_code_state(self.log_dir, self.git_status_repos)
        c = self.cfg
        stats = {"ep_len": [], "term": [], "done": [], "jl_soft": [], "rec_episodes": [], "rec_samples": []}
        self.collect(c["warmup_steps"], stats)  # also refills the (unsaved) replay buffer after a resume
        t_start = time.time() - self.wall_offset
        t_log = time.time()
        step_acc = 0.0
        phase_times = []
        wm_keys = ("wm", "wm8", "bce", *(("term_bce", "term_pos", "term_pred") if self.term_head is not None else ()))
        agg: dict[str, list] = {k: [] for k in ("wm", "wm8", "bce", "pi", "v", "ir", "gn", "surv", "term_bce", "term_pos", "term_pred")}
        terms_acc: dict[str, list] = {}
        planner_acc: dict[str, list] = {}
        while self.it < num_learning_iterations:
            it = self.it
            step_acc += 1.0 / self.utd
            n = int(step_acc)
            step_acc -= n
            if c.get("profile_phases", False):
                torch.cuda.synchronize()
                t_phase = time.perf_counter()
            self.collect(n, stats)
            if c.get("profile_phases", False):
                torch.cuda.synchronize()
                t_collect = time.perf_counter()
            m = None
            for _ in range(c["wm_batches"]):
                m_ = self.wm_update()
                if m_ is None:  # no complete window (the buffer does not change within the iteration: none for the other batches either)
                    m = None
                    break
                m = m_
            if c.get("profile_phases", False):
                torch.cuda.synchronize()
                t_wm = time.perf_counter()
            if m is not None:
                for k_ in wm_keys:
                    agg[k_].append(m[k_])
            if self.planner_trainer is not None:
                planner_losses = self.planner_update()
                if planner_losses is not None:
                    for k_, v_ in planner_losses.items():
                        planner_acc.setdefault(k_, []).append(v_)
            if m is None:  # no complete window: this iteration trained neither the world model nor the policy, so it does not count (lr, eta, the iteration budget and the save / log
                # intervals follow the effective updates); the collection above went on, so the windows appear sooner or later
                self._skipped_streak += 1
                if self._skipped_streak >= self.MAX_SKIPPED_STREAK:
                    raise RuntimeError(f"{self._skipped_streak} consecutive iterations (it {self.it}, {self.num_skipped_iterations} skipped in total) without a complete {c['wm_window']}-transition "
                                       f"window in the replay buffer: no world-model / policy update is possible. Episode lengths in the replay buffer: {self.buffer.episode_length_stats()}")
                continue
            self._skipped_streak = 0
            if it >= c["wm_only_iters"]:  # (no policy update without a world-model update)
                for _ in range(c["policy_updates_per_iter"]):
                    pl, vl, ir, gn, surv, terms = self.policy_update()
                    agg["pi"].append(pl), agg["v"].append(vl), agg["ir"].append(ir), agg["gn"].append(gn), agg["surv"].append(surv)
                    for k, v in terms.items():
                        terms_acc.setdefault(k, []).append(v)
            if c.get("profile_phases", False):
                torch.cuda.synchronize()
                t_policy = time.perf_counter()
                phase_times.append(((t_collect-t_phase)*1000, (t_wm-t_collect)*1000, (t_policy-t_wm)*1000))
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
                    "policy/loss": mean(agg_h["pi"]), "policy/grad_norm": mean(agg_h["gn"]), "critic/value_loss": mean(agg_h["v"]),
                    "imagination/reward": mean(agg_h["ir"]), "imagination/survival": mean(agg_h["surv"]),
                    "sim/episode_length": mean(stats_h["ep_len"]), "sim/terminated_fraction": term_frac,
                    "perf/it_per_s": ips, "perf/lr": c["lr"], "perf/sim_steps": float(self.sim_steps), "perf/wall_h": wall_h,
                    "perf/max_mem_gb": torch.cuda.max_memory_allocated() / 2**30,
                    "stage2/eta_obs": self.eta()[0], "stage2/eta_rew": self.eta()[1],
                    "perf/skipped_iterations": float(self.num_skipped_iterations),
                }  # fmt: skip
                if "inject" in stats:
                    row["sim/inject_rate"] = mean(stats_h["inject"])  # share of the envs moved to the reference per control step
                pen_msg = ""
                probes = [q for q in getattr(self.unwrapped, "_reset_rejection_probes", {}).values() if q is not None]
                if probes:  # reset calls whose batched probe overflowed its contact buffer and were redone sequentially (since the last log)
                    n_calls, n_over = sum(q.num_calls for q in probes), sum(q.num_overflow_calls for q in probes)
                    d_calls, d_over = n_calls - getattr(self, "_probe_calls_logged", 0), n_over - getattr(self, "_probe_over_logged", 0)
                    self._probe_calls_logged, self._probe_over_logged = n_calls, n_over
                    row["sim/reset_penetration/probe_overflow_fallback_rate"] = d_over / max(d_calls, 1)
                pen = getattr(self.unwrapped, "reset_penetration_stats", None)  # reset penetration rejection (task option): counts since the last log
                if pen is not None:
                    n_reset, n_resampled, n_fallback, n_ref_pen = (pen - getattr(self, "_pen_logged", torch.zeros_like(pen))).tolist()
                    self._pen_logged = pen.clone()
                    row["sim/reset_penetration/resample_rate"], row["sim/reset_penetration/fallback_rate"] = n_resampled / max(n_reset, 1), n_fallback / max(n_reset, 1)
                    row["sim/reset_penetration/reference_penetrating_rate"] = n_ref_pen / max(n_reset, 1)
                    pen_msg = f"pen {n_resampled / max(n_reset, 1):.3f}/{n_fallback / max(n_reset, 1):.4f} "  # share of resets that were resampled / fell back to the reference
                if self.reset_bins is not None:  # ReturnSampler bins: concentration of the bin distribution, mass and mean value V_b per clip family
                    bins = self.reset_bins
                    p_b = bins.bin_p
                    row["bins/p_entropy"], row["bins/p_entropy_uniform"] = float(-(p_b * p_b.clamp(min=1e-12).log()).sum()), math.log(len(p_b))
                    row["bins/p_max"], row["bins/G_mean"] = float(p_b.max()), float(bins.bin_G.mean())
                    for i, name in enumerate(self.family_names):
                        fam = self.clip_family[bins.bin_clip] == i
                        row[f"bins/G_mean/{name}"], row[f"bins/p_mass/{name}"] = float(bins.bin_G[fam].mean()), float(p_b[fam].sum())
                if self.term_head is not None:  # learned termination head: BCE / base rate / mean prediction on the world-model windows (true labels, predicted states)
                    row["term/bce_train"], row["term/positive_share_in_batch"], row["term/predicted_mean_in_batch"] = mean(agg_h["term_bce"]), mean(agg_h["term_pos"]), mean(agg_h["term_pred"])
                for k, v in terms_h.items():
                    row[f"imagination/{'joint_limit/' + k if k.startswith('jl_') else 'termination/' + k if k.startswith('term_') else 'reward_term/' + k}"] = mean(v)
                row["sim/joint_limit/jl_soft"] = mean(stats_h["jl_soft"])
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
                      f"imag_r {row['imagination/reward']:.3f} surv {row['imagination/survival']:.3f} | "
                      f"ep_len {row['sim/episode_length']:.1f} term {term_frac:.2f} "
                      f"eta {row['stage2/eta_obs']:.2f}/{row['stage2/eta_rew']:.2f} {pen_msg}{planner_msg}| {ips:.2f} it/s mem {row['perf/max_mem_gb']:.1f}G"
                      + (f" | skipped {self.num_skipped_iterations}" if self.num_skipped_iterations else ""), flush=True)  # fmt: skip
                if self.term_head is not None:
                    print(f"  term_head it {self.it} bce {row['term/bce_train']:.6f} positive {row['term/positive_share_in_batch']:.6f} "
                          f"term_learned_mean {row.get('imagination/termination/term_learned_mean', float('nan')):.6f} "
                          f"term_analytic_mean {row.get('imagination/termination/term_analytic_mean', float('nan')):.6f}", flush=True)
                if phase_times:
                    values = [sum(v[j] for v in phase_times)/len(phase_times) for j in range(3)]
                    print(f"  phases it {self.it} collect_ms {values[0]:.3f} wm_term_ms {values[1]:.3f} policy_critic_ms {values[2]:.3f}" + (f" policy_graph_captures {self._captured_policy.num_captures} mode {self._policy_mode}" if self._captured_policy is not None else ""), flush=True)
                    cp = getattr(self, "_collect_prof", None)
                    if cp is not None:  # per-iteration means over the log window: before env.step (obs, policy), env.step, after it (reads, bins, buffer), WM statistics
                        n_it = len(phase_times)
                        print(f"  collect it {self.it} pre_ms {1000 * cp[0] / n_it:.1f} env_step_ms {1000 * cp[1] / n_it:.1f} post_ms {1000 * cp[2] / n_it:.1f} wm_stats_ms {1000 * cp[3] / n_it:.1f}", flush=True)
                        self._collect_prof = [0.0] * 4
                    phase_times.clear()
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
        """``path``: ``OnPolicyRunner`` checkpoint format (directly playable). Sidecar: world model, critic, their optimizers
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
            "motion_names": self.motion_names,
            **({"return_count": self.reset_bins.count, "return_effort": self.reset_bins.effort} if self.cfg["return_resume_stats"] and self.reset_bins is not None else {}),
            **({"bin_G": self.reset_bins.bin_G, "bin_w_avg": self.reset_bins.bin_w_avg, "bin_clip": self.reset_bins.bin_clip, "bin_local": self.reset_bins.bin_local, "bins_v2": self.reset_bins.v2} if self.reset_bins is not None else {}),
            **({"planner_trainer": self.planner_trainer.state_dict()} if self.planner_trainer is not None else {}),
            **({"term_head": self.term_head.state_dict(), "opt_term": self.opt_term.state_dict()} if self.term_head is not None else {}),
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
        opt_state = loaded_dict["optimizer_state_dict"]
        if opt_state.get("muon_adam", False):
            self.opt_pi.load_state_dict(opt_state)
        else:  # play / export only need the actor: the plain-Adam optimizer state of a checkpoint from before the Muon + Adam optimizer is not loaded (fresh actor optimizer)
            print("[load] actor optimizer state not loaded: the checkpoint has a plain Adam state (this runner: Muon + fused Adam)", flush=True)
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
        if self.wm_contact_spec:
            self.wm.load_compatible(ck["wm"])
            self.wm.load_optimizer(self.opt_wm, ck["opt_wm"])
        else:
            self.wm.load_state_dict(ck["wm"])
            self.opt_wm.load_state_dict(ck["opt_wm"])
        self.critic.load_state_dict(ck["critic"])
        self.opt_v.load_state_dict(ck["opt_v"])
        self.wall_offset, self.sim_steps = ck["wall_time"], ck["sim_steps"]
        assert self.checkpoint_it == ck["iteration"]
        if not (torch.allclose(ck["xi_mean"].to(self.device), self.mdp.xi_mean) and torch.allclose(ck["xi_scale"].to(self.device), self.mdp.xi_scale)):  # the encoding is fixed by the env's DR ranges
            raise ValueError("the sidecar's xi encoding differs from the DR ranges of this env (another DR config, or a checkpoint from before the DR-range encoding)")
        bins = self.reset_bins
        if bins is not None and "motion_names" in ck:
            # the same dataset (same clip list in the same order) is matched by position, so stems repeated in different folders
            # (e.g. high-step/ and high-climb-76/) resume too; another dataset is matched by name, which needs unique stems
            same_clips = list(ck["motion_names"]) == list(self.motion_names)
            names = list(range(len(self.motion_names))) if same_clips else self.motion_names
            saved_names = names if same_clips else ck["motion_names"]
            assert same_clips or len(set(names)) == len(names), "clip file names (stems) must be unique: the bins are matched by name"
            index = {n: i for i, n in enumerate(saved_names)}
            if self.cfg["clip_aliases"] and not same_clips:  # a renamed clip keeps its bins
                for old, new in yaml.safe_load(open(os.path.expanduser(self.cfg["clip_aliases"]))).items():
                    if old in index and new not in index:
                        index[new] = index[old]
            saved_of = [index.get(n) for n in names]  # clip j of the dataset -> clip of the sidecar (None: a new clip, its bins start at G_b = 0)
            restored_bins = 0
            if "bin_G" in ck and ck.get("bins_v2", False) == bins.v2:  # (bins of another layout are not restored) the bins are matched by clip and bin index
                current = {(saved_of[c_], b_): i for i, (c_, b_) in enumerate(zip(bins.bin_clip.tolist(), bins.bin_local.tolist())) if saved_of[c_] is not None}
                for j, (c_, b_) in enumerate(zip(ck["bin_clip"].tolist(), ck["bin_local"].tolist())):
                    i = current.get((c_, b_))
                    if i is not None:
                        bins.bin_G[i], bins.bin_w_avg[i] = ck["bin_G"][j], ck["bin_w_avg"][j]
                        if self.cfg["return_resume_stats"]:
                            if "return_count" in ck and "return_effort" in ck:
                                bins.count[i] = ck["return_count"][j]
                                bins.effort[i] = ck["return_effort"][j]
                        restored_bins += 1
            if self.cfg["return_resume_stats"]:
                if "return_count" not in ck or "return_effort" not in ck:
                    print("[resume] legacy ReturnBins checkpoint: count / effort unavailable; values restored only", flush=True)
                focus = torch.softmax(-bins.steepness * bins.value / bins.max_return, 0)
                total = bins.effort.sum()
                keep = (bins.effort / total.clamp(min=1e-300)).float() if bool(total > 0) else torch.full_like(focus, 1 / len(focus))
                # Do not call _distribution(): recomputing probabilities is not another effort observation.
                bins.bin_p = bins.explore / len(focus) + (1 - bins.explore - bins.keep) * focus + bins.keep * keep
                bins.update_probabilities()
            print(f"[resume] bins: {restored_bins} of {len(bins.bin_clip)} restored; new clips: {[self.motion_names[j] for j, k in enumerate(saved_of) if k is None]}", flush=True)
        elif bins is not None:
            print("[resume] the sidecar has no motion names: the bins start at G_b = 0 (migrate_sidecar_names.py)", flush=True)
        if self.term_head is not None and "term_head" in ck:  # a sidecar of a run without the learned termination head: the head starts from its initialization
            self.term_head.load_state_dict(ck["term_head"]), self.opt_term.load_state_dict(ck["opt_term"])
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
                out = self.planner.forward_eager(proprio, goal, points)  # (never the torch.compile'd callable that setup_training installs for the training)
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
