"""Numerical parity of the optimized ``policy_update`` (compiled imagination step / reward / critic features, gradient
checkpointing, bf16 policy autocast) against the eager reference on FIXED inputs: same sampled starts and histories, same RNG
seed, all random noise sources of the imagination switched off (goal / proprio / camera noise; the ensemble member draw is
seeded), no optimizer step. Compares the policy loss J and the gradient of every actor parameter.

    .venv/bin/python tests/supermimic/check_policy_update_parity.py
Small GPU job (32 envs); fp32 without TF32 for the exact comparisons (TF32 / bf16 variants are reported separately)."""

from __future__ import annotations

import copy
import os

import torch

from instinct_mj.rl.supermimic import networks
from instinct_mj.rl.supermimic.runner import SuperMimicRunner
from instinct_mj.scripts.supermimic.common import TASK_ID, build_env_cfg, make_vec_env
from instinct_mj.tasks.registry import load_instinct_rl_cfg

os.environ.setdefault("MUJOCO_GL", "egl")
cfg = load_instinct_rl_cfg(TASK_ID)
cfg.device = "cuda:0"
for k, v in dict(buffer_steps=120, warmup_steps=64, pi_batch=128, wm_batch=256, fail_windows=64, wm_only_iters=0).items():
    setattr(cfg, k, v)
env_cfg = build_env_cfg(TASK_ID, 32, "~/Datasets/20251116_50cm_kneeClimbStep1")
env_cfg.seed = 42
vec = make_vec_env(TASK_ID, env_cfg, "cuda:0")
runner = SuperMimicRunner(vec, cfg.to_dict(), log_dir=None, device="cuda:0")
runner.setup_training()
runner.collect(64, {"r": [], "r_gap": [], "ep_len": [], "term": [], "done": []})
runner.it = 10**6  # failure head active in imagination
# make the fail head non-trivial (p_fail ~ 0.3%, input dependent) so that its gradient path is compared as well
with torch.no_grad():
    for p in runner.fail_head.net.parameters():
        p.add_(0.003 * torch.randn_like(p))
    runner.fail_head.net[-1].bias.fill_(-6.0)

# ---- determinism: fixed starts, no stochastic noise, no optimizer steps ----
runner.mdp.goal_noise.zero_(), runner.mdp.proprio_noise.zero_()
runner.depth_pipe.noised = runner.depth_pipe.clean  # no stochastic camera noise in the rendered images
starts = runner.sample_valid_starts(cfg.pi_batch)
runner.sample_valid_starts = lambda B: starts
runner.opt_pi.step = lambda *a, **k: None
runner.opt_v.step = lambda *a, **k: None
critic_state = copy.deepcopy(runner.critic.state_dict())


def eager(fn):
    return getattr(fn, "_torchdynamo_orig_callable", fn)


def run(label, *, compiled, checkpoint=False, autocast=False, amp=None, tf32=False):
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    networks.AMP_DTYPE = amp
    runner.cfg.update(compile_policy=compiled, checkpoint_imagination=checkpoint, policy_autocast=autocast)
    runner._build_fns()
    if not compiled:
        runner.imag_core, runner.reward_term, runner.critic_input = eager(runner.imag_core), eager(runner.reward_term), eager(runner.critic_input)
    runner.critic.load_state_dict(critic_state)
    torch.manual_seed(1234)
    loss, v_loss, ir, gn, surv, terms = runner.policy_update()
    grads = torch.cat([p.grad.flatten() for p in runner.ac.parameters() if p.grad is not None])
    return label, float(loss), grads.clone(), float(surv), float(terms["p_fail_mean"])


ref = run("eager fp32 (reference)", compiled=False)
print(f"reference: loss {ref[1]:.6f}, |grad| max {float(ref[2].abs().max()):.4f}, survival {ref[3]:.4f}, mean p_fail {ref[4]:.4f}")
failed = []
for variant, tol in (
    (dict(label="torch.compile (imag step, reward, critic features)", compiled=True), 1e-4),
    (dict(label="gradient checkpointing (eager step)", compiled=False, checkpoint=True), 1e-5),
    (dict(label="compile + gradient checkpointing", compiled=True, checkpoint=True), 1e-4),
):
    out = run(**variant)
    rel = float((out[2] - ref[2]).abs().max() / ref[2].abs().max())
    print(f"{out[0]:52s}: loss diff {abs(out[1] - ref[1]):.2e}, grad max rel diff {rel:.2e} (tol {tol:.0e})")
    if rel > tol or abs(out[1] - ref[1]) > tol * max(1.0, abs(ref[1])):
        failed.append(out[0])
print("--- reduced-precision variants (informational; not exact by construction) ---")
amp_eager = None
for variant in (
    dict(label="TF32 matmuls (training default), eager", compiled=False, tf32=True),
    dict(label="bf16 autocast of the policy net, eager", compiled=False, autocast=True),
    dict(label="bf16 autocast of WM/critic/heads (training default) eager", compiled=False, amp=torch.bfloat16),
    dict(label="  same, compiled", compiled=True, amp=torch.bfloat16),
):
    out = run(**variant)
    ref_amp = amp_eager if variant.get("amp") is not None and variant["compiled"] else ref
    if variant.get("amp") is not None and not variant["compiled"]:
        amp_eager = out
    print(f"{out[0]:58s}: loss diff vs {'fp32 eager' if ref_amp is ref else 'amp eager'} {abs(out[1] - ref_amp[1]):.2e}, grad max rel diff {float((out[2] - ref_amp[2]).abs().max() / ref_amp[2].abs().max()):.2e}")
vec.close()
assert not failed, f"parity failed: {failed}"
print("PARITY OK")
