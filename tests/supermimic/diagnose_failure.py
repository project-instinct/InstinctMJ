"""Data-driven diagnosis of a failing motion (default: kneelClimbStep1) for a stock-format SuperMimic checkpoint.

Pins a few envs of the Play-like eval env to the motion (+ some other motions as baselines), rolls the policy out and logs
per step: reference time / frame, fired terminations (and per-link errors of ``link_pos_too_far``), robot vs reference
heights of root / knees / ankles / wrists, contacts, and the world-model (1-step, 8-step) state and visual-latent errors on the
REAL transitions. It also checks the reference itself against the terrain geometry (robot collision geoms placed in the reference
pose, contacts against the terrain), and re-runs the pinned envs with the tracking terminations disabled.

    .venv/bin/python tests/supermimic/diagnose_failure.py <run>/model_4000.pt --motion-folder ~/Datasets/20251116_50cm_kneeClimbStep1
Small GPU job (16 envs).

Written for checkpoints of the visual-latent design (commit 4c4a836: ``encode_depth``, ``z_dim``, ``latent_dec`` in the sidecar);
not ported yet to the rendered-depth runner, whose world model has no latent.
"""

from __future__ import annotations

import argparse
import copy
import os

import mujoco
import numpy as np
import torch

from instinct_mj.rl.supermimic.depth import DepthPipeline
from instinct_mj.rl.supermimic.mdp import DEPTH_DIM, flatten_state, unflatten_state
from instinct_mj.rl.supermimic.networks import FailureHead, LatentDecoder, WorldModel
from instinct_mj.rl.supermimic.rotations import rot_angle_sq
from instinct_mj.rl.supermimic.sim import SimReader, build_mdp
from instinct_mj.scripts.supermimic.evaluate import EvalArgs, Evaluator

parser = argparse.ArgumentParser()
parser.add_argument("checkpoint")
parser.add_argument("--motion-folder", required=True)
parser.add_argument("--max-steps", type=int, default=700)
parser.add_argument("--out", default="/tmp/sm_work/diag")
args = parser.parse_args()
os.makedirs(args.out, exist_ok=True)
torch.manual_seed(0)
dev = "cuda:0"
np.set_printoptions(precision=3, suppress=True, linewidth=200)

# ---- env / policy (motions: 4 x kneel(wall), 4 x kneel(noWall), jumpsit2 x2, rollVault x2, diveroll x4) ----
ev = Evaluator(EvalArgs(motion_folder=args.motion_folder, checkpoint=args.checkpoint, num_envs=16, min_episodes_per_motion=1))
names = ev.motion_files
KNEEL = [i for i, n in enumerate(names) if "kneelClimbStep1" in n]
layout = [KNEEL[0]] * 4 + [KNEEL[1]] * 4 + [3, 7, 4, 8] + [0, 1, 5, 9]
ev.motion_of_env = torch.tensor(layout, device=dev)
env, vec, mgr, robot = ev.env, ev.vec, ev.mgr, ev.robot
ev.runner.load(args.checkpoint)
policy = ev.runner.get_inference_policy(device=dev)
robot.data.default_joint_pos_nominal = robot.data.default_joint_pos[0].clone()  # Play removes the default-pos DR event that sets it
sidecar = torch.load(os.path.join(os.path.dirname(args.checkpoint), "supermimic_state_" + os.path.basename(args.checkpoint)[len("model_"):]), map_location=dev, weights_only=False)
motion, kin, scanner, mdp = build_mdp(env, sidecar["cfg"]["heightmap_resolution"], sidecar["cfg"]["heightmap_cache_dir"], dev)
mdp.xi_mean, mdp.xi_scale = sidecar["xi_mean"].to(dev), sidecar["xi_scale"].to(dev)
reader = SimReader(env)
xi = reader.context(mdp)
xi_n = mdp.normalize_xi(xi)
depth_pipe = DepthPipeline(env)
encode = ev.runner.encode_depth  # policy depth encoder E (normalized as in the policy observation)
wm = WorldModel(scanner.num_rays, len(reader.contact_body_ids), mdp.xi_dim, ev.runner.z_dim).to(dev)
wm.load_state_dict(sidecar["wm"])
dec = LatentDecoder(ev.runner.z_dim).to(dev)
dec.load_state_dict(sidecar["latent_dec"])
wm.eval(), dec.eval()
fail_head = None
if "fail_head" in sidecar:  # checkpoints written after the failure head was added
    fail_head = FailureHead(mdp.critic_dim()).to(dev)
    fail_head.load_state_dict(sidecar["fail_head"])
    fail_head.eval()
N = 16
dt = env.step_dt
tm = env.termination_manager
TERMS = ["base_pos_too_far", "base_pg_too_far", "link_pos_too_far", "illegal_reset_contact", "dataset_exhausted", "out_of_border", "time_out"]
body_names = list(robot.body_names)
ref_links = list(mgr.cfg.link_of_interests)
robot_link_ids = robot.find_bodies(ref_links, preserve_order=True)[0]
term_links = [ref_links[i] for i in mdp.term_link_ids]
KEY = {"root(torso)": "torso_link", "pelvis": "pelvis", "L knee": "left_knee_link", "R knee": "right_knee_link", "L ankle": "left_ankle_roll_link",
       "R ankle": "right_ankle_roll_link", "L wrist": "left_wrist_yaw_link", "R wrist": "right_wrist_yaw_link"}
fmt = vec.get_obs_format()["policy"]
off = 0
for k, shp in fmt.items():
    if k == "depth_image":
        break
    off += int(np.prod(shp))
dsl = slice(off, off + 576)


@torch.no_grad()
def rollout(max_steps: int, record: bool):
    ev.assign_motions()
    obs, _ = vec.get_observations()
    active = torch.ones(N, dtype=torch.bool, device=dev)
    log = {k: [] for k in ["S", "S1", "a", "valid", "t_ref", "frame", "z_robot", "z_ref", "link_err", "base_err", "joint_err", "flags", "contact", "depth_anchor", "depth_clean1", "origin", "ref_pos"]}
    for _ in range(max_steps):
        S = flatten_state(reader.state())
        f, phase, origin = motion.current_frame()
        anchor = obs[:, dsl].view(N, 1, 18, 32).clone()
        t_ref = mgr._timestamp.clone()
        a = policy(obs)
        obs, rew, dones, _ = vec.step(a)
        done = dones.bool()
        S1 = flatten_state(reader.state())
        ref = mgr.reference_frame
        zr = robot.data.body_link_pos_w[:, robot_link_ids][..., 2]
        zf = ref.link_pos_w[:, 0][..., 2]
        flags = torch.stack([tm.get_term(n) for n in TERMS], -1)
        force = torch.norm(ev.env.scene["contact_forces"].data.force_history, dim=-1).max(-1).values  # (N, bodies)
        log["S"].append(S.cpu()), log["S1"].append(S1.cpu()), log["a"].append(a.cpu()), log["valid"].append((active & ~done).cpu())
        log["t_ref"].append(t_ref.cpu()), log["frame"].append((f - motion.clip_start[motion.clip_id[f]]).cpu())
        log["z_robot"].append(zr.cpu()), log["z_ref"].append(zf.cpu())
        log["link_err"].append((zr[:, mdp.term_link_ids] - zf[:, mdp.term_link_ids]).cpu())
        log["base_err"].append((robot.data.root_link_pos_w - ref.base_pos_w[:, 0]).cpu())
        log["joint_err"].append((robot.data.joint_pos - ref.joint_pos[:, 0]).abs().mean(-1).cpu())
        log["flags"].append((flags & active[:, None]).cpu()), log["contact"].append(force.cpu())
        log["depth_anchor"].append(anchor.cpu()), log["depth_clean1"].append(depth_pipe.clean_from_sensor().cpu())
        log["origin"].append(origin.cpu()), log["ref_pos"].append(ref.base_pos_w[:, 0].cpu())
        active &= ~done
        if not bool(active.any()):
            break
    return {k: torch.stack(v) for k, v in log.items()}


log = rollout(args.max_steps, True)
T = log["S"].shape[0]
valid = log["valid"]  # (T, N): transition usable (env active and not finishing at this step)
first_done = torch.where((~valid).any(0), (~valid).float().argmax(0), torch.full((N,), T))  # step index of the finishing step
lengths = mgr.complete_motion_lengths.cpu()
print("\n=== per-env outcome (kneel envs 0-7) ===")
print("env motion                                         done_step ref_t  frac  fired")
for e in range(N):
    d = int(first_done[e])
    fired = [n for n, v in zip(TERMS, log["flags"][min(d, T - 1), e]) if v]
    print(f"{e:3d} {names[layout[e]][:48]:48s} {d:5d} {float(log['t_ref'][min(d, T - 1), e]):6.2f} {float(log['t_ref'][min(d, T - 1), e] / lengths[e]):5.2f}  {fired}")

# ---- heights / terminations around the failure (kneel envs) ----
print("\n=== link_pos_too_far / base: errors (robot - reference, z) at the failure step, kneel envs ===")
for e in range(8):
    d = min(int(first_done[e]), T - 1) - 1  # last step before the terminal one (the terminal step's state is already the reset state)
    le = log["link_err"][d, e]
    print(f"env {e}: step {d} ref_t {float(log['t_ref'][d, e]):.2f}s fired(next step) {[n for n, v in zip(TERMS, log['flags'][d + 1, e]) if v]}  "
          + "  ".join(f"{n}:{float(x):+.3f}" for n, x in zip(term_links, le)) + f"  | root z err {float(log['base_err'][d, e, 2]):+.3f} base dist {float(log['base_err'][d, e].norm()):.3f}")
print("\n=== robot vs reference z around the failure (env 0 = wall terrain, env 4 = noWall terrain) ===")
keyidx = {k: ref_links.index(v) for k, v in KEY.items() if v in ref_links}
for e in (0, 4):
    d = min(int(first_done[e]), T - 1) - 1
    print(f"env {e} ({names[layout[e]][:30]}), last pre-reset step {d}")
    print("  step  t_ref  " + "  ".join(f"{k:>14s}" for k in keyidx))
    for s in list(range(max(d - 60, 0), d + 1, 6)) + [d]:
        print(f"  {s:4d} {float(log['t_ref'][s, e]):5.2f}  " + "  ".join(f"{float(log['z_robot'][s, e, i]):6.2f}/{float(log['z_ref'][s, e, i]):6.2f}" for i in keyidx.values()))
    print("  (robot z / reference z)")

# ---- contacts: bodies touching (>1 N) in the steps before failure ----
print("\n=== bodies in contact (force > 1 N) before failure, kneel envs (every 4th step, last 24 steps) ===")
for e in (0, 1, 4, 5):
    d = min(int(first_done[e]), T - 1) - 1
    print(f"env {e}:")
    for s in range(max(d - 40, 0), d + 1, 4):
        touch = [body_names[i] for i in (log["contact"][s, e] > 1.0).nonzero().flatten().tolist()]
        print(f"  step {s:4d} t_ref {float(log['t_ref'][s, e]):.2f}: {touch}")

# ---- world-model state / visual-latent errors on the real transitions ----
@torch.no_grad()
def wm_errors():
    S_all, S1_all, a_all = log["S"].to(dev), log["S1"].to(dev), log["a"].to(dev)
    anch, clean1 = log["depth_anchor"].to(dev), log["depth_clean1"].to(dev)
    out = {}
    for group, envs in (("kneel", range(0, 8)), ("others", range(8, 16))):
        pairs = [(t, e) for e in envs for t in range(int(first_done[e])) if bool(valid[t, e])]
        t_i = torch.tensor([p[0] for p in pairs], device=dev)
        e_i = torch.tensor([p[1] for p in pairs], device=dev)
        d_i = first_done.to(dev)[e_i]
        near = (d_i - t_i) <= 30  # within 30 steps (0.6 s) of the finishing step
        S0 = unflatten_state(S_all[t_i, e_i])
        S1p, _, z1p = wm.step(S0, a_all[t_i, e_i], xi_n[e_i], mdp.scan(S0, noisy=False), encode(anch[t_i, e_i]))
        S1 = unflatten_state(S1_all[t_i, e_i])
        e1 = torch.stack([(S1p[0] - S1[0]).norm(dim=-1), rot_angle_sq(S1p[1], S1[1]).sqrt(), (S1p[4] - S1[4]).abs().mean(-1), (S1p[5] - S1[5]).abs().mean(-1)], -1)
        img = dec(z1p)  # decoded predicted latent
        dd1 = torch.stack([(img - clean1[t_i, e_i]).abs().mean((1, 2, 3)), (anch[t_i, e_i] - clean1[t_i, e_i]).abs().mean((1, 2, 3))], -1)
        # 8- and 32-step open-loop windows with the real actions
        res = {}
        for H in (8, 32):
            ok = torch.tensor([t + H <= int(first_done[e]) and bool(valid[t:t + H, e].all()) for t, e in pairs], device=dev)
            tw, ew = t_i[ok], e_i[ok]
            S, z = unflatten_state(S_all[tw, ew]), encode(anch[tw, ew])
            for j in range(H):
                S, _, z = wm.step(S, a_all[tw + j, ew], xi_n[ew], mdp.scan(S, noisy=False), z)
            Sr = unflatten_state(S1_all[tw + H - 1, ew])
            eH = torch.stack([(S[0] - Sr[0]).norm(dim=-1), rot_angle_sq(S[1], Sr[1]).sqrt(), (S[4] - Sr[4]).abs().mean(-1), (S[5] - Sr[5]).abs().mean(-1)], -1)
            imgH = dec(z)
            dH = torch.stack([(imgH - clean1[tw + H - 1, ew]).abs().mean((1, 2, 3)), (anch[tw, ew] - clean1[tw + H - 1, ew]).abs().mean((1, 2, 3))], -1)
            res[H] = (eH.cpu(), dH.cpu(), near[ok].cpu())
        out[group] = (e1.cpu(), dd1.cpu(), near.cpu(), res)
    return out

res = wm_errors()
print("\n=== world model on REAL transitions: root pos [m] / root rot [rad] / joint pos [rad] / joint vel [rad/s] mean errors; decoded latent L1 vs CLEAN image (world model | copy-anchor) ===")
print("group  subset                       n  |   pos     rot     q       qd    | depth 1-step")
for group in ("kneel", "others"):
    e1, dd1, near, res_g = res[group]
    for tag, sel in (("all", torch.ones_like(near)), ("last 0.6 s before finish", near), ("earlier", ~near)):
        if group == "others" and tag != "all":
            continue
        print(f"{group:6s} {tag:24s} {int(sel.sum()):5d} | " + " ".join(f"{float(x):7.4f}" for x in e1[sel].mean(0)) + f" | {float(dd1[sel][:, 0].mean()):.4f} | {float(dd1[sel][:, 1].mean()):.4f}")
    for H in (8, 32):
        eH, dH, nearH = res_g[H]
        for tag, sel in (("all", torch.ones_like(nearH)), ("window ends <=0.6s before finish", nearH)):
            if group == "others" and tag != "all":
                continue
            if int(sel.sum()) == 0:
                continue
            print(f"{group:6s} {H:2d}-step open loop, {tag:32s} n={int(sel.sum()):5d} | " + " ".join(f"{float(x):7.4f}" for x in eH[sel].mean(0)) + f" | depth {float(dH[sel][:, 0].mean()):.4f} (copy {float(dH[sel][:, 1].mean()):.4f})")


# ---- imagination optimism: does the world model think the policy survives where the real robot does not? ----
@torch.no_grad()
def imagination_vs_real(starts=(60, 80, 100, 110, 120, 140), H=32, real_depth=False):
    ac, norm = ev.runner.ac, ev.runner.normalizers["policy"]
    d0, d1 = ev.runner.depth_slice.start, ev.runner.depth_slice.stop
    rows = [(e, t) for e in range(N) for t in starts if t + 1 < int(first_done[e])]
    # batched imagination
    es = torch.tensor([r[0] for r in rows], device=dev)
    ts = torch.tensor([r[1] for r in rows], device=dev)
    S = unflatten_state(log["S"].to(dev)[ts, es])
    # reference frame index (global) of the start state: reference time -> frame
    clip = torch.tensor(layout, device=dev)[es]
    f = motion.clip_start[clip] + log["frame"].to(dev)[ts, es]
    f_end = motion.clip_end[clip]
    phase = torch.zeros(len(rows), device=dev)
    origin = log["origin"].to(dev)[ts, es]
    # proprio history from the logged states / actions (oldest first)
    hist = []
    for k in range(7, -1, -1):
        tk = (ts - k).clamp(min=0)
        hist.append(mdp.obs(unflatten_state(log["S"].to(dev)[tk, es]), log["a"].to(dev)[(tk - 1).clamp(min=0), es] * (tk > 0)[:, None], xi[es]))
    o_hist = mdp.add_noise(torch.stack(hist, 1))
    z = encode(log["depth_anchor"].to(dev)[ts, es])
    scan = mdp.scan(S, noisy=False)
    fell = torch.full((len(rows),), H, device=dev)
    survive = torch.ones(len(rows), device=dev)
    survive_soft = torch.ones(len(rows), device=dev)
    for k in range(H):
        # actor MLP on the policy observation with the latent in place of the encoder output (as in the runner)
        obs = norm(mdp.policy_obs(o_hist, mdp.goal(S, f, phase), z.new_zeros(z.shape[0], DEPTH_DIM)))
        a = torch.clamp(ac.actor(torch.cat((obs[:, :d0], obs[:, d1:], z), -1)), -20, 20)
        S, _, z = wm.step(S, a, xi_n[es], scan, z)
        scan = mdp.scan(S, noisy=False)
        if real_depth:  # feed the latent of the REAL (noised) depth image of that future step instead of the predicted one
            z = encode(log["depth_anchor"].to(dev)[(ts + k + 1).clamp(max=T - 1), es])
        o_hist = torch.cat((o_hist[:, 1:], mdp.add_noise(mdp.obs(S, a, xi[es])).unsqueeze(1)), 1)
        f = torch.minimum(f + 1, f_end)
        term = mdp.hard_termination(S, f, origin)
        survive_soft = survive_soft * (1 - mdp.soft_termination(S, f, origin))
        if fail_head is not None:
            feats = mdp.critic_features(S, f, origin, xi_n[es], mdp.goal(S, f, phase, noisy=False), mdp.scan(S, noisy=False))
            survive = survive * (1 - mdp.soft_termination(S, f, origin)) * (1 - torch.sigmoid(fail_head(feats)).clamp(max=0.95))
        fell = torch.where((fell == H) & term, torch.full_like(fell, k + 1), fell)
    real_fail_in = (first_done.to(dev)[es] - ts)  # steps until the real finishing step
    return es.cpu(), ts.cpu(), fell.cpu(), real_fail_in.cpu(), (1 - survive_soft).cpu(), ((1 - survive).cpu() if fail_head is not None else None)


for tag, rd in (("vision = world-model latent (as in training)", False), ("vision = latent of the REAL image of each future step (isolates dynamics)", True)):
    es_, ts_, fell_, real_, p_soft_, p_head_ = imagination_vs_real(real_depth=rd)
    print(f"\n=== imagination (world model + policy) vs reality from the same real start states, horizon 32; {tag} ===")
    print("group    n   imagined terminates within 32 steps   real terminates within 32 steps   mean imagined step of termination (when it does)")
    for group, sel in (("kneel", es_ < 8), ("others", es_ >= 8)):
        im = fell_[sel] < 32
        rl = real_[sel] <= 32
        print(f"{group:7s} {int(sel.sum()):3d}   {float(im.float().mean()) * 100:5.1f}%                              {float(rl.float().mean()) * 100:5.1f}%                          {float(fell_[sel][im].float().mean()) if im.any() else float('nan'):.1f}")
    for group, sel in (("kneel", es_ < 8), ("others", es_ >= 8)):
        line = f"{group:7s} P(terminate within 32) imagined: analytic soft {float(p_soft_[sel].mean()):.3f}"
        if p_head_ is not None:
            line += f", analytic soft + failure head {float(p_head_[sel].mean()):.3f}"
        print(line + f"; real {float((real_[sel] <= 32).float().mean()):.3f}")
    sel = (es_ < 8) & (real_ <= 32)
    if sel.any():
        print("kneel starts with a real failure within 32 steps: imagined termination steps vs real failure steps:")
        print("   imagined:", fell_[sel].tolist(), "\n   real:    ", real_[sel].tolist())

# ---- contact head on real transitions ----
with torch.no_grad():
    cb = torch.tensor(reader.contact_body_ids, device=dev)
    for group, envs in (("kneel", range(0, 8)), ("others", range(8, 16))):
        pairs = [(t, e) for e in envs for t in range(int(first_done[e])) if bool(valid[t, e])]
        t_i = torch.tensor([p[0] for p in pairs], device=dev)
        e_i = torch.tensor([p[1] for p in pairs], device=dev)
        S0 = unflatten_state(log["S"].to(dev)[t_i, e_i])
        _, lg, _ = wm.step(S0, log["a"].to(dev)[t_i, e_i], xi_n[e_i], mdp.scan(S0, noisy=False), encode(log["depth_anchor"].to(dev)[t_i, e_i]))
        label = (log["contact"].to(dev)[t_i, e_i][:, cb] > 1.0).float()
        pred = torch.sigmoid(lg)
        pos = label > 0.5
        near = ((first_done.to(dev)[e_i] - t_i) <= 30)
        for tag, m in (("all", torch.ones_like(near)), ("last 0.6s", near)):
            if group == "others" and tag != "all":
                continue
            l, p_ = label[m], pred[m]
            print(f"contact head [{group} {tag}]: BCE {float(torch.nn.functional.binary_cross_entropy(p_, l)):.4f}, positives {float(l.mean()) * 100:.2f}% of body-steps, "
                  f"recall@0.5 {float(((p_ > 0.5) & (l > 0.5)).sum() / (l > 0.5).sum().clamp(min=1)) * 100:.0f}%, precision@0.5 {float(((p_ > 0.5) & (l > 0.5)).sum() / (p_ > 0.5).sum().clamp(min=1)) * 100:.0f}%")

# ---- reference vs terrain heights (heightmap) around the failure ----
print("\n=== z of reference / robot vs terrain surface below (heightmap), R knee, L ankle, wrists; kneel env 0 (wall) / env 4 (noWall) ===")
origins = log["origin"][0].numpy()
for e in (0, 4):
    d = int(first_done[e]) - 1
    print(f"env {e}: step | ref_t | R knee: ref z, ground, ref-ground | robot z, robot-ground || L wrist: ref z, ground, ref-ground | robot z, robot-ground")
    for st in range(max(d - 60, 0), d + 1, 6):
        row = []
        for link in ("right_knee_link", "left_wrist_yaw_link"):
            i = ref_links.index(link)
            # reference link world position from the reference tensors (local + origin) at this step: z_ref logged, xy from reference_frame is not logged -> use motion lib
            fl = int(log["frame"][st, e]); gi = int(motion.clip_start[layout[e]]) + fl
            ref_xy = (motion.link_p[gi, i, :2] + torch.tensor(origins[e][:2], device=dev))[None]
            gz = float(scanner.ground_height(ref_xy[None])[0, 0])
            zr_, zf_ = float(log["z_robot"][st, e, i]), float(log["z_ref"][st, e, i])
            row.append(f"{zf_:5.2f} {gz:5.2f} {zf_ - gz:+5.2f} | {zr_:5.2f} {zr_ - gz:+5.2f}")
        print(f"  {st:4d} {float(log['t_ref'][st, e]):5.2f} | " + " || ".join(row))

# ---- reference vs terrain geometry ----
print("\n=== reference vs terrain: robot collision geoms placed in the REFERENCE pose, contacts with terrain geoms ===")
cpu = copy.deepcopy(env.sim.mj_model)
terrain_body = cpu.body("terrain").id
data = mujoco.MjData(cpu)
free_q = robot.indexing.free_joint_q_adr.cpu().numpy()
joint_q = robot.indexing.joint_q_adr.cpu().numpy()
seqs = next(iter(mgr.motion_buffers.values()))._all_motion_sequences
bodies_of_geom = cpu.geom_bodyid


def reference_contacts(motion_idx: int, origin: np.ndarray, stride: int = 3):
    pos = seqs.base_pos_w[motion_idx].cpu().numpy()
    quat = seqs.base_quat_w[motion_idx].cpu().numpy()
    q = seqs.joint_pos[motion_idx].cpu().numpy()
    rows = []
    for fr in range(0, len(pos), stride):
        data.qpos[:] = cpu.qpos0
        data.qpos[free_q[:3]] = pos[fr] + origin
        data.qpos[free_q[3:7]] = quat[fr]
        data.qpos[joint_q] = q[fr]
        mujoco.mj_forward(cpu, data)
        pen, who, touching = 0.0, "", set()
        for c in data.contact[: data.ncon]:
            g1, g2 = int(c.geom1), int(c.geom2)
            b1, b2 = int(bodies_of_geom[g1]), int(bodies_of_geom[g2])
            if (b1 == terrain_body) == (b2 == terrain_body):
                continue
            rb = b2 if b1 == terrain_body else b1
            nm = mujoco.mj_id2name(cpu, mujoco.mjtObj.mjOBJ_BODY, rb).split("/")[-1]
            if c.dist < 0.01:
                touching.add(nm)
            if -c.dist > pen:
                pen, who = float(-c.dist), nm
        rows.append((fr, pen, who, sorted(touching)))
    return rows

origins = next(iter(mgr.motion_buffers.values()))._assigned_env_origins.cpu().numpy()
for tag, e in (("kneel (wall terrain)", 0), ("kneel (noWall terrain)", 4), ("jumpsit2 (wall)", 8), ("rollVault (wall)", 10), ("diveroll (simpleLab)", 12)):
    rows = reference_contacts(layout[e], origins[e])
    pens = np.array([r[1] for r in rows])
    n = len(rows)
    print(f"{tag}: frames {n * 3}, ref penetration >1cm in {np.mean(pens > 0.01) * 100:.0f}% of frames, >3cm in {np.mean(pens > 0.03) * 100:.0f}%, max {pens.max() * 100:.1f} cm "
          f"(at frame {rows[int(pens.argmax())][0]}, body {rows[int(pens.argmax())][2]})")
    if e in (0, 4):
        d = min(int(first_done[e]), T - 1)
        f_fail = int(log["frame"][d, e])
        print(f"   failure at reference frame {f_fail} (t={f_fail / 50:.2f}s). reference contacts (dist<1cm) / penetration around it:")
        for fr, pen, who, touch in rows:
            if abs(fr - f_fail) <= 45:
                print(f"   frame {fr:4d}: max penetration {pen * 100:5.1f} cm ({who}), bodies touching: {touch}")

# ---- re-run without the tracking terminations ----
print("\n=== kneel envs re-run with link_pos_too_far / base_pos_too_far / base_pg_too_far disabled (thresholds -> 100) ===")
for name in ("link_pos_too_far", "base_pos_too_far", "base_pg_too_far"):
    cfg_term = tm.get_term_cfg(name)
    key = "distance_threshold" if "distance_threshold" in cfg_term.params else "projected_gravity_threshold"
    cfg_term.params[key] = 100.0
log2 = rollout(args.max_steps, False)
T2 = log2["S"].shape[0]
v2 = log2["valid"]
d2 = torch.where((~v2).any(0), (~v2).float().argmax(0), torch.full((N,), T2))
print("env  steps  ref_t_end  max|link z err|(ankle/wrist)  [time of max, ref t]   max|root z err|   err at end   finished(dataset_exhausted)")
for e in range(16):
    d = int(d2[e])
    le = log2["link_err"][:d, e].abs()
    peak = float(le.max())
    where = int(le.max(-1).values.argmax())
    rz = log2["base_err"][:d, e, 2].abs()
    print(f"{e:3d} {d:6d} {float(log2['t_ref'][min(d, T2 - 1), e]):7.2f}   {peak:6.3f} (t={float(log2['t_ref'][where, e]):5.2f}s, link {term_links[int(le[where].argmax())]})   {float(rz.max()):6.3f}   "
          f"{float(le[-5:].mean()):6.3f}   {bool(log2['flags'][min(d, T2 - 1), e, TERMS.index('dataset_exhausted')])}")
torch.save({k: v for k, v in log.items() if k in ("t_ref", "z_robot", "z_ref", "link_err", "flags", "valid")}, os.path.join(args.out, "log_main.pt"))

# ---- plot ----
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axs = plt.subplots(2, 2, figsize=(14, 8))
for col, e in enumerate((0, 4)):
    d = min(int(first_done[e]), T - 1)
    t = log["t_ref"][:d + 1, e]
    for ax, names_ in ((axs[0, col], ["L ankle", "R ankle", "L wrist", "R wrist"]), (axs[1, col], ["L knee", "R knee", "root(torso)", "pelvis"])):
        for k in names_:
            i = keyidx[k]
            ax.plot(t, log["z_robot"][:d + 1, e, i], label=f"{k} robot")
            ax.plot(t, log["z_ref"][:d + 1, e, i], "--", label=f"{k} ref")
        ax.axvline(float(t[-1]), color="k", lw=0.5)
        ax.set_title(f"env {e} {names[layout[e]][:28]} (fails at t={float(t[-1]):.2f}s)")
        ax.set_xlabel("reference time [s]"), ax.set_ylabel("z [m]"), ax.legend(fontsize=6, ncol=2)
plt.tight_layout()
plt.savefig(os.path.join(args.out, "kneel_heights.png"), dpi=90)
print("saved", os.path.join(args.out, "kneel_heights.png"))
