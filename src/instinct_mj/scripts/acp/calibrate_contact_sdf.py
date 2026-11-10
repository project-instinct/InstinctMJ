"""Calibrate / check the "on" distances ``d_on_b`` of the SDF contact (``rl/acp/sdf_contact.py``).

    ACP_DATASET=<dir> python -m instinct_mj.scripts.acp.calibrate_contact_sdf reference [--out stats.json]
    ACP_DATASET=<dir> python -m instinct_mj.scripts.acp.calibrate_contact_sdf rollout <checkpoint> [--num-envs 256] [--steps 600]

``reference``: d_b of the nine bodies on every frame of every reference clip (placed at its tile) against the sidecar labels ``contact_flag``: the distance distribution on the
labelled-contact frames, the agreement of ``d_b < d_on`` with the labels per body and per clip family at the current ``D_ON``, and the threshold that maximizes the agreement.
``rollout``: the simulated robot (the checkpoint's policy, 50 Hz): d_b at the steps where the contact sensor measures |F_z| > 1 N (max over the 4 substeps) and where it does not."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import torch

import instinct_mj.tasks.acp.config.g1.acp_tracking_cfg as acp_g1

dataset = os.path.expanduser(os.environ["ACP_DATASET"])
buffer = acp_g1.motion_reference_cfg.motion_buffers["TerrainMotion"]
buffer.path, buffer.metadata_yaml = dataset, os.path.join(dataset, "metadata.yaml")

import instinct_mj.tasks  # noqa: E402,F401
from instinct_mj.rl import InstinctRlVecEnvWrapper  # noqa: E402
from instinct_mj.rl.acp.planner import CONTACT_BODIES  # noqa: E402
from instinct_mj.rl.acp.runner import clip_family  # noqa: E402
from instinct_mj.rl.acp.sdf_contact import D_ON, env_contact_distance, get_sdf_contact  # noqa: E402
from instinct_mj.tasks.registry import load_env_cfg, load_env_cls, load_instinct_rl_cfg, load_runner_cls  # noqa: E402

TASK = "Instinct-ACP-Track-G1-v1"
GRID = np.arange(0.0005, 0.0305, 0.0005)


def best_threshold(d: np.ndarray, label: np.ndarray) -> dict:
    """Accuracy / F1 of ``d < t`` against ``label`` for the thresholds GRID; the best-accuracy one."""
    out = []
    for t in GRID:
        p = d < t
        tp, fp, fn = (p & label).sum(), (p & ~label).sum(), (~p & label).sum()
        out.append(((p == label).mean(), 2 * tp / max(2 * tp + fp + fn, 1)))
    acc = np.array([o[0] for o in out])
    i = int(acc.argmax())
    return {"t_best": float(GRID[i]), "acc": float(acc[i]), "f1": float(out[i][1])}


def summarize(d: np.ndarray, label: np.ndarray, title: str, groups: np.ndarray | None = None, group_names: list[str] | None = None) -> dict:
    print(f"\n== {title}: {len(d)} samples")
    q = [0.5, 0.9, 0.95, 0.99]
    print(f"{'body':14s} {'pos%':>5s} {'d|pos q50/90/95/99 [mm]':>28s} {'d|neg q10/50 [mm]':>20s} {'acc@D_ON':>9s} {'F1@D_ON':>8s}  best t[mm] acc  F1")
    res = {}
    for b, name in enumerate(CONTACT_BODIES):
        db, lb = d[:, b], label[:, b]
        p = db < D_ON[b]
        tp, fp, fn = (p & lb).sum(), (p & ~lb).sum(), (~p & lb).sum()
        pos = np.quantile(db[lb], q) * 1e3 if lb.any() else np.full(4, np.nan)
        neg = np.quantile(db[~lb], [0.1, 0.5]) * 1e3 if (~lb).any() else np.full(2, np.nan)
        bt = best_threshold(db, lb)
        res[name] = {"pos_frac": float(lb.mean()), "acc": float((p == lb).mean()), "f1": float(2 * tp / max(2 * tp + fp + fn, 1)), **bt, "d_pos_q": pos.tolist()}
        print(f"{name:14s} {100 * lb.mean():5.1f} {pos[0]:7.1f}{pos[1]:7.1f}{pos[2]:7.1f}{pos[3]:7.1f} {neg[0]:10.1f}{neg[1]:9.1f} {res[name]['acc']:9.3f} {res[name]['f1']:8.3f}  {bt['t_best']*1e3:6.1f} {bt['acc']:.3f} {bt['f1']:.3f}")
    allp = d < np.array(D_ON)
    print(f"all bodies: agreement {(allp == label).mean():.4f}")
    res["all"] = float((allp == label).mean())
    if groups is not None:
        print(f"{'family':12s} " + " ".join(f"{n[:9]:>9s}" for n in CONTACT_BODIES) + "      all   frames")
        for g, gname in enumerate(group_names):
            m = groups == g
            if m.any():
                per = (allp[m] == label[m]).mean(0)
                print(f"{gname:12s} " + " ".join(f"{x:9.3f}" for x in per) + f" {per.mean():8.3f} {m.sum():8d}")
                res[f"family_{gname}"] = per.tolist()
    return res


def reference(env, out: str | None) -> None:
    from instinct_mj.rl.acp.motion import ACPMotion

    device = str(env.device)
    manager = env.scene["motion_reference"]
    contacts = env.command_manager.get_term("acp_reference_contacts")
    motion = ACPMotion(manager, contacts, device)
    sdf = get_sdf_contact(env)
    start = env.curriculum_manager.get_term_cfg("episode_start").func
    mb = start.motion_buffer
    names = [os.path.splitext(os.path.basename(f))[0] for f in next(iter(manager.motion_buffers.values()))._all_motion_files]
    families_all = [clip_family(e.get("group", "")) for e in mb.yaml_data["motion_files"]]
    fam_names = sorted(set(families_all))
    w0 = mb._motion_weights.clone()
    D, L, G, CL = [], [], [], []
    from mjlab.utils.lab_api.math import matrix_from_quat  # noqa: F401

    for k in range(len(names)):
        mb._motion_weights[:] = 0.0
        mb._motion_weights[k] = 1.0
        env.reset(env_ids=torch.arange(env.num_envs, device=device))
        origin = motion.current_frame()[2][0]
        T = int(motion.clip_len[k])
        f = motion.clip_start[k] + torch.arange(T, device=device)
        bp = motion.link_p[f] + origin
        bR = motion.link_R[f]
        d = sdf.distance(bp, bR)
        flag = motion.flag(f, torch.zeros(T, device=device))
        D.append(d.cpu()), L.append(flag.cpu()), G.append(torch.full((T,), fam_names.index(families_all[k]))), CL.append(torch.full((T,), k))
    mb._motion_weights[:] = w0
    d, lab, grp = torch.cat(D).numpy(), torch.cat(L).numpy().astype(bool), torch.cat(G).numpy()
    res = summarize(d, lab, f"reference frames of {len(names)} clips", grp, fam_names)
    if out:
        json.dump(res, open(out, "w"), indent=1)


def rollout(env, vec, checkpoint: str, steps: int, out: str | None) -> None:
    device = str(env.device)
    agent = load_instinct_rl_cfg(TASK).to_dict()
    agent.update(compile=False, compile_policy=False, buffer_steps=64, warmup_steps=32, wm_scan="planner")  # (the checkpoint was trained with the 17 x 11 world-model scan)
    runner = load_runner_cls(TASK)(vec, agent, None, device)
    runner.load(checkpoint)
    runner.resume_path = None  # the policy only: no sidecar (world model, bins) of the checkpoint's own dataset
    runner.setup_training()
    policy = runner.get_inference_policy(device=device)
    torch.set_grad_enabled(False)
    env.cfg.auto_reset = False
    contacts = env.command_manager.get_term("acp_reference_contacts")
    obs, _ = vec.get_observations()
    D, F, R = [], [], []
    for _ in range(steps):
        a = torch.clamp(policy(obs) + 0.1 * torch.randn(env.num_envs, 23, device=device), -20, 20)
        obs, _, dones, _ = vec.step(a)
        d = env_contact_distance(env, contacts.contact_body_ids)
        force = contacts.contact_sensor.data.force_history[:, contacts.contact_sensor_ids, :, 2].abs().amax(-1) > 1.0
        D.append(d.cpu()), F.append(force.cpu()), R.append(contacts.ref_contact_flag.cpu())
        if dones.any():
            env.reset(env_ids=dones.nonzero().squeeze(-1))
            obs, _ = vec.get_observations()
    d, force, ref = torch.cat(D).numpy(), torch.cat(F).numpy(), torch.cat(R).numpy()
    if out:
        np.savez(os.path.splitext(out)[0] + "_raw.npz", d=d, force=force, ref=ref)
    res = summarize(d, force, "simulated robot: label = measured |F_z| > 1 N")
    res["vs_ref"] = summarize(d, ref, "simulated robot: label = reference flag")
    if out:
        json.dump(res, open(out, "w"), indent=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("reference", "rollout"))
    ap.add_argument("checkpoint", nargs="?")
    ap.add_argument("--num-envs", type=int, default=256)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = load_env_cfg(TASK)
    cfg.scene.num_envs = args.num_envs if args.mode == "rollout" else 8
    env = load_env_cls(TASK)(cfg=cfg, device="cuda:0")
    vec = InstinctRlVecEnvWrapper(env, policy_group="policy", critic_group="critic")
    vec.reset()
    with torch.no_grad():
        reference(env, args.out) if args.mode == "reference" else rollout(env, vec, args.checkpoint, args.steps, args.out)


if __name__ == "__main__":
    main()
