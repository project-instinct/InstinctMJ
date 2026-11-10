"""Record the rollout of an ACP tracker checkpoint on every clip of the dataset as a motion dataset the web viewer (``acp_player/server.py``) can play next to the references.

The setting is that of ``eval_tracker --mode paper`` (no domain randomization / sensor noise / action delay / action noise / reset noise, only the fall terminations end an episode early),
ONE deterministic episode per clip from the clip START (phase 0) until the clip end or a fall; ``--num-envs`` clips run in parallel (one env per clip).

The output ``<out-root>/<run>_<iteration>/`` mirrors the layout of ``ACP_DATASET``: the same group folders, the clip ``<name>.npz`` becomes ``<name>__policy.npz`` (``framerate``, ``joint_names``,
``joint_pos``, ``base_pos_w``, ``base_quat_w`` like the dataset clips; the world positions of the env have the terrain origin of the env subtracted, so the rollout sits exactly where the
reference sits in the terrain frame of its clip), ``metadata.yaml`` with the SAME terrain files (``terrains`` is a symlink to the one of the dataset) and the reference ``.contacts.npz`` sidecars as
symlinks. ``rollouts.yaml`` has the per-clip outcome (survived / fell, frame of the fall, success per the paper criterion) and ``curation.yaml`` marks it in the clip list of the viewer
(keep = success, edited = reached the clip end but missed the goal, reject = fell).

    ACP_DATASET=<dir> python -m instinct_mj.scripts.acp.record_rollouts [checkpoint] [--task ch5] [--out-root ~/Datasets/acp/rollouts] [--num-envs 128] [--clip-regex .*]

``checkpoint`` defaults to the newest ``logs/instinct_rl/g1_acp_track_final6/*/model_*.pt``.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import shutil
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
import torch
import yaml

import instinct_mj.tasks.acp.config.g1.acp_tracking_cfg as acp_g1

dataset = os.path.expanduser(os.environ["ACP_DATASET"])
buffer = acp_g1.motion_reference_cfg.motion_buffers["TerrainMotion"]
buffer.path, buffer.metadata_yaml = dataset, os.path.join(dataset, "metadata.yaml")

import instinct_mj.tasks  # noqa: E402,F401  (registers the tasks)
from instinct_mj.rl import InstinctRlVecEnvWrapper  # noqa: E402
from instinct_mj.rl.acp.runner import clip_family  # noqa: E402
from instinct_mj.tasks.registry import load_env_cfg, load_env_cls, load_instinct_rl_cfg, load_runner_cls  # noqa: E402

SUFFIX = "__policy"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", nargs="?", default=None, help="model_<iteration>.pt (default: the newest of logs/instinct_rl/g1_acp_track_final6)")
    parser.add_argument("--task", default="Instinct-ACP-Track-G1-v1", help="task id (the terminations of the run the checkpoint comes from)")
    parser.add_argument("--out-root", default="~/Datasets/acp/rollouts")
    parser.add_argument("--num-envs", type=int, default=128, help="clips recorded in parallel (one env per clip)")
    parser.add_argument("--max-steps", type=int, default=1100, help="control steps per batch of clips (longer than the longest clip)")
    parser.add_argument("--xy-end", type=float, default=0.5, help="success: horizontal pelvis error at the episode end below this [m]")
    parser.add_argument("--z-end", type=float, default=0.1, help="get_up clips: success also needs the pelvis height error at the episode end below this [m] (ACP: vertical error < 0.1 m)")
    parser.add_argument("--clip-regex", default=".*", help="only the clips whose name matches")
    args = parser.parse_args()
    if args.checkpoint is None:
        args.checkpoint = max(glob.glob("logs/instinct_rl/g1_acp_track_final6/*/model_*.pt"), key=lambda p: int(re.search(r"model_(\d+)\.pt", p).group(1)))
    args.checkpoint = os.path.abspath(args.checkpoint)
    device = "cuda:0"
    cfg = load_env_cfg(args.task)
    cfg.scene.num_envs = args.num_envs
    cfg.episode_length_s = 120.0  # the episodes last until the clip end
    cfg.disable_randomization()  # no physics / sensor / odometry randomization, no pose and joint reset noise
    for actuator in cfg.scene.entities["robot"].articulation.actuators:
        actuator.delay_max_lag = 0
    cfg.events["reset_robot"].params["randomize_velocity_range"] = {}
    env = load_env_cls(args.task)(cfg=cfg, device=device)
    vec = InstinctRlVecEnvWrapper(env, policy_group="policy", critic_group="critic")
    agent = load_instinct_rl_cfg(args.task).to_dict()
    agent.update(compile=False, compile_policy=False, buffer_steps=64, warmup_steps=32)
    runner = load_runner_cls(args.task)(vec, agent, None, device)
    runner.load(args.checkpoint)
    runner.setup_training()
    policy = runner.get_inference_policy(device=device)
    torch.set_grad_enabled(False)
    start = env.curriculum_manager.get_term_cfg("episode_start").func
    start.mode = "start"  # every episode starts at phase 0 of its clip
    mb, mdp, reader, motion = start.motion_buffer, runner.mdp, runner.reader, runner.motion
    names = runner.motion_names
    metadata = yaml.safe_load(open(os.path.join(dataset, "metadata.yaml")))
    entries = metadata["motion_files"]
    assert [os.path.splitext(os.path.basename(e["motion_file"]))[0] for e in entries] == names
    families = [clip_family(e.get("group", "")) for e in entries]
    N = args.num_envs
    weights0 = mb._motion_weights.clone()
    env.cfg.auto_reset = False
    get_up = torch.tensor([f == "get_up" for f in families], device=device)
    selected = [k for k, n in enumerate(names) if re.search(args.clip_regex, n)]
    term_names = [n for n in env.termination_manager.active_terms if n not in ("time_out", "dataset_exhausted")]
    joint_names = list(env.scene["robot"].joint_names)
    fps = motion.framerate
    free_q, joint_q = reader.free_q, reader.joint_q

    experiment = os.path.basename(os.path.dirname(os.path.dirname(args.checkpoint))).removeprefix("g1_acp_track_")
    iteration = re.search(r"model_(\d+)\.pt", args.checkpoint).group(1)
    out = os.path.join(os.path.expanduser(args.out_root), f"{experiment}_{iteration}")
    os.makedirs(out, exist_ok=True)

    def reset_clip(env_ids: torch.Tensor, clip: int) -> None:
        mb._motion_weights[:] = 0.0
        mb._motion_weights[clip] = 1.0
        env.reset(env_ids=env_ids)

    def pose() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pelvis position, pelvis quaternion (w, x, y, z) and joint positions of every env, exact from ``qpos`` (world frame of the env)."""
        qpos = env.sim.data.qpos
        return qpos[:, free_q[:3]].clone(), qpos[:, free_q[3:7]].clone(), qpos[:, joint_q].clone()

    results: dict[str, dict] = {}
    for first in range(0, len(selected), N):
        batch = selected[first : first + N]
        env_clip = torch.full((N,), batch[0], device=device, dtype=torch.long)
        active = torch.zeros(N, dtype=torch.bool, device=device)  # the envs that have not finished their episode yet
        origin = torch.zeros(N, 3, device=device)
        for i, k in enumerate(batch):
            env_clip[i] = k
            active[i] = True
            reset_clip(torch.tensor([i], device=device), k)
            origin[i] = motion.current_frame()[2][i]
        if len(batch) < N:
            reset_clip(torch.arange(len(batch), N, device=device), batch[0])  # the idle envs run along and are ignored
        obs, _ = vec.get_observations()
        T = args.max_steps + 1
        rec_p, rec_q, rec_j = torch.zeros(N, T, 3, device=device), torch.zeros(N, T, 4, device=device), torch.zeros(N, T, len(joint_names), device=device)
        p, q, j = pose()
        rec_p[:, 0], rec_q[:, 0], rec_j[:, 0] = p - origin, q, j
        for i, k in enumerate(batch):  # the frame convention: the recorded start pose is the reference start pose of the clip file
            ref = np.load(os.path.join(dataset, entries[k]["motion_file"]))
            assert np.abs(rec_p[i, 0].cpu().numpy() - ref["base_pos_w"][0]).max() < 2e-2, f"{names[k]}: start pose differs from the reference"
        frames = torch.zeros(N, dtype=torch.long, device=device)
        fall_frame = torch.full((N,), -1, dtype=torch.long, device=device)
        survived, success = torch.zeros(N, dtype=torch.bool, device=device), torch.zeros(N, dtype=torch.bool, device=device)
        final_xy, final_z = torch.zeros(N, device=device), torch.zeros(N, device=device)
        fired = torch.zeros(N, len(term_names) + 1, dtype=torch.bool, device=device)  # terminations of every finished env (last: non-finite state)
        steps = 0
        while steps < args.max_steps and bool(active.any()):
            S0 = reader.state()
            a = policy(obs)
            obs, _, _, _ = vec.step(torch.clamp(a, -20, 20))
            steps += 1
            S1 = reader.state()
            bad = ~torch.stack([torch.isfinite(x).flatten(1).all(-1) for x in S1]).all(0)
            S1 = tuple(torch.where(bad.view(-1, *[1] * (x0.dim() - 1)), x0, x1) for x0, x1 in zip(S0, S1))
            p, q, j = pose()
            rec_p[:, steps], rec_q[:, steps], rec_j[:, steps] = p - origin, q, j
            f1, phase1, origin1 = motion.current_frame()
            pelvis_err = S1[0] - mdp.reference_state(f1, origin1)[0]
            xy = pelvis_err[:, :2].norm(dim=-1)  # horizontal pelvis error to the reference
            tm = env.termination_manager
            done = tm.dones | bad
            if done.any():
                ids = done.nonzero().squeeze(-1)
                first_ids = ids[active[ids]]  # only the first completion of an env counts
                if len(first_ids) > 0:
                    failed = env.reset_terminated[first_ids] | bad[first_ids]  # (the time-outs and the clip end are no failures)
                    reached = (tm.get_term("dataset_exhausted")[first_ids] | tm.get_term("time_out")[first_ids]) & ~failed
                    clip = env_clip[first_ids]
                    progress = (xy[first_ids] < args.xy_end) & (get_up[clip].logical_not() | (pelvis_err[first_ids, 2].abs() < args.z_end))
                    survived[first_ids] = reached
                    success[first_ids] = reached & progress
                    frames[first_ids] = steps + 1
                    fall_frame[first_ids] = torch.where(failed, steps, -1)
                    final_xy[first_ids], final_z[first_ids] = xy[first_ids], pelvis_err[first_ids, 2]
                    fired[first_ids] = torch.stack([tm.get_term(n)[first_ids] for n in term_names] + [bad[first_ids]], -1)
                    active[first_ids] = False
                reset_clip(ids, batch[0])  # (the finished envs run along, ignored)
                obs, _ = vec.get_observations()
        unfinished = active.clone()
        frames[unfinished] = steps + 1  # not finished within the step limit: counts as a failure
        print(f"batch {first // N + 1} of {-(-len(selected) // N)}: {steps} steps, unfinished {int(unfinished.sum())}", flush=True)
        rec_p, rec_q, rec_j = rec_p.cpu().numpy(), rec_q.cpu().numpy(), rec_j.cpu().numpy()
        for i, k in enumerate(batch):
            entry = entries[k]
            length = min(int(frames[i]), int(motion.clip_len[k]))  # (the reference is clamped to its last frame past the clip end)
            rel = entry["motion_file"][: -len(".npz")] + SUFFIX
            os.makedirs(os.path.dirname(os.path.join(out, rel)), exist_ok=True)
            ref = np.load(os.path.join(dataset, entry["motion_file"]))  # (the clip files have all 29 joints, the robot of the task only the actuated ones: the other joints keep the reference values)
            columns = [list(ref["joint_names"]).index(n) for n in joint_names]
            joint_pos = ref["joint_pos"][np.minimum(np.arange(length), len(ref["joint_pos"]) - 1)].copy()
            joint_pos[:, columns] = rec_j[i, :length]
            np.savez(os.path.join(out, rel + ".npz"), framerate=np.float64(fps), joint_names=ref["joint_names"], joint_pos=joint_pos.astype(np.float32),
                     base_pos_w=rec_p[i, :length].astype(np.float64), base_quat_w=rec_q[i, :length].astype(np.float32))
            sidecar = os.path.join(out, rel + ".contacts.npz")
            if os.path.lexists(sidecar):
                os.remove(sidecar)
            os.symlink(os.path.join(dataset, entry["motion_file"][: -len(".npz")] + ".contacts.npz"), sidecar)
            fell = int(fall_frame[i]) >= 0 or (not bool(survived[i]))
            results[rel + ".npz"] = dict(clip=names[k], family=families[k], group=entry.get("group", ""), frames=length, clip_frames=int(motion.clip_len[k]), survived=bool(survived[i]),
                                         fell=bool(fell), fall_frame=int(fall_frame[i]), unfinished=bool(unfinished[i]), success=bool(success[i]), final_xy_error=float(final_xy[i]),
                                         final_z_error=float(final_z[i]), ended_by=[n for n, f in zip([*term_names, "nonfinite"], fired[i].tolist()) if f])
    mb._motion_weights[:] = weights0

    selected_files = {e["motion_file"] for k, e in enumerate(entries) if k in set(selected)}
    out_meta = {key: value for key, value in metadata.items() if key != "motion_files"}
    out_meta["motion_files"] = [dict(e, motion_file=e["motion_file"][: -len(".npz")] + SUFFIX + ".npz") for e in entries if e["motion_file"] in selected_files]
    yaml.safe_dump(out_meta, open(os.path.join(out, "metadata.yaml"), "w"), sort_keys=False)
    if not os.path.lexists(os.path.join(out, "terrains")):
        os.symlink(os.path.join(dataset, "terrains"), os.path.join(out, "terrains"))
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    curation = {}
    for rel, r in results.items():
        if r["fell"]:
            status, note = "reject", f"fell / terminated at frame {r['fall_frame'] if r['fall_frame'] >= 0 else r['frames'] - 1} of {r['clip_frames']} ({', '.join(r['ended_by']) or 'step limit'})"
        elif r["success"]:
            status, note = "keep", f"success: end error xy {r['final_xy_error']:.2f} m, z {r['final_z_error']:+.2f} m"
        else:
            status, note = "edited", f"survived but missed the goal: end error xy {r['final_xy_error']:.2f} m, z {r['final_z_error']:+.2f} m"
        curation[os.path.splitext(rel)[0] + ".npz"] = dict(status=status, note=note, time=now)
    yaml.safe_dump(curation, open(os.path.join(out, "curation.yaml"), "w"), allow_unicode=True, sort_keys=False)
    yaml.safe_dump(dict(checkpoint=args.checkpoint, task=args.task, dataset=dataset, xy_end=args.xy_end, z_end=args.z_end, clips=results), open(os.path.join(out, "rollouts.yaml"), "w"), sort_keys=False)

    fam = defaultdict(lambda: np.zeros(4))
    for r in results.values():
        fam[r["family"]] += [1, r["fell"], r["survived"], r["success"]]
    print(f"\ncheckpoint {args.checkpoint}\nwritten to {out}")
    print(f"{'family':12s} {'clips':>5s} {'fell':>5s} {'survived':>8s} {'success':>7s}")
    for name, v in sorted(fam.items()):
        print(f"{name:12s} {int(v[0]):5d} {int(v[1]):5d} {int(v[2]):8d} {int(v[3]):7d}")
    total = sum(fam.values())
    print(f"{'all':12s} {int(total[0]):5d} {int(total[1]):5d} {int(total[2]):8d} {int(total[3]):7d}")
    print(f"peak GPU memory allocated by torch: {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")


if __name__ == "__main__":
    main()
