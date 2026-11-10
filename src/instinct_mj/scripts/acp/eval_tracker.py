"""Per-clip evaluation of an ACP tracker checkpoint (the decision to stop Stage 1): every clip of the dataset from the clip START (phase 0), episodes of the full clip length
(no episode cap), a FIXED COHORT per clip: ``--episodes`` envs run exactly one episode of the clip each, only the first completion of every env is counted and a batch of clips is
finished when its whole cohort is (no re-runs of the early failures, which would over-weight the fast failures and censor the long successful episodes).

``--mode robust``: the training settings (domain randomization, observation noise, action delay, Gaussian action noise ``--noise``, reset noise).
``--mode paper``: the ACP evaluation: no sensor noise, no physics randomization, no action delay, no action noise, no reset noise, and only the FALL terminations (root height, tilt) can end an
episode early (the contact and horizontal-position terminations of the training task are removed); success = within ``--xy-end`` of the goal (the reference pelvis at the last frame) horizontally at the
end of the clip; for the get_up clips INSTEAD within ``--z-end`` vertically (the ACP text; ``--get-up-xy``: horizontally and vertically).

Per episode: survive = the clip end is reached without a termination (root height / tilt / non-finite state / the task's other terminations); success = survive and the horizontal pelvis
error to the reference is below ``--xy-end`` at the end (robust mode: and below ``--xy-max`` at all times: a robot that stalls in front of an obstacle survives but fails); for the get_up clips
also the vertical pelvis error at the end is below ``--z-end``; strict success = success and the largest height deviation of the nine contact links from the reference stays below
``--strict-eps``. Per clip and family: the rates, the median (over episodes) of the largest horizontal pelvis error, the agreement of the measured contact with the reference flags and
the position error of a contact link at the moment its measured contact starts (median / p90). Prints a table and writes a JSON.

    ACP_DATASET=<dir> python -m instinct_mj.scripts.acp.eval_tracker <checkpoint> [--task sm|xy|bm|c|ch5|<id>] [--mode paper|robust] [--num-envs 128] [--episodes 8] [--clip-regex .*] [--out eval.json]
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict

import torch

import instinct_mj.tasks.acp.config.g1.acp_tracking_cfg as acp_g1

dataset = os.path.expanduser(os.environ["ACP_DATASET"])
buffer = acp_g1.motion_reference_cfg.motion_buffers["TerrainMotion"]
buffer.path, buffer.metadata_yaml = dataset, os.path.join(dataset, "metadata.yaml")

import instinct_mj.tasks  # noqa: E402,F401  (registers the tasks)
from instinct_mj.rl import InstinctRlVecEnvWrapper  # noqa: E402
from instinct_mj.rl.acp.runner import clip_family  # noqa: E402
from instinct_mj.tasks.registry import load_env_cfg, load_env_cls, load_instinct_rl_cfg, load_runner_cls  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--task", default="Instinct-ACP-Track-G1-v1", help="task id (the terminations of the run the checkpoint comes from)")
    parser.add_argument("--mode", default="robust", choices=("robust", "paper"))
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--episodes", type=int, default=8, help="cohort size: envs (episodes) per clip")
    parser.add_argument("--max-steps", type=int, default=1500, help="control steps per batch of clips (a cohort member that is not finished by then counts as a failure)")
    parser.add_argument("--strict-eps", type=float, default=0.35)
    parser.add_argument("--xy-end", type=float, default=0.5, help="success: horizontal pelvis error at the episode end below this [m]")
    parser.add_argument("--xy-max", type=float, default=1.0, help="robust mode success: largest horizontal pelvis error of the episode below this [m]")
    parser.add_argument("--z-end", type=float, default=0.1, help="get_up clips: success also needs the pelvis height error at the episode end below this [m] (ACP: vertical error < 0.1 m)")
    parser.add_argument("--get-up-xy", action="store_true", help="paper mode: the get_up clips also need the horizontal error below --xy-end (ACP: instead the vertical error below --z-end only)")
    parser.add_argument("--noise", type=float, default=None, help="std of the Gaussian action noise (default: 0.1 in robust mode, 0 in paper mode)")
    parser.add_argument("--clip-regex", default=".*", help="only the clips whose name matches")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    paper = args.mode == "paper"
    device = "cuda:0"
    cfg = load_env_cfg(args.task)
    cfg.scene.num_envs = args.num_envs
    cfg.episode_length_s = 120.0  # the episodes last until the clip end
    if paper:
        cfg.disable_randomization()  # no physics / sensor / odometry randomization, no pose and joint reset noise
        for actuator in cfg.scene.entities["robot"].articulation.actuators:
            actuator.delay_max_lag = 0
        cfg.events["reset_robot"].params["randomize_velocity_range"] = {}
    env = load_env_cls(args.task)(cfg=cfg, device=device)
    vec = InstinctRlVecEnvWrapper(env, policy_group="policy", critic_group="critic")
    agent = load_instinct_rl_cfg(args.task).to_dict()
    _saved = os.path.join(os.path.dirname(os.path.abspath(args.checkpoint)), "params", "agent.yaml")
    if os.path.isfile(_saved):  # the runner options of the run the checkpoint comes from (world-model inputs, critic size, termination source)
        import yaml
        agent.update({k: v for k, v in yaml.safe_load(open(_saved)).items() if not isinstance(v, dict) and k in agent})
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
    families = [clip_family(e.get("group", "")) for e in mb.yaml_data["motion_files"]]
    num_clips, N, E = len(names), args.num_envs, args.episodes
    noise = args.noise if args.noise is not None else (0.0 if paper else 0.1)
    weights0 = mb._motion_weights.clone()
    env.cfg.auto_reset = False
    get_up = torch.tensor([f == "get_up" for f in families], device=device)
    per_batch = max(1, N // E)  # clips per batch: E envs each
    selected = [k for k, n in enumerate(names) if re.search(args.clip_regex, n)]
    survive, succ, strict, agree_n, agree_d, count, unfinished = (torch.zeros(num_clips, device=device) for _ in range(7))
    term_names = [n for n in env.termination_manager.active_terms if n not in ("time_out", "dataset_exhausted")]
    cause = torch.zeros(len(term_names) + 1, num_clips, device=device)  # episodes ended by every failure termination (last row: non-finite state); an episode can have several
    final_xy_episodes: list[list[float]] = [[] for _ in range(num_clips)]  # horizontal pelvis error at the end of every finished episode
    onsets: list[list[torch.Tensor]] = [[] for _ in range(num_clips)]
    max_xy_episodes: list[list[float]] = [[] for _ in range(num_clips)]  # per clip: the largest horizontal pelvis error of every finished episode

    def reset_cohort(env_ids: torch.Tensor, clip: int) -> None:
        mb._motion_weights[:] = 0.0
        mb._motion_weights[clip] = 1.0
        env.reset(env_ids=env_ids)

    for first in range(0, len(selected), per_batch):
        batch = selected[first : first + per_batch]
        env_clip = torch.full((N,), batch[0], device=device, dtype=torch.long)
        active = torch.zeros(N, dtype=torch.bool, device=device)  # the cohort members that have not finished their episode yet
        for i, k in enumerate(batch):
            ids = torch.arange(i * E, (i + 1) * E, device=device)
            env_clip[ids] = k
            active[ids] = True
            reset_cohort(ids, k)
        if len(batch) * E < N:
            reset_cohort(torch.arange(len(batch) * E, N, device=device), batch[0])  # the idle envs run along and are ignored
        obs, _ = vec.get_observations()
        maxdev, maxxy, prev_c = torch.zeros(N, device=device), torch.zeros(N, device=device), torch.zeros(N, 9, device=device)
        steps = 0
        while steps < args.max_steps and bool(active.any()):
            S0 = reader.state()
            a = policy(obs)
            a = torch.clamp(a + noise * torch.randn_like(a), -20, 20)
            obs, _, _, _ = vec.step(a)
            steps += 1
            S1 = reader.state()
            bad = ~torch.stack([torch.isfinite(x).flatten(1).all(-1) for x in S1]).all(0)
            S1 = tuple(torch.where(bad.view(-1, *[1] * (x0.dim() - 1)), x0, x1) for x0, x1 in zip(S0, S1))
            f1, phase1, origin1 = motion.current_frame()
            c = reader.contact()
            maxdev = torch.maximum(maxdev, mdp.link_deviation(S1, f1, origin1))
            pelvis_err = S1[0] - mdp.reference_state(f1, origin1)[0]
            xy = pelvis_err[:, :2].norm(dim=-1)  # horizontal pelvis error to the reference
            maxxy = torch.maximum(maxxy, xy)
            err = (mdp.kin.forward(*S1)[0] - (motion.link_p[f1] + origin1.unsqueeze(1))).norm(dim=-1)
            new = (c > 0.5) & (prev_c < 0.5) & active[:, None]
            live = active.float()
            agree_n.index_add_(0, env_clip, (c == motion.flag(f1, phase1).float()).float().sum(-1) * live)
            agree_d.index_add_(0, env_clip, 9.0 * live)
            for k in env_clip[new.any(-1)].unique().tolist():
                rows = (env_clip == k) & new.any(-1)
                onsets[k].append(err[rows][new[rows]].cpu())
            prev_c = c.clone()
            tm = env.termination_manager
            done = tm.dones | bad
            if done.any():
                ids = done.nonzero().squeeze(-1)
                first_ids = ids[active[ids]]  # only the first completion of an env counts
                if len(first_ids) > 0:
                    failed = env.reset_terminated[first_ids] | bad[first_ids]  # (the time-outs and the clip end are no failures)
                    reached = (tm.get_term("dataset_exhausted")[first_ids] | tm.get_term("time_out")[first_ids]) & ~failed
                    clip = env_clip[first_ids]
                    z_ok = pelvis_err[first_ids, 2].abs() < args.z_end
                    progress = (xy[first_ids] < args.xy_end) & (get_up[clip].logical_not() | z_ok)
                    if paper and not args.get_up_xy:
                        progress = torch.where(get_up[clip], z_ok, xy[first_ids] < args.xy_end)  # ACP: get-up / sit / stand are judged by the vertical error instead
                    if not paper:
                        progress &= maxxy[first_ids] < args.xy_max
                    count.index_add_(0, clip, torch.ones(len(first_ids), device=device))
                    survive.index_add_(0, clip, reached.float())
                    succ.index_add_(0, clip, (reached & progress).float())
                    strict.index_add_(0, clip, (reached & progress & (maxdev[first_ids] < args.strict_eps)).float())
                    for k, v, v_end in zip(clip.tolist(), maxxy[first_ids].tolist(), xy[first_ids].tolist()):
                        max_xy_episodes[k].append(v)
                        final_xy_episodes[k].append(v_end)
                    fired = torch.stack([tm.get_term(n)[first_ids] for n in term_names] + [bad[first_ids]]).float()
                    cause.index_add_(1, clip, fired)
                    active[first_ids] = False
                reset_cohort(ids, batch[0])  # (the finished envs run along, ignored)
                obs, _ = vec.get_observations()
                maxdev[ids], maxxy[ids], prev_c[ids] = 0.0, 0.0, 0.0
        if bool(active.any()):  # cohort members that did not finish within the step limit
            unfinished.index_add_(0, env_clip[active], torch.ones(int(active.sum()), device=device))
            count.index_add_(0, env_clip[active], torch.ones(int(active.sum()), device=device))
        print(f"batch {first // per_batch + 1} of {-(-len(selected) // per_batch)}: {steps} steps, unfinished {int(active.sum())}", flush=True)
    mb._motion_weights[:] = weights0
    rows, fam, fam_xy, fam_onset, fam_final_xy = [], defaultdict(lambda: defaultdict(float)), defaultdict(list), defaultdict(list), defaultdict(list)
    fam_cause = defaultdict(lambda: torch.zeros(len(term_names) + 1, device=device))
    for k, name in enumerate(names):
        if k not in selected:
            continue
        o = torch.cat(onsets[k]) if onsets[k] else torch.zeros(0)
        row = dict(clip=name, family=families[k], episodes=int(count[k]), unfinished=int(unfinished[k]), survive=float(survive[k] / count[k].clamp(min=1)), success=float(succ[k] / count[k].clamp(min=1)),
                   strict_success=float(strict[k] / count[k].clamp(min=1)), max_xy_median=float(torch.tensor(max_xy_episodes[k]).median()) if max_xy_episodes[k] else float("nan"),
                   label_agreement=float(agree_n[k] / agree_d[k].clamp(min=1)), onset_err_median=float(o.median()) if len(o) else float("nan"),
                   onset_err_p90=float(torch.quantile(o, 0.9)) if len(o) else float("nan"), onsets=len(o))
        rows.append(row)
        if row["episodes"] > 0:
            for key in ("survive", "success", "strict_success", "label_agreement"):
                fam[row["family"]][key] += row[key]
            fam[row["family"]]["clips"] += 1
            fam_xy[row["family"]] += max_xy_episodes[k]
            fam_final_xy[row["family"]] += final_xy_episodes[k]
            fam_onset[row["family"]].append(o)
            fam_cause[row["family"]] += cause[:, k]
    print(f"\nmode {args.mode}, checkpoint {args.checkpoint}")
    print(f"{'clip':56s} {'n':>3s} {'surv':>5s} {'succ':>5s} {'strict':>6s} {'xy_max':>6s} {'agree':>6s} {'onset med/p90 [m]':>18s}")
    for r in rows:
        print(f"{r['clip'][:56]:56s} {r['episodes']:3d} {r['survive']:5.2f} {r['success']:5.2f} {r['strict_success']:6.2f} {r['max_xy_median']:6.2f} {r['label_agreement']:6.3f} {r['onset_err_median']:9.3f}/{r['onset_err_p90']:.3f}")
    print(f"\n{'family':12s} {'clips':>5s} {'surv':>6s} {'succ':>6s} {'strict':>7s} {'agree':>6s} {'max pelvis xy error [m]: median p90 p99 (episodes)':>52s}")
    summary = {}
    for name, v in sorted(fam.items()):
        xy_q = torch.quantile(torch.tensor(fam_xy[name]), torch.tensor([0.5, 0.9, 0.99])).tolist()
        final_med = float(torch.tensor(fam_final_xy[name]).median())
        onset = torch.cat(fam_onset[name])
        onset_q = torch.quantile(onset, torch.tensor([0.5, 0.9])).tolist() if len(onset) else [float("nan")] * 2
        summary[name] = {key: v[key] / v["clips"] for key in ("survive", "success", "strict_success", "label_agreement")} | {"clips": int(v["clips"]), "max_xy_median_p90_p99": xy_q, "episodes": len(fam_xy[name]), "onset_err_median_p90": onset_q, "final_xy_median": final_med,
                                                                                                                       "ended_by": {n: float(c) / len(fam_xy[name]) for n, c in zip([*term_names, "nonfinite"], fam_cause[name])}}
        print(f"{name:12s} {int(v['clips']):5d} {summary[name]['survive']:6.3f} {summary[name]['success']:6.3f} {summary[name]['strict_success']:7.3f} {summary[name]['label_agreement']:6.3f} {xy_q[0]:20.2f} {xy_q[1]:.2f} {xy_q[2]:.2f} ({len(fam_xy[name])})  final xy median {final_med:.2f}  onset err med/p90 {onset_q[0]:.3f}/{onset_q[1]:.3f}")
        print(f"{'':12s} ended by: " + ", ".join(f"{n} {float(c) / len(fam_xy[name]):.2f}" for n, c in zip([*term_names, "nonfinite"], fam_cause[name]) if float(c) > 0))
    valid = [r for r in rows if r["episodes"] > 0]
    overall = {key: sum(r[key] for r in valid) / len(valid) for key in ("survive", "success", "strict_success", "label_agreement")}
    print(f"overall (mean over {len(valid)} clips, {int(count.sum())} episodes, {int(unfinished.sum())} unfinished): survive {overall['survive']:.3f} success {overall['success']:.3f} strict {overall['strict_success']:.3f} label agreement {overall['label_agreement']:.3f}")
    print(f"peak GPU memory allocated by torch: {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")
    if args.out:
        json.dump(dict(checkpoint=args.checkpoint, task=args.task, mode=args.mode, overall=overall, families=summary, clips=rows, max_xy_episodes={names[k]: v for k, v in enumerate(max_xy_episodes) if v}), open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
