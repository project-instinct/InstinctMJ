"""Parity of the analytic ACP MDP (``instinct_mj.rl.acp.mdp.ACPMDP``) against the live env.

Runs the ``Instinct-ACP-Track-G1-v0`` env (observation noise off, uniform start phases, domain randomization off unless ``--dr``) with a
slightly noisy zero policy and compares, per step, on the REAL states: the imagined-format ``policy`` observation (noise-free) vs the
env's, every Table 3 reward term (with the measured contacts instead of the world model's probabilities) vs the env's reward manager, the
three terminations, and the goal-target resampling decision.

    python -m instinct_mj.scripts.acp.check_mdp_parity [--num-envs 32] [--steps 200] [--motion-folder DIR] [--dr]
"""

from __future__ import annotations

import argparse

import torch

import instinct_mj.tasks  # noqa: F401  (registers the tasks)
from instinct_mj.rl import InstinctRlVecEnvWrapper
from instinct_mj.rl.acp.mdp import ODO_HIST, POLICY_OBS_DIM
from instinct_mj.rl.supermimic.rotations import mm, so3_exp
from instinct_mj.rl.acp.sim import ACPSimReader, build_mdp
from instinct_mj.tasks.registry import load_env_cfg, load_env_cls

TASK = "Instinct-ACP-Track-G1-v0"
OBS_SLICES = {
    "base_ang_vel": (0, 15), "projected_gravity": (15, 30), "joint_pos": (30, 145), "joint_vel": (145, 260), "last_action": (260, 375),
    "tracker_height_scan": (375, 438), "contact_targets": (438, 465), "goal_command": (465, 468),
}  # fmt: skip
TERMS = ("base_pos_too_far", "base_pg_too_far", "link_pos_too_far", "base_xy_too_far", "contact_far_from_ref")


def main() -> None:
    global TERMS
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default=TASK)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--action-noise", type=float, default=0.1)
    parser.add_argument("--motion-folder", type=str, default=None)
    parser.add_argument("--dr", action="store_true", help="keep the domain randomization and the action delay (observation noise stays off)")
    args = parser.parse_args()

    if args.motion_folder is not None:
        import os

        import instinct_mj.tasks.acp.config.g1.acp_tracking_cfg as acp_g1

        buf = acp_g1.motion_reference_cfg.motion_buffers["TerrainMotion"]
        buf.path = args.motion_folder
        buf.metadata_yaml = os.path.join(args.motion_folder, "metadata.yaml")
    device = "cuda:0"
    cfg = load_env_cfg(args.task)
    cfg.scene.num_envs = args.num_envs
    if args.dr:
        for group in cfg.observations.values():
            group.enable_corruption = False
        cfg.observations["policy"].terms["contact_targets"].params["noise"] = 0.0
        cfg.observations["planner"].terms["planner_height_scan"].params["z_noise_std"] = 0.0
    else:
        cfg.disable_randomization()
        for actuator in cfg.scene.entities["robot"].articulation.actuators:
            actuator.delay_max_lag = 0
    cfg.curriculum["curriculum_state_init"].params["mode"] = "uniform"
    env = load_env_cls(args.task)(cfg=cfg, device=device)
    vec = InstinctRlVecEnvWrapper(env, policy_group="policy", critic_group="critic")
    unwrapped = vec.unwrapped
    N = args.num_envs
    obs, _ = vec.reset()
    unwrapped.cfg.auto_reset = False
    motion, kin, scanner, mdp = build_mdp(unwrapped, 0.05, "~/.cache/instinct_mj/supermimic_heightmaps", device, exact_scan=True)
    reader = ACPSimReader(unwrapped)
    static = reader.static_context(mdp)
    from instinct_mj.rl.acp.mdp import make_xi

    rm, tm = unwrapped.reward_manager, unwrapped.termination_manager
    w = mdp.w

    hist = None
    pose_hist = None  # true pelvis poses, newest first, as the odometry sensor keeps them
    lp_prev = None
    obs_err = {k: [] for k in OBS_SLICES}  # (max abs error per env) rows, split by episode age
    obs_err_young = {k: [] for k in OBS_SLICES}
    rew_gap = {k: [] for k in w}
    rew_scale = {k: [] for k in w}
    rew_gap_stale = {k: [] for k in w}
    TERMS = tuple(n for n in TERMS if n in tm.active_terms)  # the NoLinkTerm / SM tasks have no link_pos_too_far
    marg_stale_mismatch = {k: 0 for k in TERMS}
    term_mismatch = {k: 0 for k in TERMS}
    term_count = {k: 0 for k in TERMS}
    goal_mismatch, goal_count, goal_resamples = 0, 0, 0
    f_mismatch = 0
    scan_miss_height = unwrapped.scene["height_scanner"].cfg.max_distance - unwrapped.scene["height_scanner"].cfg.ray_offset - 0.5
    scan_miss_cells = scan_cells = scan_miss_rows = scan_edge_cells = 0
    for step in range(args.steps):
        S = reader.state()
        f, phase, origin = motion.current_frame()
        a_prev = reader.last_action()
        a_prev2 = unwrapped.action_manager.prev_action.clone()
        ep = unwrapped.episode_length_buf.clone()
        xi = make_xi(static, reader.dynamic_context())
        goal = reader.goal(motion.framerate)
        fk = kin.forward(*S)
        base_p, base_R, _, base_w = mdp.base_of(fk)
        # the contacts term's frame index must be the clip-local frame of ours
        env_frame = unwrapped.command_manager.get_term("acp_reference_contacts").frame_index
        f_mismatch += int((env_frame != motion.table_index(f, phase)[1]).sum())
        o_t = mdp.proprio(base_R, base_w, S[4], S[5], a_prev, xi)
        if hist is None:
            hist = o_t.unsqueeze(1).expand(-1, 5, -1).clone()
        fresh = (ep == 0).view(-1, 1, 1)
        hist = torch.where(fresh, o_t.unsqueeze(1).expand(-1, 5, -1), torch.cat((hist[:, 1:], o_t.unsqueeze(1)), 1))
        # actor-side pose: the estimated pelvis pose (odometry delay and drift; zero without DR)
        new_hist = mdp.pose_history(base_p.unsqueeze(1).expand(-1, ODO_HIST, -1), base_R.unsqueeze(1).expand(-1, ODO_HIST, -1, -1))
        pose_hist = new_hist if pose_hist is None else tuple(torch.where(fresh.view(-1, 1, 1), n, o) for n, o in zip(new_hist, mdp.push_pose(pose_hist, base_p, base_R)))
        drift, delay = reader.odometry()
        est_p, est_yaw = mdp.estimated_pose(pose_hist, delay, drift)
        est_R = mdp.yaw_rotation(est_yaw)
        scan = mdp.scan(est_p, est_R)
        tgt, valid = mdp.targets(est_p, est_R, f, phase, origin)
        mine = mdp.policy_obs(hist, mdp.tracker_scan(scan), tgt.flatten(1), mdp.goal(est_p, est_R, goal[:, :3], goal[:, 3]))
        assert mine.shape[-1] == POLICY_OBS_DIM == obs.shape[-1], (mine.shape, obs.shape)
        err = (mine - obs).abs()
        # the env's ray cast returns isolated spurious misses (miss height) on mesh seams: count them, compare the other cells
        a, b = OBS_SLICES["tracker_height_scan"]
        miss = (obs[:, a:b] > scan_miss_height - 0.01) & (mine[:, a:b] < scan_miss_height - 0.01)
        scan_miss_cells += int(miss.sum())
        scan_cells += miss.numel()
        scan_miss_rows += int(miss.any(-1).sum())
        err[:, a:b] = torch.where(miss, torch.zeros_like(err[:, a:b]), err[:, a:b])
        scan_edge_cells += int((err[:, a:b] > 0.05).sum())
        for k, (a, b) in OBS_SLICES.items():
            e = err[:, a:b].amax(-1)
            obs_err[k].append(e[ep >= 4])
            obs_err_young[k].append(e[ep < 4])
        lp0 = fk[2 if mdp.contact_acc_vel else 0]  # positions, link velocities with contact_acc vel1
        lp_hist = torch.stack((lp0 if lp_prev is None else lp_prev, lp0), 0)
        lp_prev = lp0

        a = args.action_noise * torch.randn(N, 23, device=device)
        obs1, rew, dones, _ = vec.step(a)
        a_env = unwrapped.action_manager.action.clone()
        S1 = reader.state()
        contact = reader.contact()
        f1, phase1, origin1 = motion.current_frame()
        args_r = (S1, S[5], f1, phase1, origin1, a_env, a_prev, a_prev2, xi, contact, lp_hist, ep, 1)
        _, t_mine = mdp.reward(*args_r, terms=True)
        # the env evaluates rewards / terminations on derived quantities (xpos, xvel) one physics substep stale: step the state back by one
        dt_p = unwrapped.physics_dt
        p1, R1, v1, w1, q1, qd1 = S1
        S1_stale = (p1 - dt_p * v1, mm(so3_exp(-dt_p * w1), R1), v1, w1, q1 - dt_p * qd1, qd1)
        t_stale = mdp.reward(S1_stale, S[5], f1, phase1, origin1, a_env, a_prev, a_prev2, xi, contact, lp_hist, ep, 1, terms=True)[1]
        planned_mine = mdp.reward(*args_r, eta_rew=torch.ones((), device=device), terms=True)[1]["track_planned_contact"] / 1.5
        for n in w:
            if n == "track_planned_contact":
                c = rm.get_term_cfg(n)
                env_v = c.func(unwrapped, **c.params)  # weight 0 in Stage 1: evaluated directly
                mine_v = planned_mine
            else:
                env_v = rm._termwise_reward_buf["rewards"][n]
                mine_v = t_mine[n]
            rew_gap[n].append((mine_v - env_v).abs())
            rew_gap_stale[n].append(((t_stale[n] if n != 'track_planned_contact' else planned_mine) - env_v).abs())
            rew_scale[n].append(env_v.abs())
        marg = mdp.termination(S1, f1, origin1, phase1)
        marg_s = mdp.termination(S1_stale, f1, origin1, phase1)
        for n in TERMS:
            env_flag = tm.get_term(n)
            marg_stale_mismatch[n] += int(((marg_s[n] > 0) != env_flag).sum())
            term_mismatch[n] += int(((marg[n] > 0) != env_flag).sum())
            term_count[n] += int(env_flag.sum())
        # goal resampling decision (the new goal of an env is a random draw: compare only whether it was resampled)
        g1 = reader.goal(motion.framerate)
        local1 = (f1 - motion.clip_start[motion.clip_id[f1]]).float() + phase1
        passed_mine = goal[:, 4] - local1 < mdp.goal_lookahead[0] * motion.framerate
        changed = (g1[:, 4] != goal[:, 4]) | (g1[:, :3] != goal[:, :3]).any(-1)
        # a goal capped at the clip end is re-drawn to the same value every step (not a change)
        at_end = goal[:, 4] >= (motion.clip_len[motion.clip_id[f]] - 1).float() - 1e-3
        ok = ~dones.bool() & ~at_end
        goal_mismatch += int(((passed_mine != changed) & ok).sum())
        goal_count += int(ok.sum())
        goal_resamples += int((changed & ok).sum())
        obs = obs1
        done_ids = dones.bool().nonzero(as_tuple=False).squeeze(-1)
        if len(done_ids) > 0:
            unwrapped.reset(env_ids=done_ids)
            obs = torch.where(dones.bool()[:, None], vec.get_observations()[0], obs1)

    cat = lambda xs: torch.cat(xs) if len(xs) else torch.zeros(1)  # noqa: E731
    print(f"\n== parity: {args.num_envs} envs, {args.steps} steps, dr={args.dr} ==")
    print("frame index mismatches (contacts term vs MotionLib):", f_mismatch)
    print(f"\n{'obs term':22s} {'max|err| (age>=4)':>20s} {'mean|err|':>12s} {'max|err| (age<4)':>20s}")
    for k in OBS_SLICES:
        e, y = cat(obs_err[k]), cat(obs_err_young[k])
        print(f"{k:22s} {float(e.max()):20.2e} {float(e.mean()):12.2e} {float(y.max()) if len(y) else float('nan'):20.2e}")
    print(f"tracker_height_scan: env ray-cast misses (spurious, excluded above): {scan_miss_cells} of {scan_cells} cells "
          f"({100 * scan_miss_cells / scan_cells:.2f} %), in {scan_miss_rows} of {args.steps * N} env-steps; cells with |err| > 0.05 "
          f"(heightmap bilinear blur at step edges): {scan_edge_cells} ({100 * scan_edge_cells / scan_cells:.2f} %)")
    print(f"\n{'reward term':26s} {'weight':>8s} {'mean|env|':>11s} {'mean|gap|':>11s} {'max|gap|':>11s} {'gap/mean|env|':>14s} {'rel. gap (stale-corrected)':>27s}")
    for n in w:
        g, s = cat(rew_gap[n]), cat(rew_scale[n])
        print(f"{n:26s} {float(w[n]):8.3g} {float(s.mean()):11.3e} {float(g.mean()):11.3e} {float(g.max()):11.3e} {float(g.mean() / s.mean().clamp(min=1e-12)):14.3e} {float(cat(rew_gap_stale[n]).mean() / s.mean().clamp(min=1e-12)):27.3e}")
    print(f"\n{'termination':20s} {'env true':>9s} {'mismatches':>11s} {'(stale-corrected)':>18s}")
    for n in TERMS:
        print(f"{n:20s} {term_count[n]:9d} {term_mismatch[n]:11d} {marg_stale_mismatch[n]:18d}")
    print(f"goal resample decision: {goal_mismatch} mismatches of {goal_count} steps ({goal_resamples} resamples)")


if __name__ == "__main__":
    main()
