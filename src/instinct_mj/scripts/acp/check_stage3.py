"""Checks of the ACP Stage 3 task (``Instinct-ACP-Stage3-G1-v0``) on the live env (small: 64 envs by default).

1. env layout: reference / goal-reaching envs, the spawn of every goal env on its tile (surface height from the boxes), the goal on the surface,
2. the two-region analytic heightmap (``RegionHeightScanner``) against mujoco ray casts on the arena and on the reference terrains,
3. goal reward terms: the env's reward manager vs the analytic ``ACPStage3MDP.reward_mixed`` (indicators exact) on real states, including
   envs teleported next to their goal so that the gates of ``move2goal`` / ``standatgoal`` switch on; soft (sigmoid) vs hard gates,
4. terminations: the env's goal fall / time-out vs the analytic hard termination,
5. terrain curriculum: EMA, promotion (success and s > 0.5), demotion (> 1.0 m), clipping.

    python -m instinct_mj.scripts.acp.check_stage3 [--num-envs 64] [--steps 200]
"""

from __future__ import annotations

import argparse
import math

import mujoco
import numpy as np
import torch

import instinct_mj.tasks  # noqa: F401  (registers the tasks)
from instinct_mj.rl import InstinctRlVecEnvWrapper
from instinct_mj.rl.acp.mdp import make_xi
from instinct_mj.rl.acp.sim import ACPSimReader
from instinct_mj.rl.acp.stage3_config import ACPStage3Cfg
from instinct_mj.rl.supermimic.rotations import mm, so3_exp
from instinct_mj.rl.acp.stage3_mdp import GOAL_TERMS, REF_ONLY_TERMS, build_stage3_mdp
from instinct_mj.tasks.acp.goal_terrains import BASE_STAND_HEIGHT, NUM_LEVELS, TERRAIN_TYPES, TILE_SIZE, surface_height
from instinct_mj.tasks.acp.mdp import goal as goal_mdp
from instinct_mj.tasks.registry import load_env_cfg, load_env_cls

TASK = "Instinct-ACP-Stage3-G1-v0"


def check_layout(env, failures: list) -> None:
    gt = env.command_manager.get_term("acp_goal")
    arena = env.cfg.scene.arena
    goal = gt.is_goal
    robot = env.scene["robot"]
    print(f"\n== layout: {int(goal.sum())} goal envs of {env.num_envs}; types {gt.terrain_type[goal].unique().tolist()} ==")
    pos = robot.data.root_link_pos_w
    for e in goal.nonzero().flatten().tolist():
        t, level = int(gt.terrain_type[e]), int(gt.level[e])
        tile = arena.tiles[t][level]
        c = arena.tile_center(t, level)
        local = lambda p: np.asarray(p.cpu() if torch.is_tensor(p) else p, dtype=np.float64) - c  # noqa: E731
        surf = surface_height(tile.boxes, local(pos[e])[None, :2])[0]
        lying = bool(gt.tables.lying[gt.tile_id[e]])
        z_over = float(pos[e, 2]) - surf
        lo, hi = (0.2, 0.40) if lying else (BASE_STAND_HEIGHT - 0.01, BASE_STAND_HEIGHT + 0.06)
        # the reset height is relative to the patch surface: the surface under the pelvis may be higher (uneven tops)
        if not lo <= z_over <= hi + 0.15 and TERRAIN_TYPES[t] not in ("uneven_climb",):
            failures.append(f"env {e} ({TERRAIN_TYPES[t]}[{level}]): pelvis {z_over:.3f} m over the surface at the reset")
        g = gt.target_pos_w[e]
        g_surf = surface_height(tile.boxes, local(g)[None, :2])[0]
        if abs(float(g[2]) - BASE_STAND_HEIGHT - g_surf) > 0.5 * tile.params.get("roughness", 0.0) + 0.02:
            failures.append(f"env {e} ({TERRAIN_TYPES[t]}[{level}]): goal z {float(g[2]):.3f} vs surface {g_surf:.3f} + {BASE_STAND_HEIGHT}")
        d = (g[:2].cpu().numpy() - c[:2])
        if abs(d[0]) > TILE_SIZE[0] / 2 or abs(d[1]) > TILE_SIZE[1] / 2:
            failures.append(f"env {e}: goal outside its tile")
    print("spawn and goal on the surface of the tile:", "OK" if not failures else failures[-3:])


def check_scanner(env, scanner, failures: list, n: int = 4000) -> None:
    model = env.sim.mj_model
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    group = np.zeros(6, dtype=np.uint8)
    group[0] = 1
    arena = env.cfg.scene.arena
    rng = np.random.default_rng(0)
    print(f"\n== analytic heightmap vs ray casts ({n} points per region) ==")
    geomid = np.zeros(1, dtype=np.int32)

    def cast(x, y):
        z0 = 20.0
        d = mujoco.mj_ray(model, data, np.array([x, y, z0]), np.array([0.0, 0.0, -1.0]), group, 1, -1, geomid)
        return z0 - d if d >= 0 else -100.0

    for name, pts in (("arena", None), ("reference terrains", "ref")):
        errs, types = [], []
        for _ in range(n):
            if pts is None:
                i, level = rng.integers(arena.num_types), rng.integers(NUM_LEVELS)
                c = arena.tile_center(i, level)
                x, y = c[0] + rng.uniform(-TILE_SIZE[0] / 2, TILE_SIZE[0] / 2), c[1] + rng.uniform(-TILE_SIZE[1] / 2, TILE_SIZE[1] / 2)
                types.append(i)
            else:
                tile = rng.integers(29)
                x, y = -420.0 + 30.0 * tile + rng.uniform(-6, 6), rng.uniform(-6, 6)
            truth = cast(x, y)
            est = float(scanner.ground_height(torch.tensor([[[x, y]]], dtype=torch.float32, device=scanner.heights.device)))
            errs.append(abs(est - truth) if truth > -50 else 0.0)
        errs = np.array(errs)
        print(f"{name:20s} median |err| {np.median(errs):.4f} m   95 % {np.percentile(errs, 95):.4f} m   > 5 cm: {100 * (errs > 0.05).mean():.2f} %")
        if pts is None:
            types = np.array(types)
            worst = sorted(((float((errs[types == i] > 0.05).mean()), TERRAIN_TYPES[i]) for i in range(arena.num_types)), reverse=True)[:5]
            print("  share of points > 5 cm off (edges of boxes, thin rungs) by type:", ", ".join(f"{n_}: {100 * s:.1f} %" for s, n_ in worst))
        if np.percentile(errs, 95) > 0.1:
            failures.append(f"heightmap of the {name}: 95 % error {np.percentile(errs, 95):.3f} m")


def check_exact_scan(env, failures: list, device: str, n: int = 128) -> None:
    """The exact ray-cast scanner (``exact_scan``) on the arena vs mujoco ray casts, and its straight-through gradient."""
    print("\n== exact ray-cast height scan on the arena ==")
    cfg = ACPStage3Cfg().to_dict()
    _, _, exact, _ = build_stage3_mdp(env, 0.05, "~/.cache/instinct_mj/supermimic_heightmaps", device, cfg, exact_scan=True)
    arena = env.cfg.scene.arena
    model = env.sim.mj_model
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    group = np.zeros(6, dtype=np.uint8)
    group[0] = 1
    rng = np.random.default_rng(1)
    centers = []
    for _ in range(n):
        i, level = rng.integers(arena.num_types), rng.integers(NUM_LEVELS)
        c = arena.tile_center(i, level)
        centers.append([c[0] + rng.uniform(-4, 4), c[1] + rng.uniform(-1.5, 1.5), 1.0, rng.uniform(-math.pi, math.pi)])
    centers = np.array(centers)
    p = torch.tensor(centers[:, :3], dtype=torch.float32, device=device, requires_grad=True)
    yaw = torch.tensor(centers[:, 3], dtype=torch.float32, device=device)
    c_, s_ = torch.cos(yaw), torch.sin(yaw)
    R = torch.zeros(n, 3, 3, device=device)
    R[:, 0, 0], R[:, 0, 1], R[:, 1, 0], R[:, 1, 1], R[:, 2, 2] = c_, -s_, s_, c_, 1.0
    scan = exact.height_scan(p, R)
    offs = exact.offsets.cpu().numpy()
    bad = total = 0
    geomid = np.zeros(1, dtype=np.int32)
    for b in range(n):
        for k in range(offs.shape[0]):
            x = centers[b, 0] + math.cos(centers[b, 3]) * offs[k, 0] - math.sin(centers[b, 3]) * offs[k, 1]
            y = centers[b, 1] + math.sin(centers[b, 3]) * offs[k, 0] + math.cos(centers[b, 3]) * offs[k, 1]
            d = mujoco.mj_ray(model, data, np.array([x, y, centers[b, 2] + 5.0]), np.array([0.0, 0.0, -1.0]), group, 1, -1, geomid)
            if d < 0:
                continue
            total += 1
            bad += abs(float(scan[b, k]) - (d - 5.0)) > 2e-3
    scan.sum().backward()
    ok_grad = bool(torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0)
    print(f"exact scan vs mj_ray: {bad} of {total} rays differ by > 2 mm; gradient w.r.t. the pelvis position finite and non-zero: {ok_grad}")
    if bad > 0.002 * total or not ok_grad:
        failures.append(f"exact scan: {bad} of {total} rays differ, gradient {ok_grad}")


def check_rewards(env, vec, mdp, args, failures: list) -> None:
    unwrapped = env
    device = env.device
    N = env.num_envs
    gt = unwrapped.command_manager.get_term("acp_goal")
    reader = ACPSimReader(unwrapped)
    static = reader.static_context(mdp)
    rm, tm = unwrapped.reward_manager, unwrapped.termination_manager
    robot = unwrapped.scene["robot"]
    ids = unwrapped.action_manager.get_term("joint_pos").target_ids
    goal_envs = gt.is_goal.nonzero().flatten()
    teleport = goal_envs[: len(goal_envs) // 2]  # these goal envs are put next to their goal, standing
    names = [n for n in rm.active_terms["rewards"]]
    gap = {n: [] for n in names}
    gap_stale = {n: [] for n in names}
    scale = {n: [] for n in names}
    since_tp = torch.full((N,), 99, device=device)  # control steps since the env was teleported (finite differences are polluted)
    gap_soft_hard = []
    reached = 0
    lp_prev, contact_prev = None, torch.zeros(N, 9, device=device)
    obs, _ = vec.reset()
    unwrapped.cfg.auto_reset = False
    fall_mismatch, fall_count, fall_debug = 0, 0, []
    for step in range(args.steps):
        if step % 40 == 0:  # teleport the chosen goal envs to the goal (default pose), heading at the goal heading
            z = gt.target_pos_w[teleport, 2:3]
            pose = torch.cat((gt.target_pos_w[teleport, :2] + 0.2 * torch.randn(len(teleport), 2, device=device), z + 0.01 * torch.randn(len(teleport), 1, device=device)), -1)
            yaw = gt.target_yaw[teleport]
            quat = torch.stack((torch.cos(yaw / 2), torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(yaw / 2)), -1)
            robot.write_root_link_pose_to_sim(torch.cat((pose, quat), -1), env_ids=teleport)
            robot.write_root_link_velocity_to_sim(torch.zeros(len(teleport), 6, device=device), env_ids=teleport)
            robot.write_joint_state_to_sim(robot.data.default_joint_pos[teleport], torch.zeros_like(robot.data.default_joint_pos[teleport]), env_ids=teleport)
            unwrapped.sim.forward()
            since_tp[teleport] = 0
        S = reader.state()
        f, phase, origin = mdp.motion.current_frame()
        a_prev, a_prev2 = reader.last_action(), unwrapped.action_manager.prev_action.clone()
        ep = unwrapped.episode_length_buf.clone()
        xi = make_xi(static, reader.dynamic_context())
        lp0 = mdp.kin.forward(*S)[0]
        lp_hist = torch.stack((lp0 if lp_prev is None else lp_prev, lp0), 0)
        lp_prev = lp0
        a = args.action_noise * torch.randn(N, 23, device=device)
        obs1, rew, dones, _ = vec.step(a)
        a_env = unwrapped.action_manager.action.clone()
        S1 = reader.state()
        contact = reader.contact()
        goal1 = reader.goal(mdp.motion.framerate)
        f1, phase1, origin1 = mdp.motion.current_frame()
        tile = gt.tile_id.clone()
        stumble = goal_mdp.stumbling(unwrapped)
        _, t_hard = mdp.reward_mixed(S1, S[5], f1, phase1, origin1, a_env, a_prev, a_prev2, xi, contact, contact_prev, lp_hist, ep, 1, gt.is_goal,
                                     goal1[:, :3], goal1[:, 3], tile, hard=True, stumbling=stumble)  # fmt: skip
        # the env evaluates rewards on derived quantities (xpos, xvel) one physics substep stale: step the state back by one
        dt_p = unwrapped.physics_dt
        p1, R1, v1, w1, q1, qd1 = S1
        S1_stale = (p1 - dt_p * v1, mm(so3_exp(-dt_p * w1), R1), v1, w1, q1 - dt_p * qd1, qd1)
        _, t_stale = mdp.reward_mixed(S1_stale, S[5], f1, phase1, origin1, a_env, a_prev, a_prev2, xi, contact, contact_prev, lp_hist, ep, 1, gt.is_goal,
                                      goal1[:, :3], goal1[:, 3], tile, hard=True, stumbling=stumble)  # fmt: skip
        valid = since_tp >= 3
        _, t_soft = mdp.reward_mixed(S1, S[5], f1, phase1, origin1, a_env, a_prev, a_prev2, xi, contact, contact_prev, lp_hist, ep, 1, gt.is_goal,
                                     goal1[:, :3], goal1[:, 3], tile, hard=False, stumbling=stumble)  # fmt: skip
        for n in names:
            env_v = rm._termwise_reward_buf["rewards"][n]
            gap[n].append(((t_hard[n] - env_v).abs())[valid]), gap_stale[n].append(((t_stale[n] - env_v).abs())[valid]), scale[n].append(env_v.abs()[valid])
        gap_soft_hard.append(torch.stack([(t_soft[n] - t_hard[n]).abs() for n in GOAL_TERMS]).sum(0))
        reached += int(((gt.target_pos_w[:, :2] - robot.data.root_link_pos_w[:, :2]).norm(dim=-1) < 0.5)[gt.is_goal].sum())
        fall_env = tm.get_term("goal_fall")
        fall_mine = mdp.goal_termination(S1, tile, hard=True) > 0.5
        fall_stale = mdp.goal_termination(S1_stale, tile, hard=True) > 0.5
        bad = (fall_env != fall_stale) & gt.is_goal
        if bad.any() and len(fall_debug) < 6:
            for e in bad.nonzero().flatten().tolist()[:2]:
                fall_debug.append(f"env {e} {TERRAIN_TYPES[int(gt.terrain_type[e])]}[{int(gt.level[e])}] z {float(S1[0][e, 2]):.3f} env {bool(fall_env[e])} mine {bool(fall_mine[e])} stale {bool(fall_stale[e])}")
        fall_mismatch += int(bad.sum())
        fall_count += int(fall_env.sum())
        since_tp += 1
        contact_prev = torch.where(dones.bool()[:, None], torch.zeros_like(contact), contact)
        done_ids = dones.bool().nonzero(as_tuple=False).squeeze(-1)
        if len(done_ids) > 0:
            unwrapped.reset(env_ids=done_ids)
    cat = lambda xs: torch.cat(xs)  # noqa: E731
    print(f"\n== rewards (hard gates) vs the env, {args.steps} steps; goal-env samples within 0.5 m of the goal: {reached} ==")
    print(f"{'term':26s} {'weight':>9s} {'mean|env|':>11s} {'mean|gap|':>11s} {'max|gap|':>11s} {'gap/mean|env|':>14s} {'stale-corrected':>16s}")
    for n in names:
        g, gs, s = cat(gap[n]), cat(gap_stale[n]), cat(scale[n])
        rel, rel_s = float(g.mean() / s.mean().clamp(min=1e-12)), float(gs.mean() / s.mean().clamp(min=1e-12))
        w = rm.get_term_cfg(n).weight
        print(f"{n:26s} {w:9.3g} {float(s.mean()):11.3e} {float(g.mean()):11.3e} {float(g.max()):11.3e} {rel:14.3e} {rel_s:16.3e}")
        # tolerance: the env reads velocities one physics substep stale; the indicator terms flip at their thresholds (e.g. feet moving at ~0.5 m/s)
        tol = 0.25 if n == "slippage" else 0.15 if n in ("ang_vel_xy", "track_base_vel", "joint_torque", "contact_acc") else 0.05  # slippage: rare indicator flips (11 - 19 % between runs)
        if n in (*GOAL_TERMS, *REF_ONLY_TERMS, "contact_acc", "joint_torque") and min(rel, rel_s) > tol and float(s.mean()) > 1e-6:
            failures.append(f"reward term {n}: analytic vs env relative gap {min(rel, rel_s):.3f}")
    print(f"soft vs hard gates (tau = {mdp.tau}): mean |sum of goal terms diff| {float(cat(gap_soft_hard).mean()):.4f}")
    print(f"goal fall: {fall_mismatch} mismatches of {fall_count} env terminations (analytic hard, stale-corrected, vs env)")
    for line in fall_debug:
        print("   ", line)
    saved = dict(mdp.tau)
    mdp.tau = {k: 1e-6 for k in saved}  # sigmoid gates -> indicators
    S_ = reader.state()
    kw = (S_[5], f1, phase1, origin1, a_env, a_prev, a_prev2, xi, contact, contact_prev, lp_hist, ep, 1, gt.is_goal, goal1[:, :3], goal1[:, 3], tile)
    r_soft = mdp.reward_mixed(S_, *kw, hard=False, stumbling=stumble)[0]
    r_hard = mdp.reward_mixed(S_, *kw, hard=True, stumbling=stumble)[0]
    mdp.tau = saved
    print(f"soft gates with tau -> 0 vs indicators: max |diff| {float((r_soft - r_hard).abs().max()):.2e}")
    if float((r_soft - r_hard).abs().max()) > 1e-3:
        failures.append("soft gates do not reduce to the indicators for tau -> 0")
    if fall_mismatch > 0.05 * max(fall_count, 1) + 2:
        failures.append(f"goal fall mismatches: {fall_mismatch} of {fall_count}")


def check_formulas(mdp, device, failures: list) -> None:
    """Direct check of the gate logic of the goal terms on constructed states (standing robot with set velocities)."""
    print("\n== formulas on constructed states (hard gates) ==")
    gt_tables = mdp.tables
    ids = mdp.act_ids
    q0 = mdp.default_q.clone()
    n = 8
    p = torch.zeros(n, 3, device=device)
    R = torch.eye(3, device=device).expand(n, 3, 3).clone()
    v = torch.zeros(n, 3, device=device)
    w = torch.zeros(n, 3, device=device)
    q = q0.expand(n, -1).clone()
    qd = torch.zeros_like(q)
    p[:, 2] = 0.78
    # goal 3 m ahead (x), v toward goal 0.5 m/s in row 0, away in row 1, too fast in row 2 (stairs cap 0.8), in the goal circle in row 3
    goal_pos = torch.tensor([3.0, 0.0, 0.78], device=device).expand(n, 3).clone()
    goal_yaw = torch.zeros(n, device=device)
    v[0, 0], v[1, 0], v[2, 0] = 0.5, -0.5, 1.5
    p[3, 0] = 2.8
    p[4, 0], p[4, 2] = 2.8, 0.5  # in the circle but 0.28 m too low: no stand reward
    p[5, 0], goal_yaw[5] = 2.8, 1.0  # heading error 1 rad: no stand reward
    contact = torch.ones(n, 9, device=device)
    tile = torch.full((n,), TERRAIN_TYPES.index("platform") * NUM_LEVELS, dtype=torch.long, device=device)
    tile[2] = TERRAIN_TYPES.index("stairs_up") * NUM_LEVELS  # speed cap 0.8 m/s
    S = (p, R, v, w, q, qd)
    t = {k: x / mdp.w[k] for k, x in mdp.goal_terms(S, contact, contact, goal_pos, goal_yaw, tile, hard=True).items()}
    exp_move = torch.tensor([1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0], device=device)  # (1 - g_xy) g_v + g_xy g_z, g_z = 1 off the get-up terrains
    ok_move = torch.allclose(t["move2goal"], exp_move)
    # standatgoal: rows 3 standing in the circle at the right height and heading, upright, both feet in contact, default pose: 1
    stand = t["standatgoal"]
    ok_stand = torch.allclose(stand[3], torch.tensor(1.0, device=device), atol=1e-3) and stand[4] == 0 and stand[5] == 0 and (stand[[0, 1, 2]] == 0).all()
    # feet without contact: d_foot = 1 -> exp(-1/3)
    S2 = (p[3:4], R[3:4], v[3:4], w[3:4], q[3:4], qd[3:4])
    stand_nc = mdp.goal_terms(S2, torch.zeros(1, 9, device=device), torch.zeros(1, 9, device=device), goal_pos[3:4], goal_yaw[3:4], tile[3:4], hard=True)["standatgoal"] / mdp.w["standatgoal"]
    ok_nc = abs(float(stand_nc) - math.exp(-1 / 3)) < 1e-3
    print(f"move2goal gates: {'OK' if ok_move else 'FAIL ' + str(t['move2goal'].tolist())}; standatgoal gates/kernel: {'OK' if ok_stand and ok_nc else 'FAIL'} "
          f"({stand.tolist()}, no-contact {float(stand_nc):.4f} vs {math.exp(-1 / 3):.4f})")
    if not (ok_move and ok_stand and ok_nc):
        failures.append("goal reward formulas on constructed states")
    # get-up g_z: below the height tolerance without vertical speed -> 0 away from the goal ring, with rise speed -> 1
    getup = torch.full((n,), TERRAIN_TYPES.index("getup") * NUM_LEVELS, dtype=torch.long, device=device)
    p2, v2 = p.clone(), v.clone()
    p2[:, :2], p2[:, 2] = 0.0, 0.3
    goal_g = torch.tensor([0.1, 0.0, 0.78], device=device).expand(n, 3).clone()  # inside the ring, 0.48 m below the target height
    v2[:, 2] = torch.tensor([0.0, 0.3, -0.3, 0.0, 0.0, 0.0, 0.0, 0.0], device=device)
    mg = mdp.goal_terms((p2, R, v2, w, q, qd), contact, contact, goal_g, goal_yaw, getup, hard=True)["move2goal"] / mdp.w["move2goal"]
    ok_getup = bool(mg[0] == 0 and mg[1] == 1 and mg[2] == 0)
    print(f"get-up g_z (rising towards the target height): {'OK' if ok_getup else 'FAIL ' + str(mg.tolist())}")
    if not ok_getup:
        failures.append("get-up g_z")


def check_curriculum(env, failures: list) -> None:
    print("\n== terrain curriculum ==")
    gt = env.command_manager.get_term("acp_goal")
    cur = env.curriculum_manager.get_term_cfg("goal_terrain").func
    goal = gt.is_goal.nonzero().flatten()[:6]
    gt.level[goal] = torch.tensor([0, 3, 3, 9, 5, 5], device=env.device)
    gt.success_ema[goal] = torch.tensor([0.6, 0.6, 0.0, 0.9, 0.0, 0.9], device=env.device)
    # env 0: success, s -> 0.8 * 0.6 + 0.2 = 0.68 > 0.5 -> level 1. env 1: success but early termination -> s = 0.48, min distance far (3 m) -> demoted to 2.
    # env 2: success, s -> 0.2 < 0.5 -> stays 3. env 3: success at the top level -> stays 9. env 4: min distance 2 m -> demoted to 4.
    # env 5: reached (0.4 m) but terminated early -> not a success: s = 0.72, distance <= 1 m -> unchanged.
    gt.min_xy_ok[goal] = torch.tensor([0.3, 3.0, 0.4, 0.1, 2.0, 0.4], device=env.device)
    terminated = torch.tensor([False, True, False, False, False, True], device=env.device)
    success = cur.update(goal, terminated)
    exp_level = [1, 2, 3, 9, 4, 5]
    exp_ema = [0.68, 0.48, 0.2, 0.92, 0.0, 0.72]
    ok = gt.level[goal].tolist() == exp_level and np.allclose(gt.success_ema[goal].cpu().numpy(), exp_ema, atol=1e-6)
    print(f"levels {gt.level[goal].tolist()} (expected {exp_level}), success {success.tolist()}, EMA {[round(x, 3) for x in gt.success_ema[goal].tolist()]}: {'OK' if ok else 'FAIL'}")
    if not ok:
        failures.append("terrain curriculum update")
    gt.level[goal] = torch.tensor([0, 0, 0, 0, 0, 0], device=env.device)
    gt.success_ema[goal] = 0.0
    gt.min_xy_ok[goal] = 5.0
    cur.update(goal, torch.zeros(6, dtype=torch.bool, device=env.device))
    ok0 = (gt.level[goal] == 0).all()
    print("demotion at level 0 clips at 0:", "OK" if ok0 else "FAIL")
    if not ok0:
        failures.append("curriculum clip at 0")
    gt.level[goal] = 0
    gt.success_ema[goal] = 0.0


def check_curriculum_wiring(env, vec, failures: list, steps: int = 820) -> None:
    """The curriculum through the env's reset (the manager call with the real episode statistics): goal envs of the even terrain types start
    at their goal with a zero action (standing in the default pose), the odd types start where they spawn; at every episode end the new level
    / EMA must follow the supplement's rule applied to the pre-reset ``min_xy_ok`` / termination."""
    print("\n== curriculum through the env resets (zero action, 15 s episodes) ==")
    device = env.device
    gt = env.command_manager.get_term("acp_goal")
    robot = env.scene["robot"]
    obs, _ = vec.reset()
    env.cfg.auto_reset = False
    goal = gt.is_goal
    gt.level[goal] = 3
    gt.success_ema[goal] = torch.where(gt.terrain_type[goal] % 4 == 0, 0.45, 0.0)  # 0.45 -> 0.56 after a success: promoted
    even = goal & (gt.terrain_type % 2 == 0)
    seen = {"promoted": 0, "demoted": 0, "kept": 0, "success": 0}
    bad = 0
    ids = even.nonzero().flatten()
    gt.sample_reset(ids)  # (the goal itself stays; this only re-draws it for the teleport below)
    def teleport(ids):
        pos = torch.cat((gt.target_pos_w[ids, :2], gt.target_pos_w[ids, 2:3]), -1)
        yaw = gt.target_yaw[ids]
        quat = torch.stack((torch.cos(yaw / 2), torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(yaw / 2)), -1)
        robot.write_root_link_pose_to_sim(torch.cat((pos, quat), -1), env_ids=ids)
        robot.write_root_link_velocity_to_sim(torch.zeros(len(ids), 6, device=device), env_ids=ids)
        robot.write_joint_state_to_sim(robot.data.default_joint_pos[ids], torch.zeros_like(robot.data.default_joint_pos[ids]), env_ids=ids)
        env.sim.forward()
    teleport(ids)
    for step in range(steps):
        obs, rew, dones, _ = vec.step(torch.zeros(env.num_envs, 23, device=device))
        done_all = dones.bool()
        if not done_all.any():
            continue
        d = (done_all & goal).nonzero().flatten()
        min_xy, term = gt.min_xy_ok[d].clone(), env.reset_terminated[d].clone()
        level0, ema0 = gt.level[d].clone(), gt.success_ema[d].clone()
        env.reset(env_ids=done_all.nonzero().flatten())
        if len(d) > 0:
            success = (min_xy <= 0.5) & ~term
            ema1 = 0.8 * ema0 + 0.2 * success.float()
            level1 = (level0 + (success & (ema1 > 0.5)).long() - (min_xy > 1.0).long()).clamp(0, NUM_LEVELS - 1)
            bad += int((gt.level[d] != level1).sum()) + int(((gt.success_ema[d] - ema1).abs() > 1e-5).sum())
            seen["success"] += int(success.sum())
            seen["promoted"] += int((level1 > level0).sum())
            seen["demoted"] += int((level1 < level0).sum())
            seen["kept"] += int((level1 == level0).sum())
        e2 = even & done_all
        if e2.any():
            teleport(e2.nonzero().flatten())
    print(f"episode ends of goal envs {sum(seen[k] for k in ('promoted', 'demoted', 'kept'))}: promoted {seen['promoted']}, demoted {seen['demoted']}, kept {seen['kept']}, "
          f"successes {seen['success']}; mismatches with the rule: {bad}")
    if bad or seen["promoted"] == 0 or seen["demoted"] == 0:
        failures.append(f"curriculum through env resets: {seen}, mismatches {bad}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--action-noise", type=float, default=0.05)
    args = parser.parse_args()
    device = "cuda:0"
    cfg = load_env_cfg(TASK)
    cfg.scene.num_envs = args.num_envs
    cfg.disable_randomization()
    for actuator in cfg.scene.entities["robot"].articulation.actuators:
        actuator.delay_max_lag = 0
    env = load_env_cls(TASK)(cfg=cfg, device=device)
    vec = InstinctRlVecEnvWrapper(env, policy_group="policy", critic_group="critic")
    obs, _ = vec.reset()
    failures: list[str] = []
    check_layout(env, failures)
    mdp_cfg = ACPStage3Cfg().to_dict()
    motion, kin, scanner, mdp = build_stage3_mdp(env, 0.05, "~/.cache/instinct_mj/supermimic_heightmaps", device, mdp_cfg)
    check_scanner(env, scanner, failures)
    check_exact_scan(env, failures, device)
    check_formulas(mdp, device, failures)
    check_rewards(env, vec, mdp, args, failures)
    check_curriculum(env, failures)
    check_curriculum_wiring(env, vec, failures)
    if failures:
        print("\nFAILURES:")
        for f in failures:
            print("  ", f)
        raise SystemExit(1)
    print("\nOK")


if __name__ == "__main__":
    main()
