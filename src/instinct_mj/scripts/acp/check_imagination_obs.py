"""End-to-end check of the imagined ``policy`` observation of :class:`~instinct_mj.rl.acp.runner.ACPRunner` against the env's.

Collects real transitions (domain randomization, odometry delay / drift and the action delay ON, observation noise zeroed), samples
imagination starts from the replay buffer (also starts shortly after an episode start) and runs the runner's ``_imagine`` with
TEACHER FORCING: the actor returns the recorded actions and the world model step returns the recorded next states, so the imagined
observation of step ``t`` must equal the env's observation recorded ``t`` steps after the start (same episode). This validates the history
gather of the buffer (proprio, pelvis poses, back-fill at episode starts), the pose history push, the odometry model, the contact
targets, the goal resampling and the scan of the imagination.

    python -m instinct_mj.scripts.acp.check_imagination_obs [--num-envs 32] [--collect-steps 120] [--motion-folder DIR]
"""

from __future__ import annotations

import argparse
import os

import torch

import instinct_mj.tasks  # noqa: F401  (registers the tasks)
from instinct_mj.rl import InstinctRlVecEnvWrapper
from instinct_mj.rl.acp.mdp import ODO_HIST
from instinct_mj.rl.supermimic.mdp import unflatten_state
from instinct_mj.tasks.registry import load_env_cfg, load_env_cls, load_instinct_rl_cfg

TASK = "Instinct-ACP-Track-G1-v1"
OBS_SLICES = {
    "base_ang_vel": (0, 15), "projected_gravity": (15, 30), "joint_pos": (30, 145), "joint_vel": (145, 260), "last_action": (260, 375),
    "tracker_height_scan": (375, 438), "contact_targets": (438, 465), "goal_command": (465, 468),
}  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--collect-steps", type=int, default=120)
    parser.add_argument("--starts", type=int, default=512)
    parser.add_argument("--motion-folder", type=str, default=None)
    args = parser.parse_args()
    if args.motion_folder is not None:
        import instinct_mj.tasks.acp.config.g1.acp_tracking_cfg as acp_g1

        buf = acp_g1.motion_reference_cfg.motion_buffers["TerrainMotion"]
        buf.path = args.motion_folder
        buf.metadata_yaml = os.path.join(args.motion_folder, "metadata.yaml")
    from instinct_mj.rl.acp.runner import ACPRunner

    device = "cuda:0"
    cfg = load_env_cfg(TASK)
    cfg.scene.num_envs = args.num_envs
    cfg.curriculum["episode_start"].params["mode"] = "uniform"
    cfg.observations["policy"].enable_corruption = False  # observation noise off, DR / odometry / action delay stay on
    cfg.observations["policy"].terms["contact_targets"].params["noise"] = 0.0
    env = load_env_cls(TASK)(cfg=cfg, device=device)
    vec = InstinctRlVecEnvWrapper(env, policy_group="policy", critic_group="critic")
    agent = load_instinct_rl_cfg(TASK).to_dict()
    agent.update(buffer_steps=args.collect_steps + 8, warmup_steps=64, compile=False, compile_policy=False, horizon=8, action_noise=0.1)
    runner = ACPRunner(vec, agent, log_dir=None, device=device)
    runner.setup_training()
    mdp = runner.mdp
    mdp.proprio_noise.zero_()  # observation noise off in the imagination (the env's: ``enable_corruption`` False, contact target noise 0)
    mdp.scan_noise = mdp.target_noise = mdp.goal_noise = 0.0

    # collect, recording the env's policy observation at every transition start
    obs_log = []
    stats = {k: [] for k in ("r", "ep_len", "term", "done", "jl_soft")}
    runner.env.reset()
    runner.obs, _ = vec.get_observations()
    for _ in range(args.collect_steps):
        obs_log.append(runner.obs.clone())
        runner.collect(1, stats)
    buf = runner.buffer
    assert buf.size == args.collect_steps
    obs_log = torch.stack(obs_log)  # (T, N, 468), age-ordered like the buffer

    # the odometry sensor's pelvis pose history vs the one rebuilt from the buffered states (newest first; back-fill at the episode start)
    sensor = runner.unwrapped.scene["height_scanner"]
    S_now = runner.reader.state()
    rows_all = buf._row(torch.arange(buf.size, device=device))
    ep_now = runner.unwrapped.episode_length_buf
    pelvis = lambda S_: runner.mdp.base_of(runner.kin.forward(*S_))  # noqa: E731
    hist_p, hist_q = sensor.history_pos_w, sensor.history_quat_w
    err_hist = torch.zeros(ODO_HIST, device=device)
    for k in range(ODO_HIST):
        back = torch.minimum(torch.full_like(ep_now, k), ep_now)
        if k == 0:
            S_k = S_now
        else:
            S_k = unflatten_state(buf.data["s"][rows_all[(buf.size - back).clamp(min=0, max=buf.size - 1)], torch.arange(args.num_envs, device=device)])
            S_k = tuple(torch.where((back == 0).view(-1, *([1] * (x.dim() - 1))), a_, x) for x, a_ in zip(S_k, S_now))
        pp = pelvis(S_k)[0]
        err_hist[k] = (pp - hist_p[:, k]).abs().max()
    print("pose history: max |FK from buffer - sensor history| per slot (newest first):", [f"{float(e):.1e}" for e in err_hist])
    H = runner.horizon
    clip_ok = lambda f: f + H <= runner.motion.clip_end[runner.motion.clip_id[f]]  # noqa: E731
    eligible = buf.eligible_starts(clip_ok, max_age=buf.size - H - 1)
    done = buf.data["done"][buf._row(torch.arange(buf.size, device=device))]  # (T, N)
    ages = eligible[:, 0].unsqueeze(1) + torch.arange(H, device=device)
    no_done = ~done[ages[:, : H - 1], eligible[:, 1:2]].any(1)  # no episode end inside the imagined window
    eligible = eligible[no_done]
    young = buf.data["ep_step"][buf._row(eligible[:, 0]), eligible[:, 1]] < 8
    pick = torch.cat((eligible[young][: args.starts // 2], eligible[~young][: args.starts // 2]))
    print(f"imagination starts: {len(pick)} ({int(young[: len(pick)].sum())} within 8 steps of an episode start) of {len(eligible)} eligible")
    s, h = buf.starts_at(pick[:, 0], pick[:, 1], 6)
    rows = buf._row(pick[:, 0].unsqueeze(-1) + torch.arange(H, device=device))
    env_i = pick[:, 1].unsqueeze(-1).expand_as(rows)
    real_a = buf.data["a"][rows, env_i]  # (B, H, 23)
    real_s1 = buf.data["s1"][rows, env_i]  # (B, H, STATE_DIM)
    real_obs = obs_log[pick[:, 0].unsqueeze(-1) + torch.arange(H, device=device), env_i]  # (B, H, 468)

    seen = []
    step = {"t": 0}

    class Actor(torch.nn.Module):
        def forward(self, obs):
            seen.append(obs.detach())
            return real_a[:, step["t"]]

    def step_wm(S, fk, a, a_prev, xi_n):
        S1 = unflatten_state(real_s1[:, step["t"]])
        step["t"] += 1
        return real_a[:, step["t"] - 1], S1, runner.fk_all(S1), torch.zeros(len(a), 9, device=device), torch.zeros((), device=device)

    runner.ac.actor = Actor()
    runner.pol_norm = lambda x: x
    runner.step_wm = step_wm
    runner.cfg["contact_source"] = "reference"
    with torch.no_grad():
        im = runner._imagine(s, h)
    real_goal = buf.data["goal"][rows, env_i]  # (B, H, 5): the env's goal target (position, heading, time) at every step
    imag_goal = torch.stack(im["Gs"][:H], 1)
    same_goal = (imag_goal - real_goal).abs().amax(-1) < 1e-4  # the random draw of a goal resample differs between env and imagination
    imagined = torch.stack(seen, 1)  # (B, H, 468) (identity normalizer)
    err = (imagined - real_obs).abs()
    first = buf.data["ep_step"][buf._row(pick[:, 0]), pick[:, 1]] < 8
    _debug(err, pick, buf, runner)
    _debug(err, pick, buf, runner, (438, 465))
    print(f"\\n{'obs term':22s} {'t=0 max':>10s} {'t=0 mean':>10s} {'t>=1 max':>10s} {'t>=1 mean':>10s} {'young starts t=0 max':>22s}")
    for k, (a, b) in OBS_SLICES.items():
        e = err[..., a:b]
        print(f"{k:22s} {float(e[:, 0].max()):10.2e} {float(e[:, 0].mean()):10.2e} {float(e[:, 1:].max()):10.2e} {float(e[:, 1:].mean()):10.2e} {float(e[first, 0].max()):22.2e}")
        if k == "goal_command":
            e3 = torch.where(same_goal.unsqueeze(-1), e, torch.zeros_like(e))
            print(f"{'  (same goal target)':22s} {float(e3[:, 0].max()):10.2e} {float(e3[:, 0].mean()):10.2e} {float(e3[:, 1:].max()):10.2e} {float(e3[:, 1:].mean()):10.2e}   steps with a different goal target (resample draws): {int((~same_goal).sum())} of {same_goal.numel()}")
        if k == "tracker_height_scan":
            miss = (real_obs[..., a:b] > 4.4) | (imagined[..., a:b] > 4.4)
            e2 = torch.where(miss, torch.zeros_like(e), e)
            print(f"{'  (without miss cells)':22s} {float(e2[:, 0].max()):10.2e} {float(e2[:, 0].mean()):10.2e} {float(e2[:, 1:].max()):10.2e} {float(e2[:, 1:].mean()):10.2e}   missing cells {int(miss.sum())} of {miss.numel()}; cells >0.05: {int(((e2 > 0.05).sum()))}")


def _debug(err, pick, buf, runner, sl=(465, 468)):
    e = err[:, 0, sl[0] : sl[1]].amax(-1)
    top = e.argsort(descending=True)[:6]
    for i in top.tolist():
        env_i, age = int(pick[i, 1]), int(pick[i, 0])
        ep = int(buf.data["ep_step"][buf._row(pick[i, 0]), pick[i, 1]])
        print(f"  worst start {i}: err {float(e[i]):.2e} ep_step {ep} delay {float(buf.data['odo_delay'][buf._row(pick[i, 0]), pick[i, 1]]):.4f}")


if __name__ == "__main__":
    main()
