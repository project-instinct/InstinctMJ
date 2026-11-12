#!/usr/bin/env bash
# ACP v1, Stage 3: reference tracking + goal reaching on the procedural terrains, planner throughout (eta_obs = eta_rew = 1).
#
# Task Instinct-ACP-Stage3-G1-v1, runner ACPStage3Runner / G1AcpStage3RunnerCfg. Every agent parameter is spelled out below (nothing is left to the config defaults); a trailing
# "# != config default" marks the values this stage sets differently from the dataclass default.
# Starts from a Stage 2 checkpoint (STAGE2_CKPT=<abs path of model_<it>.pt of a g1AcpStage2V1 run: actor, normalizer, world model, critic Vm, termination head,
# planner) and counts its iterations from 0. The loop is the official HybridRobotics/supermimic (see run_stage1.sh); both critics (Vm, Vfs) are LN + GELU MLPs on raw features, trained with
# Muon + fused Adam, with targets clamped at the analytic bound of their own objective. The Stage 3 runner does not implement the Track options (start sampling, stop-gradient terms,
# replay / hazard / injection, ...). Resume a stopped Stage 3 run with RESUME_CKPT.
# Environment: ACP_DATASET (motion folder), NUM_ENVS, MAX (iterations), SAVE_INTERVAL, EXPERIMENT_NAME, UNIT (systemd unit), MEMORY_MAX, STAGE2_CKPT (required).
# COMPILE (default True), CUDA_GRAPH_WM (default True), CUDA_GRAPH_POLICY (default True), NORM_AFTER_COLLECT.
# ACP_FOREGROUND=1 runs in the foreground (no systemd) instead of `systemd-run --user`.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../../.."
ACP_DATASET="${ACP_DATASET:-/home/duanxin/Datasets/acp/v4_train_natural_20261110_v1}"
export ACP_DATASET
[ -f "$ACP_DATASET/metadata.yaml" ] || { echo "ACP_DATASET=$ACP_DATASET has no metadata.yaml (dataset not built yet?)" >&2; exit 1; }
UNIT="${UNIT:-acp-s3-v1}"
if [[ -z "${ACP_FOREGROUND:-}" ]] && systemctl --user is-active --quiet "$UNIT"; then echo "$UNIT is already active"; exit 0; fi
CKPT="${RESUME_CKPT:-${STAGE2_CKPT:-}}"
if [[ -z "$CKPT" ]]; then echo "set STAGE2_CKPT (the checkpoint of the previous stage) or RESUME_CKPT" >&2; exit 2; fi
CKPT_ARGS=(--agent.load-run "$(dirname -- "$CKPT")" --agent.load-checkpoint "$(basename -- "$CKPT")")
# env cfg options spelled out as well (the agent's contact_detect must equal the env's)
ENV_ARGS=(--env.contact-detect force_head --env.sim.mujoco.iterations 10 --env.sim.mujoco.ls-iterations 20)
AGENT_ARGS=(
  --agent.buffer-steps 1000
  --agent.replay-ratio 16   # official supermimic Ratio: utd = num_envs * 16 / (wm_batches * wm_batch * wm_window) = 0.4 at 4096 envs
  --agent.warmup-steps 64
  --agent.hist 5
  --agent.wm-window 8
  --agent.wm-batches 10
  --agent.wm-batch 2048
  --agent.wm-failure-frac 0.0   # != config default; head windows only (the world model trains on complete windows)
  --agent.wm-failure-horizon 16
  --agent.wm-balance-frac 0.0
  --agent.wm-holdout-envs 0.0   # != config default; every env trains the world model
  --agent.wm-only-iters 0
  --agent.wm-physics-inputs ""
  --agent.wm-ref-contacts False
  --agent.wm-contact-feedback False
  --agent.profile-phases True   # != config default
  --agent.lazy-exact-heightmap False
  --agent.fail-head False   # != config default
  --agent.fail-head-warmup-iters 0
  --agent.fail-windows 512
  --agent.fail-clip 0.95
  --agent.fail-pos-weight 1.0
  --agent.horizon 32
  --agent.pi-batch 2048   # != config default
  --agent.failure-window-fix False
  --agent.scan-slope-gradient False
  --agent.imagine-clamp-joints False   # != config default
  --agent.term-source analytic   # the normalized sigmoid of the env thresholds (soft-term-softness)
  --agent.start-balance-frac 0.0
  --agent.start-sampling uniform
  --agent.gamma 0.99
  --agent.lam 0.95
  --agent.action-noise 0.1   # x the per-joint SuperMimic action scale (0.035-0.055 rad)
  --agent.geo-contact False
  --agent.geo-contact-d0 0.008
  --agent.geo-contact-tau 0.008
  --agent.contact-detect force_head   # c^meas = the contact sensor's |F_z| > 1 N (env, hindsight labels); the imagination uses the world model's contact head (ACP author)
  --agent.sdf-tau 0.001
  --agent.wm-contact-loss True   # BCE of the world model's contact head against c^meas
  --agent.wm-scan planner   # 17 x 11 scan (+-0.8 x +-0.5 m, centred on the pelvis) for the world model and the critics; "wide" = 31 x 15 forward-only (0 .. 1.5 m), blind behind the pelvis
  --agent.clip-aliases ""
  --agent.imagine-ref-replay-eps 0.0
  --agent.imagine-link-hazard-eps 0.0
  --agent.imagine-link-hazard-tau 0.02
  --agent.real-ref-inject-eps 0.0
  --agent.abpt-one-step-weight 0.0
  --agent.bptt-stopgrad-terms ""
  --agent.pi-lr-scale 1.0
  --agent.policy-updates-per-iter 1
  --agent.imagine-action-noise 0.0
  --agent.critic-epochs 5
  --agent.critic-minibatches 4
  --agent.critic-hidden 256,128
  --agent.no-critic False
  --agent.wmres-cut False
  --agent.wmres-cut-theta 0.5
  --agent.wmres-cut-flux 0.0
  --agent.wmres-cut-alpha 1.0
  --agent.wmres-cut-warmup-iters 500
  --agent.wmres-cut-max-frac 0.9
  --agent.wmres-cap-clips ""
  --agent.lr 0.0002
  --agent.compile "${COMPILE:-True}"   # default-mode compiled rollout + loss, captured by the world-model graph (official: compile, then capture)
  --agent.compile-policy True
  --agent.planner-imagination-subset True
  --agent.checkpoint-imagination False
  --agent.heightmap-resolution 0.05
  --agent.exact-scan True
  --agent.heightmap-cache-dir $HOME/.cache/instinct_mj/supermimic_heightmaps
  --agent.contact-source planner
  --agent.eta-obs-start-iter 0
  --agent.eta-rew-start-iter 0
  --agent.eta-ann-iters 1
  --agent.train-planner True
  --agent.planner-lr 0.0001
  --agent.planner-updates-per-iter 1   # != config default (review 2026-11-07: 10 per iteration = ~39x the paper update rate per transition)
  --agent.planner-batch 4096
  --agent.planner-sub-batches 8
  --agent.planner-min-entries 1000
  --agent.planner-replay-capacity 200000
  --agent.planner-ref-tile-capacity 5000
  --agent.planner-success-only True
  --agent.restart-iteration True
  --agent.wm-nominal 0.01,0.02,0.1,3.14   # Bryson nominal errors: root position, root rotation, joint position, joint velocity (official)
  --agent.soft-term-softness 0.05   # official: one normalized sigmoid, sigmoid((max_i e_i / theta_i - 1) / 0.05)
  --agent.return-horizon 20
  --agent.return-memory 256
  --agent.return-steepness 25.0
  --agent.return-explore 0.5
  --agent.return-keep 0.1
  --agent.cuda-graph-wm "${CUDA_GRAPH_WM:-True}"   # the world-model gradient step as one CUDA graph (official CapturedUpdate)
  --agent.cuda-graph-policy "${CUDA_GRAPH_POLICY:-True}"   # the Stage 3 policy update (both regimes, both critics, the return scales) as one CUDA graph; the sampling of the starts stays eager
  --agent.norm-after-collect "${NORM_AFTER_COLLECT:-False}"   # True: normalizer statistics after the collection of an iteration (official _flush_normalization)
  --agent.return-resume-stats True
  --agent.onnx-deploy-joint-order False
  --agent.vfs-lr 0.001
  --agent.vfs-warmup-iters 500
  --agent.pi-goal-fraction 0.5
  --agent.stumble-lr 0.001
  --agent.stumble-eval-states 16384
  --agent.stumble-eval-chunk 2048
  --agent.return-norm-momentum 0.99
  --agent.goal-soft-tau-dist 0.05
  --agent.goal-soft-tau-speed 0.05
  --agent.goal-soft-tau-height 0.02
  --agent.goal-soft-tau-angle 0.05
  --agent.planner-tile-capacity 2000
  --agent.max-iterations "${MAX:-10000}"
  --agent.save-interval "${SAVE_INTERVAL:-250}"   # != config default
  --agent.log-interval 10
  --agent.experiment-name "${EXPERIMENT_NAME:-g1AcpStage3V1}"
  --agent.policy-observation-group policy
  --agent.critic-observation-group critic
)
CMD=("$PWD/.venv/bin/python" -m instinct_mj.scripts.acp.train_on_dataset Instinct-ACP-Stage3-G1-v1 --num-envs "${NUM_ENVS:-2048}" "${ENV_ARGS[@]}" "${AGENT_ARGS[@]}" "${CKPT_ARGS[@]}")
if [[ -n "${ACP_FOREGROUND:-}" ]]; then
  exec env PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 "${CMD[@]}"
fi
LOG="${LOG:-/tmp/$UNIT.log}"
systemctl --user reset-failed "$UNIT" 2>/dev/null || true
printf '==== %s launch %s dataset=%s envs=%s\n' "$UNIT" "$(date -Is)" "$ACP_DATASET" "${NUM_ENVS:-2048}" >> "$LOG"
systemd-run --user --unit="$UNIT" -p MemoryMax="${MEMORY_MAX:-40G}" -p WorkingDirectory="$PWD" \
  -p "StandardOutput=append:$LOG" -p "StandardError=append:$LOG" \
  /usr/bin/env ACP_DATASET="$ACP_DATASET" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 "${CMD[@]}"
echo "started $UNIT (log $LOG)"
