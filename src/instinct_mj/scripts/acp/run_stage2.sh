#!/usr/bin/env bash
# ACP v1, Stage 2: the tracker follows the planner's contact commands (eta_obs / eta_rew schedules), the planner trains on hindsight labels.
#
# Task Instinct-ACP-Stage2-G1-v1, runner ACPRunner / G1AcpStage2RunnerCfg. Every agent parameter is spelled out below (nothing is left to the config defaults); a trailing
# "# != config default" marks the values this stage sets differently from the dataclass default.
# Starts from a Stage 1 checkpoint (STAGE1_CKPT=<abs path of model_<it>.pt of a g1AcpTrackV1 run>: actor, normalizer, world model, critic, termination head)
# and counts its iterations from 0. The world-model / critic / term-head options are those of Stage 1 (they define the network shapes); the
# episode cap of the task is 20 s, the planner trains with 1 optimizer step per iteration. Resume a stopped Stage 2 run with RESUME_CKPT.
# Environment: ACP_DATASET (motion folder), NUM_ENVS, MAX (iterations), SAVE_INTERVAL, EXPERIMENT_NAME, UNIT (systemd unit), MEMORY_MAX, STAGE1_CKPT (required).
# ACP_FOREGROUND=1 runs in the foreground (no systemd) instead of `systemd-run --user`.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../../.."
ACP_DATASET="${ACP_DATASET:-/home/duanxin/Datasets/acp/v3_train11}"
export ACP_DATASET
[ -f "$ACP_DATASET/metadata.yaml" ] || { echo "ACP_DATASET=$ACP_DATASET has no metadata.yaml (dataset not built yet?)" >&2; exit 1; }
UNIT="${UNIT:-acp-s2-v1}"
if [[ -z "${ACP_FOREGROUND:-}" ]] && systemctl --user is-active --quiet "$UNIT"; then echo "$UNIT is already active"; exit 0; fi
CKPT="${RESUME_CKPT:-${STAGE1_CKPT:-}}"
if [[ -z "$CKPT" ]]; then echo "set STAGE1_CKPT (the checkpoint of the previous stage) or RESUME_CKPT" >&2; exit 2; fi
CKPT_ARGS=(--agent.load-run "$(dirname -- "$CKPT")" --agent.load-checkpoint "$(basename -- "$CKPT")")
# env cfg options spelled out as well (the agent's contact_detect must equal the env's)
ENV_ARGS=(--env.contact-detect force_head --env.sim.mujoco.iterations 10 --env.sim.mujoco.ls-iterations 20)
AGENT_ARGS=(
  --agent.buffer-steps 1000
  --agent.utd 0.4
  --agent.warmup-steps 64
  --agent.hist 5
  --agent.wm-window 8
  --agent.wm-batches 10
  --agent.wm-batch 2048
  --agent.wm-failure-frac 0.0   # != config default; uniform world-model windows, supermimic_vibed
  --agent.wm-failure-horizon 16
  --agent.wm-balance-frac 0.0
  --agent.wm-holdout-envs 0.0   # != config default; every env trains the world model, supermimic_vibed
  --agent.wm-only-iters 0
  --agent.wm-physics-inputs ""
  --agent.wm-ref-contacts False
  --agent.wm-contact-feedback False
  --agent.profile-phases True   # != config default
  --agent.lazy-exact-heightmap True   # != config default
  --agent.fail-head False   # != config default
  --agent.fail-head-warmup-iters 0
  --agent.fail-windows 512
  --agent.fail-clip 0.95
  --agent.fail-pos-weight 1.0
  --agent.horizon 32
  --agent.pi-batch 2048   # != config default
  --agent.imagine-past-clip-end False
  --agent.clip-end-bootstrap False
  --agent.reset-sampling bins
  --agent.bins-v2 False
  --agent.bins-difficulty-k 0.0
  --agent.clip-end-exact-last False
  --agent.failure-window-fix False
  --agent.scan-slope-gradient False
  --agent.soft-term-tau-dist 0.02
  --agent.soft-term-tau-tilt 0.04
  --agent.imagine-clamp-joints False   # != config default
  --agent.term-source analytic   # supermimic_vibed: sigmoid of the env thresholds (height tau 0.02 m, tilt tau 0.04)
  --agent.clip-end-v2 False
  --agent.start-balance-frac 0.0
  --agent.start-sampling uniform
  --agent.gamma 0.99
  --agent.lam 0.95
  --agent.action-noise 0.1   # x the per-joint SuperMimic action scale (0.035-0.055 rad), supermimic_vibed
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
  --agent.value-max 0.0
  --agent.wmres-cut False
  --agent.wmres-cut-theta 0.5
  --agent.wmres-cut-flux 0.0
  --agent.wmres-cut-alpha 1.0
  --agent.wmres-cut-warmup-iters 500
  --agent.wmres-cut-max-frac 0.9
  --agent.wmres-cap-clips ""
  --agent.lr 0.0002
  --agent.lr-decay-iters 250000
  --agent.lr-final-ratio 0.03
  --agent.amp bf16
  --agent.compile True
  --agent.compile-policy True
  --agent.planner-imagination-subset True
  --agent.policy-autocast False
  --agent.checkpoint-imagination False
  --agent.heightmap-resolution 0.05
  --agent.exact-scan True
  --agent.heightmap-cache-dir $HOME/.cache/instinct_mj/supermimic_heightmaps
  --agent.contact-source mixed
  --agent.eta-obs-start-iter "${ETA_OBS_START:-1000}"
  --agent.eta-rew-start-iter "${ETA_REW_START:-3000}"
  --agent.eta-ann-iters "${ETA_ANN:-4000}"
  --agent.train-planner True
  --agent.planner-lr 0.0001
  --agent.planner-updates-per-iter 1
  --agent.planner-batch 4096
  --agent.planner-sub-batches 8
  --agent.planner-min-entries 1000
  --agent.planner-replay-capacity 200000
  --agent.planner-ref-tile-capacity 5000
  --agent.planner-success-only True
  --agent.restart-iteration True
  --agent.onnx-deploy-joint-order False
  --agent.max-iterations "${MAX:-10000}"
  --agent.save-interval "${SAVE_INTERVAL:-250}"   # != config default
  --agent.log-interval 10
  --agent.experiment-name "${EXPERIMENT_NAME:-g1AcpStage2V1}"
  --agent.policy-observation-group policy
  --agent.critic-observation-group critic
)
CMD=("$PWD/.venv/bin/python" -m instinct_mj.scripts.acp.train_on_dataset Instinct-ACP-Stage2-G1-v1 --num-envs "${NUM_ENVS:-2048}" "${ENV_ARGS[@]}" "${AGENT_ARGS[@]}" "${CKPT_ARGS[@]}")
if [[ -n "${ACP_FOREGROUND:-}" ]]; then
  exec env PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 "${CMD[@]}"
fi
LOG="${LOG:-/tmp/$UNIT.log}"
systemctl --user reset-failed "$UNIT" 2>/dev/null || true
printf '==== %s launch %s dataset=%s envs=%s\n' "$UNIT" "$(date -Is)" "$ACP_DATASET" "${NUM_ENVS:-2048}" >> "$LOG"
systemd-run --user --unit="$UNIT" -p MemoryMax="${MEMORY_MAX:-32G}" -p WorkingDirectory="$PWD" \
  -p "StandardOutput=append:$LOG" -p "StandardError=append:$LOG" \
  /usr/bin/env ACP_DATASET="$ACP_DATASET" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 "${CMD[@]}"
echo "started $UNIT (log $LOG)"
