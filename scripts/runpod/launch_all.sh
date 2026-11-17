#!/usr/bin/env bash
# Every run of the full experiment (docs/preregistration.md), spread over GPUs
# 1..N-1: each run uses its GPU's model server and trains on that GPU. GPU 0
# holds the shared grader and runs the baselines (scripts/runpod/baselines.sh).
set -euo pipefail
cd "$(dirname "$0")/../.."
export HF_HOME=/workspace/hf
LOGS=/workspace/logs; mkdir -p "$LOGS"
GPUS=$(nvidia-smi -L | wc -l)
RUNS=(
  full:1 full:2 full:3
  no-memory:1 no-memory:2 no-memory:3
  neutral:1 neutral:2 neutral:3
  simple-credit:1 simple-credit:2 simple-credit:3
  no-sharing:1 no-sharing:2 no-sharing:3
  solo:1 solo:2 solo:3
  no-star:1 no-practice:1 no-replay:1
)
i=0
for run in "${RUNS[@]}"; do
  config=${run%%:*}; replicate=${run##*:}; gpu=$((1 + i % (GPUS - 1)))
  CUDA_VISIBLE_DEVICES=$gpu \
  INFERENCE_BASE_URL=http://localhost:$((8200 + gpu))/v1 \
  VERIFIER_URL=http://localhost:8300/v1 \
  TRAINING_SLOTS_DIR=/workspace/slots/gpu$gpu \
    nohup python3 -m cli.main run "config/runs/$config.yaml" --replicate "$replicate" \
    > "$LOGS/$config-r$replicate.log" 2>&1 < /dev/null &
  echo "$config-r$replicate on GPU $gpu"
  i=$((i + 1))
done
