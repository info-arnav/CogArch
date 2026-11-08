#!/usr/bin/env bash
# Reference systems for the comparison, evaluated on the same benchmarks while
# experiments keep running (small GPU share, separate vLLM server on :8001):
#   1.5B × 9 answers with majority vote (same calls as the brain), on the main server
#   Qwen2.5-3B and Qwen2.5-7B, single greedy answer (bigger models)
# Results: data/runs/references/report.json · log: data/runs/logs/references.log
set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
LOG=data/runs/logs/references.log
EXP="docker exec -w $REPO cogarch-exp"
step() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }

step "1.5B majority@9 (main server)"
$EXP python -m cli.main reference --model Qwen/Qwen2.5-1.5B-Instruct --label "Qwen2.5-1.5B" \
  --majority 9 >> "$LOG" 2>&1 || step "majority@9 failed"

for size in 7B 3B; do
  model="Qwen/Qwen2.5-${size}-Instruct"
  step "serving $model on :8001"
  MODEL="$model" PORT=8001 NAME=cogarch-vllm-ref NO_LORA=1 GPU_MEMORY=0.14 \
    scripts/serve_vllm.sh >> "$LOG" 2>&1 || { step "serving $model failed"; continue; }
  step "$model plain"
  $EXP python -m cli.main reference --model "$model" --label "Qwen2.5-${size}" \
    --base-url http://localhost:8001/v1 >> "$LOG" 2>&1 || step "$model failed"
done
docker rm -f cogarch-vllm-ref > /dev/null 2>&1
step "references done"
