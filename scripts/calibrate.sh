#!/usr/bin/env bash
# Pick the model: for each candidate, serve it, then measure (without training)
# how often the team solves training problems, validation Pass@1, and whether
# the coordinator and probe outputs parse. Results: data/calibration/*.json
#   scripts/calibrate.sh Qwen/Qwen2.5-Coder-1.5B-Instruct Qwen/Qwen2.5-Coder-3B-Instruct ...
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
for model in "$@"; do
  MODEL="$model" scripts/serve_vllm.sh
  docker exec -w "$REPO" cogarch-exp python -m cli.main calibrate config/pilot.yaml --model "$model"
done
