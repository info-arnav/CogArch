#!/usr/bin/env bash
# Launch the experiment inside the container: two agents together vs one agent
# alone, N replicates each (default 3). All runs share the vLLM server.
#   scripts/launch_pilot.sh [replicates] [model]
# Logs: data/runs/logs/<run>.log · watch live with `make dashboard`.
set -euo pipefail

REPLICATES="${1:-3}"
MODEL_ARG=""
if [ -n "${2:-}" ]; then MODEL_ARG="--model $2"; fi
REPO="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$REPO/data/runs/logs"

for replicate in $(seq 1 "$REPLICATES"); do
  for config in general general_solo; do
    name="$(grep -E '^  name:' "$REPO/config/${config}.yaml" | awk '{print $2}')-r${replicate}"
    docker exec -d -w "$REPO" cogarch-exp sh -c \
      "python -m cli.main run config/${config}.yaml --replicate ${replicate} ${MODEL_ARG} > data/runs/logs/${name}.log 2>&1"
    echo "started ${name} → data/runs/logs/${name}.log"
  done
done
