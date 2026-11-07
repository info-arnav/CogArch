#!/usr/bin/env bash
# Launch the pilot inside the experiment container: two agents together vs one
# agent alone, three replicates each. All runs share the vLLM server.
# Logs: data/runs/logs/<run>.log · watch live with `make dashboard`.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$REPO/data/runs/logs"

for replicate in 1 2 3; do
  for config in pilot pilot_solo; do
    name="$(grep -E '^  name:' "$REPO/config/${config}.yaml" | awk '{print $2}')-r${replicate}"
    docker exec -d -w "$REPO" cogarch-exp sh -c \
      "python -m cli.main run config/${config}.yaml --replicate ${replicate} > data/runs/logs/${name}.log 2>&1"
    echo "started ${name} → data/runs/logs/${name}.log"
  done
done
