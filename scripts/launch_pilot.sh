#!/usr/bin/env bash
# Launch the pilot: two agents together vs one agent alone, two replicates each.
# All runs share the vLLM server; logs go to data/runs/logs/.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
mkdir -p data/runs/logs

for replicate in 1 2; do
  for config in pilot pilot_solo; do
    log="data/runs/logs/${config}-r${replicate}.log"
    nohup .venv/bin/python -m cli.main run "config/${config}.yaml" \
      --replicate "$replicate" >"$log" 2>&1 &
    echo "started ${config} r${replicate} (pid $!) → $log"
  done
done
