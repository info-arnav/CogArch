#!/usr/bin/env bash
# The untrained team, and published methods on the same model, on GPU 0's
# server, graded like the trained teams.
set -euo pipefail
cd "$(dirname "$0")/../.."
export HF_HOME=/workspace/hf
LOGS=/workspace/logs; mkdir -p "$LOGS"
URL=http://localhost:8200/v1
INFERENCE_BASE_URL=$URL VERIFIER_URL=http://localhost:8300/v1 python3 -m cli.main baseline config/general.yaml > "$LOGS/baseline-team.log" 2>&1
VERIFIER_URL=http://localhost:8300/v1 python3 -m cli.main baselines config/general.yaml --method single --calls 1 --base-url $URL > "$LOGS/baseline-single.log" 2>&1
for spec in majority:5 majority:9 moa:5 debate:6; do
  method=${spec%%:*}; calls=${spec##*:}
  VERIFIER_URL=http://localhost:8300/v1 python3 -m cli.main baselines config/general.yaml \
    --method "$method" --calls "$calls" --base-url $URL > "$LOGS/baseline-$method-$calls.log" 2>&1
done
echo baselines done
