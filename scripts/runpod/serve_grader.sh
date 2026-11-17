#!/usr/bin/env bash
set -euo pipefail
export HF_HOME=/workspace/hf PIP_BREAK_SYSTEM_PACKAGES=1
: "${VERIFIER_API_KEY:?set VERIFIER_API_KEY}"
LOGS=/workspace/logs; mkdir -p "$LOGS"
pip install -q vllm==0.10.2 transformers==4.56.2
nohup vllm serve IAAR-Shanghai/xVerify-3B-Ia --host 0.0.0.0 --port 8300 --api-key "$VERIFIER_API_KEY" \
  --gpu-memory-utilization 0.85 --max-model-len 8192 --max-num-seqs 512 > "$LOGS/vllm-grader.log" 2>&1 &
until curl -sf -H "Authorization: Bearer $VERIFIER_API_KEY" http://localhost:8300/v1/models >/dev/null; do sleep 5; done
echo "grader up on :8300"
