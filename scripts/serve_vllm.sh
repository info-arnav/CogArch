#!/usr/bin/env bash
# Serve the base model with vLLM (AMD ROCm image) and runtime LoRA loading.
# Every specialist/coordinator adapter is loaded into this one server by name.
#
# The repo is mounted at the same absolute path inside the container, so the
# adapter paths the experiment sends to /v1/load_lora_adapter resolve there too.
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
IMAGE="${VLLM_IMAGE:-rocm/vllm:latest}"
GPU_MEMORY="${GPU_MEMORY:-0.45}"   # leave the rest of the GPU for DPO training
PORT="${PORT:-8000}"
NAME="${NAME:-cogarch-vllm}"
# Reference models (scripts/run_references.sh) need no adapters.
LORA_ARGS="--enable-lora --max-lora-rank 16 --max-loras 16 --max-cpu-loras 64"
if [ -n "${NO_LORA:-}" ]; then LORA_ARGS=""; fi
REPO="$(cd "$(dirname "$0")/.." && pwd)"

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" \
  --network=host --ipc=host \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  --security-opt seccomp=unconfined \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$REPO:$REPO" \
  -e VLLM_ALLOW_RUNTIME_LORA_UPDATING=True \
  "$IMAGE" \
  vllm serve "$MODEL" \
    --port "$PORT" \
    $LORA_ARGS \
    --gpu-memory-utilization "$GPU_MEMORY" \
    --max-model-len 8192

echo "Waiting for vLLM on :$PORT ..."
until curl -sf "http://localhost:$PORT/v1/models" >/dev/null; do sleep 5; done
echo "vLLM is up."
