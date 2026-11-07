#!/usr/bin/env bash
# One-time setup on the GPU server (AMD ROCm; tested on 1× MI300X, Ubuntu 24.04).
#
# Both vLLM and the experiment run from AMD's vLLM image, in two containers:
#   cogarch-vllm  — the inference server (scripts/serve_vllm.sh)
#   cogarch-exp   — the experiment code, DPO training, EvalPlus, dashboard
# The repo is mounted at the same path in both, so adapter paths match.
set -euo pipefail

IMAGE="${VLLM_IMAGE:-rocm/vllm:latest}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"

if ! command -v docker >/dev/null; then
  apt-get update -qq && apt-get install -y -qq docker.io
  systemctl enable --now docker
fi
docker pull "$IMAGE"
mkdir -p "$HOME/.cache/huggingface"

docker rm -f cogarch-exp >/dev/null 2>&1 || true
docker run -d --name cogarch-exp --init \
  --network=host --ipc=host \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  --security-opt seccomp=unconfined \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$REPO:$REPO" -w "$REPO" \
  --entrypoint sleep "$IMAGE" infinity

docker exec cogarch-exp pip install -q -e ".[research,dev]"

# Pre-download datasets and the embedding model so runs start immediately.
docker exec cogarch-exp python - <<'PY'
import torch
from evalplus.data import get_human_eval_plus, get_mbpp_plus
from sentence_transformers import SentenceTransformer
from src.eval.benchmarks.mbpp import load_pool
assert torch.cuda.is_available(), "PyTorch cannot see the GPU"
print("GPU:", torch.cuda.get_device_name(0))
print("HumanEval+:", len(get_human_eval_plus()), "MBPP+:", len(get_mbpp_plus()))
print("MBPP training pool:", len(load_pool()))
SentenceTransformer("BAAI/bge-small-en-v1.5")
PY
