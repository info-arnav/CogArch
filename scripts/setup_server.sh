#!/usr/bin/env bash
# One-time setup of the experiment environment on the GPU server.
# vLLM runs in its own container (scripts/serve_vllm.sh); this venv holds the
# experiment code, PyTorch for DPO training, TRL/PEFT, EvalPlus and embeddings.
set -euo pipefail

TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/rocm6.4}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install torch --index-url "$TORCH_INDEX"
.venv/bin/pip install -e ".[research,dev]"

.venv/bin/python - <<'PY'
import torch
assert torch.cuda.is_available(), "PyTorch cannot see the GPU"
print("GPU:", torch.cuda.get_device_name(0))
PY

# Pre-download datasets and the embedding model so runs start immediately.
.venv/bin/python - <<'PY'
from evalplus.data import get_human_eval_plus, get_mbpp_plus
from sentence_transformers import SentenceTransformer
from src.eval.benchmarks.mbpp import load_pool
print("HumanEval+:", len(get_human_eval_plus()), "MBPP+:", len(get_mbpp_plus()))
print("MBPP training pool:", len(load_pool()))
SentenceTransformer("BAAI/bge-small-en-v1.5")
PY
