#!/usr/bin/env bash
# One-time setup of a RunPod pod (PyTorch 2.8 template): pinned packages and models.
set -euo pipefail
export HF_HOME=/workspace/hf PIP_BREAK_SYSTEM_PACKAGES=1
cd "$(dirname "$0")/../.."
pip install -q vllm==0.10.2 transformers==4.56.2 peft==0.17.1 trl==0.23.1 datasets==4.0.0 \
  accelerate sentence-transformers "math-verify[antlr4_13_2]" evalplus fastapi uvicorn \
  typer rich pyyaml httpx pydantic pytest pytest-asyncio matplotlib
pip install -q --no-deps -e .
python3 -c "from huggingface_hub import snapshot_download as s; [s(r) for r in ['Qwen/Qwen2.5-1.5B-Instruct', 'IAAR-Shanghai/xVerify-3B-Ia', 'BAAI/bge-small-en-v1.5']]"
python3 -m pytest -q tests
