# CogArch — Python + training dependencies
# Usage: docker compose up
# Works with or without GPU — fine-tuning requires NVIDIA GPU and the
# nvidia-container-toolkit on the host.

# Use the official CUDA image so GPU is available when present.
# On CPU-only machines this still works — unsloth just won't be importable,
# and the finetuner degrades gracefully.
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    python3.11-venv \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && python -m pip install --upgrade pip

WORKDIR /app

# Install core deps first (better layer caching)
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Install training deps (unsloth needs torch first)
# These are optional — if CUDA is unavailable they'll warn but not crash
RUN pip install torch --index-url https://download.pytorch.org/whl/cu121 || \
    pip install torch  # CPU fallback

RUN pip install \
    "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" \
    "trl>=0.8.0" \
    "transformers>=4.40.0" \
    "datasets>=2.19.0" \
    || echo "WARNING: unsloth install failed — fine-tuning disabled"

# Install the project itself
COPY . .
RUN pip install -e .

# Data and model directories (volumes are mounted over these at runtime)
RUN mkdir -p data/benchmarks data/experiments/code models

CMD ["python", "-m", "cli.main", "--help"]
