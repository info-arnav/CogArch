#!/usr/bin/env bash
# One base-model server per GPU (runtime LoRA) on port 8200+i, and one shared
# xVerify grader on GPU 0, port 8300. (Ports 8001 and 8081 are RunPod's own.)
# Each GPU keeps ~65% of its memory for DPO training (2 jobs at a time).
set -euo pipefail
export HF_HOME=/workspace/hf VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
LOGS=/workspace/logs; mkdir -p "$LOGS"
GPUS=$(nvidia-smi -L | wc -l)
for i in $(seq 0 $((GPUS - 1))); do
  CUDA_VISIBLE_DEVICES=$i nohup vllm serve Qwen/Qwen2.5-1.5B-Instruct --port $((8200 + i)) \
    --enable-lora --max-lora-rank 16 --max-loras 16 --max-cpu-loras 64 --max-num-seqs 512 \
    --gpu-memory-utilization 0.30 --max-model-len 16384 > "$LOGS/vllm-gpu$i.log" 2>&1 &
done
CUDA_VISIBLE_DEVICES=0 nohup vllm serve IAAR-Shanghai/xVerify-3B-Ia --port 8300 \
  --gpu-memory-utilization 0.25 --max-model-len 8192 > "$LOGS/vllm-grader.log" 2>&1 &
for port in $(seq 8200 $((8200 + GPUS - 1))) 8300; do
  until curl -sf "http://localhost:$port/v1/models" >/dev/null; do sleep 5; done
  echo "up :$port"
done
