#!/usr/bin/env bash
set -euo pipefail

# serve_vllm.sh — поднять ДООБУЧЕННУЮ модель через vLLM (OpenAI API на :8001).
# Запускать на GPU-боксе, где лежит слитая fp16-модель qwen-mif-merged/.
#
# Почему именно такие переменные окружения (важно!):
#   VLLM_USE_FLASHINFER_SAMPLER=0 / VLLM_USE_FLASHINFER=0
#       flashinfer пытается JIT-компилировать CUDA-кернел сэмплера и требует nvcc
#       (CUDA toolkit), которого на голом боксе с одним лишь драйвером НЕТ.
#       Без отключения vLLM падает: "Could not find nvcc ...". Отключаем →
#       нативный PyTorch-сэмплер (чуть медленнее, но без зависимости от nvcc).
#       (Поставишь cuda-toolkit / nvcc — можно вернуть flashinfer ради скорости.)
#   VLLM_ATTENTION_BACKEND=FLASH_ATTN
#       precompiled flash-attention из vllm, тоже без JIT.
#
# Параметры под RTX 4090 48GB (модель делит карту со STT/TTS в проде — тогда
# снизь --gpu-memory-utilization до ~0.55, как в .env.prod.example).

MODEL_DIR=${MODEL_DIR:-/root/mif-train/qwen-mif-merged}
VENV=${VENV:-/root/mif-train/.venv-vllm}
PORT=${LLM_PORT:-8001}
API_KEY=${LLM_API_KEY:-local-token}
GPU_UTIL=${LLM_GPU_MEMORY_UTILIZATION:-0.90}
MAX_LEN=${LLM_MAX_MODEL_LEN:-8192}

echo "vLLM serve: model=$MODEL_DIR port=$PORT gpu_util=$GPU_UTIL"

exec env \
  VLLM_USE_FLASHINFER_SAMPLER=0 \
  VLLM_USE_FLASHINFER=0 \
  VLLM_ATTENTION_BACKEND=FLASH_ATTN \
  "$VENV/bin/vllm" serve "$MODEL_DIR" \
    --served-model-name mif \
    --host 127.0.0.1 --port "$PORT" --api-key "$API_KEY" \
    --max-model-len "$MAX_LEN" --max-num-seqs 1 \
    --gpu-memory-utilization "$GPU_UTIL" \
    --enforce-eager --trust-remote-code
