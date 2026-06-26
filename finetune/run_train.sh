#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# run_train.sh — поднимает окружение и обучает LoRA Qwen2.5-14B на GPU-сервере.
# Запускать НА арендованном боксе (A100 40GB / A6000 48GB / L40S и т.п.).
#
# ШАГ 1. Закинь на сервер в ОДНУ папку (выполни У СЕБЯ на маке):
#   scp -P <PORT> \
#       train_lora_qwen.py \
#       requests.finetune.tts.jsonl \
#       requests.finetune.tts.val.jsonl \
#       run_train.sh \
#       root@<SERVER_IP>:/workspace/
#
# ШАГ 2. Зайди на сервер и запусти:
#   ssh -p <PORT> root@<SERVER_IP>
#   cd /workspace && chmod +x run_train.sh && ./run_train.sh
#
# ОПЦИОНАЛЬНО — авто-пуш адаптера в ПРИВАТНЫЙ Hugging Face repo:
#   export HF_TOKEN=hf_xxxxxxxx
#   export HF_REPO=твой_ник/qwen-mif-lora
#   ./run_train.sh
#
# ОПЦИИ окружения:
#   SKIP_VENV=1   — не создавать venv (если на боксе уже стоят torch+unsloth)
#   PYBIN=python3.11 — какой python использовать (по умолчанию python3)
#
# ВНИМАНИЕ: базовая Qwen2.5-14B качается с HF (~28 ГБ). Нужно ~60 ГБ свободного
# диска (модель + чекпойнты). Проверь: df -h .
# ============================================================================

cd "$(dirname "$0")"
PYBIN=${PYBIN:-python3}

echo "==> [1/6] Проверка GPU"
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "!! nvidia-smi не найден — это точно GPU-сервер?" >&2
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "Свободно на диске:"; df -h . | tail -1

echo "==> [2/6] Проверка файлов"
for f in train_lora_qwen.py requests.finetune.tts.jsonl requests.finetune.tts.val.jsonl; do
  [ -f "$f" ] || { echo "!! нет файла: $f (закинь через scp, см. шапку скрипта)" >&2; exit 1; }
done
echo "train: $(wc -l < requests.finetune.tts.jsonl) строк | val: $(wc -l < requests.finetune.tts.val.jsonl) строк"

echo "==> [3/6] Окружение + зависимости"
if [ "${SKIP_VENV:-0}" = "1" ]; then
  echo "SKIP_VENV=1 — ставлю в текущее окружение"
  PY="$PYBIN"
else
  "$PYBIN" -m venv .venv-train
  # shellcheck disable=SC1091
  source .venv-train/bin/activate
  PY=python
fi
$PY -m pip install -q --upgrade pip
# unsloth тянет совместимые torch/transformers сам;
# trl/peft/accelerate/bitsandbytes ставим --no-deps, чтобы не сломать torch
$PY -m pip install -q unsloth
$PY -m pip install -q --no-deps trl peft accelerate bitsandbytes
if [ -n "${HF_REPO:-}" ]; then
  $PY -m pip install -q "huggingface_hub[cli]"
fi

echo "==> [4/6] Обучение (полный лог -> train.log)"
echo "    Следи за eval_loss: если на 2-й эпохе растёт при падающем train — переобучение."
$PY train_lora_qwen.py 2>&1 | tee train.log

echo "==> [5/6] Проверка результата"
[ -d qwen-mif-lora ] || { echo "!! адаптер qwen-mif-lora/ не создан — смотри train.log" >&2; exit 1; }
echo "Размер адаптера:"; du -sh qwen-mif-lora

echo "==> [6/6] Выгрузка адаптера"
if [ -n "${HF_TOKEN:-}" ] && [ -n "${HF_REPO:-}" ]; then
  echo "Пушу в приватный HF repo: $HF_REPO"
  $PY -m huggingface_hub.commands.huggingface_cli upload \
      "$HF_REPO" qwen-mif-lora . \
      --repo-type model --private --token "$HF_TOKEN" \
    || huggingface-cli upload "$HF_REPO" qwen-mif-lora . \
         --repo-type model --private --token "$HF_TOKEN"
  echo "Готово: https://huggingface.co/$HF_REPO"
else
  echo "HF_TOKEN/HF_REPO не заданы — адаптер лежит локально в ./qwen-mif-lora"
  echo "Скачать к себе (выполни У СЕБЯ):"
  echo "  scp -P <PORT> -r root@<SERVER_IP>:$(pwd)/qwen-mif-lora ."
fi

echo ""
echo "✅ ВСЁ ГОТОВО. Адаптер: ./qwen-mif-lora  | лог: ./train.log"
