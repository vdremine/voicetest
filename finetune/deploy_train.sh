#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# deploy_train.sh — ПУЛЬТ управления обучением с локального мака.
# Заливает файлы на GPU-сервер, запускает обучение в ФОНЕ (переживает обрыв
# SSH), показывает логи, забирает адаптер.
#
# Запускать У СЕБЯ на маке из папки logs/.
#
# Команды:
#   ./deploy_train.sh up       # залить + установить + запустить + показать лог
#   ./deploy_train.sh logs     # подключиться к логам идущего обучения
#   ./deploy_train.sh status   # GPU + жив ли процесс обучения
#   ./deploy_train.sh pull     # скачать готовый адаптер к себе
#   ./deploy_train.sh shell    # просто зайти на сервер
#
# Авто-пуш адаптера в приватный HF (опционально, перед 'up'):
#   export HF_TOKEN=hf_xxxx
#   export HF_REPO=твой_ник/qwen-mif-lora
#   ./deploy_train.sh up
# ============================================================================

# ---- доступ к серверу ------------------------------------------------------
KEY=/Users/dr_emin/.ssh/vps_new
HOST=root@45.157.161.161
PORT=22
REMOTE_DIR=/root/mif-train

SSH_OPTS=(-i "$KEY" -p "$PORT" -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15)
SCP_OPTS=(-i "$KEY" -P "$PORT" -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15)

ssh_() { ssh "${SSH_OPTS[@]}" "$HOST" "$@"; }

cd "$(dirname "$0")"

FILES=(train_lora_qwen.py requests.finetune.tts.jsonl requests.finetune.tts.val.jsonl run_train.sh)

cmd_up() {
  echo "==> Проверка локальных файлов"
  for f in "${FILES[@]}"; do
    [ -f "$f" ] || { echo "!! нет файла: $f" >&2; exit 1; }
  done

  echo "==> Проверка связи с $HOST"
  ssh_ "echo OK: \$(hostname) && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true"

  echo "==> Создаю $REMOTE_DIR и заливаю файлы (~30 МБ)"
  ssh_ "mkdir -p $REMOTE_DIR"
  scp "${SCP_OPTS[@]}" "${FILES[@]}" "$HOST:$REMOTE_DIR/"

  # пробрасываем HF-переменные и SKIP_VENV на сервер, если заданы
  local env_prefix=""
  [ -n "${HF_TOKEN:-}" ] && env_prefix+="HF_TOKEN=${HF_TOKEN} "
  [ -n "${HF_REPO:-}" ]  && env_prefix+="HF_REPO=${HF_REPO} "
  [ -n "${SKIP_VENV:-}" ] && env_prefix+="SKIP_VENV=${SKIP_VENV} "

  echo "==> Запускаю обучение в ФОНЕ (nohup) на сервере"
  ssh_ "cd $REMOTE_DIR && chmod +x run_train.sh && \
        nohup env ${env_prefix}./run_train.sh > run.console.log 2>&1 & \
        echo \$! > run.pid; sleep 1; echo 'PID:' \$(cat run.pid)"

  echo ""
  echo "Обучение пошло в фоне. Подключаюсь к логам (Ctrl-C — отключиться от лога,"
  echo "обучение ПРОДОЛЖИТСЯ; вернуться: ./deploy_train.sh logs)"
  echo "-----------------------------------------------------------------------"
  sleep 2
  ssh_ "tail -f $REMOTE_DIR/run.console.log"
}

cmd_logs() {
  ssh_ "tail -n 80 -f $REMOTE_DIR/run.console.log"
}

cmd_status() {
  ssh_ "echo '== GPU =='; nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader; \
        echo; echo '== Процесс обучения =='; \
        if [ -f $REMOTE_DIR/run.pid ] && kill -0 \$(cat $REMOTE_DIR/run.pid) 2>/dev/null; then \
          echo 'РАБОТАЕТ, PID' \$(cat $REMOTE_DIR/run.pid); \
        else echo 'не запущен / завершён'; fi; \
        echo; echo '== Последние строки лога =='; tail -n 15 $REMOTE_DIR/run.console.log 2>/dev/null || true"
}

cmd_pull() {
  echo "==> Скачиваю адаптер с сервера в ./qwen-mif-lora"
  scp "${SCP_OPTS[@]}" -r "$HOST:$REMOTE_DIR/qwen-mif-lora" .
  echo "Готово: ./qwen-mif-lora"
}

cmd_shell() {
  ssh_
}

case "${1:-up}" in
  up)     cmd_up ;;
  logs)   cmd_logs ;;
  status) cmd_status ;;
  pull)   cmd_pull ;;
  shell)  cmd_shell ;;
  *) echo "Использование: $0 {up|logs|status|pull|shell}"; exit 1 ;;
esac
