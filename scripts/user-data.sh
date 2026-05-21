#!/bin/bash
set -e

export DEBIAN_FRONTEND=noninteractive
SERVER_TIMEZONE="${SERVER_TIMEZONE:-Europe/Moscow}"
REPO_URL="${REPO_URL:-https://github.com/your-org/livekit-transport-scaffold.git}"
APP_DIR="${APP_DIR:-/opt/voice-agent}"

timedatectl set-timezone "${SERVER_TIMEZONE}" || true

apt-get update
apt-get install -y git curl

if [ ! -d "${APP_DIR}" ]; then
  git clone "${REPO_URL}" "${APP_DIR}"
fi

cd "${APP_DIR}"
chmod +x scripts/bootstrap_server.sh
SERVER_TIMEZONE="${SERVER_TIMEZONE}" ./scripts/bootstrap_server.sh

if [ ! -f .env ]; then
  cp .env.prod.example .env
fi

docker compose -f docker-compose.prod.yml up --build -d
