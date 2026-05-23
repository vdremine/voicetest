#!/bin/bash
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive

SERVER_TIMEZONE="${SERVER_TIMEZONE:-Europe/Moscow}"
APP_ROOT="${APP_ROOT:-/opt/voicetest}"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-/opt/models}"
VOICE_AGENT_DATA_DIR="${VOICE_AGENT_DATA_DIR:-/opt/voice-agent-data}"
INSTALL_NVIDIA_DRIVER="${INSTALL_NVIDIA_DRIVER:-true}"
INSTALL_NVIDIA_TOOLKIT="${INSTALL_NVIDIA_TOOLKIT:-true}"
REBOOT_AFTER_BOOTSTRAP="${REBOOT_AFTER_BOOTSTRAP:-false}"
NEEDS_REBOOT=0

if [ "$(id -u)" -ne 0 ]; then
  echo "Run this script as root" >&2
  exit 1
fi

apt-get update
apt-get install -y \
  git \
  curl \
  wget \
  htop \
  tmux \
  unzip \
  ca-certificates \
  gnupg \
  jq \
  iproute2 \
  gettext-base \
  software-properties-common \
  apt-transport-https \
  ubuntu-drivers-common

if ! command -v docker >/dev/null 2>&1; then
  curl -fsSL https://get.docker.com | sh
fi

apt-get install -y docker-compose-plugin || true

timedatectl set-timezone "${SERVER_TIMEZONE}" || true

systemctl enable --now docker

if [ "${INSTALL_NVIDIA_DRIVER}" = "true" ] && ! command -v nvidia-smi >/dev/null 2>&1; then
  ubuntu-drivers autoinstall
  NEEDS_REBOOT=1
fi

if [ "${INSTALL_NVIDIA_TOOLKIT}" = "true" ]; then
  if [ ! -f /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg ]; then
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
      | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  fi

  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    > /etc/apt/sources.list.d/nvidia-container-toolkit.list

  apt-get update
  apt-get install -y nvidia-container-toolkit
  nvidia-ctk runtime configure --runtime=docker
  systemctl restart docker
fi

mkdir -p "${APP_ROOT}"
mkdir -p "${MODEL_CACHE_DIR}"
mkdir -p "${VOICE_AGENT_DATA_DIR}"

cat <<EOF
Bootstrap completed.

Next commands after OS bootstrap:
1. Reboot the host if NVIDIA driver was just installed:
   reboot

2. Verify host GPU:
   nvidia-smi

3. Verify Docker GPU runtime:
   docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi

4. Deploy the app:
   cd ${APP_ROOT}
   git clone <your_repo_url> .
   cp .env.prod.example .env
   docker compose -f docker-compose.prod.yml -f docker-compose.llm.yml up --build -d

5. Tail agent logs:
   docker compose -f docker-compose.prod.yml -f docker-compose.llm.yml logs -f agent
EOF

if [ "${REBOOT_AFTER_BOOTSTRAP}" = "true" ] && [ "${NEEDS_REBOOT}" -eq 1 ]; then
  reboot
fi
