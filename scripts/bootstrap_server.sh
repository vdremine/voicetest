#!/bin/bash
set -e

export DEBIAN_FRONTEND=noninteractive

SERVER_TIMEZONE="${SERVER_TIMEZONE:-Europe/Moscow}"

apt-get update
apt-get install -y git curl wget htop tmux unzip ca-certificates gnupg jq iproute2 gettext-base

if ! command -v docker >/dev/null 2>&1; then
  curl -fsSL https://get.docker.com | sh
fi

apt-get install -y docker-compose-plugin || true

timedatectl set-timezone "${SERVER_TIMEZONE}" || true

systemctl enable docker
systemctl start docker

mkdir -p /opt/voice-agent
mkdir -p /opt/models
mkdir -p /opt/logs

echo "Bootstrap completed"
