#!/bin/bash
set -e

export DEBIAN_FRONTEND=noninteractive

SERVER_TIMEZONE="${SERVER_TIMEZONE:-Europe/Moscow}"

apt-get update
apt-get install -y git curl wget htop tmux unzip ca-certificates gnupg ufw jq iproute2 gettext-base

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

ufw allow 22/tcp
ufw allow 80/tcp
ufw allow 443/tcp
ufw allow 7880/tcp
ufw allow 7881/tcp
ufw allow 3478/udp
ufw allow 50000:60000/udp
ufw --force enable

echo "Bootstrap completed"
