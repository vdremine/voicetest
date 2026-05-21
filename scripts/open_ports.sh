#!/bin/bash
set -e

echo "=== UFW ==="
ufw status verbose || true

echo "=== TCP LISTEN ==="
ss -lnt || true

echo "=== UDP LISTEN ==="
ss -lnu || true

echo "=== DOCKER COMPOSE ==="
docker compose ps || true
