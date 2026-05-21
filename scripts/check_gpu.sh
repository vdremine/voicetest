#!/bin/bash
set -e

echo "=== NVIDIA ==="
nvidia-smi || true

echo "=== Docker ==="
docker --version || true

echo "=== Disk ==="
df -h

echo "=== RAM ==="
free -h
