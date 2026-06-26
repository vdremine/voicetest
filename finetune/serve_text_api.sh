#!/usr/bin/env bash
set -euo pipefail

# serve_text_api.sh — поднять диалоговый движок call_agent (text_api) на :8787,
# который ходит в дообученную модель через vLLM (:8001).
# Запускать на том же боксе, что и vLLM. Нужен код agent/call_agent рядом.
#
# Зависит от: fastapi, uvicorn, openai, httpx, pydantic (GPU НЕ нужен).

AGENT_DIR=${AGENT_DIR:-/root/mif-train/agent}     # где лежит пакет call_agent/
PYBIN=${PYBIN:-/root/mif-train/.venv-train/bin/python}

export LLM_BASE_URL=${LLM_BASE_URL:-http://127.0.0.1:8001/v1}
export LLM_API_KEY=${LLM_API_KEY:-local-token}
export LLM_MODEL=${LLM_MODEL:-mif}                # имя модели в vLLM (--served-model-name)
# После фикса call_agent/llm.py guided использует response_format json_schema,
# который vLLM принуждает. Можно держать включённым (=1). =0 → режим json_object.
export LLM_GUIDED_JSON=${LLM_GUIDED_JSON:-1}
export LLM_TEMPERATURE=${LLM_TEMPERATURE:-0.2}
export LLM_MAX_TOKENS=${LLM_MAX_TOKENS:-320}
export LLM_TIMEOUT_SECONDS=${LLM_TIMEOUT_SECONDS:-60}
export TOOL_GRAPH_PATH=${TOOL_GRAPH_PATH:-/root/mif-train/voice_agent_graph_v0_1/tool_graph.json}

cd "$AGENT_DIR"
echo "text_api: LLM=$LLM_MODEL @ $LLM_BASE_URL  guided=$LLM_GUIDED_JSON"
exec "$PYBIN" -m uvicorn call_agent.api:app --host 127.0.0.1 --port 8787
