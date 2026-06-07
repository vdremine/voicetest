from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any

import httpx
from openai import AsyncOpenAI
from pydantic import ValidationError

from .graph_spec import DIALOGUE_GRAPH
from .prompt import SYSTEM_PROMPT, build_node_prompt
from .schema import LlmTurnDecision


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _extract_json_object(text: str) -> dict[str, Any]:
    value = _clean_text(text)
    if not value:
        return {}
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        pass

    start = value.find("{")
    end = value.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(value[start : end + 1])
        except json.JSONDecodeError:
            return {}
    return {}


@dataclass(frozen=True, slots=True)
class LlmSettings:
    model: str
    base_url: str
    api_key: str
    temperature: float
    max_tokens: int
    timeout_seconds: float

    @classmethod
    def from_env(cls) -> "LlmSettings":
        return cls(
            model=os.getenv("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct").strip() or "Qwen/Qwen2.5-7B-Instruct",
            base_url=os.getenv("LLM_BASE_URL", "http://127.0.0.1:8001/v1").strip() or "http://127.0.0.1:8001/v1",
            api_key=os.getenv("LLM_API_KEY", "local-token").strip() or "local-token",
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.22")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "260")),
            timeout_seconds=float(os.getenv("LLM_TIMEOUT_SECONDS", "15")),
        )


class TurnLlmClient:
    def __init__(self, settings: LlmSettings) -> None:
        self._settings = settings
        self._client = AsyncOpenAI(
            base_url=settings.base_url,
            api_key=settings.api_key,
            timeout=httpx.Timeout(settings.timeout_seconds),
        )

    @property
    def model_name(self) -> str:
        return self._settings.model

    async def call_turn_llm(
        self,
        *,
        current_node: str,
        user_text: str,
        known_facts: dict[str, Any],
        history: list[dict[str, str]],
        node_repeat_count: int,
    ) -> tuple[LlmTurnDecision, int]:
        node = DIALOGUE_GRAPH[current_node]
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "system",
                "content": build_node_prompt(
                    node=node,
                    current_node_id=current_node,
                    last_messages=history[-4:],
                    known_facts=known_facts,
                    node_repeat_count=node_repeat_count,
                    user_text=user_text,
                ),
            },
        ]

        started_at = time.perf_counter()
        raw_content = ""
        try:
            completion = await self._client.chat.completions.create(
                model=self._settings.model,
                messages=messages,
                temperature=self._settings.temperature,
                max_tokens=self._settings.max_tokens,
                response_format={"type": "json_object"},
            )
            raw_content = _coerce_message_content(completion.choices[0].message.content)
            payload = _extract_json_object(raw_content)
            decision = LlmTurnDecision.model_validate(payload)
        except Exception:
            try:
                completion = await self._client.chat.completions.create(
                    model=self._settings.model,
                    messages=messages,
                    temperature=self._settings.temperature,
                    max_tokens=self._settings.max_tokens,
                )
                raw_content = _coerce_message_content(completion.choices[0].message.content)
                decision = LlmTurnDecision.model_validate_json(_clean_text(raw_content))
            except ValidationError:
                decision = self._fallback_decision(current_node=current_node)
            except Exception:
                payload = _extract_json_object(raw_content)
                if payload:
                    try:
                        decision = LlmTurnDecision.model_validate(payload)
                    except ValidationError:
                        decision = self._fallback_decision(current_node=current_node)
                else:
                    decision = self._fallback_decision(current_node=current_node)

        latency_ms = int((time.perf_counter() - started_at) * 1000)
        return decision, latency_ms

    async def call_json_prompt(self, *, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        try:
            completion = await self._client.chat.completions.create(
                model=self._settings.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=self._settings.max_tokens,
                response_format={"type": "json_object"},
            )
            raw_content = _coerce_message_content(completion.choices[0].message.content)
            return _extract_json_object(raw_content)
        except Exception:
            return {}

    def _fallback_decision(self, *, current_node: str) -> LlmTurnDecision:
        if current_node == "call_connected":
            return LlmTurnDecision(
                reply=DIALOGUE_GRAPH["cold_opening"].ask,
                node_complete=True,
                next_node="cold_opening",
                reason="LLM fallback on call_connected",
            )
        node = DIALOGUE_GRAPH[current_node]
        return LlmTurnDecision(
            reply=node.ask,
            node_complete=False,
            next_node=None,
            reason=f"LLM fallback on {current_node}",
        )


def _coerce_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                parts.append(_clean_text(item.get("text")))
            else:
                parts.append(_clean_text(getattr(item, "text", "")))
        return "".join(parts)
    return _clean_text(content)
