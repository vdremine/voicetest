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
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.08")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "320")),
            timeout_seconds=float(os.getenv("LLM_TIMEOUT_SECONDS", "15")),
        )


@dataclass(frozen=True, slots=True)
class LlmCallResult:
    source: str
    decision: LlmTurnDecision | None
    raw_output: str
    parse_error: str
    latency_ms: int


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
        last_turn_note: str,
    ) -> LlmCallResult:
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
                    last_turn_note=last_turn_note,
                    user_text=user_text,
                ),
            },
        ]
        return await self._run_messages(
            messages=messages,
            temperature=self._settings.temperature,
            source="main",
        )

    async def repair_json_llm(
        self,
        *,
        current_node: str,
        user_text: str,
        known_facts: dict[str, Any],
        history: list[dict[str, str]],
        node_repeat_count: int,
        last_turn_note: str,
        raw_output: str,
        parse_error: str,
    ) -> LlmCallResult:
        node = DIALOGUE_GRAPH[current_node]
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "system",
                "content": (
                    "Ты не ведешь новый разговор. Ты исправляешь невалидный JSON-ответ того же агента.\n"
                    "Сохрани исходный смысл ответа максимально близко.\n"
                    "Верни только валидный JSON по схеме.\n"
                    "Не придумывай новые факты от себя."
                ),
            },
            {
                "role": "system",
                "content": build_node_prompt(
                    node=node,
                    current_node_id=current_node,
                    last_messages=history[-4:],
                    known_facts=known_facts,
                    node_repeat_count=node_repeat_count,
                    last_turn_note=last_turn_note,
                    user_text=user_text,
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Текущий узел: {current_node}\n"
                    f"Ошибка парсинга/валидации: {parse_error}\n"
                    f"Сырой вывод модели:\n{raw_output or '<empty>'}\n"
                    "Преобразуй это в валидный JSON по схеме, не меняя смысл без необходимости."
                ),
            },
        ]
        return await self._run_messages(
            messages=messages,
            temperature=0.0,
            source="json_repair",
        )

    async def repair_transition_llm(
        self,
        *,
        current_node: str,
        user_text: str,
        known_facts: dict[str, Any],
        history: list[dict[str, str]],
        node_repeat_count: int,
        last_turn_note: str,
        bad_decision: dict[str, Any],
    ) -> LlmCallResult:
        node = DIALOGUE_GRAPH[current_node]
        allowed_next = ", ".join(node.allowed_next) if node.allowed_next else "нет"
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "system",
                "content": (
                    "Ты исправляешь только структурную ошибку перехода.\n"
                    "Нельзя выбирать next_node вне allowed_next.\n"
                    "Если текущий узел не завершен, поставь node_complete=false и next_node=null.\n"
                    "Если узел завершен, выбери next_node только из allowed_next.\n"
                    "Смысл reply сохраняй максимально близко."
                ),
            },
            {
                "role": "system",
                "content": build_node_prompt(
                    node=node,
                    current_node_id=current_node,
                    last_messages=history[-4:],
                    known_facts=known_facts,
                    node_repeat_count=node_repeat_count,
                    last_turn_note=last_turn_note,
                    user_text=user_text,
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Текущий узел: {current_node}\n"
                    f"allowed_next: {allowed_next}\n"
                    f"Текущий JSON: {json.dumps(bad_decision, ensure_ascii=False)}\n"
                    "Верни исправленный JSON. next_node должен быть только из allowed_next или null."
                ),
            },
        ]
        return await self._run_messages(
            messages=messages,
            temperature=0.0,
            source="transition_repair",
        )

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

    async def _run_messages(
        self,
        *,
        messages: list[dict[str, str]],
        temperature: float,
        source: str,
    ) -> LlmCallResult:
        started_at = time.perf_counter()
        raw_output = ""
        parse_error = ""

        try:
            completion = await self._client.chat.completions.create(
                model=self._settings.model,
                messages=messages,
                temperature=temperature,
                max_tokens=self._settings.max_tokens,
                response_format={"type": "json_object"},
            )
            raw_output = _coerce_message_content(completion.choices[0].message.content)
        except Exception as exc_json_mode:
            parse_error = f"json_mode_request_failed: {exc_json_mode}"
            try:
                completion = await self._client.chat.completions.create(
                    model=self._settings.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=self._settings.max_tokens,
                )
                raw_output = _coerce_message_content(completion.choices[0].message.content)
            except Exception as exc_plain:
                latency_ms = int((time.perf_counter() - started_at) * 1000)
                combined = f"{parse_error}; plain_request_failed: {exc_plain}"
                return LlmCallResult(
                    source=source,
                    decision=None,
                    raw_output=raw_output,
                    parse_error=combined,
                    latency_ms=latency_ms,
                )

        decision, validation_error = self._parse_decision(raw_output)
        latency_ms = int((time.perf_counter() - started_at) * 1000)
        return LlmCallResult(
            source=source,
            decision=decision,
            raw_output=raw_output,
            parse_error=parse_error or validation_error,
            latency_ms=latency_ms,
        )

    def _parse_decision(self, raw_output: str) -> tuple[LlmTurnDecision | None, str]:
        if not _clean_text(raw_output):
            return None, "empty_llm_output"

        payload = _extract_json_object(raw_output)
        if not payload:
            return None, "json_not_found_in_output"

        try:
            return LlmTurnDecision.model_validate(payload), ""
        except ValidationError as exc:
            return None, f"schema_validation_failed: {exc}"
