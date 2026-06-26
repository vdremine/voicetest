from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any

import httpx
from openai import AsyncOpenAI
from pydantic import ValidationError

from .prompt import SYSTEM_PROMPT, build_turn_prompt
from .schema import UNDERSTANDING_JSON_SCHEMA, TurnUnderstanding


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _repair_json(snippet: str) -> str:
    """Fix the malformed JSON weak local models emit without guided decoding:
    bareword (unquoted) keys and trailing commas. Observed on the server, e.g.
    {"reflection":"...",answer:"",branch_signal:"none"}."""
    fixed = re.sub(r'([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)', r'\1"\2"\3', snippet)
    fixed = re.sub(r',(\s*[}\]])', r'\1', fixed)
    return fixed


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
        snippet = value[start : end + 1]
        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            pass
        try:
            return json.loads(_repair_json(snippet))
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


def _truthy_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class LlmSettings:
    model: str
    base_url: str
    api_key: str
    temperature: float
    max_tokens: int
    timeout_seconds: float
    guided_json: bool

    @classmethod
    def from_env(cls) -> "LlmSettings":
        return cls(
            model=os.getenv("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct").strip() or "Qwen/Qwen2.5-7B-Instruct",
            base_url=os.getenv("LLM_BASE_URL", "http://127.0.0.1:8001/v1").strip() or "http://127.0.0.1:8001/v1",
            api_key=os.getenv("LLM_API_KEY", "local-token").strip() or "local-token",
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "320")),
            timeout_seconds=float(os.getenv("LLM_TIMEOUT_SECONDS", "15")),
            # vLLM supports guided decoding; on by default. Set 0 for backends
            # that don't (the code then falls back to plain json_object mode).
            guided_json=_truthy_env("LLM_GUIDED_JSON", True),
        )


@dataclass(frozen=True, slots=True)
class UnderstandResult:
    source: str
    understanding: TurnUnderstanding | None
    raw_output: str
    parse_error: str
    latency_ms: int


class TurnLlmClient:
    def __init__(self, settings: LlmSettings) -> None:
        self._settings = settings
        self._guided_json = settings.guided_json
        self._client = AsyncOpenAI(
            base_url=settings.base_url,
            api_key=settings.api_key,
            timeout=httpx.Timeout(settings.timeout_seconds),
        )

    @property
    def model_name(self) -> str:
        return self._settings.model

    async def understand(
        self,
        *,
        focus_node: str,
        known_facts: dict[str, Any],
        history: list[dict[str, str]],
        last_turn_note: str,
        user_text: str,
    ) -> UnderstandResult:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "system",
                "content": build_turn_prompt(
                    focus_node=focus_node,
                    known_facts=known_facts,
                    history=history[-4:],
                    last_turn_note=last_turn_note,
                    user_text=user_text,
                ),
            },
        ]
        return await self._run_messages(messages=messages, source="main")

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

    async def _create_completion(self, messages: list[dict[str, str]], *, guided: bool):
        kwargs: dict[str, Any] = {
            "model": self._settings.model,
            "messages": messages,
            "temperature": self._settings.temperature,
            "max_tokens": self._settings.max_tokens,
        }
        if guided:
            # vLLM >=0.23 убрал extra_body={"guided_json": ...} (молча игнорирует,
            # возвращая свободный текст). Структурный вывод теперь через
            # response_format json_schema — xgrammar принуждает схему TurnUnderstanding.
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "turn_understanding",
                    "schema": UNDERSTANDING_JSON_SCHEMA,
                },
            }
        else:
            # фолбэк для бэкендов без json_schema: «любой валидный JSON».
            kwargs["response_format"] = {"type": "json_object"}
        return await self._client.chat.completions.create(**kwargs)

    async def _run_messages(self, *, messages: list[dict[str, str]], source: str) -> UnderstandResult:
        started_at = time.perf_counter()
        raw_output = ""
        parse_error = ""

        # 1) guided JSON (if enabled), 2) plain json_object, 3) bare request.
        attempts = [("guided", True), ("json_object", False)] if self._guided_json else [("json_object", False)]
        for label, guided in attempts:
            try:
                completion = await self._create_completion(messages, guided=guided)
                raw_output = _coerce_message_content(completion.choices[0].message.content)
                parse_error = ""
                break
            except Exception as exc:
                parse_error = f"{label}_request_failed: {exc}"
                # If guided decoding is unsupported by the backend, stop trying it.
                if guided:
                    self._guided_json = False
                raw_output = ""

        understanding, validation_error = self._parse_understanding(raw_output)
        latency_ms = int((time.perf_counter() - started_at) * 1000)
        return UnderstandResult(
            source=source,
            understanding=understanding,
            raw_output=raw_output,
            parse_error=parse_error or validation_error,
            latency_ms=latency_ms,
        )

    def _parse_understanding(self, raw_output: str) -> tuple[TurnUnderstanding | None, str]:
        if not _clean_text(raw_output):
            return None, "empty_llm_output"
        payload = _extract_json_object(raw_output)
        if not payload:
            return None, "json_not_found_in_output"
        try:
            return TurnUnderstanding.model_validate(payload), ""
        except ValidationError as exc:
            return None, f"schema_validation_failed: {exc}"
