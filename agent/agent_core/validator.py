from __future__ import annotations

from typing import Any

from .rag import KnowledgeSnippet


def validate_llm_reply(
    *,
    reply_tts: str,
    fallback_reply: str,
    state: dict[str, Any],
    knowledge: list[KnowledgeSnippet],
    truth_rules: tuple[str, ...] = (),
) -> str:
    text = reply_tts.strip()
    if not text:
        return fallback_reply

    lowered = text.lower()
    banned = (
        "клиент сказал",
        "клиент написал",
        "пользователь сказал",
        "разберемся с этим запросом",
        "я как модель",
    )
    if any(marker in lowered for marker in banned):
        return fallback_reply

    if len(text) > 280:
        return fallback_reply

    state_summary = str(state.get("summary", "")).lower()
    if "нет залога" in state_summary and ("птс" in lowered or "залог автомобиля" in lowered):
        return fallback_reply

    if "покупка автомобиля" in state_summary and "птс" in lowered:
        return fallback_reply

    if truth_rules:
        if "стопроцент" in lowered or "гарантир" in lowered:
            return fallback_reply

    return text
