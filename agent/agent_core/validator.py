from __future__ import annotations

from typing import Any

from .rag import KnowledgeSnippet


def inspect_llm_reply(
    *,
    reply_tts: str,
    fallback_reply: str,
    state: dict[str, Any],
    knowledge: list[KnowledgeSnippet],
    truth_rules: tuple[str, ...] = (),
) -> tuple[str, str]:
    text = reply_tts.strip()
    if not text:
        return fallback_reply, "empty_reply"

    lowered = text.lower()
    banned = (
        "клиент сказал",
        "клиент написал",
        "пользователь сказал",
        "разберемся с этим запросом",
        "я как модель",
    )
    if any(marker in lowered for marker in banned):
        return fallback_reply, "banned_meta_phrase"

    if len(text) > 280:
        return fallback_reply, "reply_too_long"

    state_summary = str(state.get("summary", "")).lower()
    if "нет залога" in state_summary and ("птс" in lowered or "залог автомобиля" in lowered):
        return fallback_reply, "contradicts_no_collateral_state"

    if "покупка автомобиля" in state_summary and "птс" in lowered:
        return fallback_reply, "contradicts_auto_goal_state"

    if truth_rules:
        if "стопроцент" in lowered or "гарантир" in lowered:
            return fallback_reply, "forbidden_guarantee_claim"

    return text, "accepted"


def validate_llm_reply(
    *,
    reply_tts: str,
    fallback_reply: str,
    state: dict[str, Any],
    knowledge: list[KnowledgeSnippet],
    truth_rules: tuple[str, ...] = (),
) -> str:
    validated_text, _ = inspect_llm_reply(
        reply_tts=reply_tts,
        fallback_reply=fallback_reply,
        state=state,
        knowledge=knowledge,
        truth_rules=truth_rules,
    )
    return validated_text
