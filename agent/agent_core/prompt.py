from __future__ import annotations

import json
from typing import Any

from .knowledge_base import TrainingExample
from .rag import KnowledgeSnippet
from .session_memory import render_state_summary


def _compact_state_payload(state: dict[str, Any]) -> dict[str, Any]:
    known_facts = state.get("known_facts", {})
    if not isinstance(known_facts, dict):
        known_facts = {}
    return {
        "stage": str(state.get("stage", "")).strip(),
        "current_node": str(state.get("current_node", "")).strip(),
        "plan_name": str(state.get("plan_name", "")).strip(),
        "awaiting_field": str(state.get("awaiting_field", "")).strip(),
        "next_required_field": str(state.get("next_required_field", "")).strip(),
        "summary": str(state.get("summary", "")).strip(),
        "known_facts": known_facts,
        "session_state": state.get("session_state", {}),
    }


def build_context_messages(
    *,
    state: dict[str, Any],
    knowledge: list[KnowledgeSnippet],
    truth_rules: tuple[str, ...] = (),
    examples: list[list[dict[str, str]]] | None = None,
) -> list[dict[str, str]]:
    state_text = str(state.get("session_state_text", "")).strip() if state else ""
    if not state_text:
        state_text = render_state_summary(state or {})

    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "Структурированное состояние звонка. "
                "Это источник правды по текущему этапу и следующему нужному шагу.\n"
                f"{state_text}"
            ),
        },
        {
            "role": "system",
            "content": (
                "Компактное runtime-состояние для машинной опоры.\n"
                f"{json.dumps(_compact_state_payload(state or {}), ensure_ascii=False)}"
            ),
        },
    ]
    if knowledge:
        knowledge_text = "\n".join(f"- {snippet.text}" for snippet in knowledge)
        messages.append(
            {
                "role": "system",
                "content": (
                    "Факты и правила бизнеса. Используй только их и не придумывай новые условия.\n"
                    f"{knowledge_text}"
                ),
            }
        )
    if truth_rules:
        messages.append(
            {
                "role": "system",
                "content": "Правила достоверности:\n" + "\n".join(f"- {rule}" for rule in truth_rules),
            }
        )
    for example in examples or []:
        messages.append(
            {
                "role": "system",
                "content": "Пример желательного стиля и сценария. Не копируй дословно, но держи структуру.",
            }
        )
        messages.extend(example)
    return messages
