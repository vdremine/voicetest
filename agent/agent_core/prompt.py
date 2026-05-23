from __future__ import annotations

import json
from typing import Any

from .knowledge_base import TrainingExample
from .rag import KnowledgeSnippet


def build_context_messages(
    *,
    state: dict[str, Any],
    knowledge: list[KnowledgeSnippet],
    truth_rules: tuple[str, ...] = (),
    examples: list[list[dict[str, str]]] | None = None,
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "Текущее состояние диалога. "
                "Не игнорируй его и не сбрасывай сценарий без причины.\n"
                f"{json.dumps(state, ensure_ascii=False)}"
            ),
        }
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
