from __future__ import annotations

import json
from typing import Any

from .rag import KnowledgeSnippet


def build_context_messages(*, state: dict[str, Any], knowledge: list[KnowledgeSnippet]) -> list[dict[str, str]]:
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
    return messages
