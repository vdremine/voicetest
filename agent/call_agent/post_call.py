from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field

from .llm import TurnLlmClient


class PostCallSummary(BaseModel):
    status: str = "unknown"
    client_name: str = ""
    lead_destination: str = "unknown"
    agreements: str = ""
    agreements_time: str = ""
    client_facts: str = ""
    lead_quality: int = 0
    smsText: str = ""


def _history_text(history: list[dict[str, str]]) -> str:
    return "\n".join(
        f"{item.get('role', 'user')}: {item.get('content', '')}"
        for item in history
        if str(item.get("content", "")).strip()
    )


async def extract_post_call_summary(
    *,
    llm_client: TurnLlmClient,
    history: list[dict[str, str]],
    known_facts: dict[str, Any],
) -> PostCallSummary:
    system_prompt = """
Ты извлекаешь итог звонка по истории разговора.
Верни только JSON. Не придумывай факты.
Если данных не хватает, оставляй пустую строку или unknown.
""".strip()
    user_prompt = f"""
Верни JSON с итогом звонка:
{{
  "status": "transfer | callback | rejected | not_target | no_answer | unknown",
  "client_name": "...",
  "lead_destination": "sales | pts | partner | refinance | unknown",
  "agreements": "...",
  "agreements_time": "...",
  "client_facts": "...",
  "lead_quality": 0,
  "smsText": "..."
}}

Известные факты:
{json.dumps(known_facts, ensure_ascii=False)}

История разговора:
{_history_text(history)}
""".strip()
    payload = await llm_client.call_json_prompt(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )
    try:
        return PostCallSummary.model_validate(payload)
    except Exception:
        return PostCallSummary(
            status="unknown",
            client_name=str(known_facts.get("client_name", "")),
            client_facts=json.dumps(known_facts, ensure_ascii=False),
        )
