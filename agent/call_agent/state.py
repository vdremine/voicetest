from __future__ import annotations

from typing import Any, TypedDict


class CallState(TypedDict, total=False):
    session_id: str
    phone: str

    current_node: str
    return_to_node: str | None
    last_turn_note: str

    raw_text: str
    user_text: str

    known_facts: dict[str, Any]
    history: list[dict[str, str]]

    node_repeat_count: dict[str, int]
    recent_acks: list[str]
    last_named: bool

    reply: str
    llm_decision: dict[str, Any]

    trace: dict[str, Any]
