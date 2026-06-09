"""Full HTTP-path integration test replaying the user's reported failures.

Only the LLM's extraction is stubbed (scripted per utterance). Routing, fact
application, navigation and reply assembly are the real code.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from call_agent import api
from call_agent.llm import UnderstandResult
from call_agent.schema import TurnUnderstanding


def _u(reflection="", facts=None, answer="", branch="none", end=False):
    return UnderstandResult(
        source="main",
        understanding=TurnUnderstanding(
            reflection=reflection, facts_update=facts or {}, answer=answer,
            branch_signal=branch, should_end=end,
        ),
        raw_output="{}", parse_error="", latency_ms=1,
    )


SCRIPT = {
    "да мне нужно 200 тысяч": _u("Двести тысяч, понял.", {"desired_amount": "200000"}),
    "владимир": _u("Владимир, очень приятно.", {"client_name": "Владимир"}),
    "квартира в саратове": _u("Квартира в Саратове, понял.", {"property_type": "квартира", "region": "Саратов"}),
}


async def _scripted_understand(**kwargs):
    return SCRIPT.get(kwargs.get("user_text", "").strip().lower(), _u("Понял.", {}))


def test_full_conversation_replays_reported_bugs(monkeypatch):
    monkeypatch.setattr(api._call_graph._llm_client, "understand", _scripted_understand)
    client = TestClient(api.app)

    client.post("/session/reset", json={"session_id": "itest"})
    client.post("/session/start", json={"session_id": "itest", "phone": "70000000000"})

    # Handshake: client picks up -> agent delivers the opening that asks the amount.
    intro = client.post("/session/message", json={"session_id": "itest", "text": "да, алло"}).json()
    assert "сумму" in intro["reply"].lower()
    assert intro["current_node"] == "collect_amount"

    # Turn 1: amount -> jump to NAME, do NOT re-ask the amount.
    r1 = client.post("/session/message", json={"session_id": "itest", "text": "да мне нужно 200 тысяч"}).json()
    assert r1["current_node"] == "collect_name"
    assert r1["known_facts"]["desired_amount"] == "200000"
    assert r1["reply"].startswith("Двести тысяч, понял.")

    # Turn 2: client literally named Владимир (same as agent) -> kept.
    r2 = client.post("/session/message", json={"session_id": "itest", "text": "Владимир"}).json()
    assert r2["known_facts"]["client_name"] == "Владимир"
    assert r2["current_node"] == "collect_property_type"

    # Turn 3: "квартира в саратове" -> BOTH facts, jump over region to encumbrance.
    r3 = client.post("/session/message", json={"session_id": "itest", "text": "квартира в саратове"}).json()
    assert r3["known_facts"]["property_type"] == "квартира"
    assert r3["known_facts"]["region"] == "Саратов"
    assert r3["current_node"] == "collect_encumbrance"
    low = r3["reply"].lower()
    assert "залог" in low or "обремен" in low
    assert "район" not in low and "адрес" not in low


def test_refi_mode_opens_with_remainder_question(monkeypatch):
    monkeypatch.setattr(api._call_graph._llm_client, "understand", _scripted_understand)
    client = TestClient(api.app)
    client.post("/session/reset", json={"session_id": "refi"})
    client.post("/session/start", json={
        "session_id": "refi", "phone": "70000000001", "known_facts": {"refi_mode": "yes"},
    })
    intro = client.post("/session/message", json={"session_id": "refi", "text": "да, слушаю"}).json()
    low = intro["reply"].lower()
    assert "осталось" in low or "выплат" in low
    assert "мосинвестфинанс" in low
