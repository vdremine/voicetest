"""End-to-end orchestration tests for the runner, with a fake LLM (no network)."""

from __future__ import annotations

import asyncio

from call_agent.graph import make_call_graph
from call_agent.llm import UnderstandResult
from call_agent.metrics import MetricsCollector
from call_agent.schema import TurnUnderstanding


class FakeLlm:
    def __init__(self, result: UnderstandResult) -> None:
        self._result = result
        self.model_name = "fake"

    async def understand(self, **_kwargs) -> UnderstandResult:
        return self._result


def _run(runner, state):
    return asyncio.run(runner.ainvoke(state))


def _state(facts: dict, user_text: str, current_node: str = "in_call") -> dict:
    return {
        "session_id": "s1", "phone": "123", "current_node": current_node,
        "return_to_node": None, "last_turn_note": "", "raw_text": user_text,
        "user_text": user_text, "known_facts": dict(facts), "history": [],
        "node_repeat_count": {}, "reply": "", "llm_decision": {}, "trace": {},
    }


def _runner(understanding, *, parse_error=""):
    result = UnderstandResult(
        source="main", understanding=understanding, raw_output="{}",
        parse_error=parse_error, latency_ms=1,
    )
    return make_call_graph(llm_client=FakeLlm(result), metrics=MetricsCollector(model_name="t"))


def test_amount_after_opening_jumps_to_name():
    u = TurnUnderstanding(reflection="Двести тысяч, понял.", facts_update={"desired_amount": "200000"})
    out = _run(_runner(u), _state({"opening_done": "yes"}, "да мне нужно 200 тысяч"))
    assert out["current_node"] == "collect_name"
    assert out["known_facts"]["desired_amount"] == "200000"
    assert out["reply"].startswith("Двести тысяч, понял.")
    assert out["reply"].rstrip().endswith("?")


def test_property_plus_region_jumps_to_encumbrance():
    u = TurnUnderstanding(
        reflection="Квартира в Саратове, понял.",
        facts_update={"property_type": "квартира", "region": "Саратов"},
    )
    facts = {"opening_done": "yes", "desired_amount": "1", "client_name": "Иван"}
    out = _run(_runner(u), _state(facts, "квартира в саратове"))
    assert out["current_node"] == "collect_encumbrance"
    assert out["known_facts"]["property_type"] == "квартира"
    assert out["known_facts"]["region"] == "Саратов"


def test_invalid_output_does_not_ask_to_repeat():
    facts = {"opening_done": "yes", "desired_amount": "1", "client_name": "Иван", "property_type": "квартира"}
    out = _run(_runner(None, parse_error="bad"), _state(facts, "в саратове"))
    assert "повторите" not in out["reply"].lower()
    assert out["reply"].rstrip().endswith("?")
    assert out["current_node"] == "collect_region"


def test_branch_signal_vehicle_routes_to_vehicle_type():
    u = TurnUnderstanding(
        reflection="Понял, недвижимости нет.",
        facts_update={"no_real_estate": "yes"},
        branch_signal="vehicle",
    )
    facts = {"opening_done": "yes", "desired_amount": "1", "client_name": "Иван"}
    out = _run(_runner(u), _state(facts, "недвижимости нет, машина есть"))
    assert out["current_node"] == "collect_vehicle_type"


def test_consolidation_signal_sets_intent():
    u = TurnUnderstanding(
        reflection="Понял, хотите объединить.",
        facts_update={},
        branch_signal="consolidation",
    )
    facts = {
        "opening_done": "yes", "desired_amount": "3000000", "client_name": "Эдуард",
        "property_type": "квартира", "region": "Москва", "encumbrance": "ипотека",
        "encumbrance_details": "8 млн", "owner_status": "я",
    }
    out = _run(_runner(u), _state(facts, "хочу объединить кредиты"))
    assert out["known_facts"]["consolidation_intent"] == "yes"
    assert out["current_node"] == "collect_consolidation_summary"


def test_name_deferred_when_client_objects_instead_of_answering():
    # focus is collect_name; client objects instead of giving a name
    u = TurnUnderstanding(
        reflection="понимаю.",
        answer="Мы не банк, а кредитный брокер.",
        facts_update={},
    )
    facts = {"opening_done": "yes", "desired_amount": "1"}
    out = _run(_runner(u), _state(facts, "а вы вообще какой банк?"))
    assert out["known_facts"].get("name_deferred") == "yes"
    assert out["current_node"] == "collect_property_type"  # moved past name


def test_anti_loop_captures_encumbrance_on_first_miss():
    # Reproduces the server bug: client says "не в залоге" but the model never
    # emits the `encumbrance` fact. A yes/no slot must capture on the FIRST miss
    # (no redundant re-ask) and advance.
    u = TurnUnderstanding(reflection="понял вас.", facts_update={})  # never fills encumbrance
    facts = {
        "opening_done": "yes", "desired_amount": "1", "client_name": "Магомед",
        "property_type": "дом", "region": "Чечня",
    }
    out = _run(_runner(u), _state(facts, "не в залоге"))
    assert out["current_node"] != "collect_encumbrance"  # advanced immediately
    assert out["known_facts"].get("encumbrance") == "нет"  # inferred from "не в залоге"


def test_anti_loop_does_not_capture_when_client_asks_back():
    # If the client asks a question instead of answering, don't force-capture.
    u = TurnUnderstanding(reflection="понимаю.", answer="Залог — это когда объект уже заложен в банке.", facts_update={})
    facts = {
        "opening_done": "yes", "desired_amount": "1", "client_name": "Магомед",
        "property_type": "дом", "region": "Чечня",
    }
    out = _run(_runner(u), _state(facts, "а что значит в залоге?"))
    assert out["current_node"] == "collect_encumbrance"  # stays, re-asks after answering
    assert not str(out["known_facts"].get("encumbrance", "")).strip()


def _tail_facts():
    return {
        "opening_done": "yes", "desired_amount": "1", "client_name": "Иван",
        "property_type": "квартира", "region": "Москва", "encumbrance": "нет",
        "owner_status": "я", "pitched": "yes", "priority": "ставка",
    }


def test_callback_time_captured_at_consent_step_no_reask():
    # "да, набирайте завтра" = consent + time together. Must NOT re-ask the time.
    u = TurnUnderstanding(reflection="завтра, хорошо.", facts_update={"callback_consent": "да"})
    out = _run(_runner(u), _state(_tail_facts(), "да, набирайте завтра"))
    assert str(out["known_facts"].get("callback_time", "")).strip()  # time captured
    assert out["current_node"] == "finish"
    low = out["reply"].lower()
    assert "когда удобнее" not in low and "сегодня, завтра" not in low  # not re-asked


def test_finish_reply_keeps_reflection():
    u = TurnUnderstanding(reflection="завтра после обеда, зафиксировал.", facts_update={"callback_time": "завтра после обеда"})
    facts = {**_tail_facts(), "callback_consent": "да"}
    out = _run(_runner(u), _state(facts, "завтра после обеда"))
    assert out["current_node"] == "finish"
    assert "зафиксировал" in out["reply"].lower()  # reflection not dropped
    assert "доброго" in out["reply"].lower()        # plus a clean close


def test_should_end_finishes():
    u = TurnUnderstanding(reflection="", should_end=True)
    out = _run(_runner(u), _state({"opening_done": "yes", "desired_amount": "1"}, "не интересно, спасибо"))
    assert out["current_node"] == "finish"
    assert "?" not in out["reply"]
