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


def test_question_is_not_captured_as_name():
    # Server bug: "вы кто?" got stored as client_name. A question must never be
    # captured as a slot value.
    u = TurnUnderstanding(reflection="Владимир, компания МосИнвестФинанс.", facts_update={})
    state = _state({"opening_done": "yes", "amount_deferred": "yes"}, "вы кто?")
    state["node_repeat_count"] = {"collect_name": 1}  # already asked once -> anti-loop armed
    out = _run(_runner(u), state)
    assert out["known_facts"].get("client_name", "") != "вы кто?"
    assert not str(out["known_facts"].get("client_name", "")).strip()


def test_generic_ack_not_repeated_across_turns():
    # "угу, понял вас" every turn sounds awful — must not repeat consecutively.
    u = TurnUnderstanding(reflection="угу, понял вас.", facts_update={})
    runner = _runner(u)
    facts = {"opening_done": "yes", "desired_amount": "1", "client_name": "Иван",
             "property_type": "квартира", "region": "Москва"}
    state = _state(facts, "ага")
    out1 = _run(runner, state)
    # first generic ack may pass through
    state2 = {**out1, "user_text": "ну да", "raw_text": "ну да"}
    out2 = _run(runner, state2)
    # second consecutive generic ack must be dropped from the reply
    assert "понял вас" not in out2["reply"].lower()
    assert out2["reply"].rstrip().endswith("?")  # still asks the question


def test_name_appended_ack_is_deduped_and_name_not_every_turn():
    # Server bug: "понял вас, Лен" every turn. Name-appended generic acks must be
    # caught by the dedup, and the name must not appear two turns in a row.
    u = TurnUnderstanding(reflection="понял вас, Лен.", facts_update={})
    runner = _runner(u)
    facts = {"opening_done": "yes", "desired_amount": "1", "client_name": "Лен",
             "property_type": "квартира", "region": "Москва", "encumbrance": "нет"}
    state = _state(facts, "ага")
    state["last_named"] = True  # previous turn already addressed by name
    out1 = _run(runner, state)
    assert "лен" not in out1["reply"].lower().split("?")[0] or "понял вас, лен" not in out1["reply"].lower()
    # second consecutive name+generic ack dropped
    state2 = {**out1, "user_text": "ну да", "raw_text": "ну да"}
    out2 = _run(runner, state2)
    assert "понял вас, лен" not in out2["reply"].lower()


def test_specific_reflection_is_kept():
    # A specific mirror (carries content) is NOT treated as a generic ack.
    u = TurnUnderstanding(reflection="Двести тысяч, понял.", facts_update={"desired_amount": "200000"})
    out = _run(_runner(u), _state({"opening_done": "yes"}, "двести тысяч"))
    assert out["reply"].startswith("Двести тысяч")


def test_noise_quality_holds_node_and_drops_facts():
    # On quality_signal=noise the graph must NOT advance or apply facts.
    u = TurnUnderstanding(
        reflection="кажется, я плохо расслышал",
        facts_update={"property_type": "дом"},  # model leaked a fact on noise
        quality_signal="noise",
    )
    facts = {"opening_done": "yes", "desired_amount": "1", "client_name": "Иван"}
    out = _run(_runner(u), _state(facts, "ха-ха головка хуя"))
    assert out["current_node"] == "collect_property_type"  # stayed (not advanced)
    assert "property_type" not in out["known_facts"]        # fact dropped
    assert "плохо расслышал" in out["reply"].lower()
    assert out["reply"].rstrip().endswith("?")              # re-asks the slot


def test_vehicle_model_alias_advances_branch():
    # The prompt emits vehicle_model; navigation gates on vehicle_type (alias).
    u = TurnUnderstanding(
        reflection="Toyota RAV4, понял.",
        facts_update={"vehicle_model": "Toyota RAV4"},
        branch_signal="vehicle",
    )
    facts = {"opening_done": "yes", "vehicle_interest": "yes", "client_name": "Иван"}
    out = _run(_runner(u), _state(facts, "тойота рав четыре"))
    assert out["known_facts"].get("vehicle_type") == "Toyota RAV4"
    assert out["current_node"] == "collect_vehicle_owner"  # advanced past type


def test_clean_reply_strips_bad_start():
    from call_agent.graph import clean_reply
    assert not clean_reply("Угу, понял вас.").lower().startswith("угу, понял вас")
    assert clean_reply("Двести тысяч, понял.") == "Двести тысяч, понял."


def _finished_state(user_text):
    facts = {"opening_done": "yes", "desired_amount": "1", "client_name": "Иван",
             "property_type": "квартира", "region": "Москва", "encumbrance": "нет",
             "owner_status": "я", "pitched": "yes", "priority": "ставка",
             "callback_consent": "да", "callback_time": "завтра"}
    st = _state(facts, user_text)
    st["current_node"] = "finish"
    st["call_finished"] = True
    return st


def test_post_finish_farewell_is_silent():
    # Repeated "Всего доброго" after finish must NOT replay the finale.
    u = TurnUnderstanding(reflection="", graph_action="ignore", quality_signal="already_finished")
    out = _run(_runner(u), _finished_state("всего доброго"))
    assert out["reply"] == ""  # silent


def test_post_finish_noise_is_silent_without_llm():
    # "кайфово делать" after finish — bare/noise -> silent.
    u = TurnUnderstanding(reflection="", graph_action="ignore", quality_signal="noise")
    out = _run(_runner(u), _finished_state("и на корену так кайфово делать"))
    assert out["reply"] == ""


def test_post_finish_real_question_is_answered():
    # Client comes back with a real question after finish -> answer, no finale.
    u = TurnUnderstanding(reflection="", answer="Эксперт перезвонит завтра.", graph_action="stay")
    out = _run(_runner(u), _finished_state("а когда перезвонят?"))
    assert "эксперт" in out["reply"].lower()
    assert "всего доброго" not in out["reply"].lower()


def test_bare_filler_holds_without_advancing():
    # "ну" is not agreement — hold the slot, no LLM call needed.
    u = TurnUnderstanding(reflection="ДОЛЖНО БЫТЬ ПРОИГНОРИРОВАНО")  # LLM result ignored for bare filler
    facts = {"opening_done": "yes", "desired_amount": "1", "client_name": "Иван", "property_type": "квартира"}
    out = _run(_runner(u), _state(facts, "ну"))
    assert out["current_node"] == "collect_region"  # stayed on focus, no advance
    assert out["reply"].rstrip().endswith("?")


def test_graph_action_stay_does_not_advance():
    u = TurnUnderstanding(reflection="секунду.", facts_update={"region": "Москва"}, graph_action="stay")
    facts = {"opening_done": "yes", "desired_amount": "1", "client_name": "Иван", "property_type": "квартира"}
    out = _run(_runner(u), _state(facts, "что-то непонятное"))
    assert out["current_node"] == "collect_region"   # stayed
    assert "region" not in out["known_facts"]         # fact not applied


def test_graph_action_end_finishes():
    u = TurnUnderstanding(reflection="понял, больше не отвлекаю.", graph_action="end")
    out = _run(_runner(u), _state({"opening_done": "yes", "desired_amount": "1"}, "не звоните больше"))
    assert out["current_node"] == "finish"


def _state_conf(facts, user_text, conf):
    st = _state(facts, user_text)
    st["stt_confidence"] = conf
    return st


def test_low_confidence_blocks_branch_switch():
    # "то киа" misheard with low confidence must NOT switch to the vehicle branch.
    u = TurnUnderstanding(reflection="Киа, понял.", facts_update={}, branch_signal="vehicle")
    facts = {"opening_done": "yes", "desired_amount": "1", "client_name": "Иван",
             "property_type": "квартира", "region": "Москва", "encumbrance": "нет"}
    out = _run(_runner(u), _state_conf(facts, "ммм то киа", 0.42))
    assert out["trace"]["branch"] == "real_estate"  # stayed, no vehicle switch


def test_low_confidence_drops_garbage_name():
    # "Коротиро" (misheard) must not become the client's name on low confidence.
    u = TurnUnderstanding(reflection="Коротиро, зафиксировал.", facts_update={"client_name": "Коротиро"})
    facts = {"opening_done": "yes", "desired_amount": "1"}
    out = _run(_runner(u), _state_conf(facts, "коротиро", 0.40))
    assert out["known_facts"].get("client_name", "") != "Коротиро"


def test_noise_phrase_holds_without_advancing():
    # "удачи" / "ну и все" are short junk -> hold the slot, no advance.
    u = TurnUnderstanding(reflection="ИГНОР")
    facts = {"opening_done": "yes", "desired_amount": "1", "client_name": "Иван", "property_type": "квартира"}
    out = _run(_runner(u), _state(facts, "Удачи!"))
    assert out["current_node"] == "collect_region"  # stayed


def test_high_confidence_allows_branch_and_name():
    # With good confidence, branch switch and name still work.
    u = TurnUnderstanding(reflection="под ПТС, понял.", facts_update={"branch": "ПТС"}, branch_signal="vehicle")
    facts = {"opening_done": "yes", "desired_amount": "1", "client_name": "Иван"}
    out = _run(_runner(u), _state_conf(facts, "мне под птс машина", 0.85))
    assert out["current_node"].startswith("collect_vehicle")


def test_ack_word_varies_across_turns():
    # "…, понял" every turn is monotonous — the ack word must not repeat back-to-back.
    runner = _runner(TurnUnderstanding(reflection="десять миллионов, понял.", facts_update={"desired_amount": "10000000"}))
    out1 = _run(runner, _state({"opening_done": "yes"}, "десять миллионов"))
    # next turn also ends with "понял"
    r2 = _runner(TurnUnderstanding(reflection="квартира, понял.", facts_update={"property_type": "квартира"}))._llm_client
    out2 = _run(make_call_graph(llm_client=r2, metrics=MetricsCollector(model_name="t")),
                {**out1, "user_text": "квартира", "raw_text": "квартира"})
    assert out1["last_ack"] == "понял"
    # second reflection's ack rotated away from "понял"
    assert "понял" not in out2["reply"].split(".")[0].lower() or out2["last_ack"] != "понял"


def test_finale_strips_trailing_question():
    from call_agent.flow import assemble_reply
    reply = assemble_reply(
        reflection="понял, быстро. А когда удобнее — сегодня, завтра?",
        answer="", focus_node="finish", facts={}, should_end=True, ended_kind="success",
    )
    assert "когда удобнее" not in reply.lower()
    assert "?" not in reply
    assert "эксперт" in reply.lower()


def test_no_echo_without_a_real_fact():
    # Garbage misheard input with NO extracted fact -> don't echo it, just ask.
    u = TurnUnderstanding(reflection="Добро пожаловать, рад помочь.", facts_update={})
    facts = {"opening_done": "yes", "desired_amount": "1", "client_name": "Иван"}
    out = _run(_runner(u), _state(facts, "добро пожаловать"))
    assert "добро пожаловать" not in out["reply"].lower()
    assert "рад помочь" not in out["reply"].lower()
    assert out["reply"].rstrip().endswith("?")  # just the question


def test_real_fact_is_still_mirrored():
    u = TurnUnderstanding(reflection="Двести тысяч, понял.", facts_update={"desired_amount": "200000"})
    out = _run(_runner(u), _state({"opening_done": "yes"}, "двести тысяч"))
    assert out["reply"].startswith("Двести тысяч")


def test_objection_answer_kept_without_fact():
    # No fact, but a real answer to a question -> keep the answer.
    u = TurnUnderstanding(reflection="", answer="Мы не банк, а брокер.", facts_update={})
    out = _run(_runner(u), _state({"opening_done": "yes", "desired_amount": "1"}, "вы банк?"))
    assert "брокер" in out["reply"].lower()


def test_should_end_finishes():
    u = TurnUnderstanding(reflection="", should_end=True)
    out = _run(_runner(u), _state({"opening_done": "yes", "desired_amount": "1"}, "не интересно, спасибо"))
    assert out["current_node"] == "finish"
    assert "?" not in out["reply"]
