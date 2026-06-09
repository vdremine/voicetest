"""Navigation tests for the 4-branch fact-driven flow (from the gold-transcript spec).

Navigation is a pure function of accumulated facts. These tests encode the
funnel order distilled from the 8 reference transcripts.
"""

from __future__ import annotations

from call_agent.flow import (
    assemble_reply,
    opening_for,
    question_for,
    resolve_branch,
    resolve_focus,
)


# Convenience: a fully-opened real-estate state prelude.
def _re(**extra):
    facts = {"opening_done": "yes"}
    facts.update(extra)
    return facts


# --- branch resolution ----------------------------------------------------


def test_branch_default_real_estate():
    assert resolve_branch({}) == "real_estate"


def test_branch_refi_from_session_flag():
    assert resolve_branch({"refi_mode": "yes"}) == "refi"


def test_branch_vehicle_from_no_property():
    assert resolve_branch({"property_exists": "no"}) == "vehicle"


def test_branch_vehicle_from_interest():
    assert resolve_branch({"vehicle_interest": "yes"}) == "vehicle"


def test_branch_partner():
    assert resolve_branch({"partner_interest": "yes"}) == "partner"


# --- real_estate order ----------------------------------------------------


def test_re_after_opening_focus_amount():
    assert resolve_focus(_re(), "real_estate") == "collect_amount"


def test_re_amount_then_name():
    assert resolve_focus(_re(desired_amount="1300000"), "real_estate") == "collect_name"


def test_re_amount_deferred_skips_to_name():
    facts = _re(amount_deferred="yes")
    assert resolve_focus(facts, "real_estate") == "collect_name"


def test_re_name_then_property():
    facts = _re(desired_amount="1300000", client_name="Татьяна")
    assert resolve_focus(facts, "real_estate") == "collect_property_type"


def test_re_property_then_region():
    facts = _re(desired_amount="1", client_name="Т", property_type="квартира")
    assert resolve_focus(facts, "real_estate") == "collect_region"


def test_re_region_then_encumbrance():
    # object_value is not asked as a step; region -> encumbrance directly (gold).
    facts = _re(desired_amount="1", client_name="Т", property_type="квартира", region="Саратов")
    assert resolve_focus(facts, "real_estate") == "collect_encumbrance"


def test_re_encumbrance_negative_skips_details():
    facts = _re(
        desired_amount="1", client_name="Т", property_type="квартира", region="Саратов",
        encumbrance="нет, чистая",
    )
    # clean client, no encumbrance, no credit signal -> straight to owner
    assert resolve_focus(facts, "real_estate") == "collect_owner"


def test_re_encumbrance_negative_phrasing_does_not_loop():
    # "вне обременения" must be treated as no encumbrance -> advance, never re-ask.
    for phrasing in ("нет", "не в залоге", "вне обременения", "нигде не заложен", "чистая"):
        facts = _re(
            desired_amount="1", client_name="Т", property_type="квартира", region="Саратов",
            encumbrance=phrasing,
        )
        assert resolve_focus(facts, "real_estate") == "collect_owner", phrasing


def test_re_encumbrance_positive_offers_refi_and_other_property():
    facts = _re(
        desired_amount="1", client_name="Т", property_type="квартира", region="Саратов",
        encumbrance="да, в ипотеке",
    )
    # under encumbrance -> first offer refinancing + ask about other property
    assert resolve_focus(facts, "real_estate") == "offer_refi_or_other"
    q = question_for("offer_refi_or_other", facts).lower()
    assert "рефинансиров" in q
    assert "недвижим" in q  # asks about another property
    # then collect the remaining-debt detail
    facts["other_property"] = "нет другой"
    assert resolve_focus(facts, "real_estate") == "collect_encumbrance_details"


def test_infer_gate_value_maps_encumbrance_yes_no():
    from call_agent.flow import infer_gate_value
    assert infer_gate_value("collect_encumbrance", "нет, вне обременения") == "нет"
    assert infer_gate_value("collect_encumbrance", "да, в ипотеке у Сбера") != "нет"


def test_re_clean_client_reaches_owner_then_pitch_then_priority_then_handoff():
    facts = _re(
        desired_amount="1", client_name="Т", property_type="квартира", region="Саратов",
        encumbrance="нет",
    )
    assert resolve_focus(facts, "real_estate") == "collect_owner"
    facts["owner_status"] = "только я"
    assert resolve_focus(facts, "real_estate") == "pitch_conditions"
    facts["pitched"] = "yes"
    assert resolve_focus(facts, "real_estate") == "priority_choice"
    facts["priority"] = "ставка"
    assert resolve_focus(facts, "real_estate") == "handoff_consent"
    facts["callback_consent"] = "да"
    assert resolve_focus(facts, "real_estate") == "callback_time"
    facts["callback_time"] = "сегодня"
    assert resolve_focus(facts, "real_estate") == "finish"


def test_re_consolidation_inserts_summary_before_pitch():
    facts = _re(
        desired_amount="1", client_name="Т", property_type="квартира", region="М",
        object_value="45000000", encumbrance="ипотека", encumbrance_details="8 млн",
        consolidation_intent="объединить", owner_status="я",
    )
    # consolidation_intent present -> summary node applies before pitch
    assert resolve_focus(facts, "real_estate") == "collect_consolidation_summary"


# --- vehicle order --------------------------------------------------------


def test_vehicle_name_first_then_type():
    facts = {"opening_done": "yes", "vehicle_interest": "yes"}
    assert resolve_focus(facts, "vehicle") == "collect_name"
    facts["client_name"] = "Владимир"
    assert resolve_focus(facts, "vehicle") == "collect_vehicle_type"


def test_vehicle_owner_other_requires_reregistration_date():
    facts = {
        "opening_done": "yes", "vehicle_interest": "yes", "client_name": "В",
        "vehicle_type": "Toyota RAV4", "vehicle_owner": "бывшая жена",
    }
    assert resolve_focus(facts, "vehicle") == "collect_vehicle_reregistration_date"


def test_vehicle_owner_self_skips_reregistration():
    facts = {
        "opening_done": "yes", "vehicle_interest": "yes", "client_name": "В",
        "vehicle_type": "Toyota RAV4", "vehicle_owner": "я",
    }
    assert resolve_focus(facts, "vehicle") == "collect_vehicle_encumbrance"


def test_vehicle_amount_after_encumbrance():
    facts = {
        "opening_done": "yes", "vehicle_interest": "yes", "client_name": "В",
        "vehicle_type": "Toyota RAV4", "vehicle_owner": "я", "vehicle_encumbrance": "чистая",
    }
    assert resolve_focus(facts, "vehicle") == "collect_amount"


# --- refi order -----------------------------------------------------------


def test_refi_opening_then_amount_as_remainder():
    facts = {"refi_mode": "yes", "opening_done": "yes"}
    assert resolve_focus(facts, "refi") == "collect_amount"


def test_refi_amount_leads_toward_term_and_property():
    facts = {"refi_mode": "yes", "opening_done": "yes", "desired_amount": "3000000"}
    nxt = resolve_focus(facts, "refi")
    assert nxt in ("collect_current_payment", "collect_refi_term")


# --- partner order --------------------------------------------------------


def test_name_deferred_skips_early_name_but_re_asks_before_handoff():
    facts = _re(desired_amount="1", name_deferred="yes")  # name dodged during objection
    assert resolve_focus(facts, "real_estate") == "collect_property_type"
    # ...funnel proceeds; name is re-asked before handoff if still missing
    facts.update(
        property_type="квартира", region="М", encumbrance="нет",
        owner_status="я", pitched="yes", priority="ставка",
    )
    assert resolve_focus(facts, "real_estate") == "collect_name_late"
    facts["client_name"] = "Эдуард"
    assert resolve_focus(facts, "real_estate") == "handoff_consent"


def test_partner_format_then_name_then_experience():
    facts = {"opening_done": "yes", "partner_interest": "yes"}
    assert resolve_focus(facts, "partner") == "partner_format"
    facts["partner_format_desc"] = "выкуп залоговой"
    assert resolve_focus(facts, "partner") == "collect_name"
    facts["client_name"] = "Александр"
    assert resolve_focus(facts, "partner") == "partner_experience"


# --- openings -------------------------------------------------------------


def test_opening_cold_goes_straight_to_amount():
    text = opening_for({})
    assert "сумму" in text.lower()
    assert "Дмитрий" not in text


def test_opening_cold_introduces_company_and_name():
    text = opening_for({})
    assert "МосИнвестФинанс" in text
    assert "Владимир" in text


def test_pitch_does_not_cap_the_amount():
    # Any sum is considered calmly — no "до X можем рассматривать" cap.
    text = question_for("pitch_conditions", {"object_value": "6000000"})
    assert "по вашей оценке это порядка" not in text.lower()


def test_opening_refi_asks_remainder():
    text = opening_for({"refi_mode": "yes"})
    low = text.lower()
    assert "осталось" in low or "выплат" in low


# --- question wording -----------------------------------------------------


def test_name_question_combines_property_when_both_missing():
    facts = _re(desired_amount="1")  # name + property both empty
    q = question_for("collect_name", facts)
    low = q.lower()
    assert "зовут" in low and "недвижимост" in low  # combined question


def test_name_question_in_vehicle_does_not_mention_property():
    facts = {"opening_done": "yes", "vehicle_interest": "yes"}  # vehicle branch, no name yet
    q = question_for("collect_name", facts)
    assert "недвижим" not in q.lower()


def test_partner_asks_callback_time_before_handoff():
    facts = {
        "opening_done": "yes", "partner_interest": "yes",
        "partner_format_desc": "выкуп", "client_name": "Александр",
        "partner_experience": "да",
    }
    assert resolve_focus(facts, "partner") == "callback_time"
    facts["callback_time"] = "сегодня вечером"
    assert resolve_focus(facts, "partner") == "partner_handoff"


def test_vehicle_pitch_safety_is_about_car():
    facts = {"vehicle_type": "Toyota", "pitched": ""}
    text = question_for("pitch_conditions", facts)
    low = text.lower()
    assert "машина остаётся" in low or "птс" in low
    assert "выписыва" not in low


def test_region_question_never_asks_district():
    q = question_for("collect_region", _re())
    low = q.lower()
    assert "район" not in low and "адрес" not in low


# --- assemble_reply -------------------------------------------------------


def test_reply_reflection_plus_question():
    reply = assemble_reply(
        reflection="миллион триста. угу, понял вас.",
        answer="",
        focus_node="collect_name",
        facts=_re(desired_amount="1300000"),
        repeat_count=0,
        should_end=False,
    )
    assert reply.startswith("миллион триста")
    assert reply.rstrip().endswith("?")


def test_reply_reflection_answer_question_order():
    # name + objection in one turn: reflection (mirror) + answer (objection) + next question
    reply = assemble_reply(
        reflection="Владимир, очень приятно.",
        answer="Оплата только по факту получения кредита.",
        focus_node="collect_property_type",
        facts=_re(desired_amount="1", client_name="Владимир"),
        repeat_count=0,
        should_end=False,
    )
    assert reply.index("Владимир") < reply.index("Оплата") < reply.index("недвижим")


def test_reply_should_end_no_question():
    reply = assemble_reply(
        reflection="", answer="", focus_node="collect_amount",
        facts=_re(), repeat_count=0, should_end=True,
    )
    assert "?" not in reply


def test_reply_fallback_question_when_no_reflection():
    reply = assemble_reply(
        reflection="", answer="", focus_node="collect_region",
        facts=_re(property_type="квартира"), repeat_count=0, should_end=False,
    )
    assert reply.strip().endswith("?")
    assert "повторите" not in reply.lower()
