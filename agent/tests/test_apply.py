"""Tests for fact application and the relaxed name sanitizer.

Bug being fixed: a client genuinely named "Владимир" was dropped everywhere
except the collect_name node, because the agent is also named Владимир. The
name must be accepted on any node; only the agent's own self-identification
(company name) is filtered.
"""

from __future__ import annotations

from call_agent.apply import apply_facts, sanitize_facts_update


def test_client_name_kept_on_name_node():
    out = sanitize_facts_update({"client_name": "Владимир"}, current_node="collect_name")
    assert out["client_name"] == "Владимир"


def test_client_name_vladimir_kept_on_other_nodes():
    # The real bug: name dropped off-node because agent is also Владимир.
    out = sanitize_facts_update({"client_name": "Владимир"}, current_node="collect_amount")
    assert out.get("client_name") == "Владимир"


def test_company_self_identification_is_dropped_as_name():
    out = sanitize_facts_update({"client_name": "МосИнвестФинанс"}, current_node="collect_amount")
    assert "client_name" not in out


def test_purpose_fact_is_forbidden():
    out = sanitize_facts_update({"purpose": "ремонт"}, current_node="collect_amount")
    assert "purpose" not in out


def test_amount_aliases_applied():
    facts = apply_facts({}, {"desired_amount": "200000"}, current_node="collect_amount")
    assert facts["desired_amount"] == "200000"
    assert facts["amount"] == "200000"


def test_no_real_estate_sets_property_exists_no():
    facts = apply_facts({}, {"no_real_estate": "yes"}, current_node="collect_property_type")
    assert facts["property_exists"] == "no"
    assert "property_type" not in facts
