from __future__ import annotations

from typing import Any


FORBIDDEN_FACT_KEYS = {"purpose", "goal", "цель"}
PROTECTED_FACT_KEYS = {"phone", "session_id"}


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def sanitize_facts_update(
    facts_update: dict[str, Any],
    *,
    current_node: str = "",
) -> dict[str, Any]:
    clean: dict[str, Any] = {}

    for key, value in facts_update.items():
        if key in FORBIDDEN_FACT_KEYS or key in PROTECTED_FACT_KEYS:
            continue

        if value is None or _clean_text(value) == "":
            continue

        if key == "client_name":
            lowered = _clean_text(value).lower().replace("ё", "е")
            if current_node == "collect_name":
                clean[key] = value
                continue
            if "владимир" in lowered or "влад+имир" in lowered or "мосинвестфинанс" in lowered:
                continue

        clean[key] = value

    return clean


def apply_facts(
    known_facts: dict[str, Any],
    facts_update: dict[str, Any],
    *,
    current_node: str = "",
) -> dict[str, Any]:
    facts = dict(known_facts)

    for key, value in sanitize_facts_update(facts_update, current_node=current_node).items():
        facts[key] = value

        if key == "desired_amount":
            facts["amount"] = value
            facts["нужная_сумма"] = value

        if key == "property_type":
            facts["property_exists"] = "yes"
            facts.pop("no_real_estate", None)

        if key == "no_real_estate":
            facts["property_exists"] = "no"
            facts.pop("property_type", None)

        if key == "vehicle_type":
            facts["vehicle_interest"] = "yes"

        if key == "partner_interest":
            facts["partner_interest"] = "yes"

    for key in FORBIDDEN_FACT_KEYS:
        facts.pop(key, None)

    return facts
