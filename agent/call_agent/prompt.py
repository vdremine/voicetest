from __future__ import annotations

from pathlib import Path
from typing import Any

from .flow import question_for

# Persona / style guide distilled from the 8 gold transcripts (see system_prompt.txt).
SYSTEM_PROMPT = (Path(__file__).with_name("system_prompt.txt")).read_text(encoding="utf-8").strip()


def _trim_text(value: Any, limit: int) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


_FACT_LABELS = {
    "refi_mode": "режим рефинанса",
    "desired_amount": "сумма/остаток",
    "current_payment": "текущий платёж",
    "refi_term": "срок",
    "object_value": "оценка объекта",
    "past_application_amount": "сумма прошлой заявки",
    "prior_loan_note": "о прошлом займе",
    "client_name": "имя",
    "client_patronymic": "отчество",
    "property_type": "недвижимость",
    "region": "регион",
    "encumbrance": "обременение",
    "encumbrance_details": "детали обременения",
    "credit_history_issues": "кредитная история",
    "owner_status": "собственник",
    "consolidation_intent": "хочет объединить кредиты",
    "consolidation_targets": "что объединяет",
    "money_purpose": "цель (назвал сам)",
    "priority": "приоритет",
    "vehicle_type": "авто",
    "vehicle_owner": "собственник авто",
    "vehicle_reregistration_date": "дата переоформления",
    "vehicle_encumbrance": "залог авто",
    "vehicle_year": "год авто",
    "partner_format_desc": "формат партнёрства",
    "partner_experience": "опыт инвестора",
    "callback_consent": "согласие на звонок",
    "callback_time": "время звонка",
}

_SERVICE_KEYS = {
    "phone", "session_id", "opening_done", "pitched", "summary_done",
    "partner_handed", "consolidation_confirmed", "pts_fallback_offered",
    "property_exists", "vehicle_interest", "partner_interest", "amount",
    "нужная_сумма", "amount_deferred", "object_value_deferred",
    "credit_history_deferred", "current_payment_deferred", "urgent",
    "vehicle_handoff_note",
}


# What fact the model MUST extract for each focus slot. The biggest reliability
# lever for a weak model: name the exact key so the gate actually fills.
EXTRACTION_HINTS: dict[str, str] = {
    "collect_amount": "desired_amount — желаемую сумму (любую прими спокойно)",
    "collect_name": "client_name — имя (без фамилии); если назвал и недвижимость — ещё property_type",
    "collect_property_type": "property_type — тип недвижимости; если назвал и город — ещё region",
    "collect_region": "region — город/регион объекта",
    "collect_encumbrance": "encumbrance — есть ли залог: «нет/не в залоге/нигде/чисто»→encumbrance:\"нет\"; «в ипотеке/да»→encumbrance:\"да\"",
    "offer_refi_or_other": "other_property — есть ли другая недвижимость без обременения (да/нет/что)",
    "collect_encumbrance_details": "encumbrance_details — остаток долга/детали",
    "collect_owner": "owner_status — кто собственник",
    "collect_credit_history": "credit_history_issues — просрочки/исполнительные/состояние КИ",
    "collect_current_payment": "current_payment — текущий ежемесячный платёж",
    "collect_refi_term": "refi_term — срок; если назвал объект — ещё property_type",
    "collect_vehicle_type": "vehicle_type — марка/модель авто",
    "collect_vehicle_owner": "vehicle_owner — на кого оформлена машина",
    "collect_vehicle_reregistration_date": "vehicle_reregistration_date — дата переоформления на клиента",
    "collect_vehicle_encumbrance": "vehicle_encumbrance — в залоге/кредите: «нет/чистая»→\"нет\"",
    "collect_vehicle_year": "vehicle_year — год выпуска",
    "priority_choice": "priority — что важнее: скорость/ставка/платёж",
    "handoff_consent": "callback_consent — согласен ли на звонок эксперта",
    "callback_time": "callback_time — удобное время звонка",
    "partner_format": "partner_format_desc — формат партнёрства/инвестиций",
    "partner_experience": "partner_experience — опыт инвестора",
    "collect_consolidation_summary": "consolidation_confirmed — подтверждение «свести в один кредит», верно",
    "offer_pts_fallback": "vehicle_interest — есть ли авто для залога ПТС (да/нет)",
    "collect_name_late": "client_name — имя клиента",
}


def _focus_hint(focus_node: str, focus_question: str, known_facts: dict[str, Any]) -> str:
    if focus_node == "pitch_conditions":
        return (
            "СЕЙЧАС МОМЕНТ ПИТЧА. Система сама произнесёт условия (до 70%, срок до 25 лет, "
            "ставка от 19%, без офиц. трудоустройства, решение 1-2 дня, перс. менеджер, "
            "остаётесь собственником). Тебе — только короткое тёплое отражение последней реплики."
        )
    if focus_node in {"refi_opening", "collect_current_payment", "collect_refi_term"} or str(
        known_facts.get("refi_mode", "")
    ).strip():
        return (
            f"Система дальше спросит: «{focus_question}». Это РЕФИНАНС: не новый кредит, а тот же долг "
            "под меньший платёж/ставку. В reflection не задавай вопрос."
        )
    if focus_node == "summary_before_pitch":
        return "Система сама зачитает резюме фактов. Тебе — только короткое отражение."
    if focus_question:
        return f"Система дальше сама задаст вопрос: «{focus_question}». Ты НЕ дублируй его — дай только reflection (+ answer при встречном вопросе)."
    return "Тебе — только reflection и извлечение фактов."


def build_turn_prompt(
    *,
    focus_node: str,
    known_facts: dict[str, Any],
    history: list[dict[str, str]],
    last_turn_note: str,
    user_text: str,
) -> str:
    focus_question = question_for(focus_node, known_facts)

    facts_text = (
        "\n".join(
            f"- {_FACT_LABELS.get(key, key)}: {_trim_text(value, 60)}"
            for key, value in list(known_facts.items())[:12]
            if str(value).strip() and key not in _SERVICE_KEYS
        )
        or "- пока ничего не известно"
    )

    history_text = (
        "\n".join(
            f"{message['role']}: {_trim_text(message['content'], 140)}"
            for message in history[-3:]
            if message.get("content")
        )
        or "—"
    )

    note_text = _trim_text(last_turn_note, 140) or "—"
    extraction = EXTRACTION_HINTS.get(focus_node, "")
    extraction_block = f"СЕЙЧАС НУЖНО ИЗВЛЕЧЬ ФАКТ: {extraction}\n" if extraction else ""

    return f"""
УЖЕ ИЗВЕСТНО:
{facts_text}

ЗАМЕТКА С ПРОШЛОГО ХОДА: {note_text}

ПОСЛЕДНИЕ РЕПЛИКИ:
{history_text}

РЕПЛИКА КЛИЕНТА СЕЙЧАС:
{_trim_text(user_text, 240)}

{extraction_block}{_focus_hint(focus_node, focus_question, known_facts)}

Верни JSON: reflection (короткое тёплое отражение БЕЗ вопроса) + facts_update (ВСЕ факты из реплики,
особенно факт выше). answer — только если клиент задал встречный вопрос; не дублируй reflection.
""".strip()
