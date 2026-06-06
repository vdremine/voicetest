from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _first_nonempty(*values: Any) -> str:
    for value in values:
        cleaned = _clean_text(value)
        if cleaned:
            return cleaned
    return ""


def render_state_summary(
    state: dict[str, Any],
    *,
    last_question: str = "",
    objections: list[str] | None = None,
) -> str:
    known_facts = state.get("known_facts", {})
    if not isinstance(known_facts, dict):
        known_facts = {}

    client_name = _first_nonempty(
        known_facts.get("client_name"),
        state.get("name"),
    )
    region = _first_nonempty(
        known_facts.get("region"),
        known_facts.get("регион"),
    )
    object_type = _first_nonempty(
        state.get("object_type"),
        known_facts.get("вид_объекта"),
    )
    amount = _first_nonempty(
        known_facts.get("amount"),
        known_facts.get("нужная_сумма"),
    )
    encumbrance = _first_nonempty(
        known_facts.get("collateral"),
        known_facts.get("обременение"),
    )
    owner = _first_nonempty(
        known_facts.get("owner"),
        known_facts.get("owners"),
        known_facts.get("собственники"),
    )
    priority = _first_nonempty(known_facts.get("priority"))
    remaining_debt = _first_nonempty(
        known_facts.get("remaining_debt"),
        known_facts.get("остаток_долга"),
    )
    callback_time = _first_nonempty(known_facts.get("callback_time"))
    lead_context = _first_nonempty(known_facts.get("last_contact_context"))
    speed_emphasis = _first_nonempty(known_facts.get("speed_emphasis"))
    stage = _clean_text(state.get("stage")) or "greeting"
    current_node = _clean_text(state.get("current_node"))
    awaiting_field = _clean_text(state.get("awaiting_field"))
    next_required_field = _clean_text(state.get("next_required_field"))
    plan_name = _clean_text(state.get("plan_name")) or _clean_text(known_facts.get("plan_name"))
    summary = _clean_text(state.get("summary"))
    last_question = _clean_text(last_question)

    objection_items = [item for item in (objections or []) if _clean_text(item)]
    if state.get("complaint_active"):
        objection_items.append("есть жалоба или напряжение в разговоре")
    if not objection_items:
        objection_items.append("нет")

    lines = [
        "Состояние звонка:",
        f"- Этап: {stage}",
        f"- Текущий runtime-узел: {current_node or 'не указан'}",
        f"- План: {plan_name or 'new_loan'}",
        f"- Следующий нужный шаг: {next_required_field or 'не определён'}",
        f"- Ожидаемое поле: {awaiting_field or 'не указано'}",
        f"- Последний заданный вопрос: {last_question or 'нет'}",
        "Клиент и сделка:",
        f"- Имя клиента: {client_name or 'неизвестно'}",
        f"- Регион/город: {region or 'неизвестно'}",
        f"- Объект: {object_type or 'неизвестно'}",
        f"- Нужная сумма: {amount or 'неизвестно'}",
        f"- Обременение: {encumbrance or 'неизвестно'}",
        f"- Собственник: {owner or 'неизвестно'}",
        f"- Остаток долга: {remaining_debt or 'неизвестно'}",
        f"- Приоритет клиента: {priority or 'неизвестно'}",
        f"- Удобное время связи: {callback_time or 'не зафиксировано'}",
        f"- Контекст предыдущего контакта: {lead_context or 'не указан'}",
        f"- Акцент на скорость: {'да' if speed_emphasis else 'нет'}",
        f"- Возражения/напряжение: {', '.join(objection_items)}",
        f"- Краткая сводка: {summary or 'фактов пока мало'}",
        "Правило ответа:",
        "- не сбрасывай сценарий без причины;",
        "- сначала ответь по сути, потом продвинь разговор только на один шаг;",
        "- не повторяй приветствие и не задавай два новых вопроса сразу.",
    ]
    return "\n".join(lines)


@dataclass(slots=True)
class SessionMemory:
    max_turns: int = 12
    history: list[dict[str, str]] = field(default_factory=list)
    state: dict[str, Any] = field(default_factory=dict)
    last_question: str = ""
    objections: list[str] = field(default_factory=list)

    def add_user(self, text: str) -> None:
        value = _clean_text(text)
        if not value:
            return
        self.history.append({"role": "user", "text": value})
        self._trim()

    def add_assistant(self, text: str) -> None:
        value = _clean_text(text)
        if not value:
            return
        self.history.append({"role": "assistant", "text": value})
        self._trim()

    def remember_question(self, text: str) -> None:
        value = _clean_text(text)
        if value:
            self.last_question = value

    def add_objection(self, text: str) -> None:
        value = _clean_text(text)
        if not value:
            return
        if value not in self.objections:
            self.objections.append(value)
        self.objections = self.objections[-6:]

    def sync_from_dialogue_state(self, snapshot: dict[str, Any], *, last_question: str = "") -> None:
        self.state = dict(snapshot)
        if last_question:
            self.remember_question(last_question)

    def recent_history(self, *, limit: int | None = None) -> list[dict[str, str]]:
        if limit is None or limit <= 0:
            return list(self.history)
        return list(self.history[-limit:])

    def llm_state_payload(self) -> dict[str, Any]:
        payload = dict(self.state)
        payload["session_state"] = self.state_snapshot()
        payload["session_state_text"] = self.state_text()
        return payload

    def state_snapshot(self) -> dict[str, Any]:
        return {
            "last_question": self.last_question,
            "objections": list(self.objections),
            "history_turns": len(self.history),
        }

    def state_text(self) -> str:
        return render_state_summary(
            self.state,
            last_question=self.last_question,
            objections=self.objections,
        )

    def _trim(self) -> None:
        if self.max_turns > 0:
            self.history = self.history[-self.max_turns :]
