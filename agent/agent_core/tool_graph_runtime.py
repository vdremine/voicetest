from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class CachedNodeReply:
    reply_text: str
    reply_node: str
    next_node: str


class ToolGraphRuntime:
    _OPENING_FAMILY = {
        "opening",
        "small_talk",
        "who_are_you",
        "identity_company_faq",
        "source_of_number_faq",
        "robot_check",
        "memory_denial_faq",
        "check_convenience",
    }

    def __init__(self, *, graph_path: Path, agent_name: str) -> None:
        payload = json.loads(graph_path.read_text(encoding="utf-8"))
        self._nodes: dict[str, dict[str, Any]] = payload.get("nodes", {})
        self._start_node = str(payload.get("start_node", "opening")).strip() or "opening"
        self._agent_name = agent_name

    @classmethod
    def load(cls, *, graph_path: Path, agent_name: str) -> "ToolGraphRuntime":
        return cls(graph_path=graph_path, agent_name=agent_name)

    @property
    def start_node(self) -> str:
        return self._start_node

    def cached_reply_for_text(self, text: str, state: dict[str, Any]) -> CachedNodeReply | None:
        lowered = self._normalize(text)
        current_node = str(state.get("current_node", "")).strip() or self._start_node

        node_name = ""
        if self._matches_any(lowered, ("как дела", "как вы", "как настроение")):
            node_name = "small_talk"
        elif self._matches_any(lowered, ("кто звонит", "кто вы", "это кто", "представьтесь")):
            node_name = "who_are_you"
        elif self._matches_any(lowered, ("что за компания", "что за организация", "вы кто такие")):
            node_name = "identity_company_faq"
        elif self._matches_any(lowered, ("откуда у вас мой номер", "где взяли номер", "почему вы мне звоните")):
            node_name = "source_of_number_faq"
        elif self._matches_any(lowered, ("вы робот", "это робот", "живой ли вы", "человек или робот")):
            node_name = "robot_check"
        elif self._matches_any(lowered, ("не помню такого", "не оставлял заявку", "не помню обращения")):
            node_name = "memory_denial_faq"
        if not node_name:
            return None

        node = self._nodes.get(node_name)
        if not node:
            return None

        reply_text = self._node_reply_text(node)
        if not reply_text:
            return None

        next_node = self._default_followup_node(node_name)
        return CachedNodeReply(reply_text=reply_text, reply_node=node_name, next_node=next_node)

    def question_for_state(self, state: dict[str, Any]) -> tuple[str, str] | None:
        known_facts = state.get("known_facts", {})
        if not isinstance(known_facts, dict):
            known_facts = {}

        plan_name = str(state.get("plan_name", "")).strip() or str(known_facts.get("plan_name", "")).strip()
        credit_closed = str(known_facts.get("credit_closed", "")).strip()
        if str(known_facts.get("amount_needs_clarification", "")).strip() == "yes":
            return self._fallback_question(
                "clarify_amount_units",
                "Подскажите, пожалуйста, речь про рубли, тысячи или миллионы?",
            )

        if plan_name == "refinance_or_returning_customer":
            if not credit_closed:
                return self._fallback_question(
                    "refinance_credit_closed",
                    "Подскажите, пожалуйста, текущий кредит уже закрыт или вы его ещё выплачиваете?",
                )
            if credit_closed == "no":
                if not self._has_value(known_facts, "остаток_долга", "remaining_debt"):
                    return self._fallback_question(
                        "refinance_collect_current_credit",
                        "Подскажите, пожалуйста, какой сейчас остаток долга по текущему кредиту?",
                    )
                if not self._has_value(known_facts, "вид_объекта", "object_type"):
                    return self._slot_question("collect_object_type")
                if not self._has_value(known_facts, "регион", "region"):
                    return self._slot_question("collect_region")
                if not self._has_value(known_facts, "обременение", "encumbrance", "collateral"):
                    return self._slot_question("collect_encumbrance")
                if not self._has_value(known_facts, "собственники", "owner", "owners"):
                    return self._slot_question("collect_owner")
                if not self._has_value(known_facts, "priority"):
                    return self._fallback_question(
                        "collect_priority",
                        "Что для вас сейчас важнее: скорость получения денег или минимальная ставка?",
                    )
                return None

        if not self._has_value(known_facts, "нужная_сумма", "amount"):
            return self._slot_question("collect_amount")
        if not self._has_value(known_facts, "вид_объекта", "object_type"):
            return self._slot_question("collect_object_type")
        if not self._has_value(known_facts, "регион", "region"):
            return self._slot_question("collect_region")
        if not self._has_value(known_facts, "обременение", "encumbrance", "collateral"):
            return self._slot_question("collect_encumbrance")
        if not self._has_value(known_facts, "собственники", "owner", "owners"):
            return self._slot_question("collect_owner")
        if not self._has_value(known_facts, "priority"):
            return self._fallback_question(
                "collect_priority",
                "Что для вас сейчас важнее: скорость получения денег или минимальная ставка?",
            )
        return None

    def llm_context_for_text(self, text: str, state: dict[str, Any]) -> dict[str, Any]:
        lowered = self._normalize(text)
        current_node = str(state.get("current_node", "")).strip() or self._start_node

        node_name = "handle_questions"
        if self._matches_any(lowered, ("не помню такого", "не оставлял заявку", "не помню обращения")):
            node_name = "memory_denial_faq"
        elif self._matches_any(
            lowered,
            (
                "нет недвижимости",
                "не являюсь собственником",
                "собственником не являюсь",
                "ничего своего нет",
                "у меня ничего нет",
                "автомобиль не мой",
                "машина не моя",
            ),
        ):
            node_name = "no_collateral_partner_prescreen"
        elif self._matches_any(lowered, ("неудобно", "занят", "не могу говорить", "не сейчас")):
            node_name = "soft_micro_qualification_or_callback"
        elif self._matches_any(lowered, ("неинтересно", "не нужно", "не звоните", "отстаньте")):
            node_name = "handle_not_interested"
        elif current_node in {"collect_owner", "collect_encumbrance", "collect_region"}:
            node_name = "pitch_short"

        node = self._nodes.get(node_name, {})
        resume = self.question_for_state(state)
        return {
            "node_name": node_name,
            "node_type": str(node.get("type", "")),
            "goal": str(node.get("goal", "")),
            "plan_name": str(state.get("plan_name", "")).strip(),
            "next_required_field": str(state.get("next_required_field", "")).strip(),
            "resume_node": resume[0] if resume else "",
            "resume_question": resume[1] if resume else "",
            "rules": [str(item).strip() for item in node.get("rules", []) if str(item).strip()],
            "good_example": str(node.get("good_example", "")).strip(),
            "allowed_topics": [str(item).strip() for item in node.get("allowed_topics", []) if str(item).strip()],
            "transitions": [item.get("next", "") for item in node.get("transitions", []) if isinstance(item, dict)],
        }

    def _slot_question(self, node_name: str) -> tuple[str, str] | None:
        node = self._nodes.get(node_name)
        if not node:
            return None
        question = str(node.get("question", "")).strip()
        if not question:
            return None
        return node_name, question

    @staticmethod
    def _has_value(known_facts: dict[str, Any], *keys: str) -> bool:
        for key in keys:
            value = str(known_facts.get(key, "")).strip()
            if value:
                return True
        return False

    def _node_reply_text(self, node: dict[str, Any]) -> str:
        ready_answers = node.get("ready_answers", [])
        if isinstance(ready_answers, list):
            candidates = [str(item).strip() for item in ready_answers if str(item).strip()]
            if candidates:
                best = max(candidates, key=len)
                return self._render_template(best)
        template = str(node.get("reply_template", "")).strip()
        if template:
            return self._render_template(template)
        good_example = str(node.get("good_example", "")).strip()
        if good_example:
            return self._render_template(good_example)
        return ""

    def _default_followup_node(self, node_name: str) -> str:
        mapping = {
            "opening": "check_convenience",
            "small_talk": "check_convenience",
            "who_are_you": "check_convenience",
            "identity_company_faq": "check_convenience",
            "source_of_number_faq": "check_convenience",
            "robot_check": "check_convenience",
            "memory_denial_faq": "check_convenience",
            "callback_reentry": "detect_scenario",
        }
        return mapping.get(node_name, node_name)

    def _fallback_question(self, node_name: str, question: str) -> tuple[str, str]:
        return node_name, question

    def _render_template(self, text: str) -> str:
        return text.replace("[AGENT_NAME]", self._agent_name)

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"\s+", " ", text.lower().replace("ё", "е")).strip()

    @staticmethod
    def _matches_any(text: str, patterns: tuple[str, ...]) -> bool:
        return any(pattern in text for pattern in patterns)

    @staticmethod
    def _looks_ready(text: str) -> bool:
        ready = {"да", "да слушаю", "слушаю", "говорите", "удобно", "слушаю вас"}
        return text in ready or text.startswith(("да ", "слушаю", "говорите"))
