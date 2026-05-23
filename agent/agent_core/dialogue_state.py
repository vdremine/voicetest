from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .knowledge_base import KnowledgeBase


_AMOUNT_RE = re.compile(
    r"(?P<num>\d+(?:[\.,]\d+)?)\s*(?P<unit>млн|миллион|миллиона|миллионов|тыс|тысяч|тысячи)?",
    flags=re.IGNORECASE,
)


@dataclass(slots=True)
class DialogueState:
    stage: str = "greeting"
    scenario: str = ""
    object_type: str = ""
    amount_text: str = ""
    goal: str = ""
    collateral: str = ""
    city: str = ""
    name: str = ""
    official_employment: str = ""
    callback_time: str = ""
    complaint_active: bool = False
    last_user_text: str = ""
    last_agent_text: str = ""
    awaiting_field: str = ""
    next_required_field: str = ""
    known_facts: dict[str, str] = field(default_factory=dict)

    def update_from_user(self, raw_text: str, normalized_text: str, *, kb: KnowledgeBase | None = None) -> None:
        text = normalized_text.strip()
        if not text:
            return

        self.last_user_text = raw_text.strip()
        lowered = text.lower()

        amount = self._extract_amount(text)
        if amount:
            self.amount_text = amount
            self.known_facts["amount"] = amount
            self.known_facts["нужная_сумма"] = amount

        goal = self._detect_goal(lowered)
        if goal:
            self.goal = goal
            self.known_facts["goal"] = goal
            self.known_facts["цель"] = goal

        collateral = self._detect_collateral(lowered)
        if collateral:
            self.collateral = collateral
            self.known_facts["collateral"] = collateral
            if collateral == "недвижимость":
                self.known_facts["вид_объекта"] = self.object_type or "недвижимость"
            elif collateral == "автомобиль/птс":
                self.known_facts["вид_объекта"] = "автомобиль"

        employment = self._detect_employment(lowered)
        if employment:
            self.official_employment = employment
            self.known_facts["official_employment"] = employment

        if "не сегодня" in lowered or "послезавтра" in lowered or "после двенадцати" in lowered:
            self.callback_time = raw_text.strip()
            self.known_facts["callback_time"] = raw_text.strip()

        if any(marker in lowered for marker in ("жалоб", "грубо", "нехорош", "запись разговора")):
            self.complaint_active = True

        if kb is not None:
            self.object_type = self._detect_object_type(lowered)
            if self.object_type:
                self.known_facts["вид_объекта"] = self.object_type
            self.scenario = self._detect_scenario(lowered, kb)
            if self.scenario:
                self.known_facts["сценарий"] = self.scenario

        self._advance_stage()
        if kb is not None:
            self.next_required_field = kb.next_required_field(self.snapshot())

    def update_from_agent(self, reply_tts: str, next_step: str, *, kb: KnowledgeBase | None = None) -> None:
        self.last_agent_text = reply_tts.strip()
        self.awaiting_field = self._detect_awaiting_field(reply_tts, next_step)
        self._advance_stage()
        if kb is not None:
            self.next_required_field = kb.next_required_field(self.snapshot())

    def snapshot(self) -> dict[str, Any]:
        summary_parts: list[str] = []
        if self.amount_text:
            summary_parts.append(f"сумма: {self.amount_text}")
        if self.goal:
            summary_parts.append(f"цель: {self.goal}")
        if self.scenario:
            summary_parts.append(f"сценарий: {self.scenario}")
        if self.object_type:
            summary_parts.append(f"тип объекта: {self.object_type}")
        if self.collateral:
            summary_parts.append(f"залог/объект: {self.collateral}")
        if self.city:
            summary_parts.append(f"город: {self.city}")
        if self.official_employment:
            summary_parts.append(f"работа: {self.official_employment}")
        if self.callback_time:
            summary_parts.append(f"callback: {self.callback_time}")
        if self.complaint_active:
            summary_parts.append("есть жалоба на прошлый разговор")

        summary = "; ".join(summary_parts) if summary_parts else "фактов пока мало"
        return {
            "stage": self.stage,
            "scenario": self.scenario,
            "object_type": self.object_type,
            "awaiting_field": self.awaiting_field,
            "next_required_field": self.next_required_field,
            "known_facts": dict(self.known_facts),
            "summary": summary,
            "last_user_text": self.last_user_text,
            "last_agent_text": self.last_agent_text,
        }

    @staticmethod
    def _extract_amount(text: str) -> str:
        best = ""
        for match in _AMOUNT_RE.finditer(text):
            num = match.group("num")
            unit = (match.group("unit") or "").lower()
            if not num:
                continue
            if unit:
                best = f"{num} {unit}"
            elif float(num.replace(",", ".")) >= 10000:
                best = num
        return best

    @staticmethod
    def _detect_goal(text: str) -> str:
        if "машин" in text or "автомоб" in text:
            return "покупка автомобиля"
        if "рефинанс" in text:
            return "рефинансирование"
        if "ипотек" in text:
            return "ипотека"
        if "отсроч" in text or "платеж" in text or "оплат" in text:
            return "оплата или отсрочка"
        if "недвижим" in text:
            return "кредит под залог недвижимости"
        return ""

    @staticmethod
    def _detect_collateral(text: str) -> str:
        if any(marker in text for marker in ("нет недвижимости", "без недвижимости", "ничего нет", "машины нет")):
            return "нет залога"
        if "недвижим" in text or "квартир" in text or "дом" in text:
            return "недвижимость"
        if "автомоб" in text or "машин" in text or "птс" in text:
            return "автомобиль/птс"
        return ""

    @staticmethod
    def _detect_employment(text: str) -> str:
        if "официально" in text and "не" not in text.split("официально", 1)[0][-4:]:
            return "официальная работа"
        if "неофициаль" in text or "официально не" in text:
            return "неофициальная работа"
        return ""

    @staticmethod
    def _detect_awaiting_field(reply_tts: str, next_step: str) -> str:
        text = f"{reply_tts} {next_step}".lower()
        if "какая сумма" in text:
            return "amount"
        if "какая цель" in text or "на какую цель" in text:
            return "goal"
        if "как к вам обращаться" in text or "как вас зовут" in text:
            return "name"
        if "удобнее, чтобы он связался" in text or "в какое время" in text:
            return "callback_time"
        return ""

    @staticmethod
    def _detect_object_type(text: str) -> str:
        if "квартир" in text:
            return "квартира"
        if "таунхаус" in text:
            return "таунхаус"
        if "апартамент" in text:
            return "апартаменты"
        if "дом" in text:
            return "дом"
        if "земл" in text or "участ" in text:
            return "земля"
        if "коммерчес" in text or "помещени" in text:
            return "коммерческая_недвижимость"
        if "машин" in text or "автомоб" in text:
            return "автомобиль"
        if "недвижим" in text:
            return "недвижимость"
        return ""

    def _detect_scenario(self, text: str, kb: KnowledgeBase) -> str:
        if any(marker in text for marker in ("не звоните", "отмените заявку", "не беспокоить")):
            return "отказ_или_не_беспокоить"
        if "ип " in f"{text} " or "ооо" in text or "юридичес" in text:
            return "квалификация_бизнес_сценария"
        if "рефинанс" in text or "ипотек" in text or "остаток долга" in text:
            return "квалификация_рефинансирования"
        if "плохая кредитная история" in text:
            return "квалификация_сценария_с_плохой_кредитной_историей"
        if "просроч" in text or "микрозайм" in text:
            return "квалификация_сценария_с_просрочками"
        if self.object_type == "автомобиль":
            return "квалификация_автомобиля"
        if self.object_type in {"квартира", "таунхаус", "апартаменты", "дом", "земля", "коммерческая_недвижимость", "недвижимость"}:
            return "квалификация_недвижимости"
        return self.scenario

    def _advance_stage(self) -> None:
        if self.complaint_active:
            self.stage = "objection_handling"
            return
        if self.callback_time:
            self.stage = "confirmation"
            return
        if self.amount_text and self.goal:
            self.stage = "qualification"
            return
        if self.last_user_text:
            self.stage = "need_detection"
