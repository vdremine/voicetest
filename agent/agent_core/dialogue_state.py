from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


_AMOUNT_RE = re.compile(
    r"(?P<num>\d+(?:[\.,]\d+)?)\s*(?P<unit>млн|миллион|миллиона|миллионов|тыс|тысяч|тысячи)?",
    flags=re.IGNORECASE,
)


@dataclass(slots=True)
class DialogueState:
    stage: str = "greeting"
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
    known_facts: dict[str, str] = field(default_factory=dict)

    def update_from_user(self, raw_text: str, normalized_text: str) -> None:
        text = normalized_text.strip()
        if not text:
            return

        self.last_user_text = raw_text.strip()
        lowered = text.lower()

        amount = self._extract_amount(text)
        if amount:
            self.amount_text = amount
            self.known_facts["amount"] = amount

        goal = self._detect_goal(lowered)
        if goal:
            self.goal = goal
            self.known_facts["goal"] = goal

        collateral = self._detect_collateral(lowered)
        if collateral:
            self.collateral = collateral
            self.known_facts["collateral"] = collateral

        employment = self._detect_employment(lowered)
        if employment:
            self.official_employment = employment
            self.known_facts["official_employment"] = employment

        if "не сегодня" in lowered or "послезавтра" in lowered or "после двенадцати" in lowered:
            self.callback_time = raw_text.strip()
            self.known_facts["callback_time"] = raw_text.strip()

        if any(marker in lowered for marker in ("жалоб", "грубо", "нехорош", "запись разговора")):
            self.complaint_active = True

        self._advance_stage()

    def update_from_agent(self, reply_tts: str, next_step: str) -> None:
        self.last_agent_text = reply_tts.strip()
        self.awaiting_field = self._detect_awaiting_field(reply_tts, next_step)
        self._advance_stage()

    def snapshot(self) -> dict[str, Any]:
        summary_parts: list[str] = []
        if self.amount_text:
            summary_parts.append(f"сумма: {self.amount_text}")
        if self.goal:
            summary_parts.append(f"цель: {self.goal}")
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
            "awaiting_field": self.awaiting_field,
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
