from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .knowledge_base import KnowledgeBase


_AMOUNT_RE = re.compile(
    r"(?P<num>\d+(?:[\.,]\d+)?)\s*(?P<unit>млн|миллион|миллиона|миллионов|тыс|тысяч|тысячи|руб|рубль|рубля|рублей)?",
    flags=re.IGNORECASE,
)
_CITY_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bмоскв(?:а|е|ы|у|ой)?\b", flags=re.IGNORECASE), "Москва"),
    (
        re.compile(r"\bмосковск(?:ая|ой|ую)\s+област(?:ь|и)\b", flags=re.IGNORECASE),
        "Московская область",
    ),
    (
        re.compile(r"\bсанкт[\s-]?петербург(?:е|а|у|ом)?\b", flags=re.IGNORECASE),
        "Санкт-Петербург",
    ),
    (re.compile(r"\bпитер(?:е|а|у|ом)?\b", flags=re.IGNORECASE), "Санкт-Петербург"),
)


@dataclass(slots=True)
class DialogueState:
    stage: str = "greeting"
    current_node: str = "opening"
    scenario: str = ""
    plan_name: str = "new_loan"
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

    def update_from_user(
        self,
        raw_text: str,
        normalized_text: str,
        *,
        kb: KnowledgeBase | None = None,
    ) -> set[str]:
        text = normalized_text.strip()
        if not text:
            return set()

        self.last_user_text = raw_text.strip()
        lowered = text.lower()
        updated_fields: set[str] = set()

        amount = self._extract_amount(text)
        if amount:
            if amount != self.amount_text:
                updated_fields.add("нужная_сумма")
            self.amount_text = amount
            self.known_facts["amount"] = amount
            self.known_facts["нужная_сумма"] = amount
            if self._amount_needs_clarification(amount):
                self.known_facts["amount_needs_clarification"] = "yes"
            else:
                self.known_facts.pop("amount_needs_clarification", None)

        goal = self._detect_goal(lowered)
        if goal:
            if goal != self.goal:
                updated_fields.add("цель")
            self.goal = goal
            self.known_facts["goal"] = goal
            self.known_facts["цель"] = goal

        collateral = self._detect_collateral(lowered)
        if collateral:
            if collateral != self.collateral:
                updated_fields.add("обременение")
            self.collateral = collateral
            self.known_facts["collateral"] = collateral
            self.known_facts["обременение"] = collateral
            if collateral == "недвижимость":
                self.known_facts["вид_объекта"] = self.object_type or "недвижимость"
            elif collateral == "автомобиль/птс":
                self.known_facts["вид_объекта"] = "автомобиль"

        employment = self._detect_employment(lowered)
        if employment:
            if employment != self.official_employment:
                updated_fields.add("official_employment")
            self.official_employment = employment
            self.known_facts["official_employment"] = employment

        city = self._detect_city(raw_text)
        if city:
            if city != self.city:
                updated_fields.add("регион")
            self.city = city
            self.known_facts["region"] = city
            self.known_facts["регион"] = city

        if "не сегодня" in lowered or "послезавтра" in lowered or "после двенадцати" in lowered:
            if raw_text.strip() != self.callback_time:
                updated_fields.add("callback_time")
            self.callback_time = raw_text.strip()
            self.known_facts["callback_time"] = raw_text.strip()

        if any(marker in lowered for marker in ("жалоб", "грубо", "нехорош", "запись разговора")):
            self.complaint_active = True

        credit_closed = self._detect_credit_closed(lowered)
        if credit_closed:
            updated_fields.add("credit_closed")
            self.known_facts["credit_closed"] = credit_closed

        owner = self._detect_owner(lowered)
        if owner:
            updated_fields.add("собственники")
            self.known_facts["owner"] = owner
            self.known_facts["owners"] = owner
            self.known_facts["собственники"] = owner

        priority = self._detect_priority(lowered)
        if priority:
            updated_fields.add("priority")
            self.known_facts["priority"] = priority

        remaining_debt = self._detect_remaining_debt(text)
        if remaining_debt:
            updated_fields.add("остаток_долга")
            self.known_facts["remaining_debt"] = remaining_debt
            self.known_facts["остаток_долга"] = remaining_debt

        if kb is not None:
            self.object_type = self._detect_object_type(lowered)
            if self.object_type:
                updated_fields.add("вид_объекта")
                self.known_facts["вид_объекта"] = self.object_type
            self.scenario = self._detect_scenario(lowered, kb)
            if self.scenario:
                updated_fields.add("сценарий")
                self.known_facts["сценарий"] = self.scenario

        self.plan_name = self._resolve_plan_name(lowered)
        self.known_facts["plan_name"] = self.plan_name

        self._advance_stage()
        if kb is not None:
            self.next_required_field = self._resolve_next_required_field(kb)
        return updated_fields

    def update_from_agent(
        self,
        reply_tts: str,
        next_step: str,
        *,
        kb: KnowledgeBase | None = None,
        current_node: str | None = None,
    ) -> None:
        self.last_agent_text = reply_tts.strip()
        if current_node:
            self.current_node = current_node.strip()
        self.awaiting_field = self._detect_awaiting_field(reply_tts, next_step)
        self._advance_stage()
        if kb is not None:
            self.next_required_field = self._resolve_next_required_field(kb)

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
            "current_node": self.current_node,
            "scenario": self.scenario,
            "plan_name": self.plan_name,
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
    def _amount_needs_clarification(amount_text: str) -> bool:
        parts = amount_text.lower().split()
        if not parts:
            return False
        raw_num = parts[0].replace(",", ".")
        try:
            value = float(raw_num)
        except Exception:
            return False
        unit = parts[1] if len(parts) > 1 else ""
        if unit.startswith("руб") and value < 1000:
            return True
        if not unit and value < 1000:
            return True
        return False

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
    def _detect_credit_closed(text: str) -> str:
        if any(marker in text for marker in ("не закрыт", "не погашен", "еще плачу", "ещё плачу")):
            return "no"
        if any(marker in text for marker in ("закрыт", "погашен", "уже выплатил", "уже выплатили")):
            return "yes"
        return ""

    @staticmethod
    def _detect_collateral(text: str) -> str:
        if any(marker in text for marker in ("нет недвижимости", "без недвижимости", "ничего нет", "машины нет")):
            return "нет залога"
        if "залог" in text and any(marker in text for marker in ("недвижим", "квартир", "дом", "участ", "земл", "коммерчес")):
            return "недвижимость"
        if "птс" in text or ("залог" in text and any(marker in text for marker in ("автомоб", "машин"))):
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
    def _detect_owner(text: str) -> str:
        if any(marker in text for marker in ("я один", "только я", "один собственник", "собственник я")):
            return "single_owner"
        if any(marker in text for marker in ("жена и я", "мы с женой", "я и жена", "несколько собственников", "я и дочь", "я и сын")):
            return "multiple_owners"
        return ""

    @staticmethod
    def _detect_priority(text: str) -> str:
        if "скорост" in text or "быстро" in text or "срочно" in text:
            return "speed"
        if "ставк" in text or "минимальн" in text or "подешевле" in text:
            return "rate"
        return ""

    @staticmethod
    def _detect_remaining_debt(text: str) -> str:
        for match in _AMOUNT_RE.finditer(text):
            num = match.group("num")
            unit = (match.group("unit") or "").lower()
            if not num:
                continue
            if unit:
                return f"{num} {unit}"
        return ""

    def _detect_city(self, text: str) -> str:
        for pattern, city in _CITY_PATTERNS:
            if pattern.search(text):
                return city
        if self.awaiting_field == "region" or self.current_node == "collect_region":
            match = re.search(
                r"^\s*(?:в|во|из|по)\s+([A-Za-zА-Яа-яЁё-]+(?:\s+[A-Za-zА-Яа-яЁё-]+){0,2})\s*[.!?]?\s*$",
                text,
                flags=re.IGNORECASE,
            )
            if match:
                city = match.group(1).strip(" .,!?:;")
                if city:
                    return city[:1].upper() + city[1:]
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

    def _resolve_plan_name(self, text: str) -> str:
        if self.known_facts.get("credit_closed") == "no":
            return "refinance_or_returning_customer"
        if any(marker in text for marker in ("рефинанс", "остаток долга", "текущий кредит", "ипотек")):
            return "refinance_or_returning_customer"
        return "new_loan"

    def _resolve_next_required_field(self, kb: KnowledgeBase | None) -> str:
        if self.plan_name == "refinance_or_returning_customer":
            credit_closed = str(self.known_facts.get("credit_closed", "")).strip()
            if not credit_closed:
                return "credit_closed"
            if credit_closed == "no":
                refinance_sequence = (
                    "остаток_долга",
                    "вид_объекта",
                    "регион",
                    "обременение",
                    "собственники",
                    "priority",
                )
                for field in refinance_sequence:
                    if not str(self.known_facts.get(_field_alias(field), "")).strip() and not str(self.known_facts.get(field, "")).strip():
                        return field
                return ""

        if kb is not None:
            fallback = kb.next_required_field(self.snapshot())
            if fallback:
                return fallback

        for field in ("нужная_сумма", "вид_объекта", "регион", "обременение", "собственники", "priority"):
            if not str(self.known_facts.get(_field_alias(field), "")).strip() and not str(self.known_facts.get(field, "")).strip():
                return field
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


def _field_alias(field: str) -> str:
    aliases = {
        "нужная_сумма": "amount",
        "вид_объекта": "object_type",
        "регион": "region",
        "обременение": "collateral",
        "собственники": "owners",
        "остаток_долга": "remaining_debt",
    }
    return aliases.get(field, field)
