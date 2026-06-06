from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agent_core import DialogueState, KnowledgeBase, SessionMemory
from agent_core.rag import KnowledgeSnippet
from voice_loop import (
    OpenAiLlmService,
    TranscriptNormalizer,
    VoicePipelineConfig,
    normalize_for_compare,
    try_parse_json_object,
)


def log(message: str) -> None:
    print(f"[text-api] {message}", flush=True)


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _contains_any(text: str, markers: tuple[str, ...]) -> bool:
    return any(marker in text for marker in markers)


def _has_word(text: str, word: str) -> bool:
    return re.search(rf"(^|\s){re.escape(word)}($|\s)", text) is not None


def _bool_to_flag(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    lowered = _clean_text(value).lower()
    if lowered in {"1", "true", "yes", "да", "y"}:
        return "yes"
    if lowered in {"0", "false", "no", "нет", "n"}:
        return "no"
    return ""


@dataclass(frozen=True, slots=True)
class FlowStep:
    goal: str
    awaiting: str
    forbidden: tuple[str, ...]
    fallback_reply: str


@dataclass(frozen=True, slots=True)
class Skillbase:
    seller_role: str
    seller_style: str
    ready_answers: tuple[str, ...]
    plan: tuple[str, ...]
    facts: tuple[str, ...]
    rules: tuple[str, ...]


SKILLBASE = Skillbase(
    seller_role=(
        "Тебя зовут Влад+имир. Ты голосовой оператор и кредитный брокер компании "
        "МосИнвестФинанс. Ты звонишь клиенту сам. Твоя задача — спокойно пройти "
        "короткую квалификацию и передать клиента эксперту."
    ),
    seller_style=(
        "Говори живо, как оператор по телефону. Не звучать как официальный банк, юрист "
        "или жёсткий скрипт. Используй короткие связки: 'угу', 'понял вас', 'смотрите', "
        "'тогда такой момент'. Можно неполными фразами. Больше слушай, чем говори. "
        "Отражай ответ клиента перед следующим вопросом. Не используй канцелярит. "
        "Не говори 'актуальна ли потребность', 'цель кредита', 'зачем вам деньги'. "
        "Не больше одного вопроса за ход. Цифры лучше писать прописью."
    ),
    ready_answers=(
        "Алло.",
        "Да, добрый день. Это Влад+имир, МосИнвестФинанс. Мы по кредитам под залог недвижимости, хотел буквально пару моментов уточнить — удобно?",
    ),
    plan=(
        "узнать сумму",
        "узнать имя",
        "узнать, какая недвижимость в собственности",
        "узнать регион недвижимости",
        "узнать, есть ли залог, ипотека или другое обременение",
        "если есть обременение — остаток, ставка, платёж, просрочки, кредитор",
        "узнать, кто собственник",
        "коротко объяснить условия",
        "спросить, что важнее: скорость или минимальная ставка",
        "передать эксперту",
    ),
    facts=(
        "сумма может быть до семидесяти процентов от рыночной стоимости",
        "срок от одного года до двадцати пяти лет",
        "ставка от девятнадцати процентов годовых",
        "официальное трудоустройство не всегда требуется",
        "решение обычно за один-два дня после документов",
        "клиент остаётся собственником",
        "оригиналы документов остаются у клиента",
        "есть направление кредитования под ПТС",
    ),
    rules=(
        "не спрашивай клиента, на какую цель нужны деньги",
        "не обещай одобрение",
        "не говори про покупку недвижимости, если клиент сам этого не сказал",
        "не придумывай заявку, объект, сумму, регион или собственников",
        "если недвижимости нет — предложи ПТС",
        "если клиент не хочет раскрывать цель, не завершай разговор",
    ),
)


FLOW_STEPS: dict[str, FlowStep] = {
    "call_connected": FlowStep(
        goal="просто поздороваться после соединения",
        awaiting="ответ клиента на приветствие",
        forbidden=(
            "не презентуй компанию",
            "не говори про покупку недвижимости",
            "не задавай вопросы по сделке",
        ),
        fallback_reply="Алло.",
    ),
    "cold_opening": FlowStep(
        goal="коротко сказать кто мы и проверить, можно ли продолжить",
        awaiting="разрешение говорить дальше",
        forbidden=(
            "не говори про покупку недвижимости",
            "не обещай одобрение",
            "не говори про хорошую кредитную историю, если это неизвестно",
            "не перечисляй условия",
            "не спрашивай цель денег",
        ),
        fallback_reply="Да, добрый день. Это Влад+имир, МосИнвестФинанс. Мы по кредитам под залог недвижимости, хотел буквально пару моментов уточнить — удобно?",
    ),
    "collect_amount": FlowStep(
        goal="узнать сумму, которую клиент примерно рассматривает",
        awaiting="нужная сумма",
        forbidden=(
            "не спрашивай цель денег",
            "не спрашивай объект в этом же вопросе",
            "не обещай одобрение",
        ),
        fallback_reply="Скажите, какую сумму примерно рассматриваете?",
    ),
    "collect_name": FlowStep(
        goal="узнать, как обращаться к клиенту",
        awaiting="имя клиента",
        forbidden=(
            "не спрашивай фамилию",
            "не дави, если клиент не хочет называть полные данные",
        ),
        fallback_reply="Угу, понял. А как я могу к вам обращаться?",
    ),
    "collect_property_type": FlowStep(
        goal="понять, какая недвижимость есть в собственности",
        awaiting="тип недвижимости",
        forbidden=(
            "не спрашивай цель денег",
            "не ограничивай клиента только квартирой или домом",
            "не спорь с типом объекта",
        ),
        fallback_reply="А какая недвижимость у вас в собственности? Квартира, дом, земля, доля — вообще что есть?",
    ),
    "collect_region": FlowStep(
        goal="узнать регион недвижимости",
        awaiting="регион недвижимости",
        forbidden=(
            "не спрашивай цель денег",
            "не спрашивай обременение в том же вопросе",
        ),
        fallback_reply="Угу, понял. А в каком регионе находится?",
    ),
    "collect_encumbrance": FlowStep(
        goal="узнать, есть ли залог, ипотека, арест или другое обременение",
        awaiting="обременение",
        forbidden=(
            "не спрашивай цель денег",
            "не спрашивай собственника в том же вопросе",
        ),
        fallback_reply="А такой момент: она сейчас в залоге где-то? Ипотека, банк, может быть?",
    ),
    "collect_encumbrance_details": FlowStep(
        goal="если есть обременение, уточнить основные параметры",
        awaiting="детали обременения",
        forbidden=(
            "не задавай длинный список вопросов",
            "задай только один вопрос",
        ),
        fallback_reply="Понял. А остаток долга примерно какой?",
    ),
    "collect_owner": FlowStep(
        goal="узнать, кто собственник",
        awaiting="собственник",
        forbidden=(
            "не спрашивай цель денег",
            "не требуй документы",
        ),
        fallback_reply="И по собственникам: вы один собственник или ещё кто-то есть?",
    ),
    "pitch_conditions": FlowStep(
        goal="коротко объяснить условия и безопасность",
        awaiting="реакция клиента на условия",
        forbidden=(
            "не читай длинную лекцию",
            "не обещай одобрение",
            "не говори 'точно выдадим'",
        ),
        fallback_reply="Смотрите, по таким параметрам можно рассматривать кредит под залог. Обычно это до семидесяти процентов от рыночной стоимости, срок до двадцати пяти лет, ставка от девятнадцати процентов. Точнее уже эксперт рассчитает.",
    ),
    "priority_choice": FlowStep(
        goal="понять, что важнее: скорость или ставка",
        awaiting="приоритет клиента",
        forbidden=(
            "не повторяй все условия",
        ),
        fallback_reply="А для вас сейчас что важнее — скорость или минимальная ставка?",
    ),
    "handoff_consent": FlowStep(
        goal="передать клиента эксперту",
        awaiting="согласие на звонок эксперта",
        forbidden=(
            "не завершай без согласия или отказа",
            "не обещай конкретное одобрение",
        ),
        fallback_reply="Тогда я передам информацию эксперту, он уже нормально рассчитает варианты. Вам удобно, если он свяжется?",
    ),
    "callback_time": FlowStep(
        goal="уточнить удобное время",
        awaiting="удобное время звонка",
        forbidden=(
            "не навязывай время",
        ),
        fallback_reply="Угу, понял. А когда удобнее — сегодня, завтра, после обеда?",
    ),
    "no_real_estate_products": FlowStep(
        goal="если недвижимости нет, предложить ПТС",
        awaiting="есть ли автомобиль",
        forbidden=(
            "не продолжай спрашивать про недвижимость",
            "не обещай одобрение",
        ),
        fallback_reply="Понял, тогда по недвижимости не пойдём. А автомобиль у вас есть? Мы ещё можем посмотреть вариант под ПТС.",
    ),
    "collect_vehicle_type": FlowStep(
        goal="узнать, какой автомобиль можно рассмотреть под ПТС",
        awaiting="тип автомобиля",
        forbidden=(
            "не возвращайся к недвижимости",
            "не обещай одобрение",
        ),
        fallback_reply="Угу, тогда можно посмотреть вариант под ПТС. Что за автомобиль у вас?",
    ),
    "collect_vehicle_owner": FlowStep(
        goal="узнать, кто собственник автомобиля",
        awaiting="собственник автомобиля",
        forbidden=(
            "не возвращайся к недвижимости",
        ),
        fallback_reply="Понял. А по машине собственник вы или ещё кто-то есть?",
    ),
    "collect_vehicle_encumbrance": FlowStep(
        goal="узнать, есть ли залог или кредит на автомобиль",
        awaiting="обременение автомобиля",
        forbidden=(
            "не возвращайся к недвижимости",
        ),
        fallback_reply="И такой момент: машина сейчас в кредите или под залогом где-то?",
    ),
    "finish": FlowStep(
        goal="закончить разговор",
        awaiting="ничего",
        forbidden=(
            "не задавай новые вопросы",
        ),
        fallback_reply="Понял вас. Тогда не буду отвлекать, всего доброго.",
    ),
}

ALLOWED_STAGES = tuple(FLOW_STEPS.keys())
ALLOWED_FACT_KEYS = (
    "client_name",
    "desired_amount",
    "property_exists",
    "no_real_estate",
    "property_type",
    "region",
    "encumbrance",
    "remaining_debt",
    "interest_rate",
    "monthly_payment",
    "overdue",
    "creditor",
    "owner_status",
    "vehicle_interest",
    "vehicle_type",
    "vehicle_owner",
    "vehicle_encumbrance",
    "permission_to_continue",
    "interest_confirmed",
    "priority",
    "callback_consent",
    "callback_time",
    "objection",
    "credit_history_issue",
    "executive_proceeding",
)

_TEXT_DIALOGUE_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "reply": {"type": "string"},
        "intent": {"type": "string"},
        "next_stage": {"type": "string"},
        "awaiting": {"type": "string"},
        "facts_update": {"type": "object"},
        "should_end": {"type": "boolean"},
        "search_index": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["reply", "intent", "next_stage", "awaiting", "facts_update", "should_end", "search_index"],
    "additionalProperties": False,
}

_TEXT_DIALOGUE_SYSTEM_PROMPT = f"""
Ты Влад+имир, голосовой оператор и кредитный брокер компании МосИнвестФинанс.

Ты звонишь клиенту сам. На старте может быть известен только номер телефона.
Компания занимается кредитованием под залог недвижимости. Также есть направление под ПТС.

Верни только JSON:
{{"reply":"...","intent":"...","next_stage":"...","awaiting":"...","facts_update":{{}},"should_end":false,"search_index":["..."]}}

ВАЖНО:
next_stage и awaiting ты заполняешь только для debug. Состояние системы ими не управляется.
Не пытайся сам выбирать сценарий. Твоя главная задача — сформулировать короткую живую реплику под текущий этап.

sellerRole:
{SKILLBASE.seller_role}

sellerStyle:
{SKILLBASE.seller_style}

Формула ответа:
1. коротко отрази, что услышал;
2. если есть вопрос или возражение клиента — ответь по сути;
3. мягко задай один следующий вопрос по текущему этапу.

Примеры стиля:
- угу, миллион триста, понял. а как я могу к вам обращаться?
- квартира, понял. а в каком регионе она находится?
- москва, хорошо. а она сейчас в залоге где-то?
- да, понял вас, цель можно не раскрывать. тогда просто по объекту: какая недвижимость в собственности?
- понял, по недвижимости тогда не пойдём. а автомобиль у вас есть? можем посмотреть вариант под ПТС.

Воронка:
1. сумма;
2. имя;
3. какая недвижимость есть в собственности;
4. регион недвижимости;
5. есть ли залог / ипотека / обременение;
6. если есть обременение — остаток, ставка, платёж, просрочки, кредитор;
7. кто собственник;
8. коротко условия;
9. что важнее: скорость или минимальная ставка;
10. передать эксперту.

Виды недвижимости:
Принимай любые объекты, которые клиент называет недвижимостью: квартира, дом, земля, дом с землёй, участок, доля, комната, апартаменты, пентхаус, таунхаус, дача, коммерческая недвижимость, офис, склад, помещение, гараж, машино-место и любые смешанные варианты.
Не спорь с клиентом по типу объекта. Запиши как сказал и двигайся дальше.

Если недвижимости нет:
- не продолжай спрашивать про недвижимость;
- предложи вариант под ПТС;
- спроси, есть ли автомобиль;
- если авто есть, переходи к авто/ПТС;
- если авто нет, мягко завершай или предложи эксперта, если клиент всё равно хочет консультацию.

Факты продукта:
{chr(10).join(f"- {fact}" for fact in SKILLBASE.facts)}

Запрещено:
- обещать одобрение;
- говорить 'у вас хорошая кредитная история', если это неизвестно;
- говорить 'можем одобрить', если данных ещё нет;
- говорить 'покупка недвижимости', если клиент сам этого не сказал;
- спрашивать цель денег;
- завершать разговор только потому, что клиент не хочет раскрывать цель;
- придумывать заявку;
- придумывать объект, регион, сумму или собственника;
- задавать два вопроса за один ход.

Разрешённые ключи facts_update:
{chr(10).join(f"- {item}" for item in ALLOWED_FACT_KEYS)}
""".strip()


@dataclass(slots=True)
class DialogueHarnessSession:
    phone: str = ""
    stage: str = "cold_opening"
    awaiting: str = FLOW_STEPS["cold_opening"].awaiting
    dialogue_state: DialogueState = field(default_factory=DialogueState)
    session_memory: SessionMemory = field(default_factory=lambda: SessionMemory(max_turns=12))


class StartSessionRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    phone: str = Field(..., min_length=3)
    known_facts: dict[str, Any] = Field(default_factory=dict)


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)


class ResetRequest(BaseModel):
    session_id: str = Field(..., min_length=1)


app = FastAPI(title="Voice Agent Text LLM Debug", version="0.3.0")

_config = VoicePipelineConfig.from_env()
_normalizer = TranscriptNormalizer()
_llm_service = OpenAiLlmService(_config, log)
_sessions: dict[str, DialogueHarnessSession] = {}
_warmed_up = False

try:
    _kb = KnowledgeBase.load(_config.data_dir)
    log(f"loaded knowledge base from {_config.data_dir}")
except Exception as exc:
    log(f"failed to load knowledge base from {_config.data_dir}: {exc}")
    _kb = KnowledgeBase.default()


def _flow_step(stage: str) -> FlowStep:
    return FLOW_STEPS.get(stage, FLOW_STEPS["finish"])


def _filtered_known_facts(session: DialogueHarnessSession) -> dict[str, str]:
    facts = dict(session.dialogue_state.known_facts)
    for key in ("goal", "цель", "purpose"):
        facts.pop(key, None)
    facts["phone"] = session.phone
    return {str(key): _clean_text(value) for key, value in facts.items() if _clean_text(value)}


def _facts_summary(session: DialogueHarnessSession) -> str:
    facts = _filtered_known_facts(session)
    parts: list[str] = [f"телефон: {session.phone or 'неизвестно'}"]
    if facts.get("client_name"):
        parts.append(f"имя: {facts['client_name']}")
    if facts.get("amount") or facts.get("нужная_сумма"):
        parts.append(f"сумма: {facts.get('amount') or facts.get('нужная_сумма')}")
    if facts.get("вид_объекта") or facts.get("property_type"):
        parts.append(f"объект: {facts.get('вид_объекта') or facts.get('property_type')}")
    if facts.get("region") or facts.get("регион"):
        parts.append(f"регион: {facts.get('region') or facts.get('регион')}")
    if facts.get("collateral") or facts.get("обременение") or facts.get("encumbrance"):
        parts.append(f"обременение: {facts.get('collateral') or facts.get('обременение') or facts.get('encumbrance')}")
    if facts.get("owner") or facts.get("owners") or facts.get("собственники") or facts.get("owner_status"):
        parts.append(
            "собственник: "
            f"{facts.get('owner') or facts.get('owners') or facts.get('собственники') or facts.get('owner_status')}"
        )
    if facts.get("vehicle_type"):
        parts.append(f"авто: {facts['vehicle_type']}")
    if facts.get("priority"):
        parts.append(f"приоритет: {facts['priority']}")
    if facts.get("callback_time"):
        parts.append(f"перезвон: {facts['callback_time']}")
    return "; ".join(parts)


def _session_snapshot(session: DialogueHarnessSession) -> dict[str, Any]:
    facts = _filtered_known_facts(session)
    object_type = facts.get("вид_объекта") or facts.get("property_type") or session.dialogue_state.object_type
    return {
        "stage": session.stage,
        "current_node": session.stage,
        "scenario": "",
        "plan_name": "new_loan",
        "object_type": object_type,
        "awaiting_field": session.awaiting,
        "next_required_field": session.awaiting,
        "known_facts": facts,
        "summary": _facts_summary(session),
        "last_user_text": session.dialogue_state.last_user_text,
        "last_agent_text": session.dialogue_state.last_agent_text,
    }


def _sync_session_memory(session: DialogueHarnessSession) -> None:
    session.session_memory.sync_from_dialogue_state(
        _session_snapshot(session),
        last_question=session.session_memory.last_question,
    )


def _remember_question(memory: SessionMemory, text: str) -> None:
    value = _clean_text(text)
    if "?" in value:
        memory.remember_question(value)


def _is_non_answer(text: str) -> bool:
    normalized = normalize_for_compare(text)
    return normalized in {"да", "нет", "угу", "ага", "слушаю", "говорите", "алло"}


def _strip_intro_phrase(text: str) -> str:
    value = _clean_text(text)
    if not value:
        return ""
    patterns = (
        r"^(?:у меня|есть|в собственности|у нас|только|просто)\s+",
        r"^(?:это|она|он)\s+",
    )
    for pattern in patterns:
        value = re.sub(pattern, "", value, flags=re.IGNORECASE)
    return value.strip(" .,!?;:-")


def _extract_name_candidate(raw_text: str, *, current_stage: str) -> str:
    explicit_patterns = (
        r"(?:меня зовут|зовут меня|можно ко мне|обращайтесь ко мне)\s+([А-Яа-яЁё-]{2,}(?:\s+[А-Яа-яЁё-]{2,}){0,2})",
    )
    for pattern in explicit_patterns:
        match = re.search(pattern, raw_text, flags=re.IGNORECASE)
        if match:
            candidate = _clean_text(match.group(1)).strip(" .,!?;:-")
            if candidate and not _contains_any(normalize_for_compare(candidate), ("без фамилий",)):
                return candidate

    if current_stage == "collect_name":
        bare_match = re.fullmatch(r"(?:я\s+)?([А-Яа-яЁё-]{2,}(?:\s+[А-Яа-яЁё-]{2,}){0,2})", _clean_text(raw_text))
        if bare_match:
            candidate = _clean_text(bare_match.group(1)).strip(" .,!?;:-")
            if candidate and not _contains_any(normalize_for_compare(candidate), ("без фамилий",)):
                return candidate
    return ""


def _extract_property_phrase(raw_text: str, normalized_text: str) -> str:
    if _contains_any(normalized_text, ("недвижимости нет", "квартиры нет", "дома нет", "ничего нет")):
        return ""
    detected = DialogueState._detect_object_type(normalized_text)
    cleaned = _strip_intro_phrase(raw_text)
    if not cleaned:
        return detected
    if detected and detected not in normalize_for_compare(cleaned):
        return cleaned
    return cleaned or detected


def _extract_region_phrase(raw_text: str, normalized_text: str) -> str:
    city = DialogueState()._detect_city(raw_text)
    if city:
        return city
    cleaned = _strip_intro_phrase(raw_text)
    cleaned = re.sub(r"^(?:в|во|из|по)\s+", "", cleaned, flags=re.IGNORECASE).strip(" .,!?;:-")
    if cleaned and not _is_non_answer(cleaned):
        return cleaned
    return ""


def _extract_owner_phrase(raw_text: str, normalized_text: str) -> str:
    detected = DialogueState._detect_owner(normalized_text)
    if detected:
        return detected
    cleaned = _strip_intro_phrase(raw_text)
    if cleaned and not _is_non_answer(cleaned):
        return cleaned
    return ""


def _extract_encumbrance_phrase(raw_text: str, normalized_text: str) -> str:
    detected = DialogueState._detect_collateral(normalized_text)
    if detected:
        return detected
    cleaned = _strip_intro_phrase(raw_text)
    if cleaned and not _is_non_answer(cleaned):
        return cleaned
    return ""


def _extract_vehicle_phrase(raw_text: str, normalized_text: str) -> str:
    cleaned = _strip_intro_phrase(raw_text)
    if not cleaned or _is_non_answer(cleaned):
        return ""
    if _contains_any(normalized_text, ("машин", "автомоб", "птс", "тойота", "kia", "bmw", "мерсед", "лада", "хендай")):
        return cleaned
    return ""


def _extract_interest_rate(raw_text: str, normalized_text: str) -> str:
    if "ставк" not in normalized_text and "%" not in raw_text:
        return ""
    match = re.search(r"(\d+(?:[.,]\d+)?)\s*%?", raw_text)
    if not match:
        return ""
    return match.group(1).replace(",", ".")


def _extract_monthly_payment(raw_text: str, normalized_text: str) -> str:
    if "плат" not in normalized_text:
        return ""
    detected = DialogueState._detect_remaining_debt(raw_text)
    if detected:
        return detected
    return ""


def _extract_creditor(raw_text: str, normalized_text: str) -> str:
    bank_markers = (
        "сбер",
        "втб",
        "альфа",
        "тинько",
        "газпром",
        "россельхоз",
        "совком",
        "дом рф",
        "уралсиб",
        "открытие",
        "банк",
    )
    for marker in bank_markers:
        if marker in normalized_text:
            return _clean_text(raw_text)
    return ""


def _extract_callback_time(raw_text: str, normalized_text: str) -> str:
    if _contains_any(
        normalized_text,
        (
            "сегодня",
            "завтра",
            "понедельник",
            "вторник",
            "среда",
            "четверг",
            "пятница",
            "суббот",
            "воскрес",
            "утром",
            "вечером",
            "днем",
            "днём",
            "после обеда",
            "после двенадцати",
            "после часа",
            "после двух",
            "после трех",
            "после трёх",
            "после четырех",
            "после четырёх",
            "после пяти",
            "после шести",
        ),
    ) or re.search(r"после\s+\d{1,2}", normalized_text):
        return _clean_text(raw_text)
    return ""


def _extract_priority(normalized_text: str) -> str:
    if _contains_any(normalized_text, ("скорост", "быстро", "срочно", "побыстрее")):
        return "speed"
    if _contains_any(normalized_text, ("ставк", "минимальн", "платеж", "платёж", "подешевле", "минимальный платеж")):
        return "rate"
    return ""


def _permission_flag(normalized_text: str) -> str:
    if _contains_any(normalized_text, ("неудобно", "не сейчас", "не могу", "перезвоните", "позже")):
        return "no"
    if (
        _has_word(normalized_text, "да")
        or _contains_any(normalized_text, ("слушаю", "говорите", "удобно", "можно", "давайте", "продолжайте"))
    ):
        return "yes"
    return ""


def _interest_flag(normalized_text: str) -> str:
    if _contains_any(normalized_text, ("не актуал", "не интересно", "не надо", "не нужно")):
        return "no"
    if _contains_any(normalized_text, ("актуал", "интересует", "интересно", "рассматриваю", "нужно", "надо")):
        return "yes"
    return ""


def _no_real_estate_flag(normalized_text: str) -> str:
    if _contains_any(
        normalized_text,
        (
            "недвижимости нет",
            "квартиры нет",
            "дома нет",
            "ничего в собственности нет",
            "только машина",
            "только автомобиль",
            "по недвижимости не подходит",
        ),
    ):
        return "yes"
    return ""


def _vehicle_negative_flag(normalized_text: str) -> str:
    if _contains_any(normalized_text, ("машины нет", "авто нет", "автомобиля нет", "нет машины", "нет авто")):
        return "no"
    return ""


def _vehicle_interest_flag(normalized_text: str) -> str:
    if _vehicle_negative_flag(normalized_text) == "no":
        return "no"
    if _contains_any(normalized_text, ("машина есть", "автомобиль есть", "под птс", "под авто", "есть машина", "есть авто")):
        return "yes"
    return ""


def _asks_identity_or_reason(normalized_text: str) -> bool:
    return _contains_any(
        normalized_text,
        ("кто вы", "что хотите", "откуда", "зачем звоните", "по какому поводу", "кто это"),
    )


def _objection_text(raw_text: str, normalized_text: str) -> str:
    if _contains_any(
        normalized_text,
        (
            "не хочу разглашать",
            "не хочу говорить",
            "это не важно",
            "непонятно",
            "не дадите",
            "кто вы",
            "что хотите",
            "вы робот",
            "дорого",
            "комиссия",
            "возраст",
            "лет",
            "не на покупку",
        ),
    ):
        return _clean_text(raw_text)
    return ""


def _credit_history_issue_flag(normalized_text: str) -> str:
    if _contains_any(normalized_text, ("плохая кредитная история", "просрочки", "просрочка", "плохая ки", "испорчена история")):
        return "yes"
    return ""


def _executive_proceeding_flag(normalized_text: str) -> str:
    if _contains_any(normalized_text, ("исполнительное производство", "пристав", "арест", "фссп")):
        return "yes"
    return ""


def _capture_turn_updates(
    session: DialogueHarnessSession,
    *,
    current_stage: str,
    raw_text: str,
    normalized_text: str,
    updated_fields: set[str],
) -> dict[str, str]:
    updates: dict[str, str] = {}
    asks_identity = _asks_identity_or_reason(normalized_text)

    amount = _clean_text(session.dialogue_state.amount_text or session.dialogue_state.known_facts.get("amount"))
    if amount and "нужная_сумма" in updated_fields:
        updates["desired_amount"] = amount

    name = _extract_name_candidate(raw_text, current_stage=current_stage)
    if name:
        updates["client_name"] = name

    permission = "" if asks_identity else _permission_flag(normalized_text)
    if permission:
        updates["permission_to_continue"] = permission

    interest = _interest_flag(normalized_text)
    if interest:
        updates["interest_confirmed"] = interest

    priority = _extract_priority(normalized_text)
    if priority:
        updates["priority"] = priority

    if current_stage in {"handoff_consent", "callback_time"}:
        callback_time = _extract_callback_time(raw_text, normalized_text)
        if callback_time:
            updates["callback_time"] = callback_time

    no_real_estate = _no_real_estate_flag(normalized_text)
    if no_real_estate:
        updates["no_real_estate"] = "yes"
        updates["property_exists"] = "no"

    vehicle_interest = _vehicle_interest_flag(normalized_text)
    if vehicle_interest:
        updates["vehicle_interest"] = vehicle_interest

    objection = _objection_text(raw_text, normalized_text)
    if objection:
        updates["objection"] = objection

    if _credit_history_issue_flag(normalized_text):
        updates["credit_history_issue"] = "yes"

    if _executive_proceeding_flag(normalized_text):
        updates["executive_proceeding"] = "yes"

    if current_stage == "collect_property_type" or "вид_объекта" in updated_fields:
        property_type = _extract_property_phrase(raw_text, normalized_text)
        if property_type and "no_real_estate" not in updates:
            updates["property_type"] = property_type
            updates["property_exists"] = "yes"

    if current_stage == "collect_region" or "регион" in updated_fields:
        region = _extract_region_phrase(raw_text, normalized_text)
        if region:
            updates["region"] = region

    if current_stage == "collect_encumbrance" or "обременение" in updated_fields:
        encumbrance = _extract_encumbrance_phrase(raw_text, normalized_text)
        if encumbrance:
            updates["encumbrance"] = encumbrance

    if current_stage == "collect_owner" or "собственники" in updated_fields:
        owner = _extract_owner_phrase(raw_text, normalized_text)
        if owner:
            updates["owner_status"] = owner

    if current_stage == "collect_encumbrance_details" or "остаток_долга" in updated_fields:
        remaining_debt = DialogueState._detect_remaining_debt(raw_text)
        if remaining_debt:
            updates["remaining_debt"] = remaining_debt
        rate = _extract_interest_rate(raw_text, normalized_text)
        if rate:
            updates["interest_rate"] = rate
        payment = _extract_monthly_payment(raw_text, normalized_text)
        if payment:
            updates["monthly_payment"] = payment
        if "просроч" in normalized_text:
            updates["overdue"] = _clean_text(raw_text)
        creditor = _extract_creditor(raw_text, normalized_text)
        if creditor:
            updates["creditor"] = creditor

    if current_stage == "no_real_estate_products":
        if updates.get("vehicle_interest") not in {"yes", "no"}:
            vehicle_negative = _vehicle_negative_flag(normalized_text)
            if vehicle_negative:
                updates["vehicle_interest"] = vehicle_negative

    if current_stage == "collect_vehicle_type":
        vehicle_type = _extract_vehicle_phrase(raw_text, normalized_text)
        if vehicle_type:
            updates["vehicle_type"] = vehicle_type
            updates["vehicle_interest"] = "yes"

    if current_stage == "collect_vehicle_owner":
        vehicle_owner = _extract_owner_phrase(raw_text, normalized_text)
        if vehicle_owner:
            updates["vehicle_owner"] = vehicle_owner

    if current_stage == "collect_vehicle_encumbrance":
        vehicle_encumbrance = _extract_encumbrance_phrase(raw_text, normalized_text)
        if vehicle_encumbrance:
            updates["vehicle_encumbrance"] = vehicle_encumbrance

    if current_stage == "handoff_consent":
        if _has_word(normalized_text, "да") or _contains_any(
            normalized_text,
            ("давайте", "удобно", "хорошо", "пусть свяжется", "пусть перезвонит"),
        ):
            updates["callback_consent"] = "yes"
        elif _contains_any(normalized_text, ("не надо", "не нужно", "не звоните")):
            updates["callback_consent"] = "no"

    return {key: value for key, value in updates.items() if _clean_text(value)}


def _apply_known_facts(session: DialogueHarnessSession, updates: dict[str, Any]) -> None:
    state = session.dialogue_state
    facts = state.known_facts

    for key, raw_value in updates.items():
        value = _clean_text(raw_value)
        if not value:
            continue
        if key == "client_name":
            state.name = value
            facts["client_name"] = value
        elif key == "desired_amount":
            parsed = DialogueState._extract_amount(value.lower().replace("ё", "е")) or value
            state.amount_text = parsed
            facts["amount"] = parsed
            facts["нужная_сумма"] = parsed
        elif key == "property_type":
            state.object_type = value
            facts["property_type"] = value
            facts["вид_объекта"] = value
            facts["property_exists"] = "yes"
            facts.pop("no_real_estate", None)
        elif key == "region":
            state.city = value
            facts["region"] = value
            facts["регион"] = value
        elif key == "encumbrance":
            state.collateral = value
            facts["encumbrance"] = value
            facts["collateral"] = value
            facts["обременение"] = value
        elif key == "remaining_debt":
            facts["remaining_debt"] = value
            facts["остаток_долга"] = value
        elif key == "interest_rate":
            facts["interest_rate"] = value
        elif key == "monthly_payment":
            facts["monthly_payment"] = value
        elif key == "overdue":
            facts["overdue"] = value
        elif key == "creditor":
            facts["creditor"] = value
        elif key == "owner_status":
            facts["owner_status"] = value
            facts["owner"] = value
            facts["owners"] = value
            facts["собственники"] = value
        elif key == "vehicle_interest":
            facts["vehicle_interest"] = _bool_to_flag(value) or value
        elif key == "vehicle_type":
            facts["vehicle_type"] = value
        elif key == "vehicle_owner":
            facts["vehicle_owner"] = value
        elif key == "vehicle_encumbrance":
            facts["vehicle_encumbrance"] = value
        elif key == "property_exists":
            facts["property_exists"] = _bool_to_flag(value) or value
        elif key == "no_real_estate":
            facts["no_real_estate"] = _bool_to_flag(value) or value
            facts["property_exists"] = "no"
            for stale_key in ("property_type", "вид_объекта"):
                facts.pop(stale_key, None)
        elif key == "permission_to_continue":
            facts["permission_to_continue"] = _bool_to_flag(value) or value
        elif key == "interest_confirmed":
            facts["interest_confirmed"] = _bool_to_flag(value) or value
        elif key == "priority":
            facts["priority"] = value
        elif key == "callback_consent":
            facts["callback_consent"] = _bool_to_flag(value) or value
        elif key == "callback_time":
            state.callback_time = value
            facts["callback_time"] = value
        elif key == "objection":
            facts["objection"] = value
            session.session_memory.add_objection(value)
        elif key == "credit_history_issue":
            facts["credit_history_issue"] = _bool_to_flag(value) or value
        elif key == "executive_proceeding":
            facts["executive_proceeding"] = _bool_to_flag(value) or value
        else:
            facts[key] = value

    for key in ("goal", "цель", "purpose"):
        facts.pop(key, None)


def _has_amount(session: DialogueHarnessSession) -> bool:
    facts = session.dialogue_state.known_facts
    return bool(_clean_text(facts.get("amount") or facts.get("нужная_сумма") or facts.get("desired_amount")))


def _has_name(session: DialogueHarnessSession) -> bool:
    return bool(_clean_text(session.dialogue_state.known_facts.get("client_name")))


def _has_property_type(session: DialogueHarnessSession) -> bool:
    facts = session.dialogue_state.known_facts
    return bool(_clean_text(facts.get("вид_объекта") or facts.get("property_type")))


def _has_region(session: DialogueHarnessSession) -> bool:
    facts = session.dialogue_state.known_facts
    return bool(_clean_text(facts.get("region") or facts.get("регион")))


def _has_encumbrance(session: DialogueHarnessSession) -> bool:
    facts = session.dialogue_state.known_facts
    return bool(_clean_text(facts.get("collateral") or facts.get("обременение") or facts.get("encumbrance")))


def _has_owner(session: DialogueHarnessSession) -> bool:
    facts = session.dialogue_state.known_facts
    return bool(_clean_text(facts.get("owner") or facts.get("owners") or facts.get("собственники") or facts.get("owner_status")))


def _has_priority(session: DialogueHarnessSession) -> bool:
    facts = session.dialogue_state.known_facts
    return bool(_clean_text(facts.get("priority")))


def _has_vehicle_type(session: DialogueHarnessSession) -> bool:
    return bool(_clean_text(session.dialogue_state.known_facts.get("vehicle_type")))


def _has_vehicle_owner(session: DialogueHarnessSession) -> bool:
    return bool(_clean_text(session.dialogue_state.known_facts.get("vehicle_owner")))


def _has_vehicle_encumbrance(session: DialogueHarnessSession) -> bool:
    return bool(_clean_text(session.dialogue_state.known_facts.get("vehicle_encumbrance")))


def _has_remaining_debt(session: DialogueHarnessSession) -> bool:
    facts = session.dialogue_state.known_facts
    return bool(_clean_text(facts.get("remaining_debt") or facts.get("остаток_долга")))


def _client_refuses_extra_disclosure(normalized_text: str) -> bool:
    return _contains_any(
        normalized_text,
        (
            "не хочу разглашать",
            "не хочу говорить",
            "это не важно",
            "не так важно",
            "без разницы",
            "не хочу объяснять",
        ),
    )


def _negative_interest(normalized_text: str) -> bool:
    return _contains_any(normalized_text, ("не интересно", "не надо", "не нужно", "не звоните", "до свидания"))


def _positive_encumbrance(enc: str) -> bool:
    return _contains_any(enc, ("ипотек", "залог", "арест", "банк", "обремен", "кредит"))


def _negative_encumbrance(enc: str) -> bool:
    return _contains_any(enc, ("нет", "ни у кого", "без", "чист", "свобод"))


def _bad_reply(*, reply: str, user_text: str, target_stage: str) -> bool:
    norm = normalize_for_compare(reply)
    user_norm = normalize_for_compare(user_text)

    if "покупк" in norm and "покупк" not in user_norm:
        return True
    if "цель" in norm or "зачем вам деньги" in norm:
        return True
    if reply.count("?") > 1:
        return True
    if _contains_any(norm, ("можем одобрить", "точно одобрим", "точно выдадим")):
        return True
    if target_stage == "cold_opening" and _contains_any(
        norm,
        (
            "какую сумму",
            "сумму рассматриваете",
            "как вас зовут",
            "какая недвижимость",
            "в каком регионе",
            "в залоге",
            "ипотека",
            "кто собственник",
            "автомобиль у вас есть",
        ),
    ):
        return True
    if target_stage == "collect_amount" and _contains_any(
        norm,
        (
            "здравствуйте",
            "добрый день",
            "меня зовут влад+имир",
            "мосинвестфинанс",
            "какая недвижимость",
            "есть недвижимость",
            "в каком регионе",
            "кто собственник",
        ),
    ):
        return True
    if target_stage == "collect_name" and _contains_any(
        norm,
        (
            "здравствуйте",
            "добрый день",
            "меня зовут влад+имир",
            "мосинвестфинанс",
            "какая недвижимость",
            "в каком регионе",
            "в залоге",
            "ипотека",
        ),
    ):
        return True
    if target_stage == "collect_property_type" and _contains_any(
        norm,
        (
            "на какую цель",
            "зачем вам деньги",
            "в каком регионе",
            "в залоге",
            "кто собственник",
        ),
    ):
        return True
    return False


def _fallback_reply_for_turn(
    session: DialogueHarnessSession,
    *,
    target_stage: str,
    user_text: str,
) -> str:
    norm = normalize_for_compare(user_text)
    if target_stage == "cold_opening":
        return _flow_step("cold_opening").fallback_reply
    if target_stage == "collect_amount" and "не на покупку" in norm:
        return (
            "Да, понял, речь не про покупку. Мы как раз больше про кредит под уже имеющуюся "
            "недвижимость. Скажите, какую сумму примерно рассматриваете?"
        )
    if target_stage == "collect_property_type" and _client_refuses_extra_disclosure(norm):
        return (
            "Да, понял вас, цель можно не раскрывать. Тогда просто по объекту сориентируюсь: "
            "какая недвижимость у вас в собственности?"
        )
    return _flow_step(target_stage).fallback_reply


def _resolve_next_stage_from_facts(
    session: DialogueHarnessSession,
    *,
    current_stage: str,
    user_text: str,
    should_end: bool = False,
) -> str:
    facts = session.dialogue_state.known_facts
    norm = normalize_for_compare(user_text)

    if should_end:
        return "finish"

    if current_stage == "call_connected":
        return "cold_opening"

    if _negative_interest(norm):
        return "finish"

    if current_stage == "cold_opening":
        if _asks_identity_or_reason(norm):
            return "cold_opening"
        if facts.get("permission_to_continue") == "yes" or facts.get("interest_confirmed") == "yes":
            return "collect_amount"
        return "cold_opening"

    if facts.get("callback_consent") == "yes" and facts.get("callback_time"):
        return "finish"

    if facts.get("callback_time") and current_stage == "callback_time":
        return "finish"

    if facts.get("no_real_estate") == "yes":
        if facts.get("vehicle_interest") == "yes" and not _has_vehicle_type(session):
            return "collect_vehicle_type"
        if facts.get("vehicle_interest") == "yes" and _has_vehicle_type(session) and not _has_vehicle_owner(session):
            return "collect_vehicle_owner"
        if facts.get("vehicle_interest") == "yes" and _has_vehicle_owner(session) and not _has_vehicle_encumbrance(session):
            return "collect_vehicle_encumbrance"
        if (
            facts.get("vehicle_interest") == "yes"
            and _has_vehicle_type(session)
            and _has_vehicle_owner(session)
            and _has_vehicle_encumbrance(session)
            and not _has_amount(session)
        ):
            return "collect_amount"
        if facts.get("vehicle_interest") == "yes" and _has_vehicle_type(session) and _has_vehicle_owner(session) and _has_vehicle_encumbrance(session):
            return "handoff_consent"
        if facts.get("vehicle_interest") == "no":
            return "finish"
        return "no_real_estate_products"

    if current_stage == "no_real_estate_products":
        if facts.get("vehicle_interest") == "yes":
            return "collect_vehicle_type"
        if facts.get("vehicle_interest") == "no":
            return "finish"
        return "no_real_estate_products"

    if current_stage == "collect_amount" and not _has_amount(session) and _client_refuses_extra_disclosure(norm):
        if _has_name(session):
            return "collect_property_type"
        return "collect_property_type"

    if not _has_amount(session):
        return "collect_amount"

    if not _has_name(session):
        return "collect_name"

    if not _has_property_type(session):
        return "collect_property_type"

    if not _has_region(session):
        return "collect_region"

    if not _has_encumbrance(session):
        return "collect_encumbrance"

    enc = normalize_for_compare(
        facts.get("collateral") or facts.get("обременение") or facts.get("encumbrance") or ""
    )
    if _positive_encumbrance(enc) and not _negative_encumbrance(enc):
        if not _has_remaining_debt(session):
            return "collect_encumbrance_details"

    if not _has_owner(session):
        return "collect_owner"

    if current_stage not in {"pitch_conditions", "priority_choice", "handoff_consent", "callback_time", "finish"}:
        return "pitch_conditions"

    if current_stage == "pitch_conditions":
        return "priority_choice"

    if not _has_priority(session):
        return "priority_choice"

    if current_stage == "priority_choice":
        return "handoff_consent"

    if current_stage == "handoff_consent":
        if facts.get("callback_consent") == "yes":
            return "callback_time"
        if facts.get("callback_consent") == "no":
            return "finish"
        return "handoff_consent"

    if current_stage == "callback_time":
        return "callback_time"

    return current_stage


def _format_known_facts(session: DialogueHarnessSession) -> str:
    facts = _filtered_known_facts(session)
    if not facts:
        return "- пока фактов нет"
    return "\n".join(f"- {key}: {value}" for key, value in facts.items())


def _format_examples(examples: list[list[dict[str, str]]]) -> str:
    if not examples:
        return "Нет похожих примеров."
    blocks: list[str] = []
    for index, example in enumerate(examples[:2], start=1):
        excerpt = example[-4:]
        lines = [f"Пример {index}:"]
        for item in excerpt:
            role = _clean_text(item.get("role")) or "user"
            content = _clean_text(item.get("content"))
            if content:
                lines.append(f"{role}: {content}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _format_snippets(snippets: list[KnowledgeSnippet]) -> str:
    if not snippets:
        return "- без дополнительных сниппетов"
    return "\n".join(f"- {snippet.text}" for snippet in snippets[:2])


def _build_llm_messages(
    session: DialogueHarnessSession,
    *,
    target_stage: str,
    user_text: str,
    knowledge: list[KnowledgeSnippet],
    examples: list[list[dict[str, str]]],
) -> list[dict[str, str]]:
    step = _flow_step(target_stage)
    prompt = f"""
Текущий этап:
{target_stage}

Задача этапа:
{step.goal}

Ждём от клиента:
{step.awaiting}

Запрещено:
{chr(10).join(f"- {item}" for item in step.forbidden)}

Известные факты:
{_format_known_facts(session)}

Короткие продуктовые подсказки:
{_format_snippets(knowledge)}

Похожие примеры:
{_format_examples(examples)}

Фраза клиента:
{_clean_text(user_text) or "(на старте пользователь ещё ничего не сказал)"}
""".strip()
    history = [
        {
            "role": str(item.get("role", "user")),
            "content": _clean_text(item.get("text") or item.get("content")),
        }
        for item in session.session_memory.recent_history(limit=6)
        if _clean_text(item.get("text") or item.get("content"))
    ]
    return [
        {"role": "system", "content": _TEXT_DIALOGUE_SYSTEM_PROMPT},
        {"role": "system", "content": prompt},
        *history,
    ]


async def _generate_llm_reply(
    session: DialogueHarnessSession,
    *,
    target_stage: str,
    user_text: str,
) -> tuple[str, dict[str, Any], int]:
    query = _clean_text(user_text) or target_stage
    snapshot = _session_snapshot(session)
    allow_retrieval = bool(normalize_for_compare(user_text)) and target_stage not in {"cold_opening", "collect_amount"}
    knowledge = _kb.retrieve(query, snapshot, limit=2) if allow_retrieval else []
    examples = _kb.relevant_examples(query, limit=2) if allow_retrieval else []
    client = _llm_service._ensure_client()
    messages = _build_llm_messages(
        session,
        target_stage=target_stage,
        user_text=user_text,
        knowledge=knowledge,
        examples=examples,
    )
    started_at = time.perf_counter()
    completion = await client.chat.completions.create(
        model=_config.llm_model,
        temperature=0.18,
        max_tokens=min(220, _config.llm_max_tokens),
        response_format=_llm_service._response_format(_TEXT_DIALOGUE_JSON_SCHEMA),
        messages=messages,
    )
    raw_content = completion.choices[0].message.content or ""
    if isinstance(raw_content, list):
        content = "".join(
            part.get("text", "") if isinstance(part, dict) else getattr(part, "text", "")
            for part in raw_content
        )
    else:
        content = str(raw_content)
    parsed, _ = try_parse_json_object(content)
    reply = _clean_text(parsed.get("reply")) or _fallback_reply_for_turn(
        session,
        target_stage=target_stage,
        user_text=user_text,
    )
    if _bad_reply(reply=reply, user_text=user_text, target_stage=target_stage):
        reply = _fallback_reply_for_turn(
            session,
            target_stage=target_stage,
            user_text=user_text,
        )
    latency_ms = int((time.perf_counter() - started_at) * 1000)
    return reply, parsed, latency_ms


def _effective_facts_update(updates: dict[str, str]) -> dict[str, str]:
    return {key: value for key, value in updates.items() if key in ALLOWED_FACT_KEYS and _clean_text(value)}


def _search_index(value: Any) -> list[str]:
    if isinstance(value, list):
        return [_clean_text(item) for item in value if _clean_text(item)]
    if isinstance(value, str) and _clean_text(value):
        return [_clean_text(value)]
    return []


def _response_payload(
    session_id: str,
    session: DialogueHarnessSession,
    *,
    reply: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    _sync_session_memory(session)
    payload: dict[str, Any] = {
        "session_id": session_id,
        "reply": reply,
        "stage": session.stage,
        "awaiting": session.awaiting,
        "known_facts": _filtered_known_facts(session),
        "state": session.session_memory.state_text(),
    }
    if extra:
        payload.update(extra)
    return payload


def _fresh_session(*, phone: str) -> DialogueHarnessSession:
    session = DialogueHarnessSession(phone=_clean_text(phone), stage="cold_opening")
    session.awaiting = _flow_step("cold_opening").awaiting
    session.dialogue_state.current_node = "cold_opening"
    session.dialogue_state.known_facts["phone"] = session.phone
    return session


@app.on_event("startup")
async def startup() -> None:
    global _warmed_up
    if not _warmed_up:
        try:
            await _llm_service.warmup()
        finally:
            _warmed_up = True


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/session/start")
async def start_session(request: StartSessionRequest) -> dict[str, Any]:
    session = _fresh_session(phone=request.phone)
    _apply_known_facts(session, request.known_facts)
    _sync_session_memory(session)
    _sessions[request.session_id] = session

    reply = _flow_step("call_connected").fallback_reply
    session.stage = "cold_opening"
    session.awaiting = _flow_step("cold_opening").awaiting
    session.dialogue_state.current_node = session.stage
    session.dialogue_state.last_agent_text = reply
    session.dialogue_state.awaiting_field = session.awaiting
    session.dialogue_state.next_required_field = session.awaiting

    _remember_question(session.session_memory, reply)
    session.session_memory.add_assistant(reply)

    return _response_payload(
        request.session_id,
        session,
        reply=reply,
        extra={
            "intent": "call_connected",
            "next_stage": session.stage,
            "search_index": [],
            "latency_ms": 0,
            "facts_update": {},
            "llm_next_stage": "",
            "llm_awaiting": "",
            "llm_facts_update": {},
            "should_end": False,
        },
    )


@app.post("/session/message")
async def chat(request: ChatRequest) -> dict[str, Any]:
    session = _sessions.get(request.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="session not found; call /session/start first")

    raw_text = request.text.strip()
    normalized_text = _normalizer.normalize(raw_text)
    user_text = normalized_text or raw_text
    current_stage = session.stage

    session.session_memory.add_user(user_text)
    updated_fields = session.dialogue_state.update_from_user(raw_text, user_text, kb=_kb)

    deterministic_updates = _effective_facts_update(
        _capture_turn_updates(
            session,
            current_stage=current_stage,
            raw_text=raw_text,
            normalized_text=normalize_for_compare(user_text),
            updated_fields=updated_fields,
        )
    )
    _apply_known_facts(session, deterministic_updates)

    next_stage = _resolve_next_stage_from_facts(
        session,
        current_stage=current_stage,
        user_text=user_text,
        should_end=False,
    )

    reply, llm_json, latency_ms = await _generate_llm_reply(
        session,
        target_stage=next_stage,
        user_text=user_text,
    )

    session.stage = next_stage
    session.awaiting = _flow_step(next_stage).awaiting
    session.dialogue_state.current_node = session.stage
    session.dialogue_state.last_user_text = raw_text
    session.dialogue_state.last_agent_text = reply
    session.dialogue_state.awaiting_field = session.awaiting
    session.dialogue_state.next_required_field = session.awaiting

    _remember_question(session.session_memory, reply)
    session.session_memory.add_assistant(reply)

    should_end = session.stage == "finish"

    return _response_payload(
        request.session_id,
        session,
        reply=reply,
        extra={
            "latency_ms": latency_ms,
            "intent": _clean_text(llm_json.get("intent")) or "unknown",
            "next_stage": session.stage,
            "search_index": _search_index(llm_json.get("search_index")),
            "facts_update": deterministic_updates,
            "llm_next_stage": _clean_text(llm_json.get("next_stage")),
            "llm_awaiting": _clean_text(llm_json.get("awaiting")),
            "llm_facts_update": llm_json.get("facts_update", {}),
            "updated_fields": sorted(updated_fields),
            "should_end": should_end,
        },
    )


@app.post("/session/reset")
async def reset(request: ResetRequest) -> dict[str, Any]:
    _sessions.pop(request.session_id, None)
    return {"status": "ok", "session_id": request.session_id}


@app.get("/session/state")
async def state(session_id: str) -> dict[str, Any]:
    session = _sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="session not found")
    _sync_session_memory(session)
    return {
        "session_id": session_id,
        "phone": session.phone,
        "stage": session.stage,
        "awaiting": session.awaiting,
        "known_facts": _filtered_known_facts(session),
        "history": session.session_memory.recent_history(),
        "state": session.session_memory.state_text(),
    }
