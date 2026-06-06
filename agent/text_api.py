from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agent_core import DialogueState, KnowledgeBase, SessionMemory
from agent_core.rag import KnowledgeSnippet
from voice_loop import OpenAiLlmService, TranscriptNormalizer, VoicePipelineConfig, try_parse_json_object


def log(message: str) -> None:
    print(f"[text-api] {message}", flush=True)


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


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


FLOW_STEPS: dict[str, FlowStep] = {
    "cold_opening": FlowStep(
        goal="кратко представиться, объяснить повод звонка и проверить, удобно ли говорить дальше",
        awaiting="разрешение говорить дальше",
        forbidden=(
            "не обещай одобрение",
            "не перечисляй все условия сразу",
            "не переходи к сбору сделки без согласия клиента продолжать разговор",
        ),
        fallback_reply="Алло. Это Влад+имир, МосИнвестФинанс. Удобно сейчас коротко поговорить?",
    ),
    "need_detection": FlowStep(
        goal="понять, есть ли вообще интерес к кредиту или рефинансированию и в чём потребность",
        awaiting="актуальна ли потребность и в каком виде",
        forbidden=(
            "не спорь с клиентом",
            "не собирай все данные сделки за один ход",
        ),
        fallback_reply="Подскажите, пожалуйста, вопрос по кредиту для вас вообще актуален?",
    ),
    "collect_name": FlowStep(
        goal="узнать, как обращаться к клиенту",
        awaiting="имя клиента",
        forbidden=(
            "не повторяй вопрос про имя, если имя уже известно",
        ),
        fallback_reply="Подскажите, пожалуйста, как к вам можно обращаться?",
    ),
    "collect_amount": FlowStep(
        goal="узнать, какая сумма нужна клиенту",
        awaiting="нужная сумма",
        forbidden=(
            "не спрашивай сумму, если она уже известна",
            "не задавай второй вопрос про цель в том же ходе",
        ),
        fallback_reply="Подскажите, пожалуйста, какую сумму вы рассматриваете?",
    ),
    "collect_purpose": FlowStep(
        goal="узнать, на какую цель нужна сумма",
        awaiting="цель кредита",
        forbidden=(
            "не спрашивай цель, если она уже известна",
        ),
        fallback_reply="Подскажите, пожалуйста, на какую цель нужна сумма?",
    ),
    "collect_property_type": FlowStep(
        goal="понять, есть ли объект под залог и какой именно объект рассматривается",
        awaiting="тип объекта",
        forbidden=(
            "не спрашивай объект, если он уже известен",
        ),
        fallback_reply="Подскажите, пожалуйста, какая недвижимость у вас есть: квартира, дом, участок или другой объект?",
    ),
    "collect_region": FlowStep(
        goal="узнать, в каком регионе находится объект",
        awaiting="регион объекта",
        forbidden=(
            "не спрашивай регион, если он уже известен",
        ),
        fallback_reply="Подскажите, пожалуйста, в каком регионе находится объект?",
    ),
    "collect_encumbrance": FlowStep(
        goal="узнать, свободен ли объект от залога или есть обременение",
        awaiting="обременение объекта",
        forbidden=(
            "не спрашивай обременение, если оно уже известно",
        ),
        fallback_reply="Подскажите, пожалуйста, объект сейчас свободен от залога или уже в обременении?",
    ),
    "collect_owner": FlowStep(
        goal="понять, кто собственник объекта и есть ли доли",
        awaiting="собственник объекта",
        forbidden=(
            "не спрашивай собственника, если он уже известен",
        ),
        fallback_reply="Подскажите, пожалуйста, собственник объекта вы или есть ещё кто-то?",
    ),
    "pitch_speed": FlowStep(
        goal="коротко объяснить общую механику продукта и мягко сделать акцент на скорости",
        awaiting="что для клиента важнее: скорость или ставка",
        forbidden=(
            "не читай длинную презентацию",
            "не перечисляй все условия без запроса клиента",
        ),
        fallback_reply="Если для вас важна скорость, я быстро передам кейс эксперту. Для вас важнее скорость или минимальная ставка?",
    ),
    "handoff_consent": FlowStep(
        goal="получить согласие на обратный звонок эксперта",
        awaiting="согласие на звонок эксперта",
        forbidden=(
            "не завершай разговор без явного согласия или отказа",
        ),
        fallback_reply="Если вам подходит, я передам информацию эксперту, и он с вами свяжется. Вам это удобно?",
    ),
    "callback_time": FlowStep(
        goal="уточнить удобное время для обратного звонка",
        awaiting="удобное время для звонка",
        forbidden=(
            "не навязывай конкретное время, если клиент сам просит перезвонить позже",
        ),
        fallback_reply="Хорошо. Подскажите, пожалуйста, когда вам будет удобнее принять звонок?",
    ),
    "finish": FlowStep(
        goal="корректно завершить разговор",
        awaiting="ничего",
        forbidden=(
            "не задавай новые вопросы",
        ),
        fallback_reply="Хорошо, договорились. Тогда на этом завершим.",
    ),
}

ALLOWED_STAGES = tuple(FLOW_STEPS.keys())

_TEXT_DIALOGUE_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "reply": {
            "type": "string",
            "description": "Короткий ответ менеджера, 1–2 предложения для телефонного разговора.",
        },
        "intent": {
            "type": "string",
            "description": "Короткий смысловой интент реплики клиента.",
        },
        "next_stage": {
            "type": "string",
            "description": "Следующий этап разговора.",
        },
        "awaiting": {
            "type": "string",
            "description": "Что именно мы сейчас ждём от клиента.",
        },
        "facts_update": {
            "type": "object",
            "description": "Только новые или уточнённые факты по клиенту и сделке.",
        },
        "should_end": {
            "type": "boolean",
            "description": "Нужно ли завершать сессию после этой реплики.",
        },
        "search_index": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Короткие ключи для отладки и поиска похожих кейсов.",
        },
    },
    "required": ["reply", "intent", "next_stage", "awaiting", "facts_update", "should_end", "search_index"],
    "additionalProperties": False,
}

_TEXT_DIALOGUE_SYSTEM_PROMPT = """Ты Влад+имир, голосовой менеджер компании МосИнвестФинанс.
Ты звонишь клиенту сам. Это холодный звонок: на старте известен только номер телефона, а фактов по сделке может не быть.

Верни только один JSON-объект без markdown и без текста вне JSON:
{"reply":"...","intent":"...","next_stage":"...","awaiting":"...","facts_update":{},"should_end":false,"search_index":["..."]}

Правила:
- говори только по-русски;
- отвечай коротко, как по телефону: обычно 1–2 предложения;
- сначала ответь по смыслу реплики клиента, потом продвинь разговор только на один шаг;
- за один ход задавай не больше одного нового вопроса;
- не спрашивай уже известные факты;
- не повторяй приветствие после первого сообщения;
- не обещай одобрение кредита;
- не придумывай заявку, если её нет в известных фактах;
- если клиент просит перезвонить позже, перейди к уточнению времени, а не продолжай квалификацию;
- если клиент не хочет говорить, уважительно завершай разговор.

Разрешённые next_stage:
- cold_opening
- need_detection
- collect_name
- collect_amount
- collect_purpose
- collect_property_type
- collect_region
- collect_encumbrance
- collect_owner
- pitch_speed
- handoff_consent
- callback_time
- finish

Разрешённые ключи в facts_update:
- client_name
- desired_amount
- purpose
- property_type
- region
- encumbrance
- owner_status
- permission_to_continue
- interest_confirmed
- priority
- callback_consent
- callback_time
- objection
"""


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
    stage: str = Field(default="cold_opening")
    known_facts: dict[str, Any] = Field(default_factory=dict)


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)


class ResetRequest(BaseModel):
    session_id: str = Field(..., min_length=1)


app = FastAPI(title="Voice Agent Text LLM Debug", version="0.2.0")

_config = VoicePipelineConfig.from_env()
_normalizer = TranscriptNormalizer()
_llm_service = OpenAiLlmService(_config, log)
_sessions: dict[str, DialogueHarnessSession] = {}

try:
    _kb = KnowledgeBase.load(_config.data_dir)
    log(f"loaded knowledge base from {_config.data_dir}")
except Exception as exc:
    log(f"failed to load knowledge base from {_config.data_dir}: {exc}")
    _kb = KnowledgeBase.default()

_warmed_up = False


def _flow_step(stage: str) -> FlowStep:
    return FLOW_STEPS.get(stage, FLOW_STEPS["cold_opening"])


def _fresh_session(*, phone: str, stage: str) -> DialogueHarnessSession:
    normalized_stage = stage if stage in ALLOWED_STAGES else "cold_opening"
    session = DialogueHarnessSession(phone=_clean_text(phone), stage=normalized_stage)
    session.awaiting = _flow_step(normalized_stage).awaiting
    session.dialogue_state.current_node = normalized_stage
    session.dialogue_state.known_facts["phone"] = session.phone
    return session


def _remember_question(memory: SessionMemory, text: str) -> None:
    value = _clean_text(text)
    if not value:
        return
    if "?" in value or value.lower().startswith(("подскажите", "скажите", "какая", "какой", "когда", "кто", "вам")):
        memory.remember_question(value)


def _facts_summary(session: DialogueHarnessSession) -> str:
    facts = session.dialogue_state.known_facts
    parts: list[str] = [f"телефон: {session.phone or 'неизвестно'}"]
    if _clean_text(facts.get("client_name")):
        parts.append(f"имя: {facts['client_name']}")
    if _clean_text(facts.get("amount")) or _clean_text(facts.get("нужная_сумма")):
        parts.append(f"сумма: {facts.get('amount') or facts.get('нужная_сумма')}")
    if _clean_text(facts.get("goal")) or _clean_text(facts.get("цель")):
        parts.append(f"цель: {facts.get('goal') or facts.get('цель')}")
    if _clean_text(facts.get("вид_объекта")):
        parts.append(f"объект: {facts['вид_объекта']}")
    if _clean_text(facts.get("region")) or _clean_text(facts.get("регион")):
        parts.append(f"регион: {facts.get('region') or facts.get('регион')}")
    if _clean_text(facts.get("collateral")) or _clean_text(facts.get("обременение")):
        parts.append(f"обременение: {facts.get('collateral') or facts.get('обременение')}")
    if _clean_text(facts.get("owner")) or _clean_text(facts.get("owners")) or _clean_text(facts.get("собственники")):
        parts.append(
            f"собственник: {facts.get('owner') or facts.get('owners') or facts.get('собственники')}"
        )
    if _clean_text(facts.get("priority")):
        parts.append(f"приоритет: {facts['priority']}")
    if _clean_text(facts.get("callback_time")):
        parts.append(f"callback: {facts['callback_time']}")
    if _clean_text(facts.get("objection")):
        parts.append(f"возражение: {facts['objection']}")
    return "; ".join(parts)


def _session_snapshot(session: DialogueHarnessSession) -> dict[str, Any]:
    facts = dict(session.dialogue_state.known_facts)
    facts["phone"] = session.phone
    return {
        "stage": session.stage,
        "current_node": session.stage,
        "scenario": session.dialogue_state.scenario,
        "plan_name": session.dialogue_state.plan_name or "new_loan",
        "object_type": session.dialogue_state.object_type,
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


def _format_known_facts(session: DialogueHarnessSession) -> str:
    facts = dict(session.dialogue_state.known_facts)
    facts["phone"] = session.phone
    if not facts:
        return "- пока ничего не известно"
    lines: list[str] = []
    for key, value in facts.items():
        cleaned = _clean_text(value)
        if cleaned:
            lines.append(f"- {key}: {cleaned}")
    return "\n".join(lines) if lines else "- пока ничего не известно"


def _format_rules(snippets: list[KnowledgeSnippet]) -> str:
    if not snippets:
        return "- дополнительных правил не найдено"
    return "\n".join(f"- {snippet.text}" for snippet in snippets[:2])


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


def _build_flow_prompt(
    session: DialogueHarnessSession,
    *,
    flow_step: FlowStep,
    user_text: str,
    knowledge: list[KnowledgeSnippet],
    examples: list[list[dict[str, str]]],
) -> str:
    forbidden = "\n".join(f"- {item}" for item in flow_step.forbidden) or "- нет"
    user_fragment = _clean_text(user_text) or "(пусто: нужно самому начать разговор)"
    return f"""
Известные факты:
{_format_known_facts(session)}

Текущий этап:
{session.stage}

Задача текущего шага:
{flow_step.goal}

Чего мы ждём от клиента:
{flow_step.awaiting}

Запрещено:
{forbidden}

Фраза клиента:
{user_fragment}

Короткие продуктовые правила:
{_format_rules(knowledge)}

Похожие диалоговые примеры:
{_format_examples(examples)}

Правила ответа:
- ответь сначала по смыслу фразы клиента;
- затем продвинь разговор только на один шаг;
- не задавай больше одного вопроса;
- не спрашивай уже известные факты;
- говори коротко, как по телефону;
- не обещай одобрение кредита;
- если фраза клиента пустая, сам начни звонок в рамках текущего этапа.
""".strip()


def _parse_facts_update(raw_value: Any) -> dict[str, Any]:
    if not isinstance(raw_value, dict):
        return {}
    result: dict[str, Any] = {}
    for key, value in raw_value.items():
        cleaned_key = _clean_text(key)
        if not cleaned_key:
            continue
        if isinstance(value, bool):
            result[cleaned_key] = "yes" if value else "no"
            continue
        cleaned_value = _clean_text(value)
        if cleaned_value:
            result[cleaned_key] = cleaned_value
    return result


def _apply_known_facts(session: DialogueHarnessSession, updates: dict[str, Any]) -> None:
    state = session.dialogue_state
    facts = state.known_facts
    for key, raw_value in updates.items():
        value = _clean_text(raw_value)
        if not value:
            continue
        lowered_value = value.lower().replace("ё", "е")
        if key == "client_name":
            state.name = value
            facts["client_name"] = value
        elif key == "desired_amount":
            amount = DialogueState._extract_amount(lowered_value) or value
            state.amount_text = amount
            facts["amount"] = amount
            facts["нужная_сумма"] = amount
        elif key == "purpose":
            state.goal = value
            facts["goal"] = value
            facts["цель"] = value
        elif key == "property_type":
            state.object_type = value
            facts["вид_объекта"] = value
        elif key == "region":
            state.city = value
            facts["region"] = value
            facts["регион"] = value
        elif key == "encumbrance":
            state.collateral = value
            facts["collateral"] = value
            facts["обременение"] = value
        elif key == "owner_status":
            facts["owner"] = value
            facts["owners"] = value
            facts["собственники"] = value
        elif key == "priority":
            facts["priority"] = value
        elif key == "callback_time":
            state.callback_time = value
            facts["callback_time"] = value
        elif key == "permission_to_continue":
            flag = _bool_to_flag(value) or value
            facts["permission_to_continue"] = flag
        elif key == "interest_confirmed":
            flag = _bool_to_flag(value) or value
            facts["interest_confirmed"] = flag
        elif key == "callback_consent":
            flag = _bool_to_flag(value) or value
            facts["callback_consent"] = flag
        elif key == "objection":
            facts["objection"] = value
            session.session_memory.add_objection(value)
        else:
            facts[key] = value


def _fallback_reply(stage: str) -> str:
    return _flow_step(stage).fallback_reply


async def _run_dialogue_llm(
    session: DialogueHarnessSession,
    *,
    user_text: str,
) -> tuple[dict[str, Any], int]:
    query = _clean_text(user_text) or session.stage
    flow_step = _flow_step(session.stage)
    snapshot = _session_snapshot(session)
    knowledge = _kb.retrieve(query, snapshot, limit=2)
    examples = _kb.relevant_examples(query, limit=2)
    client = _llm_service._ensure_client()
    messages: list[dict[str, str]] = [
        {"role": "system", "content": _TEXT_DIALOGUE_SYSTEM_PROMPT},
        {
            "role": "system",
            "content": _build_flow_prompt(
                session,
                flow_step=flow_step,
                user_text=user_text,
                knowledge=knowledge,
                examples=examples,
            ),
        },
        *[
            {
                "role": str(item.get("role", "user")),
                "content": _clean_text(item.get("text") or item.get("content")),
            }
            for item in session.session_memory.recent_history(limit=6)
            if _clean_text(item.get("text") or item.get("content"))
        ],
    ]

    started_at = time.perf_counter()
    completion = await client.chat.completions.create(
        model=_config.llm_model,
        temperature=0.15,
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
    latency_ms = int((time.perf_counter() - started_at) * 1000)
    return parsed, latency_ms


def _validate_stage(value: Any, *, fallback: str) -> str:
    stage = _clean_text(value)
    return stage if stage in ALLOWED_STAGES else fallback


def _response_payload(session_id: str, session: DialogueHarnessSession, *, reply: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    _sync_session_memory(session)
    payload: dict[str, Any] = {
        "session_id": session_id,
        "reply": reply,
        "stage": session.stage,
        "awaiting": session.awaiting,
        "known_facts": dict(session.dialogue_state.known_facts),
        "state": session.session_memory.state_text(),
    }
    if extra:
        payload.update(extra)
    return payload


def _search_index(value: Any) -> list[str]:
    if isinstance(value, list):
        return [_clean_text(item) for item in value if _clean_text(item)]
    if isinstance(value, str) and _clean_text(value):
        return [_clean_text(value)]
    return []


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
    session = _fresh_session(phone=request.phone, stage=request.stage)
    _apply_known_facts(session, request.known_facts)
    _sync_session_memory(session)
    _sessions[request.session_id] = session

    llm_json, latency_ms = await _run_dialogue_llm(session, user_text="")
    reply = _clean_text(llm_json.get("reply")) or _fallback_reply(session.stage)
    next_stage = _validate_stage(llm_json.get("next_stage"), fallback=session.stage)
    awaiting = _clean_text(llm_json.get("awaiting")) or _flow_step(next_stage).awaiting
    _apply_known_facts(session, _parse_facts_update(llm_json.get("facts_update")))

    session.stage = next_stage
    session.awaiting = awaiting
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
            "intent": _clean_text(llm_json.get("intent")) or "opening",
            "next_stage": session.stage,
            "search_index": _search_index(llm_json.get("search_index")),
            "latency_ms": latency_ms,
            "should_end": bool(llm_json.get("should_end", False)),
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
    session.session_memory.add_user(user_text)

    session.dialogue_state.update_from_user(raw_text, user_text, kb=_kb)
    _sync_session_memory(session)

    llm_json, latency_ms = await _run_dialogue_llm(session, user_text=user_text)
    reply = _clean_text(llm_json.get("reply")) or _fallback_reply(session.stage)
    next_stage = _validate_stage(llm_json.get("next_stage"), fallback=session.stage)
    awaiting = _clean_text(llm_json.get("awaiting")) or _flow_step(next_stage).awaiting
    facts_update = _parse_facts_update(llm_json.get("facts_update"))
    _apply_known_facts(session, facts_update)

    should_end = bool(llm_json.get("should_end", False))
    session.stage = "finish" if should_end else next_stage
    session.awaiting = _flow_step(session.stage).awaiting if should_end else awaiting
    session.dialogue_state.current_node = session.stage
    session.dialogue_state.last_user_text = raw_text
    session.dialogue_state.last_agent_text = reply
    session.dialogue_state.awaiting_field = session.awaiting
    session.dialogue_state.next_required_field = session.awaiting

    if "objection" in facts_update:
        session.session_memory.add_objection(str(facts_update["objection"]))

    _remember_question(session.session_memory, reply)
    session.session_memory.add_assistant(reply)

    return _response_payload(
        request.session_id,
        session,
        reply=reply,
        extra={
            "latency_ms": latency_ms,
            "intent": _clean_text(llm_json.get("intent")) or "unknown",
            "next_stage": session.stage,
            "search_index": _search_index(llm_json.get("search_index")),
            "facts_update": facts_update,
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
        "known_facts": dict(session.dialogue_state.known_facts),
        "history": session.session_memory.recent_history(),
        "state": session.session_memory.state_text(),
    }
