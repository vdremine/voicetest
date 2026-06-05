from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from agent_core import DialogueState, KnowledgeBase, SessionMemory, ToolGraphRuntime
from voice_loop import OpenAiLlmService, TranscriptNormalizer, VoicePipelineConfig


def log(message: str) -> None:
    print(f"[text-api] {message}", flush=True)


@dataclass(slots=True)
class DialogueHarnessSession:
    dialogue_state: DialogueState = field(default_factory=DialogueState)
    session_memory: SessionMemory = field(default_factory=lambda: SessionMemory(max_turns=12))


class StartSessionRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    lead_profile: dict[str, Any] = Field(default_factory=dict)


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)
    lead_profile: dict[str, Any] = Field(default_factory=dict)


class ResetRequest(BaseModel):
    session_id: str = Field(..., min_length=1)


app = FastAPI(title="Voice Agent Text LLM Debug", version="0.1.0")

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

try:
    _graph_path = Path(os.getenv("TOOL_GRAPH_PATH", str(_config.data_dir / "tool_graph.json")))
    _tool_graph = ToolGraphRuntime.load(graph_path=_graph_path, agent_name="Влад+имир")
    log(f"loaded tool graph from {_graph_path}")
except Exception as exc:
    log(f"failed to load tool graph: {exc}")
    _tool_graph = None

_warmed_up = False


def _ensure_session(session_id: str) -> DialogueHarnessSession:
    session = _sessions.get(session_id)
    if session is None:
        session = DialogueHarnessSession()
        if _tool_graph is not None:
            session.dialogue_state.current_node = _tool_graph.start_node
        _sessions[session_id] = session
    return session


def _remember_question(memory: SessionMemory, text: str) -> None:
    value = text.strip()
    if not value:
        return
    if "?" in value or value.lower().startswith(("подскажите", "скажите", "какая", "какой", "кто", "в каком")):
        memory.remember_question(value)


def _build_context_sentence(facts: dict[str, Any]) -> str:
    context = str(facts.get("last_contact_context", "")).strip()
    if context:
        context = context.rstrip(".!? ")
        return f"Мы с вами уже говорили {context}."

    amount_phrase = str(facts.get("lead_amount_phrase", "")).strip()
    property_hint = str(facts.get("property_hint", "")).strip()
    object_type = str(facts.get("вид_объекта", "")).strip()
    if amount_phrase and property_hint:
        return f"Мы с вами уже говорили по вопросу кредита на {amount_phrase}. Тогда речь шла про {property_hint}."
    if amount_phrase and object_type:
        return f"Мы с вами уже говорили по вопросу кредита на {amount_phrase} под залог {object_type}."
    if amount_phrase:
        return f"Мы с вами уже говорили по вопросу кредита на {amount_phrase}."
    if property_hint:
        return f"Мы с вами уже говорили по вопросу кредита. Тогда речь шла про {property_hint}."
    return "Мы с вами уже говорили по вопросу кредита."


def _contextual_opening(session: DialogueHarnessSession) -> str:
    facts = session.dialogue_state.known_facts
    if str(facts.get("prefilled_lead", "")).strip() != "yes":
        opening = _tool_graph.opening_prompt() if _tool_graph is not None else None
        return opening[1] if opening else "Алл+о. Да, добрый день. Удобно сейчас коротко обсудить вопрос по кредиту?"
    name = str(facts.get("client_name", "")).strip()
    greeting = f"Да, добрый день, {name}." if name else "Да, добрый день."
    speed_sentence = (
        "Если для вас важна скорость, я уточню только главное и быстро передам кейс эксперту."
        if str(facts.get("speed_emphasis", "")).strip() == "yes"
        else ""
    )
    text = " ".join(
        part
        for part in (
            "Алл+о.",
            greeting,
            "Это Влад+имир, МосИнвестФинанс.",
            _build_context_sentence(facts),
            speed_sentence,
            "Удобно сейчас коротко продолжить?",
        )
        if part
    )
    _remember_question(session.session_memory, "Удобно сейчас коротко продолжить?")
    session.dialogue_state.update_from_agent(text, "", kb=_kb, current_node="callback_reentry")
    session.session_memory.add_assistant(text)
    session.session_memory.sync_from_dialogue_state(
        session.dialogue_state.snapshot(),
        last_question=session.session_memory.last_question,
    )
    return text


def _contextual_identity_reply(session: DialogueHarnessSession) -> str:
    facts = session.dialogue_state.known_facts
    speed_sentence = (
        "Если для вас важна скорость, я уточню только главное и быстро передам кейс эксперту."
        if str(facts.get("speed_emphasis", "")).strip() == "yes"
        else ""
    )
    text = " ".join(
        part
        for part in (
            "Это Влад+имир, МосИнвестФинанс.",
            _build_context_sentence(facts),
            speed_sentence,
            "Вам ещё актуален этот вопрос?",
        )
        if part
    )
    _remember_question(session.session_memory, "Вам ещё актуален этот вопрос?")
    return text


def _state_payload(session: DialogueHarnessSession, snapshot: dict[str, Any]) -> dict[str, Any]:
    session.session_memory.sync_from_dialogue_state(
        snapshot,
        last_question=session.session_memory.last_question,
    )
    return session.session_memory.llm_state_payload()


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
    session = _ensure_session(request.session_id)
    if request.lead_profile:
        session.dialogue_state.bootstrap_lead_profile(request.lead_profile, kb=_kb)
    reply = _contextual_opening(session)
    return {
        "session_id": request.session_id,
        "reply": reply,
        "state": session.session_memory.state_text(),
    }


@app.post("/session/message")
async def chat(request: ChatRequest) -> dict[str, Any]:
    session = _ensure_session(request.session_id)
    if request.lead_profile:
        session.dialogue_state.bootstrap_lead_profile(request.lead_profile, kb=_kb)

    raw_text = request.text.strip()
    normalized_text = _normalizer.normalize(raw_text)
    session.session_memory.add_user(normalized_text or raw_text)

    updated_fields = session.dialogue_state.update_from_user(raw_text, normalized_text, kb=_kb)
    forced_fields = session.dialogue_state.force_capture_expected_slot(raw_text, normalized_text)
    if forced_fields:
        updated_fields |= forced_fields

    snapshot = session.dialogue_state.snapshot()
    facts = snapshot.get("known_facts", {})

    if not session.dialogue_state.last_agent_text and normalized_text in {"алло", "ало", "але"}:
        reply = _contextual_opening(session)
        return {
            "session_id": request.session_id,
            "reply": reply,
            "updated_fields": sorted(updated_fields),
            "state": session.session_memory.state_text(),
        }

    if any(marker in normalized_text for marker in ("кто вы", "это кто", "по какому поводу", "по какому вопросу")):
        reply = _contextual_identity_reply(session)
        session.dialogue_state.update_from_agent(reply, "", kb=_kb, current_node="check_convenience")
        session.session_memory.add_assistant(reply)
        session.session_memory.sync_from_dialogue_state(
            session.dialogue_state.snapshot(),
            last_question=session.session_memory.last_question,
        )
        return {
            "session_id": request.session_id,
            "reply": reply,
            "updated_fields": sorted(updated_fields),
            "state": session.session_memory.state_text(),
        }

    knowledge = _kb.retrieve(normalized_text, snapshot, limit=2)
    examples = _kb.relevant_examples(normalized_text, limit=1)
    graph_context = (
        _tool_graph.llm_context_for_text(normalized_text, snapshot)
        if _tool_graph is not None
        else None
    )
    llm_state = _state_payload(session, snapshot)
    llm_reply, latency_ms = await _llm_service.generate_response(
        normalized_text=normalized_text,
        history=session.session_memory.recent_history(limit=6),
        dialogue_state=llm_state,
        knowledge=knowledge,
        truth_rules=_kb.truth_rules,
        examples=examples,
        graph_context=graph_context,
        max_tokens_override=min(120, _config.llm_max_tokens),
    )
    reply = llm_reply.reply_tts.strip() or _config.fallback_complex_text
    _remember_question(session.session_memory, reply)

    current_node = str((graph_context or {}).get("node_name", "")).strip() or session.dialogue_state.current_node
    session.dialogue_state.update_from_agent(reply, llm_reply.next_step, kb=_kb, current_node=current_node)
    session.session_memory.add_assistant(reply)
    session.session_memory.sync_from_dialogue_state(
        session.dialogue_state.snapshot(),
        last_question=session.session_memory.last_question,
    )

    return {
        "session_id": request.session_id,
        "reply": reply,
        "latency_ms": latency_ms,
        "llm_intent": llm_reply.intent,
        "next_step": llm_reply.next_step,
        "search_index": llm_reply.search_index,
        "updated_fields": sorted(updated_fields),
        "known_facts": facts,
        "state": session.session_memory.state_text(),
    }


@app.post("/session/reset")
async def reset(request: ResetRequest) -> dict[str, Any]:
    _sessions.pop(request.session_id, None)
    return {"status": "ok", "session_id": request.session_id}


@app.get("/session/state")
async def state(session_id: str) -> dict[str, Any]:
    session = _ensure_session(session_id)
    session.session_memory.sync_from_dialogue_state(
        session.dialogue_state.snapshot(),
        last_question=session.session_memory.last_question,
    )
    return {
        "session_id": session_id,
        "state": session.session_memory.state_text(),
        "known_facts": session.dialogue_state.snapshot().get("known_facts", {}),
    }
