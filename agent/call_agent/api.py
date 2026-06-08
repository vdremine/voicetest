from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .apply import apply_facts
from .graph import make_call_graph
from .graph_spec import DIALOGUE_GRAPH
from .llm import LlmSettings, TurnLlmClient
from .metrics import MetricsCollector
from .post_call import extract_post_call_summary
from .state import CallState


class StartSessionRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    phone: str = Field(..., min_length=3)
    known_facts: dict[str, Any] = Field(default_factory=dict)


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)


class ResetRequest(BaseModel):
    session_id: str = Field(..., min_length=1)


app = FastAPI(title="Call Agent Text API", version="1.0.0")

_settings = LlmSettings.from_env()
_metrics = MetricsCollector(model_name=_settings.model)
_llm_client = TurnLlmClient(_settings)
_call_graph = make_call_graph(llm_client=_llm_client, metrics=_metrics)
_sessions: dict[str, CallState] = {}


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/metrics")
async def metrics() -> dict[str, object]:
    return _metrics.snapshot()


@app.post("/session/start")
async def start_session(request: StartSessionRequest) -> dict[str, Any]:
    known_facts = apply_facts({}, request.known_facts, current_node="call_connected")
    known_facts["phone"] = request.phone

    state: CallState = {
        "session_id": request.session_id,
        "phone": request.phone,
        "current_node": "call_connected",
        "return_to_node": None,
        "last_turn_note": "Звонок только начался. Агент сказал 'Алло.' и ждёт, пока клиент ответит на звонок.",
        "raw_text": "",
        "user_text": "",
        "known_facts": known_facts,
        "history": [
            {"role": "assistant", "content": DIALOGUE_GRAPH["call_connected"].ask},
        ],
        "node_repeat_count": {},
        "reply": DIALOGUE_GRAPH["call_connected"].ask,
        "llm_decision": {},
        "trace": {
            "start": True,
            "note": "start does not call LLM",
        },
    }

    _sessions[request.session_id] = state

    return {
        "session_id": request.session_id,
        "reply": DIALOGUE_GRAPH["call_connected"].ask,
        "current_node": state["current_node"],
        "known_facts": state["known_facts"],
        "last_turn_note": state["last_turn_note"],
        "trace": state["trace"],
    }


@app.post("/session/message")
async def message(request: ChatRequest) -> dict[str, Any]:
    state = _sessions.get(request.session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="session not found; call /session/start first")

    input_state: CallState = {
        **state,
        "raw_text": request.text,
        "user_text": request.text.strip(),
        "trace": {
            "input_node": state.get("current_node"),
            "input_text": request.text,
        },
    }

    output_state = await _call_graph.ainvoke(input_state)
    _sessions[request.session_id] = output_state

    return {
        "session_id": request.session_id,
        "reply": output_state.get("reply", ""),
        "current_node": output_state.get("current_node", ""),
        "known_facts": output_state.get("known_facts", {}),
        "last_turn_note": output_state.get("last_turn_note", ""),
        "history": output_state.get("history", [])[-4:],
        "llm_decision": output_state.get("llm_decision", {}),
        "trace": output_state.get("trace", {}),
    }


@app.post("/session/reset")
async def reset(request: ResetRequest) -> dict[str, Any]:
    _sessions.pop(request.session_id, None)
    return {"status": "ok", "session_id": request.session_id}


@app.get("/session/state")
async def session_state(session_id: str) -> dict[str, Any]:
    state = _sessions.get(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="session not found")
    return state


@app.get("/session/post_call")
async def session_post_call(session_id: str) -> dict[str, Any]:
    state = _sessions.get(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="session not found")

    summary = await extract_post_call_summary(
        llm_client=_llm_client,
        history=state.get("history", []),
        known_facts=state.get("known_facts", {}),
    )
    return summary.model_dump()
