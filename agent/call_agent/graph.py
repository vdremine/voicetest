from __future__ import annotations

from typing import Any

from .apply import apply_facts
from .flow import (
    SOFT_SLOTS,
    DEFER_FLAGS,
    assemble_reply,
    gate_fact_for,
    infer_gate_value,
    is_auto_complete,
    is_yes_no_slot,
    looks_like_time,
    opening_for,
    resolve_branch,
    resolve_focus,
)
from .llm import TurnLlmClient
from .metrics import MetricsCollector
from .state import CallState


def _apply_branch_signal(facts_update: dict[str, Any], signal: str, facts: dict[str, Any]) -> dict[str, Any]:
    """Translate the model's coarse branch hint into a fact the deterministic
    branch resolver understands."""
    merged = dict(facts_update)
    if signal == "vehicle" and not str(facts.get("property_type", "")).strip():
        merged.setdefault("vehicle_interest", "yes")
    elif signal == "partner":
        merged.setdefault("partner_interest", "yes")
    elif signal == "consolidation":
        merged.setdefault("consolidation_intent", "yes")
    return merged


def _ended_kind(facts: dict[str, Any]) -> str:
    if any(
        str(facts.get(k, "")).strip()
        for k in ("callback_consent", "callback_time", "partner_handed")
    ):
        return "success"
    return "refusal"


class CallGraphRunner:
    def __init__(self, *, llm_client: TurnLlmClient, metrics: MetricsCollector) -> None:
        self._llm_client = llm_client
        self._metrics = metrics

    async def ainvoke(self, state: CallState) -> CallState:
        current_node = state.get("current_node", "call_connected")

        if current_node == "call_connected":
            return self._ready_intro(state)

        facts = state.get("known_facts", {})
        branch = resolve_branch(facts)
        focus_before = resolve_focus(facts, branch)

        result = await self._llm_client.understand(
            focus_node=focus_before,
            known_facts=facts,
            history=state.get("history", [])[-4:],
            last_turn_note=state.get("last_turn_note", ""),
            user_text=state.get("user_text", ""),
        )
        self._metrics.record("llm_answer", result.latency_ms)
        self._metrics.record("first_answer_time", result.latency_ms)

        if result.understanding is None:
            return self._graceful_reask(state, focus_before, branch, result)

        return self._commit(state, focus_before, branch, result)

    # --- intro --------------------------------------------------------------

    def _ready_intro(self, state: CallState) -> CallState:
        facts = dict(state.get("known_facts", {}))
        reply = opening_for(facts)
        facts["opening_done"] = "yes"
        next_node = resolve_focus(facts, resolve_branch(facts))
        self._metrics.record("cached_answer", 0)
        self._metrics.record("first_answer_time", 0)
        history = list(state.get("history", []))
        history.append({"role": "assistant", "content": reply})
        return {
            **state,
            "reply": reply,
            "known_facts": facts,
            "current_node": next_node,
            "last_turn_note": "Открытие произнесено, ждём ответа клиента.",
            "history": history[-12:],
            "node_repeat_count": {},
            "trace": {**state.get("trace", {}), "source": "ready_intro"},
        }

    # --- fallback (model returned invalid JSON) -----------------------------

    def _graceful_reask(self, state: CallState, focus_before: str, branch: str, result) -> CallState:
        facts = state.get("known_facts", {})
        repeat = state.get("node_repeat_count", {}).get(focus_before, 0)
        reply = assemble_reply(
            reflection="", answer="", focus_node=focus_before, facts=facts,
            repeat_count=repeat, should_end=False,
        )
        repeat_count = dict(state.get("node_repeat_count", {}))
        repeat_count[focus_before] = repeat + 1
        return {
            **state,
            "reply": reply,
            "current_node": focus_before,
            "node_repeat_count": repeat_count,
            "history": self._append_history(state, reply),
            "trace": {
                **state.get("trace", {}),
                "source": "fallback_reask",
                "parse_error": result.parse_error,
                "raw_llm_output": result.raw_output,
            },
        }

    # --- normal commit ------------------------------------------------------

    def _commit(self, state: CallState, focus_before: str, branch: str, result) -> CallState:
        understanding = result.understanding
        facts = state.get("known_facts", {})
        prior_repeat = state.get("node_repeat_count", {}).get(focus_before, 0)

        facts_update = _apply_branch_signal(understanding.facts_update, understanding.branch_signal, facts)
        new_facts = apply_facts(facts, facts_update, current_node=focus_before)

        # Tail capture: if the client gives the callback time together with the
        # consent ("да, набирайте завтра"), grab the time too so we never re-ask
        # it. Scheduling implies consent.
        user_text = str(state.get("user_text", "")).strip()
        if (
            focus_before in ("handoff_consent", "partner_experience", "callback_time")
            and not str(new_facts.get("callback_time", "")).strip()
            and looks_like_time(user_text)
        ):
            new_facts = dict(new_facts)
            new_facts["callback_time"] = user_text[:80]
            new_facts.setdefault("callback_consent", "да")

        # Name deferral: if the client goes into an objection/branch pivot instead
        # of giving a name, don't keep asking it mid-storm — defer to the handoff.
        objecting = bool(understanding.answer.strip()) or understanding.branch_signal != "none"
        if focus_before == "collect_name" and objecting and not str(new_facts.get("client_name", "")).strip():
            new_facts = {**new_facts, "name_deferred": "yes"}

        # Anti-loop guard (the server bug): never re-ask a slot the client already
        # answered. If the model failed to fill the gate but the client gave a
        # statement (not a counter-question), infer/capture it and move on.
        #  - yes/no slots (encumbrance): capture on the FIRST miss (no re-ask).
        #  - other slots: allow one clarifying re-ask, then capture.
        #  - soft slots (amount, ...): defer instead of capturing raw text.
        gate = gate_fact_for(focus_before)
        threshold = 0 if is_yes_no_slot(focus_before) else 1
        if (
            gate
            and not is_auto_complete(focus_before)
            and not str(new_facts.get(gate, "")).strip()
            and not understanding.answer.strip()  # client answered, didn't ask back
            and str(state.get("user_text", "")).strip()
            and prior_repeat >= threshold
        ):
            if focus_before in SOFT_SLOTS:
                new_facts = {**new_facts, DEFER_FLAGS[focus_before]: "yes"}
            else:
                new_facts = {**new_facts, gate: infer_gate_value(focus_before, state.get("user_text", ""))}

        branch_after = resolve_branch(new_facts)
        if understanding.should_end:
            focus_after = "finish"
        else:
            focus_after = resolve_focus(new_facts, branch_after)
            if is_auto_complete(focus_after):
                gate = gate_fact_for(focus_after)
                if gate:
                    new_facts = {**new_facts, gate: "yes"}

        repeat_count = dict(state.get("node_repeat_count", {}))
        repeat = repeat_count.get(focus_after, 0) if focus_after == focus_before else 0
        reply = assemble_reply(
            reflection=understanding.reflection,
            answer=understanding.answer,
            focus_node=focus_after,
            facts=new_facts,
            repeat_count=repeat,
            should_end=understanding.should_end,
            ended_kind=_ended_kind(new_facts),
        )
        if focus_after == focus_before:
            repeat_count[focus_after] = repeat_count.get(focus_after, 0) + 1
        else:
            repeat_count[focus_after] = 0

        note = (understanding.reflection or "").strip() or f"Перешли к узлу {focus_after}."

        return {
            **state,
            "reply": reply,
            "known_facts": new_facts,
            "current_node": focus_after,
            "node_repeat_count": repeat_count,
            "last_turn_note": note,
            "history": self._append_history(state, reply),
            "llm_decision": understanding.model_dump(),
            "trace": {
                **state.get("trace", {}),
                "source": result.source,
                "llm_latency_ms": result.latency_ms,
                "parse_error": result.parse_error,
                "focus_before": focus_before,
                "focus_after": focus_after,
                "branch": branch_after,
                "facts_update": facts_update,
                "understanding": understanding.model_dump(),
            },
        }

    def _append_history(self, state: CallState, reply: str) -> list[dict[str, str]]:
        history = list(state.get("history", []))
        user_text = state.get("user_text", "").strip()
        if user_text:
            history.append({"role": "user", "content": user_text})
        if reply.strip():
            history.append({"role": "assistant", "content": reply.strip()})
        return history[-12:]


def make_call_graph(*, llm_client: TurnLlmClient, metrics: MetricsCollector) -> CallGraphRunner:
    return CallGraphRunner(llm_client=llm_client, metrics=metrics)
