from __future__ import annotations

from typing import Any

from .apply import apply_facts, sanitize_facts_update
from .graph_spec import DIALOGUE_GRAPH
from .llm import TurnLlmClient
from .metrics import MetricsCollector
from .schema import LlmTurnDecision
from .state import CallState


def choose_next_node(
    *,
    current_node: str,
    decision_next_node: str | None,
    node_complete: bool,
    temporary_exit: bool,
    should_end: bool,
) -> str:
    if should_end:
        return "finish"

    if current_node == "call_connected":
        return "cold_opening"

    if temporary_exit:
        return current_node

    if not node_complete:
        return current_node

    node = DIALOGUE_GRAPH[current_node]
    if decision_next_node in node.allowed_next:
        return decision_next_node
    return current_node


class CallGraphRunner:
    def __init__(self, *, llm_client: TurnLlmClient, metrics: MetricsCollector) -> None:
        self._llm_client = llm_client
        self._metrics = metrics

    async def ainvoke(self, state: CallState) -> CallState:
        after_llm = await self._llm_turn_node(state)
        after_apply = await self._apply_node(after_llm)
        after_commit = await self._commit_node(after_apply)
        return after_commit

    async def _llm_turn_node(self, state: CallState) -> CallState:
        current_node = state.get("current_node", "call_connected")
        repeat_count = state.get("node_repeat_count", {}).get(current_node, 0)

        decision, latency_ms = await self._llm_client.call_turn_llm(
            current_node=current_node,
            user_text=state.get("user_text", ""),
            known_facts=state.get("known_facts", {}),
            history=state.get("history", [])[-4:],
            node_repeat_count=repeat_count,
        )
        self._metrics.record("llm_answer", latency_ms)
        self._metrics.record("first_answer_time", latency_ms)

        trace = {
            **state.get("trace", {}),
            "llm_latency_ms": latency_ms,
            "llm_decision": decision.model_dump(),
        }
        return {
            **state,
            "reply": decision.reply,
            "llm_decision": decision.model_dump(),
            "return_to_node": decision.return_to_node,
            "trace": trace,
        }

    async def _apply_node(self, state: CallState) -> CallState:
        current_node = state.get("current_node", "call_connected")
        decision = LlmTurnDecision.model_validate(state.get("llm_decision", {}))

        clean_facts_update = sanitize_facts_update(decision.facts_update)
        facts = apply_facts(
            state.get("known_facts", {}),
            clean_facts_update,
        )

        next_node = choose_next_node(
            current_node=current_node,
            decision_next_node=decision.next_node,
            node_complete=decision.node_complete,
            temporary_exit=decision.temporary_exit,
            should_end=decision.should_end,
        )

        repeat = dict(state.get("node_repeat_count", {}))
        if next_node == current_node:
            repeat[current_node] = repeat.get(current_node, 0) + 1
        else:
            repeat[next_node] = 0

        trace = {
            **state.get("trace", {}),
            "facts_after_apply": facts,
            "next_node_after_apply": next_node,
            "repeat_count": repeat,
            "sanitized_facts_update": clean_facts_update,
        }

        return {
            **state,
            "known_facts": facts,
            "current_node": next_node,
            "node_repeat_count": repeat,
            "trace": trace,
        }

    async def _commit_node(self, state: CallState) -> CallState:
        history = list(state.get("history", []))
        user_text = state.get("user_text", "").strip()
        reply = state.get("reply", "").strip()

        if user_text:
            history.append({"role": "user", "content": user_text})
        if reply:
            history.append({"role": "assistant", "content": reply})

        trace = {
            **state.get("trace", {}),
            "committed": True,
            "history_length": len(history[-12:]),
        }
        return {
            **state,
            "history": history[-12:],
            "trace": trace,
        }


def make_call_graph(*, llm_client: TurnLlmClient, metrics: MetricsCollector) -> CallGraphRunner:
    return CallGraphRunner(llm_client=llm_client, metrics=metrics)
