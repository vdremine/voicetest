from __future__ import annotations

from typing import Any

from .apply import apply_facts, sanitize_facts_update
from .graph_spec import DIALOGUE_GRAPH
from .llm import LlmCallResult, TurnLlmClient
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

    if temporary_exit:
        return current_node

    if not node_complete:
        return current_node

    node = DIALOGUE_GRAPH[current_node]
    if decision_next_node in node.allowed_next:
        return decision_next_node

    return current_node


def _decision_structure_errors(
    *,
    current_node: str,
    decision: LlmTurnDecision,
) -> list[str]:
    errors: list[str] = []
    allowed_next = DIALOGUE_GRAPH[current_node].allowed_next

    if not decision.node_complete and decision.next_node is not None:
        errors.append("next_node_set_but_node_complete_false")

    if decision.next_node is not None and decision.next_node not in allowed_next:
        errors.append("next_node_not_allowed")

    if (
        decision.reply_asks_node is not None
        and decision.reply_asks_node != current_node
        and not decision.node_complete
    ):
        errors.append("reply_asks_next_node_but_node_not_complete")

    if (
        decision.reply_asks_node is not None
        and decision.next_node is not None
        and decision.reply_asks_node != current_node
        and decision.reply_asks_node != decision.next_node
    ):
        errors.append("reply_asks_node_mismatch_next_node")

    return errors


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
        known_facts = state.get("known_facts", {})
        history = state.get("history", [])[-4:]
        last_turn_note = state.get("last_turn_note", "")

        if current_node == "call_connected":
            decision = LlmTurnDecision(
                reply=DIALOGUE_GRAPH["cold_opening"].ask,
                heard_summary="Клиент ответил на звонок.",
                turn_note="Клиент взял трубку. Агент представился и перевёл разговор в cold_opening.",
                facts_update={},
                node_complete=True,
                next_node="cold_opening",
                reply_asks_node="cold_opening",
                temporary_exit=False,
                return_to_node=None,
                client_question_answered=False,
                client_resistance=None,
                should_end=False,
                confidence=1.0,
                repeat_note="Ready answer after call pickup.",
                reason="ready_answer_call_connected",
            )
            trace = {
                **state.get("trace", {}),
                "source": "ready_answer",
                "llm_latency_ms": 0,
                "raw_llm_output": "",
                "parse_error": "",
                "last_turn_note": decision.turn_note,
                "llm_decision": decision.model_dump(),
            }
            self._metrics.record("cached_answer", 0)
            self._metrics.record("first_answer_time", 0)
            return {
                **state,
                "reply": decision.reply,
                "llm_decision": decision.model_dump(),
                "return_to_node": decision.return_to_node,
                "trace": trace,
            }

        primary_result = await self._llm_client.call_turn_llm(
            current_node=current_node,
            user_text=state.get("user_text", ""),
            known_facts=known_facts,
            history=history,
            node_repeat_count=repeat_count,
            last_turn_note=last_turn_note,
        )
        self._metrics.record("llm_answer", primary_result.latency_ms)

        total_latency_ms = primary_result.latency_ms
        final_result = primary_result
        json_repair_trace: dict[str, Any] | None = None
        transition_repair_trace: dict[str, Any] | None = None

        if final_result.decision is None:
            repaired_json = await self._llm_client.repair_json_llm(
                current_node=current_node,
                user_text=state.get("user_text", ""),
                known_facts=known_facts,
                history=history,
                node_repeat_count=repeat_count,
                last_turn_note=last_turn_note,
                raw_output=final_result.raw_output,
                parse_error=final_result.parse_error,
            )
            total_latency_ms += repaired_json.latency_ms
            self._metrics.record("llm_repair", repaired_json.latency_ms)
            json_repair_trace = {
                "source": repaired_json.source,
                "raw_llm_output": repaired_json.raw_output,
                "parse_error": repaired_json.parse_error,
                "llm_decision": repaired_json.decision.model_dump() if repaired_json.decision else None,
            }
            if repaired_json.decision is not None:
                final_result = repaired_json

        structure_errors = (
            _decision_structure_errors(current_node=current_node, decision=final_result.decision)
            if final_result.decision is not None
            else []
        )

        if final_result.decision is not None and structure_errors:
            repaired_transition = await self._llm_client.repair_transition_llm(
                current_node=current_node,
                user_text=state.get("user_text", ""),
                known_facts=known_facts,
                history=history,
                node_repeat_count=repeat_count,
                last_turn_note=last_turn_note,
                bad_decision=final_result.decision.model_dump(),
                errors=structure_errors,
            )
            total_latency_ms += repaired_transition.latency_ms
            self._metrics.record("llm_repair", repaired_transition.latency_ms)
            transition_repair_trace = {
                "source": repaired_transition.source,
                "raw_llm_output": repaired_transition.raw_output,
                "parse_error": repaired_transition.parse_error,
                "errors": structure_errors,
                "llm_decision": repaired_transition.decision.model_dump() if repaired_transition.decision else None,
            }
            repaired_errors = (
                _decision_structure_errors(current_node=current_node, decision=repaired_transition.decision)
                if repaired_transition.decision is not None
                else ["repair_returned_invalid_json"]
            )
            if repaired_transition.decision is not None and not repaired_errors:
                final_result = repaired_transition
            else:
                final_result = self._technical_fallback_result(
                    parse_error="; ".join(repaired_errors) or "invalid_decision_after_repair",
                    raw_output=repaired_transition.raw_output or final_result.raw_output,
                )

        elif final_result.decision is not None and self._needs_transition_repair(
            current_node=current_node,
            decision=final_result.decision,
        ):
            repaired_transition = await self._llm_client.repair_transition_llm(
                current_node=current_node,
                user_text=state.get("user_text", ""),
                known_facts=known_facts,
                history=history,
                node_repeat_count=repeat_count,
                last_turn_note=last_turn_note,
                bad_decision=final_result.decision.model_dump(),
                errors=["next_node_not_allowed"],
            )
            total_latency_ms += repaired_transition.latency_ms
            self._metrics.record("llm_repair", repaired_transition.latency_ms)
            transition_repair_trace = {
                "source": repaired_transition.source,
                "raw_llm_output": repaired_transition.raw_output,
                "parse_error": repaired_transition.parse_error,
                "llm_decision": repaired_transition.decision.model_dump() if repaired_transition.decision else None,
            }
            if repaired_transition.decision is not None and not self._needs_transition_repair(
                current_node=current_node,
                decision=repaired_transition.decision,
            ):
                final_result = repaired_transition
            else:
                final_result = self._technical_fallback_result(
                    parse_error="invalid_next_node_after_transition_repair",
                    raw_output=repaired_transition.raw_output or final_result.raw_output,
                )

        if final_result.decision is None:
            final_result = self._technical_fallback_result(
                parse_error=final_result.parse_error,
                raw_output=final_result.raw_output,
            )

        self._metrics.record("first_answer_time", total_latency_ms)
        final_decision = final_result.decision
        assert final_decision is not None

        trace = {
            **state.get("trace", {}),
            "source": final_result.source,
            "llm_latency_ms": total_latency_ms,
            "raw_llm_output": primary_result.raw_output,
            "parse_error": primary_result.parse_error,
            "last_turn_note": final_decision.turn_note or final_decision.heard_summary,
            "llm_decision": final_decision.model_dump(),
            "json_repair": json_repair_trace,
            "transition_repair": transition_repair_trace,
        }

        if final_result.source == "fallback":
            trace["fallback"] = {
                "source": "fallback",
                "parse_error": final_result.parse_error,
                "raw_llm_output": final_result.raw_output,
            }

        return {
            **state,
            "reply": final_decision.reply,
            "llm_decision": final_decision.model_dump(),
            "return_to_node": final_decision.return_to_node,
            "last_turn_note": final_decision.turn_note or final_decision.heard_summary,
            "trace": trace,
        }

    async def _apply_node(self, state: CallState) -> CallState:
        current_node = state.get("current_node", "call_connected")
        decision = LlmTurnDecision.model_validate(state.get("llm_decision", {}))

        clean_facts_update = sanitize_facts_update(
            decision.facts_update,
            current_node=current_node,
        )
        facts = apply_facts(
            state.get("known_facts", {}),
            clean_facts_update,
            current_node=current_node,
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
            "sanitized_facts_update": clean_facts_update,
            "facts_after_apply": facts,
            "next_node_after_apply": next_node,
            "repeat_count": repeat,
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
            "last_turn_note": state.get("last_turn_note", ""),
            "trace": trace,
        }

    def _needs_transition_repair(
        self,
        *,
        current_node: str,
        decision: LlmTurnDecision,
    ) -> bool:
        if decision.should_end or decision.temporary_exit or not decision.node_complete:
            return False
        return decision.next_node not in DIALOGUE_GRAPH[current_node].allowed_next

    def _technical_fallback_result(
        self,
        *,
        parse_error: str,
        raw_output: str,
    ) -> LlmCallResult:
        decision = LlmTurnDecision(
            reply="Секунду, повторите, пожалуйста, я не совсем корректно понял.",
            heard_summary="",
            turn_note="На этом ходу модель не вернула валидный ответ. Агент попросил клиента повторить.",
            facts_update={},
            node_complete=False,
            next_node=None,
            reply_asks_node=None,
            temporary_exit=False,
            return_to_node=None,
            client_question_answered=False,
            client_resistance=None,
            should_end=False,
            confidence=0.0,
            repeat_note=None,
            reason="technical_fallback",
        )
        return LlmCallResult(
            source="fallback",
            decision=decision,
            raw_output=raw_output,
            parse_error=parse_error or "unknown_llm_failure",
            latency_ms=0,
        )


def make_call_graph(*, llm_client: TurnLlmClient, metrics: MetricsCollector) -> CallGraphRunner:
    return CallGraphRunner(llm_client=llm_client, metrics=metrics)
