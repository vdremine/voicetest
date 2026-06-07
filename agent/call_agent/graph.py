from __future__ import annotations

from typing import Any

from .apply import apply_facts, sanitize_facts_update
from .graph_spec import DIALOGUE_GRAPH
from .llm import TurnLlmClient
from .metrics import MetricsCollector
from .schema import LlmTurnDecision
from .state import CallState


FACT_ALIASES: dict[str, tuple[str, ...]] = {
    "desired_amount": ("desired_amount", "amount", "нужная_сумма"),
    "property_type": ("property_type", "no_real_estate"),
    "region": ("region",),
    "encumbrance": ("encumbrance",),
    "owner_status": ("owner_status",),
    "priority": ("priority",),
    "vehicle_type": ("vehicle_type",),
    "vehicle_owner": ("vehicle_owner",),
    "vehicle_encumbrance": ("vehicle_encumbrance",),
    "callback_consent": ("callback_consent",),
    "callback_time": ("callback_time",),
}

REQUIRED_FACT_EXEMPT_NEXT: dict[str, set[str]] = {
    "collect_amount": {"collect_vehicle_type", "partner_format", "callback_time", "finish"},
    "collect_name": {"callback_time", "finish"},
    "collect_property_type": {"callback_time", "finish"},
    "collect_region": {"callback_time", "finish"},
    "collect_encumbrance": {"callback_time", "finish"},
    "collect_encumbrance_details": {"callback_time", "finish"},
    "collect_owner": {"callback_time", "finish"},
    "priority_choice": {"callback_time", "finish"},
    "collect_vehicle_type": {"callback_time", "finish"},
    "collect_vehicle_owner": {"callback_time", "finish"},
    "collect_vehicle_encumbrance": {"callback_time", "finish"},
    "partner_format": {"callback_time", "finish"},
}


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _contains_cjk(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in text)


def _contains_any(text: str, markers: tuple[str, ...]) -> bool:
    normalized = _clean_text(text).lower()
    return any(marker in normalized for marker in markers)


def _fact_present(key: str, facts_update: dict[str, Any], known_facts: dict[str, Any]) -> bool:
    aliases = FACT_ALIASES.get(key, (key,))
    for alias in aliases:
        if _clean_text(facts_update.get(alias)) or _clean_text(known_facts.get(alias)):
            return True
    return False


def _missing_required_fact_keys(
    current_node: str,
    decision: LlmTurnDecision,
    known_facts: dict[str, Any],
) -> list[str]:
    if not decision.node_complete or decision.temporary_exit or decision.should_end:
        return []

    if decision.next_node in REQUIRED_FACT_EXEMPT_NEXT.get(current_node, set()):
        return []

    node = DIALOGUE_GRAPH[current_node]
    missing: list[str] = []
    for key in node.required_fact_keys:
        if not _fact_present(key, decision.facts_update, known_facts):
            missing.append(key)
    return missing


def _decision_errors(
    current_node: str,
    decision: LlmTurnDecision,
    known_facts: dict[str, Any],
) -> list[str]:
    errors: list[str] = []

    missing_required = _missing_required_fact_keys(current_node, decision, known_facts)
    if missing_required:
        errors.append(
            "node_complete=true but required facts are missing: "
            + ", ".join(missing_required)
        )

    if (
        not decision.temporary_exit
        and decision.reply_asks_node
        and decision.next_node
        and decision.node_complete
        and decision.reply_asks_node != decision.next_node
    ):
        errors.append(
            "reply_asks_node does not match next_node: "
            f"{decision.reply_asks_node} != {decision.next_node}"
        )

    if (
        not decision.temporary_exit
        and decision.reply_asks_node
        and not decision.node_complete
        and decision.reply_asks_node != current_node
    ):
        errors.append(
            "reply_asks_node points away from current node while current node is not complete"
        )

    if _contains_cjk(decision.reply):
        errors.append("reply contains CJK characters")

    if decision.reply.count("?") > 1:
        errors.append("reply asks more than one question")

    if _contains_any(
        decision.reply,
        (
            "как я могу вам помочь",
            "чем могу помочь",
            "как могу вам помочь",
        ),
    ):
        errors.append("reply slipped into inbound support style")

    if _contains_any(
        decision.reply,
        (
            "хата",
            "тачка",
            "бабки",
            "налик",
        ),
    ):
        errors.append("reply mirrors client slang")

    if _contains_any(
        decision.reply,
        (
            "точно одобрим",
            "можем одобрить",
            "точно получится",
            "у вас хорошая кредитная история",
        ),
    ):
        errors.append("reply promises approval or invents credit quality")

    if _contains_any(
        decision.reply,
        (
            "цель кредита",
            "зачем вам деньги",
            "для чего вам деньги",
        ),
    ):
        errors.append("reply asks forbidden money purpose question")

    if not _clean_text(decision.reply):
        errors.append("reply is empty")

    return errors


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

        if current_node == "call_connected":
            decision = LlmTurnDecision(
                reply=DIALOGUE_GRAPH["cold_opening"].ask,
                heard_summary="Клиент ответил на звонок.",
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
                repeat_note="Deterministic ready answer after call pickup.",
                reason="Deterministic cold opening after answer.",
            )
            self._metrics.record("cached_answer", 0)
            self._metrics.record("first_answer_time", 0)
            trace = {
                **state.get("trace", {}),
                "llm_latency_ms": 0,
                "llm_initial_decision": decision.model_dump(),
                "llm_decision": decision.model_dump(),
                "used_deterministic_call_connected": True,
            }
            return {
                **state,
                "reply": decision.reply,
                "llm_decision": decision.model_dump(),
                "return_to_node": decision.return_to_node,
                "trace": trace,
            }

        decision, latency_ms = await self._llm_client.call_turn_llm(
            current_node=current_node,
            user_text=state.get("user_text", ""),
            known_facts=known_facts,
            history=history,
            node_repeat_count=repeat_count,
        )
        self._metrics.record("llm_answer", latency_ms)

        final_decision = decision
        repair_errors = _decision_errors(current_node, decision, known_facts)
        repair_trace: dict[str, Any] = {
            "triggered": False,
            "errors": repair_errors,
        }
        total_latency_ms = latency_ms

        if repair_errors:
            repaired_decision, repair_latency_ms = await self._llm_client.repair_turn_llm(
                current_node=current_node,
                user_text=state.get("user_text", ""),
                known_facts=known_facts,
                history=history,
                node_repeat_count=repeat_count,
                bad_decision=decision.model_dump(),
                errors=repair_errors,
            )
            total_latency_ms += repair_latency_ms
            self._metrics.record("llm_repair", repair_latency_ms)
            repair_trace["triggered"] = True
            repair_trace["latency_ms"] = repair_latency_ms
            repair_trace["repaired_decision"] = (
                repaired_decision.model_dump() if repaired_decision is not None else None
            )
            if repaired_decision is not None:
                final_decision = repaired_decision

        self._metrics.record("first_answer_time", total_latency_ms)

        trace = {
            **state.get("trace", {}),
            "llm_latency_ms": total_latency_ms,
            "llm_initial_decision": decision.model_dump(),
            "llm_decision": final_decision.model_dump(),
            "repair": repair_trace,
        }
        return {
            **state,
            "reply": final_decision.reply,
            "llm_decision": final_decision.model_dump(),
            "return_to_node": final_decision.return_to_node,
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
