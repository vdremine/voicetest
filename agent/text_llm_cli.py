from __future__ import annotations

import asyncio
import os
from pathlib import Path

from agent_core import DialogueState, KnowledgeBase, SessionMemory, ToolGraphRuntime
from voice_loop import OpenAiLlmService, TranscriptNormalizer, VoicePipelineConfig


def log(message: str) -> None:
    print(f"[text-llm] {message}", flush=True)


async def run() -> None:
    config = VoicePipelineConfig.from_env()
    llm_service = OpenAiLlmService(config, log)
    normalizer = TranscriptNormalizer()
    dialogue_state = DialogueState()
    session_memory = SessionMemory(max_turns=12)

    try:
        kb = KnowledgeBase.load(config.data_dir)
        log(f"loaded knowledge base from {config.data_dir}")
    except Exception as exc:
        log(f"failed to load knowledge base from {config.data_dir}: {exc}")
        kb = KnowledgeBase.default()

    try:
        graph_path = Path(
            os.getenv(
                "TOOL_GRAPH_PATH",
                str(config.data_dir / "tool_graph.json"),
            )
        )
        tool_graph = ToolGraphRuntime.load(graph_path=graph_path, agent_name="Влад+имир")
        dialogue_state.current_node = tool_graph.start_node
        log(f"loaded tool graph from {graph_path}")
    except Exception as exc:
        log(f"failed to load tool graph: {exc}")
        tool_graph = None

    await llm_service.warmup()
    print("Text LLM mode ready. Commands: /state /reset /exit")

    while True:
        try:
            raw_text = input("you> ").strip()
        except EOFError:
            break

        if not raw_text:
            continue
        if raw_text == "/exit":
            break
        if raw_text == "/reset":
            dialogue_state = DialogueState()
            session_memory = SessionMemory(max_turns=12)
            if tool_graph is not None:
                dialogue_state.current_node = tool_graph.start_node
            print("assistant> Сессия сброшена.")
            continue
        if raw_text == "/state":
            print(session_memory.state_text() if session_memory.state else "assistant> state is empty")
            continue

        normalized_text = normalizer.normalize(raw_text)
        updated_fields = dialogue_state.update_from_user(raw_text, normalized_text, kb=kb)
        forced_fields = dialogue_state.force_capture_expected_slot(raw_text, normalized_text)
        if forced_fields:
            updated_fields |= forced_fields

        state_snapshot = dialogue_state.snapshot()
        session_memory.add_user(normalized_text or raw_text)
        session_memory.sync_from_dialogue_state(state_snapshot)

        knowledge = kb.retrieve(normalized_text, state_snapshot, limit=2)
        examples = kb.relevant_examples(normalized_text, limit=1)
        graph_context = (
            tool_graph.llm_context_for_text(normalized_text, state_snapshot)
            if tool_graph is not None
            else None
        )
        llm_state = session_memory.llm_state_payload()

        llm_reply, latency_ms = await llm_service.generate_response(
            normalized_text=normalized_text,
            history=session_memory.recent_history(limit=6),
            dialogue_state=llm_state,
            knowledge=knowledge,
            truth_rules=kb.truth_rules,
            examples=examples,
            graph_context=graph_context,
            max_tokens_override=min(120, config.llm_max_tokens),
        )

        reply_text = llm_reply.reply_tts.strip() or config.fallback_complex_text
        dialogue_state.update_from_agent(
            reply_text,
            llm_reply.next_step,
            kb=kb,
            current_node=str((graph_context or {}).get("node_name", "")).strip() or dialogue_state.current_node,
        )
        session_memory.add_assistant(reply_text)
        if "?" in reply_text or reply_text.lower().startswith(("подскажите", "скажите", "какая", "какой", "кто", "в каком")):
            session_memory.remember_question(reply_text)
        session_memory.sync_from_dialogue_state(
            dialogue_state.snapshot(),
            last_question=session_memory.last_question,
        )

        print(f"assistant> {reply_text}")
        print(
            f"[latency={latency_ms}ms stage={dialogue_state.stage} "
            f"next={dialogue_state.next_required_field or '-'} "
            f"updated={sorted(updated_fields)!r}]"
        )


if __name__ == "__main__":
    asyncio.run(run())
