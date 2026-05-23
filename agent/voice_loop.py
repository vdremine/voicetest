from __future__ import annotations

import asyncio
import json
import math
import os
import random
import re
import threading
import time
import uuid
import wave
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable
import numpy as np
import torch
import torchaudio.functional as torchaudio_f
from agent_core import DialogueState, KnowledgeBase, build_context_messages, inspect_llm_reply
from faster_whisper import WhisperModel
from livekit import rtc
from openai import AsyncOpenAI
from silero_vad import load_silero_vad


def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def env_nonempty(name: str, default: str = "") -> str:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip()
    return value if value else default


_AMOUNT_TOKEN_RE = re.compile(
    r"\b\d+(?:[\.,]\d+)?\s*(?:млн|миллион|миллиона|миллионов|тыс|тысяч|тысячи)?\b",
    flags=re.IGNORECASE,
)


@dataclass(slots=True)
class VoicePipelineConfig:
    sample_rate: int
    num_channels: int
    frame_size_ms: int
    vad_threshold: float
    vad_min_speech_duration_ms: int
    vad_min_silence_duration_ms: int
    vad_speech_pad_ms: int
    processing_resume_min_speech_duration_ms: int
    barge_in_min_speech_duration_ms: int
    vad_use_onnx: bool
    torch_num_threads: int
    stt_enabled: bool
    stt_model: str
    stt_device: str
    stt_compute_type_cpu: str
    stt_compute_type_gpu: str
    stt_language: str
    stt_beam_size: int
    stt_confidence_floor: float
    debug_save_wav: bool
    llm_enabled: bool
    llm_provider: str
    llm_model: str
    llm_reasoning_effort: str
    llm_timeout_seconds: float
    llm_base_url: str
    llm_api_key: str
    llm_project: str
    llm_temperature: float
    llm_max_tokens: int
    adaptive_classifier_enabled: bool
    llm_orchestrates_all: bool
    llm_debug_direct_mode: bool
    tts_enabled: bool
    tts_model_path: Path
    tts_model_url: str
    tts_speaker: str
    tts_sample_rate: int
    tts_publish_sample_rate: int
    tts_frame_ms: int
    tts_provider: str
    tts_segment_pause_ms: int
    tts_normalize_peak: float
    tts_fade_ms: int
    half_duplex: bool
    barge_in_enabled: bool
    voice_fillers_enabled: bool
    voice_fillers_level: str
    voice_fillers_probability: float
    voice_bridge_on_llm: bool
    data_dir: Path
    utterance_dir: Path
    session_log_dir: Path
    events_topic: str
    fallback_repeat_text: str
    fallback_low_confidence_text: str
    fallback_complex_text: str

    @classmethod
    def from_env(cls) -> "VoicePipelineConfig":
        llm_provider = env_nonempty("LLM_PROVIDER", "openai").lower()

        tts_provider = env_nonempty("TTS_PROVIDER").lower()
        if not tts_provider:
            tts_provider = "silero"

        default_llm_model = "Qwen/Qwen3-8B"
        default_llm_base_url = "http://127.0.0.1:8001/v1"
        default_llm_api_key = "local-token"
        llm_project = env_nonempty("LLM_PROJECT")

        if llm_provider == "yandex":
            yandex_api_key = env_nonempty("YANDEX_API_KEY")
            yandex_project_id = env_nonempty("YANDEX_PROJECT_ID")
            yandex_base_url = env_nonempty("YANDEX_BASE_URL", "https://ai.api.cloud.yandex.net/v1")
            llm_project = llm_project or yandex_project_id or env_nonempty("YANDEX_CLOUD_FOLDER")
            if llm_project:
                default_llm_model = f"gpt://{llm_project}/yandexgpt-lite/latest"
            default_llm_base_url = yandex_base_url
            default_llm_api_key = yandex_api_key

        return cls(
            sample_rate=int(os.getenv("AUDIO_SAMPLE_RATE", "16000")),
            num_channels=int(os.getenv("AUDIO_NUM_CHANNELS", "1")),
            frame_size_ms=int(os.getenv("AUDIO_FRAME_SIZE_MS", "20")),
            vad_threshold=float(os.getenv("VAD_THRESHOLD", "0.45")),
            vad_min_speech_duration_ms=int(os.getenv("VAD_MIN_SPEECH_DURATION_MS", "250")),
            vad_min_silence_duration_ms=int(os.getenv("VAD_MIN_SILENCE_DURATION_MS", "900")),
            vad_speech_pad_ms=int(os.getenv("VAD_SPEECH_PAD_MS", "200")),
            processing_resume_min_speech_duration_ms=int(
                os.getenv("PROCESSING_RESUME_MIN_SPEECH_DURATION_MS", "1200")
            ),
            barge_in_min_speech_duration_ms=int(os.getenv("BARGE_IN_MIN_SPEECH_DURATION_MS", "500")),
            vad_use_onnx=env_bool("VAD_USE_ONNX", False),
            torch_num_threads=int(os.getenv("TORCH_NUM_THREADS", "1")),
            stt_enabled=env_bool("STT_ENABLED", True),
            stt_model=os.getenv("STT_MODEL", "Systran/faster-whisper-medium"),
            stt_device=os.getenv("STT_DEVICE", "auto"),
            stt_compute_type_cpu=os.getenv("STT_COMPUTE_TYPE_CPU", "int8"),
            stt_compute_type_gpu=os.getenv("STT_COMPUTE_TYPE_GPU", "float16"),
            stt_language=os.getenv("STT_LANGUAGE", "ru"),
            stt_beam_size=int(os.getenv("STT_BEAM_SIZE", "3")),
            stt_confidence_floor=float(os.getenv("STT_CONFIDENCE_FLOOR", "0.25")),
            debug_save_wav=env_bool("DEBUG_SAVE_WAV", True),
            llm_enabled=env_bool("LLM_ENABLED", True),
            llm_provider=llm_provider,
            llm_model=env_nonempty("LLM_MODEL", default_llm_model),
            llm_reasoning_effort=os.getenv("LLM_REASONING_EFFORT", "low"),
            llm_timeout_seconds=float(os.getenv("LLM_TIMEOUT_SECONDS", "15")),
            llm_base_url=env_nonempty("LLM_BASE_URL", default_llm_base_url),
            llm_api_key=env_nonempty("LLM_API_KEY", default_llm_api_key),
            llm_project=llm_project,
            llm_temperature=float(os.getenv("LLM_TEMPERATURE", "0.2")),
            llm_max_tokens=int(os.getenv("LLM_MAX_TOKENS", "4096")),
            adaptive_classifier_enabled=env_bool("ADAPTIVE_CLASSIFIER_ENABLED", False),
            llm_orchestrates_all=env_bool("LLM_ORCHESTRATES_ALL", True),
            llm_debug_direct_mode=env_bool("LLM_DEBUG_DIRECT_MODE", False),
            tts_enabled=env_bool("TTS_ENABLED", True),
            tts_model_path=Path(os.getenv("TTS_MODEL_PATH", "/models/silero-tts/ru/v5_4_ru.pt")),
            tts_model_url=os.getenv(
                "TTS_MODEL_URL",
                "https://models.silero.ai/models/tts/ru/v5_4_ru.pt",
            ),
            tts_speaker=os.getenv("TTS_SPEAKER", "aidar"),
            tts_sample_rate=int(os.getenv("TTS_SAMPLE_RATE", "24000")),
            tts_publish_sample_rate=int(os.getenv("TTS_PUBLISH_SAMPLE_RATE", "24000")),
            tts_frame_ms=int(os.getenv("TTS_FRAME_MS", "20")),
            tts_provider=tts_provider,
            tts_segment_pause_ms=int(os.getenv("TTS_SEGMENT_PAUSE_MS", "120")),
            tts_normalize_peak=float(os.getenv("TTS_NORMALIZE_PEAK", "0.8")),
            tts_fade_ms=int(os.getenv("TTS_FADE_MS", "8")),
            half_duplex=env_bool("HALF_DUPLEX", True),
            barge_in_enabled=env_bool("BARGE_IN_ENABLED", False),
            voice_fillers_enabled=env_bool("VOICE_FILLERS_ENABLED", False),
            voice_fillers_level=os.getenv("VOICE_FILLERS_LEVEL", "light").strip().lower() or "light",
            voice_fillers_probability=float(os.getenv("VOICE_FILLERS_PROBABILITY", "0.35")),
            voice_bridge_on_llm=env_bool("VOICE_BRIDGE_ON_LLM", False),
            data_dir=Path(os.getenv("AGENT_DATA_DIR", "/app/data")),
            utterance_dir=Path(os.getenv("UTTERANCE_DIR", "/tmp/voice-agent/utterances")),
            session_log_dir=Path(os.getenv("SESSION_LOG_DIR", "/tmp/voice-agent/session-logs")),
            events_topic=os.getenv("AGENT_EVENTS_TOPIC", "agent_events"),
            fallback_repeat_text=os.getenv(
                "FALLBACK_REPEAT_TEXT",
                "Извините, я не расслышал. Повторите, пожалуйста.",
            ),
            fallback_low_confidence_text=os.getenv(
                "FALLBACK_LOW_CONFIDENCE_TEXT",
                "Извините, я не расслышал. Повторите, пожалуйста.",
            ),
            fallback_complex_text=os.getenv(
                "FALLBACK_COMPLEX_TEXT",
                "Подскажите, пожалуйста, какая сумма вам нужна и на какую цель.",
            ),
        )


@dataclass(slots=True)
class TranscriptResult:
    text: str
    language: str
    confidence: float
    duration_ms: int
    stt_latency_ms: int


@dataclass(slots=True)
class IntentResult:
    intent: str
    confidence: float
    use_llm: bool
    action: str


@dataclass(slots=True)
class LlmReply:
    reply_tts: str
    search_index: list[str]
    intent: str
    next_step: str
    raw_text: str = ""


@dataclass(slots=True)
class TtsRequest:
    text: str
    emotion: str = "friendly"
    style: str = "consultant"
    speed: float = 1.0
    speaker: str = "xenia"


@dataclass(slots=True)
class VoiceStyleResult:
    styled_text: str
    filler_added: bool
    filler_type: str
    original_text: str


class Intent(str, Enum):
    GREETING = "greeting"
    READY_TO_TALK = "ready_to_talk"
    CONFIRM_INTEREST = "confirm_interest"
    SLOT_ANSWER = "slot_answer"
    IDENTIFY_SELF = "identify_self"
    IDENTITY_MISMATCH = "identity_mismatch"
    LINE_ISSUE = "line_issue"
    WHY_NEED_INFO = "why_need_info"
    LATENCY_QUESTION = "latency_question"
    SERVICE_COMPLAINT = "service_complaint"
    PAYMENT_HELP = "payment_help"
    REPEAT = "repeat"
    WAIT = "wait"
    HUMAN_HANDOFF = "human_handoff"
    CANCEL = "cancel"
    REJECT = "reject"
    END_SESSION = "end_session"
    UNKNOWN_SHORT = "unknown_short"
    COMPLEX_REQUEST = "complex_request"
    CONFIRM = "confirm"
    AMOUNT_PROVIDED = "amount_provided"
    CLARIFY = "clarify"


class Action(str, Enum):
    CONTINUE_OPENING = "continue_opening"
    ASK_NEXT_SLOT = "ask_next_slot"
    INTRODUCE_SELF = "introduce_self"
    CLARIFY_IDENTITY = "clarify_identity"
    REPEAT_LAST_AGENT_MESSAGE = "repeat_last_agent_message"
    EXPLAIN_QUESTION = "explain_question"
    EXPLAIN_DELAY_AND_CONTINUE = "explain_delay_and_continue"
    ACK_COMPLAINT_AND_REFOCUS = "ack_complaint_and_refocus"
    HANDOFF_PAYMENT_SUPPORT = "handoff_payment_support"
    HANDOFF_TO_HUMAN = "handoff_to_human"
    CANCEL_ACTION = "cancel_action"
    ACK_REJECT = "ack_reject"
    END_SESSION = "end_session"
    ACK_WAIT = "ack_wait"
    ASK_REPEAT = "ask_repeat"
    CALL_LLM = "call_llm"
    ACK_GREETING = "ack_greeting"
    ACK_CONFIRM = "ack_confirm"
    ACK_AMOUNT_AND_CONTINUE = "ack_amount_and_continue"


class Int16ChunkBuffer:
    def __init__(self) -> None:
        self._chunks: deque[np.ndarray] = deque()
        self._size = 0

    @property
    def size(self) -> int:
        return self._size

    def append(self, samples: np.ndarray, *, limit: int | None = None) -> None:
        if samples.size == 0:
            return
        self._chunks.append(samples.copy())
        self._size += int(samples.size)
        if limit is not None:
            self._trim_left(limit)

    def prepend(self, samples: np.ndarray, *, limit: int | None = None) -> None:
        if samples.size == 0:
            return
        self._chunks.appendleft(samples.copy())
        self._size += int(samples.size)
        if limit is not None:
            self._trim_right(limit)

    def pop_front(self, count: int) -> np.ndarray:
        count = min(max(0, count), self._size)
        if count <= 0:
            return np.empty(0, dtype=np.int16)
        parts: list[np.ndarray] = []
        remaining = count
        while remaining > 0 and self._chunks:
            chunk = self._chunks[0]
            if chunk.size <= remaining:
                parts.append(chunk)
                self._chunks.popleft()
                self._size -= int(chunk.size)
                remaining -= int(chunk.size)
            else:
                parts.append(chunk[:remaining].copy())
                self._chunks[0] = chunk[remaining:].copy()
                self._size -= remaining
                remaining = 0
        return np.concatenate(parts) if len(parts) > 1 else parts[0]

    def to_array(self) -> np.ndarray:
        if not self._chunks:
            return np.empty(0, dtype=np.int16)
        if len(self._chunks) == 1:
            return self._chunks[0].copy()
        return np.concatenate(list(self._chunks))

    def clear(self) -> None:
        self._chunks.clear()
        self._size = 0

    def _trim_left(self, limit: int) -> None:
        while self._size > limit and self._chunks:
            overflow = self._size - limit
            chunk = self._chunks[0]
            if chunk.size <= overflow:
                self._chunks.popleft()
                self._size -= int(chunk.size)
            else:
                self._chunks[0] = chunk[overflow:].copy()
                self._size -= overflow
                break

    def _trim_right(self, limit: int) -> None:
        while self._size > limit and self._chunks:
            overflow = self._size - limit
            chunk = self._chunks[-1]
            if chunk.size <= overflow:
                self._chunks.pop()
                self._size -= int(chunk.size)
            else:
                self._chunks[-1] = chunk[:-overflow].copy()
                self._size -= overflow
                break


class AgentEventBus:
    def __init__(self, room: rtc.Room, *, topic: str, log: Callable[[str], None]) -> None:
        self._room = room
        self._topic = topic
        self._log = log

    async def publish_json(
        self,
        payload: dict[str, Any],
        *,
        destination_identities: list[str] | None = None,
    ) -> None:
        encoded = json.dumps(payload, ensure_ascii=False)
        try:
            await self._room.local_participant.publish_data(
                encoded,
                reliable=True,
                destination_identities=destination_identities or [],
                topic=self._topic,
            )
        except Exception as exc:
            self._log(f"failed to publish event topic={self._topic}: {exc}")

    async def publish_status(
        self,
        state: str,
        *,
        participant_identity: str,
        status: str,
        destination_identities: list[str] | None = None,
    ) -> None:
        await self.publish_json(
            {
                "type": "agent_status",
                "status": status,
                "state": state,
                "participant_identity": participant_identity,
                "ts_ms": int(time.time() * 1000),
            },
            destination_identities=destination_identities,
        )

    async def publish_error(
        self,
        *,
        stage: str,
        message: str,
        participant_identity: str,
        destination_identities: list[str] | None = None,
        utterance_id: str | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "type": "error",
            "stage": stage,
            "message": message,
            "participant_identity": participant_identity,
            "ts_ms": int(time.time() * 1000),
        }
        if utterance_id:
            payload["utterance_id"] = utterance_id
        await self.publish_json(payload, destination_identities=destination_identities)


class SessionLogger:
    def __init__(self, log_path: Path) -> None:
        self._log_path = log_path
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, payload: dict[str, Any]) -> None:
        with self._log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


class OpenAiLlmService:
    _LLM_JSON_SCHEMA: dict[str, Any] = {
        "type": "object",
        "properties": {
            "reply_tts": {
                "type": "string",
                "description": "Одна короткая реплика для клиента, готовая для озвучки.",
            },
            "search_index": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Короткие поисковые/контекстные ключи без дублей.",
            },
            "intent": {
                "type": "string",
                "description": "Короткий смысловой интент клиента.",
            },
            "next_step": {
                "type": "string",
                "description": "Один короткий следующий шаг менеджера.",
            },
        },
        "required": ["reply_tts", "search_index", "intent", "next_step"],
        "additionalProperties": False,
    }
    _CLASSIFIER_JSON_SCHEMA: dict[str, Any] = {
        "type": "object",
        "properties": {
            "intent": {
                "type": "string",
                "description": "Короткий тип реплики клиента.",
            },
            "action": {
                "type": "string",
                "description": "Что должен сделать агент на этом ходе.",
            },
            "use_llm": {
                "type": "boolean",
                "description": "Нужен ли полный ответ основного менеджерского LLM-ответчика.",
            },
            "confidence": {
                "type": "number",
                "description": "Уверенность от 0 до 1.",
            },
        },
        "required": ["intent", "action", "use_llm", "confidence"],
        "additionalProperties": False,
    }
    _CLASSIFIER_ALLOWED_INTENTS = {
        Intent.GREETING.value,
        Intent.READY_TO_TALK.value,
        Intent.CONFIRM_INTEREST.value,
        Intent.SLOT_ANSWER.value,
        Intent.IDENTIFY_SELF.value,
        Intent.IDENTITY_MISMATCH.value,
        Intent.LINE_ISSUE.value,
        Intent.WHY_NEED_INFO.value,
        Intent.LATENCY_QUESTION.value,
        Intent.SERVICE_COMPLAINT.value,
        Intent.PAYMENT_HELP.value,
        Intent.REPEAT.value,
        Intent.WAIT.value,
        Intent.HUMAN_HANDOFF.value,
        Intent.CANCEL.value,
        Intent.REJECT.value,
        Intent.END_SESSION.value,
        Intent.UNKNOWN_SHORT.value,
        Intent.COMPLEX_REQUEST.value,
    }
    _CLASSIFIER_ALLOWED_ACTIONS = {
        Action.CONTINUE_OPENING.value,
        Action.ASK_NEXT_SLOT.value,
        Action.INTRODUCE_SELF.value,
        Action.CLARIFY_IDENTITY.value,
        Action.REPEAT_LAST_AGENT_MESSAGE.value,
        Action.EXPLAIN_QUESTION.value,
        Action.EXPLAIN_DELAY_AND_CONTINUE.value,
        Action.ACK_COMPLAINT_AND_REFOCUS.value,
        Action.HANDOFF_PAYMENT_SUPPORT.value,
        Action.HANDOFF_TO_HUMAN.value,
        Action.CANCEL_ACTION.value,
        Action.ACK_REJECT.value,
        Action.END_SESSION.value,
        Action.ACK_WAIT.value,
        Action.ASK_REPEAT.value,
        Action.CALL_LLM.value,
        Action.ACK_GREETING.value,
        Action.ACK_CONFIRM.value,
        Action.ACK_AMOUNT_AND_CONTINUE.value,
    }
    _CLASSIFIER_PROMPT = """Ты sidecar-классификатор реплик клиента в голосовом кредитном звонке.
Ты НЕ отвечаешь клиенту. Ты только решаешь, что означает текущая реплика и какой следующий режим обработки нужен.

Верни только один JSON-объект строго по схеме.
Без markdown.
Без комментариев.
Без текста вне JSON.

intent используй только из списка:
- greeting
- ready_to_talk
- confirm_interest
- slot_answer
- identify_self
- identity_mismatch
- line_issue
- why_need_info
- latency_question
- service_complaint
- payment_help
- repeat
- wait
- human_handoff
- cancel
- reject
- end_session
- unknown_short
- complex_request

action используй только из списка:
- continue_opening
- ask_next_slot
- introduce_self
- clarify_identity
- repeat_last_agent_message
- explain_question
- explain_delay_and_continue
- ack_complaint_and_refocus
- handoff_payment_support
- handoff_to_human
- cancel_action
- ack_reject
- end_session
- ack_wait
- ask_repeat
- call_llm

Правила:
- если клиент просто подтверждает актуальность, согласие на разговор или отвечает на предыдущий квалификационный вопрос, это не complex_request;
- если в реплике уже есть сумма, цель, объект, регион, обременение или другая квалификационная деталь, обычно выбирай intent=slot_answer и action=ask_next_slot;
- если в реплике есть и сумма, и цель одновременно, всё равно выбирай intent=slot_answer и action=ask_next_slot;
- если клиент повторяет цель, которую агент уже должен был понять, не возвращай тот же вопрос снова;
- identify_self выбирай для "кто это", "кто вы", "с кем говорю";
- why_need_info выбирай для "зачем вам это", "для чего эта информация";
- line_issue выбирай для плохой связи и "вас не слышно";
- service_complaint выбирай для жалоб на прошлого менеджера и грубость;
- payment_help выбирай, если человек спрашивает про оплату, отсрочку, реквизиты, перевод или последний платеж;
- repeat, wait, cancel, human_handoff, reject, end_session используй для прямых управляющих команд;
- unknown_short ставь только если реплика реально шумовая, пустая или неинформативная;
- если реплика требует содержательного ответа по продукту, возражению или нестандартному кейсу, выбирай complex_request и action=call_llm."""

    _LOCAL_MANAGER_PROMPT = """Ты — Влад+имир, дежурный менеджер компании МосИнвестФинанс.
Если спрашивают, живой ли ты, отвечай: да.

Верни только один JSON-объект строго по схеме.
Без markdown.
Без комментариев.
Без текста вне JSON.

Правила ответа:
- reply_tts: одна короткая реплика для клиента, готовая для озвучки;
- search_index: массив из 1-5 коротких строк без дублей;
- intent: короткий смысловой интент клиента;
- next_step: один короткий следующий шаг менеджера;
- если данных мало, всё равно верни валидный JSON по схеме.

Твоя задача:
- активно вести разговор, а не ждать;
- выявить потребность клиента;
- подобрать подходящий продукт;
- продвинуть разговор на один шаг вперёд.
- самостоятельно оркестрировать разговор на каждом ходе, опираясь на историю и состояние диалога;
- корректно трактовать короткие ответы в контексте последнего вопроса, например: "да", "нет", "Москва", "в Москве", "квартира", "не сегодня".

Стиль:
- говори коротко, живо, уверенно;
- на «вы»;
- одна реплика = одна мысль;
- без канцелярита;
- без длинных монологов;
- не повторяй уже известное;
- после ответа клиента либо коротко ответь по сути, либо задай один следующий вопрос.

Старт звонка:
- первую реплику начинай с мягкого, чуть протяжного "Алл+о";
- после этого говори быстрее и естественнее;
- если это первый заход, начинай так:
  "Алл+о. Это Влад+имир, МосИнвестФинанс. Мы с вами созванивались на прошлой неделе по поводу кредита. Подскажите, пожалуйста, вопрос для вас ещё актуален?"
- если разговор уже идёт, не повторяй стартовую реплику.

Что делать в разговоре:
- если клиент задал прямой вопрос, сначала ответь на него;
- потом мягко верни разговор к цели звонка;
- задай один следующий короткий вопрос;
- если клиент отвечает общо, сам переводи разговор в конкретику;
- если клиент говорит коротко, продолжай разговор сам;
- если клиент возражает, коротко сними напряжение и веди дальше.
- если клиент ответил коротко, но по делу, не проси повторить тот же вопрос, а используй этот ответ как следующий слот разговора;
- если клиент спрашивает "это кто", сразу коротко представься и напомни причину звонка;
- если клиент жалуется на грубый прошлый разговор, коротко извинись, признай проблему и верни разговор к практическому решению;
- если клиент исправил имя, один раз извинись и дальше используй только правильное имя;
- если клиент говорит, что плохо слышит, повтори одну короткую фразу без длинного объяснения;
- если клиент просит не сегодня, зафиксируй удобное окно и подтверди обратный звонок;
- если клиент спрашивает, куда переводить или как оплатить, не придумывай реквизиты и не обещай детали, которых нет; переведи на персонального менеджера с согласованием времени.
- если клиент жалуется на прошлый разговор или грубость сотрудника, сначала коротко извинись и зафиксируй, что передашь жалобу, потом вернись к решению вопроса;
- если клиент говорит, что ему нужна отсрочка, перенос последнего платежа или порядок оплаты, не уводи разговор в новый кредит, а помоги довести до менеджера по платежам;
- если клиент сказал, что у него нет недвижимости, машины, ПТС или официальной работы, не предлагай продукты под такой залог и не спорь с этим.
- если клиент говорит, что его с кем-то перепутали, не спорь и не дави; уточни, как к нему обращаться, и актуален ли вопрос по кредиту вообще;
- если клиент хочет взять деньги на покупку автомобиля, не предлагай залог ПТС, если у него ещё нет автомобиля;
- если клиент сказал, что работает официально, а до этого распознавание ошиблось, коротко прими исправление и опирайся на новую информацию.
- если клиент грубит, посылает или явно требует прекратить разговор, не возвращайся к квалификации и спокойно заверши разговор.

Что нужно выяснить:
- какая сумма нужна;
- цель кредита;
- насколько срочно нужны деньги;
- есть ли недвижимость;
- какой объект;
- есть ли обременение;
- в каком городе или регионе объект;
- если недвижимости нет, есть ли автомобиль, ПТС или спецтехника;
- если речь о текущих кредитах, подходит ли рефинансирование.

Основные направления:
- кредит под залог недвижимости;
- кредит под залог автомобиля;
- займ под залог ПТС;
- кредит для ИП и ООО;
- кредит под залог коммерческой недвижимости;
- рефинансирование;
- потребительский кредит без подтверждения дохода;
- ипотека по двум документам.

Ключевые факты:
- по недвижимости сумма может быть до 70% от рыночной стоимости;
- срок от 1 года до 25 лет;
- ставка от 5% годовых;
- официальное трудоустройство не требуется;
- решение обычно за 1–2 дня после документов;
- клиент остаётся собственником;
- документы и оригиналы остаются у клиента.

Ограничения:
- не выдумывай продукты и условия;
- не предлагай инвестиции, вклады, брокерские продукты и другие нерелевантные услуги;
- не используй неправильные формы вроде "звОним" или "звонем";
- правильно: "звоним", "я звоню", "мы звоним".
- не предлагай залог автомобиля, ПТС, спецтехнику или недвижимость, если клиент прямо сказал, что этого нет;
- не предлагай ПТС, если клиент просит деньги именно на покупку автомобиля и ещё не владеет машиной;
- не дави на новый кредит, если клиент говорит только про закрытие долга, отсрочку или порядок оплаты;
- не озвучивай внутренние рассуждения, сводки разговора или пересказ в стиле "клиент сказал...".

TTS:
- reply_tts должен быть сразу пригоден для озвучки;
- TTS-разметку используй только если она реально нужна;
- обязательно помогай с произношением: Влад+имир;
- если нужно, размечай суммы, проценты, сроки, сложные названия и слово зал+ог."""
    _LOCAL_MANAGER_DEBUG_PROMPT = """Ты — Влад+имир, дежурный менеджер компании МосИнвестФинанс.
Если спрашивают, живой ли ты, отвечай: да.

Отвечай только plain text, сразу пригодным для озвучки.
Без JSON.
Без markdown.
Без комментариев.
Без служебных префиксов.

Правила:
- одна реплика, обычно 1-2 коротких предложения;
- держи ответ коротким, обычно до 220 символов;
- сначала ответь по сути, потом при необходимости задай один следующий вопрос;
- не повторяй уже известное;
- не пересказывай историю звонка и не озвучивай служебные поля;
- если это первый заход, начни с "Алл+о";
- не выдумывай продукты и условия;
- не предлагай ПТС, если у клиента нет автомобиля;
- вопросы оплаты, отсрочки и реквизитов переводи на персонального менеджера;
- если клиент грубит или просит прекратить разговор, спокойно заверши разговор."""

    def __init__(self, config: VoicePipelineConfig, log: Callable[[str], None]) -> None:
        self._config = config
        self._log = log
        self._client: AsyncOpenAI | None = None

    @property
    def enabled(self) -> bool:
        return self._config.llm_enabled and bool(self._config.llm_model)

    def _is_yandex_provider(self) -> bool:
        if self._config.llm_provider == "yandex":
            return True
        return "yandex.cloud" in self._config.llm_base_url or "ai.api.cloud.yandex.net" in self._config.llm_base_url

    def _ensure_client(self) -> AsyncOpenAI:
        if self._client is None:
            kwargs: dict[str, Any] = {"timeout": self._config.llm_timeout_seconds}
            if self._config.llm_base_url:
                kwargs["base_url"] = self._config.llm_base_url
                kwargs["api_key"] = self._config.llm_api_key or "local-token"
                if self._config.llm_project:
                    kwargs["default_headers"] = {"OpenAI-Project": self._config.llm_project}
            else:
                api_key = os.getenv("OPENAI_API_KEY", "").strip()
                if not api_key:
                    raise RuntimeError("OPENAI_API_KEY is not configured")
                kwargs["api_key"] = api_key
            self._client = AsyncOpenAI(**kwargs)
        return self._client

    @staticmethod
    def _history_to_messages(history: list[dict[str, str]]) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        for item in history[-8:]:
            role = str(item.get("role", "user")).strip().lower()
            if role not in {"user", "assistant", "system"}:
                role = "user"
            content = str(item.get("text", "")).strip()
            if not content:
                continue
            messages.append({"role": role, "content": content})
        return messages

    def _response_format(self, schema: dict[str, Any]) -> dict[str, Any]:
        if self._is_yandex_provider():
            return {"type": "json_schema", "json_schema": schema}
        return {"type": "json_object"}

    def _chat_extra_body(self) -> dict[str, Any] | None:
        model_name = self._config.llm_model.strip().lower()
        if "qwen3" not in model_name:
            return None

        effort = self._config.llm_reasoning_effort.strip().lower()
        enable_thinking = effort in {"on", "enabled", "thinking", "reasoning"}
        return {"chat_template_kwargs": {"enable_thinking": enable_thinking}}

    @classmethod
    def _parse_classifier_reply(
        cls,
        raw_text: str,
        *,
        fallback_intent: str,
        fallback_action: str,
        fallback_use_llm: bool,
        fallback_confidence: float,
    ) -> IntentResult:
        payload_text = extract_first_json_object(raw_text)
        parsed: dict[str, Any] = {}
        try:
            maybe_parsed = json.loads(payload_text)
            if isinstance(maybe_parsed, dict):
                parsed = maybe_parsed
        except Exception:
            parsed = {}

        intent = str(parsed.get("intent", "")).strip() or fallback_intent
        if intent not in cls._CLASSIFIER_ALLOWED_INTENTS:
            intent = fallback_intent

        action = str(parsed.get("action", "")).strip() or fallback_action
        if action not in cls._CLASSIFIER_ALLOWED_ACTIONS:
            action = fallback_action

        use_llm_value = parsed.get("use_llm", fallback_use_llm)
        if isinstance(use_llm_value, bool):
            use_llm = use_llm_value
        elif isinstance(use_llm_value, str):
            use_llm = use_llm_value.strip().lower() in {"1", "true", "yes", "да"}
        else:
            use_llm = fallback_use_llm

        confidence_value = parsed.get("confidence", fallback_confidence)
        try:
            confidence = float(confidence_value)
        except Exception:
            confidence = fallback_confidence
        confidence = max(0.0, min(1.0, confidence))

        return IntentResult(intent=intent, confidence=confidence, use_llm=use_llm, action=action)

    async def classify_intent(
        self,
        *,
        normalized_text: str,
        history: list[dict[str, str]],
        dialogue_state: dict[str, Any],
        initial_intent: IntentResult,
    ) -> tuple[IntentResult, int]:
        if not self.enabled:
            raise RuntimeError("LLM is disabled by configuration")

        client = self._ensure_client()
        started_at = time.perf_counter()
        state_json = json.dumps(dialogue_state, ensure_ascii=False)
        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": self._CLASSIFIER_PROMPT,
            },
            {
                "role": "system",
                "content": f"Текущее состояние звонка: {state_json}",
            },
            {
                "role": "system",
                "content": (
                    "Подсказка эвристического роутера: "
                    f"intent={initial_intent.intent}, action={initial_intent.action}, "
                    f"use_llm={str(initial_intent.use_llm).lower()}, confidence={initial_intent.confidence:.2f}."
                ),
            },
            *self._history_to_messages(history[-6:]),
        ]
        if not messages or messages[-1]["role"] != "user":
            messages.append({"role": "user", "content": normalized_text})

        request_kwargs: dict[str, Any] = {
            "model": self._config.llm_model,
            "temperature": 0.0,
            "max_tokens": min(160, self._config.llm_max_tokens),
            "response_format": self._response_format(self._CLASSIFIER_JSON_SCHEMA),
            "messages": messages,
        }
        extra_body = self._chat_extra_body()
        if extra_body is not None:
            request_kwargs["extra_body"] = extra_body

        completion = await client.chat.completions.create(**request_kwargs)
        latency_ms = int((time.perf_counter() - started_at) * 1000)
        content = completion.choices[0].message.content or ""
        if isinstance(content, list):
            text = "".join(
                part.get("text", "") if isinstance(part, dict) else getattr(part, "text", "")
                for part in content
            )
        else:
            text = str(content)

        return self._parse_classifier_reply(
            text,
            fallback_intent=initial_intent.intent,
            fallback_action=initial_intent.action,
            fallback_use_llm=initial_intent.use_llm,
            fallback_confidence=initial_intent.confidence,
        ), latency_ms

    async def generate_response(
        self,
        *,
        normalized_text: str,
        history: list[dict[str, str]],
        dialogue_state: dict[str, Any] | None = None,
        knowledge: list[KnowledgeSnippet] | None = None,
        truth_rules: tuple[str, ...] = (),
        examples: list[list[dict[str, str]]] | None = None,
    ) -> tuple[LlmReply, int]:
        if not self.enabled:
            raise RuntimeError("LLM is disabled by configuration")

        client = self._ensure_client()
        started_at = time.perf_counter()
        debug_direct_mode = self._config.llm_debug_direct_mode
        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    self._LOCAL_MANAGER_DEBUG_PROMPT
                    if debug_direct_mode
                    else self._LOCAL_MANAGER_PROMPT
                ),
            },
            {
                "role": "system",
                "content": (
                    "Верни только plain-text реплику для TTS. "
                    "Никакого JSON, markdown и служебного текста."
                    if debug_direct_mode
                    else (
                        "Строго соблюдай JSON-схему ответа. "
                        'Формат: {"reply_tts":"строка","search_index":["строка"],"intent":"строка","next_step":"строка"}. '
                        "Никакого текста вне JSON."
                    )
                ),
            },
            *build_context_messages(
                state=dialogue_state or {},
                knowledge=knowledge or [],
                truth_rules=truth_rules,
                examples=examples or [],
            ),
            *self._history_to_messages(history),
        ]
        if not messages or messages[-1]["role"] != "user":
            messages.append({"role": "user", "content": normalized_text})

        request_kwargs: dict[str, Any] = {
            "model": self._config.llm_model,
            "temperature": self._config.llm_temperature,
            "max_tokens": self._config.llm_max_tokens,
            "messages": messages,
        }
        if not debug_direct_mode:
            request_kwargs["response_format"] = self._response_format(self._LLM_JSON_SCHEMA)
        extra_body = self._chat_extra_body()
        if extra_body is not None:
            request_kwargs["extra_body"] = extra_body

        completion = await client.chat.completions.create(**request_kwargs)
        latency_ms = int((time.perf_counter() - started_at) * 1000)
        content = completion.choices[0].message.content or ""
        if isinstance(content, list):
            text = "".join(
                part.get("text", "") if isinstance(part, dict) else getattr(part, "text", "")
                for part in content
            )
        else:
            text = str(content)
        return parse_llm_reply(
            text,
            fallback_reply=self._config.fallback_complex_text,
            fallback_intent="complex_request",
            fallback_next_step="уточнить потребность клиента",
            fallback_search_seed=normalized_text,
            prefer_raw_text=debug_direct_mode,
        ), latency_ms


def sanitize_voice_response(text: str, *, fallback: str) -> str:
    value = text.strip()
    if not value:
        return fallback

    value = re.sub(r"<think>.*?</think>", " ", value, flags=re.IGNORECASE | re.DOTALL)
    value = re.sub(r"</?think>", " ", value, flags=re.IGNORECASE)
    value = re.sub(r"```.*?```", " ", value, flags=re.DOTALL)
    value = re.sub(r"<[^>]+>", " ", value)
    value = re.sub(r"(?im)^(assistant|system|user)\s*:\s*", "", value)
    value = re.sub(r"\s+", " ", value).strip()

    if not value:
        return fallback

    if value.startswith(("Хорошо, пользователь", "Пользователь", "Нужно ", "Стоит ", "Okay,", "The user")):
        return fallback

    suspicious_meta_markers = (
        "клиент написал",
        "клиент сказал",
        "запрос клиента",
        "разберемся с этим запросом",
        "давайте разберемся с этим запросом",
        "в этом запросе",
        "the user said",
        "the assistant",
        "user said",
        "assistant initially",
        "means in russian",
        "chain-of-thought",
        "hidden reasoning",
    )
    lowered = value.lower()
    if any(marker in lowered for marker in suspicious_meta_markers):
        return fallback

    ascii_letters = sum(1 for char in value if "a" <= char.lower() <= "z")
    cyrillic_letters = sum(1 for char in value if "а" <= char.lower() <= "я")
    if ascii_letters > max(8, cyrillic_letters):
        return fallback

    sentences = re.split(r"(?<=[.!?])\s+", value)
    short_text = " ".join(sentence.strip() for sentence in sentences[:2] if sentence.strip()).strip()
    return short_text or fallback


def extract_first_json_object(text: str) -> str:
    value = text.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", value, flags=re.DOTALL | re.IGNORECASE)
    if fence_match:
        return fence_match.group(1)

    start = value.find("{")
    if start < 0:
        return value

    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(value)):
        char = value[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return value[start : index + 1]

    return value


def dedupe_compact_strings(items: list[str], *, limit: int) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        value = re.sub(r"\s+", " ", item.strip())
        if not value:
            continue
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(value[:120])
        if len(result) >= limit:
            break
    return result


def _decode_loose_json_string(value: str) -> str:
    candidate = value.strip()
    if not candidate:
        return ""
    try:
        return json.loads(f'"{candidate}"')
    except Exception:
        return (
            candidate.replace('\\"', '"')
            .replace("\\n", " ")
            .replace("\\r", " ")
            .replace("\\t", " ")
            .replace("\\/", "/")
            .strip()
        )


def _extract_json_string_field(text: str, field_name: str) -> str:
    match = re.search(rf'"{re.escape(field_name)}"\s*:\s*"', text)
    if not match:
        return ""

    chars: list[str] = []
    escape = False
    index = match.end()
    while index < len(text):
        char = text[index]
        index += 1
        if escape:
            chars.append(char)
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if char == '"':
            break
        chars.append(char)

    return _decode_loose_json_string("".join(chars))


def _extract_json_array_strings_field(text: str, field_name: str) -> list[str]:
    match = re.search(rf'"{re.escape(field_name)}"\s*:\s*\[', text)
    if not match:
        return []

    values: list[str] = []
    index = match.end()
    while index < len(text):
        while index < len(text) and text[index] in " \t\r\n,":
            index += 1
        if index >= len(text) or text[index] == "]":
            break
        if text[index] != '"':
            break

        index += 1
        chars: list[str] = []
        escape = False
        while index < len(text):
            char = text[index]
            index += 1
            if escape:
                chars.append(char)
                escape = False
                continue
            if char == "\\":
                escape = True
                continue
            if char == '"':
                break
            chars.append(char)

        value = _decode_loose_json_string("".join(chars))
        if value:
            values.append(value)

    return dedupe_compact_strings(values, limit=5)


def parse_llm_reply(
    raw_text: str,
    *,
    fallback_reply: str,
    fallback_intent: str,
    fallback_next_step: str,
    fallback_search_seed: str,
    prefer_raw_text: bool = False,
) -> LlmReply:
    payload_text = extract_first_json_object(raw_text)
    parsed: dict[str, Any] = {}
    json_like_response = raw_text.lstrip().startswith("{") or payload_text.lstrip().startswith("{")
    try:
        maybe_parsed = json.loads(payload_text)
        if isinstance(maybe_parsed, dict):
            parsed = maybe_parsed
    except Exception:
        parsed = {}

    extracted_reply = _extract_json_string_field(raw_text, "reply_tts")
    extracted_intent = _extract_json_string_field(raw_text, "intent")
    extracted_next_step = _extract_json_string_field(raw_text, "next_step")
    extracted_search_index = _extract_json_array_strings_field(raw_text, "search_index")

    raw_reply_value = parsed.get("reply_tts", "")
    reply_source = str(raw_reply_value).strip() if raw_reply_value is not None else ""
    if not reply_source:
        reply_source = extracted_reply
    if not reply_source:
        reply_source = raw_text if prefer_raw_text or not json_like_response else fallback_reply
    if reply_source.lstrip().startswith("{") or '"reply_tts"' in reply_source:
        reply_source = extracted_reply or (fallback_reply if json_like_response else raw_text)

    reply_tts = sanitize_voice_response(reply_source, fallback=fallback_reply)
    intent = str(parsed.get("intent", "")).strip() or extracted_intent or fallback_intent
    next_step = str(parsed.get("next_step", "")).strip() or extracted_next_step or fallback_next_step

    raw_search_index = parsed.get("search_index", [])
    search_values: list[str] = []
    if isinstance(raw_search_index, list):
        search_values = [str(item) for item in raw_search_index]
    elif isinstance(raw_search_index, str):
        search_values = [raw_search_index]
    elif extracted_search_index:
        search_values = extracted_search_index

    if fallback_search_seed:
        search_values.append(fallback_search_seed)

    search_index = dedupe_compact_strings(search_values, limit=5)
    if not search_index:
        search_index = [fallback_intent]

    return LlmReply(
        reply_tts=reply_tts,
        search_index=search_index,
        intent=intent[:120],
        next_step=next_step[:160],
        raw_text=raw_text,
    )


def is_low_information_transcript(raw_text: str, normalized_text: str) -> bool:
    if not raw_text.strip():
        return True

    normalized = normalized_text.strip()
    if not normalized:
        return True

    raw_compact = re.sub(r"[^а-яa-z]", "", raw_text.lower().replace("ё", "е"))
    if not raw_compact:
        return True

    if len(raw_compact) >= 4 and len(set(raw_compact)) == 1:
        return True

    if len(normalized.split()) == 1:
        token = normalized
        if len(token) >= 4 and set(token) <= set("аоуыэеияюм"):
            return True
        if len(token) >= 4 and len(set(token)) == 1:
            return True

    return False


def split_into_tts_segments(text: str) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return []
    segments = [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", stripped) if segment.strip()]
    if not segments:
        return [stripped]
    return segments


def silence_ms(ms: int, sample_rate: int) -> np.ndarray:
    samples = max(0, int(sample_rate * ms / 1000))
    return np.zeros(samples, dtype=np.int16)


def trim_silence(pcm: np.ndarray, *, threshold: int = 96) -> np.ndarray:
    if len(pcm) == 0:
        return pcm
    indexes = np.flatnonzero(np.abs(pcm.astype(np.int32)) > threshold)
    if indexes.size == 0:
        return pcm
    return pcm[indexes[0] : indexes[-1] + 1]


def normalize_peak(pcm: np.ndarray, *, target_peak: float) -> np.ndarray:
    if len(pcm) == 0:
        return pcm
    peak = float(np.max(np.abs(pcm.astype(np.float32))))
    if peak < 1.0:
        return pcm
    gain = target_peak * 32767.0 / peak
    out = pcm.astype(np.float32) * gain
    return np.clip(out, -32768.0, 32767.0).astype(np.int16)


def apply_fade(pcm: np.ndarray, *, sample_rate: int, fade_ms: int) -> np.ndarray:
    if len(pcm) == 0 or fade_ms <= 0:
        return pcm
    n = min(len(pcm), int(sample_rate * fade_ms / 1000))
    if n <= 1:
        return pcm
    out = pcm.astype(np.float32)
    fade_in = np.linspace(0.0, 1.0, n, dtype=np.float32)
    fade_out = np.linspace(1.0, 0.0, n, dtype=np.float32)
    out[:n] *= fade_in
    out[-n:] *= fade_out
    return np.clip(out, -32768.0, 32767.0).astype(np.int16)


class TtsMarkupService:
    def __init__(self) -> None:
        self._pronunciation = {
            "владимир": "Влад+имир",
            "мосинвестфинанс": "Мос Инвест Фин+анс",
            "мфо": "эм эф о",
            "птс": "пэ тэ эс",
            "crm": "си ар эм",
            "livekit": "лайв кит",
        }
        self._stress = {
            "залог": "зал+ог",
            "договор": "догов+ор",
            "каталог": "катал+ог",
            "звонит": "звон+ит",
        }

    def prepare(self, request: TtsRequest) -> str:
        text = self._clean(request.text)
        text = self._normalize_numbers(text)
        text = self._apply_pronunciation(text)
        text = self._apply_stress(text)
        text = self._split_for_speech(text)
        return text

    @staticmethod
    def _clean(text: str) -> str:
        return re.sub(r"\s+", " ", text.strip())

    @staticmethod
    def _normalize_numbers(text: str) -> str:
        return text.replace("%", " процентов")

    def _apply_pronunciation(self, text: str) -> str:
        result = text
        for source, target in self._pronunciation.items():
            result = re.sub(rf"\b{re.escape(source)}\b", target, result, flags=re.IGNORECASE)
        return result

    def _apply_stress(self, text: str) -> str:
        result = text
        for source, target in self._stress.items():
            result = re.sub(rf"\b{re.escape(source)}\b", target, result, flags=re.IGNORECASE)
        return result

    @staticmethod
    def _split_for_speech(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()


class VoiceStyleAdapter:
    _SENSITIVE_INTENTS = {
        "service_complaint",
        "reject",
        "human_handoff",
        "payment_help",
        "cancel",
        "end_session",
        "repeat",
        "unknown_short",
        "clarify",
        "low_confidence",
    }

    def __init__(self, config: VoicePipelineConfig) -> None:
        self._config = config
        self._random = random.Random()
        self._previous_had_filler = False

    def adapt(
        self,
        text: str,
        *,
        intent: str,
        stage: str,
        can_use_greeting_prefix: bool,
    ) -> VoiceStyleResult:
        original = text.strip()
        if not original:
            return VoiceStyleResult("", False, "", original)

        lowered_original = original.lower()
        if "не расслышал" in lowered_original:
            self._previous_had_filler = False
            return VoiceStyleResult(original, False, "", original)

        if can_use_greeting_prefix:
            styled = original if original.lower().startswith(("алло", "алл")) else f"Алл+о. {original}"
            self._previous_had_filler = True
            return VoiceStyleResult(styled, True, "greeting", original)

        if not self._config.voice_fillers_enabled or self._config.voice_fillers_level == "off":
            self._previous_had_filler = False
            return VoiceStyleResult(original, False, "", original)

        if self._previous_had_filler or intent in self._SENSITIVE_INTENTS:
            self._previous_had_filler = False
            return VoiceStyleResult(original, False, "", original)

        probability = max(0.0, min(1.0, self._config.voice_fillers_probability))
        if self._random.random() > probability:
            self._previous_had_filler = False
            return VoiceStyleResult(original, False, "", original)

        variants = self._variants_for(intent=intent, stage=stage)
        if not variants:
            self._previous_had_filler = False
            return VoiceStyleResult(original, False, "", original)

        prefix = self._random.choice(variants)
        styled = original if original.startswith(prefix) else f"{prefix} {original}"
        self._previous_had_filler = True
        return VoiceStyleResult(styled, True, intent or stage, original)

    def _variants_for(self, *, intent: str, stage: str) -> list[str]:
        level = self._config.voice_fillers_level
        light = {
            "slot_answer": ["Ага, понял.", "Так, понял.", "Хорошо, понял."],
            "amount_provided": ["Ага, понял.", "Так, понял."],
            "confirm": ["Хорошо.", "Понял."],
            "ready_to_talk": ["Да, здравствуйте.", "Хорошо."],
            "complex_request": ["Смотрите.", "Да, смотрите."],
            "line_issue": ["Да, конечно.", "Повторю коротко."],
            "latency_question": ["Так, сейчас.", "Секунду."],
        }
        medium = {
            **light,
            "slot_answer": ["Ага, понял.", "Так, понял.", "Такс.", "Давайте тогда."],
            "complex_request": ["Смотрите.", "Нуу, смотрите.", "Так, сейчас сориентирую."],
            "clarification": ["А, понял.", "Тогда уточню."],
        }
        high = {
            **medium,
            "slot_answer": ["Ага, понял.", "Так, понял.", "Такс.", "Угу.", "Давайте тогда."],
            "complex_request": ["Смотрите.", "Нуу, смотрите.", "Так, секунду.", "Сейчас сориентирую."],
        }
        table = light if level == "light" else medium if level == "medium" else high
        if intent in table:
            return table[intent]
        if stage == "qualification":
            return table.get("slot_answer", [])
        if stage == "need_detection":
            return table.get("complex_request", [])
        return []


class SileroTtsService:
    def __init__(self, config: VoicePipelineConfig, log: Callable[[str], None]) -> None:
        self._config = config
        self._log = log
        self._lock = threading.Lock()
        self._model: Any | None = None

    def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model

        self._config.tts_model_path.parent.mkdir(parents=True, exist_ok=True)
        if not self._config.tts_model_path.is_file():
            self._log(f"downloading silero tts model to {self._config.tts_model_path}")
            torch.hub.download_url_to_file(self._config.tts_model_url, str(self._config.tts_model_path))

        model = torch.package.PackageImporter(str(self._config.tts_model_path)).load_pickle(
            "tts_models",
            "model",
        )
        model.to(torch.device("cpu"))
        self._model = model
        self._log(
            f"initialized silero tts model={self._config.tts_model_path} speaker={self._config.tts_speaker}"
        )
        return self._model

    def synthesize_segments(self, request: TtsRequest, segments: list[str]) -> tuple[np.ndarray, int, int]:
        started_at = time.perf_counter()
        with self._lock:
            model = self._ensure_model()
            rendered_segments: list[np.ndarray] = []
            for index, segment in enumerate(segments):
                audio = model.apply_tts(
                    text=segment,
                    speaker=request.speaker or self._config.tts_speaker,
                    sample_rate=self._config.tts_sample_rate,
                )
                if isinstance(audio, torch.Tensor):
                    audio_np = audio.detach().cpu().numpy()
                else:
                    audio_np = np.asarray(audio)
                audio_np = np.clip(audio_np, -1.0, 1.0)
                pcm16 = (audio_np * 32767.0).astype(np.int16)
                pcm16 = trim_silence(pcm16)
                pcm16 = apply_fade(
                    pcm16,
                    sample_rate=self._config.tts_sample_rate,
                    fade_ms=self._config.tts_fade_ms,
                )
                rendered_segments.append(pcm16)
                if index < len(segments) - 1 and self._config.tts_segment_pause_ms > 0:
                    rendered_segments.append(
                        silence_ms(self._config.tts_segment_pause_ms, self._config.tts_sample_rate)
                    )

        if not rendered_segments:
            pcm16 = np.zeros(0, dtype=np.int16)
        else:
            pcm16 = np.concatenate(rendered_segments)
        pcm16 = normalize_peak(pcm16, target_peak=self._config.tts_normalize_peak)
        latency_ms = int((time.perf_counter() - started_at) * 1000)
        return pcm16, self._config.tts_sample_rate, latency_ms


def build_tts_service(config: VoicePipelineConfig, log: Callable[[str], None]) -> SileroTtsService:
    return SileroTtsService(config, log)


class LiveKitAudioPublisher:
    def __init__(self, room: rtc.Room, config: VoicePipelineConfig, log: Callable[[str], None]) -> None:
        self._room = room
        self._config = config
        self._log = log
        self._source: rtc.AudioSource | None = None
        self._track: rtc.LocalAudioTrack | None = None
        self._published = False
        self._lock = asyncio.Lock()
        self._playback_generation = 0

    async def ensure_published(self) -> None:
        if self._published:
            return

        async with self._lock:
            if self._published:
                return

            self._source = rtc.AudioSource(
                self._config.tts_publish_sample_rate,
                self._config.num_channels,
                queue_size_ms=1000,
            )
            self._track = rtc.LocalAudioTrack.create_audio_track("agent-voice", self._source)

            options = None
            try:
                options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
            except Exception:
                options = None

            if options is not None:
                await self._room.local_participant.publish_track(self._track, options)
            else:
                await self._room.local_participant.publish_track(self._track)

            self._published = True
            self._log("published local audio track for agent voice")

    def interrupt_playback(self) -> None:
        self._playback_generation += 1
        if self._source is not None:
            self._source.clear_queue()
        self._log("interrupted current agent audio playback")

    async def speak_pcm(self, pcm16: np.ndarray, sample_rate: int) -> bool:
        await self.ensure_published()
        assert self._source is not None
        playback_generation = self._playback_generation

        target = pcm16
        if sample_rate != self._config.tts_publish_sample_rate:
            target = self._resample(
                pcm16,
                orig_rate=sample_rate,
                target_rate=self._config.tts_publish_sample_rate,
            )

        samples_per_channel = int(self._config.tts_publish_sample_rate * self._config.tts_frame_ms / 1000)
        if samples_per_channel <= 0:
            samples_per_channel = 480

        cursor = 0
        while cursor < len(target):
            if playback_generation != self._playback_generation:
                return False

            chunk = target[cursor : cursor + samples_per_channel]
            if len(chunk) < samples_per_channel:
                chunk = np.pad(chunk, (0, samples_per_channel - len(chunk)))

            frame = rtc.AudioFrame(
                data=memoryview(chunk.tobytes()),
                sample_rate=self._config.tts_publish_sample_rate,
                num_channels=self._config.num_channels,
                samples_per_channel=samples_per_channel,
            )
            await self._source.capture_frame(frame)
            cursor += samples_per_channel

        if playback_generation != self._playback_generation:
            return False

        await self._source.wait_for_playout()
        return playback_generation == self._playback_generation

    @staticmethod
    def _resample(pcm16: np.ndarray, *, orig_rate: int, target_rate: int) -> np.ndarray:
        if orig_rate == target_rate or len(pcm16) == 0:
            return pcm16
        source = torch.from_numpy(pcm16.astype(np.float32)).view(1, -1)
        resampled = torchaudio_f.resample(
            source,
            orig_freq=orig_rate,
            new_freq=target_rate,
        )
        return np.clip(resampled.view(-1).cpu().numpy(), -32768.0, 32767.0).astype(np.int16)


class WhisperSttService:
    def __init__(self, config: VoicePipelineConfig, log: Callable[[str], None]) -> None:
        self._config = config
        self._log = log
        self._model: WhisperModel | None = None
        self._device: str | None = None
        self._compute_type: str | None = None
        self._lock = threading.Lock()

    def _resolve_device(self) -> tuple[str, str]:
        if self._config.stt_device in {"cpu", "cuda"}:
            device = self._config.stt_device
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        compute_type = (
            self._config.stt_compute_type_gpu
            if device == "cuda"
            else self._config.stt_compute_type_cpu
        )
        return device, compute_type

    def _ensure_model(self) -> WhisperModel:
        if self._model is not None:
            return self._model

        device, compute_type = self._resolve_device()

        try:
            self._model = WhisperModel(self._config.stt_model, device=device, compute_type=compute_type)
            self._device = device
            self._compute_type = compute_type
            self._log(
                f"initialized faster-whisper model={self._config.stt_model} "
                f"device={device} compute_type={compute_type}"
            )
            return self._model
        except Exception as exc:
            if device == "cuda":
                self._log(f"cuda init failed for faster-whisper, falling back to cpu: {exc}")
                self._model = WhisperModel(
                    self._config.stt_model,
                    device="cpu",
                    compute_type=self._config.stt_compute_type_cpu,
                )
                self._device = "cpu"
                self._compute_type = self._config.stt_compute_type_cpu
                return self._model
            raise

    @staticmethod
    def _estimate_confidence(segments: list[Any], transcript_text: str) -> float:
        values: list[float] = []
        for segment in segments:
            avg_logprob = getattr(segment, "avg_logprob", None)
            if avg_logprob is None:
                continue
            values.append(max(0.0, min(1.0, math.exp(min(0.0, float(avg_logprob))))))

        if values:
            return round(sum(values) / len(values), 3)
        if transcript_text:
            return 0.5
        return 0.0

    def transcribe(self, audio_samples: np.ndarray, duration_ms: int) -> TranscriptResult:
        started_at = time.perf_counter()
        audio_f32 = audio_samples.astype(np.float32) / 32768.0
        with self._lock:
            model = self._ensure_model()
            segments_iter, info = model.transcribe(
                audio_f32,
                language=self._config.stt_language,
                beam_size=self._config.stt_beam_size,
                condition_on_previous_text=False,
                vad_filter=False,
            )
            segments = list(segments_iter)

        text = " ".join(segment.text.strip() for segment in segments if segment.text.strip()).strip()
        latency_ms = int((time.perf_counter() - started_at) * 1000)
        return TranscriptResult(
            text=text,
            language=getattr(info, "language", self._config.stt_language) or self._config.stt_language,
            confidence=self._estimate_confidence(segments, text),
            duration_ms=duration_ms,
            stt_latency_ms=latency_ms,
        )


class TranscriptNormalizer:
    _punct_re = re.compile(r"[^\w\s]+", flags=re.UNICODE)
    _space_re = re.compile(r"\s+")

    def __init__(self) -> None:
        self._replacements = {
            "щё": "ще",
            "ещё": "еще",
        }
        self._filler_words = {"эм", "ээ", "мм", "ну"}

    def normalize(self, text: str) -> str:
        value = text.lower().replace("ё", "е")
        for source, target in self._replacements.items():
            value = value.replace(source, target)
        value = self._punct_re.sub(" ", value)
        tokens = [token for token in self._space_re.sub(" ", value).strip().split(" ") if token]
        tokens = [token for token in tokens if token not in self._filler_words]
        return " ".join(tokens)


class SimpleIntentRouter:
    _affirmation_tokens = {
        "да",
        "ага",
        "угу",
        "конечно",
        "хорошо",
        "ладно",
        "поехали",
        "договорились",
        "безусловно",
    }
    _rejection_tokens = {"нет", "не", "неа", "не буду", "не надо", "не нужно"}

    def __init__(self) -> None:
        self._greeting = {
            "алло",
            "ало",
            "привет",
            "всем привет",
            "здравствуйте",
            "добрый день",
            "добрый вечер",
            "доброе утро",
            "доброй ночи",
        }
        self._confirm = {"да", "угу", "ага", "подтверждаю", "конечно", "хорошо", "супер", "отлично", "безусловно"}
        self._reject = {"нет", "неа", "не надо"}
        self._cancel = {"отмена", "отменить", "отбой"}
        self._repeat = {
            "повтори",
            "повтори пожалуйста",
            "повторите",
            "повторите пожалуйста",
            "повторяю",
            "повторяю пожалуйста",
            "еще раз",
            "ещё раз",
            "не понял",
        }
        self._wait = {"подожди", "секунду", "одну секунду"}
        self._ready_to_talk = {"я слушаю", "слушаю вас", "говорите", "да слушаю", "слушаю", "удобно"}
        self._identify = {"это кто", "кто это", "кто вы", "представьтесь"}
        self._identity_mismatch_markers = {
            "с кем то",
            "с кем-то",
            "перепутали",
            "не меня",
            "ошиблись номером",
            "ошиблись",
        }
        self._line_issue = {"не слышу", "плохо слышно", "связь плохая", "вас не слышно"}
        self._why_need_info_markers = {
            "зачем тебе эта информация",
            "зачем вам эта информация",
            "для чего эта информация",
            "почему вам это нужно",
        }
        self._latency_markers = {
            "почему так долго",
            "долго отвечал",
            "что так долго",
            "почему долго",
        }
        self._service_complaint_markers = {
            "грубо",
            "груб",
            "жалоб",
            "запись разговора",
            "плохо пообщалась",
            "нехорошая",
            "девушка",
        }
        self._payment_help_markers = {
            "куда переводить",
            "куда перевести",
            "как оплатить",
            "порядок оплаты",
            "куда мне",
            "последний платеж",
            "отсроч",
            "перенос платеж",
        }
        self._end_session = {"стоп", "завершить", "закончить"}
        self._handoff_tokens = {"оператор", "человек"}

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return [token for token in text.split() if token]

    @classmethod
    def _is_affirmation_phrase(cls, text: str) -> bool:
        tokens = cls._tokens(text)
        if not tokens:
            return False
        return all(token in cls._affirmation_tokens or token == "а" for token in tokens)

    @classmethod
    def _is_rejection_phrase(cls, text: str) -> bool:
        tokens = cls._tokens(text)
        if not tokens:
            return False
        return all(token in cls._rejection_tokens or token == "буду" for token in tokens)

    @staticmethod
    def _looks_like_amount(text: str) -> bool:
        return bool(_AMOUNT_TOKEN_RE.search(text))

    def route(self, text: str) -> IntentResult:
        if not text:
            return IntentResult(Intent.CLARIFY.value, 0.0, False, Action.ASK_REPEAT.value)

        if text in self._greeting or text.startswith(("привет", "здравствуйте", "добрый ", "алло", "ало")):
            return IntentResult(Intent.GREETING.value, 0.99, True, Action.CALL_LLM.value)
        if text in self._ready_to_talk or text.startswith(("я слушаю", "слушаю вас", "говорите", "да слушаю")):
            return IntentResult(Intent.READY_TO_TALK.value, 0.99, True, Action.CALL_LLM.value)
        if text in self._identify or text.startswith(("это кто", "кто это", "кто вы", "представьтесь", "кто со мной")):
            return IntentResult(Intent.IDENTIFY_SELF.value, 0.99, True, Action.CALL_LLM.value)
        if any(marker in text for marker in self._identity_mismatch_markers):
            return IntentResult(Intent.IDENTITY_MISMATCH.value, 0.95, True, Action.CALL_LLM.value)
        if text in self._line_issue or "не слышу" in text or "вас не слышно" in text:
            return IntentResult(Intent.LINE_ISSUE.value, 0.98, False, Action.REPEAT_LAST_AGENT_MESSAGE.value)
        if any(marker in text for marker in self._why_need_info_markers):
            return IntentResult(Intent.WHY_NEED_INFO.value, 0.95, True, Action.CALL_LLM.value)
        if any(marker in text for marker in self._latency_markers):
            return IntentResult(Intent.LATENCY_QUESTION.value, 0.94, True, Action.CALL_LLM.value)
        if any(marker in text for marker in self._service_complaint_markers):
            return IntentResult(Intent.SERVICE_COMPLAINT.value, 0.92, True, Action.CALL_LLM.value)
        if any(marker in text for marker in self._payment_help_markers):
            return IntentResult(Intent.PAYMENT_HELP.value, 0.92, True, Action.CALL_LLM.value)
        if text in self._confirm or self._is_affirmation_phrase(text):
            return IntentResult(Intent.CONFIRM.value, 0.99, True, Action.CALL_LLM.value)
        if text in self._reject or self._is_rejection_phrase(text):
            return IntentResult(Intent.REJECT.value, 0.99, True, Action.CALL_LLM.value)
        if text in self._cancel or "отмена" in text:
            return IntentResult(Intent.CANCEL.value, 0.99, False, Action.CANCEL_ACTION.value)
        if text in self._repeat or text.startswith("повтор") or "повтор" in text:
            return IntentResult(Intent.REPEAT.value, 0.99, False, Action.REPEAT_LAST_AGENT_MESSAGE.value)
        if text in self._wait or text.startswith("подожди"):
            return IntentResult(Intent.WAIT.value, 0.98, False, Action.ACK_WAIT.value)
        if text in self._end_session or text.startswith("заверши") or text.startswith("стоп"):
            return IntentResult(Intent.END_SESSION.value, 0.98, False, Action.END_SESSION.value)
        if any(token in text for token in self._handoff_tokens):
            return IntentResult(Intent.HUMAN_HANDOFF.value, 0.99, False, Action.HANDOFF_TO_HUMAN.value)
        if self._looks_like_amount(text):
            return IntentResult(Intent.AMOUNT_PROVIDED.value, 0.94, True, Action.CALL_LLM.value)
        if len(text.split()) <= 2:
            return IntentResult(Intent.UNKNOWN_SHORT.value, 0.45, False, Action.ASK_REPEAT.value)

        return IntentResult(Intent.COMPLEX_REQUEST.value, 0.8, True, Action.CALL_LLM.value)


class CannedResponseEngine:
    def __init__(self, config: VoicePipelineConfig) -> None:
        self._config = config

    @staticmethod
    def _first_sentence(text: str | None) -> str:
        value = (text or "").strip()
        if not value:
            return ""
        parts = re.split(r"(?<=[.!?])\s+", value, maxsplit=1)
        return parts[0].strip()

    def choose(self, intent: IntentResult, *, last_agent_message: str | None) -> str:
        if intent.intent == "greeting":
            if not last_agent_message:
                return (
                    "Алл+о. Это Влад+имир, МосИнвестФинанс. "
                    "Мы с вами созванивались по вопросу кредита. "
                    "Подскажите, пожалуйста, вопрос для вас ещё актуален?"
                )
            return "Добрый день. Слушаю вас."
        if intent.intent == "ready_to_talk":
            return (
                "Это Влад+имир, МосИнвестФинанс. "
                "Мы созванивались по вопросу кредита. "
                "Подскажите, пожалуйста, вопрос для вас ещё актуален?"
            )
        if intent.intent == "identify_self":
            return "Это Влад+имир, МосИнвестФинанс. Мы созванивались по вопросу кредита. Удобно сейчас говорить?"
        if intent.intent == "identity_mismatch":
            return (
                "Понял вас. Давайте уточним. "
                "Подскажите, пожалуйста, как к вам обращаться и вопрос по кредиту для вас вообще актуален?"
            )
        if intent.intent == "line_issue":
            repeated = self._first_sentence(last_agent_message)
            return repeated or "Повторю коротко. Подскажите, пожалуйста, удобно сейчас говорить?"
        if intent.intent == "why_need_info":
            return (
                "Чтобы не гонять вас по лишним вопросам и сразу подобрать подходящий вариант. "
                "Подскажите, пожалуйста, это покупка автомобиля или другая цель?"
            )
        if intent.intent == "latency_question":
            return "Связь чуть задержалась. Продолжим. Подскажите, пожалуйста, это покупка автомобиля или другая цель?"
        if intent.intent == "service_complaint":
            return (
                "Понял вас. Извините, пожалуйста, за этот разговор. "
                "Я зафиксирую жалобу. "
                "Подскажите, пожалуйста, вам сейчас важнее уточнить оплату или договориться о звонке менеджера?"
            )
        if intent.intent == "payment_help":
            return (
                "Понял вас. Реквизиты и точную сумму должен подтвердить персональный менеджер. "
                "Подскажите, пожалуйста, вам удобнее, чтобы он связался сегодня или в другое время?"
            )
        if intent.intent == "amount_provided":
            return "Понял. Подскажите, пожалуйста, на какую цель планируете использовать эту сумму?"
        if intent.intent == "confirm":
            return "Хорошо."
        if intent.intent == "reject":
            return "Хорошо, не будем."
        if intent.intent == "cancel":
            return "Хорошо, отменяю."
        if intent.intent == "repeat":
            return last_agent_message or self._config.fallback_repeat_text
        if intent.intent == "wait":
            return "Хорошо. Подожду."
        if intent.intent == "human_handoff":
            return "Хорошо. Передаю на оператора."
        if intent.intent == "end_session":
            return "Хорошо. Завершаю разговор."
        if intent.intent in {"clarify", "unknown_short"}:
            return self._config.fallback_low_confidence_text
        return self._config.fallback_complex_text

    @staticmethod
    def rescue_prompt() -> str:
        return "Алл+о. Я вас слушаю. Повторите, пожалуйста, коротко."


class SileroVadEngine:
    def __init__(self, config: VoicePipelineConfig) -> None:
        torch.set_num_threads(max(1, config.torch_num_threads))
        self._model = load_silero_vad(onnx=config.vad_use_onnx)
        self._sample_rate = config.sample_rate
        self._window_size = 512 if self._sample_rate == 16000 else 256

    @property
    def window_size(self) -> int:
        return self._window_size

    def speech_probability(self, chunk: np.ndarray) -> float:
        chunk_f32 = chunk.astype(np.float32) / 32768.0
        tensor = torch.from_numpy(chunk_f32)
        with torch.no_grad():
            return float(self._model(tensor, self._sample_rate).item())

    def reset_states(self) -> None:
        reset = getattr(self._model, "reset_states", None)
        if callable(reset):
            reset()


class ParticipantAudioSession:
    def __init__(
        self,
        *,
        room: rtc.Room,
        participant: rtc.RemoteParticipant,
        config: VoicePipelineConfig,
        event_bus: AgentEventBus,
        stt_service: WhisperSttService,
        llm_service: OpenAiLlmService,
        tts_service: SileroTtsService,
        audio_publisher: LiveKitAudioPublisher,
        log: Callable[[str], None],
    ) -> None:
        self._room = room
        self._participant = participant
        self._config = config
        self._event_bus = event_bus
        self._stt_service = stt_service
        self._llm_service = llm_service
        self._tts_service = tts_service
        self._audio_publisher = audio_publisher
        self._log = log

        self._normalizer = TranscriptNormalizer()
        self._router = SimpleIntentRouter()
        self._responses = CannedResponseEngine(config)
        self._vad = SileroVadEngine(config)
        self._tts_markup = TtsMarkupService()
        self._voice_style = VoiceStyleAdapter(config)
        self._dialogue_state = DialogueState()
        try:
            self._kb = KnowledgeBase.load(config.data_dir)
            self._log(f"loaded agent knowledge base from {config.data_dir}")
        except Exception as exc:
            self._log(f"failed to load knowledge base from {config.data_dir}: {exc}")
            self._kb = KnowledgeBase.default()

        self._session_id = f"{participant.identity}-{int(time.time())}-{uuid.uuid4().hex[:8]}"
        self._session_logger = SessionLogger(config.session_log_dir / f"{self._session_id}.jsonl")

        self._audio_task: asyncio.Task[None] | None = None
        self._active_track_sid: str | None = None
        self._sample_buffer = Int16ChunkBuffer()
        self._pre_speech_chunks: deque[np.ndarray] = deque()
        self._utterance_chunks: list[np.ndarray] = []
        self._utterance_probs: list[float] = []
        self._last_speech_chunk_index = 0
        self._speech_ms = 0
        self._silence_ms = 0
        self._first_frame_logged = False
        self._active_utterance_id: str | None = None
        self._utterance_counter = 0
        self._last_agent_message: str | None = None
        self._history: list[dict[str, str]] = []
        self._is_speaking = False
        self._is_processing = False
        self._state = "agent_ready"
        self._speech_started_at_ms = 0
        self._speech_detected_published = False
        self._interrupt_speech_ms = 0
        self._barge_in_pending = False
        self._resume_speech_ms = 0
        self._turn_revision = 0
        self._active_utterance_revision = 0
        self._resume_buffer = Int16ChunkBuffer()
        self._resume_probe_buffer = Int16ChunkBuffer()
        self._resume_buffer_limit = self._config.sample_rate * 6
        self._needs_rescue_prompt = False
        self._spoken_turn_count = 0
        self._greeting_was_spoken = False
        self._processing_can_be_interrupted = False

        self._chunk_ms = int(self._vad.window_size * 1000 / self._config.sample_rate)
        self._pad_chunks = max(1, math.ceil(self._config.vad_speech_pad_ms / self._chunk_ms))
        self._min_speech_chunks = max(
            1,
            math.ceil(self._config.vad_min_speech_duration_ms / self._chunk_ms),
        )

        (self._config.utterance_dir / self._session_id).mkdir(parents=True, exist_ok=True)

    @property
    def participant_identity(self) -> str:
        return self._participant.identity

    async def ensure_started(self, track: rtc.Track, *, track_sid: str | None = None) -> None:
        if self._audio_task and not self._audio_task.done():
            if track_sid and self._active_track_sid == track_sid:
                self._log(
                    f"audio stream already active: participant={self._participant.identity} track_sid={track_sid}"
                )
                return
            self._audio_task.cancel()
            try:
                await self._audio_task
            except asyncio.CancelledError:
                pass

        self._active_track_sid = track_sid or getattr(track, "sid", None)
        self._audio_task = asyncio.create_task(self._consume_audio(track))
        self._audio_task.add_done_callback(self._on_audio_task_done)
        await self._publish_status("waiting_for_speech")

    def _on_audio_task_done(self, task: asyncio.Task[None]) -> None:
        try:
            task.result()
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            self._log(f"audio session task crashed for {self._participant.identity}: {exc}")

    async def aclose(self) -> None:
        if self._audio_task:
            self._audio_task.cancel()
            try:
                await self._audio_task
            except asyncio.CancelledError:
                pass
        self._active_track_sid = None
        self._vad.reset_states()

    async def _publish_status(self, state: str) -> None:
        self._state = state
        status = {
            "agent_ready": "ready",
            "waiting_for_speech": "listening",
            "speech_detected": "listening",
            "recording_utterance": "listening",
            "utterance_finalized": "transcribing",
            "stt_processing": "transcribing",
            "routing_intent": "thinking",
            "simple_intent_detected": "thinking",
            "complex_request_detected": "thinking",
            "speaking": "speaking",
            "error": "error",
        }.get(state, state)
        await self._event_bus.publish_status(
            state,
            participant_identity=self._participant.identity,
            status=status,
            destination_identities=[self._participant.identity],
        )

    async def _consume_audio(self, track: rtc.Track) -> None:
        stream = None
        try:
            self._log(
                "starting audio stream: "
                f"participant={self._participant.identity} "
                f"track_sid={getattr(track, 'sid', '-')}"
            )
            try:
                stream = rtc.AudioStream.from_participant(
                    participant=self._participant,
                    track_source=rtc.TrackSource.SOURCE_MICROPHONE,
                    sample_rate=self._config.sample_rate,
                    num_channels=self._config.num_channels,
                    frame_size_ms=self._config.frame_size_ms,
                )
                self._log(f"audio stream source=participant participant={self._participant.identity}")
            except Exception as participant_exc:
                self._log(
                    f"audio stream from_participant failed for {self._participant.identity}: {participant_exc}"
                )
                try:
                    stream = rtc.AudioStream.from_track(
                        track=track,
                        sample_rate=self._config.sample_rate,
                        num_channels=self._config.num_channels,
                        frame_size_ms=self._config.frame_size_ms,
                    )
                    self._log(f"audio stream source=track participant={self._participant.identity}")
                except Exception:
                    # Keep compatibility with older Python SDK builds that only expose the constructor.
                    stream = rtc.AudioStream(
                        track=track,
                        sample_rate=self._config.sample_rate,
                        num_channels=self._config.num_channels,
                    )
                    self._log(f"audio stream source=ctor participant={self._participant.identity}")

            async for frame_event in stream:
                frame = frame_event.frame
                if not self._first_frame_logged:
                    self._first_frame_logged = True
                    self._log(
                        "audio frame activity: "
                        f"participant={self._participant.identity} "
                        f"sample_rate={frame.sample_rate} "
                        f"channels={frame.num_channels} "
                        f"samples_per_channel={frame.samples_per_channel}"
                    )
                await self._push_frame(frame)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._log(f"audio stream failed for {self._participant.identity}: {exc}")
            await self._event_bus.publish_error(
                stage="audio_stream",
                message=str(exc),
                participant_identity=self._participant.identity,
                destination_identities=[self._participant.identity],
            )
        finally:
            if stream is not None:
                maybe_aclose = getattr(stream, "aclose", None)
                if callable(maybe_aclose):
                    result = maybe_aclose()
                    if asyncio.iscoroutine(result):
                        await result

    async def _push_frame(self, frame: rtc.AudioFrame) -> None:
        samples = np.array(frame.data, dtype=np.int16, copy=True)
        if frame.num_channels > 1:
            samples = samples.reshape(-1, frame.num_channels).mean(axis=1).astype(np.int16)

        if self._is_speaking:
            if self._config.half_duplex or not self._config.barge_in_enabled:
                return
            await self._handle_barge_in_samples(samples)
            return

        if self._is_processing:
            await self._handle_processing_samples(samples)
            return

        self._sample_buffer.append(samples)

        while self._sample_buffer.size >= self._vad.window_size:
            window = self._sample_buffer.pop_front(self._vad.window_size)
            await self._process_vad_window(window)

    def _append_resume_samples(self, samples: np.ndarray) -> None:
        self._resume_buffer.append(samples, limit=self._resume_buffer_limit)
        self._resume_probe_buffer.append(samples, limit=self._resume_buffer_limit)

    async def _detect_resumed_speech(self, *, speaking: bool) -> None:
        while self._resume_probe_buffer.size >= self._vad.window_size:
            window = self._resume_probe_buffer.pop_front(self._vad.window_size)
            probability = self._vad.speech_probability(window)
            if probability >= self._config.vad_threshold:
                if speaking:
                    self._interrupt_speech_ms += self._chunk_ms
                else:
                    self._resume_speech_ms += self._chunk_ms
            else:
                if speaking:
                    self._interrupt_speech_ms = 0
                else:
                    self._resume_speech_ms = 0

            speech_ms = self._interrupt_speech_ms if speaking else self._resume_speech_ms
            min_required_ms = (
                self._config.barge_in_min_speech_duration_ms
                if speaking
                else max(1200, self._config.processing_resume_min_speech_duration_ms)
            )
            if speech_ms < min_required_ms:
                continue

            if not speaking and not self._processing_can_be_interrupted:
                self._resume_speech_ms = 0
                self._resume_probe_buffer.clear()
                return

            if not self._barge_in_pending:
                self._turn_revision += 1
            self._barge_in_pending = True
            self._needs_rescue_prompt = True
            self._interrupt_speech_ms = 0
            self._resume_speech_ms = 0
            self._reset_utterance_state()
            await self._publish_status("waiting_for_speech")

            if speaking:
                self._log(
                    f"barge-in detected participant={self._participant.identity} "
                    f"vad_probability={probability:.3f}"
                )
                self._audio_publisher.interrupt_playback()
                self._is_speaking = False
            else:
                self._log(
                    f"speech resumed during processing participant={self._participant.identity} "
                    f"vad_probability={probability:.3f}"
                )
            return

    async def _handle_barge_in_samples(self, samples: np.ndarray) -> None:
        self._append_resume_samples(samples)
        await self._detect_resumed_speech(speaking=True)

    async def _handle_processing_samples(self, samples: np.ndarray) -> None:
        self._append_resume_samples(samples)
        await self._detect_resumed_speech(speaking=False)

    def _is_stale_turn(self, turn_revision: int) -> bool:
        return turn_revision != self._turn_revision

    def _append_pre_speech(self, chunk: np.ndarray) -> None:
        self._pre_speech_chunks.append(chunk.copy())
        while len(self._pre_speech_chunks) > self._pad_chunks:
            self._pre_speech_chunks.popleft()

    async def _process_vad_window(self, chunk: np.ndarray) -> None:
        probability = self._vad.speech_probability(chunk)
        is_speech = probability >= self._config.vad_threshold

        if not self._utterance_chunks:
            self._append_pre_speech(chunk)
            if not is_speech:
                return

            self._turn_revision += 1
            self._active_utterance_revision = self._turn_revision
            self._utterance_counter += 1
            self._active_utterance_id = f"utt-{self._utterance_counter:04d}"
            self._speech_started_at_ms = int(time.time() * 1000)
            self._speech_detected_published = True
            self._utterance_chunks = [buffered.copy() for buffered in self._pre_speech_chunks]
            if self._utterance_chunks:
                self._utterance_probs = [0.0] * (len(self._utterance_chunks) - 1) + [probability]
            else:
                self._utterance_probs = [probability]
            self._speech_ms = self._chunk_ms
            self._silence_ms = 0
            self._last_speech_chunk_index = len(self._utterance_chunks)
            await self._publish_status("speech_detected")
            await self._event_bus.publish_json(
                {
                    "type": "speech_detected",
                    "utterance_id": self._active_utterance_id,
                    "participant_identity": self._participant.identity,
                    "vad_probability": round(probability, 4),
                    "ts_ms": int(time.time() * 1000),
                },
                destination_identities=[self._participant.identity],
            )
            return

        self._utterance_chunks.append(chunk.copy())
        self._utterance_probs.append(probability)

        if is_speech:
            self._speech_ms += self._chunk_ms
            self._silence_ms = 0
            self._last_speech_chunk_index = len(self._utterance_chunks)
        else:
            self._silence_ms += self._chunk_ms

        if self._speech_ms >= self._config.vad_min_speech_duration_ms and self._state != "recording_utterance":
            await self._publish_status("recording_utterance")

        if self._silence_ms < self._config.vad_min_silence_duration_ms:
            return

        if self._speech_ms < self._config.vad_min_speech_duration_ms:
            self._log(
                f"dropped short speech candidate participant={self._participant.identity} "
                f"speech_ms={self._speech_ms}"
            )
            self._reset_utterance_state()
            await self._publish_status("waiting_for_speech")
            return

        keep_chunks = min(
            len(self._utterance_chunks),
            self._last_speech_chunk_index + self._pad_chunks,
        )
        utterance_audio = np.concatenate(self._utterance_chunks[:keep_chunks])
        vad_confidence = max(self._utterance_probs[:keep_chunks], default=0.0)
        utterance_id = self._active_utterance_id or f"utt-{self._utterance_counter:04d}"
        duration_ms = int(len(utterance_audio) * 1000 / self._config.sample_rate)
        speech_end_time_ms = int(time.time() * 1000)

        wav_path: Path | None = None
        if self._config.debug_save_wav:
            wav_path = self._config.utterance_dir / self._session_id / f"{utterance_id}.wav"
            self._save_wav(wav_path, utterance_audio)

        await self._publish_status("utterance_finalized")
        await self._event_bus.publish_json(
            {
                "type": "utterance_finalized",
                "utterance_id": utterance_id,
                "participant_identity": self._participant.identity,
                "duration_ms": duration_ms,
                "vad_confidence": round(vad_confidence, 4),
                "ts_ms": speech_end_time_ms,
            },
            destination_identities=[self._participant.identity],
        )

        self._reset_utterance_state()
        asyncio.create_task(
            self._process_utterance(
                utterance_id=utterance_id,
                audio_samples=utterance_audio,
                wav_path=wav_path,
                duration_ms=duration_ms,
                vad_confidence=vad_confidence,
                speech_start_time_ms=self._speech_started_at_ms,
                speech_end_time_ms=speech_end_time_ms,
                turn_revision=self._active_utterance_revision,
            )
        )

    def _reset_utterance_state(self) -> None:
        self._utterance_chunks = []
        self._utterance_probs = []
        self._last_speech_chunk_index = 0
        self._speech_ms = 0
        self._silence_ms = 0
        self._interrupt_speech_ms = 0
        self._resume_speech_ms = 0
        self._active_utterance_id = None
        self._speech_detected_published = False
        self._pre_speech_chunks.clear()

    def _save_wav(self, path: Path, samples: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(self._config.num_channels)
            handle.setsampwidth(2)
            handle.setframerate(self._config.sample_rate)
            handle.writeframes(samples.astype(np.int16).tobytes())

    def _state_fallback_reply(self) -> str:
        snapshot = self._dialogue_state.snapshot()
        next_field = self._kb.next_required_field(snapshot) or self._dialogue_state.next_required_field
        if next_field:
            return self._kb.question_for_field(next_field)
        return self._config.fallback_complex_text

    def _llm_bridge_text(self) -> str:
        name = self._dialogue_state.name.strip()
        if name:
            return f"Так, {name}, секунду."
        if self._dialogue_state.stage == "qualification":
            return "Так, секунду, быстро сориентируюсь."
        return "Смотрите, секунду."

    @staticmethod
    def _is_opening_intent(intent_value: str) -> bool:
        return intent_value in {
            Intent.GREETING.value,
            Intent.READY_TO_TALK.value,
            Intent.IDENTIFY_SELF.value,
        }

    def _prepare_tts_output(self, text: str, *, intent_value: str) -> tuple[VoiceStyleResult, str, list[str]]:
        style_result = self._voice_style.adapt(
            text,
            intent=intent_value,
            stage=self._dialogue_state.stage,
            can_use_greeting_prefix=(
                self._is_opening_intent(intent_value) and not self._greeting_was_spoken
            ),
        )
        prepared_text = self._tts_markup.prepare(
            TtsRequest(
                text=style_result.styled_text,
                speaker=self._config.tts_speaker,
            )
        )
        segments = split_into_tts_segments(prepared_text)
        return style_result, prepared_text, segments

    async def _speak_response(
        self,
        *,
        utterance_id: str,
        response_text: str,
        intent_value: str,
    ) -> tuple[str, str, list[str], int, bool, bool, str]:
        style_result, prepared_text, segments = self._prepare_tts_output(
            response_text,
            intent_value=intent_value,
        )
        if not segments:
            segments = [prepared_text] if prepared_text else [style_result.styled_text]
        tts_pcm16, tts_sample_rate, tts_latency_ms = await asyncio.to_thread(
            self._tts_service.synthesize_segments,
            TtsRequest(text=style_result.styled_text, speaker=self._config.tts_speaker),
            segments,
        )
        playback_completed = await self._audio_publisher.speak_pcm(tts_pcm16, tts_sample_rate)
        self._spoken_turn_count += 1
        if not self._greeting_was_spoken and style_result.styled_text.lower().startswith(("алло", "алл")):
            self._greeting_was_spoken = True
        return (
            style_result.styled_text,
            prepared_text,
            segments,
            tts_latency_ms,
            playback_completed,
            style_result.filler_added,
            style_result.filler_type,
        )

    def _should_use_adaptive_classifier(self, intent: IntentResult, normalized_text: str) -> bool:
        if (
            not self._config.adaptive_classifier_enabled
            or not self._llm_service.enabled
            or not normalized_text.strip()
        ):
            return False
        if intent.intent == Intent.COMPLEX_REQUEST.value:
            return False
        if self._dialogue_state.awaiting_field and intent.intent in {
            Intent.CONFIRM.value,
            Intent.AMOUNT_PROVIDED.value,
            Intent.UNKNOWN_SHORT.value,
        }:
            return True
        return False

    def _should_use_llm_orchestrator(
        self,
        *,
        transcript: TranscriptResult,
        normalized_text: str,
        rescue_only: bool,
    ) -> bool:
        return (
            self._config.llm_orchestrates_all
            and self._llm_service.enabled
            and not rescue_only
            and bool(transcript.text.strip())
            and bool(normalized_text.strip())
            and transcript.confidence >= self._config.stt_confidence_floor
        )

    @staticmethod
    def _llm_orchestrator_intent() -> IntentResult:
        return IntentResult(
            Intent.COMPLEX_REQUEST.value,
            1.0,
            True,
            Action.CALL_LLM.value,
        )

    @staticmethod
    def _skip_confidence_gate_for(intent: IntentResult) -> bool:
        return intent.intent in {
            Intent.GREETING.value,
            Intent.READY_TO_TALK.value,
            Intent.IDENTIFY_SELF.value,
            Intent.REPEAT.value,
            Intent.LINE_ISSUE.value,
            Intent.WAIT.value,
            Intent.HUMAN_HANDOFF.value,
            Intent.CANCEL.value,
            Intent.END_SESSION.value,
        }

    def _should_advance_by_state(self, intent: IntentResult, updated_fields: set[str]) -> bool:
        if intent.use_llm:
            return False
        slot_fields = {
            "нужная_сумма",
            "цель",
            "вид_объекта",
            "обременение",
            "callback_time",
            "сценарий",
        }
        if intent.action == Action.ASK_NEXT_SLOT.value:
            return True
        if not (updated_fields & slot_fields):
            return False
        if self._dialogue_state.awaiting_field:
            return True
        return intent.intent in {
            Intent.AMOUNT_PROVIDED.value,
            Intent.CONFIRM_INTEREST.value,
            Intent.SLOT_ANSWER.value,
            Intent.UNKNOWN_SHORT.value,
            Intent.COMPLEX_REQUEST.value,
        }

    async def _process_utterance(
        self,
        *,
        utterance_id: str,
        audio_samples: np.ndarray,
        wav_path: Path | None,
        duration_ms: int,
        vad_confidence: float,
        speech_start_time_ms: int,
        speech_end_time_ms: int,
        turn_revision: int,
    ) -> None:
        await self._publish_status("stt_processing")
        started_at = time.perf_counter()
        self._is_processing = True
        response_text = self._config.fallback_low_confidence_text
        raw_response_text = response_text
        normalized_text = ""
        intent = IntentResult("clarify", 0.0, False, "ask_repeat")
        transcript = TranscriptResult("", self._config.stt_language, 0.0, duration_ms, 0)
        llm_reply: LlmReply | None = None
        error_stage = ""
        response_published = False
        suppress_response = False
        rescue_only = False
        updated_fields: set[str] = set()
        classifier_latency_ms = 0
        router_latency_ms = 0
        decision_started_at = 0.0
        stt_done_time_ms = 0
        intent_ready_time_ms = 0
        response_ready_time_ms = 0
        tts_synth_start_time_ms = 0
        tts_synth_done_time_ms = 0
        tts_publish_start_time_ms = 0
        tts_publish_done_time_ms = 0
        tts_prepared_text = ""
        tts_segments: list[str] = []
        filler_added = False
        filler_type = ""
        llm_latency_ms = 0
        tts_latency_ms = 0
        llm_validation_reason = ""

        try:
            if not self._config.stt_enabled:
                raise RuntimeError("STT is disabled by configuration")

            self._processing_can_be_interrupted = True

            transcript = await asyncio.to_thread(self._stt_service.transcribe, audio_samples, duration_ms)

            await self._event_bus.publish_json(
                {
                    "type": "transcript",
                    "utterance_id": utterance_id,
                    "participant_identity": self._participant.identity,
                    "text": transcript.text,
                    "language": transcript.language,
                    "confidence": transcript.confidence,
                    "final": True,
                    "duration_ms": transcript.duration_ms,
                    "stt_latency_ms": transcript.stt_latency_ms,
                },
                destination_identities=[self._participant.identity],
            )

            normalized_text = self._normalizer.normalize(transcript.text)
            stt_done_time_ms = int(time.time() * 1000)
            self._log(
                f"turn transcript participant={self._participant.identity} "
                f"utterance_id={utterance_id} raw_stt_text={transcript.text!r} "
                f"normalized_text={normalized_text!r} confidence={transcript.confidence:.3f}"
            )
            if is_low_information_transcript(transcript.text, normalized_text):
                if self._needs_rescue_prompt:
                    response_text = self._responses.rescue_prompt()
                    suppress_response = False
                    rescue_only = True
                    self._needs_rescue_prompt = False
                else:
                    suppress_response = True
                    error_stage = "ignored_low_information"
                    self._log(
                        f"ignored low-information transcript participant={self._participant.identity} "
                        f"utterance_id={utterance_id} text={transcript.text!r}"
                    )
                    return
            if self._is_stale_turn(turn_revision):
                suppress_response = True
                error_stage = "superseded_turn"
                self._log(
                    f"skipped stale utterance before routing participant={self._participant.identity} "
                    f"utterance_id={utterance_id}"
                )
                return

            if not rescue_only:
                if transcript.text:
                    updated_fields = self._dialogue_state.update_from_user(
                        transcript.text,
                        normalized_text,
                        kb=self._kb,
                    )
                    self._history.append({"role": "user", "text": normalized_text or transcript.text})
                    self._history = self._history[-12:]
                state_snapshot = self._dialogue_state.snapshot()
                self._log(
                    f"turn state participant={self._participant.identity} "
                    f"utterance_id={utterance_id} updated_fields={sorted(updated_fields)!r} "
                    f"scenario={state_snapshot.get('scenario', '')!r} "
                    f"awaiting_field={state_snapshot.get('awaiting_field', '')!r} "
                    f"next_required_field={state_snapshot.get('next_required_field', '')!r} "
                    f"known_facts={state_snapshot.get('known_facts', {})!r}"
                )
                await self._publish_status("routing_intent")
                decision_started_at = time.perf_counter()
                use_llm_orchestrator = self._should_use_llm_orchestrator(
                    transcript=transcript,
                    normalized_text=normalized_text,
                    rescue_only=rescue_only,
                )
                if use_llm_orchestrator:
                    intent = self._llm_orchestrator_intent()
                else:
                    faq_answer = self._kb.match_faq(normalized_text)
                    intent = self._router.route(normalized_text)
                    if (
                        transcript.text
                        and transcript.confidence >= self._config.stt_confidence_floor
                        and not faq_answer
                        and self._should_use_adaptive_classifier(intent, normalized_text)
                    ):
                        try:
                            intent, classifier_latency_ms = await self._llm_service.classify_intent(
                                normalized_text=normalized_text,
                                history=self._history,
                                dialogue_state=state_snapshot,
                                initial_intent=intent,
                            )
                        except Exception as exc:
                            self._log(
                                f"adaptive classifier fallback for {self._participant.identity}: {exc}"
                            )
                if decision_started_at > 0 and not use_llm_orchestrator:
                    decision_latency_ms = int((time.perf_counter() - decision_started_at) * 1000)
                    router_latency_ms = max(0, decision_latency_ms - classifier_latency_ms)
                intent_ready_time_ms = int(time.time() * 1000)

                await self._event_bus.publish_json(
                    {
                        "type": "intent",
                        "utterance_id": utterance_id,
                        "participant_identity": self._participant.identity,
                        "intent": intent.intent,
                        "confidence": intent.confidence,
                        "use_llm": intent.use_llm,
                        "action": intent.action,
                    },
                    destination_identities=[self._participant.identity],
                )

                if not transcript.text:
                    response_text = self._config.fallback_low_confidence_text
                elif (
                    transcript.confidence < self._config.stt_confidence_floor
                    and not self._skip_confidence_gate_for(intent)
                ):
                    response_text = self._config.fallback_low_confidence_text
                elif use_llm_orchestrator:
                    await self._publish_status("complex_request_detected")
                    try:
                        state_snapshot = self._dialogue_state.snapshot()
                        knowledge = self._kb.retrieve(normalized_text, state_snapshot)
                        examples = self._kb.relevant_examples(normalized_text)
                        llm_task = asyncio.create_task(
                            self._llm_service.generate_response(
                                normalized_text=normalized_text,
                                history=self._history,
                                dialogue_state=state_snapshot,
                                knowledge=knowledge,
                                truth_rules=self._kb.truth_rules,
                                examples=examples,
                            )
                        )
                        if self._config.voice_bridge_on_llm and self._config.tts_enabled:
                            bridge_text = self._llm_bridge_text()
                            if bridge_text and not self._is_stale_turn(turn_revision):
                                self._log(
                                    f"llm bridge start participant={self._participant.identity} "
                                    f"utterance_id={utterance_id} text={bridge_text!r}"
                                )
                                await self._publish_status("speaking")
                                self._is_speaking = True
                                try:
                                    _, _, _, _, _, _, _ = await self._speak_response(
                                        utterance_id=f"{utterance_id}-bridge",
                                        response_text=bridge_text,
                                        intent_value=Intent.COMPLEX_REQUEST.value,
                                    )
                                finally:
                                    self._is_speaking = False
                                self._log(
                                    f"llm bridge done participant={self._participant.identity} "
                                    f"utterance_id={utterance_id}"
                                )
                                await self._publish_status("complex_request_detected")
                        llm_reply, llm_latency_ms = await llm_task
                        self._log(
                            f"llm raw reply participant={self._participant.identity} "
                            f"utterance_id={utterance_id} raw_text={llm_reply.raw_text!r} "
                            f"reply_tts={llm_reply.reply_tts!r} intent={llm_reply.intent!r} "
                            f"next_step={llm_reply.next_step!r}"
                        )
                        response_text, llm_validation_reason = inspect_llm_reply(
                            reply_tts=llm_reply.reply_tts,
                            fallback_reply=self._state_fallback_reply(),
                            state=state_snapshot,
                            knowledge=knowledge,
                            truth_rules=self._kb.truth_rules,
                        )
                        self._log(
                            f"llm validated reply participant={self._participant.identity} "
                            f"utterance_id={utterance_id} reason={llm_validation_reason} "
                            f"result={response_text!r}"
                        )
                    except Exception as exc:
                        self._log(f"llm fallback failed for {self._participant.identity}: {exc}")
                        response_text = self._state_fallback_reply()
                elif faq_answer and not intent.use_llm:
                    await self._publish_status("simple_intent_detected")
                    response_text = faq_answer
                elif self._should_advance_by_state(intent, updated_fields):
                    await self._publish_status("simple_intent_detected")
                    response_text = self._state_fallback_reply()
                elif intent.use_llm:
                    await self._publish_status("complex_request_detected")
                    if self._llm_service.enabled:
                        try:
                            state_snapshot = self._dialogue_state.snapshot()
                            knowledge = self._kb.retrieve(normalized_text, state_snapshot)
                            examples = self._kb.relevant_examples(normalized_text)
                            llm_task = asyncio.create_task(
                                self._llm_service.generate_response(
                                    normalized_text=normalized_text,
                                    history=self._history,
                                    dialogue_state=state_snapshot,
                                    knowledge=knowledge,
                                    truth_rules=self._kb.truth_rules,
                                    examples=examples,
                                )
                            )
                            if self._config.voice_bridge_on_llm and self._config.tts_enabled:
                                bridge_text = self._llm_bridge_text()
                                if bridge_text and not self._is_stale_turn(turn_revision):
                                    self._log(
                                        f"llm bridge start participant={self._participant.identity} "
                                        f"utterance_id={utterance_id} text={bridge_text!r}"
                                    )
                                    await self._publish_status("speaking")
                                    self._is_speaking = True
                                    try:
                                        _, _, _, _, _, _, _ = await self._speak_response(
                                            utterance_id=f"{utterance_id}-bridge",
                                            response_text=bridge_text,
                                            intent_value=Intent.COMPLEX_REQUEST.value,
                                        )
                                    finally:
                                        self._is_speaking = False
                                    self._log(
                                        f"llm bridge done participant={self._participant.identity} "
                                        f"utterance_id={utterance_id}"
                                    )
                                    await self._publish_status("complex_request_detected")
                            llm_reply, llm_latency_ms = await llm_task
                            self._log(
                                f"llm raw reply participant={self._participant.identity} "
                                f"utterance_id={utterance_id} raw_text={llm_reply.raw_text!r} "
                                f"reply_tts={llm_reply.reply_tts!r} intent={llm_reply.intent!r} "
                                f"next_step={llm_reply.next_step!r}"
                            )
                            response_text, llm_validation_reason = inspect_llm_reply(
                                reply_tts=llm_reply.reply_tts,
                                fallback_reply=self._state_fallback_reply(),
                                state=state_snapshot,
                                knowledge=knowledge,
                                truth_rules=self._kb.truth_rules,
                            )
                            self._log(
                                f"llm validated reply participant={self._participant.identity} "
                                f"utterance_id={utterance_id} reason={llm_validation_reason} "
                                f"result={response_text!r}"
                            )
                        except Exception as exc:
                            self._log(f"llm fallback failed for {self._participant.identity}: {exc}")
                            response_text = self._state_fallback_reply()
                    else:
                        response_text = self._state_fallback_reply()
                else:
                    if intent.action == "ask_next_slot":
                        response_text = self._state_fallback_reply()
                    else:
                        await self._publish_status("simple_intent_detected")
                        response_text = self._responses.choose(
                            intent,
                            last_agent_message=self._last_agent_message,
                        )
                self._processing_can_be_interrupted = False
                raw_response_text = response_text
                response_ready_time_ms = int(time.time() * 1000)
                self._log(
                    f"turn decision participant={self._participant.identity} "
                    f"utterance_id={utterance_id} router_intent={intent.intent} "
                    f"action={intent.action} use_llm={str(intent.use_llm).lower()} "
                    f"final_response_text={response_text!r}"
                )

            self._processing_can_be_interrupted = False
            if self._is_stale_turn(turn_revision):
                suppress_response = True
                error_stage = "superseded_turn"
                self._log(
                    f"skipped stale utterance after routing participant={self._participant.identity} "
                    f"utterance_id={utterance_id}"
                )
                return

            self._last_agent_message = response_text
            self._dialogue_state.update_from_agent(
                response_text,
                llm_reply.next_step if llm_reply else "",
                kb=self._kb,
            )
            self._history.append({"role": "assistant", "text": response_text})
            self._history = self._history[-12:]
            self._needs_rescue_prompt = False

            await self._event_bus.publish_json(
                {
                    "type": "agent_response_text",
                    "utterance_id": utterance_id,
                    "participant_identity": self._participant.identity,
                    "text": response_text,
                    "use_llm": intent.use_llm,
                    "llm_intent": llm_reply.intent if llm_reply else "",
                    "search_index": llm_reply.search_index if llm_reply else [],
                    "next_step": llm_reply.next_step if llm_reply else "",
                },
                destination_identities=[self._participant.identity],
            )
            response_published = True
            if self._config.tts_enabled:
                if self._is_stale_turn(turn_revision):
                    suppress_response = True
                    error_stage = "superseded_turn"
                    self._log(
                        f"skipped stale utterance before tts participant={self._participant.identity} "
                        f"utterance_id={utterance_id}"
                    )
                    return
                await self._publish_status("speaking")
                self._is_speaking = True
                tts_synth_start_time_ms = int(time.time() * 1000)
                self._log(
                    f"tts synth start participant={self._participant.identity} "
                    f"utterance_id={utterance_id} text={response_text!r}"
                )
                (
                    spoken_text,
                    tts_prepared_text,
                    tts_segments,
                    tts_latency_ms,
                    playback_completed,
                    filler_added,
                    filler_type,
                ) = await self._speak_response(
                    utterance_id=utterance_id,
                    response_text=response_text,
                    intent_value=intent.intent,
                )
                response_text = spoken_text
                self._log(
                    f"tts style participant={self._participant.identity} "
                    f"utterance_id={utterance_id} filler_added={str(filler_added).lower()} "
                    f"filler_type={filler_type or '-'} raw={raw_response_text!r} styled={response_text!r}"
                )
                tts_synth_done_time_ms = int(time.time() * 1000)
                self._log(
                    f"tts synth done participant={self._participant.identity} "
                    f"utterance_id={utterance_id} segments={len(tts_segments)} "
                    f"prepared_text={tts_prepared_text!r}"
                )
                tts_publish_start_time_ms = int(time.time() * 1000)
                self._log(
                    f"tts publish start participant={self._participant.identity} "
                    f"utterance_id={utterance_id}"
                )
                tts_publish_done_time_ms = int(time.time() * 1000)
                self._log(
                    f"tts publish done participant={self._participant.identity} "
                    f"utterance_id={utterance_id}"
                )
                if not playback_completed:
                    self._log(
                        f"tts publish interrupted participant={self._participant.identity} "
                        f"utterance_id={utterance_id}"
                    )
        except Exception as exc:
            error_stage = "stt"
            self._log(f"utterance processing failed participant={self._participant.identity}: {exc}")
            response_text = self._config.fallback_low_confidence_text
            await self._event_bus.publish_error(
                stage=error_stage,
                message=str(exc),
                participant_identity=self._participant.identity,
                destination_identities=[self._participant.identity],
                utterance_id=utterance_id,
            )
            await self._event_bus.publish_json(
                {
                    "type": "agent_response_text",
                    "utterance_id": utterance_id,
                    "participant_identity": self._participant.identity,
                    "text": response_text,
                    "use_llm": False,
                    "llm_intent": "",
                    "search_index": [],
                    "next_step": "",
                },
                destination_identities=[self._participant.identity],
            )
            response_published = True
        finally:
            total_latency_ms = int((time.perf_counter() - started_at) * 1000)
            finalized_to_intent_ms = max(0, intent_ready_time_ms - speech_end_time_ms) if intent_ready_time_ms else 0
            finalized_to_response_ms = max(0, response_ready_time_ms - speech_end_time_ms) if response_ready_time_ms else 0
            finalized_to_tts_start_ms = (
                max(0, tts_synth_start_time_ms - speech_end_time_ms) if tts_synth_start_time_ms else 0
            )
            perceived_latency_ms = finalized_to_tts_start_ms or finalized_to_response_ms
            finalized_to_tts_publish_ms = (
                max(0, tts_publish_start_time_ms - speech_end_time_ms) if tts_publish_start_time_ms else 0
            )
            finalized_to_tts_done_ms = (
                max(0, tts_publish_done_time_ms - speech_end_time_ms) if tts_publish_done_time_ms else 0
            )
            if not response_published and not suppress_response:
                await self._event_bus.publish_json(
                    {
                        "type": "agent_response_text",
                        "utterance_id": utterance_id,
                        "participant_identity": self._participant.identity,
                        "text": response_text,
                        "use_llm": intent.use_llm,
                        "llm_intent": llm_reply.intent if llm_reply else "",
                        "search_index": llm_reply.search_index if llm_reply else [],
                        "next_step": llm_reply.next_step if llm_reply else "",
                    },
                        destination_identities=[self._participant.identity],
                )
            await self._event_bus.publish_json(
                {
                    "type": "turn_timing",
                    "utterance_id": utterance_id,
                    "participant_identity": self._participant.identity,
                    "stt_latency_ms": transcript.stt_latency_ms,
                    "router_latency_ms": router_latency_ms,
                    "classifier_latency_ms": classifier_latency_ms,
                    "llm_latency_ms": llm_latency_ms,
                    "tts_latency_ms": tts_latency_ms,
                    "perceived_latency_ms": perceived_latency_ms,
                    "finalized_to_intent_ms": finalized_to_intent_ms,
                    "finalized_to_response_ms": finalized_to_response_ms,
                    "finalized_to_tts_start_ms": finalized_to_tts_start_ms,
                    "finalized_to_tts_publish_ms": finalized_to_tts_publish_ms,
                    "finalized_to_tts_done_ms": finalized_to_tts_done_ms,
                    "total_latency_ms": total_latency_ms,
                    "error_stage": error_stage,
                },
                destination_identities=[self._participant.identity],
            )
            self._log(
                "turn timing "
                f"participant={self._participant.identity} "
                f"utterance_id={utterance_id} "
                f"stt_ms={transcript.stt_latency_ms} "
                f"router_ms={router_latency_ms} "
                f"classifier_ms={classifier_latency_ms} "
                f"llm_ms={llm_latency_ms} "
                f"tts_ms={tts_latency_ms} "
                f"perceived_latency_ms={perceived_latency_ms} "
                f"finalized_to_intent_ms={finalized_to_intent_ms} "
                f"finalized_to_response_ms={finalized_to_response_ms} "
                f"finalized_to_tts_start_ms={finalized_to_tts_start_ms} "
                f"finalized_to_tts_publish_ms={finalized_to_tts_publish_ms} "
                f"finalized_to_tts_done_ms={finalized_to_tts_done_ms} "
                f"total_ms={total_latency_ms}"
            )
            self._log(
                "turn timestamps "
                f"participant={self._participant.identity} "
                f"utterance_id={utterance_id} "
                f"speech_start_ms={speech_start_time_ms} "
                f"speech_end_ms={speech_end_time_ms} "
                f"stt_done_ms={stt_done_time_ms} "
                f"intent_ready_ms={intent_ready_time_ms} "
                f"response_ready_ms={response_ready_time_ms} "
                f"tts_synth_start_ms={tts_synth_start_time_ms} "
                f"tts_synth_done_ms={tts_synth_done_time_ms} "
                f"tts_publish_start_ms={tts_publish_start_time_ms} "
                f"tts_publish_done_ms={tts_publish_done_time_ms}"
            )
            self._session_logger.write(
                {
                    "session_id": self._session_id,
                    "room_name": self._room.name,
                    "participant_identity": self._participant.identity,
                    "utterance_id": utterance_id,
                    "speech_start_time": speech_start_time_ms,
                    "speech_end_time": speech_end_time_ms,
                    "utterance_duration_ms": duration_ms,
                    "vad_confidence": round(vad_confidence, 4),
                    "stt_text": transcript.text,
                    "stt_confidence": transcript.confidence,
                    "normalized_text": normalized_text,
                    "detected_intent": intent.intent,
                    "intent_confidence": intent.confidence,
                    "use_llm": intent.use_llm,
                    "stt_latency_ms": transcript.stt_latency_ms,
                    "router_latency_ms": router_latency_ms,
                    "classifier_latency_ms": classifier_latency_ms,
                    "total_latency_ms": total_latency_ms,
                    "error_stage": error_stage,
                    "response_text": response_text,
                    "raw_response_text": raw_response_text,
                    "tts_prepared_text": tts_prepared_text,
                    "tts_segments": tts_segments,
                    "filler_added": filler_added,
                    "filler_type": filler_type,
                    "llm_latency_ms": llm_latency_ms,
                    "llm_reply_intent": llm_reply.intent if llm_reply else "",
                    "llm_reply_search_index": llm_reply.search_index if llm_reply else [],
                    "llm_reply_next_step": llm_reply.next_step if llm_reply else "",
                    "llm_raw_text": llm_reply.raw_text if llm_reply else "",
                    "llm_validation_reason": llm_validation_reason,
                    "tts_latency_ms": tts_latency_ms,
                    "stt_done_time_ms": stt_done_time_ms,
                    "intent_ready_time_ms": intent_ready_time_ms,
                    "response_ready_time_ms": response_ready_time_ms,
                    "tts_synth_start_time_ms": tts_synth_start_time_ms,
                    "tts_synth_done_time_ms": tts_synth_done_time_ms,
                    "tts_publish_start_time_ms": tts_publish_start_time_ms,
                    "tts_publish_done_time_ms": tts_publish_done_time_ms,
                    "finalized_to_intent_ms": finalized_to_intent_ms,
                    "finalized_to_response_ms": finalized_to_response_ms,
                    "finalized_to_tts_start_ms": finalized_to_tts_start_ms,
                    "finalized_to_tts_publish_ms": finalized_to_tts_publish_ms,
                    "finalized_to_tts_done_ms": finalized_to_tts_done_ms,
                }
            )
            self._is_speaking = False
            self._is_processing = False
            had_barge_in_pending = self._barge_in_pending
            self._barge_in_pending = False
            await self._publish_status("waiting_for_speech")
            if had_barge_in_pending:
                if self._resume_buffer.size:
                    self._sample_buffer.prepend(self._resume_buffer.to_array())
                while (
                    not self._is_processing
                    and not self._is_speaking
                    and self._sample_buffer.size >= self._vad.window_size
                ):
                    window = self._sample_buffer.pop_front(self._vad.window_size)
                    await self._process_vad_window(window)
            self._resume_buffer.clear()
            self._resume_probe_buffer.clear()
            self._resume_speech_ms = 0
            self._processing_can_be_interrupted = False


class VoiceSessionManager:
    def __init__(
        self,
        *,
        room: rtc.Room,
        config: VoicePipelineConfig,
        event_bus: AgentEventBus,
        llm_service: OpenAiLlmService,
        tts_service: SileroTtsService,
        audio_publisher: LiveKitAudioPublisher,
        log: Callable[[str], None],
    ) -> None:
        self._room = room
        self._config = config
        self._event_bus = event_bus
        self._llm_service = llm_service
        self._tts_service = tts_service
        self._audio_publisher = audio_publisher
        self._log = log
        self._stt = WhisperSttService(config, log)
        self._sessions: dict[str, ParticipantAudioSession] = {}

    async def start_audio_track(
        self,
        *,
        track: rtc.Track,
        participant: rtc.RemoteParticipant,
        track_sid: str | None = None,
    ) -> None:
        session = self._sessions.get(participant.identity)
        if session is None:
            session = ParticipantAudioSession(
                room=self._room,
                participant=participant,
                config=self._config,
                event_bus=self._event_bus,
                stt_service=self._stt,
                llm_service=self._llm_service,
                tts_service=self._tts_service,
                audio_publisher=self._audio_publisher,
                log=self._log,
            )
            self._sessions[participant.identity] = session

        await session.ensure_started(track, track_sid=track_sid)

    async def participant_disconnected(self, participant_identity: str) -> None:
        session = self._sessions.pop(participant_identity, None)
        if session is not None:
            await session.aclose()

    async def aclose(self) -> None:
        sessions = list(self._sessions.values())
        self._sessions.clear()
        await asyncio.gather(*(session.aclose() for session in sessions), return_exceptions=True)
