from __future__ import annotations

import asyncio
import contextlib
import json
import math
import os
import random
import re
import threading
import time
import traceback
import uuid
import wave
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable
import httpx
import numpy as np
import torch
import torchaudio.functional as torchaudio_f
from agent_core import (
    DialogueState,
    KnowledgeBase,
    SessionMemory,
    ToolGraphRuntime,
    build_context_messages,
    inspect_llm_reply,
)
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
    r"\b(?:\d+(?:[\.,]\d+)?|один|одна|два|две|три|четыре|пять|шесть|семь|восемь|девять|десять|полтора)\s*"
    r"(?:млн|миллион|миллиона|миллионов|тыс|тысяч|тысячи|руб|рубль|рубля|рублей)?\b",
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
    dialogue_backend: str
    llm_provider: str
    llm_model: str
    llm_reasoning_effort: str
    llm_timeout_seconds: float
    llm_base_url: str
    llm_api_key: str
    llm_project: str
    llm_temperature: float
    llm_max_tokens: int
    text_api_url: str
    text_api_timeout_seconds: float
    adaptive_classifier_enabled: bool
    tts_enabled: bool
    tts_model_path: Path
    tts_model_url: str
    tts_speaker: str
    tts_speed: float
    tts_sample_rate: int
    tts_publish_sample_rate: int
    tts_frame_ms: int
    tts_provider: str
    tts_segment_pause_ms: int
    tts_normalize_peak: float
    tts_fade_ms: int
    omnivoice_model: str
    omnivoice_device: str
    omnivoice_dtype: str
    omnivoice_num_step: int
    omnivoice_instruct: str
    omnivoice_ref_audio: str
    piper_model: str
    piper_model_dir: str
    piper_use_cuda: bool
    piper_length_scale: float
    half_duplex: bool
    barge_in_enabled: bool
    barge_in_cue: str
    barge_in_cue_ms: int
    barge_in_cue_volume: float
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
        dialogue_backend = env_nonempty("DIALOGUE_BACKEND", "text_api").lower()

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
                os.getenv("PROCESSING_RESUME_MIN_SPEECH_DURATION_MS", "450")
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
            dialogue_backend=dialogue_backend,
            llm_provider=llm_provider,
            llm_model=env_nonempty("LLM_MODEL", default_llm_model),
            llm_reasoning_effort=os.getenv("LLM_REASONING_EFFORT", "low"),
            llm_timeout_seconds=float(os.getenv("LLM_TIMEOUT_SECONDS", "15")),
            llm_base_url=env_nonempty("LLM_BASE_URL", default_llm_base_url),
            llm_api_key=env_nonempty("LLM_API_KEY", default_llm_api_key),
            llm_project=llm_project,
            llm_temperature=float(os.getenv("LLM_TEMPERATURE", "0.2")),
            llm_max_tokens=int(os.getenv("LLM_MAX_TOKENS", "180")),
            text_api_url=env_nonempty("TEXT_API_URL", "http://127.0.0.1:8787"),
            text_api_timeout_seconds=float(os.getenv("TEXT_API_TIMEOUT_SECONDS", "20")),
            adaptive_classifier_enabled=env_bool("ADAPTIVE_CLASSIFIER_ENABLED", False),
            tts_enabled=env_bool("TTS_ENABLED", True),
            tts_model_path=Path(os.getenv("TTS_MODEL_PATH", "/models/silero-tts/ru/v5_4_ru.pt")),
            tts_model_url=os.getenv(
                "TTS_MODEL_URL",
                "https://models.silero.ai/models/tts/ru/v5_4_ru.pt",
            ),
            tts_speaker=os.getenv("TTS_SPEAKER", "aidar"),
            tts_speed=float(os.getenv("TTS_SPEED", "1.12")),
            tts_sample_rate=int(os.getenv("TTS_SAMPLE_RATE", "24000")),
            tts_publish_sample_rate=int(os.getenv("TTS_PUBLISH_SAMPLE_RATE", "24000")),
            tts_frame_ms=int(os.getenv("TTS_FRAME_MS", "20")),
            tts_provider=tts_provider,
            tts_segment_pause_ms=int(os.getenv("TTS_SEGMENT_PAUSE_MS", "240")),
            tts_normalize_peak=float(os.getenv("TTS_NORMALIZE_PEAK", "0.8")),
            tts_fade_ms=int(os.getenv("TTS_FADE_MS", "8")),
            omnivoice_model=os.getenv("OMNIVOICE_MODEL", "k2-fsa/OmniVoice").strip() or "k2-fsa/OmniVoice",
            omnivoice_device=os.getenv("OMNIVOICE_DEVICE", "cuda:0").strip() or "cuda:0",
            omnivoice_dtype=os.getenv("OMNIVOICE_DTYPE", "float16").strip() or "float16",
            omnivoice_num_step=int(os.getenv("OMNIVOICE_NUM_STEP", "8")),
            # instruct = английские атрибуты через запятую ("male, low pitch"); voice-design
            # обучен на ZH/EN, для русского даёт акцент. По умолчанию пусто -> auto-voice
            # (модель берёт родной русский голос). Для стабильного голоса — OMNIVOICE_REF_AUDIO.
            omnivoice_instruct=os.getenv("OMNIVOICE_INSTRUCT", "").strip(),
            omnivoice_ref_audio=os.getenv("OMNIVOICE_REF_AUDIO", "").strip(),
            piper_model=os.getenv("PIPER_MODEL", "ru_RU-dmitri-medium").strip() or "ru_RU-dmitri-medium",
            piper_model_dir=os.getenv("PIPER_MODEL_DIR", "/models/piper").strip() or "/models/piper",
            piper_use_cuda=env_bool("PIPER_USE_CUDA", True),
            piper_length_scale=float(os.getenv("PIPER_LENGTH_SCALE", "0.95")),
            half_duplex=env_bool("HALF_DUPLEX", True),
            barge_in_enabled=env_bool("BARGE_IN_ENABLED", False),
            barge_in_cue=os.getenv("BARGE_IN_CUE", "click").strip().lower() or "click",
            barge_in_cue_ms=int(os.getenv("BARGE_IN_CUE_MS", "45")),
            barge_in_cue_volume=float(os.getenv("BARGE_IN_CUE_VOLUME", "0.18")),
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


@dataclass(slots=True)
class PlaybackState:
    utterance_id: str = ""
    response_text: str = ""
    prepared_text: str = ""
    segments: list[str] = field(default_factory=list)
    next_segment_index: int = 0
    interrupted: bool = False
    last_interrupted_at_ms: int = 0
    completed: bool = True

    def reset(self) -> None:
        self.utterance_id = ""
        self.response_text = ""
        self.prepared_text = ""
        self.segments = []
        self.next_segment_index = 0
        self.interrupted = False
        self.last_interrupted_at_ms = 0
        self.completed = True

    def has_remaining(self) -> bool:
        return self.next_segment_index < len(self.segments)


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


class SpeechSituation(str, Enum):
    OPENING = "opening"
    SLOT_BRIDGE = "slot_bridge"
    CLARIFICATION = "clarification"
    OBJECTION = "objection"
    REPAIR = "repair"
    THINKING = "thinking"
    HANDOFF = "handoff"
    CLOSING = "closing"
    DEFAULT = "default"


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

    _LOCAL_MANAGER_PROMPT = """Ты Влад+имир, менеджер МосИнвестФинанс.
Ты сам звонишь клиенту по вопросу кредита или рефинансирования под залог недвижимости.

Верни только JSON:
{"reply_tts":"...","search_index":["..."],"intent":"...","next_step":"..."}

Правила:
- только русский язык;
- reply_tts: 1–2 коротких предложения для устной речи;
- один смысловой шаг за ход;
- один новый вопрос за раз;
- не повторяй приветствие после первого сообщения;
- не говори "вы позвонили", потому что звонишь ты;
- сначала ответь по сути, если клиент спросил или возразил, потом мягко вернись к следующему шагу;
- не выдумывай условия и продукты;
- не сбрасывай сценарий без причины;
- если данных мало, всё равно верни валидный JSON.

Цель звонка:
1. коротко объяснить повод;
2. подтвердить интерес;
3. узнать нужную сумму;
4. узнать объект и регион;
5. понять обременение и собственника;
6. передать кейс эксперту.

Базовые факты:
- до 70% от рыночной стоимости объекта;
- срок от 1 года до 25 лет;
- ставка от 19% годовых;
- клиент остаётся собственником;
- оригиналы документов остаются у клиента.

Ограничения:
- не перечисляй все условия сразу;
- не спорь с клиентом о старой заявке;
- не придумывай реквизиты, комиссии, одобрения и сроки вне известных фактов;
- не озвучивай внутренние рассуждения."""

    _LOCAL_RENDER_PROMPT = """Ты формулируешь реплику внутри уже выбранного runtime-шага.
Верни только JSON.
Не меняй сценарий и не выбирай новый слот сам.
Сначала ответь по сути, если клиент спросил или возразил, потом вернись к текущему шагу.
reply_tts: 1–2 коротких предложения.
next_step: один короткий следующий шаг менеджера.
Без текста вне JSON."""

    _LOCAL_REPAIR_PROMPT = """Ты быстрый repair-слой агента. Верни только JSON.
Задача: увидеть, что диалог застрял, клиент уже ответил или агент повторяется.
Коротко признай это и продолжи без сброса сценария.
Нельзя повторять тот же вопрос теми же словами.
reply_tts: максимум 1–2 коротких предложения.
next_step: один короткий следующий шаг.
Без текста вне JSON."""

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
            content = str(item.get("content", "") or item.get("text", "")).strip()
            if not content:
                continue
            messages.append({"role": role, "content": content})
        return messages

    def _response_format(self, schema: dict[str, Any]) -> dict[str, Any]:
        if self._is_yandex_provider():
            return {"type": "json_schema", "json_schema": schema}
        return {"type": "json_object"}

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
        parsed, _ = try_parse_json_object(raw_text)

        raw_intent = parsed.get("intent")
        intent = raw_intent.strip() if isinstance(raw_intent, str) else fallback_intent
        if intent not in cls._CLASSIFIER_ALLOWED_INTENTS:
            intent = fallback_intent

        raw_action = parsed.get("action")
        action = raw_action.strip() if isinstance(raw_action, str) else fallback_action
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
        state_text = str(dialogue_state.get("session_state_text", "")).strip()
        compact_state = {
            "stage": str(dialogue_state.get("stage", "")).strip(),
            "current_node": str(dialogue_state.get("current_node", "")).strip(),
            "awaiting_field": str(dialogue_state.get("awaiting_field", "")).strip(),
            "next_required_field": str(dialogue_state.get("next_required_field", "")).strip(),
            "known_facts": dialogue_state.get("known_facts", {}),
            "session_state": dialogue_state.get("session_state", {}),
        }
        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": self._CLASSIFIER_PROMPT,
            },
            {
                "role": "system",
                "content": (
                    "Структурированное состояние звонка. "
                    "Используй его как источник контекста.\n"
                    f"{state_text or json.dumps(compact_state, ensure_ascii=False)}"
                ),
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

        completion = await client.chat.completions.create(
            model=self._config.llm_model,
            temperature=0.0,
            max_tokens=min(160, self._config.llm_max_tokens),
            response_format=self._response_format(self._CLASSIFIER_JSON_SCHEMA),
            messages=messages,
        )
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
        graph_context: dict[str, Any] | None = None,
        temperature_override: float | None = None,
        max_tokens_override: int | None = None,
    ) -> tuple[LlmReply, int]:
        if not self.enabled:
            raise RuntimeError("LLM is disabled by configuration")

        client = self._ensure_client()
        started_at = time.perf_counter()
        prompt_text = self._LOCAL_MANAGER_PROMPT
        if graph_context:
            if graph_context.get("mode") == "adaptive_repair" or graph_context.get("stalled_slot_recovery"):
                prompt_text = self._LOCAL_REPAIR_PROMPT
            else:
                prompt_text = self._LOCAL_RENDER_PROMPT
        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": prompt_text,
            },
            {
                "role": "system",
                "content": (
                    "Строго соблюдай JSON-схему ответа. "
                    'Формат: {"reply_tts":"строка","search_index":["строка"],"intent":"строка","next_step":"строка"}. '
                    "Никакого текста вне JSON."
                ),
            },
            *(
                [
                    {
                        "role": "system",
                        "content": (
                            "Текущий runtime-узел сценария.\n"
                            f"{json.dumps(graph_context, ensure_ascii=False)}\n"
                            "Отвечай только в рамках этого узла. "
                            "Не перепрыгивай через следующий шаг и не расширяй сценарий без причины."
                        ),
                    }
                ]
                if graph_context
                else []
            ),
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

        completion = await client.chat.completions.create(
            model=self._config.llm_model,
            temperature=self._config.llm_temperature if temperature_override is None else temperature_override,
            max_tokens=self._config.llm_max_tokens if max_tokens_override is None else max_tokens_override,
            response_format=self._response_format(self._LLM_JSON_SCHEMA),
            messages=messages,
        )
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
        ), latency_ms

    async def warmup(self) -> None:
        if not self.enabled:
            return
        client = self._ensure_client()
        started_at = time.perf_counter()
        try:
            await client.chat.completions.create(
                model=self._config.llm_model,
                temperature=0.0,
                max_tokens=8,
                response_format=self._response_format(self._LLM_JSON_SCHEMA),
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Верни только JSON с полями "
                            "reply_tts, search_index, intent, next_step."
                        ),
                    },
                    {
                        "role": "user",
                        "content": "ok",
                    },
                ],
            )
            latency_ms = int((time.perf_counter() - started_at) * 1000)
            self._log(f"llm warmup done latency_ms={latency_ms}")
        except Exception as exc:
            self._log(f"llm warmup skipped: {exc}")


class TextApiLlmService:
    def __init__(self, config: VoicePipelineConfig, log: Callable[[str], None]) -> None:
        self._config = config
        self._log = log
        self._client: httpx.AsyncClient | None = None

    @property
    def enabled(self) -> bool:
        return self._config.llm_enabled and bool(self._config.text_api_url)

    @property
    def uses_text_api_backend(self) -> bool:
        return True

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            base_url = self._config.text_api_url.rstrip("/")
            self._client = httpx.AsyncClient(
                base_url=base_url,
                timeout=self._config.text_api_timeout_seconds,
            )
        return self._client

    async def warmup(self) -> None:
        if not self.enabled:
            return
        started_at = time.perf_counter()
        try:
            response = await self._ensure_client().get("/healthz")
            response.raise_for_status()
            latency_ms = int((time.perf_counter() - started_at) * 1000)
            self._log(f"text_api warmup done latency_ms={latency_ms}")
        except Exception as exc:
            self._log(f"text_api warmup skipped: {exc}")

    async def start_session(
        self,
        *,
        session_id: str,
        phone: str,
        known_facts: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        response = await self._ensure_client().post(
            "/session/start",
            json={
                "session_id": session_id,
                "phone": phone,
                "known_facts": known_facts or {},
            },
        )
        response.raise_for_status()
        return response.json()

    async def message(self, *, session_id: str, text: str) -> tuple[dict[str, Any], int]:
        started_at = time.perf_counter()
        response = await self._ensure_client().post(
            "/session/message",
            json={"session_id": session_id, "text": text},
        )
        latency_ms = int((time.perf_counter() - started_at) * 1000)
        response.raise_for_status()
        return response.json(), latency_ms

    async def reset_session(self, *, session_id: str) -> None:
        if not self.enabled:
            return
        try:
            await self._ensure_client().post("/session/reset", json={"session_id": session_id})
        except Exception as exc:
            self._log(f"text_api reset skipped session_id={session_id}: {exc}")

    async def aclose(self) -> None:
        if self._client is None:
            return
        await self._client.aclose()
        self._client = None


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


def try_parse_json_object(text: str) -> tuple[dict[str, Any], str]:
    payload_text = extract_first_json_object(text)
    try:
        maybe_parsed = json.loads(payload_text)
    except Exception:
        return {}, payload_text
    if not isinstance(maybe_parsed, dict):
        return {}, payload_text
    return maybe_parsed, payload_text


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

def normalize_for_compare(text: str) -> str:
    value = text.lower().replace("ё", "е")
    value = re.sub(r"[^\wа-яa-z0-9\s]", " ", value, flags=re.IGNORECASE)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def looks_like_meaningful_unknown(text: str) -> bool:
    value = normalize_for_compare(text)
    if not value:
        return False

    if value in {"ммм", "эм", "ээ", "а", "и", "ну"}:
        return False

    if len(value.split()) >= 2:
        return True

    meaningful_singletons = {
        "москва",
        "москвe",
        "питерe",
        "спб",
        "авто",
        "дом",
        "квартира",
        "свободен",
        "свободна",
        "залог",
        "залоге",
        "да",
        "нет",
    }
    return value in meaningful_singletons


def detect_repair_markers(text: str) -> list[str]:
    lowered = normalize_for_compare(text)
    flags: list[str] = []

    not_actual_markers = (
        "не актуален",
        "не актуально",
        "не нужно",
        "не интересно",
        "нет спасибо",
        "вопрос закрыт",
        "уже решил",
    )
    if any(marker in lowered for marker in not_actual_markers):
        flags.append("client_not_actual")

    confusion_markers = (
        "не понял",
        "что это за вопрос",
        "по какому поводу",
        "что вы хотите",
        "не понимаю",
        "зачем это",
    )
    if any(marker in lowered for marker in confusion_markers):
        flags.append("client_confused")

    repeat_complaint_markers = (
        "нельзя один и тот же",
        "одно и то же",
        "повторяете",
        "я уже сказал",
        "я же говорю",
        "я ответил",
    )
    if any(marker in lowered for marker in repeat_complaint_markers):
        flags.append("client_complains_repeat")

    irritation_markers = (
        "ну и что",
        "и что",
        "отстаньте",
        "не звоните",
    )
    if any(marker in lowered for marker in irritation_markers):
        flags.append("client_pressure")

    return flags


def should_force_llm_repair(
    *,
    intent: IntentResult,
    normalized_text: str,
    state: dict[str, Any],
    candidate_response: str,
) -> tuple[bool, str]:
    text = normalize_for_compare(normalized_text)
    awaiting = str(state.get("awaiting_field", "") or state.get("next_required_field", "")).strip()
    last_agent = str(state.get("last_agent_text", "")).strip()

    repair_flags = detect_repair_markers(normalized_text)
    if repair_flags:
        return True, ",".join(repair_flags)

    if intent.use_llm or intent.action == Action.CALL_LLM.value:
        return True, "intent_requested_llm"

    if intent.intent in {Intent.UNKNOWN_SHORT.value, Intent.CLARIFY.value} and looks_like_meaningful_unknown(normalized_text):
        return True, "meaningful_unknown_short"

    if text in {"да", "нет", "угу", "ага"} and awaiting:
        return True, "short_answer_needs_context"

    if candidate_response and last_agent:
        if normalize_for_compare(candidate_response) == normalize_for_compare(last_agent):
            return True, "repeated_agent_response"

    return False, ""

def parse_llm_reply(
    raw_text: str,
    *,
    fallback_reply: str,
    fallback_intent: str,
    fallback_next_step: str,
    fallback_search_seed: str,
) -> LlmReply:
    parsed, _ = try_parse_json_object(raw_text)

    raw_reply_value = parsed.get("reply_tts")
    reply_source = raw_reply_value.strip() if isinstance(raw_reply_value, str) else ""
    if not reply_source or reply_source.lstrip().startswith("{") or '"reply_tts"' in reply_source:
        reply_source = fallback_reply

    reply_tts = sanitize_voice_response(reply_source, fallback=fallback_reply)

    raw_intent = parsed.get("intent")
    intent = raw_intent.strip() if isinstance(raw_intent, str) else ""
    if not intent:
        intent = fallback_intent

    raw_next_step = parsed.get("next_step")
    next_step = raw_next_step.strip() if isinstance(raw_next_step, str) else ""
    if not next_step:
        next_step = fallback_next_step

    raw_search_index = parsed.get("search_index", [])
    search_values: list[str] = []
    if isinstance(raw_search_index, list):
        search_values = [item for item in raw_search_index if isinstance(item, str)]
    elif isinstance(raw_search_index, str):
        search_values = [raw_search_index]

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
    merged: list[str] = []
    index = 0
    while index < len(segments):
        current = segments[index]
        normalized = normalize_for_compare(current.replace("+", ""))
        is_short_lead = normalized in {
            "алло",
            "ало",
            "ага",
            "так",
            "такс",
            "понял",
            "хорошо",
            "смотрите",
            "да",
        }
        if index == 0 and is_short_lead and index + 1 < len(segments):
            merged.append(f"{current} {segments[index + 1]}".strip())
            index += 2
            continue
        merged.append(current)
        index += 1
    return merged


def pause_after_segment_ms(segment: str, default_ms: int) -> int:
    value = segment.strip().lower()
    if not value:
        return default_ms
    if value.endswith("?"):
        return max(default_ms, 180)
    if value in {"ага.", "так.", "понял.", "хорошо.", "смотрите.", "алло."}:
        return 120
    if "извините" in value or "вы правы" in value:
        return 220
    if len(value) < 18:
        return 100
    return default_ms


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


def _ssml_escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# Filler words that sound bad at the START of a spoken reply — stripped there,
# kept only mid-sentence (where they read as natural hesitation).
_LEADING_FILLERS = ("ну смотрите", "ну вот", "ну", "вот", "так вот", "угу", "ага", "э", "эм", "ааа", "аа")
# Emphasis / transition words that get an accent pause before them mid-sentence.
_ACCENT_WORDS = (
    "хорошо", "понятно", "понял", "поняла", "ясно", "отлично", "конечно",
    "смотрите", "значит", "итак", "так вот", "договорились", "замечательно",
)


class TtsMarkupService:
    def __init__(self, provider: str = "silero") -> None:
        self._provider = (provider or "silero").lower()
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
        text = self._strip_leading_filler(text)
        text = self._accent_pauses(text)
        text = self._apply_pronunciation(text)
        text = self._apply_stress(text)
        text = self._split_for_speech(text)
        # Stress marks ("+") are Silero syntax; other engines read them literally.
        if self._provider != "silero":
            text = text.replace("+", "")
        return text

    @staticmethod
    def _clean(text: str) -> str:
        return re.sub(r"\s+", " ", text.strip())

    @staticmethod
    def _normalize_numbers(text: str) -> str:
        return text.replace("%", " процентов")

    @staticmethod
    def _strip_leading_filler(text: str) -> str:
        value = text.strip()
        low = value.lower()
        for filler in _LEADING_FILLERS:
            if low.startswith(filler + ",") or low.startswith(filler + " ") or low == filler:
                rest = value[len(filler):].lstrip(" ,—-").strip()
                if rest:
                    return rest[0].upper() + rest[1:]
        return value

    @staticmethod
    def _accent_pauses(text: str) -> str:
        # Insert a comma-pause before an emphasis word when it appears mid-sentence
        # (preceded by a word and not already after a pause). Accent only in the
        # middle — at the start it sounds bad, so we don't touch sentence openings.
        result = text
        for word in _ACCENT_WORDS:
            result = re.sub(
                rf"(\w)\s+({re.escape(word)})\b",
                lambda m: f"{m.group(1)}, {m.group(2)}",
                result,
                flags=re.IGNORECASE,
            )
        return re.sub(r"\s*,\s*,", ",", result)  # collapse accidental double commas

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
        value = re.sub(r"\s+", " ", text).strip()
        return value.replace(" — ", ". ")


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
        if self._starts_with_spoken_prefix(lowered_original):
            self._previous_had_filler = False
            return VoiceStyleResult(original, False, "", original)
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

    @staticmethod
    def _starts_with_spoken_prefix(lowered_text: str) -> bool:
        prefixes = (
            "алло.",
            "алл+о.",
            "ага.",
            "так.",
            "такс.",
            "хорошо.",
            "понял.",
            "смотрите.",
            "да, смотрите.",
            "нуу, смотрите.",
            "так, секунду.",
            "сейчас сориентирую.",
            "понимаю вас.",
            "да, вы правы",
            "извините,",
        )
        return any(lowered_text.startswith(prefix) for prefix in prefixes)


class SalesSpeechStyler:
    def __init__(self, config: VoicePipelineConfig) -> None:
        self._config = config
        self._random = random.Random()
        self._last_filler = ""

    def style(
        self,
        text: str,
        *,
        intent: str,
        stage: str,
        situation: str,
        known_facts: dict[str, Any] | None = None,
        repair_reason: str = "",
    ) -> str:
        value = re.sub(r"\s+", " ", text.strip())
        if not value:
            return value

        known_facts = known_facts or {}
        lowered = value.lower()
        if "не расслышал" in lowered or "повторите" in lowered:
            return value

        if situation == SpeechSituation.REPAIR.value:
            return self._style_repair(value, repair_reason=repair_reason)
        if situation == SpeechSituation.SLOT_BRIDGE.value:
            return self._style_slot_bridge(value, known_facts=known_facts)
        if situation == SpeechSituation.OBJECTION.value:
            return self._style_objection(value)
        if situation == SpeechSituation.THINKING.value:
            return self._style_thinking(value)
        if situation == SpeechSituation.HANDOFF.value:
            return self._style_handoff(value)
        if situation == SpeechSituation.OPENING.value and not lowered.startswith(("алло", "алл")):
            return f"Алл+о. {value}"
        return self._light_prefix(value, intent=intent, stage=stage)

    def speech_rate(self, *, situation: str, intent: str) -> float:
        return 1.0

    def _choose(self, variants: list[str]) -> str:
        items = [item for item in variants if item and item != self._last_filler]
        if not items:
            items = [item for item in variants if item]
        if not items:
            return ""
        picked = self._random.choice(items)
        self._last_filler = picked
        return picked

    def _light_prefix(self, text: str, *, intent: str, stage: str) -> str:
        if intent in {
            Intent.LINE_ISSUE.value,
            Intent.REPEAT.value,
            Intent.GREETING.value,
            Intent.UNKNOWN_SHORT.value,
            Intent.CLARIFY.value,
        }:
            return text
        if text.endswith("?") and len(text) <= 80:
            return text
        variants = ["Ага.", "Так.", "Хорошо.", "Понял."]
        if intent in {Intent.COMPLEX_REQUEST.value, Intent.CLARIFY.value}:
            variants = ["Смотрите.", "Да, смотрите.", "Так, сейчас."]
        elif stage == "qualification":
            variants = ["Ага.", "Так.", "Понял."]
        prefix = self._choose(variants)
        return f"{prefix} {text}".strip() if prefix else text

    def _style_slot_bridge(self, text: str, *, known_facts: dict[str, Any]) -> str:
        lowered = text.lower()
        amount = str(known_facts.get("amount") or known_facts.get("нужная_сумма") or "").strip()
        object_type = str(known_facts.get("вид_объекта") or known_facts.get("object_type") or "").strip()
        region = str(known_facts.get("region") or known_facts.get("регион") or "").strip()
        encumbrance = str(known_facts.get("обременение") or known_facts.get("collateral") or "").strip()

        if amount and "какая недвижимость" in lowered:
            return (
                f"Ага, {amount} понял. С такой суммой, скорее всего, можно работать. "
                f"Какая недвижимость есть в собственности?"
            )
        if object_type and "регион" in lowered:
            return f"Понял, {object_type}. В каком регионе находится объект?"
        if region and ("залог" in lowered or "обремен" in lowered):
            return f"Хорошо, объект в {region}. Он сейчас свободен от залога или уже в обременении?"
        if encumbrance and "собствен" in lowered:
            enc_lower = encumbrance.lower()
            if "свобод" in enc_lower or "без" in enc_lower or "нет" in enc_lower:
                return "Понял, объект свободен от залога. Собственник вы?"
            return "Понял, по обременению зафиксировал. Собственник объекта вы?"
        return self._light_prefix(text, intent=Intent.SLOT_ANSWER.value, stage="qualification")

    def _style_repair(self, text: str, *, repair_reason: str) -> str:
        if "repeat" in repair_reason or "repeated" in repair_reason:
            return f"Да, вы правы, извините, повторился. {text}"
        if "confused" in repair_reason or "не понял" in repair_reason:
            return f"Да, понял вас. Переформулирую проще. {text}"
        if "not_actual" in repair_reason:
            return f"Понял, не буду давить. {text}"
        return text

    def _style_objection(self, text: str) -> str:
        prefix = self._choose(
            [
                "Понимаю вас.",
                "Да, логичный вопрос.",
                "Смотрите, объясню коротко.",
            ]
        )
        return f"{prefix} {text}".strip() if prefix else text

    def _style_thinking(self, text: str) -> str:
        prefix = self._choose(
            [
                "Сейчас сориентирую.",
                "Смотрите.",
            ]
        )
        return f"{prefix} {text}".strip() if prefix else text

    def _style_handoff(self, text: str) -> str:
        return f"Отлично. {text}"


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

    def _valid_speaker(self, model: Any, requested: str) -> str:
        """Silero v5_4_ru speakers: aidar/baya/kseniya/xenia/eugene vary by build.
        Fall back to a known male voice so an unknown name never kills synthesis."""
        speaker = (requested or "").strip() or "aidar"
        available = getattr(model, "speakers", None)
        if available and speaker not in available:
            fallback = "aidar" if "aidar" in available else available[0]
            self._log(f"tts speaker {speaker!r} not available {list(available)} -> {fallback!r}")
            return fallback
        return speaker

    def _render_segment_pcm(self, model: Any, request: TtsRequest, segment: str) -> np.ndarray:
        speaker = self._valid_speaker(model, request.speaker or self._config.tts_speaker)
        sr = self._config.tts_sample_rate
        # base speech rate (faster on average) unless the request overrides it
        speed = request.speed if (request.speed and abs(request.speed - 1.0) > 1e-3) else self._config.tts_speed

        audio = None
        if abs(speed - 1.0) > 1e-3:
            # Silero supports SSML prosody rate; pitch-preserving speed change.
            rate = max(50, min(200, int(round(speed * 100))))
            ssml = f'<speak><prosody rate="{rate}%">{_ssml_escape(segment)}</prosody></speak>'
            try:
                audio = model.apply_tts(
                    ssml_text=ssml, speaker=speaker, sample_rate=sr, put_accent=True, put_yo=True
                )
            except Exception as exc:  # SSML unsupported -> fall back to plain (no crash)
                self._log(f"tts ssml rate failed, fallback to plain: {exc}")
                audio = None
        if audio is None:
            # put_accent/put_yo = автоматические ударения и «ё» по всему тексту
            audio = model.apply_tts(
                text=segment, speaker=speaker, sample_rate=sr, put_accent=True, put_yo=True
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
        return pcm16

    def synthesize_segment(
        self,
        request: TtsRequest,
        segment: str,
        *,
        trailing_pause_ms: int = 0,
    ) -> tuple[np.ndarray, int, int]:
        started_at = time.perf_counter()
        with self._lock:
            model = self._ensure_model()
            pcm16 = self._render_segment_pcm(model, request, segment)
        if trailing_pause_ms > 0:
            pcm16 = np.concatenate(
                [pcm16, silence_ms(trailing_pause_ms, self._config.tts_sample_rate)]
            )
        pcm16 = normalize_peak(pcm16, target_peak=self._config.tts_normalize_peak)
        latency_ms = int((time.perf_counter() - started_at) * 1000)
        return pcm16, self._config.tts_sample_rate, latency_ms

    def synthesize_segments(self, request: TtsRequest, segments: list[str]) -> tuple[np.ndarray, int, int]:
        started_at = time.perf_counter()
        with self._lock:
            model = self._ensure_model()
            rendered_segments: list[np.ndarray] = []
            for index, segment in enumerate(segments):
                pcm16 = self._render_segment_pcm(model, request, segment)
                rendered_segments.append(pcm16)
                if index < len(segments) - 1:
                    pause_ms = pause_after_segment_ms(segment, self._config.tts_segment_pause_ms)
                    if pause_ms <= 0:
                        continue
                    rendered_segments.append(
                        silence_ms(pause_ms, self._config.tts_sample_rate)
                    )

        if not rendered_segments:
            pcm16 = np.zeros(0, dtype=np.int16)
        else:
            pcm16 = np.concatenate(rendered_segments)
        pcm16 = normalize_peak(pcm16, target_peak=self._config.tts_normalize_peak)
        latency_ms = int((time.perf_counter() - started_at) * 1000)
        return pcm16, self._config.tts_sample_rate, latency_ms


class OmniVoiceTtsService:
    """TTS via k2-fsa/OmniVoice (zero-shot multilingual, Russian supported).

    Same interface as SileroTtsService: synthesize_segment/synthesize_segments ->
    (pcm16 int16, sample_rate, latency_ms). Language is implicit in the (Russian)
    text. Voice is set once via `instruct` (voice design) or `ref_audio` (clone);
    with neither, OmniVoice picks an auto voice.
    """

    SAMPLE_RATE = 24000

    # Seed line used to lock a single voice for the whole process when no ref
    # audio is provided. We know its text, so it can serve as the clone reference.
    _SEED_TEXT = "Добрый день. Меня зовут Владимир, компания МосИнвестФинанс."

    def __init__(self, config: VoicePipelineConfig, log: Callable[[str], None]) -> None:
        self._config = config
        self._log = log
        self._lock = threading.Lock()
        self._model: Any | None = None
        # voice lock: fixed reference so every utterance uses the SAME voice
        self._ref_audio: str = config.omnivoice_ref_audio
        self._ref_text: str = ""

    def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model
        import torch  # local import: heavy, only when this provider is used
        from omnivoice import OmniVoice

        dtype = getattr(torch, self._config.omnivoice_dtype, torch.float16)
        self._log(f"loading omnivoice model={self._config.omnivoice_model} device={self._config.omnivoice_device}")
        self._model = OmniVoice.from_pretrained(
            self._config.omnivoice_model,
            device_map=self._config.omnivoice_device,
            dtype=dtype,
        )
        self._log("omnivoice model ready")
        return self._model

    def _raw_generate(self, model: Any, text: str, *, speed: float, **extra: Any) -> np.ndarray:
        kwargs: dict[str, Any] = {
            "text": text,
            "num_step": self._config.omnivoice_num_step,
            "speed": float(speed),
            **extra,
        }
        try:
            audio = model.generate(**kwargs)
        except Exception as exc:
            if "instruct" in kwargs or "ref_audio" in kwargs:
                self._log(f"omnivoice generate failed ({exc}); retrying auto-voice")
                kwargs.pop("instruct", None)
                kwargs.pop("ref_audio", None)
                kwargs.pop("ref_text", None)
                audio = model.generate(**kwargs)
            else:
                raise
        if isinstance(audio, list):
            audio = np.concatenate([np.asarray(a).reshape(-1) for a in audio]) if audio else np.zeros(0)
        return np.asarray(audio, dtype=np.float32).reshape(-1)

    def _ensure_voice_lock(self, model: Any) -> None:
        """Lock one voice for the whole process: synthesize a seed line once and
        reuse it as the clone reference so the voice never changes per call."""
        if self._ref_audio:
            return
        try:
            import soundfile as sf

            seed_kwargs: dict[str, Any] = {}
            if self._config.omnivoice_instruct:
                seed_kwargs["instruct"] = self._config.omnivoice_instruct
            else:
                seed_kwargs["instruct"] = "male, low pitch"  # bias the locked voice to male
            seed = self._raw_generate(model, self._SEED_TEXT, speed=1.0, **seed_kwargs)
            ref_dir = Path("/tmp/voice-agent")
            ref_dir.mkdir(parents=True, exist_ok=True)
            ref_path = ref_dir / "omnivoice_ref.wav"
            sf.write(str(ref_path), seed.astype(np.float32), self.SAMPLE_RATE)
            self._ref_audio = str(ref_path)
            self._ref_text = self._SEED_TEXT
            self._log(f"omnivoice voice locked -> {ref_path}")
        except Exception as exc:
            # If locking fails, leave ref empty; auto-voice still produces sound.
            self._log(f"omnivoice voice lock failed ({exc}); using auto-voice")

    def _render_segment_pcm(self, model: Any, request: TtsRequest, segment: str) -> np.ndarray:
        speed = request.speed if (request.speed and abs(request.speed - 1.0) > 1e-3) else self._config.tts_speed
        self._ensure_voice_lock(model)
        extra: dict[str, Any] = {}
        if self._ref_audio:
            extra["ref_audio"] = self._ref_audio
            if self._ref_text:
                extra["ref_text"] = self._ref_text
        elif self._config.omnivoice_instruct:
            extra["instruct"] = self._config.omnivoice_instruct
        audio_np = self._raw_generate(model, segment, speed=speed, **extra)
        audio_np = np.clip(audio_np, -1.0, 1.0)
        pcm16 = (audio_np * 32767.0).astype(np.int16)
        pcm16 = trim_silence(pcm16)
        pcm16 = apply_fade(pcm16, sample_rate=self.SAMPLE_RATE, fade_ms=self._config.tts_fade_ms)
        return pcm16

    def synthesize_segment(
        self,
        request: TtsRequest,
        segment: str,
        *,
        trailing_pause_ms: int = 0,
    ) -> tuple[np.ndarray, int, int]:
        started_at = time.perf_counter()
        with self._lock:
            model = self._ensure_model()
            pcm16 = self._render_segment_pcm(model, request, segment)
        if trailing_pause_ms > 0:
            pcm16 = np.concatenate([pcm16, silence_ms(trailing_pause_ms, self.SAMPLE_RATE)])
        pcm16 = normalize_peak(pcm16, target_peak=self._config.tts_normalize_peak)
        latency_ms = int((time.perf_counter() - started_at) * 1000)
        return pcm16, self.SAMPLE_RATE, latency_ms

    def synthesize_segments(self, request: TtsRequest, segments: list[str]) -> tuple[np.ndarray, int, int]:
        # Diffusion synthesis is expensive per call — render the WHOLE reply in a
        # single generate() (faster than per-segment, and one consistent voice).
        started_at = time.perf_counter()
        text = " ".join(s.strip() for s in segments if s and s.strip())
        with self._lock:
            model = self._ensure_model()
            pcm16 = self._render_segment_pcm(model, request, text) if text else np.zeros(0, dtype=np.int16)
        pcm16 = normalize_peak(pcm16, target_peak=self._config.tts_normalize_peak)
        latency_ms = int((time.perf_counter() - started_at) * 1000)
        return pcm16, self.SAMPLE_RATE, latency_ms

    def warmup(self) -> None:
        """Load the model and lock the voice BEFORE taking calls, so the first
        turn isn't slowed by model load + voice-lock."""
        try:
            with self._lock:
                model = self._ensure_model()
                self._ensure_voice_lock(model)
        except Exception as exc:
            self._log(f"omnivoice warmup skipped: {exc}")


class PiperTtsService:
    """Fast local neural TTS (Piper, ONNX). Near real-time, native Russian voices,
    no cloud. Same interface as Silero: synthesize_segment/_segments -> pcm16/sr/ms.
    Voice is a single fixed ONNX model, so it never changes between turns."""

    _HF_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main"

    def __init__(self, config: VoicePipelineConfig, log: Callable[[str], None]) -> None:
        self._config = config
        self._log = log
        self._lock = threading.Lock()
        self._voice: Any | None = None
        self._sample_rate = 22050

    def _voice_paths(self) -> tuple[Path, Path]:
        # ru_RU-dmitri-medium -> ru/ru_RU/dmitri/medium/ru_RU-dmitri-medium.onnx
        name = self._config.piper_model
        lang_full = name.split("-")[0]              # ru_RU
        lang = lang_full.split("_")[0]              # ru
        speaker = name.split("-")[1] if "-" in name else "dmitri"
        quality = name.split("-")[2] if name.count("-") >= 2 else "medium"
        rel = f"{lang}/{lang_full}/{speaker}/{quality}/{name}.onnx"
        base = Path(self._config.piper_model_dir)
        return base / f"{name}.onnx", base / rel

    def _ensure_voice(self) -> Any:
        if self._voice is not None:
            return self._voice
        from piper import PiperVoice

        base = Path(self._config.piper_model_dir)
        base.mkdir(parents=True, exist_ok=True)
        onnx = base / f"{self._config.piper_model}.onnx"
        cfg = base / f"{self._config.piper_model}.onnx.json"
        if not onnx.is_file() or not cfg.is_file():
            _, rel = self._voice_paths()
            url_onnx = f"{self._HF_BASE}/{rel.relative_to(base).as_posix()}"
            self._log(f"downloading piper voice {self._config.piper_model} from {url_onnx}")
            import torch

            torch.hub.download_url_to_file(url_onnx, str(onnx))
            torch.hub.download_url_to_file(url_onnx + ".json", str(cfg))
        self._log(f"loading piper voice={onnx} cuda={self._config.piper_use_cuda}")
        try:
            self._voice = PiperVoice.load(str(onnx), use_cuda=self._config.piper_use_cuda)
        except Exception as exc:
            self._log(f"piper cuda load failed ({exc}); loading on CPU")
            self._voice = PiperVoice.load(str(onnx), use_cuda=False)
        sr = getattr(getattr(self._voice, "config", None), "sample_rate", None)
        if isinstance(sr, int) and sr > 0:
            self._sample_rate = sr
        self._log(f"piper voice ready sample_rate={self._sample_rate}")
        return self._voice

    def _syn_config(self, request: TtsRequest) -> Any:
        from piper import SynthesisConfig

        # length_scale < 1.0 => faster speech. Map request.speed/config to it.
        speed = request.speed if (request.speed and abs(request.speed - 1.0) > 1e-3) else self._config.tts_speed
        length_scale = self._config.piper_length_scale
        if speed and speed > 0:
            length_scale = max(0.5, min(2.0, self._config.piper_length_scale / speed))
        return SynthesisConfig(length_scale=length_scale, normalize_audio=False)

    def _render_pcm(self, voice: Any, request: TtsRequest, text: str) -> np.ndarray:
        syn = self._syn_config(request)
        chunks: list[np.ndarray] = []
        for chunk in voice.synthesize(text, syn_config=syn):
            pcm = np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16)
            chunks.append(pcm)
            sr = getattr(chunk, "sample_rate", None)
            if isinstance(sr, int) and sr > 0:
                self._sample_rate = sr
        pcm16 = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int16)
        return apply_fade(pcm16, sample_rate=self._sample_rate, fade_ms=self._config.tts_fade_ms)

    def synthesize_segment(
        self, request: TtsRequest, segment: str, *, trailing_pause_ms: int = 0
    ) -> tuple[np.ndarray, int, int]:
        started_at = time.perf_counter()
        with self._lock:
            voice = self._ensure_voice()
            pcm16 = self._render_pcm(voice, request, segment)
        if trailing_pause_ms > 0:
            pcm16 = np.concatenate([pcm16, silence_ms(trailing_pause_ms, self._sample_rate)])
        pcm16 = normalize_peak(pcm16, target_peak=self._config.tts_normalize_peak)
        latency_ms = int((time.perf_counter() - started_at) * 1000)
        return pcm16, self._sample_rate, latency_ms

    def synthesize_segments(self, request: TtsRequest, segments: list[str]) -> tuple[np.ndarray, int, int]:
        started_at = time.perf_counter()
        with self._lock:
            voice = self._ensure_voice()
            rendered: list[np.ndarray] = []
            for index, segment in enumerate(segments):
                rendered.append(self._render_pcm(voice, request, segment))
                if index < len(segments) - 1:
                    pause_ms = pause_after_segment_ms(segment, self._config.tts_segment_pause_ms)
                    if pause_ms > 0:
                        rendered.append(silence_ms(pause_ms, self._sample_rate))
        pcm16 = np.concatenate(rendered) if rendered else np.zeros(0, dtype=np.int16)
        pcm16 = normalize_peak(pcm16, target_peak=self._config.tts_normalize_peak)
        latency_ms = int((time.perf_counter() - started_at) * 1000)
        return pcm16, self._sample_rate, latency_ms

    def warmup(self) -> None:
        try:
            with self._lock:
                self._ensure_voice()
        except Exception as exc:
            self._log(f"piper warmup skipped: {exc}")


def build_tts_service(config: VoicePipelineConfig, log: Callable[[str], None]):
    provider = config.tts_provider
    if provider == "omnivoice":
        try:
            import importlib.util

            if importlib.util.find_spec("omnivoice") is None:
                raise ImportError("omnivoice package not installed")
            log("tts provider: omnivoice")
            return OmniVoiceTtsService(config, log)
        except Exception as exc:
            log(f"omnivoice unavailable ({exc}); falling back to silero TTS")
    elif provider == "piper":
        try:
            import importlib.util

            if importlib.util.find_spec("piper") is None:
                raise ImportError("piper-tts package not installed")
            log("tts provider: piper")
            return PiperTtsService(config, log)
        except Exception as exc:
            log(f"piper unavailable ({exc}); falling back to silero TTS")
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

    def _build_cue(self) -> np.ndarray:
        """A short, soft phone-like cue played on barge-in so the agent voice
        doesn't cut abruptly. 'click' = gentle blip, 'beep' = soft tone."""
        kind = self._config.barge_in_cue
        if kind in ("", "none", "off"):
            return np.zeros(0, dtype=np.int16)
        sr = self._config.tts_publish_sample_rate
        n = max(1, int(sr * self._config.barge_in_cue_ms / 1000))
        t = np.arange(n, dtype=np.float32) / sr
        freq = 1500.0 if kind == "beep" else 900.0
        wave = np.sin(2 * np.pi * freq * t).astype(np.float32)
        if kind != "beep":
            # short two-tone "tk" click — quick decay
            wave = wave * np.exp(-t * 60.0).astype(np.float32)
        # smooth fade in/out to avoid pops
        fade = max(1, int(sr * 0.006))
        if 2 * fade < n:
            wave[:fade] *= np.linspace(0.0, 1.0, fade, dtype=np.float32)
            wave[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
        wave *= max(0.0, min(1.0, self._config.barge_in_cue_volume))
        return np.clip(wave * 32767.0, -32768.0, 32767.0).astype(np.int16)

    async def interrupt_with_cue(self) -> None:
        """Stop current playback and play the barge-in cue (phone-like switch)."""
        cue = self._build_cue()
        if len(cue) == 0 or self._source is None:
            self.interrupt_playback()
            return
        # Bump generation to stop the producer loop, but DON'T clear_queue here —
        # clearing puts the source in a transient InvalidState where capture fails.
        self._playback_generation += 1
        self._source.clear_queue()
        spc = int(self._config.tts_publish_sample_rate * self._config.tts_frame_ms / 1000) or 480
        for attempt in range(3):
            try:
                await asyncio.sleep(0.01)  # let the source settle after clear_queue
                for cursor in range(0, len(cue), spc):
                    chunk = cue[cursor : cursor + spc]
                    if len(chunk) < spc:
                        chunk = np.pad(chunk, (0, spc - len(chunk)))
                    frame = rtc.AudioFrame(
                        data=memoryview(chunk.tobytes()),
                        sample_rate=self._config.tts_publish_sample_rate,
                        num_channels=self._config.num_channels,
                        samples_per_channel=spc,
                    )
                    await self._source.capture_frame(frame)
                self._log("interrupted current agent audio playback (with cue)")
                return
            except Exception as exc:
                if attempt == 2:
                    self._log(f"barge-in cue skipped: {exc}")

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
    _affirmation_tokens = {"да", "ага", "угу", "конечно", "хорошо", "ладно", "поехали", "договорились"}
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
        self._confirm = {"да", "угу", "ага", "подтверждаю", "конечно", "хорошо", "супер", "отлично"}
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
        self._ready_to_talk = {
            "я слушаю",
            "я вас слушаю",
            "слушаю вас",
            "говорите",
            "да слушаю",
            "слушаю",
            "удобно",
            "да удобно",
        }
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
            "что это за вопрос",
            "не понял что это за вопрос",
            "по какому поводу",
            "по какому вопросу",
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
            "один и тот же ответ",
            "один и тот же вопрос",
            "повторяете одно и то же",
            "нельзя один и тот же ответ",
        }
        self._abuse_markers = {
            "тупец",
            "тупая",
            "тупой",
            "идиот",
            "дебил",
            "придурок",
            "отстань",
            "отвали",
            "пошел",
            "пошла",
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

    def route(self, text: str, *, current_node: str = "", stage: str = "") -> IntentResult:
        if not text:
            return IntentResult(Intent.CLARIFY.value, 0.0, False, Action.ASK_REPEAT.value)

        opening_nodes = {
            "opening",
            "check_convenience",
            "small_talk",
            "who_are_you",
            "identity_company_faq",
            "source_of_number_faq",
            "robot_check",
            "memory_denial_faq",
            "callback_reentry",
        }
        is_opening_context = current_node in opening_nodes or stage == "greeting"
        if text in {"але", "алло", "ало"}:
            if is_opening_context:
                return IntentResult(Intent.GREETING.value, 0.9, False, Action.ACK_GREETING.value)
            return IntentResult(Intent.LINE_ISSUE.value, 0.88, False, Action.REPEAT_LAST_AGENT_MESSAGE.value)

        if (
            text in self._greeting
            or text.startswith(("привет", "здравствуйте", "добрый ", "алло", "ало"))
            or any(marker in text for marker in ("добрый день", "добрый вечер", "здравствуйте", " привет"))
            or "привет" in text.split()
        ):
            return IntentResult(Intent.GREETING.value, 0.99, False, Action.ACK_GREETING.value)
        if (
            text in self._ready_to_talk
            or text.startswith(("я слушаю", "я вас слушаю", "слушаю вас", "говорите", "да слушаю"))
            or ("слушаю" in text and any(token in text for token in ("да", "удобно", "говорите")))
        ):
            return IntentResult(Intent.READY_TO_TALK.value, 0.99, False, Action.CONTINUE_OPENING.value)
        if (
            text in self._identify
            or text.startswith(("это кто", "кто это", "кто вы", "представьтесь", "кто со мной"))
            or any(marker in text for marker in ("это кто", "кто звонит", "кто вы", "представьтесь", "по какому поводу", "по какому вопросу"))
        ):
            return IntentResult(Intent.IDENTIFY_SELF.value, 0.99, False, Action.INTRODUCE_SELF.value)
        if any(marker in text for marker in self._identity_mismatch_markers):
            return IntentResult(Intent.IDENTITY_MISMATCH.value, 0.95, False, Action.CLARIFY_IDENTITY.value)
        if text in self._line_issue or "не слышу" in text or "вас не слышно" in text:
            return IntentResult(Intent.LINE_ISSUE.value, 0.98, False, Action.REPEAT_LAST_AGENT_MESSAGE.value)
        if any(marker in text for marker in self._why_need_info_markers):
            return IntentResult(Intent.WHY_NEED_INFO.value, 0.95, False, Action.EXPLAIN_QUESTION.value)
        if any(marker in text for marker in self._latency_markers):
            return IntentResult(Intent.LATENCY_QUESTION.value, 0.94, False, Action.EXPLAIN_DELAY_AND_CONTINUE.value)
        if any(marker in text for marker in self._service_complaint_markers):
            return IntentResult(Intent.SERVICE_COMPLAINT.value, 0.92, False, Action.ACK_COMPLAINT_AND_REFOCUS.value)
        if any(marker in text for marker in self._abuse_markers):
            return IntentResult(Intent.REJECT.value, 0.97, False, Action.ACK_REJECT.value)
        if any(marker in text for marker in self._payment_help_markers):
            return IntentResult(Intent.PAYMENT_HELP.value, 0.92, False, Action.HANDOFF_PAYMENT_SUPPORT.value)
        if text.startswith(("да да", "угу да", "ага да")) or "я понял" in text or "понял вас" in text:
            return IntentResult(Intent.CONFIRM.value, 0.96, False, Action.ACK_CONFIRM.value)
        if text in self._confirm or self._is_affirmation_phrase(text):
            return IntentResult(Intent.CONFIRM.value, 0.99, False, Action.ACK_CONFIRM.value)
        if text in self._reject or self._is_rejection_phrase(text):
            return IntentResult(Intent.REJECT.value, 0.99, False, Action.ACK_REJECT.value)
        if text in self._cancel or "отмена" in text:
            return IntentResult(Intent.CANCEL.value, 0.99, False, Action.CANCEL_ACTION.value)
        if text in self._repeat or text.startswith("повтор") or "повтор" in text:
            return IntentResult(Intent.REPEAT.value, 0.99, False, Action.REPEAT_LAST_AGENT_MESSAGE.value)
        if text in self._wait or text.startswith("подожди"):
            return IntentResult(Intent.WAIT.value, 0.98, False, Action.ACK_WAIT.value)
        if any(marker in text for marker in ("спасибо за внимание", "всего доброго", "до свидания", "до свиданья")):
            return IntentResult(Intent.END_SESSION.value, 0.98, False, Action.END_SESSION.value)
        if text in self._end_session or text.startswith("заверши") or text.startswith("стоп"):
            return IntentResult(Intent.END_SESSION.value, 0.98, False, Action.END_SESSION.value)
        if any(token in text for token in self._handoff_tokens):
            return IntentResult(Intent.HUMAN_HANDOFF.value, 0.99, False, Action.HANDOFF_TO_HUMAN.value)
        if self._looks_like_amount(text):
            return IntentResult(Intent.AMOUNT_PROVIDED.value, 0.94, False, Action.ACK_AMOUNT_AND_CONTINUE.value)
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
            repeated = (last_agent_message or "").strip()
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
        llm_service: OpenAiLlmService | TextApiLlmService,
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
        self._tts_markup = TtsMarkupService(provider=config.tts_provider)
        self._sales_speech_styler = SalesSpeechStyler(config)
        self._voice_style = VoiceStyleAdapter(config)
        self._dialogue_state = DialogueState()
        try:
            self._kb = KnowledgeBase.load(config.data_dir)
            self._log(f"loaded agent knowledge base from {config.data_dir}")
        except Exception as exc:
            self._log(f"failed to load knowledge base from {config.data_dir}: {exc}")
            self._kb = KnowledgeBase.default()
        try:
            graph_path = Path(
                os.getenv(
                    "TOOL_GRAPH_PATH",
                    str(config.data_dir / "tool_graph.json"),
                )
            )
            self._tool_graph = ToolGraphRuntime.load(
                graph_path=graph_path,
                agent_name="Влад+имир",
            )
            self._dialogue_state.current_node = self._tool_graph.start_node
            self._log("loaded tool graph runtime")
        except Exception as exc:
            self._log(f"failed to load tool graph runtime: {exc}")
            self._tool_graph = None

        self._session_id = f"{participant.identity}-{int(time.time())}-{uuid.uuid4().hex[:8]}"
        self._session_logger = SessionLogger(config.session_log_dir / f"{self._session_id}.jsonl")
        self._text_api_mode = bool(getattr(llm_service, "uses_text_api_backend", False))
        self._text_api_started = False
        self._text_api_phone = participant.identity
        self._text_api_current_node = "call_connected"
        self._text_api_last_turn_note = ""
        self._text_api_known_facts: dict[str, Any] = {}

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
        self._last_semantic_agent_message: str | None = None
        self._session_memory = SessionMemory(max_turns=12)
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
        self._playback_state = PlaybackState()
        self._spoken_turn_count = 0
        self._greeting_was_spoken = False
        self._last_question_text = ""

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

    def _uses_text_api_backend(self) -> bool:
        return self._text_api_mode

    @staticmethod
    def _jsonable_known_facts(values: dict[str, Any]) -> dict[str, Any]:
        clean: dict[str, Any] = {}
        for key, value in values.items():
            if value is None:
                continue
            if isinstance(value, (str, int, float, bool)):
                clean[key] = value
                continue
            if isinstance(value, list) and all(isinstance(item, (str, int, float, bool)) for item in value):
                clean[key] = value
        return clean

    def _text_api_start_payload(self) -> dict[str, Any]:
        snapshot = self._dialogue_state.snapshot()
        known_facts = self._jsonable_known_facts(snapshot.get("known_facts", {}))
        if self._text_api_known_facts:
            known_facts.update(self._jsonable_known_facts(self._text_api_known_facts))
        return known_facts

    def _sync_text_api_shadow_state(
        self,
        *,
        current_node: str,
        known_facts: dict[str, Any],
        last_turn_note: str,
        reply_text: str,
    ) -> None:
        self._text_api_current_node = current_node or self._text_api_current_node
        self._text_api_last_turn_note = last_turn_note
        self._text_api_known_facts = dict(known_facts or {})
        if isinstance(getattr(self._dialogue_state, "known_facts", None), dict):
            self._dialogue_state.known_facts.update(self._text_api_known_facts)
        if hasattr(self._dialogue_state, "current_node"):
            self._dialogue_state.current_node = self._text_api_current_node
        if hasattr(self._dialogue_state, "last_agent_text"):
            self._dialogue_state.last_agent_text = reply_text
        client_name = str(self._text_api_known_facts.get("client_name", "")).strip()
        if client_name and hasattr(self._dialogue_state, "name"):
            self._dialogue_state.name = client_name

    async def _ensure_text_api_session_started(self) -> None:
        if not self._uses_text_api_backend() or self._text_api_started:
            return
        start_payload = self._text_api_start_payload()
        response = await self._llm_service.start_session(
            session_id=self._session_id,
            phone=self._text_api_phone,
            known_facts=start_payload,
        )
        self._text_api_started = True
        self._sync_text_api_shadow_state(
            current_node=str(response.get("current_node", "call_connected")).strip() or "call_connected",
            known_facts=response.get("known_facts", {}) or {},
            last_turn_note=str(response.get("last_turn_note", "")).strip(),
            reply_text=str(response.get("reply", "")).strip(),
        )

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
        if self._uses_text_api_backend():
            with contextlib.suppress(Exception):
                await self._llm_service.reset_session(session_id=self._session_id)
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
                else self._config.processing_resume_min_speech_duration_ms
            )
            if speech_ms < min_required_ms:
                continue

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
                await self._audio_publisher.interrupt_with_cue()
                self._mark_playback_interrupted()
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

    def _recent_history(self, *, limit: int | None = None) -> list[dict[str, str]]:
        return self._session_memory.recent_history(limit=limit)

    def _llm_state_payload(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        self._session_memory.sync_from_dialogue_state(snapshot, last_question=self._last_question_text)
        return self._session_memory.llm_state_payload()

    def _remember_agent_question(self, text: str) -> None:
        value = text.strip()
        if not value:
            return
        if "?" in value or value.lower().startswith(("подскажите", "скажите", "какая", "какой", "кто", "в каком")):
            self._last_question_text = value
            self._session_memory.remember_question(value)

    def apply_lead_profile(self, profile: dict[str, Any]) -> None:
        updated = self._dialogue_state.bootstrap_lead_profile(profile, kb=self._kb)
        snapshot = self._dialogue_state.snapshot()
        for key in ("phone", "phone_number", "client_phone", "tel"):
            value = str(profile.get(key, "")).strip()
            if value:
                self._text_api_phone = value
                break
        self._text_api_known_facts.update(
            self._jsonable_known_facts(snapshot.get("known_facts", {}))
        )
        self._session_memory.sync_from_dialogue_state(snapshot, last_question=self._last_question_text)
        self._log(
            f"lead profile applied participant={self._participant.identity} "
            f"updated_fields={sorted(updated)!r} known_facts={snapshot.get('known_facts', {})!r}"
        )

    def _has_prefilled_lead(self) -> bool:
        return str(self._dialogue_state.known_facts.get("prefilled_lead", "")).strip() == "yes"

    def _lead_speed_emphasis(self) -> bool:
        return str(self._dialogue_state.known_facts.get("speed_emphasis", "")).strip() == "yes"

    def _build_context_sentence_for_lead(self) -> str:
        facts = self._dialogue_state.known_facts
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

    def _contextual_opening_reply(self) -> tuple[str, str] | None:
        if not self._has_prefilled_lead():
            return None
        facts = self._dialogue_state.known_facts
        name = str(facts.get("client_name", "")).strip()
        greeting = f"Да, добрый день, {name}." if name else "Да, добрый день."
        context_sentence = self._build_context_sentence_for_lead()
        speed_sentence = (
            "Если для вас важна скорость, я уточню только главное и быстро передам кейс эксперту."
            if self._lead_speed_emphasis()
            else ""
        )
        text = " ".join(
            part
            for part in (
                "Алл+о.",
                greeting,
                "Это Влад+имир, МосИнвестФинанс.",
                context_sentence,
                speed_sentence,
                "Удобно сейчас коротко продолжить?",
            )
            if part
        )
        self._remember_agent_question("Удобно сейчас коротко продолжить?")
        return "callback_reentry", text

    def _contextual_identity_reply(self) -> str:
        context_sentence = self._build_context_sentence_for_lead()
        speed_sentence = (
            "Если для вас важна скорость, я уточню только главное и быстро передам кейс эксперту."
            if self._lead_speed_emphasis()
            else ""
        )
        response = " ".join(
            part
            for part in (
                "Это Влад+имир, МосИнвестФинанс.",
                context_sentence,
                speed_sentence,
                "Вам ещё актуален этот вопрос?",
            )
            if part
        )
        self._remember_agent_question("Вам ещё актуален этот вопрос?")
        return response

    def _state_fallback_reply(self) -> str:
        snapshot = self._dialogue_state.snapshot()
        if self._tool_graph is not None:
            graph_question = self._tool_graph.question_for_state(snapshot)
            if graph_question is not None:
                node_name, question = graph_question
                self._dialogue_state.current_node = node_name
                self._remember_agent_question(question)
                return question
        next_field = self._kb.next_required_field(snapshot) or self._dialogue_state.next_required_field
        if next_field:
            question = self._kb.question_for_field(next_field)
            self._remember_agent_question(question)
            return question
        return self._config.fallback_complex_text

    def _state_guided_reply(self, intent: IntentResult) -> str:
        next_question = self._state_fallback_reply()
        known_facts = self._dialogue_state.known_facts
        amount = self._dialogue_state.amount_text.strip()
        goal = self._dialogue_state.goal.strip()
        object_type = self._dialogue_state.object_type.strip()
        region = self._dialogue_state.city.strip()
        speed_emphasis = self._lead_speed_emphasis()

        if str(known_facts.get("amount_needs_clarification", "")).strip() == "yes":
            return next_question

        if intent.intent == Intent.AMOUNT_PROVIDED.value and amount:
            if speed_emphasis:
                return f"Понял. Чтобы не тянуть, уточню только главное и быстро передам кейс эксперту. {next_question}"
            if not goal:
                return f"Хмм, понял. Сумму {amount} вижу. С такой суммой, скорее всего, работаем. {next_question}"
            if not object_type:
                return (
                    f"Хмм, понял. Сумму {amount} зафиксировал. "
                    f"Под такую цель, скорее всего, посмотрим варианты. {next_question}"
                )

        if intent.intent in {Intent.SLOT_ANSWER.value, Intent.COMPLEX_REQUEST.value}:
            if speed_emphasis and next_question:
                if object_type and not region:
                    return f"Понял. Чтобы быстрее передать кейс, задам только ключевые вопросы. {next_question}"
                if region and not self._dialogue_state.collateral:
                    return f"Хорошо. Иду коротко и по делу, чтобы не тянуть. {next_question}"
            if goal and not object_type:
                return f"Понял, цель {goal}. {next_question}"
            if object_type and not region:
                return f"Понял, {object_type}. {next_question}"
            if region and not self._dialogue_state.collateral:
                return f"Понял, объект в {region}. {next_question}"
            if self._dialogue_state.collateral and not str(known_facts.get('owners', '') or known_facts.get('owner', '')).strip():
                collateral = self._dialogue_state.collateral.strip()
                return f"Понял, по обременению отметил: {collateral}. {next_question}"

        return next_question

    def _repair_and_resume_reply(self, intent: IntentResult) -> str:
        next_question = self._state_fallback_reply()
        if intent.intent == Intent.WHY_NEED_INFO.value:
            return f"Да, поясню. Это нужно, чтобы сразу понять, подойдём ли мы вам по условиям. {next_question}"
        if intent.intent == Intent.SERVICE_COMPLAINT.value:
            return f"Да, понимаю. Извините, если прозвучало неудачно. Давайте коротко и по делу. {next_question}"
        return next_question

    def _detect_speech_situation(
        self,
        *,
        intent: IntentResult,
        updated_fields: set[str],
        repair_reason: str = "",
    ) -> str:
        if repair_reason:
            return SpeechSituation.REPAIR.value
        if intent.intent in {Intent.GREETING.value, Intent.READY_TO_TALK.value, Intent.IDENTIFY_SELF.value}:
            return SpeechSituation.OPENING.value
        if intent.intent in {Intent.REJECT.value, Intent.SERVICE_COMPLAINT.value, Intent.WHY_NEED_INFO.value}:
            return SpeechSituation.OBJECTION.value
        if updated_fields:
            return SpeechSituation.SLOT_BRIDGE.value
        if intent.action == Action.HANDOFF_TO_HUMAN.value:
            return SpeechSituation.HANDOFF.value
        if intent.action == Action.END_SESSION.value:
            return SpeechSituation.CLOSING.value
        if intent.use_llm or intent.action == Action.CALL_LLM.value:
            return SpeechSituation.THINKING.value
        if intent.intent in {Intent.CLARIFY.value, Intent.UNKNOWN_SHORT.value, Intent.LINE_ISSUE.value}:
            return SpeechSituation.CLARIFICATION.value
        return SpeechSituation.DEFAULT.value

    def _validate_and_log_llm_reply(
        self,
        *,
        reply_tts: str,
        fallback_reply: str,
        state: dict[str, Any],
        knowledge: list[Any],
        truth_rules: tuple[str, ...],
        utterance_id: str,
        mode: str,
        reason_hint: str = "",
    ) -> str:
        validated_text, validation_reason = inspect_llm_reply(
            reply_tts=reply_tts,
            fallback_reply=fallback_reply,
            state=state,
            knowledge=knowledge,
            truth_rules=truth_rules,
        )
        was_replaced = normalize_for_compare(validated_text) != normalize_for_compare(reply_tts)
        self._log(
            f"llm validation participant={self._participant.identity} "
            f"utterance_id={utterance_id} "
            f"mode={mode} "
            f"reason_hint={reason_hint!r} "
            f"validation_reason={validation_reason!r} "
            f"was_replaced={str(was_replaced).lower()} "
            f"raw_reply={reply_tts!r} "
            f"validated_reply={validated_text!r} "
            f"fallback_reply={fallback_reply!r}"
        )
        return validated_text

    def _start_playback_state(
        self,
        *,
        utterance_id: str,
        response_text: str,
        prepared_text: str,
        segments: list[str],
    ) -> None:
        self._playback_state.utterance_id = utterance_id
        self._playback_state.response_text = response_text
        self._playback_state.prepared_text = prepared_text
        self._playback_state.segments = list(segments)
        self._playback_state.next_segment_index = 0
        self._playback_state.interrupted = False
        self._playback_state.last_interrupted_at_ms = 0
        self._playback_state.completed = False

    def _mark_playback_segment_completed(self, segment_index: int) -> None:
        if segment_index + 1 > self._playback_state.next_segment_index:
            self._playback_state.next_segment_index = segment_index + 1

    def _mark_playback_interrupted(self) -> None:
        if not self._playback_state.segments:
            return
        self._playback_state.interrupted = True
        self._playback_state.completed = False
        self._playback_state.last_interrupted_at_ms = int(time.time() * 1000)

    def _complete_playback_state(self) -> None:
        self._playback_state.completed = True
        self._playback_state.interrupted = False
        self._playback_state.next_segment_index = len(self._playback_state.segments)

    def _should_resume_previous_playback(self, raw_text: str, normalized_text: str) -> bool:
        if not self._needs_rescue_prompt:
            return False
        if not self._playback_state.interrupted or not self._playback_state.has_remaining():
            return False
        if not self._playback_state.last_interrupted_at_ms:
            return False
        if int(time.time() * 1000) - self._playback_state.last_interrupted_at_ms > 2500:
            return False
        lowered = normalize_for_compare(normalized_text)
        if is_low_information_transcript(raw_text, normalized_text):
            return True
        return lowered in {"алло", "ало", "але", "слышно", "меня слышно", "вы тут"}

    def _build_resume_playback_text(self, normalized_text: str) -> str:
        remaining_segments = self._playback_state.segments[self._playback_state.next_segment_index :]
        if not remaining_segments:
            return ""
        lowered = normalize_for_compare(normalized_text)
        if "слыш" in lowered:
            bridge = "Да, слышно."
        elif lowered in {"алло", "ало", "але"}:
            bridge = "Да, алло."
        elif self._playback_state.next_segment_index == 0:
            bridge = "Да, смотрите."
        else:
            bridge = "Так вот."
        return f"{bridge} {' '.join(remaining_segments)}".strip()

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
        tts_speed: float = 1.0,
    ) -> tuple[str, str, list[str], int, bool, bool, str]:
        style_result, prepared_text, segments = self._prepare_tts_output(
            response_text,
            intent_value=intent_value,
        )
        if not segments:
            segments = [prepared_text] if prepared_text else [style_result.styled_text]
        self._start_playback_state(
            utterance_id=utterance_id,
            response_text=style_result.styled_text,
            prepared_text=prepared_text,
            segments=segments,
        )
        self._log(
            f"tts plan participant={self._participant.identity} "
            f"utterance_id={utterance_id} speed={tts_speed:.2f} "
            f"prepared_text={prepared_text!r} segments={segments!r}"
        )
        request = TtsRequest(text=style_result.styled_text, speaker=self._config.tts_speaker, speed=tts_speed)
        if len(segments) == 1:
            tts_pcm16, tts_sample_rate, tts_latency_ms = await asyncio.to_thread(
                self._tts_service.synthesize_segments,
                request,
                segments,
            )
            playback_completed = await self._audio_publisher.speak_pcm(tts_pcm16, tts_sample_rate)
            if playback_completed:
                self._mark_playback_segment_completed(0)
        else:
            first_segment = segments[0]
            remaining_segments = segments[1:]
            first_pcm16, tts_sample_rate, first_latency_ms = await asyncio.to_thread(
                self._tts_service.synthesize_segment,
                request,
                first_segment,
                trailing_pause_ms=pause_after_segment_ms(first_segment, self._config.tts_segment_pause_ms)
                if remaining_segments
                else 0,
            )
            rest_task: asyncio.Task[tuple[np.ndarray, int, int]] | None = None
            if remaining_segments:
                rest_task = asyncio.create_task(
                    asyncio.to_thread(
                        self._tts_service.synthesize_segments,
                        request,
                        remaining_segments,
                    )
                )
            first_completed = await self._audio_publisher.speak_pcm(first_pcm16, tts_sample_rate)
            playback_completed = first_completed
            tts_latency_ms = first_latency_ms
            if first_completed:
                self._mark_playback_segment_completed(0)
            if first_completed and rest_task is not None:
                try:
                    rest_pcm16, rest_sample_rate, _rest_latency_ms = await rest_task
                    if len(rest_pcm16) > 0:
                        playback_completed = await self._audio_publisher.speak_pcm(rest_pcm16, rest_sample_rate)
                        if playback_completed:
                            self._mark_playback_segment_completed(len(segments) - 1)
                except Exception as exc:
                    self._log(
                        f"tts tail synthesis failed participant={self._participant.identity} "
                        f"utterance_id={utterance_id} error={str(exc) or repr(exc)}"
                    )
                    playback_completed = first_completed
            elif rest_task is not None:
                rest_task.cancel()
        if playback_completed:
            self._complete_playback_state()
        else:
            self._mark_playback_interrupted()
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
        if intent.use_llm or intent.action == Action.CALL_LLM.value:
            return False
        slot_fields = {
            "нужная_сумма",
            "цель",
            "вид_объекта",
            "регион",
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
        }

    def _should_try_stalled_slot_rescue(
        self,
        *,
        normalized_text: str,
        intent: IntentResult,
        updated_fields: set[str],
        state_snapshot: dict[str, Any],
    ) -> bool:
        if updated_fields or not normalized_text.strip():
            return False
        if intent.intent in {
            Intent.GREETING.value,
            Intent.READY_TO_TALK.value,
            Intent.IDENTIFY_SELF.value,
            Intent.LINE_ISSUE.value,
            Intent.WAIT.value,
            Intent.REPEAT.value,
            Intent.END_SESSION.value,
            Intent.REJECT.value,
            Intent.CANCEL.value,
            Intent.HUMAN_HANDOFF.value,
        }:
            return False
        next_field = str(state_snapshot.get("next_required_field", "")).strip()
        current_node = str(state_snapshot.get("current_node", "")).strip()
        if not next_field and not current_node.startswith("collect_"):
            return False
        if is_low_information_transcript(normalized_text, normalized_text):
            return False
        return True

    async def _generate_adaptive_repair_reply(
    self,
    *,
    participant_identity: str,
    normalized_text: str,
    raw_text: str,
    reason: str,
    candidate_response: str,
    utterance_id: str,
) -> tuple[str, LlmReply | None, int]:
        state = self._dialogue_state.snapshot()
        llm_state = self._llm_state_payload(state)
        graph_context = self._tool_graph.llm_context_for_text(normalized_text, state) if self._tool_graph else {}
        resume_question = ""
        resume_node = ""

        if self._tool_graph:
            resume = self._tool_graph.question_for_state(state)
            if resume:
                resume_node, resume_question = resume

        repair_context = {
            "mode": "adaptive_repair",
            "reason": reason,
            "client_text": raw_text,
            "normalized_text": normalized_text,
            "candidate_bad_response": candidate_response,
            "last_agent_text": state.get("last_agent_text", ""),
            "awaiting_field": state.get("awaiting_field", ""),
            "next_required_field": state.get("next_required_field", ""),
            "current_node": state.get("current_node", ""),
            "resume_node": resume_node,
            "resume_question": resume_question,
            "known_facts": state.get("known_facts", {}),
            "rules": [
                "Если клиент уже ответил на вопрос, не повторяй этот вопрос.",
                "Если клиент раздражён или не понял вопрос, сначала коротко признай это.",
                "Если candidate_bad_response повторяет прошлый вопрос, запрещено его использовать.",
                "Если клиент сказал, что вопрос не актуален, не спрашивай сумму.",
                "Если клиент дал слот, подтверди его и перейди к следующему слоту.",
                "Ответ — максимум 1–2 коротких предложения.",
            ],
        }

        try:
            safe_repair_fallback = "Понял вас. Давайте уточню по-другому."
            llm_reply, llm_latency_ms = await self._llm_service.generate_response(
                normalized_text=normalized_text,
                history=self._recent_history(limit=4),
                dialogue_state=llm_state,
                knowledge=[],
                truth_rules=self._kb.truth_rules if self._kb else (),
                examples=[],
                graph_context={**graph_context, **repair_context},
                temperature_override=0.0,
                max_tokens_override=min(96, self._config.llm_max_tokens),
            )
            response_text = self._validate_and_log_llm_reply(
                reply_tts=llm_reply.reply_tts,
                fallback_reply=safe_repair_fallback,
                state=state,
                knowledge=[],
                truth_rules=self._kb.truth_rules if self._kb else (),
                utterance_id=utterance_id,
                mode="adaptive_repair",
                reason_hint=reason,
            )
            return response_text, llm_reply, llm_latency_ms

        except Exception as exc:
            self._log(f"adaptive repair failed utterance={utterance_id}: {exc}")

            return "Понял вас. Давайте уточню по-другому.", None, 0
    
    async def _generate_stalled_slot_rescue(
        self,
        *,
        normalized_text: str,
        history: list[dict[str, str]],
        state_snapshot: dict[str, Any],
        utterance_id: str,
    ) -> tuple[str, LlmReply | None, int]:
        resume_question = self._state_fallback_reply()
        if not self._llm_service.enabled:
            return self._repair_and_resume_reply(IntentResult(Intent.SERVICE_COMPLAINT.value, 0.0, False, "")), None, 0
        try:
            llm_state = self._llm_state_payload(state_snapshot)
            graph_context = (
                self._tool_graph.llm_context_for_text(normalized_text, state_snapshot)
                if self._tool_graph is not None
                else {}
            )
            graph_context = dict(graph_context or {})
            graph_context["node_name"] = graph_context.get("node_name") or "repair_stalled_slot"
            graph_context["goal"] = "Если клиент уже, вероятно, ответил по текущему шагу, мягко признай возможную ошибку, коротко переформулируй и продолжи."
            graph_context["resume_question"] = resume_question
            graph_context["next_required_field"] = str(state_snapshot.get("next_required_field", "")).strip()
            graph_context["stalled_slot_recovery"] = True
            graph_context["rules"] = list(graph_context.get("rules", [])) + [
                "Не начинай сценарий заново.",
                "Если клиент уже дал ответ по текущему шагу, кратко это признай.",
                "Если ответ недостаточно точный, переформулируй только текущий вопрос.",
                "Не перепрыгивай на другой слот без причины.",
            ]
            llm_reply, llm_latency_ms = await self._llm_service.generate_response(
                normalized_text=normalized_text,
                history=history[-4:],
                dialogue_state=llm_state,
                knowledge=[],
                truth_rules=self._kb.truth_rules,
                examples=[],
                graph_context=graph_context,
                temperature_override=0.0,
                max_tokens_override=min(96, self._config.llm_max_tokens),
            )
            response_text = self._validate_and_log_llm_reply(
                reply_tts=llm_reply.reply_tts,
                fallback_reply=f"Да, возможно, я неточно понял. {resume_question}",
                state=state_snapshot,
                knowledge=[],
                truth_rules=self._kb.truth_rules,
                utterance_id=utterance_id,
                mode="stalled_slot_rescue",
                reason_hint=str(state_snapshot.get("next_required_field", "")),
            )
            self._log(
                f"stalled-slot rescue participant={self._participant.identity} "
                f"utterance_id={utterance_id} next_required_field={state_snapshot.get('next_required_field', '')!r}"
            )
            return response_text, llm_reply, llm_latency_ms
        except Exception as exc:
            self._log(f"stalled-slot rescue fallback for {self._participant.identity}: {exc}")
            return f"Да, возможно, я неточно понял. {resume_question}", None, 0

    async def _run_text_api_turn(
        self,
        *,
        utterance_id: str,
        transcript_text: str,
        normalized_text: str,
        speech_end_time_ms: int,
        turn_revision: int,
    ) -> dict[str, Any]:
        if not transcript_text.strip():
            intent = IntentResult(Intent.CLARIFY.value, 0.0, False, Action.ASK_REPEAT.value)
            return {
                "intent": intent,
                "llm_reply": None,
                "response_text": "",
                "raw_response_text": "",
                "response_published": False,
                "suppress_response": True,
                "router_latency_ms": 0,
                "intent_ready_time_ms": 0,
                "response_ready_time_ms": 0,
                "llm_latency_ms": 0,
                "tts_latency_ms": 0,
                "tts_prepared_text": "",
                "tts_segments": [],
                "filler_added": False,
                "filler_type": "",
                "tts_synth_start_time_ms": 0,
                "tts_synth_done_time_ms": 0,
                "tts_publish_start_time_ms": 0,
                "tts_publish_done_time_ms": 0,
                "current_node": self._text_api_current_node,
                "known_facts": dict(self._text_api_known_facts),
                "last_turn_note": self._text_api_last_turn_note,
                "trace": {},
                "speech_end_time_ms": speech_end_time_ms,
            }

        if transcript_text.strip():
            self._session_memory.add_user(normalized_text or transcript_text)

        await self._publish_status("complex_request_detected")
        await self._ensure_text_api_session_started()

        intent_started_at = time.perf_counter()
        intent = IntentResult("text_api", 1.0, True, Action.CALL_LLM.value)
        router_latency_ms = int((time.perf_counter() - intent_started_at) * 1000)
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
                "ts_ms": intent_ready_time_ms,
            },
            destination_identities=[self._participant.identity],
        )

        payload, llm_latency_ms = await self._llm_service.message(
            session_id=self._session_id,
            text=transcript_text,
        )
        response_ready_time_ms = int(time.time() * 1000)

        reply_text = str(payload.get("reply", "")).strip()
        if not reply_text:
            raise RuntimeError("text_api returned empty reply")

        current_node = str(payload.get("current_node", "")).strip() or self._text_api_current_node
        last_turn_note = str(payload.get("last_turn_note", "")).strip()
        known_facts = payload.get("known_facts", {}) if isinstance(payload.get("known_facts"), dict) else {}
        llm_decision = payload.get("llm_decision", {}) if isinstance(payload.get("llm_decision"), dict) else {}
        trace = payload.get("trace", {}) if isinstance(payload.get("trace"), dict) else {}
        self._log(
            "text_api turn "
            f"participant={self._participant.identity} "
            f"session_id={self._session_id} "
            f"utterance_id={utterance_id} "
            f"input_text={transcript_text!r} "
            f"current_node={current_node!r} "
            f"trace_source={str(trace.get('source', ''))!r} "
            f"reply={reply_text!r}"
        )

        search_values: list[str] = []
        if isinstance(llm_decision.get("search_index"), list):
            search_values.extend(
                item for item in llm_decision.get("search_index", []) if isinstance(item, str)
            )
        if current_node:
            search_values.append(current_node)
        search_index = dedupe_compact_strings(search_values, limit=5) or ["text_api"]

        llm_reply = LlmReply(
            reply_tts=reply_text,
            search_index=search_index,
            intent=current_node or "text_api",
            next_step=last_turn_note or current_node or "text_api_turn",
        )

        self._sync_text_api_shadow_state(
            current_node=current_node,
            known_facts=known_facts,
            last_turn_note=last_turn_note,
            reply_text=reply_text,
        )
        self._last_semantic_agent_message = reply_text
        self._last_agent_message = reply_text
        self._remember_agent_question(reply_text)
        self._session_memory.add_assistant(reply_text)
        self._needs_rescue_prompt = False

        await self._event_bus.publish_json(
            {
                "type": "agent_response_text",
                "utterance_id": utterance_id,
                "participant_identity": self._participant.identity,
                "text": reply_text,
                "use_llm": True,
                "llm_intent": llm_reply.intent,
                "search_index": llm_reply.search_index,
                "next_step": llm_reply.next_step,
                "text_api_trace": trace,
            },
            destination_identities=[self._participant.identity],
        )

        tts_prepared_text = ""
        tts_segments: list[str] = []
        filler_added = False
        filler_type = ""
        tts_latency_ms = 0
        tts_synth_start_time_ms = 0
        tts_synth_done_time_ms = 0
        tts_publish_start_time_ms = 0
        tts_publish_done_time_ms = 0
        spoken_text = reply_text

        if self._config.tts_enabled:
            if self._is_stale_turn(turn_revision):
                return {
                    "intent": intent,
                    "llm_reply": llm_reply,
                    "response_text": reply_text,
                    "raw_response_text": reply_text,
                    "response_published": True,
                    "suppress_response": True,
                    "router_latency_ms": router_latency_ms,
                    "intent_ready_time_ms": intent_ready_time_ms,
                    "response_ready_time_ms": response_ready_time_ms,
                    "llm_latency_ms": llm_latency_ms,
                    "tts_latency_ms": 0,
                    "tts_prepared_text": "",
                    "tts_segments": [],
                    "filler_added": False,
                    "filler_type": "",
                    "tts_synth_start_time_ms": 0,
                    "tts_synth_done_time_ms": 0,
                    "tts_publish_start_time_ms": 0,
                    "tts_publish_done_time_ms": 0,
                    "current_node": current_node,
                    "known_facts": known_facts,
                    "last_turn_note": last_turn_note,
                    "trace": trace,
                    "speech_end_time_ms": speech_end_time_ms,
                }
            await self._publish_status("speaking")
            self._is_speaking = True
            tts_synth_start_time_ms = int(time.time() * 1000)
            (
                spoken_text,
                tts_prepared_text,
                tts_segments,
                tts_latency_ms,
                _playback_completed,
                filler_added,
                filler_type,
            ) = await self._speak_response(
                utterance_id=utterance_id,
                response_text=reply_text,
                intent_value="text_api",
                tts_speed=1.0,
            )
            tts_synth_done_time_ms = int(time.time() * 1000)
            tts_publish_start_time_ms = tts_synth_done_time_ms
            tts_publish_done_time_ms = int(time.time() * 1000)

        return {
            "intent": intent,
            "llm_reply": llm_reply,
            "response_text": spoken_text,
            "raw_response_text": reply_text,
            "response_published": True,
            "suppress_response": False,
            "router_latency_ms": router_latency_ms,
            "intent_ready_time_ms": intent_ready_time_ms,
            "response_ready_time_ms": response_ready_time_ms,
            "llm_latency_ms": llm_latency_ms,
            "tts_latency_ms": tts_latency_ms,
            "tts_prepared_text": tts_prepared_text,
            "tts_segments": tts_segments,
            "filler_added": filler_added,
            "filler_type": filler_type,
            "tts_synth_start_time_ms": tts_synth_start_time_ms,
            "tts_synth_done_time_ms": tts_synth_done_time_ms,
            "tts_publish_start_time_ms": tts_publish_start_time_ms,
            "tts_publish_done_time_ms": tts_publish_done_time_ms,
            "current_node": current_node,
            "known_facts": known_facts,
            "last_turn_note": last_turn_note,
            "trace": trace,
            "speech_end_time_ms": speech_end_time_ms,
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
        updated_fields: set[str] = set()
        classifier_latency_ms = 0
        router_latency_ms = 0
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
        next_graph_node = self._dialogue_state.current_node

        try:
            error_stage = "stt"
            transcript = await asyncio.to_thread(self._stt_service.transcribe, audio_samples, duration_ms)
            stt_done_time_ms = int(time.time() * 1000)
            normalized_text = self._normalizer.normalize(transcript.text)
            self._log(
                f"turn transcript participant={self._participant.identity} "
                f"utterance_id={utterance_id} raw_stt_text={transcript.text!r} "
                f"normalized_text={normalized_text!r} confidence={transcript.confidence:.3f}"
            )
            await self._event_bus.publish_json(
                {
                    "type": "transcript",
                    "utterance_id": utterance_id,
                    "participant_identity": self._participant.identity,
                    "text": transcript.text,
                    "normalized_text": normalized_text,
                    "final": True,
                    "confidence": transcript.confidence,
                    "duration_ms": duration_ms,
                    "ts_ms": stt_done_time_ms,
                },
                destination_identities=[self._participant.identity],
            )

            if self._uses_text_api_backend():
                error_stage = "text_api"
                text_api_result = await self._run_text_api_turn(
                    utterance_id=utterance_id,
                    transcript_text=transcript.text,
                    normalized_text=normalized_text,
                    speech_end_time_ms=speech_end_time_ms,
                    turn_revision=turn_revision,
                )
                intent = text_api_result["intent"]
                llm_reply = text_api_result["llm_reply"]
                response_text = text_api_result["response_text"]
                raw_response_text = text_api_result["raw_response_text"]
                response_published = bool(text_api_result["response_published"])
                suppress_response = bool(text_api_result.get("suppress_response", False))
                router_latency_ms = int(text_api_result["router_latency_ms"])
                intent_ready_time_ms = int(text_api_result["intent_ready_time_ms"])
                response_ready_time_ms = int(text_api_result["response_ready_time_ms"])
                llm_latency_ms = int(text_api_result["llm_latency_ms"])
                tts_latency_ms = int(text_api_result["tts_latency_ms"])
                tts_prepared_text = str(text_api_result["tts_prepared_text"])
                tts_segments = list(text_api_result["tts_segments"])
                filler_added = bool(text_api_result["filler_added"])
                filler_type = str(text_api_result["filler_type"])
                tts_synth_start_time_ms = int(text_api_result["tts_synth_start_time_ms"])
                tts_synth_done_time_ms = int(text_api_result["tts_synth_done_time_ms"])
                tts_publish_start_time_ms = int(text_api_result["tts_publish_start_time_ms"])
                tts_publish_done_time_ms = int(text_api_result["tts_publish_done_time_ms"])
                next_graph_node = str(text_api_result["current_node"])
                return

            error_stage = "routing"
            updated_fields = self._dialogue_state.update_from_user(
                transcript.text,
                normalized_text,
                kb=self._kb,
            )

            # 3. Принудительно пытаемся поймать ожидаемый слот
            forced_fields = self._dialogue_state.force_capture_expected_slot(
                transcript.text,
                normalized_text,
            )

            if forced_fields:
                updated_fields |= forced_fields
                self._log(f"force captured expected slot: {sorted(forced_fields)}")

            state_snapshot = self._dialogue_state.snapshot()
            self._log(
                f"turn state participant={self._participant.identity} "
                f"utterance_id={utterance_id} updated_fields={sorted(updated_fields)!r} "
                f"scenario={state_snapshot.get('scenario', '')!r} "
                f"awaiting_field={state_snapshot.get('awaiting_field', '')!r} "
                f"next_required_field={state_snapshot.get('next_required_field', '')!r} "
                f"known_facts={state_snapshot.get('known_facts', {})!r}"
            )

            if transcript.text.strip():
                self._session_memory.add_user(normalized_text or transcript.text)
            self._session_memory.sync_from_dialogue_state(state_snapshot, last_question=self._last_question_text)

            await self._publish_status("routing_intent")
            router_started_at = time.perf_counter()
            intent = self._router.route(
                normalized_text,
                current_node=str(state_snapshot.get("current_node", "")).strip(),
                stage=str(state_snapshot.get("stage", "")).strip(),
            )
            router_latency_ms = int((time.perf_counter() - router_started_at) * 1000)
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
                    "ts_ms": intent_ready_time_ms,
                },
                destination_identities=[self._participant.identity],
            )
            if intent.intent in {
                Intent.SERVICE_COMPLAINT.value,
                Intent.WHY_NEED_INFO.value,
                Intent.REJECT.value,
                Intent.HUMAN_HANDOFF.value,
            }:
                self._session_memory.add_objection(normalized_text or transcript.text)

            candidate_response = ""
            candidate_node = self._dialogue_state.current_node
            playback_resume_selected = False
            graph_question = (
                self._tool_graph.question_for_state(state_snapshot)
                if self._tool_graph is not None
                else None
            )

            opening_intents = {
                Intent.GREETING.value,
                Intent.READY_TO_TALK.value,
                Intent.IDENTIFY_SELF.value,
                Intent.IDENTITY_MISMATCH.value,
            }
            canned_intents = opening_intents | {
                Intent.LINE_ISSUE.value,
                Intent.REPEAT.value,
                Intent.WAIT.value,
                Intent.HUMAN_HANDOFF.value,
                Intent.CANCEL.value,
                Intent.REJECT.value,
                Intent.END_SESSION.value,
            }

            if self._should_resume_previous_playback(transcript.text, normalized_text):
                candidate_response = self._build_resume_playback_text(normalized_text)
                if candidate_response:
                    playback_resume_selected = True
                    intent = IntentResult("playback_resume", 0.99, False, "resume_playback")
                    self._log(
                        f"playback resume participant={self._participant.identity} "
                        f"utterance_id={utterance_id} next_segment_index={self._playback_state.next_segment_index} "
                        f"resume_text={candidate_response!r}"
                    )
                    self._needs_rescue_prompt = False

            cached = None
            if not playback_resume_selected:
                cached = (
                    self._tool_graph.cached_reply_for_text(normalized_text, state_snapshot)
                    if self._tool_graph is not None
                    else None
                )

            if cached:
                candidate_response = cached.reply_text
                candidate_node = cached.next_node

            elif not normalized_text:
                candidate_response = (
                    self._responses.rescue_prompt()
                    if self._dialogue_state.stage == "greeting"
                    else self._config.fallback_low_confidence_text
                )

            elif intent.intent in canned_intents:
                if intent.intent == Intent.GREETING.value and not (self._dialogue_state.last_agent_text or self._last_agent_message):
                    opening = self._contextual_opening_reply()
                    if opening is None:
                        opening = self._tool_graph.opening_prompt() if self._tool_graph is not None else None
                    if opening is not None:
                        candidate_node, candidate_response = opening
                    else:
                        candidate_response = self._responses.choose(
                            intent,
                            last_agent_message=None,
                        )
                elif intent.intent == Intent.READY_TO_TALK.value and self._dialogue_state.last_agent_text:
                    if graph_question:
                        candidate_node, candidate_response = graph_question
                    else:
                        candidate_response = self._state_fallback_reply()
                else:
                    if intent.intent == Intent.IDENTIFY_SELF.value and self._has_prefilled_lead():
                        candidate_response = self._contextual_identity_reply()
                    else:
                        candidate_response = self._responses.choose(
                            intent,
                            last_agent_message=self._last_semantic_agent_message
                            or self._dialogue_state.last_agent_text
                            or self._last_agent_message,
                        )
                    if intent.intent in opening_intents:
                        candidate_node = "check_convenience"

            elif intent.intent in {Intent.WHY_NEED_INFO.value, Intent.SERVICE_COMPLAINT.value}:
                candidate_response = self._repair_and_resume_reply(intent)

            elif intent.intent == Intent.LATENCY_QUESTION.value:
                candidate_response = f"Связь чуть задержалась. {self._state_fallback_reply()}"

            elif updated_fields and not playback_resume_selected:
                if graph_question:
                    candidate_node, _graph_question_text = graph_question
                candidate_response = self._state_guided_reply(intent)

            elif not playback_resume_selected and not intent.use_llm and intent.action != Action.CALL_LLM.value:
                if graph_question:
                    candidate_node, candidate_response = graph_question
                else:
                    candidate_response = self._responses.choose(
                        intent,
                        last_agent_message=self._last_semantic_agent_message
                        or self._dialogue_state.last_agent_text
                        or self._last_agent_message,
                    )

            force_llm, repair_reason = should_force_llm_repair(
                intent=intent,
                normalized_text=normalized_text,
                state=state_snapshot,
                candidate_response=candidate_response,
            )
            if updated_fields and repair_reason == "intent_requested_llm":
                force_llm = False
                repair_reason = ""

            if force_llm:
                await self._publish_status("complex_request_detected")

                response_text, llm_reply, llm_latency_ms = await self._generate_adaptive_repair_reply(
                    participant_identity=self._participant.identity,
                    normalized_text=normalized_text,
                    raw_text=transcript.text,
                    reason=repair_reason,
                    candidate_response=candidate_response,
                    utterance_id=utterance_id,
                )

                next_graph_node = candidate_node or self._dialogue_state.current_node

            elif candidate_response:
                await self._publish_status("simple_intent_detected")

                response_text = candidate_response
                llm_reply = None
                llm_latency_ms = 0
                next_graph_node = candidate_node or self._dialogue_state.current_node

            elif self._should_try_stalled_slot_rescue(
                normalized_text=normalized_text,
                intent=intent,
                updated_fields=updated_fields,
                state_snapshot=state_snapshot,
            ):
                await self._publish_status("complex_request_detected")
                response_text, llm_reply, llm_latency_ms = await self._generate_stalled_slot_rescue(
                    normalized_text=normalized_text,
                    history=self._recent_history(),
                    state_snapshot=state_snapshot,
                    utterance_id=utterance_id,
                )
                next_graph_node = candidate_node or self._dialogue_state.current_node

            else:
                await self._publish_status("complex_request_detected")

                state_snapshot = self._dialogue_state.snapshot()
                knowledge = self._kb.retrieve(normalized_text, state_snapshot, limit=2)
                examples = self._kb.relevant_examples(normalized_text)[:2]

                graph_context = (
                    self._tool_graph.llm_context_for_text(normalized_text, state_snapshot)
                    if self._tool_graph is not None
                    else None
                )

                if graph_context:
                    next_graph_node = str(graph_context.get("node_name", "")).strip() or next_graph_node

                llm_state = self._llm_state_payload(state_snapshot)
                llm_reply, llm_latency_ms = await self._llm_service.generate_response(
                    normalized_text=normalized_text,
                    history=self._recent_history(limit=4),
                    dialogue_state=llm_state,
                    knowledge=knowledge,
                    truth_rules=self._kb.truth_rules,
                    examples=examples,
                    graph_context=graph_context,
                    max_tokens_override=min(140, self._config.llm_max_tokens),
                )

                response_text = self._validate_and_log_llm_reply(
                    reply_tts=llm_reply.reply_tts,
                    fallback_reply=self._state_fallback_reply(),
                    state=state_snapshot,
                    knowledge=knowledge,
                    truth_rules=self._kb.truth_rules,
                    utterance_id=utterance_id,
                    mode="normal_llm",
                    reason_hint="normal_generation",
                )

            if normalize_for_compare(response_text) == normalize_for_compare(self._dialogue_state.last_agent_text):
                response_text, llm_reply, llm_latency_ms = await self._generate_adaptive_repair_reply(
                    participant_identity=self._participant.identity,
                    normalized_text=normalized_text,
                    raw_text=transcript.text,
                    reason="repeated_final_response",
                    candidate_response=response_text,
                    utterance_id=utterance_id,
                )

            speech_situation = self._detect_speech_situation(
                intent=intent,
                updated_fields=updated_fields,
                repair_reason=repair_reason,
            )
            raw_response_text = response_text
            self._last_semantic_agent_message = raw_response_text
            response_text = self._sales_speech_styler.style(
                response_text,
                intent=intent.intent,
                stage=str(self._dialogue_state.stage),
                situation=speech_situation,
                known_facts=state_snapshot.get("known_facts", {}),
                repair_reason=repair_reason,
            )
            tts_speed = self._sales_speech_styler.speech_rate(
                situation=speech_situation,
                intent=intent.intent,
            )
            response_ready_time_ms = int(time.time() * 1000)
            self._log(
                f"turn decision participant={self._participant.identity} "
                f"utterance_id={utterance_id} router_intent={intent.intent} "
                f"action={intent.action} use_llm={str(intent.use_llm).lower()} "
                f"final_response_text={response_text!r}"
            )

            self._dialogue_state.update_from_agent(
                response_text,
                llm_reply.next_step if llm_reply else "",
                kb=self._kb,
                current_node=next_graph_node,
            )
            self._remember_agent_question(raw_response_text)
            self._last_agent_message = response_text
            self._session_memory.add_assistant(response_text)
            self._session_memory.sync_from_dialogue_state(
                self._dialogue_state.snapshot(),
                last_question=self._last_question_text,
            )
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
                error_stage = "tts"
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
                    tts_speed=tts_speed,
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
            if not error_stage:
                error_stage = "processing"
            error_message = str(exc) or repr(exc) or exc.__class__.__name__
            self._log(
                f"utterance processing failed participant={self._participant.identity} "
                f"stage={error_stage} error={error_message}"
            )
            self._log(traceback.format_exc())
            response_text = self._config.fallback_low_confidence_text
            await self._event_bus.publish_error(
                stage=error_stage,
                message=error_message,
                participant_identity=self._participant.identity,
                destination_identities=[self._participant.identity],
                utterance_id=utterance_id,
            )
            if self._uses_text_api_backend():
                response_text = ""
                raw_response_text = ""
                suppress_response = True
            else:
                response_text = self._config.fallback_low_confidence_text
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
            self._log(
                "turn timing "
                f"participant={self._participant.identity} "
                f"utterance_id={utterance_id} "
                f"stt_ms={transcript.stt_latency_ms} "
                f"router_ms={router_latency_ms} "
                f"classifier_ms={classifier_latency_ms} "
                f"llm_ms={llm_latency_ms} "
                f"tts_first_segment_ms={tts_latency_ms} "
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
                    "tts_latency_ms": tts_latency_ms,
                    "tts_first_segment_latency_ms": tts_latency_ms,
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


class VoiceSessionManager:
    def __init__(
        self,
        *,
        room: rtc.Room,
        config: VoicePipelineConfig,
        event_bus: AgentEventBus,
        llm_service: OpenAiLlmService | TextApiLlmService,
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

    def _ensure_session(self, participant: rtc.RemoteParticipant) -> ParticipantAudioSession:
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
        return session

    async def start_audio_track(
        self,
        *,
        track: rtc.Track,
        participant: rtc.RemoteParticipant,
        track_sid: str | None = None,
    ) -> None:
        session = self._ensure_session(participant)
        await session.ensure_started(track, track_sid=track_sid)

    async def apply_lead_profile(self, participant: rtc.RemoteParticipant, profile: dict[str, Any]) -> None:
        session = self._ensure_session(participant)
        session.apply_lead_profile(profile)

    async def participant_disconnected(self, participant_identity: str) -> None:
        session = self._sessions.pop(participant_identity, None)
        if session is not None:
            await session.aclose()

    async def aclose(self) -> None:
        sessions = list(self._sessions.values())
        self._sessions.clear()
        await asyncio.gather(*(session.aclose() for session in sessions), return_exceptions=True)
        llm_close = getattr(self._llm_service, "aclose", None)
        if callable(llm_close):
            with contextlib.suppress(Exception):
                await llm_close()
