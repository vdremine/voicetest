from __future__ import annotations

import asyncio
import json
import math
import os
import re
import threading
import time
import uuid
import wave
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from faster_whisper import WhisperModel
from livekit import rtc
from openai import AsyncOpenAI
from silero_vad import load_silero_vad


def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(slots=True)
class VoicePipelineConfig:
    sample_rate: int
    num_channels: int
    frame_size_ms: int
    vad_threshold: float
    vad_min_speech_duration_ms: int
    vad_min_silence_duration_ms: int
    vad_speech_pad_ms: int
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
    llm_enabled: bool
    llm_model: str
    llm_reasoning_effort: str
    llm_timeout_seconds: float
    llm_base_url: str
    llm_api_key: str
    llm_temperature: float
    llm_max_tokens: int
    tts_enabled: bool
    tts_model_path: Path
    tts_model_url: str
    tts_speaker: str
    tts_sample_rate: int
    tts_publish_sample_rate: int
    tts_frame_ms: int
    utterance_dir: Path
    session_log_dir: Path
    events_topic: str
    fallback_repeat_text: str
    fallback_low_confidence_text: str
    fallback_complex_text: str

    @classmethod
    def from_env(cls) -> "VoicePipelineConfig":
        return cls(
            sample_rate=int(os.getenv("AUDIO_SAMPLE_RATE", "16000")),
            num_channels=int(os.getenv("AUDIO_NUM_CHANNELS", "1")),
            frame_size_ms=int(os.getenv("AUDIO_FRAME_SIZE_MS", "20")),
            vad_threshold=float(os.getenv("VAD_THRESHOLD", "0.45")),
            vad_min_speech_duration_ms=int(os.getenv("VAD_MIN_SPEECH_DURATION_MS", "200")),
            vad_min_silence_duration_ms=int(os.getenv("VAD_MIN_SILENCE_DURATION_MS", "500")),
            vad_speech_pad_ms=int(os.getenv("VAD_SPEECH_PAD_MS", "120")),
            vad_use_onnx=env_bool("VAD_USE_ONNX", False),
            torch_num_threads=int(os.getenv("TORCH_NUM_THREADS", "1")),
            stt_enabled=env_bool("STT_ENABLED", True),
            stt_model=os.getenv("STT_MODEL", "Systran/faster-whisper-small"),
            stt_device=os.getenv("STT_DEVICE", "auto"),
            stt_compute_type_cpu=os.getenv("STT_COMPUTE_TYPE_CPU", "int8"),
            stt_compute_type_gpu=os.getenv("STT_COMPUTE_TYPE_GPU", "float16"),
            stt_language=os.getenv("STT_LANGUAGE", "ru"),
            stt_beam_size=int(os.getenv("STT_BEAM_SIZE", "1")),
            stt_confidence_floor=float(os.getenv("STT_CONFIDENCE_FLOOR", "0.35")),
            llm_enabled=env_bool("LLM_ENABLED", True),
            llm_model=os.getenv("LLM_MODEL", "Qwen/Qwen3-8B"),
            llm_reasoning_effort=os.getenv("LLM_REASONING_EFFORT", "low"),
            llm_timeout_seconds=float(os.getenv("LLM_TIMEOUT_SECONDS", "15")),
            llm_base_url=os.getenv("LLM_BASE_URL", "http://127.0.0.1:8001/v1").strip(),
            llm_api_key=os.getenv("LLM_API_KEY", "local-token").strip(),
            llm_temperature=float(os.getenv("LLM_TEMPERATURE", "0.2")),
            llm_max_tokens=int(os.getenv("LLM_MAX_TOKENS", "96")),
            tts_enabled=env_bool("TTS_ENABLED", True),
            tts_model_path=Path(os.getenv("TTS_MODEL_PATH", "/models/silero-tts/ru/v5_4_ru.pt")),
            tts_model_url=os.getenv(
                "TTS_MODEL_URL",
                "https://models.silero.ai/models/tts/ru/v5_4_ru.pt",
            ),
            tts_speaker=os.getenv("TTS_SPEAKER", "xenia"),
            tts_sample_rate=int(os.getenv("TTS_SAMPLE_RATE", "24000")),
            tts_publish_sample_rate=int(os.getenv("TTS_PUBLISH_SAMPLE_RATE", "24000")),
            tts_frame_ms=int(os.getenv("TTS_FRAME_MS", "20")),
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
                "Я понял запрос. Сейчас подключу обработку следующего уровня.",
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
    def __init__(self, config: VoicePipelineConfig, log: Callable[[str], None]) -> None:
        self._config = config
        self._log = log
        self._client: AsyncOpenAI | None = None

    @property
    def enabled(self) -> bool:
        return self._config.llm_enabled and bool(self._config.llm_model)

    def _ensure_client(self) -> AsyncOpenAI:
        if self._client is None:
            kwargs: dict[str, Any] = {"timeout": self._config.llm_timeout_seconds}
            if self._config.llm_base_url:
                kwargs["base_url"] = self._config.llm_base_url
                kwargs["api_key"] = self._config.llm_api_key or "local-token"
            else:
                api_key = os.getenv("OPENAI_API_KEY", "").strip()
                if not api_key:
                    raise RuntimeError("OPENAI_API_KEY is not configured")
                kwargs["api_key"] = api_key
            self._client = AsyncOpenAI(**kwargs)
        return self._client

    async def generate_response(
        self,
        *,
        normalized_text: str,
        history: list[dict[str, str]],
    ) -> tuple[str, int]:
        if not self.enabled:
            raise RuntimeError("LLM is disabled by configuration")

        client = self._ensure_client()
        started_at = time.perf_counter()
        history_lines = [
            f"{item['role']}: {item['text']}"
            for item in history[-8:]
            if item.get("text")
        ]
        history_block = "\n".join(history_lines) if history_lines else "history: <empty>"

        completion = await client.chat.completions.create(
            model=self._config.llm_model,
            temperature=self._config.llm_temperature,
            max_tokens=self._config.llm_max_tokens,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Ты голосовой помощник. Отвечай по-русски коротко, естественно, без канцелярита. "
                        "Ответ должен быть удобен для озвучивания: 1-2 коротких предложения. "
                        "Не показывай размышления, рассуждения, служебные теги, XML, markdown, "
                        "скрытые планы, chain-of-thought и текст в стиле <think>...</think>. "
                        "Верни только финальный ответ для пользователя."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"{history_block}\n"
                        f"user_request: {normalized_text}\n"
                        "Верни только текст ответа для голоса."
                    ),
                },
            ],
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
        return sanitize_voice_response(text, fallback="Уточните, пожалуйста, что именно вы хотите сделать."), latency_ms


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

    if value.startswith(("Хорошо, пользователь", "Пользователь", "Нужно ", "Стоит ")):
        return fallback

    sentences = re.split(r"(?<=[.!?])\s+", value)
    short_text = " ".join(sentence.strip() for sentence in sentences[:2] if sentence.strip()).strip()
    return short_text or fallback


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

    def synthesize(self, text: str) -> tuple[np.ndarray, int, int]:
        started_at = time.perf_counter()
        with self._lock:
            model = self._ensure_model()
            audio = model.apply_tts(
                text=text,
                speaker=self._config.tts_speaker,
                sample_rate=self._config.tts_sample_rate,
            )

        if isinstance(audio, torch.Tensor):
            audio_np = audio.detach().cpu().numpy()
        else:
            audio_np = np.asarray(audio)

        audio_np = np.clip(audio_np, -1.0, 1.0)
        pcm16 = (audio_np * 32767.0).astype(np.int16)
        latency_ms = int((time.perf_counter() - started_at) * 1000)
        return pcm16, self._config.tts_sample_rate, latency_ms


class LiveKitAudioPublisher:
    def __init__(self, room: rtc.Room, config: VoicePipelineConfig, log: Callable[[str], None]) -> None:
        self._room = room
        self._config = config
        self._log = log
        self._source: rtc.AudioSource | None = None
        self._track: rtc.LocalAudioTrack | None = None
        self._published = False
        self._lock = asyncio.Lock()

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

    async def speak_pcm(self, pcm16: np.ndarray, sample_rate: int) -> None:
        await self.ensure_published()
        assert self._source is not None

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

        await self._source.wait_for_playout()

    @staticmethod
    def _resample(pcm16: np.ndarray, *, orig_rate: int, target_rate: int) -> np.ndarray:
        if orig_rate == target_rate or len(pcm16) == 0:
            return pcm16

        duration = len(pcm16) / float(orig_rate)
        target_len = max(1, int(round(duration * target_rate)))
        source_x = np.linspace(0.0, 1.0, num=len(pcm16), endpoint=False)
        target_x = np.linspace(0.0, 1.0, num=target_len, endpoint=False)
        resampled = np.interp(target_x, source_x, pcm16.astype(np.float32))
        return np.clip(resampled, -32768.0, 32767.0).astype(np.int16)


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

    def transcribe(self, wav_path: Path, duration_ms: int) -> TranscriptResult:
        started_at = time.perf_counter()
        with self._lock:
            model = self._ensure_model()
            segments_iter, info = model.transcribe(
                str(wav_path),
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
    def __init__(self) -> None:
        self._greeting = {
            "алло",
            "ало",
            "привет",
            "здравствуйте",
            "добрый день",
            "добрый вечер",
            "доброе утро",
            "доброй ночи",
        }
        self._confirm = {"да", "угу", "ага", "подтверждаю", "конечно", "хорошо"}
        self._reject = {"нет", "неа", "не надо"}
        self._cancel = {"отмена", "отменить", "отбой"}
        self._repeat = {"повтори", "повтори пожалуйста", "еще раз", "ещё раз", "не понял"}
        self._wait = {"подожди", "секунду", "одну секунду"}
        self._end_session = {"стоп", "завершить", "закончить"}
        self._handoff_tokens = {"оператор", "человек"}

    def route(self, text: str) -> IntentResult:
        if not text:
            return IntentResult("clarify", 0.0, False, "ask_repeat")

        if text in self._greeting:
            return IntentResult("greeting", 0.99, False, "ack_greeting")
        if text in self._confirm:
            return IntentResult("confirm", 0.99, False, "ack_confirm")
        if text in self._reject:
            return IntentResult("reject", 0.99, False, "ack_reject")
        if text in self._cancel or "отмена" in text:
            return IntentResult("cancel", 0.99, False, "cancel_action")
        if text in self._repeat or text.startswith("повтори") or text.startswith("повтори"):
            return IntentResult("repeat", 0.99, False, "repeat_last_agent_message")
        if text in self._wait or text.startswith("подожди"):
            return IntentResult("wait", 0.98, False, "ack_wait")
        if text in self._end_session or text.startswith("заверши") or text.startswith("стоп"):
            return IntentResult("end_session", 0.98, False, "end_session")
        if any(token in text for token in self._handoff_tokens):
            return IntentResult("human_handoff", 0.99, False, "handoff_to_human")
        if len(text.split()) <= 2:
            return IntentResult("unknown_short", 0.45, False, "ask_repeat")

        return IntentResult("complex_request", 0.8, True, "call_llm")


class CannedResponseEngine:
    def __init__(self, config: VoicePipelineConfig) -> None:
        self._config = config

    def choose(self, intent: IntentResult, *, last_agent_message: str | None) -> str:
        if intent.intent == "greeting":
            return "Добрый день. Слушаю вас."
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

        self._session_id = f"{participant.identity}-{int(time.time())}-{uuid.uuid4().hex[:8]}"
        self._session_logger = SessionLogger(config.session_log_dir / f"{self._session_id}.jsonl")

        self._audio_task: asyncio.Task[None] | None = None
        self._sample_buffer = np.empty(0, dtype=np.int16)
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

    async def ensure_started(self, track: rtc.Track) -> None:
        if self._audio_task and not self._audio_task.done():
            return
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
        if self._is_speaking or self._is_processing:
            self._sample_buffer = np.empty(0, dtype=np.int16)
            self._pre_speech_chunks.clear()
            return

        samples = np.array(frame.data, dtype=np.int16, copy=True)
        if frame.num_channels > 1:
            samples = samples.reshape(-1, frame.num_channels).mean(axis=1).astype(np.int16)

        if self._sample_buffer.size == 0:
            self._sample_buffer = samples
        else:
            self._sample_buffer = np.concatenate((self._sample_buffer, samples))

        while self._sample_buffer.size >= self._vad.window_size:
            window = self._sample_buffer[: self._vad.window_size]
            self._sample_buffer = self._sample_buffer[self._vad.window_size :]
            await self._process_vad_window(window)

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
                wav_path=wav_path,
                duration_ms=duration_ms,
                vad_confidence=vad_confidence,
                speech_start_time_ms=self._speech_started_at_ms,
                speech_end_time_ms=speech_end_time_ms,
            )
        )

    def _reset_utterance_state(self) -> None:
        self._utterance_chunks = []
        self._utterance_probs = []
        self._last_speech_chunk_index = 0
        self._speech_ms = 0
        self._silence_ms = 0
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

    async def _process_utterance(
        self,
        *,
        utterance_id: str,
        wav_path: Path,
        duration_ms: int,
        vad_confidence: float,
        speech_start_time_ms: int,
        speech_end_time_ms: int,
    ) -> None:
        await self._publish_status("stt_processing")
        started_at = time.perf_counter()
        self._is_processing = True
        response_text = self._config.fallback_low_confidence_text
        normalized_text = ""
        intent = IntentResult("clarify", 0.0, False, "ask_repeat")
        transcript = TranscriptResult("", self._config.stt_language, 0.0, duration_ms, 0)
        error_stage = ""
        response_published = False
        llm_latency_ms = 0
        tts_latency_ms = 0

        try:
            if not self._config.stt_enabled:
                raise RuntimeError("STT is disabled by configuration")

            transcript = await asyncio.to_thread(self._stt_service.transcribe, wav_path, duration_ms)

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
            if transcript.text:
                self._history.append({"role": "user", "text": normalized_text or transcript.text})
                self._history = self._history[-12:]
            await self._publish_status("routing_intent")
            intent = self._router.route(normalized_text)
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

            if not transcript.text or transcript.confidence < self._config.stt_confidence_floor:
                response_text = self._config.fallback_low_confidence_text
            else:
                if intent.use_llm:
                    await self._publish_status("complex_request_detected")
                    if self._llm_service.enabled:
                        try:
                            response_text, llm_latency_ms = await self._llm_service.generate_response(
                                normalized_text=normalized_text,
                                history=self._history,
                            )
                        except Exception as exc:
                            self._log(f"llm fallback failed for {self._participant.identity}: {exc}")
                            response_text = self._responses.choose(
                                intent,
                                last_agent_message=self._last_agent_message,
                            )
                    else:
                        response_text = self._responses.choose(
                            intent,
                            last_agent_message=self._last_agent_message,
                        )
                else:
                    await self._publish_status("simple_intent_detected")
                    response_text = self._responses.choose(
                        intent,
                        last_agent_message=self._last_agent_message,
                    )

            self._last_agent_message = response_text
            self._history.append({"role": "assistant", "text": response_text})
            self._history = self._history[-12:]

            await self._event_bus.publish_json(
                {
                    "type": "agent_response_text",
                    "utterance_id": utterance_id,
                    "participant_identity": self._participant.identity,
                    "text": response_text,
                    "use_llm": intent.use_llm,
                },
                destination_identities=[self._participant.identity],
            )
            response_published = True
            if self._config.tts_enabled:
                await self._publish_status("speaking")
                self._is_speaking = True
                self._log(
                    f"tts synth start participant={self._participant.identity} "
                    f"utterance_id={utterance_id} text={response_text!r}"
                )
                tts_pcm16, tts_sample_rate, tts_latency_ms = await asyncio.to_thread(
                    self._tts_service.synthesize,
                    response_text,
                )
                self._log(
                    f"tts synth done participant={self._participant.identity} "
                    f"utterance_id={utterance_id} samples={len(tts_pcm16)} sample_rate={tts_sample_rate}"
                )
                self._log(
                    f"tts publish start participant={self._participant.identity} "
                    f"utterance_id={utterance_id}"
                )
                await self._audio_publisher.speak_pcm(tts_pcm16, tts_sample_rate)
                self._log(
                    f"tts publish done participant={self._participant.identity} "
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
                },
                destination_identities=[self._participant.identity],
            )
            response_published = True
        finally:
            total_latency_ms = int((time.perf_counter() - started_at) * 1000)
            if not response_published:
                await self._event_bus.publish_json(
                    {
                        "type": "agent_response_text",
                        "utterance_id": utterance_id,
                        "participant_identity": self._participant.identity,
                        "text": response_text,
                        "use_llm": intent.use_llm,
                    },
                    destination_identities=[self._participant.identity],
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
                    "router_latency_ms": 0,
                    "total_latency_ms": total_latency_ms,
                    "error_stage": error_stage,
                    "response_text": response_text,
                    "llm_latency_ms": llm_latency_ms,
                    "tts_latency_ms": tts_latency_ms,
                }
            )
            self._is_speaking = False
            self._is_processing = False
            await self._publish_status("waiting_for_speech")


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

        await session.ensure_started(track)

    async def participant_disconnected(self, participant_identity: str) -> None:
        session = self._sessions.pop(participant_identity, None)
        if session is not None:
            await session.aclose()

    async def aclose(self) -> None:
        sessions = list(self._sessions.values())
        self._sessions.clear()
        await asyncio.gather(*(session.aclose() for session in sessions), return_exceptions=True)
