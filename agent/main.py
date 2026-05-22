import asyncio
import os
import signal
from typing import Any
from urllib.parse import urlencode

import httpx
from livekit import rtc

from voice_loop import (
    AgentEventBus,
    LiveKitAudioPublisher,
    OpenAiLlmService,
    SileroTtsService,
    VoicePipelineConfig,
    VoiceSessionManager,
)


TOKEN_SERVER_URL = os.getenv("TOKEN_SERVER_URL", "http://token_server:8000/token")
LIVEKIT_URL_INTERNAL = os.getenv("LIVEKIT_URL_INTERNAL", "ws://livekit:7880")
AGENT_ROOM = os.getenv("AGENT_ROOM", "demo-room")
AGENT_IDENTITY = os.getenv("AGENT_IDENTITY", "agent-001")
AGENT_NAME = os.getenv("AGENT_NAME", "Room Agent")
AGENT_READY_TOPIC = os.getenv("AGENT_READY_TOPIC", "presence")
TOKEN_REQUEST_TIMEOUT = float(os.getenv("TOKEN_REQUEST_TIMEOUT", "10"))
CONNECT_RETRY_DELAY = float(os.getenv("CONNECT_RETRY_DELAY", "2"))


def log(message: str) -> None:
    print(f"[agent] {message}", flush=True)


def is_audio_kind(kind: Any) -> bool:
    kind_text = str(kind).lower()
    if "audio" in kind_text:
        return True

    kind_name = str(getattr(kind, "name", "")).lower()
    if "audio" in kind_name:
        return True

    # LiveKit Python SDK may expose TrackKind as an enum-like integer where 1 == audio.
    if kind == 1:
        return True

    track_kind_audio = getattr(getattr(rtc, "TrackKind", object()), "KIND_AUDIO", None)
    if track_kind_audio is not None and kind == track_kind_audio:
        return True

    return False


async def fetch_token() -> dict[str, Any]:
    params = urlencode({"room": AGENT_ROOM, "identity": AGENT_IDENTITY})
    url = f"{TOKEN_SERVER_URL}?{params}"

    async with httpx.AsyncClient(timeout=TOKEN_REQUEST_TIMEOUT) as client:
        response = await client.get(url)
        response.raise_for_status()
        payload = response.json()

    if "token" not in payload:
        raise RuntimeError("token server response does not contain 'token'")

    return payload


async def publish_ready(room: rtc.Room, destination_identities: list[str] | None = None) -> None:
    try:
        await room.local_participant.publish_data(
            "agent_ready",
            reliable=True,
            destination_identities=destination_identities or [],
            topic=AGENT_READY_TOPIC,
        )
        target = ",".join(destination_identities) if destination_identities else "room"
        log(f"published agent_ready to {target}")
    except Exception as exc:
        log(f"failed to publish agent_ready: {exc}")


async def publish_agent_event(
    room: rtc.Room,
    *,
    payload: str,
    topic: str,
    destination_identities: list[str] | None = None,
) -> None:
    try:
        await room.local_participant.publish_data(
            payload,
            reliable=True,
            destination_identities=destination_identities or [],
            topic=topic,
        )
        target = ",".join(destination_identities) if destination_identities else "room"
        log(f"published {payload} on topic={topic} to {target}")
    except Exception as exc:
        log(f"failed to publish {payload} on topic={topic}: {exc}")


def ensure_audio_subscription(publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant) -> None:
    if not is_audio_kind(publication.kind):
        return
    try:
        publication.set_subscribed(True)
        log(f"requested audio subscription: participant={participant.identity} track={publication.sid}")
    except Exception as exc:
        log(f"failed to subscribe audio track for {participant.identity}: {exc}")


async def run() -> None:
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    room = rtc.Room()
    pipeline_config = VoicePipelineConfig.from_env()
    event_bus = AgentEventBus(room, topic=pipeline_config.events_topic, log=log)
    llm_service = OpenAiLlmService(pipeline_config, log)
    tts_service = SileroTtsService(pipeline_config, log)
    audio_publisher = LiveKitAudioPublisher(room, pipeline_config, log)
    voice_sessions = VoiceSessionManager(
        room=room,
        config=pipeline_config,
        event_bus=event_bus,
        llm_service=llm_service,
        tts_service=tts_service,
        audio_publisher=audio_publisher,
        log=log,
    )

    def schedule(coro: Any, *, label: str) -> None:
        def _spawn() -> None:
            task = loop.create_task(coro)

            def _on_done(done_task: asyncio.Task[Any]) -> None:
                try:
                    done_task.result()
                except asyncio.CancelledError:
                    pass
                except Exception as exc:
                    log(f"scheduled task failed label={label}: {exc}")

            task.add_done_callback(_on_done)

        loop.call_soon_threadsafe(_spawn)

    @room.on("participant_connected")
    def on_participant_connected(participant: rtc.RemoteParticipant) -> None:
        log(f"participant connected: {participant.identity}")
        for publication in participant.track_publications.values():
            ensure_audio_subscription(publication, participant)

    @room.on("participant_active")
    def on_participant_active(participant: rtc.RemoteParticipant) -> None:
        log(f"participant active: {participant.identity}")
        schedule(publish_ready(room, [participant.identity]), label="publish_ready_participant_active")
        schedule(
            event_bus.publish_status(
                "agent_ready",
                participant_identity=participant.identity,
                status="ready",
                destination_identities=[participant.identity],
            ),
            label="publish_status_agent_ready",
        )

    @room.on("participant_disconnected")
    def on_participant_disconnected(participant: rtc.RemoteParticipant) -> None:
        log(f"participant disconnected: {participant.identity}")
        schedule(
            voice_sessions.participant_disconnected(participant.identity),
            label="participant_disconnected_cleanup",
        )

    @room.on("data_received")
    def on_data_received(data_packet: rtc.DataPacket) -> None:
        payload = data_packet.data.decode("utf-8", errors="replace")
        sender = data_packet.participant.identity if data_packet.participant else "server"
        topic = data_packet.topic or "-"
        log(f"data received from {sender} on topic={topic}: {payload}")

    @room.on("track_published")
    def on_track_published(
        publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant
    ) -> None:
        log(
            "track published: "
            f"participant={participant.identity} track={publication.sid} kind={publication.kind}"
        )
        ensure_audio_subscription(publication, participant)

    @room.on("track_subscribed")
    def on_track_subscribed(
        track: rtc.Track, publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant
    ) -> None:
        log(
            "track subscribed: "
            f"participant={participant.identity} track={publication.sid} kind={publication.kind}"
        )
        if not is_audio_kind(publication.kind):
            return

        schedule(
            publish_agent_event(
                room,
                payload=f"user_audio_track_detected:{participant.identity}",
                topic="agent_status",
                destination_identities=[participant.identity],
            ),
            label="publish_user_audio_track_detected",
        )
        schedule(
            voice_sessions.start_audio_track(track=track, participant=participant, track_sid=publication.sid),
            label="start_audio_track",
        )

    @room.on("track_unsubscribed")
    def on_track_unsubscribed(
        track: rtc.Track, publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant
    ) -> None:
        log(
            "track unsubscribed: "
            f"participant={participant.identity} track={publication.sid} kind={publication.kind}"
        )

    @room.on("track_subscription_failed")
    def on_track_subscription_failed(
        participant: rtc.RemoteParticipant, track_sid: str, error: str
    ) -> None:
        log(f"track subscription failed: participant={participant.identity} track={track_sid} error={error}")

    @room.on("connection_state_changed")
    def on_connection_state_changed(connection_state: rtc.ConnectionState) -> None:
        log(f"connection state changed: {connection_state}")

    @room.on("disconnected")
    def on_disconnected(reason: rtc.DisconnectReason) -> None:
        log(f"disconnected: {reason}")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            asyncio.get_running_loop().add_signal_handler(sig, stop_event.set)
        except NotImplementedError:
            pass

    while True:
        try:
            token_payload = await fetch_token()
            break
        except Exception as exc:
            log(f"token fetch failed, retrying in 2s: {exc}")
            await asyncio.sleep(2)

    token = token_payload["token"]
    livekit_url = LIVEKIT_URL_INTERNAL or token_payload.get("url")

    while True:
        try:
            log(f"connecting to room={AGENT_ROOM} as identity={AGENT_IDENTITY}")
            await room.connect(livekit_url, token)
            log(f"connected to livekit room={room.name}")
            break
        except Exception as exc:
            log(f"room connect failed, retrying in {CONNECT_RETRY_DELAY}s: {exc}")
            await asyncio.sleep(CONNECT_RETRY_DELAY)

    if pipeline_config.tts_enabled:
        try:
            await audio_publisher.ensure_published()
        except Exception as exc:
            log(f"failed to publish local audio track for agent voice: {exc}")

    await publish_ready(room)

    if room.remote_participants:
        for participant in room.remote_participants.values():
            await publish_ready(room, [participant.identity])
            await event_bus.publish_status(
                "agent_ready",
                participant_identity=participant.identity,
                status="ready",
                destination_identities=[participant.identity],
            )
            for publication in participant.track_publications.values():
                ensure_audio_subscription(publication, participant)
                track = getattr(publication, "track", None)
                if track is not None and is_audio_kind(publication.kind):
                    await voice_sessions.start_audio_track(
                        track=track,
                        participant=participant,
                        track_sid=publication.sid,
                    )

    await stop_event.wait()

    log("shutting down")
    await voice_sessions.aclose()
    await room.disconnect()


if __name__ == "__main__":
    asyncio.run(run())
