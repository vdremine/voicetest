import os
from datetime import timedelta

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from livekit import api


def get_required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"missing required environment variable: {name}")
    return value


app = FastAPI(title="LiveKit Token Server", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in os.getenv("CORS_ALLOW_ORIGINS", "*").split(",")],
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/token")
async def token(
    room: str = Query(default="demo-room", min_length=1),
    identity: str = Query(default="user-123", min_length=1),
) -> dict[str, str]:
    try:
        api_key = get_required_env("LIVEKIT_API_KEY")
        api_secret = get_required_env("LIVEKIT_API_SECRET")
        livekit_url_public = get_required_env("LIVEKIT_URL_PUBLIC")
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    ttl_minutes = int(os.getenv("TOKEN_TTL_MINUTES", "60"))

    access_token = (
        api.AccessToken(api_key, api_secret)
        .with_identity(identity)
        .with_name(identity)
        .with_ttl(timedelta(minutes=ttl_minutes))
        .with_grants(
            api.VideoGrants(
                room_join=True,
                room=room,
                can_publish=True,
                can_subscribe=True,
                can_publish_data=True,
            )
        )
    )

    return {
        "url": livekit_url_public,
        "token": access_token.to_jwt(),
        "room": room,
        "identity": identity,
    }
