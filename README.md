# LiveKit Transport-First Scaffold

Минимальный каркас для transport-check и первого browser-safe AI voice loop.

Текущая цель:

1. Браузер заходит в комнату по `https://vdremin.ru`.
2. Агент заходит в ту же комнату.
3. Frontend видит `agent-001`.
4. Frontend получает `agent_ready`.
5. Агент видит audio track пользователя.
6. Агент умеет отправлять data-message обратно.

Что сознательно не делаем на этом этапе:

- TURN/TLS
- отдельный `turn.vdremin.ru`
- полный production perimeter
- multi-node routing
- прикладной AI orchestration поверх транспорта

## Что есть в репозитории

- `docker-compose.yml` — локальный smoke-test
- `docker-compose.prod.yml` — доменный baseline для Ubuntu
- `token_server` на FastAPI
- простой `frontend` без React
- Python `agent`, который входит в комнату и отправляет `agent_ready`
- bootstrap/user-data скрипты
- `Caddy` для HTTPS и WSS

## Структура

```text
.
├── .env.example
├── .env.prod.example
├── docker-compose.yml
├── docker-compose.prod.yml
├── README.md
├── token_server/
│   ├── main.py
│   ├── requirements.txt
│   └── Dockerfile
├── agent/
│   ├── main.py
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── index.html
│   ├── nginx.conf
│   └── nginx.prod.conf
└── scripts/
    ├── bootstrap_server.sh
    ├── check_gpu.sh
    ├── open_ports.sh
    ├── user-data.sh
    └── user-data.cloud-config.yaml
```

## Режимы запуска

### 1. Локальный smoke-test

Файл: [docker-compose.yml](/Users/dr_emin/Desktop/livekit/docker-compose.yml)

URL:

- frontend: `http://localhost:8080`
- token server: `http://localhost:8000`
- LiveKit: `ws://localhost:7880`

Запуск:

```bash
cd /Users/dr_emin/Desktop/livekit
cp .env.example .env
docker compose up --build
```

### 2. Ubuntu domain baseline

Файл: [docker-compose.prod.yml](/Users/dr_emin/Desktop/livekit/docker-compose.prod.yml)

Сейчас это основной серверный режим для проверки транспорта и микрофона без `localhost` tunnel.

URL:

- frontend: `https://vdremin.ru`
- token server через frontend: `https://vdremin.ru/token`
- health: `https://vdremin.ru/healthz`
- LiveKit: `wss://livekit.vdremin.ru`

## Схема baseline

```mermaid
flowchart LR
    U["Browser / User"] -->|HTTPS 443| C["Caddy"]
    C -->|proxy| F["frontend nginx :8080"]
    F -->|GET /token| T["token_server :8000"]
    T -->|JWT + wss URL| U
    U -->|WSS 443| C
    C -->|proxy| LK["LiveKit signal :7880"]
    A["agent"] -->|WS 127.0.0.1:7880| LK
    U -->|WebRTC UDP 50000-60000| LK
    U -->|WebRTC TCP 7881 fallback| LK
```

Что важно:

- frontend отдается по `https://vdremin.ru`
- token server наружу напрямую не нужен, frontend проксирует `/token`
- browser получает secure context и может запрашивать микрофон без tunnel
- signal идет через `wss://livekit.vdremin.ru`
- media по-прежнему идет напрямую в LiveKit по UDP/TCP

## DNS

Для этого baseline нужны две A-записи:

- `vdremin.ru -> 193.39.168.244`
- `livekit.vdremin.ru -> 193.39.168.244`

`turn.vdremin.ru` пока не нужен.

## Domain baseline на Ubuntu 24.04

Минимальный `.env`:

```env
SERVER_TIMEZONE=Europe/Moscow
SERVER_PUBLIC_IP=193.39.168.244
APP_DOMAIN=vdremin.ru
LIVEKIT_DOMAIN=livekit.vdremin.ru
LIVEKIT_API_KEY=voice-agent-prod
LIVEKIT_API_SECRET=replace-with-long-random-secret
LIVEKIT_URL_PUBLIC=wss://livekit.vdremin.ru
LIVEKIT_USE_EXTERNAL_IP=true
TOKEN_TTL_MINUTES=60
CORS_ALLOW_ORIGINS=*
AGENT_ROOM=demo-room
AGENT_IDENTITY=agent-001
AGENT_NAME=Room Agent
AGENT_READY_TOPIC=presence
TOKEN_REQUEST_TIMEOUT=10
```

Запуск:

```bash
cd /opt/voice-agent
cp .env.prod.example .env
docker compose -f docker-compose.prod.yml up --build -d
```

## Порты

Для domain baseline должны быть доступны:

- `80/tcp` — ACME / HTTP challenge
- `443/tcp` — HTTPS frontend и WSS signal
- `7881/tcp` — WebRTC TCP fallback
- `50000-60000/udp` — media traffic

`3478/udp` можно держать открытым заранее, но текущая конфигурация его не использует.
`7880/tcp` снаружи больше не нужен. По LiveKit docs этот порт должен быть за SSL termination layer, а наружу для клиентов нужен `wss://...` endpoint.

## Проверка

1. Откройте `https://vdremin.ru`.
2. Нажмите `Join`.
3. Проверьте, что frontend подключился к комнате.
4. Проверьте, что в participants виден `agent-001`.
5. Проверьте, что в логе есть `agent_ready`.
6. Включите `Mic On`.
7. Убедитесь по логам агента, что он подписался на пользовательский audio track.
8. Проверьте, что frontend получил data-message `user_audio_track_detected:<identity>` на topic `agent_status`.

## Что делаем сразу после этого

После transport-check переходим в AI loop:

1. Получение аудиофреймов от пользователя.
2. VAD на аудиопотоке.
3. Буферизация utterance.
4. STT.
5. Keyword/intent router без LLM для простых команд.
6. LLM только для сложных запросов.
7. TTS с разметкой.
8. Публикация аудиоответа обратно в LiveKit.

## Token API

### `GET /healthz`

```json
{
  "status": "ok"
}
```

### `GET /token?room=demo-room&identity=user-123`

```json
{
  "url": "wss://livekit.vdremin.ru",
  "token": "<jwt>",
  "room": "demo-room",
  "identity": "user-123"
}
```

## Источники

По официальной документации LiveKit secure deployment требует домен, SSL termination и `wss://` endpoint для SDK-клиентов:

- [Deployment](https://docs.livekit.io/home/self-hosting/deployment/)
- [Virtual machines](https://docs.livekit.io/transport/self-hosting/vm/)
- [Ports and firewall](https://docs.livekit.io/transport/self-hosting/ports-firewall/)
