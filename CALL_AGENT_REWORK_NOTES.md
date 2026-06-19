# Call Agent — Rework Notes & Handoff

> Рабочие заметки по переделке диалогового движка `agent/call_agent/`.
> Цель: продолжить работу в новой сессии без потери контекста.
> Последнее обновление: 2026-06-19.

---

## 0. TL;DR (что сделано и что дальше)

- Переписан `agent/call_agent/` — диалоговый движок холодного звонка (брокер «МосИнвестФинанс», кредиты под залог недвижимости/ПТС, рефинанс, инвесторы).
- **Архитектура:** навигация детерминирована (код выбирает следующий вопрос из накопленных фактов), LLM пишет только тёплую `reflection` + извлекает факты. Это даёт стабильность даже на слабой модели (Qwen2.5-7B) и человеческий тон.
- Исправлены баги с прод-сервера (зацикливание на вопросе про залог, «живой оператор», битый JSON, странный расчёт 70%).
- **59 тестов, все зелёные.** Тестируется детерминированное ядро без обращения к LLM.
- **ГОЛОС:** обёртка voice_loop → call_agent (8787) УЖЕ написана (`TextApiLlmService`, `_run_text_api_turn`), включается флагом `DIALOGUE_BACKEND=text_api` (раздел 7). Остаётся инфра: LiveKit-ключи и способ подачи аудио (браузер/SIP).
- Открытый вопрос: смена TTS (раздел 8).

---

## 1. Где что лежит

```
/Users/dr_emin/Desktop/livekit/
├── agent/
│   ├── call_agent/              # <-- ДВИЖОК, который переделывали (текстовый API, порт 8787)
│   │   ├── api.py               # FastAPI: /session/start, /session/message, /session/reset, /metrics, /post_call
│   │   ├── flow.py              # ЯДРО: ветки, слоты, resolve_focus, openings, pitch, assemble_reply, анти-цикл helpers
│   │   ├── graph.py             # Раннер: ainvoke -> understand(LLM) -> apply -> commit; анти-цикл, deferral
│   │   ├── schema.py            # TurnUnderstanding (reflection/facts_update/answer/branch_signal/should_end) + NodeId + JSON-schema
│   │   ├── prompt.py            # SYSTEM_PROMPT (из файла) + build_turn_prompt + EXTRACTION_HINTS
│   │   ├── system_prompt.txt    # Персона Владимир, формула реплики, возражения, запреты, few-shot
│   │   ├── llm.py               # TurnLlmClient.understand(); vLLM guided_json + ТОЛЕРАНТНЫЙ JSON-парсер
│   │   ├── apply.py             # apply_facts/sanitize_facts_update (нормализация фактов, фильтр имени)
│   │   ├── post_call.py         # извлечение итога звонка (отдельный LLM-вызов)
│   │   ├── metrics.py, state.py # метрики латентности; TypedDict состояния сессии
│   │   └── graph_spec.py        # СТАРЫЙ граф; сейчас из него используется только call_connected.ask ("Алло.")
│   ├── tests/                   # pytest: test_flow / test_apply / test_runner / test_api_integration / test_llm
│   ├── .venv-dev/               # лёгкий dev-venv (pydantic/openai/httpx/fastapi/pytest, БЕЗ torch) для тестов
│   ├── voice_loop.py            # ГОЛОСОВОЙ цикл (LiveKit, STT/TTS/VAD) — использует agent_core, НЕ call_agent (см. раздел 7)
│   ├── main.py                  # точка входа голосового агента
│   ├── text_api.py              # `from call_agent.api import app` — то, что слушает 8787
│   ├── text_llm_cli.py          # CLI для отладки (agent_core)
│   └── agent_core/              # ДРУГОЙ диалоговый движок (ToolGraphRuntime, OpenAiLlmService) — его юзает voice_loop
├── docker-compose.prod.yml      # прод: llm(vLLM), text_llm(8787), agent(голос), livekit, token_server, frontend, caddy
├── docker-compose.llm.yml       # оверрайд только для vLLM
├── .env.prod.example            # все env-переменные (STT/TTS/LLM/LiveKit)
└── scripts/bootstrap_server.sh  # развёртывание на Ubuntu
```

Сервер: `root@Bohr:/opt/voicetest`, текстовый API проверяется `curl http://127.0.0.1:8787/...`.

---

## 2. Архитектура call_agent (главное)

**Инвариант:** позиция в звонке — чистая функция накопленных `known_facts`. LLM НЕ выбирает узел и НЕ формулирует вопрос.

Каждый ход (`graph.py::ainvoke`):
1. `resolve_branch(facts)` → ветка (`real_estate` / `refi` / `vehicle` / `partner`).
2. `resolve_focus(facts, branch)` → первый незаполненный слот (`focus_before`).
3. `llm.understand(focus_before, facts, history, user_text)` → `TurnUnderstanding`:
   - `reflection` — тёплое отражение БЕЗ вопроса,
   - `facts_update` — ВСЕ факты из реплики,
   - `answer` — короткий ответ, только если клиент задал встречный вопрос,
   - `branch_signal` — `vehicle/partner/consolidation/none`,
   - `should_end`.
4. `apply_facts` (нормализация, фильтр имени), `branch_signal` → факт (`vehicle_interest`/`partner_interest`/`consolidation_intent`).
5. **Анти-цикл** (см. 4) + deferral мягких слотов.
6. `resolve_focus(new_facts)` → `focus_after`; auto_complete-узлы (открытие/питч/резюме/partner_handoff) проставляют свой гейт при озвучке.
7. `assemble_reply(reflection, answer, focus_after, facts)` = reflection [+ answer] + **детерминированный вопрос** (`question_for`).

Почему так: слабая модель не может застрять, рассинхронизировать состояние, придумать узел или зациклиться — навигация в коде; reflection даёт «человечность».

### Ветки и порядок слотов (`flow.py::_FLOWS`)
- **real_estate:** opening → amount → name(+property) → property_type → region → encumbrance → [offer_refi_or_other | encumbrance_details если в залоге] → owner → [consolidation_summary] → [offer_pts_fallback] → [summary] → pitch → priority → [name_late] → handoff → callback_time.
- **refi** (флаг `refi_mode` в known_facts на старте): refi_opening(«сколько выплатить осталось?») → amount(остаток) → current_payment → refi_term(+property) → [consolidation_summary] → [refi_to_vehicle] → pitch → handoff → callback.
- **vehicle** (branch_signal=vehicle / нет недвижимости): opening → name → vehicle_type → vehicle_owner → [rereg_date если не на клиенте] → vehicle_encumbrance → amount → vehicle_year → credit_history → [summary] → pitch → priority → [name_late] → handoff → callback.
- **partner:** opening → partner_format → name → partner_experience → callback_time → partner_handoff.

Условные слоты: `encumbrance_details`/`offer_refi_or_other` только при залоге; `offer_refi_or_other` НЕ показывается при consolidation_intent; `summary` только для vehicle/consolidation; `callback_time` пропускается при срочности; имя откладывается при возражении (`name_deferred`) и переспрашивается перед хэндофом (`collect_name_late`).

---

## 3. История изменений по сессиям

**Сессия 1 — первичная переделка (исходные жалобы пользователя):**
- Бот «мыслил по одному узлу», переспрашивал уже сказанное, застревал на `node_complete=false`, придумывал несуществующие узлы, ронял разговор в fallback «повторите».
- Решение: выкинули хранимый `current_node`/`node_complete`, перешли на fact-driven `resolve_focus`. Слим-схема, guided_json, graceful fallback (переспрос чистого вопроса вместо «повторите»), ослаблен фильтр имени (клиент Владимир принимается).

**Сессия 2 — «как реальный человек» (по `logs_learn.txt`, 8 эталонных диалогов):**
- Запущен воркфлоу (14 агентов) — разобрал транскрипты, собрал персону/питч/возражения/4 ветки.
- Восстановили богатую LLM-реплику (reflection+answer) поверх детерминированной навигации.
- Добавили ветки refi/vehicle/partner, питч, отработку возражений, отложенное имя.
- Решения пользователя: имя оператора = **Владимир** (НЕ Дмитрий), открытие сразу к сумме, поддержать refi-режим.

**Сессия 3 — фиксы по `resultat_.txt` (реальный прогон на сервере):** см. раздел 4.

---

## 4. Фиксы из прод-лога (`resultat_.txt`) — текущая сессия

| Баг | Причина | Фикс (файл) |
|---|---|---|
| Зацикливание на «в залоге?» (5 раз) | модель не писала факт `encumbrance` → слот не заполнялся | **Анти-цикл** в `graph.py`: да/нет-слот (залог) захватывается с ПЕРВОГО ответа через `infer_gate_value` (flow.py); прочие слоты — 1 уточнение, потом захват/дефер. Гейт `not understanding.answer` (если клиент сам задал вопрос — не захватываем). |
| «Живой оператор, Владимир» | так было в промпте | `system_prompt.txt`: «Вы кто?»→«Владимир, компания МосИнвестФинанс». Открытие (`flow.py::OPENING_COLD/REFI`) теперь представляет компанию. |
| Битый JSON (`answer:""` без кавычек) → fallback | guided_json на сервере не сработал | `llm.py::_repair_json` — чинит bareword-ключи и хвостовые запятые. |
| «можем рассмотреть до 46 млн» (70% от суммы займа!) | расчёт-кэп 70% | Убрано в `flow.py::_pitch_text` и в промпте. «Любую сумму рассматриваем спокойно». |
| Дубли reflection/answer, многословие | промпт | `system_prompt.txt` ужат, правило краткости, «не дублируй вопрос системы». |
| Модель не заполняла нужный ключ | промпт не называл ключ | `prompt.py::EXTRACTION_HINTS` — «СЕЙЧАС НУЖНО ИЗВЛЕЧЬ ФАКТ: encumbrance — …» на каждый focus-слот. |
| Плохо завершает: «да, набирайте завтра» → опять «когда удобнее?» | время названо вместе с согласием, модель отдала только согласие → слот callback_time пуст → переспрос. Финал отбрасывал reflection | `graph.py`: на шаге handoff_consent/callback_time захватываем время через `flow.looks_like_time(user_text)` (scheduling ⇒ consent). `flow.assemble_reply`: финал теперь = reflection + закрытие («завтра, зафиксировал. Спасибо, ожидайте звонка…»). |

**Новое поведение по заданию пользователя (покрыто тестами):**
- **вне залога** → не переспрашивает, сразу к собственнику.
- **под залогом** → узел `offer_refi_or_other`: «можем рассмотреть рефинансирование этого кредита. А есть ещё недвижимость без обременения?».

---

## 5. Тесты и как их гонять

```bash
cd /Users/dr_emin/Desktop/livekit/agent
. .venv-dev/bin/activate          # лёгкий venv (если нет: python3 -m venv .venv-dev && pip install pydantic openai httpx fastapi pytest)
python -m pytest tests/ -q        # 59 passed
```
- `test_flow.py` — навигация по 4 веткам, условные слоты, openings, assemble_reply, encumbrance да/нет.
- `test_runner.py` — оркестрация с фейковым LLM, анти-цикл, deferral, branch_signal.
- `test_api_integration.py` — полный HTTP-путь (реплей багов с сервера).
- `test_apply.py` — фильтр имени, факты. `test_llm.py` — толерантный JSON-парсер.

Богатость реплик НЕ тестируется локально (нужна реальная модель) — проверяется на сервере.

---

## 6. Деплой / пересборка контейнеров

`call_agent` = сервис `text_llm` (`uvicorn text_api:app --host 127.0.0.1 --port 8787`, build `./agent`).

```bash
cd /opt/voicetest
git pull                                              # или scp/rsync кода с Mac
# пересобрать только текстовый API:
docker compose -f docker-compose.prod.yml build text_llm
docker compose -f docker-compose.prod.yml up -d text_llm
docker compose -f docker-compose.prod.yml logs -f text_llm
# проверка:
curl -s http://127.0.0.1:8787/healthz
curl -s http://127.0.0.1:8787/session/start -H 'Content-Type: application/json' -d '{"session_id":"c1","phone":"+79990030000"}'
curl -s http://127.0.0.1:8787/session/message -H 'Content-Type: application/json' -d '{"session_id":"c1","text":"нет, вне обременения"}'
# refi-режим: добавить "known_facts":{"refi_mode":"yes"} в /session/start

# полный стек:
docker compose -f docker-compose.prod.yml -f docker-compose.llm.yml up --build -d
```

**Env (call_agent), в `.env`:** `LLM_MODEL=Qwen/Qwen2.5-7B-Instruct`, `LLM_BASE_URL=http://127.0.0.1:8001/v1`, `LLM_API_KEY=local-token`, `LLM_GUIDED_JSON=1`, `LLM_TEMPERATURE≈0.1`, `LLM_MAX_TOKENS≈320`.

**Чтобы JSON был валиден на источнике:** в `docker-compose.llm.yml` к команде vLLM добавить `--guided-decoding-backend outlines` (или `xgrammar`). Толерантный парсер оставлен как страховка.

---

## 7. Обёртка voice_loop → call_agent — УЖЕ ГОТОВА, включается флагом

**Важно (исправление прошлой записи):** интеграция голоса с текстовым API `call_agent` (8787) **уже написана и закоммичена** на ветке `local-orchestrator-1llm`. НЕ надо ничего строить — надо включить флаг.

Что есть в коде:
- `voice_loop.py::TextApiLlmService` (~стр. 985) — httpx-клиент к 8787: `start_session` / `message` / `reset_session` / `warmup`.
- `voice_loop.py::ParticipantAudioSession._run_text_api_turn` (~стр. 3522) — STT-транскрипт → `POST /session/message` → `reply` → `LlmReply(reply_tts=...)` → TTS; синк теневого состояния (`_text_api_*`).
- Диспетч (~стр. 3807): `if self._uses_text_api_backend(): _run_text_api_turn(...)`.
- `main.py:126`: `if dialogue_backend == "text_api": llm_service = TextApiLlmService(...)` иначе `OpenAiLlmService` (legacy).
- Конфиг: `DIALOGUE_BACKEND` (дефолт `legacy`), `TEXT_API_URL` (дефолт `http://127.0.0.1:8787`).

**Как включить голос на новом движке:**
1. В `.env`: `DIALOGUE_BACKEND=text_api` (уже дефолт в `.env.prod.example`), `TEXT_API_URL=http://127.0.0.1:8787`.
2. Поднять стек: `llm` + `text_llm` (мозг) + `agent` (STT/TTS/LiveKit) + `livekit` + `token_server`. `agent` host-network → дотягивается до 8787.
3. Вписать реальные `LIVEKIT_API_KEY/SECRET/DOMAIN` — без них `agent` не зайдёт в комнату.

Состояние сессии call_agent — в памяти (`api.py::_sessions`); voice_loop гоняет `session_id`+`text`. Рестарт `text_llm` теряет сессии (для коротких звонков ок).

**Остаётся для реальных звонков:** как подаётся аудио в LiveKit-комнату — браузер (frontend + домен + Caddy TLS) или телефон (SIP-транк → LiveKit, в текущем compose НЕТ). Это инфра-вопрос, не код движка.

---

## 8. Открытый вопрос: другой TTS?

Сейчас: **Silero** (`v5_4_ru`, спикер `aidar`, 24кГц) — быстрый, CPU-friendly, бесплатный, но звучит роботично.

Варианты замены (для русского, телефонный бот = важна латентность <300–500мс до первого аудио):

| TTS | Качество | Латентность/ресурсы | Стоимость | Когда брать |
|---|---|---|---|---|
| **Silero (сейчас)** | средне | очень низкая, CPU | free | базовый, минимум ресурсов |
| **Piper (ru_RU)** | средне-выше | низкая, CPU/ONNX | free | лёгкий апгрейд Silero без GPU |
| **XTTS v2 (Coqui)** | высокое, клон голоса | выше, нужен GPU | free | есть GPU, хотим живой голос |
| **Fish-Speech / F5-TTS** | высокое | GPU, средняя | free | новее XTTS, хорошее RU на GPU |
| **Yandex SpeechKit** | очень высокое, телефонное RU | низкая (облако) | платно | прод-звонки, бюджет на облако |
| **ElevenLabs** | топ | средняя (облако) | платно $$ | максимум качества, не критична задержка |

**Рекомендация:**
- Бюджет на облако и нужен телефонный продакшн → **Yandex SpeechKit** (нативный русский, телефонное качество, низкая задержка, есть SSML).
- Только локально + есть GPU → **XTTS v2** или **Fish-Speech**.
- Только CPU, апгрейд без боли → **Piper (ru)**.

TTS меняется в `voice_loop.py` (`SileroTtsService`, ~стр. 1659; `build_tts_service`, ~стр. 1759) + env `TTS_PROVIDER`/`TTS_*`. Архитектурно TTS отвязан от диалога — менять можно независимо от пункта 7.

---

## 9. Сознательные упрощения (не баги)
- Питч — одна тёплая реплика, не прогрессивный по кускам (barge-in живёт в voice_loop).
- Стресс-маркеры TTS (`татьЯна`) — задача голосового слоя, не текстовой логики call_agent.
- Метраж/комнаты/адрес не спрашиваем — города/региона достаточно.
- `graph_spec.py` — легаси, используется только `call_connected.ask` ("Алло."). Можно почистить позже.

---

## 10. Следующие шаги (приоритет)
1. **Завернуть voice_loop на call_agent (8787)** — без этого голос не на новой логике (раздел 7).
2. Решить по TTS (раздел 8) — при желании улучшить звук.
3. Включить `--guided-decoding-backend` в vLLM (раздел 6) — чтобы JSON был валиден на источнике.
4. Прогнать новый прод-лог звонков, собрать остаточные косяки экстракции фактов (промпт `EXTRACTION_HINTS` — главный рычаг донастройки).
