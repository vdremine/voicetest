# Дообученная модель «Дмитрий» → LiveKit (статус и инструкции)

Дообучение Qwen2.5-14B (LoRA) на 2810 реальных звонках колл-центра и интеграция
в твой LiveKit-стек: vLLM (OpenAI API) → `call_agent` (text_api) → STT/TTS.

Модель = твоя же база `Qwen2.5-14B-Instruct` (прод крутит её AWQ-вариант), поэтому
дообученный адаптер ложится в стек без смены архитектуры.

---

## 1. Что сделано (итог сессии)

1. **Данные:** лог вебхуков → чистый JSONL. 4481 записей → дедуп → **2810 уникальных**
   диалогов (1622 дубля-ретрая выкинуты). Принцип «LLM перед TTS»: user-турны = сырой
   ASR, assistant-турны чистим, ударения (`одОбрить`) и числа прописью сохраняем.
   Скрипт: `extract_finetune.py`.
2. **Обучение:** QLoRA на Qwen2.5-14B (rank 32, 2 эпохи, train_on_responses_only).
   Результат: train loss 2.25→0.77, **eval_loss 0.83→0.81** (переобучения нет).
   Скрипт: `train_lora_qwen.py`. Адаптер: 550 МБ.
3. **Деплой:** LoRA слита в fp16 (`export_awq.py merge`) → vLLM отдаёт OpenAI API.
4. **Интеграция:** `call_agent` (text_api) подключён к vLLM, извлекает факты,
   генерирует реплики Дмитрия, ведёт по графу. **Проверено end-to-end.**

Пример живого ответа модели:
> «Сумма — до семидесяти процентов от рыночной стоимости квартиры. Срок — от одного
> года до двадцати пяти лет. Ставка — от девятнадцати процентов годовых. Официальное
> трудоустройство не требуется.»

---

## 2. Текущее состояние сервера (GPU-бокс)

**Хост:** `root@45.157.161.161` (ключ `~/.ssh/vps_new`, `-o IdentitiesOnly=yes`)
**Железо:** RTX 4090 **48 ГБ**, 12 CPU, 62 ГБ RAM, Ubuntu 24.04, драйвер 580 (CUDA 13).

| Что | Путь / адрес |
|---|---|
| Слитая fp16-модель (в прод через vLLM) | `/root/mif-train/qwen-mif-merged/` (~28 ГБ) |
| LoRA-адаптер | `/root/mif-train/qwen-mif-lora/` (550 МБ) |
| venv для vLLM | `/root/mif-train/.venv-vllm/` (vllm 0.23.0) |
| venv для обучения/merge (unsloth) | `/root/mif-train/.venv-train/` |
| Код call_agent (копия) | `/root/mif-train/agent/call_agent/` |
| **vLLM (OpenAI API)** | `http://127.0.0.1:8001/v1`, модель `mif`, ключ `local-token` |
| **call_agent (text_api)** | `http://127.0.0.1:8787` |

Оба сервиса запущены в `setsid nohup` — переживают обрыв SSH.
Логи на боксе: `vllm_serve.log`, `text_api.log`.

> ⚠️ Диск на боксе ~63 ГБ, занят почти весь (fp16-модель 28 ГБ + два venv). Для AWQ
> или чистоты — докинуть SSD (ресайз требует ребута).

---

## 3. Как поднять заново (на боксе)

```bash
# 1) vLLM с дообученной моделью (OpenAI API :8001)
cd /root/mif-train && bash /path/to/finetune/serve_vllm.sh &
#   — или вручную, ВАЖНО env против flashinfer/nvcc (см. serve_vllm.sh):
#   VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_USE_FLASHINFER=0 VLLM_ATTENTION_BACKEND=FLASH_ATTN \
#     .venv-vllm/bin/vllm serve qwen-mif-merged --served-model-name mif \
#     --host 127.0.0.1 --port 8001 --api-key local-token \
#     --max-model-len 8192 --max-num-seqs 1 --gpu-memory-utilization 0.90 \
#     --enforce-eager --trust-remote-code

# 2) call_agent text_api (:8787), ходит в vLLM
bash /path/to/finetune/serve_text_api.sh &
```

`serve_vllm.sh` / `serve_text_api.sh` лежат в этой папке — копируй на бокс или
запускай по абсолютному пути.

---

## 4. ВАЖНО: фикс `call_agent/llm.py` (уже внесён)

**Проблема:** `call_agent` слал структурный запрос через `extra_body={"guided_json": ...}`
— это **старый API vLLM**. В vLLM **0.23 он молча игнорируется** (возвращает 200 + свободный
текст), из-за чего понимание не парсилось → диалог сваливался в скриптовый фолбэк, а
дообученная модель фактически не использовалась.

**Фикс (внесён в `agent/call_agent/llm.py`, метод `_create_completion`):**
вместо `guided_json` теперь `response_format: json_schema` (новый API vLLM, xgrammar
принуждает схему `TurnUnderstanding`). Фолбэк на `json_object` сохранён.

```python
# было:  kwargs["extra_body"] = {"guided_json": UNDERSTANDING_JSON_SCHEMA}
# стало:
kwargs["response_format"] = {
    "type": "json_schema",
    "json_schema": {"name": "turn_understanding", "schema": UNDERSTANDING_JSON_SCHEMA},
}
```

После фикса `LLM_GUIDED_JSON=1` (дефолт) работает корректно. Без фикса — обходись
`LLM_GUIDED_JSON=0` (режим `json_object`, vLLM его тоже принуждает).

---

## 5. Конфиг для `.env.prod` (под дообученную модель)

```ini
LLM_MODEL=mif                          # имя в vLLM (--served-model-name mif)
LLM_BASE_URL=http://127.0.0.1:8001/v1
LLM_API_KEY=local-token
LLM_GUIDED_JSON=1                       # после фикса llm.py; иначе 0
# модель делит карту со STT/TTS — снизь утилизацию, как и было:
LLM_GPU_MEMORY_UTILIZATION=0.55
LLM_MAX_MODEL_LEN=8192
LLM_MAX_NUM_SEQS=1
```

Остальное (STT whisper/gigaam, TTS omnivoice/piper) — без изменений.

---

## 6. Проверка (с бокса)

```bash
# прямой LLM
curl -s http://127.0.0.1:8001/v1/chat/completions \
  -H 'Authorization: Bearer local-token' -H 'Content-Type: application/json' \
  -d '{"model":"mif","messages":[{"role":"user","content":"Условия займа под квартиру?"}],"max_tokens":120}'

# полный диалог через call_agent
curl -s http://127.0.0.1:8787/session/start   -H 'Content-Type: application/json' \
  -d '{"session_id":"call-001","phone":"+79990000000"}'
curl -s http://127.0.0.1:8787/session/message -H 'Content-Type: application/json' \
  -d '{"session_id":"call-001","text":"мне нужно два миллиона под квартиру в москве"}'
```
Признак, что используется модель (а не скрипт): в ответе `trace.source = "main"` и
непустой `known_facts`/`llm_decision`. Первый ход после `start` — скриптовое
приветствие (`source: ready_intro`), это нормально.

---

## 7. Репродукция пайплайна (скрипты в этой папке)

| Файл | Назначение |
|---|---|
| `extract_finetune.py` | лог → чистый JSONL (`--clean`, дедуп, `--val-split`) |
| `train_lora_qwen.py` | QLoRA-обучение Qwen2.5-14B (Unsloth) |
| `export_awq.py` | `merge` (LoRA→fp16) и `awq` (квантизация под прод) |
| `serve_vllm.sh` | запуск vLLM с дообученной моделью (обход flashinfer) |
| `serve_text_api.sh` | запуск call_agent поверх vLLM |
| `run_train.sh` / `deploy_train.sh` | обучение на боксе / пульт деплоя с мака |
| `gen_strategy_a.py` | (опц.) синтетика-дистилляция на реальных скелетах |
| `serve_openai.py` | (запасной) лёгкий OpenAI-сервер на transformers без vLLM |
| `PIPELINE.md` | подробная методичка по данным/обучению |

Команда подготовки данных (режим LLM перед TTS — без `--normalize-stress`):
```bash
python3 extract_finetune.py requests.log data.jsonl --clean --val-split 0.1
```

---

## 8. Известные нюансы / TODO

- **flashinfer/скорость.** Сейчас vLLM без flashinfer (нет nvcc). Поставишь
  `cuda-toolkit` (nvcc) — вернёшь flashinfer + CUDA-graphs → быстрее инференс.
  Сейчас ~4–5 с на реплику (enforce-eager). Для звонков терпимо, но можно ускорить.
- **AWQ под прод.** Сейчас отдаём fp16 (28 ГБ VRAM). Если карта делит ресурсы со
  STT/TTS впритык — собрать AWQ (`export_awq.py awq`, ~10 ГБ). Нужен +SSD и autoawq.
- **Структурный вывод vs стиль.** Файнтюн учил свободную речь; в call_agent она
  попадает в поле `answer` JSON под принуждением схемы. Если захочешь, чтобы модель
  «думала» в формате `TurnUnderstanding` нативнее — отдельный заход: дообучить на
  датасете в JSON-формате understanding (не свободный текст).
- **PII.** Датасет содержит имена/суммы/телефоны. Перед любой выгрузкой наружу —
  прогнать через санитайзер (`a.py` в исходной папке логов). Адаптер/веса — ок.
- **Дедуп train/val.** Дедуп идёт ДО сплита (нет протечки). См. `extract_finetune.py`.

---

## 9. Операционка (шпаргалка)

```bash
# подключиться
ssh -i ~/.ssh/vps_new -o IdentitiesOnly=yes root@45.157.161.161

# что слушает
ss -ltnp | grep -E ':8001|:8787'
# GPU
nvidia-smi
# рестарт сервиса (пример vLLM): убить порт и поднять заново
fuser -k 8001/tcp; cd /root/mif-train && bash serve_vllm.sh > vllm_serve.log 2>&1 &
# логи
tail -f /root/mif-train/vllm_serve.log
tail -f /root/mif-train/text_api.log
```
