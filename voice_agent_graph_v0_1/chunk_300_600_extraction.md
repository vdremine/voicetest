# Chunk 300-600 Extraction Map

Источник: [request_sanitized2.txt](/Users/dr_emin/Desktop/livekit/voice_agent_graph_v0_1/request_sanitized2.txt)

Цель этого файла:
- не парсить весь лог сразу;
- фиксировать, как разбирать диапазон строк `300-600`;
- вытащить полезные сущности для graph/state/dataset;
- отбрасывать шум webhook-обвязки, который не нужен для оркестрации.

## 1. Формат блока

Внутри диапазона `300-600` встречаются полные webhook-блоки такого вида:

```text
========| DD.MM.YYYY HH:MM:SS |========
REFERER:
INPUT: {...}
REQUEST: []
HEADERS: {...}
=======================================
```

Базовая единица парсинга:
- один лог-блок между `========|` и `=======================================`

## 2. Что брать обязательно

Из `INPUT`:
- `timestamp`
- `callList.name`
- `callList.skillbase.sellerRole`
- `callList.skillbase.sellerStyle`
- `callList.skillbase.readyAnswers`
- `callList.skillbase.plan`
- `callList.skillbase.facts`
- `callList.skillbase.rules`
- `contact.phone`
- `contact.dadataPhoneInfo.region`
- `contact.additionalFields.source`
- `call.status`
- `call.duration`
- `call.recordUrl`
- `call.callDetails.provider`
- `call.callDetails.chatHistory`
- `agreements`
- `metrics`

## 3. Что брать как derived fields

Из `chatHistory` и `agreements` надо собирать:
- `scenario_guess`
- `client_name`
- `amount`
- `object_type`
- `region`
- `encumbrance`
- `owners`
- `goal`
- `urgency`
- `employment_or_income_style`
- `handoff_result`
- `lead_quality`

## 4. Что игнорировать

Не нужно тянуть в рабочий граф:
- все `organization.id`, `createdBy`, `responsibleUser`
- почти все `createdAt` / `updatedAt` во вложенных сущностях
- `originateData`, `originateResponse`
- `channelId`, `platformCallId`
- `X-Webhook-*` заголовки
- весь `HEADERS`, кроме случаев отдельного аудита интеграции

## 5. Нормализованный target schema

```json
{
  "chunk_range": "300-600",
  "blocks": [
    {
      "log_timestamp": "21.01.2026 12:22:53",
      "source_type": "lead_only",
      "skillbase": {
        "role": "...",
        "style": ["..."],
        "ready_answers": ["..."],
        "plan_steps": ["..."],
        "facts": ["..."],
        "rules": ["..."]
      },
      "contact": {
        "region": "...",
        "source": "...",
        "phone_masked": "[PHONE]"
      },
      "call": {
        "provider": "...",
        "status": "completed",
        "duration_ms": 0,
        "chat_history": []
      },
      "derived": {
        "scenario_guess": "...",
        "client_name": "...",
        "slots": {},
        "handoff_result": "...",
        "lead_quality": 0
      },
      "agreements": {},
      "metrics": {}
    }
  ]
}
```

## 6. Slot map для оркестрации

Обязательные слоты:
- `amount`
- `object_type`
- `region`
- `encumbrance`
- `owner_count_or_owner_type`

Контекстные слоты:
- `property_value_estimate`
- `not_only_home`
- `arrears_other`
- `goal`
- `current_creditor`
- `monthly_payment`
- `preferred_contact_channel`
- `callback_time`

Итоговые системные поля:
- `handoff_needed`
- `handoff_target`
- `sms_needed`
- `lead_quality`

## 7. Найденные кейсы в строках 300-600

### Case A. "Не для себя, а для подруги"

Наблюдаемые факты:
- сценарий: посредник / третье лицо
- объект: квартира
- регион: Калининград
- район: Ленинградский
- залога нет
- собственник: подруга сама
- жилье не единственное
- есть задолженность по квартплате
- оценка стоимости: `7-8 млн`
- цель: продать/переложиться в новостройку, высвободить деньги

Что важно для graph:
- нужен переход в ветку `third_party_proxy`
- не надо требовать имя в начале как обязательный слот
- сначала ценнее собрать объект / регион / залог / стоимость

Рекомендуемый `scenario_guess`:
- `proxy_for_friend_with_property`

Рекомендуемые extracted slots:

```json
{
  "amount": null,
  "object_type": "квартира",
  "region": "Калининград",
  "district": "Ленинградский",
  "encumbrance": "none",
  "owner_type": "friend_single_owner",
  "property_value_estimate": "7-8 млн",
  "not_only_home": true,
  "arrears_other": "квартплата 62-64 тыс",
  "goal": "высвободить деньги и вложиться в новое жилье"
}
```

### Case B. "Ольга, 600 тысяч, Омск"

Наблюдаемые факты:
- клиент: Ольга
- сумма: `600 тысяч`
- объект: квартира
- регион: Омск
- обременения нет
- собственники: клиент + дочь
- интерес: кредит без подтверждения дохода
- важны и скорость, и ставка
- результат: передача эксперту

Рекомендуемый `scenario_guess`:
- `new_loan_property_secured`

Рекомендуемые extracted slots:

```json
{
  "client_name": "Ольга",
  "amount": "600000",
  "object_type": "квартира",
  "region": "Омск",
  "encumbrance": "none",
  "owners": "client_plus_daughter",
  "goal": "новый кредит под залог квартиры",
  "income_confirmation_needed": false,
  "priority": "speed_and_rate"
}
```

## 8. Mapping chatHistory -> graph events

Каждую пару `user/assistant` надо потом маппить не только в текст, но и в event layer:

```json
{
  "event_type": "slot_fill",
  "slot": "region",
  "source_turn_text": "город омск",
  "normalized_value": "Омск"
}
```

Поддерживаемые event types:
- `opening`
- `identity_question`
- `slot_fill`
- `slot_correction`
- `proxy_disclosure`
- `objection`
- `product_question`
- `handoff_offer`
- `handoff_accept`
- `callback_request`
- `messenger_request`
- `finish`

## 9. Что важно для следующего чанка

При разборе следующих диапазонов по 300 строк держать тот же порядок:
1. выделить лог-блоки
2. вынуть `skillbase`
3. вынуть `chatHistory`
4. собрать `derived slots`
5. собрать `agreements`
6. собрать `metrics`
7. пометить `scenario_guess`

## 10. Короткий operational rule

Для этого файла правильный pipeline такой:
- сначала `log block extraction`
- потом `chatHistory extraction`
- потом `slot/event normalization`
- потом `graph mapping`

Неправильно:
- пытаться читать весь `INPUT` как единый полезный JSON без фильтрации
- тащить все метаданные webhook в runtime graph
- смешивать `skillbase prompt`, `dialogue trace` и `handoff outcome` в один плоский текст
