from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .rag import KnowledgeSnippet


@dataclass(slots=True)
class FaqEntry:
    code: str
    patterns: tuple[str, ...]
    answer: str


@dataclass(slots=True)
class TrainingExample:
    messages: list[dict[str, str]]


class KnowledgeBase:
    _GENERIC_SEQUENCE: tuple[str, ...] = (
        "нужная_сумма",
        "цель",
        "вид_объекта",
        "регион",
        "обременение",
    )

    def __init__(
        self,
        *,
        snippets: list[KnowledgeSnippet],
        faq_entries: list[FaqEntry],
        qualification_sequences: dict[str, list[str]],
        product_synonyms: dict[str, tuple[str, ...]],
        lead_fields: tuple[str, ...],
        truth_rules: tuple[str, ...],
        examples: list[TrainingExample],
    ) -> None:
        self._snippets = snippets
        self._faq_entries = faq_entries
        self._qualification_sequences = qualification_sequences
        self._product_synonyms = product_synonyms
        self._lead_fields = lead_fields
        self._truth_rules = truth_rules
        self._examples = examples

    @classmethod
    def load(cls, data_dir: Path) -> "KnowledgeBase":
        snippets: list[KnowledgeSnippet] = []
        faq_entries: list[FaqEntry] = []
        qualification_sequences: dict[str, list[str]] = {}
        product_synonyms: dict[str, tuple[str, ...]] = {}
        lead_fields: list[str] = []
        truth_rules: list[str] = []
        examples: list[TrainingExample] = []

        product_catalog = _load_json(data_dir / "product_catalog_v2.json")
        for product in product_catalog.get("продукты", []):
            code = str(product.get("код", "")).strip()
            name = str(product.get("наименование", "")).strip()
            synonyms = tuple(str(item).strip().lower() for item in product.get("синонимы", []) if str(item).strip())
            qual_fields = [str(item).strip() for item in product.get("поля_квалификации", []) if str(item).strip()]
            text_parts = [name]
            if product.get("базовые_формулировки"):
                text_parts.extend(str(item).strip() for item in product.get("базовые_формулировки", []))
            if qual_fields:
                text_parts.append(f"Для квалификации нужно: {', '.join(qual_fields)}.")
            keyword_pool = list(synonyms)
            keyword_pool.extend(_keywords_from_text(name))
            keyword_pool.extend(_keywords_from_text(code.replace("_", " ")))
            keyword_pool.extend(_keywords_from_text(" ".join(qual_fields)))
            snippets.append(
                KnowledgeSnippet(
                    key=code or name,
                    text=" ".join(part for part in text_parts if part),
                    keywords=tuple(_dedupe_preserve(keyword_pool)) or (name.lower(),),
                )
            )
            if code:
                product_synonyms[code] = synonyms

        faq_data = _load_json(data_dir / "faq_knowledge_v2.json")
        for item in faq_data.get("элементы", []):
            patterns = tuple(
                str(pattern).strip().lower()
                for pattern in item.get("шаблоны_вопросов", [])
                if str(pattern).strip()
            )
            answer = str(item.get("ответ", "")).strip()
            if patterns and answer:
                faq_entries.append(
                    FaqEntry(
                        code=str(item.get("код", "")).strip(),
                        patterns=patterns,
                        answer=answer,
                    )
                )
                snippets.append(
                    KnowledgeSnippet(
                        key=str(item.get("код", "")).strip() or "faq",
                        text=answer,
                        keywords=tuple(
                            _dedupe_preserve(
                                list(patterns) + _keywords_from_text(" ".join(patterns)) + _keywords_from_text(answer)
                            )
                        ),
                    )
                )

        kb_chunks = _load_json(data_dir / "kb_chunks_v2.json")
        for item in kb_chunks.get("фрагменты", []):
            text = str(item.get("текст", "")).strip()
            if not text:
                continue
            key = str(item.get("код", "")).strip() or str(item.get("категория", "")).strip()
            keywords = tuple(
                _dedupe_preserve(
                    _keywords_from_text(text)
                    + _keywords_from_text(str(item.get("код", "")).replace("_", " "))
                    + _keywords_from_text(str(item.get("категория", "")))
                )[:24]
            )
            snippets.append(KnowledgeSnippet(key=key, text=text, keywords=keywords))

        routing_data = _load_json(data_dir / "routing_rules_v2.json")
        for scenario, fields in routing_data.get("последовательности_квалификации", {}).items():
            field_list = [
                str(field).strip() for field in fields if str(field).strip()
            ]
            qualification_sequences[str(scenario)] = field_list
            if field_list:
                snippets.append(
                    KnowledgeSnippet(
                        key=str(scenario),
                        text=(
                            f"Сценарий {scenario}. "
                            f"Последовательность квалификации: {', '.join(field_list)}."
                        ),
                        keywords=tuple(
                            _dedupe_preserve(
                                _keywords_from_text(str(scenario).replace("_", " "))
                                + _keywords_from_text(" ".join(field_list))
                            )
                        ),
                    )
                )

        lead_schema = _load_json(data_dir / "lead_schema_v2.json")
        for field in lead_schema.get("поля", []):
            name = str(field.get("имя", "")).strip()
            if name:
                lead_fields.append(name)
                snippets.append(
                    KnowledgeSnippet(
                        key=f"lead_field:{name}",
                        text=f"Поле лида: {name}. {str(field.get('описание', '')).strip()}",
                        keywords=tuple(_keywords_from_text(name.replace("_", " "))),
                    )
                )

        risk_rules = _load_json(data_dir / "risk_rules_v2.json")
        for rule in risk_rules.get("жесткие_ограничения", []):
            text = str(rule.get("правило_ответа", "")).strip()
            if text:
                snippets.append(
                    KnowledgeSnippet(
                        key=str(rule.get("код", "")).strip(),
                        text=text,
                        keywords=tuple(
                            _dedupe_preserve(
                                _keywords_from_text(str(rule.get("условие", "")))
                                + _keywords_from_text(text)
                            )[:18]
                        ),
                    )
                )
        for rule in risk_rules.get("мягкие_ограничения", []):
            text = str(rule.get("правило_ответа", "")).strip()
            if text:
                snippets.append(
                    KnowledgeSnippet(
                        key=str(rule.get("код", "")).strip(),
                        text=text,
                        keywords=tuple(
                            _dedupe_preserve(
                                _keywords_from_text(str(rule.get("условие", "")))
                                + _keywords_from_text(text)
                            )[:18]
                        ),
                    )
                )
        for rule in risk_rules.get("правила_достоверности", []):
            text = str(rule).strip()
            if text:
                truth_rules.append(text)

        train_path = data_dir / "train_yandex_v2.jsonl"
        if train_path.is_file():
            example_index = 0
            for raw_line in train_path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except Exception:
                    continue
                items = payload.get("сообщения", [])
                messages: list[dict[str, str]] = []
                for item in items:
                    role_map = {"система": "system", "пользователь": "user", "помощник": "assistant"}
                    role = role_map.get(str(item.get("роль", "")).strip().lower())
                    text = str(item.get("текст", "")).strip()
                    if role and text:
                        messages.append({"role": role, "content": text})
                if messages:
                    example_index += 1
                    examples.append(TrainingExample(messages=messages))
                    joined = " ".join(f"{msg['role']}: {msg['content']}" for msg in messages)
                    snippets.append(
                        KnowledgeSnippet(
                            key=f"train_example:{example_index}",
                            text=joined,
                            keywords=tuple(_keywords_from_text(joined)[:36]),
                        )
                    )

        return cls(
            snippets=snippets,
            faq_entries=faq_entries,
            qualification_sequences=qualification_sequences,
            product_synonyms=product_synonyms,
            lead_fields=tuple(lead_fields),
            truth_rules=tuple(truth_rules),
            examples=examples,
        )

    @classmethod
    def default(cls) -> "KnowledgeBase":
        return cls(
            snippets=[],
            faq_entries=[],
            qualification_sequences={},
            product_synonyms={},
            lead_fields=(),
            truth_rules=(),
            examples=[],
        )

    def retrieve(self, query: str, state: dict[str, Any], *, limit: int = 4) -> list[KnowledgeSnippet]:
        known_facts = state.get("known_facts", {})
        if not isinstance(known_facts, dict):
            known_facts = {}
        query_text = _normalize_text(query)
        summary_text = _normalize_text(str(state.get("summary", "")))
        scenario = _normalize_text(str(state.get("scenario", "")))
        stage = _normalize_text(str(state.get("stage", "")))
        awaiting_field = _normalize_text(str(state.get("awaiting_field", "")))
        facts_text = _normalize_text(" ".join(str(value) for value in known_facts.values()))
        haystack = " ".join(part for part in (query_text, summary_text, scenario, stage, awaiting_field, facts_text) if part)
        query_tokens = set(_keywords_from_text(query_text))
        state_tokens = set(_keywords_from_text(" ".join(part for part in (summary_text, scenario, awaiting_field, facts_text) if part)))
        scored: list[tuple[int, KnowledgeSnippet]] = []
        for snippet in self._snippets:
            snippet_text = _normalize_text(snippet.text)
            snippet_keywords = tuple(_dedupe_preserve(list(snippet.keywords) + _keywords_from_text(snippet_text)))
            if not snippet_keywords and not snippet_text:
                continue
            score = 0
            query_overlap = sum(
                1 for token in query_tokens if token in snippet_text or token in snippet_keywords
            )
            contextual_overlap = sum(
                1 for token in state_tokens if token in snippet_text or token in snippet_keywords
            )
            if scenario:
                scenario_hint = scenario.replace("_", " ")
                if query_overlap > 0 and scenario_hint and (
                    scenario_hint in snippet_text or scenario_hint in _normalize_text(snippet.key)
                ):
                    score += 4
            if awaiting_field:
                awaiting_hint = awaiting_field.replace("_", " ")
                if query_overlap > 0 and awaiting_hint and (
                    awaiting_hint in snippet_text or awaiting_hint in " ".join(snippet_keywords)
                ):
                    score += 3
            for keyword in snippet_keywords:
                if not keyword or len(keyword) < 3:
                    continue
                if " " in keyword:
                    if keyword in query_text:
                        score += 10
                    elif keyword in haystack:
                        score += 4
                else:
                    if keyword in query_tokens:
                        score += 5
                    elif keyword in state_tokens:
                        score += 2
                    elif keyword in query_text and keyword in snippet_text:
                        score += 3
            score += query_overlap * 3
            score += min(2, contextual_overlap)
            if query_text and len(query_text) >= 12 and query_text in snippet_text:
                score += 12
            if awaiting_field and awaiting_field in snippet.key:
                score += 3
            if score > 0:
                scored.append((score, snippet))
        scored.sort(key=lambda item: (item[0], len(item[1].keywords), len(item[1].text)), reverse=True)
        return [snippet for _, snippet in scored[:limit]]

    def match_faq(self, text: str) -> str:
        lowered = _normalize_text(text)
        for entry in self._faq_entries:
            normalized_patterns = tuple(_normalize_text(pattern) for pattern in entry.patterns)
            if any(pattern and pattern in lowered for pattern in normalized_patterns):
                return entry.answer
            if entry.code == "статус_компании" and any(
                marker in lowered
                for marker in (
                    "о компании",
                    "что за компания",
                    "расскажи о компании",
                    "расскажите о компании",
                    "чем занимаетесь",
                    "кто вы такие",
                )
            ):
                return entry.answer
        return ""

    def next_required_field(self, state: dict[str, Any]) -> str:
        scenario = str(state.get("scenario", "")).strip()
        known_facts = state.get("known_facts", {})
        if not isinstance(known_facts, dict):
            known_facts = {}
        sequence = self._qualification_sequences.get(scenario, [])
        if not sequence:
            sequence = list(self._GENERIC_SEQUENCE)
        for field in sequence:
            value = str(known_facts.get(field, "")).strip()
            if not value:
                return field
        return ""

    def question_for_field(self, field: str) -> str:
        prompts = {
            "вид_объекта": "Подскажите, пожалуйста, какой объект рассматриваете: квартира, дом, земля или что-то другое?",
            "регион": "Подскажите, пожалуйста, в каком регионе находится объект?",
            "оценка_стоимости": "Подскажите, пожалуйста, какая примерная стоимость объекта?",
            "нужная_сумма": "Подскажите, пожалуйста, какая сумма вам нужна?",
            "обременение": "Подскажите, пожалуйста, объект уже в залоге, ипотеке или без обременения?",
            "собственники": "Подскажите, пожалуйста, собственник один или их несколько?",
            "детские_доли": "Подскажите, пожалуйста, есть ли детские доли?",
            "материнский_капитал": "Подскажите, пожалуйста, объект связан с материнским капиталом или нет?",
            "цель": "Подскажите, пожалуйста, на какую цель вам нужна сумма?",
            "марка_и_модель": "Подскажите, пожалуйста, какая марка и модель автомобиля?",
            "год": "Подскажите, пожалуйста, какого года автомобиль?",
            "остаток_долга": "Подскажите, пожалуйста, какой сейчас остаток долга?",
            "ежемесячный_платеж": "Подскажите, пожалуйста, какой у вас текущий ежемесячный платеж?",
            "нужна_ли_дополнительная_сумма": "Подскажите, пожалуйста, нужна только сумма на закрытие или еще деньги сверху?",
            "описание_кредитной_истории": "Подскажите, пожалуйста, какая сейчас ситуация по кредитной истории?",
            "размер_просрочек": "Подскажите, пожалуйста, какой сейчас размер просрочек?",
            "тип_клиента": "Подскажите, пожалуйста, вы как физлицо, ИП или организация?",
            "название_организации_или_статус_ип": "Подскажите, пожалуйста, вы ИП или юридическое лицо?",
        }
        return prompts.get(field, "Подскажите, пожалуйста, уточняющую деталь по вашему запросу.")

    def relevant_examples(self, query: str, *, limit: int = 2) -> list[list[dict[str, str]]]:
        lowered = _normalize_text(query)
        query_tokens = set(_keywords_from_text(lowered))
        ranked: list[tuple[int, list[dict[str, str]]]] = []
        for example in self._examples:
            text = _normalize_text(" ".join(item["content"] for item in example.messages))
            score = 0
            for token in ("рефинанс", "плохая", "доли", "материн", "отмените", "не звоните", "таунхаус"):
                if token in lowered and token in text:
                    score += 2
            for word in query_tokens:
                if word in text:
                    score += 1
            if score > 0:
                ranked.append((score, example.messages))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return [messages for _, messages in ranked[:limit]]

    @property
    def truth_rules(self) -> tuple[str, ...]:
        return self._truth_rules

    @property
    def lead_fields(self) -> tuple[str, ...]:
        return self._lead_fields


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _keywords_from_text(text: str) -> list[str]:
    words = [word.strip() for word in re.split(r"[^0-9a-zа-яё_]+", _normalize_text(text)) if len(word.strip()) >= 3]
    deduped: list[str] = []
    seen: set[str] = set()
    for word in words:
        if word in seen:
            continue
        seen.add(word)
        deduped.append(word)
    return deduped


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().replace("ё", "е")).strip()


def _dedupe_preserve(items: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        value = _normalize_text(item)
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result
