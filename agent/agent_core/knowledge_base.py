from __future__ import annotations

import json
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
            snippets.append(
                KnowledgeSnippet(
                    key=code or name,
                    text=" ".join(part for part in text_parts if part),
                    keywords=synonyms or (name.lower(),),
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

        kb_chunks = _load_json(data_dir / "kb_chunks_v2.json")
        for item in kb_chunks.get("фрагменты", []):
            text = str(item.get("текст", "")).strip()
            if not text:
                continue
            key = str(item.get("код", "")).strip() or str(item.get("категория", "")).strip()
            keywords = tuple(_keywords_from_text(text)[:8])
            snippets.append(KnowledgeSnippet(key=key, text=text, keywords=keywords))

        routing_data = _load_json(data_dir / "routing_rules_v2.json")
        for scenario, fields in routing_data.get("последовательности_квалификации", {}).items():
            qualification_sequences[str(scenario)] = [
                str(field).strip() for field in fields if str(field).strip()
            ]

        lead_schema = _load_json(data_dir / "lead_schema_v2.json")
        for field in lead_schema.get("поля", []):
            name = str(field.get("имя", "")).strip()
            if name:
                lead_fields.append(name)

        risk_rules = _load_json(data_dir / "risk_rules_v2.json")
        for rule in risk_rules.get("жесткие_ограничения", []):
            text = str(rule.get("правило_ответа", "")).strip()
            if text:
                snippets.append(
                    KnowledgeSnippet(
                        key=str(rule.get("код", "")).strip(),
                        text=text,
                        keywords=tuple(_keywords_from_text(str(rule.get("условие", "")))[:6]),
                    )
                )
        for rule in risk_rules.get("мягкие_ограничения", []):
            text = str(rule.get("правило_ответа", "")).strip()
            if text:
                snippets.append(
                    KnowledgeSnippet(
                        key=str(rule.get("код", "")).strip(),
                        text=text,
                        keywords=tuple(_keywords_from_text(str(rule.get("условие", "")))[:6]),
                    )
                )
        for rule in risk_rules.get("правила_достоверности", []):
            text = str(rule).strip()
            if text:
                truth_rules.append(text)

        train_path = data_dir / "train_yandex_v2.jsonl"
        if train_path.is_file():
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
                    examples.append(TrainingExample(messages=messages))

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
        haystack = " ".join(
            [
                query.lower(),
                str(state.get("summary", "")).lower(),
                str(state.get("scenario", "")).lower(),
                str(state.get("stage", "")).lower(),
                str(state.get("awaiting_field", "")).lower(),
            ]
        )
        scored: list[tuple[int, KnowledgeSnippet]] = []
        for snippet in self._snippets:
            if not snippet.keywords:
                continue
            score = sum(1 for keyword in snippet.keywords if keyword and keyword in haystack)
            if score > 0:
                scored.append((score, snippet))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [snippet for _, snippet in scored[:limit]]

    def match_faq(self, text: str) -> str:
        lowered = text.lower()
        for entry in self._faq_entries:
            if any(pattern in lowered for pattern in entry.patterns):
                return entry.answer
        return ""

    def next_required_field(self, state: dict[str, Any]) -> str:
        scenario = str(state.get("scenario", "")).strip()
        known_facts = state.get("known_facts", {})
        if not isinstance(known_facts, dict):
            known_facts = {}
        sequence = self._qualification_sequences.get(scenario, [])
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
        lowered = query.lower()
        ranked: list[tuple[int, list[dict[str, str]]]] = []
        for example in self._examples:
            text = " ".join(item["content"].lower() for item in example.messages)
            score = 0
            for token in ("рефинанс", "плохая", "доли", "материн", "отмените", "не звоните", "таунхаус"):
                if token in lowered and token in text:
                    score += 2
            for word in lowered.split():
                if len(word) >= 4 and word in text:
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
    cleaned = (
        text.lower()
        .replace(",", " ")
        .replace(".", " ")
        .replace(":", " ")
        .replace(";", " ")
        .replace("(", " ")
        .replace(")", " ")
    )
    words = [word.strip() for word in cleaned.split() if len(word.strip()) >= 4]
    deduped: list[str] = []
    seen: set[str] = set()
    for word in words:
        if word in seen:
            continue
        seen.add(word)
        deduped.append(word)
    return deduped
