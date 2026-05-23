from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class KnowledgeSnippet:
    key: str
    text: str
    keywords: tuple[str, ...]


class StaticRagIndex:
    def __init__(self, snippets: list[KnowledgeSnippet]) -> None:
        self._snippets = snippets

    @classmethod
    def default(cls) -> "StaticRagIndex":
        return cls(
            [
                KnowledgeSnippet(
                    key="core_products",
                    text=(
                        "Компания работает с кредитом под залог недвижимости, кредитом под залог автомобиля, "
                        "займом под залог ПТС, рефинансированием, кредитом для ИП и ООО, "
                        "потребительским кредитом без подтверждения дохода и ипотекой по двум документам."
                    ),
                    keywords=("кредит", "недвижим", "автомоб", "птс", "рефинанс", "ипотек"),
                ),
                KnowledgeSnippet(
                    key="real_estate_terms",
                    text=(
                        "По недвижимости сумма может быть до 70 процентов от рыночной стоимости, "
                        "срок от одного года до двадцати пяти лет, ставка от 5 процентов годовых. "
                        "Клиент остаётся собственником, оригиналы документов остаются у клиента."
                    ),
                    keywords=("недвижим", "квартир", "дом", "коммерческ"),
                ),
                KnowledgeSnippet(
                    key="payment_support",
                    text=(
                        "Если клиент говорит про отсрочку, порядок оплаты, последний платеж или реквизиты, "
                        "нельзя придумывать сумму и реквизиты. Нужно согласовать звонок персонального менеджера по платежам."
                    ),
                    keywords=("отсроч", "платеж", "оплат", "реквизит", "перевод"),
                ),
                KnowledgeSnippet(
                    key="no_collateral",
                    text=(
                        "Если клиент прямо сказал, что у него нет недвижимости, машины или ПТС, "
                        "нельзя предлагать залог под то, чего у него нет. "
                        "Нужно честно сказать, что вариант надо уточнить у специалиста, либо перейти к другому релевантному продукту."
                    ),
                    keywords=("нет недвижимости", "без недвижимости", "машины нет", "ничего нет", "птс"),
                ),
                KnowledgeSnippet(
                    key="voice_style",
                    text=(
                        "Ответ должен быть коротким, живым и голосовым: одна мысль, один следующий шаг. "
                        "Нельзя выдавать внутренние рассуждения, мета-комментарии и пересказ разговора."
                    ),
                    keywords=("кто", "зачем", "почему", "повтори", "голос"),
                ),
                KnowledgeSnippet(
                    key="callback",
                    text=(
                        "Если клиент просит не сегодня или задаёт удобное окно, нужно подтвердить окно обратного звонка "
                        "и кратко зафиксировать, что менеджер свяжется в это время."
                    ),
                    keywords=("не сегодня", "послезавтра", "после двенадцати", "перезвон", "удобно"),
                ),
            ]
        )

    def retrieve(self, query: str, state: dict[str, Any], *, limit: int = 3) -> list[KnowledgeSnippet]:
        haystack = " ".join(
            [
                query.lower(),
                str(state.get("summary", "")).lower(),
                str(state.get("stage", "")).lower(),
                str(state.get("awaiting_field", "")).lower(),
            ]
        )
        scored: list[tuple[int, KnowledgeSnippet]] = []
        for snippet in self._snippets:
            score = sum(1 for keyword in snippet.keywords if keyword in haystack)
            if score > 0:
                scored.append((score, snippet))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [snippet for _, snippet in scored[:limit]]
