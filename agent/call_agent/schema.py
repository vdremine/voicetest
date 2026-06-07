from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


NodeId = Literal[
    "call_connected",
    "cold_opening",
    "collect_amount",
    "collect_name",
    "collect_property_type",
    "collect_region",
    "collect_encumbrance",
    "collect_encumbrance_details",
    "collect_owner",
    "pitch_conditions",
    "priority_choice",
    "handoff_consent",
    "callback_time",
    "no_real_estate_products",
    "collect_vehicle_type",
    "collect_vehicle_owner",
    "collect_vehicle_encumbrance",
    "partner_format",
    "partner_handoff",
    "finish",
]


class LlmTurnDecision(BaseModel):
    reply: str = Field(..., description="Живая короткая реплика ассистента")
    heard_summary: str = Field(
        default="",
        description="Коротко, что именно модель поняла из текущей реплики клиента",
    )
    facts_update: dict[str, Any] = Field(
        default_factory=dict,
        description="Только факты, явно сказанные клиентом в текущей реплике",
    )
    node_complete: bool = Field(
        default=False,
        description="Выполнена ли задача текущего узла",
    )
    next_node: NodeId | None = Field(
        default=None,
        description="Следующий узел, если текущий узел завершён",
    )
    reply_asks_node: NodeId | None = Field(
        default=None,
        description="Какой узел фактически спрашивает текущая reply-реплика",
    )
    temporary_exit: bool = Field(
        default=False,
        description="true, если клиент задал дополнительный вопрос, и агент временно вышел из узла",
    )
    return_to_node: NodeId | None = Field(
        default=None,
        description="Куда вернуться после ответа на дополнительный вопрос",
    )
    client_question_answered: bool = False
    client_resistance: str | None = None
    should_end: bool = False
    confidence: float = Field(
        default=0.0,
        description="Уверенность модели в своём решении от 0.0 до 1.0",
    )
    repeat_note: str | None = Field(
        default=None,
        description="Как модель изменила ответ, если это повтор на том же узле",
    )
    reason: str = Field(
        default="",
        description="Короткое объяснение решения для debug",
    )
