from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# Navigation never depends on the model naming a node (see flow.py). NodeId is
# kept for telemetry / typing only.
NodeId = Literal[
    "call_connected",
    "cold_opening",
    "refi_opening",
    "collect_amount",
    "collect_current_payment",
    "collect_refi_term",
    "collect_name",
    "collect_property_type",
    "collect_region",
    "collect_object_value",
    "collect_encumbrance",
    "collect_encumbrance_details",
    "collect_credit_history",
    "collect_owner",
    "collect_consolidation_summary",
    "offer_pts_fallback",
    "summary_before_pitch",
    "pitch_conditions",
    "priority_choice",
    "handoff_consent",
    "callback_time",
    "collect_vehicle_type",
    "collect_vehicle_owner",
    "collect_vehicle_reregistration_date",
    "collect_vehicle_encumbrance",
    "collect_vehicle_year",
    "refi_to_vehicle",
    "partner_format",
    "partner_experience",
    "partner_handoff",
    "finish",
]

# "consolidation" = client wants to merge/refinance several loans (switches the
# pitch framing). "refi" is NOT here — it is a session flag in known_facts.
BranchSignal = Literal["real_estate", "vehicle", "partner", "consolidation", "none"]


class TurnUnderstanding(BaseModel):
    """Everything the LLM is responsible for on a turn — and nothing more.

    Deliberately small so a weak model can fill it reliably: a warm reflection,
    extracted facts, an optional objection answer, a coarse branch hint, and an
    end flag. The model does NOT choose the next node or word the next question
    — that is deterministic (flow.py).
    """

    reflection: str = Field(
        default="",
        description="Тёплое отражение реплики клиента БЕЗ вопроса. При обрывке/поддакивании — короткий ак.",
    )
    facts_update: dict[str, Any] = Field(
        default_factory=dict,
        description="ВСЕ факты, явно названные клиентом в этой реплике (можно несколько).",
    )
    answer: str = Field(
        default="",
        description="Короткий ответ по сути (1-2 фразы), только если клиент задал встречный вопрос/возражение.",
    )
    branch_signal: BranchSignal = Field(
        default="none",
        description="vehicle — про авто/ПТС; partner — инвестор/партнёр; consolidation — объединить кредиты; иначе none.",
    )
    should_end: bool = Field(
        default=False,
        description="true только при явном прощании/жёстком отказе. Обрыв/тишина — НЕ конец.",
    )


# JSON schema for vLLM guided decoding (guaranteed-valid output on a weak model).
UNDERSTANDING_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "reflection": {"type": "string"},
        "facts_update": {"type": "object"},
        "answer": {"type": "string"},
        "branch_signal": {
            "type": "string",
            "enum": ["real_estate", "vehicle", "partner", "consolidation", "none"],
        },
        "should_end": {"type": "boolean"},
    },
    "required": ["reflection", "facts_update"],
    "additionalProperties": False,
}
