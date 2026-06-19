"""Deterministic dialogue navigation across 4 branches (real_estate / refi /
vehicle / partner), distilled from the 8 gold transcripts.

Contract (unchanged architectural invariant):
  The position in the call is a pure function of accumulated facts. The LLM only
  extracts facts and writes a warm `reflection`/`answer`; this module decides the
  next slot and the question wording. So a weak model cannot desync the state,
  invent a node, or re-ask a filled slot.
"""

from __future__ import annotations

from typing import Callable

# --- fact helpers ---------------------------------------------------------


def _has(facts: dict, key: str) -> bool:
    return bool(str(facts.get(key, "")).strip())


def _val(facts: dict, key: str) -> str:
    return str(facts.get(key, "")).strip().lower().replace("ё", "е")


_NEGATIVE = (
    "нет", "без", "чист", "свободн", "не в залог", "не заложен", "не заклад",
    "вне обремен", "вне залог", "отсутств", "никаких", "no",
)
_REFUSAL = ("нет", "не надо", "не нужно", "не звоните", "откажусь", "не хочу", "не буду")
_CALLBACK_OK = (
    "да", "хорошо", "удобно", "согласен", "давайте", "можно", "ок", "перезвон",
    "позже", "потом", "завтра", "сегодня", "вечер", "утро", "обед", "понедельн",
    "подума", "мягк", "callback",
)


def _encumbrance_is_positive(facts: dict) -> bool:
    value = _val(facts, "encumbrance")
    if not value:
        return False
    return not any(m in value for m in _NEGATIVE)


_ENCUMBRANCE_SLOTS = {"collect_encumbrance", "collect_vehicle_encumbrance"}


_TIME_MARKERS = (
    "сегодня", "завтра", "послезавтра", "вечер", "утро", "утром", "обед",
    "в любое", "любое время", "после обеда", "до обеда", "ближайш",
    "понедельн", "вторник", "сред", "четверг", "пятниц", "суббот", "воскрес",
    "выходн", "на следующей", "час дня", "часов", "сейчас", "прямо",
)


def looks_like_time(text: str) -> bool:
    """True if the utterance names a callback time (so we don't re-ask it)."""
    low = (text or "").lower().replace("ё", "е")
    return any(m in low for m in _TIME_MARKERS)


_QUESTION_WORDS = (
    "кто", "что", "чем", "чего", "как", "почему", "зачем", "сколько", "какой",
    "какая", "какие", "когда", "где", "куда", "откуда", "а вы", "вы что",
    "что за", "о чем", "вы кто", "это что", "а что",
)


def looks_like_question(text: str) -> bool:
    """The client asked something instead of answering — must NOT be captured as
    a slot value (e.g. 'вы кто?' is not a name)."""
    t = (text or "").strip().lower().replace("ё", "е")
    if not t:
        return False
    if t.endswith("?"):
        return True
    padded = f" {t} "
    return any(t.startswith(w) or f" {w} " in padded or f" {w}" == padded[: len(w) + 1] for w in _QUESTION_WORDS)


def is_yes_no_slot(node_id: str) -> bool:
    """Yes/no slots where the client's first statement IS the answer — so the
    runner captures it immediately rather than allowing even one clarifying re-ask."""
    return node_id in _ENCUMBRANCE_SLOTS


def infer_gate_value(node_id: str, text: str) -> str:
    """Best-effort fact value when the model failed to extract it but the client
    clearly responded — used by the runner's anti-loop guard so the same slot is
    never asked a third time."""
    t = (text or "").strip()
    if node_id in _ENCUMBRANCE_SLOTS:
        low = t.lower().replace("ё", "е")
        if any(m in low for m in _NEGATIVE):
            return "нет"
        return t[:80] or "да"
    return t[:80]


def _wants_callback_time(facts: dict) -> bool:
    consent = _val(facts, "callback_consent")
    if not consent:
        return False
    if _val(facts, "urgent") == "yes":
        return False  # urgency -> "перезвонит в ближайшее время", no time-picking
    if any(m in consent for m in _REFUSAL):
        return False
    return any(m in consent for m in _CALLBACK_OK)


def _has_consolidation(facts: dict) -> bool:
    return _has(facts, "consolidation_intent")


def _credit_signal(facts: dict) -> bool:
    """Real-estate credit-history is only asked when something hints at it —
    matching the gold transcripts (clean clients are not interrogated)."""
    return _has(facts, "credit_history_issues") or _encumbrance_is_positive(facts) or _has_consolidation(facts)


def _vehicle_not_declined(facts: dict) -> bool:
    return _val(facts, "vehicle_interest") != "no"


def _enough_for_summary(facts: dict) -> bool:
    keys = ("client_name", "vehicle_type", "vehicle_year", "vehicle_owner", "desired_amount", "vehicle_encumbrance")
    return sum(1 for k in keys if _has(facts, k)) >= 4


# --- step model -----------------------------------------------------------


class Step:
    __slots__ = ("node_id", "gate_fact", "questions", "applies", "auto_complete")

    def __init__(self, node_id, gate_fact, questions, *, applies=None, auto_complete=False):
        self.node_id = node_id
        self.gate_fact = gate_fact
        self.questions: list[str] = questions
        self.applies: Callable[[dict], bool] | None = applies
        self.auto_complete = auto_complete


# Soft slots: if the client dodges them, we don't deadlock — after repeats they
# become skippable via a deferral flag set by the runner.
SOFT_SLOTS = {"collect_amount", "collect_object_value", "collect_credit_history", "collect_current_payment"}

# Deferral flag name per soft slot (collect_amount uses the spec's `amount_deferred`).
DEFER_FLAGS = {
    "collect_amount": "amount_deferred",
    "collect_object_value": "object_value_deferred",
    "collect_credit_history": "credit_history_deferred",
    "collect_current_payment": "current_payment_deferred",
}


def _soft_done(node_id: str, gate: str, facts: dict) -> bool:
    return _has(facts, gate) or _val(facts, DEFER_FLAGS.get(node_id, f"{gate}_deferred")) == "yes"


STEPS: dict[str, Step] = {
    # openings (auto_complete statement nodes; gate set on delivery)
    "cold_opening": Step("cold_opening", "opening_done", [], auto_complete=True),
    "refi_opening": Step("refi_opening", "opening_done", [], auto_complete=True),
    # real_estate / shared
    "collect_amount": Step("collect_amount", "desired_amount", ["Скажите, какую сумму примерно рассматриваете?"]),
    "collect_name": Step(
        "collect_name", "client_name",
        ["А как я могу к вам обращаться?"],
    ),
    # Re-asked just before the expert handoff if the name was deferred because the
    # client went into objections instead of answering (gold Клиент 2 names at the end).
    "collect_name_late": Step(
        "collect_name_late", "client_name",
        ["А как я могу к вам обращаться, чтобы передать эксперту?"],
        applies=lambda f: not _has(f, "client_name"),
    ),
    "collect_property_type": Step(
        "collect_property_type", "property_type",
        ["А какая недвижимость у вас в собственности — квартира, дом, земля, доля?"],
    ),
    "collect_region": Step("collect_region", "region", ["А в каком регионе находится объект?"]),
    "collect_object_value": Step(
        "collect_object_value", "object_value",
        ["А примерно в какую сумму оценивается объект?"],
    ),
    "collect_encumbrance": Step(
        "collect_encumbrance", "encumbrance",
        ["А объект сейчас в залоге где-то — ипотека, банк?"],
    ),
    # If the object IS encumbered: offer refinancing of that loan and ask whether
    # there is another, unencumbered property to consider instead.
    "offer_refi_or_other": Step(
        "offer_refi_or_other", "other_property",
        [
            "Раз объект в залоге — можем рассмотреть рефинансирование этого кредита. "
            "А есть ещё недвижимость без обременения, которую тоже можно рассмотреть?"
        ],
        # Only when the client hasn't already said they want to consolidate/refi
        # (in that case we go straight to confirming the consolidation).
        applies=lambda f: _encumbrance_is_positive(f) and not _has_consolidation(f),
    ),
    "collect_encumbrance_details": Step(
        "collect_encumbrance_details", "encumbrance_details",
        ["А остаток долга примерно какой?"],
        applies=_encumbrance_is_positive,
    ),
    "collect_credit_history": Step(
        "collect_credit_history", "credit_history_issues",
        ["А по кредитной истории — просрочки, исполнительные есть, или в целом нормально?"],
        applies=_credit_signal,
    ),
    "collect_owner": Step(
        "collect_owner", "owner_status",
        ["А кто собственник — только вы или ещё кто-то?"],
    ),
    "collect_consolidation_summary": Step(
        "collect_consolidation_summary", "consolidation_confirmed",
        ["То есть задача — свести всё в один кредит с меньшим платежом, верно?"],
        applies=_has_consolidation,
    ),
    "offer_pts_fallback": Step(
        "offer_pts_fallback", "vehicle_interest",
        ["А автомобиль у вас есть? Можем посмотреть ещё вариант под залог ПТС."],
        applies=lambda f: _has(f, "credit_history_issues") and not _has(f, "vehicle_interest"),
    ),
    "summary_before_pitch": Step(
        "summary_before_pitch", "summary_done",
        ["Если позволите, коротко зарезюмирую, что у нас получается. Всё верно?"],
        # Only summarize complex cases (vehicle / consolidation), like the gold
        # transcripts — a simple real-estate funnel goes owner -> pitch directly.
        applies=lambda f: _enough_for_summary(f) and (_has_consolidation(f) or _has(f, "vehicle_type")),
        auto_complete=True,
    ),
    "pitch_conditions": Step("pitch_conditions", "pitched", [], auto_complete=True),
    "priority_choice": Step(
        "priority_choice", "priority",
        ["А для вас сейчас что важнее — скорость или минимальная ставка?"],
    ),
    "handoff_consent": Step(
        "handoff_consent", "callback_consent",
        ["Удобно, если эксперт свяжется и всё детально рассчитает?"],
    ),
    "callback_time": Step(
        "callback_time", "callback_time",
        ["А когда удобнее — сегодня, завтра, до или после обеда?"],
        applies=_wants_callback_time,
    ),
    # refi
    "collect_current_payment": Step(
        "collect_current_payment", "current_payment",
        ["А текущий платёж сейчас примерно какой?"],
    ),
    "collect_refi_term": Step(
        "collect_refi_term", "refi_term",
        ["А на какой срок брали, и что в залоге — квартира или другое?"],
    ),
    "refi_to_vehicle": Step(
        "refi_to_vehicle", "vehicle_handoff_note",
        ["По машине отдельно тоже посмотрим. А пока — что за автомобиль?"],
        applies=lambda f: _val(f, "vehicle_interest") == "yes",
    ),
    # vehicle
    "collect_vehicle_type": Step("collect_vehicle_type", "vehicle_type", ["А что за автомобиль у вас?"]),
    "collect_vehicle_owner": Step(
        "collect_vehicle_owner", "vehicle_owner",
        ["А машина на вас оформлена или есть ещё собственник?"],
    ),
    "collect_vehicle_reregistration_date": Step(
        "collect_vehicle_reregistration_date", "vehicle_reregistration_date",
        ["А когда переоформляете её на себя?"],
        applies=lambda f: _has(f, "vehicle_owner") and _val(f, "vehicle_owner") not in ("я", "на мне", "моя", "на меня"),
    ),
    "collect_vehicle_encumbrance": Step(
        "collect_vehicle_encumbrance", "vehicle_encumbrance",
        ["А машина сейчас в кредите или под залогом где-то?"],
    ),
    "collect_vehicle_year": Step("collect_vehicle_year", "vehicle_year", ["А какого года машина?"]),
    # partner
    "partner_format": Step(
        "partner_format", "partner_format_desc",
        ["А какой формат сотрудничества вам интересен?"],
    ),
    "partner_experience": Step(
        "partner_experience", "partner_experience",
        ["А вы уже работали в этом направлении, есть опыт?"],
    ),
    "partner_handoff": Step("partner_handoff", "partner_handed", [], auto_complete=True),
}


_FLOWS: dict[str, list[str]] = {
    # Note: real_estate does NOT force a credit-history question — gold transcripts
    # never interrogate clean clients (Клиент 1, 3). credit_history_issues is only
    # captured when the client volunteers it, and then gates offer_pts_fallback.
    # object_value is NOT asked as a step — gold goes region->encumbrance and the
    # value is volunteered later (used for the 70% calc in the pitch when present).
    "real_estate": [
        "cold_opening", "collect_amount", "collect_name", "collect_property_type",
        "collect_region", "collect_encumbrance",
        "offer_refi_or_other", "collect_encumbrance_details", "collect_owner",
        "collect_consolidation_summary", "offer_pts_fallback", "summary_before_pitch",
        "pitch_conditions", "priority_choice", "collect_name_late", "handoff_consent", "callback_time",
    ],
    "refi": [
        "refi_opening", "collect_amount", "collect_current_payment", "collect_refi_term",
        "collect_property_type", "collect_consolidation_summary", "refi_to_vehicle",
        "pitch_conditions", "priority_choice", "handoff_consent", "callback_time",
    ],
    "vehicle": [
        "cold_opening", "collect_name", "collect_vehicle_type", "collect_vehicle_owner",
        "collect_vehicle_reregistration_date", "collect_vehicle_encumbrance",
        "collect_amount", "collect_vehicle_year", "collect_credit_history",
        "summary_before_pitch", "pitch_conditions", "priority_choice",
        "collect_name_late", "handoff_consent", "callback_time",
    ],
    "partner": [
        "cold_opening", "partner_format", "collect_name", "partner_experience",
        "callback_time", "partner_handoff",
    ],
}

# vehicle credit-history is always asked (gold Клиент 5); real_estate only on signal.
_VEHICLE_ALWAYS_CREDIT = True


# --- public navigation ----------------------------------------------------


def resolve_branch(facts: dict) -> str:
    if _val(facts, "refi_mode") in ("yes", "true", "1"):
        return "refi"
    if _has(facts, "partner_interest"):
        return "partner"
    if (
        _val(facts, "property_exists") == "no"
        or _val(facts, "vehicle_interest") == "yes"
        or _has(facts, "vehicle_type")
    ):
        return "vehicle"
    return "real_estate"


def _step_done(node_id: str, facts: dict, branch: str) -> bool:
    step = STEPS[node_id]
    # branch-specific applicability override for credit history
    if node_id == "collect_credit_history" and branch == "vehicle" and _VEHICLE_ALWAYS_CREDIT:
        return _soft_done(node_id, step.gate_fact, facts)
    # In the partner branch the handoff implies consent, so we always ask the time.
    if node_id == "callback_time" and branch == "partner":
        return _has(facts, step.gate_fact)
    # Early name slot is skippable if it was deferred during an objection storm
    # (the late name node re-asks it before handoff).
    if node_id == "collect_name":
        return _has(facts, step.gate_fact) or _val(facts, "name_deferred") == "yes"
    if step.applies is not None and not step.applies(facts):
        return True
    if node_id in SOFT_SLOTS:
        return _soft_done(node_id, step.gate_fact, facts)
    return _has(facts, step.gate_fact)


def resolve_focus(facts: dict, branch: str) -> str:
    for node_id in _FLOWS.get(branch, _FLOWS["real_estate"]):
        if not _step_done(node_id, facts, branch):
            return node_id
    return "finish"


def is_auto_complete(node_id: str) -> bool:
    step = STEPS.get(node_id)
    return bool(step and step.auto_complete)


def gate_fact_for(node_id: str) -> str:
    step = STEPS.get(node_id)
    return step.gate_fact if step else ""


# --- openings -------------------------------------------------------------

OPENING_COLD = (
    "Да, добрый день. Меня зовут Владимир, компания МосИнвестФинанс. "
    "Вы интересовались кредитом под залог недвижимости — давайте подберём условия. "
    "Скажите, какую сумму рассматриваете?"
)
OPENING_REFI = (
    "Да, добрый день. Меня зовут Владимир, компания МосИнвестФинанс. "
    "Вы брали у нас кредит — можем предложить рефинансирование на более выгодных условиях. "
    "Скажите, сколько выплатить осталось?"
)


def is_warm_lead(facts: dict) -> bool:
    """Warm callback: a lead with pre-loaded data (name + amount/object). Set as a
    stable flag at session start so it doesn't flip mid-call."""
    if _val(facts, "refi_mode") in ("yes", "true", "1"):
        return False
    if _val(facts, "lead_mode") == "warm":
        return True
    return _has(facts, "client_name") and (_has(facts, "desired_amount") or _has(facts, "property_type"))


def _greeting_name(facts: dict) -> str:
    name = str(facts.get("client_name", "")).strip()
    patr = str(facts.get("client_patronymic", "")).strip()
    if name and patr and patr.lower() not in name.lower():
        return f"{name} {patr}"
    return name


def opening_for(facts: dict) -> str:
    if _val(facts, "refi_mode") in ("yes", "true", "1"):
        return OPENING_REFI
    if is_warm_lead(facts):
        g = _greeting_name(facts)
        prefix = f"{g}, добрый день." if g else "Добрый день."
        return (
            f"{prefix} Это Владимир, МосИнвестФинанс. Мы вчера общались по кредиту "
            "под залог недвижимости, связь прервалась. Удобно сейчас быстро продолжить?"
        )
    return OPENING_COLD


# --- pitch ----------------------------------------------------------------

_PITCH = (
    "Смотрите, по таким параметрам можно рассматривать кредит под залог. "
    "Сумма — до семидесяти процентов от рыночной стоимости, срок от года до двадцати пяти лет, "
    "ставка от девятнадцати процентов, и официальное трудоустройство не требуется. "
    "Решение даём за один-два дня после документов, и весь процесс вас сопровождает персональный менеджер. "
    "Мы девять лет на рынке и в разных ситуациях находили решение. "
    "{safety} "
    "Точнее уже эксперт рассчитает."
)
_SAFETY_PROPERTY = (
    "Вы при этом остаётесь собственником, никто вас не выписывает, оригиналы документов остаются у вас на руках."
)
_SAFETY_VEHICLE = (
    "При этом машина остаётся у вас, вы ей пользуетесь, и ПТС с документами остаются у вас на руках."
)


def _pitch_text(facts: dict) -> str:
    # Any amount is considered calmly — no 70%-of-the-number cap spoken aloud.
    safety = _SAFETY_VEHICLE if _has(facts, "vehicle_type") else _SAFETY_PROPERTY
    return _PITCH.format(safety=safety)


def _summary_text(facts: dict) -> str:
    parts = []
    label = {
        "client_name": "вас зовут", "desired_amount": "сумма", "property_type": "объект",
        "region": "регион", "object_value": "оценка", "vehicle_type": "авто",
        "vehicle_year": "год", "owner_status": "собственник", "encumbrance": "залог",
    }
    for key, lab in label.items():
        if _has(facts, key):
            parts.append(f"{lab} — {str(facts[key]).strip()}")
    body = ", ".join(parts[:5]) if parts else "всё, что вы рассказали"
    return f"Если позволите, коротко зарезюмирую: {body}. Всё верно?"


# Soft re-ask prefixes — when a slot is asked again, vary the wording so it isn't
# a verbatim repeat ("повторения это зло").
_REASK_PREFIXES = ("Подскажите, ", "Если можно, уточните: ", "Давайте ещё раз — ", "Всё-таки, ")


def _vary_reask(question: str, repeat_count: int) -> str:
    if repeat_count <= 0 or not question:
        return question
    pre = _REASK_PREFIXES[(repeat_count - 1) % len(_REASK_PREFIXES)]
    body = question.lstrip()
    # drop a leading "А " so "Подскажите, а в каком…" reads naturally
    low = body.lower()
    if low.startswith("а "):
        body = body[2:]
    return pre + body[0].lower() + body[1:]


def question_for(node_id: str, facts: dict | None = None, repeat_count: int = 0) -> str:
    facts = facts or {}
    if node_id == "finish":
        return FINISH_SUCCESS
    if node_id == "pitch_conditions":
        return _pitch_text(facts)
    if node_id == "summary_before_pitch":
        return _summary_text(facts)
    if node_id == "partner_handoff":
        return (
            "Отлично, передам вас в направление по партнёрам и инвесторам. "
            "Специалист свяжется и предметно обсудит формат."
        )
    # Combined name + property when both are still missing — only in real_estate
    # (gold Клиент 1). In vehicle/partner we never ask about недвижимость.
    if (
        node_id == "collect_name"
        and not _has(facts, "property_type")
        and resolve_branch(facts) == "real_estate"
    ):
        return _vary_reask("А как вас зовут? И какая недвижимость у вас в собственности?", repeat_count)
    if node_id == "collect_refi_term" and not _has(facts, "property_type"):
        return _vary_reask("А на какой срок брали и что в залоге — квартира или что-то другое?", repeat_count)
    step = STEPS.get(node_id)
    if step is None or not step.questions:
        return ""
    base = step.questions[repeat_count % len(step.questions)]
    return _vary_reask(base, repeat_count)


# --- reply assembly -------------------------------------------------------

# Short finale — long TTS (OmniVoice diffusion) lags on long replies.
FINISH_SUCCESS = "Передаю эксперту, ожидайте звонка. Всего доброго."
FINISH_REFUSAL = "Понял, не отвлекаю. Всего доброго."

# A question mid-reflection right before the finale ("…когда удобнее? Спасибо…")
# is wrong — strip a trailing question clause when finishing.
import re as _re


def _strip_trailing_question(text: str) -> str:
    cleaned = (text or "").strip()
    if not cleaned:
        return ""
    # drop the last clause if it is a question
    parts = _re.split(r"(?<=[.!?…])\s+", cleaned)
    parts = [p for p in parts if not p.strip().endswith("?")]
    return " ".join(parts).strip()


def _join(parts: list[str]) -> str:
    cleaned = [p.strip() for p in parts if p and p.strip()]
    out = ""
    for part in cleaned:
        if not out:
            out = part
            continue
        if out[-1] not in ".!?…":
            out += "."
        out += " " + part
    return out


def assemble_reply(
    *,
    reflection: str,
    answer: str,
    focus_node: str,
    facts: dict | None = None,
    repeat_count: int = 0,
    should_end: bool = False,
    ended_kind: str = "refusal",
) -> str:
    """Deterministic reply: warm reflection/answer (from the model) + the next
    question (from this module). The model never picks the question, so reply and
    state cannot disagree."""
    facts = facts or {}
    if should_end or focus_node == "finish":
        finish = FINISH_SUCCESS if ended_kind == "success" else FINISH_REFUSAL
        # Drop any trailing question ("…когда удобнее?") before the finale so the
        # bot doesn't ask AND say goodbye in one breath.
        refl = _strip_trailing_question(reflection)
        ans = _strip_trailing_question(answer)
        return _join([refl, ans, finish])

    question = question_for(focus_node, facts, repeat_count)
    return _join([reflection, answer, question])
