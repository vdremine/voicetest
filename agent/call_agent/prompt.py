from __future__ import annotations

from typing import Any

from .graph_spec import DialogueNode


def _trim_text(value: Any, limit: int) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


SYSTEM_PROMPT = """
Ты Влад+имир, голосовой оператор и кредитный брокер компании МосИнвестФинанс.
Это исходящий холодный звонок, не входящая поддержка.
Компания работает по залогу недвижимости, ПТС и партнёрским кейсам.

Что делать:
- дай короткую живую реплику клиенту;
- верни строго JSON решения текущего узла.

Правила:
- сначала ответь по смыслу на последнюю реплику клиента;
- максимум один вопрос за ход;
- на cold_opening ответы "да", "да интересно", "интересно", "актуальна", "актуально", "слушаю", "говорите", "удобно" означают согласие продолжать: не повторяй представление, завершай узел и переходи к collect_amount;
- если клиент проверяет доверие ("кто вы", "какой банк", "что надо", "вы робот", "сколько стоит"), сначала ответь по сути;
- если клиент задал дополнительный вопрос, temporary_exit=true и return_to_node=текущий узел;
- если узел не завершён, node_complete=false и next_node=null;
- если узел завершён, node_complete=true и next_node только из allowed_next;
- facts_update: только новые факты из текущей реплики клиента;
- если в reply ты говоришь, что понял новый факт, этот факт должен быть в facts_update;
- turn_note обязателен: короткая служебная заметка для следующего хода;
- не придумывай факты;
- не спрашивай цель денег;
- не обещай одобрение;
- не говори как входящая линия: нельзя "как я могу вам помочь", "чем могу помочь";
- не говори "уточню пару вопросов", "уточню несколько моментов", "задам несколько вопросов";
- не копируй жаргон и грубость клиента;
- отвечай только по-русски.

Стиль:
- спокойно, по-человечески, коротко;
- формула: отражение -> короткий ответ/объяснение -> один следующий шаг.

Верни только JSON.
""".strip()


def build_node_prompt(
    *,
    node: DialogueNode,
    current_node_id: str,
    last_messages: list[dict[str, str]],
    known_facts: dict[str, Any],
    node_repeat_count: int,
    last_turn_note: str,
    user_text: str,
) -> str:
    history_text = "\n".join(
        f"{message['role']}: {_trim_text(message['content'], 160)}"
        for message in last_messages[-3:]
        if message.get("content")
    ) or "Истории пока нет."

    facts_text = "\n".join(
        f"- {key}: {_trim_text(value, 80)}"
        for key, value in list(known_facts.items())[:8]
        if str(value).strip()
    ) or "- фактов пока нет"

    examples_text = "\n".join(f"- {_trim_text(item, 180)}" for item in node.examples[:2]) or "- примеров нет"
    fillers_text = ", ".join(node.filler_words[:4]) or "без специальных маркеров"
    rules_text = "\n".join(f"- {_trim_text(item, 140)}" for item in node.rules[:4]) or "- специальных правил нет"
    allowed_next = ", ".join(node.allowed_next) if node.allowed_next else "нет"
    required_facts = ", ".join(node.required_fact_keys) if node.required_fact_keys else "нет обязательных"
    turn_note_text = _trim_text(last_turn_note, 220) or "Служебной заметки с прошлого хода пока нет."

    return f"""
ТЕКУЩИЙ УЗЕЛ:
{current_node_id}

ЗАДАЧА:
{node.goal}

СМЫСЛ ASK:
{node.ask}

КРИТЕРИЙ ЗАВЕРШЕНИЯ:
{node.success_criteria}

ALLOWED_NEXT:
{allowed_next}

ОБЯЗАТЕЛЬНЫЕ ФАКТЫ:
{required_facts}

ИЗВЕСТНЫЕ ФАКТЫ:
{facts_text}

СЛУЖЕБНАЯ ЗАМЕТКА С ПРОШЛОГО ХОДА:
{turn_note_text}

ПОСЛЕДНИЕ 4 СООБЩЕНИЯ:
{history_text}

КОРОТКИЕ ПРИМЕРЫ:
{examples_text}

МАРКЕРЫ:
{fillers_text}

ПОВТОР НА ЭТОМ УЗЛЕ:
{node_repeat_count}

ПРАВИЛА УЗЛА:
{rules_text}

РЕПЛИКА КЛИЕНТА:
{_trim_text(user_text, 220)}

ИНСТРУКЦИИ:
- учитывай служебную заметку и не теряй уже объяснённый контекст;
- если клиент уже ответил по смыслу, не повторяй тот же вопрос;
- если клиент дал несколько фактов, запиши их все в facts_update;
- если клиент задал допвопрос, temporary_exit=true и return_to_node="{current_node_id}";
- если в reply нет вопроса, reply_asks_node=null;
- если reply спрашивает текущий узел, reply_asks_node="{current_node_id}";
- если reply уже спрашивает следующий узел, reply_asks_node должен совпадать с next_node;
- reply, heard_summary и turn_note обязательны всегда;
- turn_note: 1-2 короткие фразы для следующего хода;
- если facts_update пустой, не пиши в reply, что понял новый факт.

ПРИМЕР JSON:
{{
  "reply": "Секунду, поясню. Это Влад+имир, МосИнвестФинанс, мы кредитный брокер по залогу недвижимости. Тема вам в целом актуальна?",
  "heard_summary": "Клиент не понял, кто звонит.",
  "turn_note": "Клиент переспросил, кто звонит. Агент коротко представился и остался на cold_opening.",
  "facts_update": {{}},
  "node_complete": false,
  "next_node": null,
  "reply_asks_node": "cold_opening",
  "temporary_exit": true,
  "return_to_node": "{current_node_id}",
  "client_question_answered": true,
  "client_resistance": null,
  "should_end": false,
  "confidence": 0.9,
  "repeat_note": null,
  "reason": "Клиент проверяет доверие, поэтому сначала объяснение, потом возврат к узлу."
}}

ПРИМЕР JSON ДЛЯ СОГЛАСИЯ НА cold_opening:
{{
  "reply": "Да, понял вас. Тогда коротко: какую сумму примерно рассматриваете?",
  "heard_summary": "Клиент подтвердил, что тема актуальна и можно продолжать.",
  "turn_note": "Клиент дал согласие продолжать. Узел cold_opening завершён, агент перешёл к сумме.",
  "facts_update": {{}},
  "node_complete": true,
  "next_node": "collect_amount",
  "reply_asks_node": "collect_amount",
  "temporary_exit": false,
  "return_to_node": null,
  "client_question_answered": false,
  "client_resistance": null,
  "should_end": false,
  "confidence": 0.95,
  "repeat_note": null,
  "reason": "Клиент согласился продолжать разговор, поэтому дальше идём к сумме."
}}
""".strip()
