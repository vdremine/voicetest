# gen_strategy_a.py
#
# Стратегия A: дистилляция на РЕАЛЬНЫХ скелетах.
# Берём реальные диалоги (выход extract_finetune.py), сохраняем реальные
# user-турны КАК ЕСТЬ (с ASR-шумом — это реальный вход прода), а ответы
# Дмитрия переписываем сильным учителем (Claude/GPT) по системному промпту.
# Получаем датасет: реальный вход + эталонный таргет.
#
# Установка:
#   pip install anthropic          # для Claude
#   # или: pip install openai      # см. блок OpenAI ниже
#
# Ключ:
#   export ANTHROPIC_API_KEY=sk-ant-...
#
# Запуск (СНАЧАЛА мини-тест на 5 диалогах — проверь качество глазами!):
#   python3 gen_strategy_a.py data.jsonl data.distilled.jsonl --limit 5
#   # ок -> полный прогон:
#   python3 gen_strategy_a.py data.jsonl data.distilled.jsonl

import argparse
import json
import os
import time
from pathlib import Path

# учитель: сильнее ученика (Qwen-14B), иначе дистилляция бессмысленна
TEACHER_MODEL = "claude-sonnet-4-6"   # качественнее/дороже: claude-opus-4-8
TEMPERATURE = 0.6
MAX_TOKENS = 512
MAX_RETRIES = 4

# Доп-инструкция учителю поверх реального system (role/style/plan/facts/rules).
# Держит формат под TTS и заземляет на факты.
TEACHER_GUIDE = """
Ты — Дмитрий из инструкции выше. Сейчас идёт телефонный разговор.
Сгенерируй ТОЛЬКО одну следующую реплику Дмитрия в ответ на последнюю реплику клиента.

Жёсткие требования:
- Устная телефонная речь. Без списков, без markdown, без кавычек вокруг ответа.
- Числа прописью (девятнадцать процентов, а не 19%).
- Используй ТОЛЬКО условия и факты из инструкции выше. Ничего не выдумывай
  (ставки, сроки, суммы — строго как в фактах).
- Отвечай по существу последней реплики клиента, веди по плану разговора.
- Никаких служебных пометок вроде *перебили*, без сценических ремарок.
- Только текст реплики, ничего больше.
""".strip()


def make_client():
    import anthropic
    return anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


def ask_teacher(client, system, api_msgs):
    """Один вызов учителя. api_msgs начинается с user и чередуется."""
    full_system = system + "\n\n" + TEACHER_GUIDE
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.messages.create(
                model=TEACHER_MODEL,
                system=full_system,
                messages=api_msgs,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
            )
            return resp.content[0].text.strip()
        except Exception as e:  # rate limit / сеть — backoff
            wait = 2 ** attempt
            print(f"  ! retry {attempt + 1}/{MAX_RETRIES} через {wait}s: {e}")
            time.sleep(wait)
    return None


def split_system(messages):
    system = ""
    turns = []
    for m in messages:
        if m["role"] == "system":
            system = m["content"]
        else:
            turns.append(m)
    return system, turns


def regenerate(client, messages):
    """Автогрессивно переписывает assistant-турны на реальных user-турнах."""
    system, turns = split_system(messages)
    out_turns = []     # итоговый диалог (включая стартовое приветствие)
    api_msgs = []      # контекст для учителя: начинается с user, чередуется
    for t in turns:
        if t["role"] == "user":
            out_turns.append(t)                     # реальный вход — как есть
            api_msgs.append({"role": "user", "content": t["content"]})
        else:  # assistant
            if not api_msgs:
                # стартовое приветствие до первой реплики клиента — оставляем
                # оригинал (это канонический скрипт из readyAnswers)
                out_turns.append(t)
                continue
            new = ask_teacher(client, system, api_msgs)
            if new is None:
                return None                         # диалог пропускаем
            out_turns.append({"role": "assistant", "content": new})
            api_msgs.append({"role": "assistant", "content": new})
    result = []
    if system:
        result.append({"role": "system", "content": system})
    result.extend(out_turns)
    return result


def main():
    ap = argparse.ArgumentParser(description="Стратегия A: переписать ответы учителем.")
    ap.add_argument("input", help="data.jsonl (реальные диалоги)")
    ap.add_argument("output", help="выход .jsonl")
    ap.add_argument("--limit", type=int, default=None, help="обработать N диалогов (тест)")
    args = ap.parse_args()

    client = make_client()
    rows = [json.loads(l) for l in Path(args.input).read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        rows = rows[: args.limit]

    out, skipped = [], 0
    with Path(args.output).open("w", encoding="utf-8") as f:
        for i, ex in enumerate(rows, 1):
            new = regenerate(client, ex["messages"])
            if new is None:
                skipped += 1
                print(f"[{i}/{len(rows)}] SKIP")
                continue
            row = {"messages": new}
            if "meta" in ex:
                row["meta"] = ex["meta"]
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()                                # стримим — не теряем при обрыве
            out.append(row)
            print(f"[{i}/{len(rows)}] ok ({len(new)} сообщений)")

    print(f"\nГотово: {len(out)} диалогов -> {args.output} (skip: {skipped})")


# ---- OpenAI-вариант (если учитель GPT) ------------------------------------
# Замени make_client/ask_teacher на:
#   from openai import OpenAI
#   client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
#   resp = client.chat.completions.create(
#       model="gpt-4o", temperature=TEMPERATURE, max_tokens=MAX_TOKENS,
#       messages=[{"role":"system","content":full_system}, *api_msgs])
#   return resp.choices[0].message.content.strip()

if __name__ == "__main__":
    main()
