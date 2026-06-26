# extract_finetune.py
#
# Чистит лог webhook-запросов и оставляет только полезное для fine-tuning.
# Каждая запись лога -> одна строка JSONL вида:
#   {"messages": [{"role":"system",...},{"role":"assistant"|"user",...}, ...],
#    "meta": {...agreements/исход звонка...}}
#
# Из записи берём только:
#   - skillbase (sellerRole, sellerStyle, plan, facts, rules) -> system message
#   - chatHistory -> турны диалога
#   - agreements -> meta (метка исхода, не попадает в обучающие турны)
# Весь служебный мусор (организация, контакт, dadata, HEADERS, подписи,
# UUID, recordUrl, metrics) отбрасывается.
#
# Бесполезные записи (недозвоны / нет реального диалога) пропускаются.
#
# Usage:
#   python3 extract_finetune.py <input.log> [output.jsonl]

import argparse
import json
import random
import re
from pathlib import Path

# ---- настройки фильтрации полезности -------------------------------------

MIN_USER_TURNS = 2      # минимум реальных реплик клиента
MIN_MESSAGES = 4        # минимум сообщений в диалоге после очистки

# реплики клиента, которые не считаем "реальным ответом"
TRIVIAL_USER = {"", "...", "алло", "ало", "алло", "да", "слушаю"}


# ---- очистка артефактов ASR/TTS -------------------------------------------
#
# Принцип: user-турны = вход (в проде модель тоже получает сырой ASR),
# поэтому их НЕ чистим. Assistant-турны = цель генерации, чистим от мусора.

MARKER_RE = re.compile(r"\s*\*[^*]*\*\s*")       # *тут нас перебили*
WS_RE = re.compile(r"[ \t]+")
WORD_RE = re.compile(r"[а-яёА-ЯЁ]+")
# заглавная кириллица НЕ в начале слова (TTS-ударение: одОбрить, ДмИтрий)
INNER_CAP_RE = re.compile(r"(?<=[а-яёА-ЯЁ])([А-ЯЁ])")


def load_morph():
    """Пытается поднять морфологию для склейки разорванных слов."""
    for mod in ("pymorphy3", "pymorphy2"):
        try:
            m = __import__(mod)
            return m.MorphAnalyzer()
        except Exception:
            continue
    return None


def _word_known(morph, w: str) -> bool:
    w = w.strip("-").lower()
    if not w:
        return False
    try:
        return any(p.is_known for p in morph.parse(w))
    except Exception:
        return False


def rejoin_split_words(text: str, morph) -> tuple[str, int]:
    """Склеивает 'ра йон' -> 'район' только если это даёт реальное слово,
    а по отдельности куски словом не являются (защита от 'от анапы')."""
    tokens = text.split(" ")
    out, i, fixed = [], 0, 0
    while i < len(tokens):
        if i + 1 < len(tokens):
            a, b = tokens[i], tokens[i + 1]
            if WORD_RE.fullmatch(a) and WORD_RE.fullmatch(b):
                merged = a + b
                both_words = _word_known(morph, a) and _word_known(morph, b)
                if _word_known(morph, merged) and not both_words:
                    out.append(merged)
                    i += 2
                    fixed += 1
                    continue
        out.append(tokens[i])
        i += 1
    return " ".join(out), fixed


def normalize_stress(text: str) -> str:
    """одОбрить -> одобрить, ДмИтрий -> Дмитрий (опускаем TTS-ударения)."""
    return INNER_CAP_RE.sub(lambda m: m.group(1).lower(), text)


def clean_assistant(text: str, morph, do_stress: bool) -> tuple[str, int]:
    text = MARKER_RE.sub(" ", text)
    text = WS_RE.sub(" ", text).strip()
    fixed = 0
    if morph is not None:
        text, fixed = rejoin_split_words(text, morph)
    if do_stress:
        text = normalize_stress(text)
    return text, fixed


# ---- парсинг лога ---------------------------------------------------------

def split_records(text: str):
    """Разбивает лог на блоки между маркерами '====| дата |====' ... '======'."""
    # Каждая запись начинается строкой вида ========| ... |========
    parts = re.split(r"^========\|.*?\|========\s*$", text, flags=re.MULTILINE)
    # первый кусок до первого маркера — пустой/мусор
    return [p for p in parts if p.strip()]


def extract_input_json(record: str):
    """Достаёт JSON из строки 'INPUT: {...}' внутри блока записи."""
    m = re.search(r"^INPUT:\s*(\{.*\})\s*$", record, flags=re.MULTILINE | re.DOTALL)
    if not m:
        return None
    raw = m.group(1)
    # INPUT может быть на одной длинной строке; обрезаем по последней '}'
    raw = raw[: raw.rfind("}") + 1]
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # пробуем нежадно: до строки REQUEST:
        cut = re.split(r"\nREQUEST:", raw)[0]
        try:
            return json.loads(cut)
        except json.JSONDecodeError:
            return None


def find_first(obj, key):
    """Рекурсивно ищет первое значение по ключу в произвольном JSON."""
    if isinstance(obj, dict):
        if key in obj and obj[key] not in (None, "", [], {}):
            return obj[key]
        for v in obj.values():
            found = find_first(v, key)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for v in obj:
            found = find_first(v, key)
            if found is not None:
                return found
    return None


def find_skillbase(obj):
    """Ищет объект skillbase, у которого есть содержательный sellerRole."""
    if isinstance(obj, dict):
        if isinstance(obj.get("sellerRole"), str) and obj.get("sellerRole").strip():
            return obj
        for v in obj.values():
            found = find_skillbase(v)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for v in obj:
            found = find_skillbase(v)
            if found is not None:
                return found
    return None


# ---- сборка обучающего примера --------------------------------------------

def build_system(skillbase: dict) -> str:
    if not skillbase:
        return ""
    blocks = []
    role = (skillbase.get("sellerRole") or "").strip()
    style = (skillbase.get("sellerStyle") or "").strip()
    if role:
        blocks.append(role)
    if style:
        blocks.append("Стиль общения:\n" + style)

    def join_list(title, items):
        items = [str(x).strip() for x in (items or []) if str(x).strip()]
        if items:
            blocks.append(title + "\n" + "\n".join(f"- {x}" for x in items))

    join_list("План разговора:", skillbase.get("plan"))
    join_list("Факты и условия:", skillbase.get("facts"))
    join_list("Правила:", skillbase.get("rules"))
    return "\n\n".join(blocks).strip()


def clean_turns(chat_history, morph=None, do_stress=False):
    """Чистит chatHistory: убирает пустые/служебные турны.
    user-турны = сырой вход (только тримминг пробелов),
    assistant-турны = чистим артефакты ASR/TTS."""
    turns = []
    fixes = 0
    for msg in chat_history or []:
        role = msg.get("role")
        if role not in ("user", "assistant"):
            continue
        content = (msg.get("content") or "")
        if role == "assistant":
            content, n = clean_assistant(content, morph, do_stress)
            fixes += n
        else:
            content = WS_RE.sub(" ", content).strip()
        if content in ("", "..."):
            continue
        turns.append({"role": role, "content": content})
    return turns, fixes


def is_useful(turns) -> bool:
    if len(turns) < MIN_MESSAGES:
        return False
    real_user = [
        t for t in turns
        if t["role"] == "user" and t["content"].strip().lower() not in TRIVIAL_USER
    ]
    return len(real_user) >= MIN_USER_TURNS


def build_meta(agreements):
    if not isinstance(agreements, dict):
        return None
    keep = (
        "status", "lead_destination", "lead_quality",
        "agreements", "client_facts", "client_name",
    )
    meta = {k: agreements[k] for k in keep if agreements.get(k) not in (None, "")}
    return meta or None


# ---- main -----------------------------------------------------------------

def passes_quality(meta, min_quality, only_transfer) -> bool:
    """Отбор сильных звонков для обучения по меткам исхода."""
    meta = meta or {}
    if only_transfer and meta.get("status") != "transfer":
        return False
    if min_quality is not None:
        q = meta.get("lead_quality")
        if not isinstance(q, (int, float)) or q < min_quality:
            return False
    return True


def dialogue_key(turns):
    """Ключ диалога для дедупа: только турны (system у всех одинаковый)."""
    return tuple((t["role"], t["content"].strip().lower()) for t in turns)


def process(text: str, morph=None, do_stress=False,
            min_quality=None, only_transfer=False, dedup=True):
    examples = []
    seen = set()
    stats = {"records": 0, "no_input": 0, "no_chat": 0, "dropped": 0,
             "dropped_quality": 0, "dropped_dup": 0, "kept": 0, "word_fixes": 0}
    for record in split_records(text):
        stats["records"] += 1
        data = extract_input_json(record)
        if data is None:
            stats["no_input"] += 1
            continue
        chat = find_first(data, "chatHistory")
        if not chat:
            stats["no_chat"] += 1
            continue
        turns, fixes = clean_turns(chat, morph=morph, do_stress=do_stress)
        stats["word_fixes"] += fixes
        if not is_useful(turns):
            stats["dropped"] += 1
            continue
        if dedup:
            key = dialogue_key(turns)
            if key in seen:
                stats["dropped_dup"] += 1
                continue
            seen.add(key)
        meta = build_meta(find_first(data, "agreements"))
        if not passes_quality(meta, min_quality, only_transfer):
            stats["dropped_quality"] += 1
            continue
        system = build_system(find_skillbase(data))
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.extend(turns)
        example = {"messages": messages}
        if meta:
            example["meta"] = meta
        examples.append(example)
        stats["kept"] += 1
    return examples, stats


def main():
    parser = argparse.ArgumentParser(
        description="Чистит лог webhook-запросов в JSONL для fine-tuning."
    )
    parser.add_argument("input", help="входной .log")
    parser.add_argument("output", nargs="?", help="выходной .jsonl")
    parser.add_argument(
        "--clean", action="store_true",
        help="склеивать разорванные слова через морфологию (нужен pymorphy3)",
    )
    parser.add_argument(
        "--normalize-stress", action="store_true",
        help="опускать TTS-ударения (одОбрить->одобрить); НЕ нужно для LLM перед TTS",
    )
    parser.add_argument(
        "--min-quality", type=float, default=None,
        help="брать только звонки с lead_quality >= N (напр. 7)",
    )
    parser.add_argument(
        "--only-transfer", action="store_true",
        help="брать только успешные звонки (status == transfer)",
    )
    parser.add_argument(
        "--no-dedup", action="store_true",
        help="НЕ выкидывать точные дубли диалогов (по умолчанию выкидываются)",
    )
    parser.add_argument(
        "--val-split", type=float, default=0.0,
        help="доля валидации, напр. 0.1 -> отдельный *.val.jsonl",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="seed для train/val split",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = (
        Path(args.output) if args.output
        else input_path.with_suffix(".finetune.jsonl")
    )

    morph = None
    if args.clean:
        morph = load_morph()
        if morph is None:
            print("! pymorphy3 не найден — склейка разорванных слов пропущена.")
            print("  Установи: pip install pymorphy3")

    text = input_path.read_text(encoding="utf-8", errors="ignore")
    examples, stats = process(
        text, morph=morph, do_stress=args.normalize_stress,
        min_quality=args.min_quality, only_transfer=args.only_transfer,
        dedup=not args.no_dedup,
    )

    def dump(path, rows):
        with path.open("w", encoding="utf-8") as f:
            for ex in rows:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"Saved: {path}  ({len(rows)} примеров)")

    if args.val_split and 0 < args.val_split < 1 and len(examples) > 1:
        rng = random.Random(args.seed)
        rng.shuffle(examples)
        n_val = max(1, round(len(examples) * args.val_split))
        val, train = examples[:n_val], examples[n_val:]
        dump(output_path, train)
        dump(output_path.with_suffix(".val.jsonl"), val)
    else:
        dump(output_path, examples)

    print("Stats:")
    for k, v in stats.items():
        print(f"- {k}: {v}")


if __name__ == "__main__":
    main()
