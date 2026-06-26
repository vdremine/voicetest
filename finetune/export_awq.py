# export_awq.py
#
# Превращает обученный LoRA-адаптер в ПРОД-модель под твой стек
# (Qwen/Qwen2.5-14B-Instruct-AWQ). Два шага, запускать ПО ОТДЕЛЬНОСТИ
# (чтобы освобождать VRAM/диск между ними):
#
#   python3 export_awq.py merge   # LoRA + 4-bit база -> слитые fp16 (~28 ГБ)
#   python3 export_awq.py awq     # fp16 -> AWQ (~9 ГБ) c калибровкой на диалогах
#
# Результат qwen-mif-awq/ кладёшь в прод как замену Qwen2.5-14B-Instruct-AWQ
# (тот же vLLM/serving, та же конфигурация — только путь к весам меняешь).
#
# ДИСК (на 60 ГБ боксе впритык): после 'merge' можно почистить кеш базы
#   rm -rf ~/.cache/huggingface/hub/*Qwen2.5-14B*bnb*
# чтобы освободить место под AWQ.
#
# Установка:
#   pip install autoawq        # для шага awq (в том же venv, где обучали)

import json
import sys
from pathlib import Path

ADAPTER_DIR = "qwen-mif-lora"          # выход train_lora_qwen.py
MERGED_DIR = "qwen-mif-merged"         # промежуточный fp16 (~28 ГБ)
AWQ_DIR = "qwen-mif-awq"               # финал под прод (~9 ГБ)
TRAIN_FILE = "requests.finetune.tts.jsonl"
MAX_SEQ_LEN = 4096
N_CALIB = 192                          # примеров на калибровку AWQ


def step_merge():
    """LoRA + 4-bit база -> слитые fp16 веса. Через Unsloth: дегвантует уже
    скачанную 4-bit базу в памяти, fp16-базу заново НЕ качает (бережём диск)."""
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=ADAPTER_DIR,        # грузит базу + адаптер из папки обучения
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=True,
    )
    model.save_pretrained_merged(MERGED_DIR, tokenizer, save_method="merged_16bit")
    print(f"OK: слитые fp16 веса -> {MERGED_DIR}/")
    print("Если диск поджимает, перед 'awq' почисти кеш базы:")
    print("  rm -rf ~/.cache/huggingface/hub/*Qwen2.5-14B*bnb*")


def step_awq():
    """AWQ-квантизация слитой fp16-модели. Калибровка на РЕАЛЬНЫХ диалогах —
    точнее держит домен и формат под TTS, чем дефолтный калибровочный корпус."""
    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MERGED_DIR, trust_remote_code=True)

    rows = [json.loads(l) for l in
            Path(TRAIN_FILE).read_text(encoding="utf-8").splitlines() if l.strip()]
    calib = [
        tok.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=False)
        for r in rows[:N_CALIB]
    ]

    model = AutoAWQForCausalLM.from_pretrained(MERGED_DIR, safetensors=True)
    # параметры как у стандартного Qwen2.5-14B-Instruct-AWQ
    quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}
    model.quantize(tok, quant_config=quant_config, calib_data=calib)
    model.save_quantized(AWQ_DIR)
    tok.save_pretrained(AWQ_DIR)
    print(f"OK: AWQ-модель -> {AWQ_DIR}/")
    print("Клади в прод как замену Qwen2.5-14B-Instruct-AWQ (путь к весам в конфиге).")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "merge":
        step_merge()
    elif cmd == "awq":
        step_awq()
    else:
        print("Использование: python3 export_awq.py {merge|awq}")
        print("  сначала merge, потом awq")
        sys.exit(1)
