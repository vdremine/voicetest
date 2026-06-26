# train_lora_qwen.py
#
# LoRA fine-tuning Qwen2.5 на диалогах колл-центра (выход extract_finetune.py).
# Запускать НА GPU-машине (не на этом маке). ~ нужна 1 GPU:
#   Qwen2.5-7B  -> ~10-12 ГБ VRAM (4bit + LoRA)
#   Qwen2.5-14B -> ~18-22 ГБ VRAM
#
# Установка (на GPU-боксе, в venv/conda):
#   pip install unsloth
#   pip install --no-deps trl peft accelerate bitsandbytes
#
# Данные готовим так (LLM перед TTS — без --normalize-stress):
#   python3 extract_finetune.py requests_interest.log requests.finetune.tts.jsonl \
#       --clean --val-split 0.1
#   -> requests.finetune.tts.jsonl (train) + requests.finetune.tts.val.jsonl (val)
#   (для обучения только на успешных звонках добавь --only-transfer --min-quality 7)
#
# Запуск:
#   python3 train_lora_qwen.py

from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

# ---- настройки ------------------------------------------------------------

# RTX 4090 48GB (модиф. карта: nvidia-smi показал 49140 MiB), диск 60 ГБ.
# Берём ПРЕДКВАНТОВАННУЮ 4-bit модель: качать ~9 ГБ вместо 28 ГБ fp16
# (важно при 60 ГБ диска). 48 ГБ VRAM — для QLoRA 14B вагон места.
# В проде на 4090 14B крутится в 4-bit/AWQ с низкой latency — ок для звонков.
MODEL = "unsloth/Qwen2.5-14B-Instruct-bnb-4bit"   # OOM? -> Qwen2.5-7B-Instruct-bnb-4bit
MAX_SEQ_LEN = 4096                        # хватает на длинные звонки (~30 турнов)
TRAIN_FILE = "requests.finetune.tts.jsonl"
VAL_FILE = "requests.finetune.tts.val.jsonl"   # None если без валидации
OUTPUT_DIR = "qwen-mif-lora"

# Настройки под СРЕДНИЙ датасет (~2810 уникальных диалогов после дедупа,
# train ~2529 / val ~281). Данных достаточно, риск переобучения низкий:
LORA_RANK = 32          # больше ёмкости: на 4k данных не переобучит
LORA_ALPHA = 64         # alpha = 2*rank
LORA_DROPOUT = 0.05     # лёгкая регуляризация, можно и 0.0
EPOCHS = 2              # на 4k хватает 2; ОРИЕНТИРУЙСЯ на eval_loss, не на эпохи
LR = 2e-4              # стандартный LR — данных достаточно для устойчивости
BATCH = 2              # 48 ГБ VRAM — можно и 4, начнём безопасно с 2
GRAD_ACCUM = 8         # эффективный батч = 16 — стабильнее градиент

# ---- модель + LoRA --------------------------------------------------------

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL,
    max_seq_length=MAX_SEQ_LEN,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")

# ---- данные ---------------------------------------------------------------
# extract_finetune.py отдаёт {"messages":[...], "meta":{...}}.
# Берём только messages и прогоняем через chat-шаблон Qwen.

def format_example(batch):
    texts = [
        tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        for msgs in batch["messages"]
    ]
    return {"text": texts}

train_ds = load_dataset("json", data_files=TRAIN_FILE, split="train")
train_ds = train_ds.map(format_example, batched=True)

eval_ds = None
if VAL_FILE:
    eval_ds = load_dataset("json", data_files=VAL_FILE, split="train")
    eval_ds = eval_ds.map(format_example, batched=True)

# ---- тренер ---------------------------------------------------------------

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    args=SFTConfig(
        output_dir=OUTPUT_DIR,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LEN,
        per_device_train_batch_size=BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        warmup_ratio=0.05,
        logging_steps=5,
        save_strategy="epoch",
        eval_strategy="epoch" if eval_ds else "no",
        # на малых данных берём чекпойнт с лучшим eval_loss, а не последний
        load_best_model_at_end=bool(eval_ds),
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        weight_decay=0.01,
        optim="adamw_8bit",
        lr_scheduler_type="cosine",
        seed=42,
        report_to="none",
    ),
)

# Учим ТОЛЬКО на ответах ассистента (system/user маскируем) —
# модель учится генерировать реплики Дмитрия, а не воспроизводить вход.
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|im_start|>user\n",
    response_part="<|im_start|>assistant\n",
)

trainer.train()

# ---- сохранение -----------------------------------------------------------
# LoRA-адаптер (десятки МБ) — для инференса грузится поверх базовой модели.
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"LoRA сохранён в {OUTPUT_DIR}/")

# Опционально — слитые веса под vLLM/прод (раскомментируй):
# model.save_pretrained_merged("qwen-mif-merged", tokenizer, save_method="merged_16bit")
