# serve_openai.py
#
# OpenAI-совместимый сервер для дообученной модели (база bnb-4bit + LoRA-адаптер).
# Поднимается на том, что уже на боксе после обучения — без скачиваний.
# Контракт совпадает с твоим vLLM: POST /v1/chat/completions (+stream), /v1/models.
#
#   LLM_BASE_URL=http://127.0.0.1:8001/v1   LLM_API_KEY=local-token
#
# Запуск:
#   pip install fastapi uvicorn
#   python3 serve_openai.py
#
# ENV (необязательно):
#   PORT=8001 HOST=127.0.0.1 MODEL_DIR=qwen-mif-lora MAX_LEN=8192 API_KEY=local-token

import json
import os
import time
import threading

import torch
from unsloth import FastLanguageModel
from transformers import TextIteratorStreamer
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn

MODEL_DIR = os.environ.get("MODEL_DIR", "qwen-mif-lora")
MAX_LEN = int(os.environ.get("MAX_LEN", "8192"))
HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "8001"))
SERVED_NAME = os.environ.get("SERVED_NAME", "Qwen/Qwen2.5-14B-Instruct-AWQ")  # имя, что шлёт агент

print(f"Загружаю модель из {MODEL_DIR} (база + LoRA) ...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_DIR,
    max_seq_length=MAX_LEN,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)
print("Модель готова.")

app = FastAPI()


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": SERVED_NAME, "object": "model", "owned_by": "mif"}]}


@app.get("/health")
def health():
    return {"status": "ok"}


def _build_inputs(messages):
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return tokenizer(text, return_tensors="pt").to(model.device)


def _gen_kwargs(inputs, body):
    return dict(
        **inputs,
        max_new_tokens=int(body.get("max_tokens") or 256),
        temperature=float(body.get("temperature", 0.2)),
        top_p=float(body.get("top_p", 0.9)),
        do_sample=float(body.get("temperature", 0.2)) > 0,
        pad_token_id=tokenizer.eos_token_id,
    )


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    stream = bool(body.get("stream", False))
    model_name = body.get("model", SERVED_NAME)
    inputs = _build_inputs(messages)
    created = int(time.time())
    cid = f"chatcmpl-{created}"

    if not stream:
        with torch.no_grad():
            out = model.generate(**_gen_kwargs(inputs, body))
        gen = out[0][inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(gen, skip_special_tokens=True).strip()
        return JSONResponse({
            "id": cid, "object": "chat.completion", "created": created, "model": model_name,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": int(inputs["input_ids"].shape[1]),
                      "completion_tokens": int(gen.shape[0]),
                      "total_tokens": int(inputs["input_ids"].shape[1] + gen.shape[0])},
        })

    # ---- streaming (SSE), как OpenAI ----
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    threading.Thread(target=lambda: model.generate(streamer=streamer, **_gen_kwargs(inputs, body))).start()

    def event_stream():
        first = {"id": cid, "object": "chat.completion.chunk", "created": created,
                 "model": model_name,
                 "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]}
        yield f"data: {json.dumps(first, ensure_ascii=False)}\n\n"
        for piece in streamer:
            if not piece:
                continue
            chunk = {"id": cid, "object": "chat.completion.chunk", "created": created,
                     "model": model_name,
                     "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}]}
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        done = {"id": cid, "object": "chat.completion.chunk", "created": created,
                "model": model_name,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
        yield f"data: {json.dumps(done, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


if __name__ == "__main__":
    print(f"Слушаю http://{HOST}:{PORT}/v1  (модель: {SERVED_NAME})")
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
