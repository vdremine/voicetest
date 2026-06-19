FROM python:3.12-slim

WORKDIR /app

ENV LD_LIBRARY_PATH=/usr/local/lib/python3.12/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.12/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH}

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Optional OmniVoice TTS. Non-fatal: a failure here (e.g. torch conflict) must
# not break the image — the agent falls back to Silero at runtime.
COPY requirements-omnivoice.txt .
RUN pip install --no-cache-dir -r requirements-omnivoice.txt || \
    echo "WARN: omnivoice optional install failed; agent will use Silero TTS"

COPY . .

CMD ["python", "main.py"]
