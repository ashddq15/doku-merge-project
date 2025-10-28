# ---- build runtime ----
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . /app

# (opsional) Healthcheck lokal — boleh dibiarkan
HEALTHCHECK --interval=30s --timeout=3s \
  CMD python - <<'PY' || exit 1
import os, socket
port = int(os.environ.get("PORT", "8000"))
s = socket.socket(); s.settimeout(2)
try:
    s.connect(("127.0.0.1", port))
    s.close()
except Exception:
    raise SystemExit(1)
PY

# Default fallback port (Railway akan overwrite PORT)
ENV PORT=8000
EXPOSE 8000

CMD ["python", "-c", "import os, uvicorn; uvicorn.run('app:app', host='0.0.0.0', port=int(os.environ.get('PORT','8000')))"]
