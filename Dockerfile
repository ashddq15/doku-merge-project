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


# Default fallback port (Railway akan overwrite PORT)
ENV PORT=8000
EXPOSE 8000

CMD ["python", "-c", "import os, uvicorn; uvicorn.run('app:app', host='0.0.0.0', port=int(os.environ.get('PORT','8000')))"]
