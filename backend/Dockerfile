FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# libgomp is required by LightGBM/CatBoost.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/
COPY config/ ./config/
COPY scripts/ ./scripts/

# Writable locations for artefacts and uploads.
RUN mkdir -p /app/data/raw /app/data/uploads /app/runs /app/reports /app/artifacts

ENV DJANGO_SETTINGS_MODULE=config.settings \
    PYTHONPATH=/app/backend

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8000/api/v1/health/ || exit 1

CMD ["sh", "-c", "python backend/manage.py migrate --noinput && python backend/manage.py runserver 0.0.0.0:8000"]
