FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

WORKDIR /app

# Базовые пакеты для сборки python-зависимостей.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Сначала ставим зависимости для лучшего кеширования слоев.
COPY apps/requirements.txt /app/apps/requirements.txt
RUN pip install -r /app/apps/requirements.txt

# Копируем только код приложения; обязательные данные и модель подаются томами в compose.
COPY apps /app/apps
COPY src /app/src

EXPOSE 8501

CMD ["streamlit", "run", "apps/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
