# Dockerfile
FROM python:3.12-slim

# Устанавливаем git (нужен для dvc и pip зависимостей)
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Устанавливаем uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

# Копируем файлы зависимостей
COPY pyproject.toml uv.lock ./

# Устанавливаем зависимости
RUN uv sync --frozen
ENV PATH="/app/.venv/bin:$PATH"

# Копируем проект (включая .dvc папку, чтобы DVC работал)
COPY . .

# При запуске контейнера показываем версии пакетов и запускаем help
CMD ["python", "src/models/train_model.py"]