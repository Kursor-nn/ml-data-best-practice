# HW5: ClearML MLOps Workflow

Проект реализует полный цикл MLOps с использованием **ClearML**, **Docker** и **uv**.
Включает в себя трекинг экспериментов, реестр моделей и автоматизированные пайплайны.

## Требования
- Docker & Docker Compose
- Python 3.11+
- uv

## Быстрый старт (Воспроизведение)

### Подготовка инфраструктуры
Поднимаем локальный сервер ClearML.
```bash
docker compose up -d
uv run clearml-init
```

### Запуск экспериментов

```
# Запуск с параметрами по умолчанию
uv run src/train.py

# Запуск с кастомными параметрами
uv run src/train.py --n_estimators 10 --seed 42
uv run src/train.py --n_estimators 200 --seed 1
```

### Запуск пайплайна
Допущения: все запускаем локально в рамках обучения.

```
uv run src/pipeline.py
```
