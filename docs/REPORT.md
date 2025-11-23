# Отчет по ДЗ 3: Трекинг экспериментов и MLOps инфраструктура

**Инструмент:** MLflow (Self-hosted)

## 1. Архитектура и Настройка (Infrastructure)

Вместо использования managed-решений, была развернута собственная инфраструктура MLflow с использованием Docker Compose.

### 1.1. Конфигурация Сервера
Сервер развернут в Docker-контейнере с использованием **кастомного образа** (`mlflow.Dockerfile`).
```Dockerfile
# mlflow.Dockerfile
FROM python:3.12-slim

# Устанавливаем git (на всякий случай)
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Устанавливаем MLflow с поддержкой аутентификации (auth)
# и sqlalcheimy для работы с БД
RUN pip install "mlflow[auth]" sqlalchemy

# Рабочая директория
WORKDIR /app

# Порт
EXPOSE 6001

# Точка входа (запускаем сервер, слушаем 0.0.0.0)
CMD ["mlflow", "server", \
     "--host", "0.0.0.0", \
     "--port", "6001", \
     "--backend-store-uri", "sqlite:////backend_store/mlflow.db", \
     "--default-artifact-root", "/mlruns", \
     "--allowed-hosts", "asdaf.kozachuk.link,asdaf.kozachuk.link:6001,localhost,127.0.0.1", \
     "--app-name", "basic-auth"]
```
- **Backend Store:** SQLite (`sqlite:///backend_store/mlflow.db`) — персистентное хранение метрик и параметров.
- **Artifact Root:** Локальная директория (`/mlruns`), проброшенная через Docker Volumes.
- **Networking:** Сервер настроен на прослушивание `0.0.0.0` с корректной обработкой `Host Headers` (флаг `--allowed-hosts *`).
```yaml
services:
  mlflow:
    build:
      context: .
      dockerfile: mlflow.Dockerfile
    container_name: mlflow_server
    ports:
      - "6001:6001"
    volumes:
      - /srv/mlflow/mlruns:/mlruns
      - /srv/mlflow/backend_store:/backend_store
    environment:
      - MLFLOW_AUTH_DB_PATH=/backend_store/basic_auth.db
      - MLFLOW_FLASK_SERVER_SECRET_KEY=password
    restart: always
```

### 1.2. Безопасность и Аутентификация (Security)
Настроен **Native Basic Authentication**:

1.  **Серверная часть:**
    - Используется флаг `--app-name basic-auth`.
    - База пользователей (`basic_auth.db`) хранится персистентно.
    - Доступ к UI закрыт экраном логина.

**Скриншот 1: Экран авторизации (Login Screen)**
*(Вставь сюда скриншот окна ввода логина/пароля)*
![Login Screen](docs/images/auth-1.png)

## 2. Интеграция с кодом (Engineering)
Разработан модуль `src/utils/mlflow_helper.py`.

### 2.1. Декораторы и Утилиты
Реализован кастомный декоратор **`@log_run`**, который:
- Автоматически инициализирует `mlflow.start_run`.
- Поддерживает динамические имена запусков (аргумент `run_name`).
- Логирует время выполнения (`execution_time_seconds`).
- Обрабатывает исключения (ставит статус `FAILED` и логирует ошибку).
- Сохраняет датасет в MLFlow

## Трекинг
Результаты трегинга работы алгоритмов
![Result 1](docs/images/1.png)

![Result 2](docs/images/2.png)

![Result 3](docs/images/3.png)

![Result 4](docs/images/4.png)

![Result 5](docs/images/filter.png)