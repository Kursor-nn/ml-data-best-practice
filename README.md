# Intro
Для проверки проекта, необходимо установить
- uv
- docker


# Инструкция по воспроизведению
## Установить зависимости
```bash
uv sync
uv run pre-commit autoupdate
uv run pre-commit install
```

## Настроить DVC для локального использования

### Допущения
В проекте настроен локальный DVC Remote (./tmp/dvc-storage) для демонстрации.

#### Инициализация DVC:

```bash
uv run dvc init
# Создание локального хранилища
mkdir -p tmp/dvc-storage
uv run dvc remote add -d local_storage ./tmp/dvc-storage -f
curl -o data/raw/wine-quality.csv https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv
uv run dvc add data/raw/wine-quality.csv
uv run dvc push
```

### Подготовка для воспроизведения
Для последующего воспроизведения, необходимо сделать пару ручных действий:
```bash
unzip dvc_localstorage.zip
uv run dvc remote add -d local_storage ./tmp/dvc-storage -f
uv run dvc pull
uv run dvc status
# Ожидаемый результат: "Data and pipelines are up to date."
```


## Запуск обучения и трекинг (Запуск в Docker)
Для сбора метрик, соберем контейнер и запустим его
```bash
touch mlflow.db
docker build -t ml_project .
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/mlruns:/app/mlruns \
  -v $(pwd)/mlflow.db:/app/mlflow.db \
  ml_project uv run src/models/train_model.py data/raw/wine-quality.csv
```

А для просмотра результатов:
```
uv run mlflow ui --host 0.0.0.0 --port 5001
```
тогда по адресу: `http://127.0.0.1:5001` будет доступен ML Flow UI

