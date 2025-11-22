# Intro
Для проверки проекта, необходимо установить
- uv
- docker

# Инструкция по воспроизведению
## Установить зависимости
```bash
uv sync
```

## Настроить DVC для локального использования

### Допущения
В проекте настроен локальный DVC Remote (/tmp/dvc-storage) для демонстрации.

#### Инициализация DVC:

```bash
uv run dvc init
```

#### Создаем папку где-то вне проекта (имитация облака)
```bash
mkdir -p /tmp/dvc-storage
```

#### Добавляем её как remote 'local_storage'
```bash
uv run dvc remote add -d local_storage /tmp/dvc-storage
```

#### Загрузка данных
```bash
curl -o data/raw/wine-quality.csv https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv
```
#### Трекинг данных

```bash
uv run dvc add data/raw/wine-quality.csv
```
#### Пуш данных
```bash
uv run dvc push
```


### Подготовка для воспроизведения
Для последующего воспроизведения, необходимо сделать пару ручных действий:
```bash
mkdir -p data/raw
unzip dvc_localstorage.zip
uv run dvc remote add -d local_storage ./tmp/dvc-storage -f

#curl -o data/raw/wine-quality.csv https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv
uv run dvc status
# Ожидаемый результат: "Data and pipelines are up to date." или отсутствие изменений для data/raw/wine-quality.csv
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
uv run mlflow ui # 
```
тогда по адресу: `http://127.0.0.1:5000` будет доступен ML Flow UI

