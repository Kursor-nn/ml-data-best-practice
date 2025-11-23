## Requirements
- docker
## Допущения
### Допущение 1
Для простоты воспроизведения результатов, я поднял MLFlow на своем сервер: http://asdaf.kozachuk.link:6001 
Креды к учетке: admin/password1234.
Да, не безопасно, но для обучения пойдет.

### Допущение 2
dvc настроен только локально и добавлен в репозиторий - чисто ради демонстрации.
по хорошему,нужно поднимать S3 и туда все сохранять.


### Допущение 3
не хорошо хранить среды в репозитории, но для воспроизвдения эксперимента в рамках обучения - почему бы и нет.

## Воспроизведение эксепериментов
```bash
# загрузим данные
uv run dvc pull
uv run dvc status

# запустим сборку для запуска экспериментов
docker build -t ml_project_hw3 .
# запустим эксперименты
docker run --rm --env-file .env.example -v $(pwd)/data:/app/data -v $(pwd)/mlruns:/app/mlruns -v $(pwd)/mlflow.db:/app/mlflow.db ml_project_hw3 uv run src/experiments.py data/raw/wine-quality.csv
```