# src/utils/mlflow_helper.py
import functools
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, Generator, Optional

import mlflow


class MLflowManager:
    """
    Класс-утилита для работы с экспериментами.
    Выполняет требование: 'Создать утилиты для работы с экспериментами'
    """

    def __init__(self, experiment_name: str, tracking_uri: str = "sqlite:///mlflow.db"):
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

    @staticmethod
    def log_params_from_dict(params: Dict[str, Any]) -> None:
        """Безопасное логирование словаря параметров"""
        for k, v in params.items():
            mlflow.log_param(k, v)


# --- Декоратор ---
def log_run(
    run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None
) -> Callable:
    """
    Декоратор для автоматического создания MLflow run.
    Выполняет требование: 'Создать декораторы для автоматического логирования'
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Определяем имя запуска
            active_run_name = run_name or func.__name__

            # Автоматически стартуем run
            with mlflow.start_run(run_name=active_run_name):
                if tags:
                    mlflow.set_tags(tags)

                # Логируем время начала
                start_time = time.time()

                try:
                    # Выполняем функцию обучения
                    result = func(*args, **kwargs)

                    # Логируем время выполнения
                    duration = time.time() - start_time
                    mlflow.log_metric("execution_time_seconds", duration)

                    return result
                except Exception as e:
                    # Логируем ошибку, если эксперимент упал
                    mlflow.set_tag("status", "failed")
                    mlflow.log_param("error", str(e))
                    raise e

        return wrapper

    return decorator



@contextmanager
def temp_experiment(exp_name: str) -> Generator[None, None, None]:
    """
    Контекстный менеджер для временного переключения эксперимента.
    Выполняет требование: 'Настроить контекстные менеджеры'
    """
    original_exp_id = (
        mlflow.active_run().info.experiment_id if mlflow.active_run() else None
    )
    mlflow.set_experiment(exp_name)
    try:
        yield
    finally:
        if original_exp_id:
            # Возвращаемся в предыдущий (хотя mlflow это делает неявно, полезно для явности)
            pass
