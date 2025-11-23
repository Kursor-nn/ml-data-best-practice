# src/utils/mlflow_helper.py
import functools
import time
import os
from dotenv import load_dotenv
from contextlib import contextmanager
from typing import Any, Callable, Dict, Generator, Optional

import mlflow

load_dotenv()

class MLflowManager:
    """
    Класс-утилита для работы с экспериментами.
    Выполняет требование: 'Создать утилиты для работы с экспериментами'
    """

    def __init__(self, experiment: str):
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
        experiment_name = experiment or os.getenv("MLFLOW_EXPERIMENT_NAME", "Default_Experiment")

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)


    @staticmethod
    def log_params_from_dict(params: Dict[str, Any]) -> None:
        """Безопасное логирование словаря параметров"""
        for k, v in params.items():
            mlflow.log_param(k, v)


def log_run(run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None) -> Callable:
    """
    Декоратор для автоматического создания MLflow run.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # 1. Пытаемся найти run_name в аргументах функции (динамическое имя)
            # Сначала ищем в kwargs
            dynamic_run_name = kwargs.get("run_name")
            
            # Если в kwargs нет, используем то, что дали в декораторе, или имя функции
            active_run_name = dynamic_run_name or run_name or func.__name__
            
            # Автоматически стартуем run
            with mlflow.start_run(run_name=active_run_name):
                if tags:
                    mlflow.set_tags(tags)
                
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    
                    duration = time.time() - start_time
                    mlflow.log_metric("execution_time_seconds", duration)
                    return result
                except Exception as e:
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
            pass
