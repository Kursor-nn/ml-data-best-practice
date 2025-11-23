# src/experiments.py
from itertools import product
from typing import Any, Dict

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import train_test_split

# Импортируем наши утилиты (выполняем требования ДЗ)
from utils.mlflow_helper import MLflowManager, log_run
from xgboost import XGBRegressor

# Инициализация менеджера
manager = MLflowManager(experiment_name="HW3_Advanced_Tracking")


def load_data(path: str):
    df = pd.read_csv(path, sep=";")  # Wine dataset uses ; separator
    X = df.drop("quality", axis=1)
    y = df["quality"]
    return train_test_split(X, y, test_size=0.2, random_state=42)


def log_feature_importance(model, feature_names, filename="feature_importance.png"):
    """Создает и логирует график важности признаков (Artifacts)"""
    if hasattr(model, "feature_importances_"):
        plt.figure(figsize=(10, 6))
        sns.barplot(x=model.feature_importances_, y=feature_names)
        plt.title("Feature Importance")
        plt.tight_layout()
        plt.savefig(filename)
        mlflow.log_artifact(filename)
        plt.close()


# Функция обучения одной модели, обернутая в НАШ ДЕКОРАТОР
@log_run(tags={"stage": "experimentation", "author": "student"})
def train_and_log(
    model_class, params: Dict[str, Any], X_train, y_train, X_test, y_test, run_name: str
):

    # 1. Логируем параметры (используем нашу утилиту)
    MLflowManager.log_params_from_dict(params)

    # 2. Инициализация и обучение
    model = model_class(**params)
    model.fit(X_train, y_train)

    # 3. Предсказание
    preds = model.predict(X_test)

    # 4. Метрики
    rmse = root_mean_squared_error(y_test, preds)
    mae = mean_absolute_error(y_test, preds)

    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)

    # 5. Логируем модель и артефакты
    # Для XGBoost свой логгер, для Sklearn свой
    if "XGB" in str(model_class):
        mlflow.xgboost.log_model(model, "model")
    else:
        mlflow.sklearn.log_model(model, "model")

    log_feature_importance(model, X_train.columns)

    print(f"Run {run_name} finished. RMSE: {rmse:.4f}")


def run_all_experiments(data_path: str):
    X_train, X_test, y_train, y_test = load_data(data_path)

    # === Определение сетки экспериментов ===
    # Нам нужно 15+ запусков. Сделаем 3 алгоритма * 6 наборов параметров = 18 запусков.

    configs = [
        # 1. Random Forest (3 * 2 = 6 experiments)
        {
            "model": RandomForestRegressor,
            "name_prefix": "RF",
            "params_grid": {
                "n_estimators": [50, 100, 200],
                "max_depth": [5, 10],
                "random_state": [42],
            },
        },
        # 2. Gradient Boosting Sklearn (2 * 3 = 6 experiments)
        {
            "model": GradientBoostingRegressor,
            "name_prefix": "GB",
            "params_grid": {
                "learning_rate": [0.01, 0.1],
                "n_estimators": [50, 100, 150],
                "random_state": [42],
            },
        },
        # 3. XGBoost (2 * 3 = 6 experiments)
        {
            "model": XGBRegressor,
            "name_prefix": "XGB",
            "params_grid": {
                "learning_rate": [0.05, 0.1],
                "n_estimators": [50, 100, 200],
                "random_state": [42],
            },
        },
    ]

    print("Starting experiment series...")

    for config in configs:
        model_cls = config["model"]
        # Генерируем все комбинации параметров
        keys, values = zip(*config["params_grid"].items())
        param_combinations = [dict(zip(keys, v)) for v in product(*values)]

        for i, params in enumerate(param_combinations):
            run_name = f"{config['name_prefix']}_run_{i+1}"
            try:
                # Вызываем декорированную функцию
                train_and_log(
                    model_class=model_cls,
                    params=params,
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    run_name=run_name,
                )
            except Exception as e:
                print(f"Error in {run_name}: {e}")


if __name__ == "__main__":
    import os

    if not os.path.exists("data/raw/wine-quality.csv"):
        print("Dataset not found! Please run 'dvc pull' or download it.")
    else:
        run_all_experiments("data/raw/wine-quality.csv")
