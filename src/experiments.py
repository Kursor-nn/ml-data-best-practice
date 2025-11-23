# src/experiments.py
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.data 
from mlflow.data.pandas_dataset import PandasDataset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from xgboost import XGBRegressor
from itertools import product
from typing import Dict, Any
import warnings
import os

from utils.mlflow_helper import MLflowManager, log_run
warnings.filterwarnings("ignore", category=UserWarning, module="mlflow.types.utils")

# Инициализация (читает .env внутри класса)
manager = MLflowManager(experiment="HW3_Advanced_Tracking")

def load_data(path: str):
    df = pd.read_csv(path, sep=";") 
    X = df.drop("quality", axis=1)
    y = df["quality"]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def log_feature_importance(model, feature_names, filename="feature_importance.png"):
    if hasattr(model, "feature_importances_"):
        plt.figure(figsize=(10, 6))
        sns.barplot(x=model.feature_importances_, y=feature_names)
        plt.title("Feature Importance")
        plt.tight_layout()
        plt.savefig(filename)
        mlflow.log_artifact(filename)
        plt.close()

@log_run(tags={"stage": "experimentation", "author": "student"})
def train_and_log(model_class, params: Dict[str, Any], X_train, y_train, X_test, y_test, run_name: str):
    
    # 1. Логируем параметры
    MLflowManager.log_params_from_dict(params)

    # 2. Логируем Dataset (Data Lineage)
    dataset_source = mlflow.data.from_pandas(
        X_train.assign(target=y_train), 
        targets="target", 
        name="wine_quality_train"
    )
    mlflow.log_input(dataset_source, context="training")
    
    # 3. Обучение
    model = model_class(**params)
    model.fit(X_train, y_train)
    
    # 4. Метрики
    preds = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    
    # 5. Логируем модель (ФИКС 1: Используем именованные аргументы)
    input_example = X_train.iloc[:5]

    if "XGB" in str(model_class):
        mlflow.xgboost.log_model(
            xgb_model=model,
            artifact_path="model",
            input_example=input_example
        )
    else:
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=input_example
        )
        
    log_feature_importance(model, X_train.columns)
    
    print(f"Run {run_name} finished. RMSE: {rmse:.4f}")

def run_all_experiments(data_path: str):
    X_train, X_test, y_train, y_test = load_data(data_path)
    
    configs = [
        {
            "model": RandomForestRegressor,
            "name_prefix": "RF",
            "params_grid": {
                "n_estimators": [50, 100],
                "max_depth": [5, 10],
                "random_state": [42]
            }
        },
        {
            "model": GradientBoostingRegressor,
            "name_prefix": "GB",
            "params_grid": {
                "learning_rate": [0.01, 0.1],
                "n_estimators": [50, 100],
                "random_state": [42]
            }
        },
        {
            "model": XGBRegressor,
            "name_prefix": "XGB",
            "params_grid": {
                "learning_rate": [0.05, 0.1],
                "n_estimators": [50, 100],
                "random_state": [42]
            }
        }
    ]
    
    print("Starting experiment series...")
    
    for config in configs:
        model_cls = config["model"]
        keys, values = zip(*config["params_grid"].items())
        param_combinations = [dict(zip(keys, v)) for v in product(*values)]
        
        for i, params in enumerate(param_combinations):
            # Передаем run_name как аргумент, чтобы наш новый декоратор его подхватил
            run_name = f"{config['name_prefix']}_run_{i+1}"
            try:
                train_and_log(
                    model_class=model_cls,
                    params=params,
                    X_train=X_train, 
                    y_train=y_train, 
                    X_test=X_test, 
                    y_test=y_test,
                    run_name=run_name  # <--- Динамическое имя
                )
            except Exception as e:
                print(f"Error in {run_name}: {e}")

if __name__ == "__main__":
    if not os.path.exists("data/raw/wine-quality.csv"):
        print("Dataset not found! Please run 'dvc pull' or download it.")
    else:
        run_all_experiments("data/raw/wine-quality.csv")