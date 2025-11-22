import os
import sys

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import train_test_split

mlflow.set_tracking_uri("file:///" + os.path.abspath("mlruns"))

def train(data_path: str, n_estimators: int, max_depth: int) -> None:
    # 1. Загрузка данных
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path, sep=";")

    X = df.drop("quality", axis=1)
    y = df["quality"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 2. Настройка эксперимента MLflow
    mlflow.set_experiment("WineQuality_Experiment")

    with mlflow.start_run():
        # Логируем параметры
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("data_version", "v1")  # Можно брать git hash, если нужно

        # Обучение
        rf = RandomForestRegressor(
            n_estimators=n_estimators, max_depth=max_depth, random_state=42
        )
        rf.fit(X_train, y_train)

        # Предикт и метрики
        predictions = rf.predict(X_test)
        rmse = root_mean_squared_error(y_test, predictions, squared=False)
        mae = mean_absolute_error(y_test, predictions)

        print(f"RandomForest(n_estimators={n_estimators}, max_depth={max_depth}):")
        print(f"  RMSE: {rmse}")
        print(f"  MAE: {mae}")

        # Логируем метрики
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)

        # 3. Логируем модель
        mlflow.sklearn.log_model(
            rf,
            "random-forest-model",
            input_example=X_train.iloc[:1],
            registered_model_name="WineQualityModel",  # Автоматически создает версию модели в Registry
        )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/models/train_model.py <data_path>")
        sys.exit(1)

    path = sys.argv[1]

    train(path, n_estimators=50, max_depth=5)
    train(path, n_estimators=100, max_depth=10)
