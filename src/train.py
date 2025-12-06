import argparse

import joblib

from clearml import Logger, Task
from common import evaluate_model, get_data, train_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    task = Task.init(
        project_name="HW5_MLOps_Course", task_name="Wine_Training", output_uri=True
    )
    task.connect(args)

    # Используем функции из logic.py
    X_train, X_test, y_train, y_test = get_data(seed=args.seed)

    # Превращаем args в словарь
    params = vars(args)
    model = train_model(X_train, y_train, params)

    accuracy = evaluate_model(model, X_test, y_test)

    Logger.current_logger().report_scalar("Performance", "Accuracy", accuracy, 4)
    joblib.dump(model, "model.joblib")
    print(f"Done. Accuracy: {accuracy}")


if __name__ == "__main__":
    main()
