from typing import Any, Tuple

from clearml import PipelineDecorator
from common import evaluate_model, get_data, train_model


@PipelineDecorator.component(
    return_values=["X_train", "X_test", "y_train", "y_test"], cache=True
)
def data_step(seed: int) -> Tuple[Any, Any, Any, Any]:
    # Импорт ВНУТРИ функции, чтобы при десериализации на агенте python нашел путь
    return get_data(seed=seed)


@PipelineDecorator.component(return_values=["model"], cache=True)
def train_step(X_train: Any, y_train: Any, n_estimators: int) -> Any:
    # Собираем параметры
    params = {"n_estimators": n_estimators, "seed": 42}
    return train_model(X_train, y_train, params)


@PipelineDecorator.component(return_values=["accuracy"], cache=True)
def eval_step(model: Any, X_test: Any, y_test: Any) -> float:
    return evaluate_model(model, X_test, y_test)


@PipelineDecorator.pipeline(
    name="HW5_Pipeline", project="HW5_MLOps_Course", version="1.0"
)
def main_pipeline(seed: int = 42, n_estimators: int = 50):
    X_train, X_test, y_train, y_test = data_step(seed)
    model = train_step(X_train, y_train, n_estimators)
    eval_step(model, X_test, y_test)


if __name__ == "__main__":
    PipelineDecorator.run_locally()
    main_pipeline()
