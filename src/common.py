# src/logic.py
from typing import Any, Dict, Tuple

from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def get_data(seed: int = 42, test_size: float = 0.2) -> Tuple[Any, Any, Any, Any]:
    data = load_wine()
    return train_test_split(
        data.data, data.target, test_size=test_size, random_state=seed
    )


def train_model(X_train: Any, y_train: Any, params: Dict[str, Any]) -> Any:
    # params ожидаем словарь типа {'n_estimators': 100, 'seed': 42}
    model = RandomForestClassifier(
        n_estimators=params.get("n_estimators", 100),
        random_state=params.get("seed", 42),
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model: Any, X_test: Any, y_test: Any) -> float:
    y_pred = model.predict(X_test)
    return float(accuracy_score(y_test, y_pred))
