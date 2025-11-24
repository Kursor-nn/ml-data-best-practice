import pandas as pd
import os
import json
import pickle # nosec
from typing import Optional, Any

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from omegaconf import OmegaConf

# Добавляем аннотации типов: (model_config_name: Optional[str] = None) -> Any
def get_config(model_config_name: Optional[str] = None) -> Any:
    base_conf = OmegaConf.load("/opt/airflow/config/base_config.yaml")
    
    if model_config_name:
        model_path = f"/opt/airflow/config/models/{model_config_name}.yaml"
        model_conf = OmegaConf.load(model_path)
        config = OmegaConf.merge(base_conf, model_conf)
    else:
        config = base_conf
        
    if "data" not in config or "raw_path" not in config.data:
        raise ValueError("Invalid Config: 'data.raw_path' is missing")
        
    return config

# Добавляем -> None для всех тасков
def load_data_task() -> None:
    cfg = get_config()
    print("Loading Iris dataset...")
    data = load_iris(as_frame=True)
    df = data.frame
    
    df.to_csv(cfg.data.raw_path, index=False)
    print(f"Data saved to {cfg.data.raw_path}")

def preprocess_data_task() -> None:
    cfg = get_config()
    df = pd.read_csv(cfg.data.raw_path)
    
    train, test = train_test_split(
        df, 
        test_size=cfg.data.test_size, 
        random_state=cfg.data.random_state
    )
    
    train['is_train'] = True
    test['is_train'] = False
    full_df = pd.concat([train, test])
    
    full_df.to_csv(cfg.data.processed_path, index=False)
    print("Preprocessing complete.")

def train_model_task(model_config_file: str) -> None:
    cfg = get_config(model_config_file)
    print(f"Training model: {cfg.model.name}")
    
    df = pd.read_csv(cfg.data.processed_path)
    train_df = df[df['is_train']]
    
    X = train_df.drop(['target', 'is_train'], axis=1)
    y = train_df['target']

    if cfg.model.name == "RandomForestClassifier":
        params = OmegaConf.to_container(cfg.model.params, resolve=True)
        # mypy может ругаться на **params, поэтому игнорируем или кастим, 
        # но обычно с Any конфигом прокатывает.
        model = RandomForestClassifier(**params)
    elif cfg.model.name == "LogisticRegression":
        params = OmegaConf.to_container(cfg.model.params, resolve=True)
        model = LogisticRegression(**params)
    else:
        raise ValueError(f"Unknown model: {cfg.model.name}")

    model.fit(X, y)
    
    os.makedirs(cfg.paths.model_save_dir, exist_ok=True)
    save_path = os.path.join(cfg.paths.model_save_dir, f"{cfg.model.name}.pkl")
    
    with open(save_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {save_path}")

def evaluate_model_task(model_config_file: str) -> None:
    cfg = get_config(model_config_file)
    
    df = pd.read_csv(cfg.data.processed_path)
    test_df = df[not df['is_train']]
    
    X_test = test_df.drop(['target', 'is_train'], axis=1)
    y_test = test_df['target']
    
    model_path = os.path.join(cfg.paths.model_save_dir, f"{cfg.model.name}.pkl")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)  # nosec
    
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    
    metrics = {"model": cfg.model.name, "accuracy": acc}
    
    output_file = os.path.join(cfg.paths.metrics_dir, f"metrics_{cfg.model.name}.json")
    os.makedirs(cfg.paths.metrics_dir, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(metrics, f)
        
    print(f"Evaluation for {cfg.model.name}: {acc}")