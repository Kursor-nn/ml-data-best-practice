import os
import sys

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

from src.pipelines import evaluate_model_task, load_data_task, preprocess_data_task, train_model_task

airflow_home = os.environ.get("AIRFLOW_HOME", "/opt/airflow")
if airflow_home not in sys.path:
    sys.path.append(airflow_home)

sys.path.append(os.path.join(os.environ["AIRFLOW_HOME"], "src"))


default_args = {"owner": "student", "start_date": days_ago(1), "retries": 0}

with DAG(
    dag_id="ml_pipeline_omegaconf",
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
    tags=["ml", "omegaconf"],
) as dag:
    # 1. Load Data
    t_load = PythonOperator(task_id="load_data", python_callable=load_data_task)

    # 2. Preprocess
    t_preprocess = PythonOperator(
        task_id="preprocess", python_callable=preprocess_data_task
    )

    # --- Ветка 1: Random Forest ---
    t_train_rf = PythonOperator(
        task_id="train_rf",
        python_callable=train_model_task,
        op_kwargs={"model_config_file": "random_forest"},  # Имя файла в config/models/
    )

    t_eval_rf = PythonOperator(
        task_id="eval_rf",
        python_callable=evaluate_model_task,
        op_kwargs={"model_config_file": "random_forest"},
    )

    # --- Ветка 2: Logistic Regression ---
    t_train_lr = PythonOperator(
        task_id="train_lr",
        python_callable=train_model_task,
        op_kwargs={"model_config_file": "logistic_regression"},
    )

    t_eval_lr = PythonOperator(
        task_id="eval_lr",
        python_callable=evaluate_model_task,
        op_kwargs={"model_config_file": "logistic_regression"},
    )

    # Зависимости (Graph)
    t_load >> t_preprocess

    # Параллельный запуск
    t_preprocess >> [t_train_rf, t_train_lr]

    t_train_rf >> t_eval_rf
    t_train_lr >> t_eval_lr
