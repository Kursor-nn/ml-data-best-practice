FROM ghcr.io/astral-sh/uv:latest AS uv_installer

# Базовый образ
FROM apache/airflow:2.10.3-python3.12

USER root

COPY --from=uv_installer /uv /bin/uv

COPY pyproject.toml /pyproject.toml
RUN uv pip install --system --no-cache -r /pyproject.toml
ENV PYTHONPATH="/usr/local/lib/python3.12/site-packages:${PYTHONPATH}"
RUN python -c "import joblib; print(f'!!! JOBLIB SUCCESS: {joblib.__file__} !!!')"

RUN mkdir -p /opt/airflow/data /opt/airflow/models /opt/airflow/metrics && \
    chown -R airflow: /opt/airflow/data /opt/airflow/models /opt/airflow/metrics

USER airflow
