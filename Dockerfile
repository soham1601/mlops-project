FROM python:3.12-slim AS base
ENV PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app
COPY requirements.lock pyproject.toml ./
COPY src ./src
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install -r requirements.lock \
    && /opt/venv/bin/pip install --no-deps .

# Training happens inside the build, so the model always matches the library versions
FROM base AS trainer
ARG N_TRIALS=30
ENV MLOPS_ROOT=/app
RUN /opt/venv/bin/pip install -c requirements.lock optuna matplotlib
COPY configs ./configs
COPY data/processed ./data/processed
RUN /opt/venv/bin/python -m mlops_project train --config configs/cleveland.yaml \
    --n-trials ${N_TRIALS} --no-tracking

FROM python:3.12-slim AS runtime
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    PATH=/opt/venv/bin:$PATH MLOPS_ROOT=/app MODEL_PATH=/app/models/model.joblib
RUN useradd --create-home --uid 10001 app
WORKDIR /app
COPY --from=base /opt/venv /opt/venv
COPY --from=trainer /app/models ./models
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
    CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "mlops_project.api:app", "--host", "0.0.0.0", "--port", "8000"]
