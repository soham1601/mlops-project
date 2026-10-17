"""FastAPI service: /predict, health checks, JSON request logs and Prometheus metrics."""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, Field

from mlops_project.config import DESCRIPTIONS, VALID_RANGES
from mlops_project.predict import DEFAULT_MODEL, SAMPLE_PATIENT, load_model, predict_records

log = logging.getLogger("heart_api")
QUIET_PATHS = {"/metrics", "/health", "/ready"}


def _range(name: str) -> dict:
    low, high = VALID_RANGES[name]
    return {"ge": low, "le": high, "description": DESCRIPTIONS[name]}


class Patient(BaseModel):
    age: float = Field(**_range("age"))
    sex: Literal[0, 1] = Field(description=DESCRIPTIONS["sex"])
    cp: Literal[1, 2, 3, 4] = Field(description=DESCRIPTIONS["cp"])
    trestbps: float = Field(**_range("trestbps"))
    chol: float = Field(**_range("chol"))
    fbs: Literal[0, 1] = Field(description=DESCRIPTIONS["fbs"])
    restecg: Literal[0, 1, 2] = Field(description=DESCRIPTIONS["restecg"])
    thalach: float = Field(**_range("thalach"))
    exang: Literal[0, 1] = Field(description=DESCRIPTIONS["exang"])
    oldpeak: float = Field(**_range("oldpeak"))
    slope: Literal[1, 2, 3] = Field(description=DESCRIPTIONS["slope"])
    ca: Literal[0, 1, 2, 3] | None = Field(default=None, description=DESCRIPTIONS["ca"])
    thal: Literal[3, 6, 7] | None = Field(default=None, description=DESCRIPTIONS["thal"])

    model_config = {"json_schema_extra": {"examples": [SAMPLE_PATIENT]}}


class Prediction(BaseModel):
    prediction: Literal[0, 1]
    label: str
    probability: float = Field(description="Probability of heart disease")
    confidence: float = Field(description="Probability of the predicted class")
    model: str


class Metrics:
    """Prometheus metrics, kept in a registry of their own so each app instance is isolated."""

    def __init__(self):
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "http_requests_total",
            "HTTP requests",
            ["method", "path", "status"],
            registry=self.registry,
        )
        self.latency = Histogram(
            "http_request_duration_seconds", "Request latency", ["method", "path"],
            registry=self.registry,
        )  # fmt: skip
        self.predictions = Counter(
            "predictions_total", "Predictions made", ["label"], registry=self.registry
        )
        self.probability = Histogram(
            "prediction_probability", "Predicted probability of disease",
            buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], registry=self.registry,
        )  # fmt: skip
        self.model_loaded = Gauge(
            "model_loaded", "1 when a model is loaded", registry=self.registry
        )


def _configure_logging() -> None:
    if not log.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        log.addHandler(handler)
    log.setLevel(logging.INFO)


def create_app(model_path: str | Path | None = None) -> FastAPI:
    model_path = Path(model_path or os.environ.get("MODEL_PATH", DEFAULT_MODEL))
    metrics = Metrics()
    state: dict = {"model": None, "card": {}}
    _configure_logging()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            state["model"] = load_model(model_path)
            card = model_path.with_name("model_card.json")
            state["card"] = json.loads(card.read_text(encoding="utf-8")) if card.exists() else {}
            metrics.model_loaded.set(1)
            log.info(json.dumps({"event": "model_loaded", "path": str(model_path)}))
        except FileNotFoundError as exc:
            metrics.model_loaded.set(0)
            log.error(json.dumps({"event": "model_missing", "error": str(exc)}))
        yield

    app = FastAPI(title="Heart disease risk API", version="1.0.0", lifespan=lifespan)

    @app.middleware("http")
    async def observe(request: Request, call_next):
        started = time.perf_counter()
        request_id = request.headers.get("x-request-id", uuid.uuid4().hex[:12])
        request.state.extra = {}
        response = await call_next(request)
        seconds = time.perf_counter() - started
        route = request.scope.get("route")
        path = route.path if route else "unmatched"
        metrics.requests.labels(request.method, path, str(response.status_code)).inc()
        metrics.latency.labels(request.method, path).observe(seconds)
        response.headers["x-request-id"] = request_id
        if request.url.path not in QUIET_PATHS:
            log.info(
                json.dumps(
                    {
                        "event": "request",
                        "request_id": request_id,
                        "method": request.method,
                        "path": request.url.path,
                        "status": response.status_code,
                        "duration_ms": round(seconds * 1000, 2),
                        **request.state.extra,
                    }
                )
            )
        return response

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "model_loaded": state["model"] is not None}

    @app.get("/ready")
    def ready() -> dict:
        if state["model"] is None:
            raise HTTPException(status_code=503, detail="model not loaded")
        return {"status": "ready"}

    @app.get("/info")
    def info() -> dict:
        card = state["card"]
        return {k: card.get(k) for k in ("best_model", "trained_at", "test_metrics", "versions")}

    @app.post("/predict", response_model=Prediction)
    def predict(patient: Patient, request: Request) -> dict:
        if state["model"] is None:
            raise HTTPException(status_code=503, detail="model not loaded")
        features = patient.model_dump()
        result = predict_records(state["model"], [features])[0]
        metrics.predictions.labels(result["label"]).inc()
        metrics.probability.observe(result["probability"])
        request.state.extra = {"input": features, **result}
        return {**result, "model": state["card"].get("best_model", "unknown")}

    @app.get("/metrics")
    def prometheus() -> Response:
        return Response(generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()
