import json
import logging

import pytest
from fastapi.testclient import TestClient

from mlops_project.api import create_app
from mlops_project.predict import SAMPLE_PATIENT


@pytest.fixture(scope="module")
def client(trained):
    result, _ = trained
    with TestClient(create_app(result["model_path"])) as c:
        yield c


def metric_value(client, line_start: str) -> float:
    for line in client.get("/metrics").text.splitlines():
        if line.startswith(line_start):
            return float(line.rsplit(" ", 1)[1])
    return 0.0


def test_health_and_ready(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}
    assert client.get("/ready").status_code == 200


def test_predict_returns_prediction_and_confidence(client):
    response = client.post("/predict", json=SAMPLE_PATIENT)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in (0, 1)
    assert 0.5 <= body["confidence"] <= 1
    assert 0 <= body["probability"] <= 1
    assert body["model"] in ("logistic_regression", "random_forest", "xgboost")


def test_missing_ca_and_thal_are_accepted(client):
    patient = {k: v for k, v in SAMPLE_PATIENT.items() if k not in ("ca", "thal")}
    assert client.post("/predict", json=patient).status_code == 200


@pytest.mark.parametrize(
    "change",
    [{"age": 250}, {"sex": 2}, {"chol": -5}, {"cp": 9}, {"thal": 5}, {"age": "old"}],
)
def test_impossible_values_are_rejected(client, change):
    response = client.post("/predict", json={**SAMPLE_PATIENT, **change})
    assert response.status_code == 422


def test_missing_required_field_is_rejected(client):
    patient = {k: v for k, v in SAMPLE_PATIENT.items() if k != "age"}
    assert client.post("/predict", json=patient).status_code == 422


def test_metrics_count_requests_and_predictions(client):
    before = metric_value(client, 'predictions_total{label="')
    client.post("/predict", json=SAMPLE_PATIENT)
    client.post("/predict", json={**SAMPLE_PATIENT, "age": 250})
    text = client.get("/metrics").text
    assert 'http_requests_total{method="POST",path="/predict",status="200"}' in text
    assert 'http_requests_total{method="POST",path="/predict",status="422"}' in text
    assert "http_request_duration_seconds_bucket" in text
    assert "prediction_probability_bucket" in text
    assert "model_loaded 1.0" in text
    assert (
        sum(
            float(line.rsplit(" ", 1)[1])
            for line in text.splitlines()
            if line.startswith("predictions_total{")
        )
        >= before + 1
    )


def test_every_prediction_request_is_logged_as_json(client, caplog):
    with caplog.at_level(logging.INFO, logger="heart_api"):
        client.post("/predict", json=SAMPLE_PATIENT, headers={"x-request-id": "abc123"})
    entry = json.loads(next(r.message for r in caplog.records if "abc123" in r.message))
    assert entry["event"] == "request"
    assert entry["path"] == "/predict" and entry["status"] == 200
    assert entry["input"]["age"] == SAMPLE_PATIENT["age"]
    assert entry["duration_ms"] >= 0
    assert "prediction" in entry and "probability" in entry


def test_probe_and_scrape_requests_are_not_logged(client, caplog):
    with caplog.at_level(logging.INFO, logger="heart_api"):
        client.get("/health")
        client.get("/metrics")
    assert not caplog.records


def test_info_describes_the_model(client):
    body = client.get("/info").json()
    assert body["best_model"] and "scikit-learn" in body["versions"]


def test_service_reports_not_ready_without_a_model(tmp_path):
    with TestClient(create_app(tmp_path / "missing.joblib")) as c:
        assert c.get("/health").json()["model_loaded"] is False
        assert c.get("/ready").status_code == 503
        assert c.post("/predict", json=SAMPLE_PATIENT).status_code == 503
