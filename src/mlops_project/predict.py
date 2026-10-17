"""Load the saved model and predict for one or many patients."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from mlops_project.config import ROOT, TARGET_LABELS

DEFAULT_MODEL = ROOT / "models" / "model.joblib"

SAMPLE_PATIENT = {
    "age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1, "restecg": 2,
    "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3, "ca": 0, "thal": 6,
}  # fmt: skip


def load_model(source: str | Path = DEFAULT_MODEL, tracking_uri: str | None = None):
    """Load a joblib file, or an MLflow registry model such as ``models:/name@champion``."""
    if str(source).startswith("models:/"):
        import mlflow

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        return mlflow.sklearn.load_model(str(source))
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}. Train one with `train` first.")
    return joblib.load(path)


def predict_records(model, records: list[dict]) -> list[dict]:
    """Return prediction, label, probability of disease and confidence for every record."""
    proba = model.predict_proba(pd.DataFrame(records))[:, 1]
    results = []
    for p in proba:
        prediction = int(p >= 0.5)
        results.append(
            {
                "prediction": prediction,
                "label": TARGET_LABELS[prediction],
                "probability": round(float(p), 4),
                "confidence": round(float(max(p, 1 - p)), 4),
            }
        )
    return results
