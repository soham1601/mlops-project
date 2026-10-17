"""Score a saved model on any CSV from the Heart Disease family."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from mlops_project.datasets import load_xy
from mlops_project.features import clean_features

log = logging.getLogger(__name__)
METRIC_NAMES = ("accuracy", "precision", "recall", "f1", "roc_auc")


def score(y_true, proba, threshold: float = 0.5) -> tuple[dict, list]:
    """Return ({metric: value}, confusion matrix). roc_auc is None if only one class is present."""
    y_true = np.asarray(y_true)
    pred = (np.asarray(proba) >= threshold).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba)) if len(np.unique(y_true)) == 2 else None,
    }
    return metrics, confusion_matrix(y_true, pred, labels=[0, 1]).tolist()


def evaluate(
    model_path: str | Path,
    data_path: str | Path,
    target: str = "target",
    sep: str = ",",
    rename: dict[str, str] | None = None,
) -> dict:
    """Load the model, clean and score the file, and compare with the training scores."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Train one with `train` first.")
    pipeline = joblib.load(model_path)

    X, y, info = load_xy(data_path, target=target, sep=sep, rename=rename)
    _, issues = clean_features(X)
    fixed = {col: sum(c.values()) for col, c in issues.items() if sum(c.values())}

    proba = pipeline.predict_proba(X)[:, 1]
    metrics, cm = score(y, proba)
    result = {
        "model": str(model_path),
        "data": str(data_path),
        "data_info": info,
        "values_fixed_by_cleaner": fixed,
        "metrics": metrics,
        "confusion_matrix": cm,
    }

    card_path = model_path.with_name("model_card.json")
    if card_path.exists():
        reference = json.loads(card_path.read_text(encoding="utf-8"))["test_metrics"]
        result["training_test_metrics"] = reference
        result["change_vs_training"] = {
            k: None if metrics[k] is None or reference.get(k) is None else metrics[k] - reference[k]
            for k in METRIC_NAMES
        }
    return result


def format_result(result: dict) -> str:
    """Human-readable summary of an ``evaluate`` result."""
    info = result["data_info"]
    lines = [
        f"Data: {result['data']}",
        f"  rows used {info['rows_used']} of {info['rows_read']}, "
        f"class balance {info['class_balance']}",
        f"  values repaired by the cleaner: {result['values_fixed_by_cleaner'] or 'none'}",
        "",
        f"{'metric':<10}{'this data':>12}{'training test':>16}{'change':>10}",
    ]
    reference = result.get("training_test_metrics", {})
    change = result.get("change_vs_training", {})
    for name in METRIC_NAMES:
        value = result["metrics"][name]
        ref, delta = reference.get(name), change.get(name)
        lines.append(f"{name:<10}{_num(value):>12}{_num(ref):>16}{_num(delta, signed=True):>10}")
    (tn, fp), (fn, tp) = result["confusion_matrix"]
    lines += ["", f"Confusion matrix: TN={tn} FP={fp} FN={fn} TP={tp}"]
    lines += [
        "",
        "Note: rows the model was trained on give optimistic scores. For an honest comparison",
        "score data it has not seen, e.g. reports/training/holdout_test.csv or another site.",
    ]
    return "\n".join(lines)


def _num(value, signed: bool = False) -> str:
    if value is None:
        return "-"
    return f"{value:+.3f}" if signed else f"{value:.3f}"
