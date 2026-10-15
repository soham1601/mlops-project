"""Training report figures: ROC curves, confusion matrix, permutation importance.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve

MODEL_COLOURS = {
    "logistic_regression": "#2a78d6",
    "random_forest": "#eb6834",
    "xgboost": "#1baf7a",
}
INK = "#0b0b0b"
MUTED = "#52514e"


def _style(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(color="#e6e5e1", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED)


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_roc_curves(y_true, probas: dict[str, np.ndarray], aucs: dict[str, float], path: Path):
    """One ROC curve per model on the held-out test set."""
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name, proba in probas.items():
        fpr, tpr, _ = roc_curve(y_true, proba)
        ax.plot(fpr, tpr, color=MODEL_COLOURS.get(name, INK), linewidth=2,
                label=f"{name} (AUC {aucs[name]:.3f})")  # fmt: skip
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle="--", label="chance")
    ax.set(xlabel="False positive rate", ylabel="True positive rate (recall)",
           title="ROC curves on the held-out test set")  # fmt: skip
    ax.legend(frameon=False, loc="lower right")
    _style(ax)
    return _save(fig, path)


def plot_confusion_matrix(cm: np.ndarray, title: str, path: Path):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["No disease", "Disease"]).plot(
        ax=ax, cmap="Blues", colorbar=False, values_format="d"
    )
    ax.set_title(title)
    ax.grid(False)
    return _save(fig, path)


def plot_importance(importance: pd.DataFrame, title: str, path: Path):
    """Horizontal bars of mean permutation importance with a standard-deviation whisker."""
    data = importance.sort_values("mean")
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.barh(data.index, data["mean"], xerr=data["std"], color="#2a78d6", height=0.6,
            error_kw={"ecolor": MUTED, "elinewidth": 1})  # fmt: skip
    ax.set(xlabel="Drop in test ROC-AUC when the feature is shuffled", title=title)
    _style(ax)
    ax.grid(axis="y", visible=False)
    return _save(fig, path)
