"""Training report figures: ROC curves, confusion matrix, permutation importance."""

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


def plot_variant_scores(summary: pd.DataFrame, metric: str, path: Path):
    """Dot plot of each feature set's cross-validated score, one colour per model."""
    data = summary[summary["group"] != "ablation"]
    variants = list(dict.fromkeys(data["variant"]))
    models = list(dict.fromkeys(data["model"]))
    fig, ax = plt.subplots(figsize=(7.5, 1.2 + 0.7 * len(variants)))
    step = 0.22
    for j, model in enumerate(models):
        part = data[data["model"] == model].set_index("variant")
        ys = [i + (j - (len(models) - 1) / 2) * step for i in range(len(variants))]
        ax.errorbar(
            part.loc[variants, f"{metric}_mean"], ys, xerr=part.loc[variants, f"{metric}_std"],
            fmt="o", color=MODEL_COLOURS.get(model, INK), ecolor=MODEL_COLOURS.get(model, INK),
            elinewidth=1.2, capsize=0, markersize=6, label=model,
        )  # fmt: skip
    labels = [f"{v} ({int(data[data['variant'] == v]['n_features'].iloc[0])})" for v in variants]
    ax.set_yticks(range(len(variants)), labels)
    ax.invert_yaxis()
    ax.set(xlabel=f"CV {metric} (mean +/- std over folds)", title="Feature sets compared")
    ax.legend(
        frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=len(models)
    )  # below the axis so it never covers a point
    _style(ax)
    ax.grid(axis="y", visible=False)
    return _save(fig, path)


def plot_ablation(summary: pd.DataFrame, metric: str, path: Path):
    """One panel per model: change in the score when a single feature is removed."""
    data = summary[summary["group"] == "ablation"].copy()
    data["feature"] = data["variant"].str.removeprefix("drop_")
    models = list(dict.fromkeys(data["model"]))
    order = data.groupby("feature")["mean_diff"].mean().sort_values().index.tolist()
    fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models) + 0.8, 5), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, model in zip(axes, models, strict=True):
        part = data[data["model"] == model].set_index("feature").loc[order]
        colour = MODEL_COLOURS.get(model, INK)
        significant = (part["p_holm"] < 0.05).to_numpy()
        ax.barh(
            order, part["mean_diff"], height=0.6,
            color=[colour if s else "white" for s in significant], edgecolor=colour, linewidth=1.5,
            xerr=[part["mean_diff"] - part["ci_low"], part["ci_high"] - part["mean_diff"]],
            error_kw={"ecolor": MUTED, "elinewidth": 1},
        )  # fmt: skip
        ax.axvline(0, color=INK, linewidth=1)
        ax.set(title=model, xlabel=f"change in CV {metric}")
        _style(ax)
        ax.grid(axis="y", visible=False)
    axes[0].invert_yaxis()
    fig.suptitle(
        "Dropping one feature at a time (left of zero = worse without it; filled = significant)",
        y=1.0, fontsize=10,
    )  # fmt: skip
    fig.tight_layout()
    return _save(fig, path)
