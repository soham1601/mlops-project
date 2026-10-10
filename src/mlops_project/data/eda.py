"""
Exploratory data analysis: saves figures to reports/figures and a text summary to reports.

"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from mlops_project.config import (
    CATEGORICAL_FEATURES,
    DESCRIPTIONS,
    FIGURES_DIR,
    NUMERIC_FEATURES,
    REPORTS_DIR,
    TARGET,
    TARGET_LABELS,
)

PALETTE = {"No disease": "#4C78A8", "Disease": "#E45756"}
HUE_ORDER = list(TARGET_LABELS.values())


def _labelled(df: pd.DataFrame) -> pd.DataFrame:
    """Copy of df with a readable 'outcome' column for plot legends."""
    out = df.copy()
    out["outcome"] = out[TARGET].map(TARGET_LABELS)
    return out


def _save(fig: plt.Figure, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_class_balance(df: pd.DataFrame, out_dir: Path = FIGURES_DIR) -> Path:
    data = _labelled(df)
    order = list(TARGET_LABELS.values())
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.countplot(data=data, x="outcome", order=order, hue="outcome", palette=PALETTE, ax=ax)
    for bar in ax.patches:
        n = int(bar.get_height())
        ax.annotate(f"{n} ({n / len(data):.0%})", (bar.get_x() + bar.get_width() / 2, n),
                    ha="center", va="bottom")  # fmt: skip
    ax.set(title="Class balance", xlabel="", ylabel="Patients")
    return _save(fig, out_dir, "class_balance.png")


def plot_numeric_histograms(df: pd.DataFrame, out_dir: Path = FIGURES_DIR) -> Path:
    data = _labelled(df)
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, col in zip(axes.flat, NUMERIC_FEATURES, strict=False):
        sns.histplot(
            data=data, x=col, hue="outcome", hue_order=HUE_ORDER, palette=PALETTE, kde=True,
            bins=20, ax=ax,
        )  # fmt: skip
        ax.set(title=DESCRIPTIONS[col].split(" (")[0], xlabel=col)
    for ax in axes.flat[len(NUMERIC_FEATURES) :]:
        ax.set_visible(False)
    fig.suptitle("Distribution of numeric features", y=1.01)
    fig.tight_layout()
    return _save(fig, out_dir, "numeric_histograms.png")


def plot_categorical_counts(df: pd.DataFrame, out_dir: Path = FIGURES_DIR) -> Path:
    data = _labelled(df)
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, col in zip(axes.flat, CATEGORICAL_FEATURES, strict=False):
        sns.countplot(
            data=data, x=col, hue="outcome", hue_order=HUE_ORDER, palette=PALETTE, ax=ax
        )  # fmt: skip
        ax.set(title=col, ylabel="Patients")
        ax.get_legend().set_title("")
    fig.suptitle("Categorical features by outcome", y=1.01)
    fig.tight_layout()
    return _save(fig, out_dir, "categorical_counts.png")


def plot_correlation_heatmap(df: pd.DataFrame, out_dir: Path = FIGURES_DIR) -> Path:
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1,
                square=True, linewidths=0.5, ax=ax)  # fmt: skip
    ax.set_title("Correlation heatmap (Pearson)")
    return _save(fig, out_dir, "correlation_heatmap.png")


def plot_boxplots_by_target(df: pd.DataFrame, out_dir: Path = FIGURES_DIR) -> Path:
    data = _labelled(df)
    fig, axes = plt.subplots(1, len(NUMERIC_FEATURES), figsize=(16, 4))
    for ax, col in zip(axes, NUMERIC_FEATURES, strict=True):
        sns.boxplot(data=data, x="outcome", y=col, order=list(TARGET_LABELS.values()),
                    hue="outcome", palette=PALETTE, ax=ax)  # fmt: skip
        ax.set(xlabel="", title=col)
    fig.suptitle("Numeric features by outcome", y=1.03)
    fig.tight_layout()
    return _save(fig, out_dir, "boxplots_by_target.png")


def write_summary(df: pd.DataFrame, out_path: Path = REPORTS_DIR / "eda_summary.md") -> Path:
    """Write the key numbers behind the plots, as raw material for your own observations."""
    balance = df[TARGET].value_counts().sort_index()
    corr_target = df.corr(numeric_only=True)[TARGET].drop(TARGET)
    top = corr_target.reindex(corr_target.abs().sort_values(ascending=False).index)

    q1, q3 = df[NUMERIC_FEATURES].quantile(0.25), df[NUMERIC_FEATURES].quantile(0.75)
    iqr = q3 - q1
    numeric = df[NUMERIC_FEATURES]
    outliers = ((numeric < q1 - 1.5 * iqr) | (numeric > q3 + 1.5 * iqr)).sum()

    lines = [
        "# EDA summary",
        "",
        f"- Rows: {len(df)}, columns: {df.shape[1]}",
        f"- Missing values: {int(df.isna().sum().sum())}",
        "",
        "## Class balance",
        *[f"- {TARGET_LABELS[k]}: {v} ({v / len(df):.1%})" for k, v in balance.items()],
        "",
        "## Correlation with the target (Pearson, strongest first)",
        *[f"- {name}: {value:+.3f}" for name, value in top.items()],
        "",
        "## Outliers by the 1.5 x IQR rule (numeric features)",
        *[f"- {name}: {n}" for name, n in outliers.items()],
        "",
        "## Numeric summary",
        "```",
        df[NUMERIC_FEATURES].describe().round(2).to_string(),
        "```",
        "",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    return out_path


def run_all(df: pd.DataFrame, out_dir: Path = FIGURES_DIR) -> list[Path]:
    """Create every figure and return their paths."""
    sns.set_theme(style="whitegrid")
    return [
        plot_class_balance(df, out_dir),
        plot_numeric_histograms(df, out_dir),
        plot_categorical_counts(df, out_dir),
        plot_correlation_heatmap(df, out_dir),
        plot_boxplots_by_target(df, out_dir),
    ]


def main() -> int:
    plt.switch_backend("Agg")  # no display needed (WSL, CI)
    from mlops_project.data.preprocess import load_clean

    df = load_clean()
    paths = run_all(df)
    summary = write_summary(df)
    for path in [*paths, summary]:
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
