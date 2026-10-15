"""Data-quality profile of a CSV: what is wrong with it, before any model sees it.
"""

from __future__ import annotations

import pandas as pd

from mlops_project.config import FEATURES
from mlops_project.features import clean_features, normalise_columns


def profile_frame(df: pd.DataFrame, target: pd.Series | None = None) -> dict:
    """Return a dict describing the quality of ``df`` (features only, as read from a CSV)."""
    df = normalise_columns(df)
    missing_columns = [c for c in FEATURES if c not in df.columns]
    report: dict = {
        "rows": len(df),
        "missing_columns": missing_columns,
        "extra_columns": [c for c in df.columns if c not in FEATURES],
        "duplicate_rows": int(df.duplicated().sum()),
        "ready_for_model": not missing_columns,
        "features": {},
    }
    if target is not None:
        report["class_balance"] = {
            str(k): int(v) for k, v in target.value_counts().sort_index().items()
        }
    if missing_columns:
        return report

    cleaned, issues = clean_features(df)
    for col in FEATURES:
        counts = issues[col]
        report["features"][col] = {
            "markers": counts.get("marker", 0),
            "not_numeric": counts.get("not_numeric", 0),
            "invalid": counts.get("invalid", 0),
            "empty_after_cleaning": int(cleaned[col].isna().sum()),
            "empty_pct": round(100 * float(cleaned[col].isna().mean()), 1) if len(df) else 0.0,
        }
    return report


def format_profile(report: dict) -> str:
    """Plain-text table for the terminal."""
    lines = [f"Rows: {report['rows']}   duplicates: {report['duplicate_rows']}"]
    if "class_balance" in report:
        lines.append(f"Class balance: {report['class_balance']}")
    if report["extra_columns"]:
        lines.append(f"Ignored extra columns: {report['extra_columns']}")
    if report["missing_columns"]:
        lines.append(f"NOT READY: missing required columns {report['missing_columns']}")
        return "\n".join(lines)

    lines += [
        "",
        f"{'feature':<10}{'markers':>9}{'non-num':>9}{'invalid':>9}"
        f"{'empty after':>13}{'empty %':>9}",
    ]
    for col, f in report["features"].items():
        lines.append(
            f"{col:<10}{f['markers']:>9}{f['not_numeric']:>9}{f['invalid']:>9}"
            f"{f['empty_after_cleaning']:>13}{f['empty_pct']:>9}"
        )
    worst = max(report["features"].values(), key=lambda f: f["empty_pct"])["empty_pct"]
    verdict = (
        "READY"
        if worst < 50
        else "READY, but some features are mostly empty: expect weak predictions"
    )
    lines += ["", f"Verdict: {verdict}. Gaps are imputed inside the model pipeline."]
    return "\n".join(lines)
