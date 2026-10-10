"""

Clean the raw dataset: binary target, missing values, type fixes and range validation.
  --missing drop  (default) removes those rows, so the cleaned CSV has no gaps.
  --missing keep            leaves the gaps; the model pipeline then imputes them (Task 2).

"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from mlops_project.config import (
    CATEGORICAL_FEATURES,
    FEATURES,
    PROCESSED_PATH,
    RAW_PATH,
    RAW_TARGET,
    REPORTS_DIR,
    TARGET,
    VALID_CATEGORIES,
    VALID_RANGES,
)


def validate(df: pd.DataFrame) -> None:
    """Raise ValueError listing every value outside the documented ranges. NaNs are ignored."""
    problems = []
    for col, (low, high) in VALID_RANGES.items():
        bad = df[col].dropna()
        bad = bad[(bad < low) | (bad > high)]
        if len(bad):
            problems.append(
                f"{col}: {len(bad)} value(s) outside [{low}, {high}], e.g. {bad.iloc[0]}"
            )
    for col, allowed in VALID_CATEGORIES.items():
        bad = df[col].dropna()
        bad = bad[~bad.isin(allowed)]
        if len(bad):
            problems.append(
                f"{col}: {len(bad)} value(s) not in {sorted(allowed)}, e.g. {bad.iloc[0]}"
            )
    if TARGET in df.columns and not df[TARGET].isin([0, 1]).all():
        problems.append(f"{TARGET}: must be 0 or 1")
    if problems:
        raise ValueError("Data validation failed:\n  " + "\n  ".join(problems))


def clean(raw: pd.DataFrame, missing: str = "drop") -> tuple[pd.DataFrame, dict]:
    """Return (cleaned DataFrame, report dict). The input is not modified."""
    if missing not in ("drop", "keep"):
        raise ValueError("missing must be 'drop' or 'keep'")
    if raw[RAW_TARGET].isna().any():
        raise ValueError(f"Target column '{RAW_TARGET}' has missing values")

    df = raw.copy()
    report = {
        "rows_raw": len(df),
        "missing_per_column": {c: int(n) for c, n in df.isna().sum().items() if n},
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_strategy": missing,
    }

    # Binary target: 0 = no disease, 1-4 = disease present.
    df[TARGET] = (df[RAW_TARGET] > 0).astype(int)
    df = df.drop(columns=RAW_TARGET)

    if missing == "drop":
        df = df.dropna(subset=FEATURES)
    df = df.reset_index(drop=True)

    # Categorical codes arrive as floats because of the NaNs; store them as integers.
    for col in CATEGORICAL_FEATURES:
        if df[col].notna().all():
            df[col] = df[col].astype(int)

    validate(df)
    report["rows_clean"] = len(df)
    report["rows_removed"] = report["rows_raw"] - len(df)
    balance = df[TARGET].value_counts().sort_index()
    report["class_balance"] = {str(k): int(v) for k, v in balance.items()}
    return df, report


def load_clean(path: Path = PROCESSED_PATH) -> pd.DataFrame:
    """Load the cleaned dataset (used by EDA, training and tests)."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `make data` first.")
    return pd.read_csv(path)


def run(
    raw_path: Path = RAW_PATH,
    out_path: Path = PROCESSED_PATH,
    report_path: Path = REPORTS_DIR / "preprocessing_report.json",
    missing: str = "drop",
) -> dict:
    """Read the raw CSV, clean it, and write the cleaned CSV and a JSON report."""
    if not raw_path.exists():
        raise FileNotFoundError(f"{raw_path} not found. Run `python -m mlops_project.data.download`.")
    df, report = clean(pd.read_csv(raw_path), missing=missing)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--missing", choices=["drop", "keep"], default="drop")
    args = parser.parse_args(argv)
    try:
        report = run(missing=args.missing)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2))
    print(f"Cleaned data written to {PROCESSED_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
