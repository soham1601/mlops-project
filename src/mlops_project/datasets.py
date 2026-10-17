"""Load any CSV from the Heart Disease family into (features, binary target)."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from mlops_project.config import FEATURES, RAW_TARGET
from mlops_project.features import normalise_columns

RAW_COLUMNS = [*FEATURES, RAW_TARGET]


def _looks_like_data(name: object) -> bool:
    """True for a number or a '?' marker: such a 'column name' is really a data value."""
    text = str(name).strip()
    if re.fullmatch(r"\?(\.\d+)?", text):  # pandas renames a repeated '?' to '?.1', '?.2', ...
        return True
    try:
        float(text)
    except ValueError:
        return False
    return True


def read_table(
    path: str | Path, sep: str = ",", rename: dict[str, str] | None = None
) -> pd.DataFrame:
    """Read a CSV as text. Detects the header-less UCI format automatically."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    # Everything is read as text, with no automatic NA handling: the model pipeline's cleaner
    # is the single place that decides what is missing, and it reports what it repaired.
    options = {"sep": sep, "dtype": str, "keep_default_na": False, "skipinitialspace": True}
    df = pd.read_csv(path, **options)
    if len(df.columns) and all(_looks_like_data(c) for c in df.columns):
        # No header row: the first data row was read as column names. Re-read with names.
        if len(df.columns) != len(RAW_COLUMNS):
            raise ValueError(
                f"{path.name} has no header and {len(df.columns)} columns; "
                f"expected {len(RAW_COLUMNS)} ({RAW_COLUMNS})"
            )
        df = pd.read_csv(path, header=None, names=RAW_COLUMNS, **options)

    df = normalise_columns(df)
    if rename:
        df = df.rename(columns={k.strip().lower(): v for k, v in rename.items()})
    return df


def split_xy(df: pd.DataFrame, target: str = "target") -> tuple[pd.DataFrame, pd.Series, dict]:
    """Separate features from the label and turn the label into 0/1."""
    df = normalise_columns(df)
    target = target.strip().lower()
    source = next((c for c in (target, RAW_TARGET) if c in df.columns), None)
    if source is None:
        raise ValueError(
            f"Target column '{target}' (or '{RAW_TARGET}') not found. Columns: {list(df.columns)}"
        )

    label = pd.to_numeric(df[source].astype("string").str.strip(), errors="coerce")
    keep = label.notna()
    info = {
        "target_column": source,
        "rows_read": len(df),
        "rows_dropped_no_label": int((~keep).sum()),
    }

    y = (label[keep] > 0).astype(int).rename("target")
    X = df.loc[keep].drop(columns=[c for c in {target, RAW_TARGET} if c in df.columns])
    info["rows_used"] = len(X)
    info["class_balance"] = {str(k): int(v) for k, v in y.value_counts().sort_index().items()}
    return X.reset_index(drop=True), y.reset_index(drop=True), info


def load_xy(
    path: str | Path,
    target: str = "target",
    sep: str = ",",
    rename: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Read ``path`` and return (X, y, info). The one entry point used by train/evaluate."""
    return split_xy(read_table(path, sep=sep, rename=rename), target)
