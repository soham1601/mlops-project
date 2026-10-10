"""
Download the UCI Heart Disease (Cleveland) dataset and save it as a raw CSV.

"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from mlops_project.config import FEATURES, RAW_PATH, RAW_TARGET

UCI_REPO_ID = 45
UCI_DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/"
    "processed.cleveland.data"
)
EXPECTED_ROWS = 303  # the Cleveland subset has 303 patients
RAW_COLUMNS = [*FEATURES, RAW_TARGET]


def standardise(df: pd.DataFrame) -> pd.DataFrame:
    """Check that all expected columns exist, then return them in a fixed order as numbers."""
    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Downloaded data is missing columns: {missing}")
    return df[RAW_COLUMNS].apply(pd.to_numeric, errors="raise").reset_index(drop=True)


def from_ucimlrepo() -> pd.DataFrame:
    """Fetch the dataset with the official ``ucimlrepo`` package."""
    from ucimlrepo import fetch_ucirepo

    dataset = fetch_ucirepo(id=UCI_REPO_ID)
    return standardise(pd.concat([dataset.data.features, dataset.data.targets], axis=1))


def from_file(path_or_url: str | Path) -> pd.DataFrame:
    """Read the headerless ``processed.cleveland.data`` format, where '?' marks a missing value."""
    df = pd.read_csv(path_or_url, header=None, names=RAW_COLUMNS, na_values="?")
    return standardise(df)


def from_url(url: str = UCI_DATA_URL) -> pd.DataFrame:
    """Download ``processed.cleveland.data`` straight from the UCI archive."""
    return from_file(url)


def download(source: str = "auto", file: str | Path | None = None, out: Path = RAW_PATH):
    """Fetch the data from the chosen source and write it to ``out``. Returns the DataFrame."""
    if source == "file":
        if file is None:
            raise ValueError("--source file needs --file PATH")
        df = from_file(file)
    elif source == "ucimlrepo":
        df = from_ucimlrepo()
    elif source == "url":
        df = from_url()
    elif source == "auto":
        df = _auto()
    else:
        raise ValueError(f"Unknown source: {source}")

    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return df


def _auto() -> pd.DataFrame:
    errors = []
    for name, fetch in (("ucimlrepo", from_ucimlrepo), ("url", from_url)):
        try:
            print(f"Trying source: {name} ...")
            return fetch()
        except Exception as exc:  # network errors differ by library, so catch broadly here
            errors.append(f"  - {name}: {type(exc).__name__}: {exc}")
            print(f"  failed ({type(exc).__name__}), trying next source")
    raise RuntimeError(
        "All automatic sources failed:\n" + "\n".join(errors) + "\n\n"
        "Manual fallback: download 'processed.cleveland.data'"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", choices=["auto", "ucimlrepo", "url", "file"], default="auto")
    parser.add_argument("--file", help="Local file or URL (only with --source file)")
    args = parser.parse_args(argv)

    try:
        df = download(args.source, args.file)
    except (RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Saved {len(df)} rows x {df.shape[1]} columns to {RAW_PATH}")
    print(f"Missing values per column: {df.isna().sum()[df.isna().any()].to_dict() or 'none'}")
    if len(df) != EXPECTED_ROWS:
        print(f"WARNING: expected {EXPECTED_ROWS} rows for the Cleveland data, got {len(df)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
