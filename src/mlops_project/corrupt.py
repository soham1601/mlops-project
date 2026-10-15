"""Make a messy copy of a clean CSV, to test how the model copes with bad input.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlops_project.config import CATEGORICAL_FEATURES, FEATURES

MARKERS = ["?", "NA", "n/a", "", "null"]
JUNK = ["unknown", "abc", "--", "see notes"]


def corrupt(
    df: pd.DataFrame, rate: float = 0.1, seed: int = 0, messy_headers: bool = False
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Return (damaged copy with text columns, {column: number of damaged cells})."""
    if not 0 <= rate <= 1:
        raise ValueError("rate must be between 0 and 1")
    rng = np.random.default_rng(seed)
    out = df.copy().astype(object)
    damaged: dict[str, int] = {}

    for col in (c for c in FEATURES if c in out.columns):
        n = int(round(rate * len(out)))
        rows = rng.choice(len(out), size=min(n, len(out)), replace=False)
        for row in rows:
            kind = rng.choice(["marker", "junk", "impossible"])
            if kind == "marker":
                out.at[out.index[row], col] = rng.choice(MARKERS)
            elif kind == "junk":
                out.at[out.index[row], col] = rng.choice(JUNK)
            elif col in CATEGORICAL_FEATURES:
                out.at[out.index[row], col] = 99
            else:
                value = pd.to_numeric(out.at[out.index[row], col], errors="coerce")
                value = 1.0 if pd.isna(value) else float(value)
                # Always outside every valid range: negative, or far above the maximum.
                impossible = -(abs(value) + 1) if rng.random() < 0.5 else abs(value) * 100 + 1000
                out.at[out.index[row], col] = impossible
        damaged[col] = len(rows)

    if messy_headers:
        out.columns = [f" {c.upper()} " if c in FEATURES else c for c in out.columns]
    return out, damaged
