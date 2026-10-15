"""Feature handling: the in-pipeline data cleaner and the preprocessing/model pipeline.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from mlops_project.config import (
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
    VALID_CATEGORIES,
    VALID_RANGES,
)

# Text that means "no value" in raw UCI files and in typical hand-made CSVs.
MISSING_MARKERS = frozenset(
    {"", "?", "na", "n/a", "nan", "null", "none", "-", "--", "missing", "unknown"}
)


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy whose column names are stripped and lower-cased ('Age ' -> 'age')."""
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    return out


def check_columns(df: pd.DataFrame) -> None:
    """Raise a clear error when required feature columns are absent."""
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(
            f"Input is missing required column(s): {missing}. "
            f"Columns found: {list(df.columns)}. "
            "Use `data.rename` in the config file to map differently named columns."
        )


def clean_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Counter]]:
    """Turn a raw feature table into clean floats, and count what had to be fixed.
    Returns (cleaned DataFrame with exactly FEATURES columns, {column: Counter of fixes}).

    """
    df = normalise_columns(df)
    check_columns(df)

    cleaned = {}
    issues: dict[str, Counter] = {}
    for col in FEATURES:
        series = df[col]
        counts: Counter = Counter()

        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            numeric = series.astype("float64")
        else:
            text = series.astype("string").str.strip()
            is_marker = text.str.lower().isin(MISSING_MARKERS).fillna(False).to_numpy(dtype=bool)
            numeric = pd.to_numeric(text.mask(is_marker), errors="coerce").astype("float64")
            counts["marker"] = int(is_marker.sum())
            counts["not_numeric"] = int(
                (numeric.isna().to_numpy() & ~is_marker & text.notna().to_numpy()).sum()
            )

        if col in VALID_RANGES:
            low, high = VALID_RANGES[col]
            invalid = (numeric < low) | (numeric > high)
        else:
            invalid = ~numeric.isin(VALID_CATEGORIES[col]) & numeric.notna()
        counts["invalid"] = int(invalid.sum())

        cleaned[col] = numeric.mask(invalid)
        issues[col] = counts

    return pd.DataFrame(cleaned, index=df.index), issues


class RawCleaner(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        self.feature_names_in_ = np.array(FEATURES, dtype=object)
        return self

    def transform(self, X):
        cleaned, _ = clean_features(pd.DataFrame(X))
        return cleaned

    def get_feature_names_out(self, input_features=None):
        return np.array(FEATURES, dtype=object)


def build_preprocessor() -> ColumnTransformer:
    """Impute and scale numeric columns; impute and one-hot encode categorical ones."""
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [("num", numeric, NUMERIC_FEATURES), ("cat", categorical, CATEGORICAL_FEATURES)],
        sparse_threshold=0.0,  # always dense: simpler for every model
    )


def build_pipeline(estimator) -> Pipeline:
    """The complete model: cleaner -> preprocessing -> classifier."""
    return Pipeline(
        [
            ("cleaner", RawCleaner()),
            ("preprocess", build_preprocessor()),
            ("model", estimator),
        ]
    )
