"""Feature handling: the in-pipeline data cleaner and the preprocessing/model pipeline."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import mutual_info_classif
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


# Optional engineered column, used by the feature experiments (see `experiment.py`).
ENGINEERED_NUMERIC = ["hr_ratio"]


class AddEngineered(BaseEstimator, TransformerMixin):
    """Adds ``hr_ratio``: the peak heart rate reached as a share of the age-predicted maximum.

    ``hr_ratio = thalach / (220 - age)``. A lower share means the heart could not be pushed as
    far, which is clinically meaningful and not a column the data already contains. A fixed
    formula, so nothing is learned. Gaps stay NaN and are imputed in the next stage.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        out = pd.DataFrame(X).copy()
        out["hr_ratio"] = out["thalach"] / (220.0 - out["age"])
        return out


class TopKFeatures(BaseEstimator, TransformerMixin):
    """Keep the ``k`` original columns that share the most information with the label.

    Ranking uses mutual information on the rows given to ``fit`` only. Placed inside the
    pipeline, cross-validation therefore re-ranks on each training fold and never looks at the
    validation rows, which avoids the classic mistake of choosing features on all the data and
    then cross-validating. ``selected_`` records the choice, so its stability can be reported.
    """

    def __init__(self, k: int = 10, seed: int = 42):
        self.k = k
        self.seed = seed

    def fit(self, X, y):
        frame = pd.DataFrame(X)
        filled = frame.fillna(frame.median()).fillna(0.0)
        discrete = [c in CATEGORICAL_FEATURES for c in frame.columns]
        scores = mutual_info_classif(filled, y, discrete_features=discrete, random_state=self.seed)
        order = np.argsort(-scores, kind="stable")[: self.k]
        self.scores_ = dict(zip(frame.columns, map(float, scores), strict=True))
        self.selected_ = [frame.columns[i] for i in sorted(order)]
        return self

    def transform(self, X):
        return pd.DataFrame(X)[self.selected_]

    def get_feature_names_out(self, input_features=None):
        return np.array(self.selected_, dtype=object)


def build_preprocessor(drop=(), engineer: bool = False, dynamic: bool = False) -> ColumnTransformer:
    """Impute and scale numeric columns; impute and one-hot encode categorical ones.

    ``drop`` omits original features; ``engineer`` adds the ``hr_ratio`` column. With
    ``dynamic=True`` the columns are whatever a previous selection step let through.
    """
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
    num_cols = [c for c in NUMERIC_FEATURES if c not in drop]
    cat_cols = [c for c in CATEGORICAL_FEATURES if c not in drop]
    if engineer:
        num_cols = [*num_cols, *ENGINEERED_NUMERIC]
    if dynamic:  # columns are only known once the selector has been fitted
        num_cols = _present(num_cols)
        cat_cols = _present(cat_cols)
    return ColumnTransformer(
        [("num", numeric, num_cols), ("cat", categorical, cat_cols)],
        sparse_threshold=0.0,  # always dense: simpler for every model
    )


def _present(wanted: list[str]):
    """Column selector that keeps the wanted columns which are present in the data."""
    return lambda frame: [c for c in wanted if c in frame.columns]


def build_pipeline(
    estimator, *, drop=(), top_k: int | None = None, engineer: bool = False, seed: int = 42
) -> Pipeline:
    """The complete model: cleaner -> (optional steps) -> preprocessing -> classifier.

    With no options this is exactly the model that ``train`` saves. The options exist for the
    feature experiments: ``drop`` removes original features, ``engineer`` adds ``hr_ratio``,
    and ``top_k`` keeps the k most informative columns, re-chosen on every training fold.
    """
    steps = [("cleaner", RawCleaner())]
    if engineer:
        steps.append(("engineer", AddEngineered()))
    if top_k is not None:
        steps.append(("select", TopKFeatures(k=top_k, seed=seed)))
    steps.append(
        ("preprocess", build_preprocessor(drop=drop, engineer=engineer, dynamic=top_k is not None))
    )
    steps.append(("model", estimator))
    return Pipeline(steps)
