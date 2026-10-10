"""The data here is SYNTHETIC and only used to exercise the code."""

import matplotlib
import numpy as np
import pandas as pd
import pytest

from mlops_project.config import FEATURES, RAW_TARGET

matplotlib.use("Agg")  # tests must run without a display


@pytest.fixture
def synthetic_raw() -> pd.DataFrame:
    """120 fake patients in the raw format: 0-4 target, 3 missing 'ca' and 2 missing 'thal'."""
    rng = np.random.default_rng(42)
    n = 120
    df = pd.DataFrame(
        {
            "age": rng.integers(29, 78, n),
            "sex": rng.integers(0, 2, n),
            "cp": rng.integers(1, 5, n),
            "trestbps": rng.integers(94, 201, n),
            "chol": rng.integers(126, 565, n),
            "fbs": rng.integers(0, 2, n),
            "restecg": rng.integers(0, 3, n),
            "thalach": rng.integers(71, 203, n),
            "exang": rng.integers(0, 2, n),
            "oldpeak": rng.integers(0, 40, n) / 10,
            "slope": rng.integers(1, 4, n),
            "ca": rng.integers(0, 4, n).astype(float),
            "thal": rng.choice([3.0, 6.0, 7.0], n),
            RAW_TARGET: rng.choice([0, 1, 2, 3, 4], n, p=[0.55, 0.2, 0.1, 0.1, 0.05]),
        }
    )
    df.loc[[3, 17, 50], "ca"] = np.nan
    df.loc[[8, 90], "thal"] = np.nan
    assert list(df.columns) == [*FEATURES, RAW_TARGET]
    return df
