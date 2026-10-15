"""Shared test fixtures. The data here is SYNTHETIC and only used to exercise the code."""

import os

# Must run before NumPy and other numerical libraries are imported.
for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_name, "1")

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # tests must run without a display

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from mlops_project.config import FEATURES, RAW_TARGET  # noqa: E402


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


def _learnable(n: int = 240, seed: int = 7) -> pd.DataFrame:
    """Fake patients whose label really depends on the features, so models can learn it."""
    rng = np.random.default_rng(seed)
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
            "ca": rng.integers(0, 4, n),
            "thal": rng.choice([3, 6, 7], n),
        }
    )
    logit = (
        0.04 * (df["age"] - 55)
        + 1.0 * (df["cp"] == 4)
        + 1.0 * df["exang"]
        + 0.8 * df["oldpeak"]
        - 0.03 * (df["thalach"] - 150)
        + 0.8 * df["ca"]
        + 1.0 * (df["thal"] == 7)
        - 1.5
    )
    df["target"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return df


@pytest.fixture(scope="session")
def learnable_df() -> pd.DataFrame:
    """240 clean synthetic rows (13 features + binary 'target') with a learnable signal."""
    return _learnable()


@pytest.fixture
def learnable_csv(tmp_path, learnable_df):
    path = tmp_path / "learnable.csv"
    learnable_df.to_csv(path, index=False)
    return path


@pytest.fixture
def make_settings():
    """Factory for small, fast training settings that write everything under a temp folder."""
    from mlops_project.models import SPECS
    from mlops_project.settings import Settings

    def _make(tmp_path, data_path, models=None, n_trials=2, folds=3) -> Settings:
        settings = Settings(name="test", models=models or list(SPECS))
        settings.data.path = str(data_path)
        settings.tuning.n_trials = n_trials
        settings.cv.folds = folds
        settings.output.model_path = str(tmp_path / "out" / "model.joblib")
        settings.output.report_dir = str(tmp_path / "out" / "reports")
        settings.validate()
        return settings

    return _make


@pytest.fixture(scope="session")
def trained(tmp_path_factory, learnable_df):
    """One small training run (3 models, 2 Optuna trials each), shared by the whole session."""
    from mlops_project import train
    from mlops_project.models import SPECS
    from mlops_project.settings import Settings

    tmp = tmp_path_factory.mktemp("trained")
    data = tmp / "data.csv"
    learnable_df.to_csv(data, index=False)
    settings = Settings(name="test", models=list(SPECS))
    settings.data.path = str(data)
    settings.tuning.n_trials = 2
    settings.cv.folds = 3
    settings.output.model_path = str(tmp / "out" / "model.joblib")
    settings.output.report_dir = str(tmp / "out" / "reports")
    return train.run(settings), tmp
