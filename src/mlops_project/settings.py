"""Run settings, read from a YAML file (see configs/) and validated."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml

from mlops_project.config import FEATURES, ROOT
from mlops_project.models import SPECS

METRICS = ("roc_auc", "f1", "accuracy", "precision", "recall")


@dataclass
class DataSettings:
    path: str = "data/processed/heart_clean.csv"
    target: str = "target"  # a 0/1 column, or the raw 0-4 'num' column (any value > 0 = disease)
    rename: dict[str, str] = field(default_factory=dict)  # {"thalch": "thalach"}
    sep: str = ","


@dataclass
class SplitSettings:
    test_size: float = 0.2
    seed: int = 42


@dataclass
class CVSettings:
    folds: int = 5


@dataclass
class TuningSettings:
    n_trials: int = 30  # 0 = no tuning, use default parameters
    metric: str = "roc_auc"  # also used to choose the best model


@dataclass
class OutputSettings:
    model_path: str = "models/model.joblib"
    report_dir: str = "reports/training"


@dataclass
class ExperimentSettings:
    """Feature experiments (``python -m mlops_project experiment``)."""

    repeats: int = 5  # repeats of the k-fold CV; folds x repeats scores per variant
    top_k: list[int] = field(default_factory=lambda: [10, 8, 6])  # in-fold top-k selection
    engineered: bool = True  # also test the extra hr_ratio column
    ablation: bool = True  # also drop each feature in turn
    output_dir: str = "reports/experiments"


@dataclass
class TrackingSettings:
    enabled: bool = False  # the shipped config turns this on
    db: str = "mlflow.db"  # SQLite file holding runs and the model registry
    artifact_dir: str = "mlruns"  # plots, reports and model files of every run
    experiment: str = "heart-disease"
    model_name: str = "heart-disease-classifier"  # name in the MLflow model registry


@dataclass
class Settings:
    name: str = "cleveland"
    data: DataSettings = field(default_factory=DataSettings)
    split: SplitSettings = field(default_factory=SplitSettings)
    cv: CVSettings = field(default_factory=CVSettings)
    tuning: TuningSettings = field(default_factory=TuningSettings)
    models: list[str] = field(default_factory=lambda: list(SPECS))
    output: OutputSettings = field(default_factory=OutputSettings)
    experiment: ExperimentSettings = field(default_factory=ExperimentSettings)
    tracking: TrackingSettings = field(default_factory=TrackingSettings)

    def validate(self) -> None:
        if not 0 < self.split.test_size < 1:
            raise ValueError("split.test_size must be between 0 and 1")
        if self.cv.folds < 2:
            raise ValueError("cv.folds must be at least 2")
        if self.tuning.n_trials < 0:
            raise ValueError("tuning.n_trials cannot be negative")
        if self.tuning.metric not in METRICS:
            raise ValueError(f"tuning.metric must be one of {METRICS}")
        if self.experiment.repeats < 1:
            raise ValueError("experiment.repeats must be at least 1")
        n_features = len(FEATURES)
        if any(not 1 <= k < n_features for k in self.experiment.top_k):
            raise ValueError(f"experiment.top_k values must be between 1 and {n_features - 1}")
        unknown = [m for m in self.models if m not in SPECS]
        if unknown or not self.models:
            raise ValueError(
                f"models must be a non-empty list from {list(SPECS)}; got {self.models}"
            )


_SECTIONS = {
    "data": DataSettings,
    "split": SplitSettings,
    "cv": CVSettings,
    "tuning": TuningSettings,
    "output": OutputSettings,
    "experiment": ExperimentSettings,
    "tracking": TrackingSettings,
}


def _build(cls, values: dict, where: str):
    valid = {f.name for f in fields(cls)}
    unknown = set(values) - valid
    if unknown:
        raise ValueError(
            f"Unknown setting(s) {sorted(unknown)} in '{where}'. Valid: {sorted(valid)}"
        )
    return cls(**values)


def load_settings(path: str | Path | None = None) -> Settings:
    """Read a YAML config (or use the defaults when ``path`` is None) and validate it."""
    raw: dict = {}
    if path is not None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"{path} must contain a YAML mapping")

    top = {k: v for k, v in raw.items() if k not in _SECTIONS}
    unknown = set(top) - {"name", "models"}
    if unknown:
        raise ValueError(f"Unknown top-level setting(s) {sorted(unknown)} in {path}")
    for key, cls in _SECTIONS.items():
        top[key] = _build(cls, raw.get(key) or {}, key)
    settings = Settings(**top)
    settings.validate()
    return settings


def resolve(path: str | Path) -> Path:
    """Absolute paths stay as they are; relative ones are relative to the project root."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p
