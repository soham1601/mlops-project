"""The candidate classifiers: default settings and the Optuna search space for each."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

SEED = 42


@dataclass(frozen=True)
class ModelSpec:
    """How to build one model, its starting parameters, and how Optuna should vary them."""

    name: str
    build: Callable[[dict], ClassifierMixin]
    defaults: dict
    suggest: Callable  # (optuna.Trial) -> dict of parameters


def _logreg(params: dict) -> LogisticRegression:
    return LogisticRegression(max_iter=2000, random_state=SEED, **params)


def _logreg_space(trial) -> dict:
    return {
        "C": trial.suggest_float("C", 1e-3, 1e2, log=True),
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
    }


def _forest(params: dict) -> RandomForestClassifier:
    return RandomForestClassifier(random_state=SEED, n_jobs=1, **params)


def _forest_space(trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
        "max_depth": trial.suggest_int("max_depth", 2, 12),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),
    }


def _xgb(params: dict) -> XGBClassifier:
    return XGBClassifier(
        random_state=SEED, n_jobs=1, tree_method="hist", eval_metric="logloss", **params
    )


def _xgb_space(trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 400, step=50),
        "max_depth": trial.suggest_int("max_depth", 2, 6),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 7),
    }


# Defaults are chosen inside each search space, so Optuna can start from them (see train.tune).
SPECS: dict[str, ModelSpec] = {
    "logistic_regression": ModelSpec(
        "logistic_regression", _logreg, {"C": 1.0, "class_weight": None}, _logreg_space
    ),
    "random_forest": ModelSpec(
        "random_forest",
        _forest,
        {"n_estimators": 300, "max_depth": 6, "min_samples_leaf": 2, "max_features": "sqrt"},
        _forest_space,
    ),
    "xgboost": ModelSpec(
        "xgboost",
        _xgb,
        {
            "n_estimators": 200,
            "max_depth": 3,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 1,
        },
        _xgb_space,
    ),
}
