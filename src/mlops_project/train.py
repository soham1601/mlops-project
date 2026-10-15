"""Train, tune and compare the candidate models; save the best one.

Procedure:
  1. Stratified train/test split.
  2. For each model: 5-fold stratified cross-validation with default parameters (baseline).
  3. Optuna searches the model's parameters, maximising cross-validated ROC-AUC. The default
     parameters are the first trial, so tuning can never end up worse than the baseline.
  4. The tuned model is cross-validated again (all five metrics) and then scored on the test set.
  5. The best model by CROSS-VALIDATED score (never by test score) is saved as one joblib file
     containing the whole pipeline, together with a model card describing how it was made.
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
import sklearn
import xgboost
from sklearn.inspection import permutation_importance
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    cross_validate,
    train_test_split,
)

from mlops_project import plots
from mlops_project.config import FEATURES
from mlops_project.datasets import load_xy
from mlops_project.evaluate import METRIC_NAMES, score
from mlops_project.features import build_pipeline
from mlops_project.models import SPECS
from mlops_project.settings import Settings, resolve

log = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _folds(settings: Settings) -> StratifiedKFold:
    return StratifiedKFold(settings.cv.folds, shuffle=True, random_state=settings.split.seed)


def cv_summary(pipeline, X, y, settings: Settings) -> dict[str, dict[str, float]]:
    """Mean and standard deviation of every metric over the cross-validation folds."""
    result = cross_validate(
        pipeline, X, y, cv=_folds(settings), scoring=list(METRIC_NAMES), error_score="raise"
    )
    return {
        m: {"mean": float(result[f"test_{m}"].mean()), "std": float(result[f"test_{m}"].std())}
        for m in METRIC_NAMES
    }


def tune(spec, X, y, settings: Settings) -> optuna.Study:
    """Optuna search for ``spec``; the first trial is the default parameter set."""
    folds = _folds(settings)

    def objective(trial: optuna.Trial) -> float:
        pipeline = build_pipeline(spec.build(spec.suggest(trial)))
        return float(
            cross_val_score(pipeline, X, y, cv=folds, scoring=settings.tuning.metric).mean()
        )

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=settings.split.seed)
    )
    study.enqueue_trial(spec.defaults)
    study.optimize(objective, n_trials=settings.tuning.n_trials)
    return study


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _check_enough_data(y_train: pd.Series, folds: int) -> None:
    smallest = int(y_train.value_counts().min())
    if y_train.nunique() < 2 or smallest < folds:
        raise ValueError(
            f"Not enough data for {folds}-fold stratified CV: the rarer class has only "
            f"{smallest} training row(s). Provide more data or lower cv.folds."
        )


def run(settings: Settings) -> dict:
    """Run the whole training procedure and return a summary dict (also written to disk)."""
    data_path = resolve(settings.data.path)
    report_dir = resolve(settings.output.report_dir)
    model_path = resolve(settings.output.model_path)
    report_dir.mkdir(parents=True, exist_ok=True)

    X, y, info = load_xy(
        data_path, target=settings.data.target, sep=settings.data.sep, rename=settings.data.rename
    )
    log.info(
        "Loaded %s: %d rows, class balance %s",
        data_path.name,
        info["rows_used"],
        info["class_balance"],
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=settings.split.test_size, stratify=y, random_state=settings.split.seed
    )
    _check_enough_data(y_train, settings.cv.folds)
    log.info("Split: %d train / %d test rows", len(X_train), len(X_test))

    # The untouched test rows, saved so the exact same rows can be re-scored later
    X_test.assign(target=y_test).to_csv(report_dir / "holdout_test.csv", index=False)

    metric = settings.tuning.metric
    results: dict[str, dict] = {}
    fitted: dict[str, object] = {}
    probas: dict[str, np.ndarray] = {}

    for name in settings.models:
        spec = SPECS[name]
        log.info("[%s] baseline cross-validation", name)
        baseline = cv_summary(build_pipeline(spec.build(spec.defaults)), X_train, y_train, settings)

        params, trials = spec.defaults, 0
        if settings.tuning.n_trials > 0:
            log.info(
                "[%s] Optuna: %d trials, maximising %s", name, settings.tuning.n_trials, metric
            )
            study = tune(spec, X_train, y_train, settings)
            params, trials = study.best_params, len(study.trials)
            study.trials_dataframe().to_csv(report_dir / f"optuna_trials_{name}.csv", index=False)

        pipeline = build_pipeline(spec.build(params))
        tuned = cv_summary(pipeline, X_train, y_train, settings)
        pipeline.fit(X_train, y_train)
        proba = pipeline.predict_proba(X_test)[:, 1]
        test_metrics, cm = score(y_test, proba)

        fitted[name], probas[name] = pipeline, proba
        results[name] = {
            "best_params": params,
            "optuna_trials": trials,
            "cv_baseline": baseline,
            "cv_tuned": tuned,
            "test_metrics": test_metrics,
            "test_confusion_matrix": cm,
        }
        log.info(
            "[%s] CV %s %.3f -> %.3f | test %s %.3f",
            name,
            metric,
            baseline[metric]["mean"],
            tuned[metric]["mean"],
            metric,
            test_metrics[metric],
        )  # fmt: skip

    best = max(results, key=lambda n: results[n]["cv_tuned"][metric]["mean"])
    log.info("Best model by cross-validated %s: %s", metric, best)

    # --- artefacts -------------------------------------------------------------------------
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted[best], model_path)

    importance = permutation_importance(
        fitted[best], X_test[FEATURES], y_test, scoring="roc_auc", n_repeats=20,
        random_state=settings.split.seed,
    )  # fmt: skip
    importance_df = pd.DataFrame(
        {"mean": importance.importances_mean, "std": importance.importances_std}, index=FEATURES
    ).sort_values("mean", ascending=False)
    importance_df.to_csv(report_dir / "permutation_importance.csv")

    aucs = {n: r["test_metrics"]["roc_auc"] for n, r in results.items()}
    figures = [
        plots.plot_roc_curves(y_test, probas, aucs, report_dir / "roc_curves.png"),
        plots.plot_confusion_matrix(
            np.array(results[best]["test_confusion_matrix"]),
            f"{best}: test confusion matrix",
            report_dir / "confusion_matrix.png",
        ),
        plots.plot_importance(
            importance_df,
            f"{best}: permutation importance",
            report_dir / "permutation_importance.png",
        ),
    ]

    card = {
        "name": settings.name,
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "best_model": best,
        "best_params": results[best]["best_params"],
        "selection_metric": metric,
        "test_metrics": results[best]["test_metrics"],
        "cv_metrics": results[best]["cv_tuned"],
        "features": FEATURES,
        "data": {"file": data_path.name, "sha256": _sha256(data_path), **info},
        "split": {
            "test_size": settings.split.test_size,
            "seed": settings.split.seed,
            "train_rows": len(X_train),
            "test_rows": len(X_test),
        },  # fmt: skip
        "cv_folds": settings.cv.folds,
        "optuna_trials": settings.tuning.n_trials,
        "versions": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }
    model_path.with_name("model_card.json").write_text(
        json.dumps(card, indent=2) + "\n", encoding="utf-8"
    )
    summary = {"best_model": best, "settings_name": settings.name, "results": results}
    (report_dir / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (report_dir / "model_comparison.md").write_text(
        _comparison_table(results, best, metric), encoding="utf-8"
    )
    log.info("Saved model to %s and reports to %s", model_path, report_dir)

    return {**summary, "model_path": str(model_path), "report_dir": str(report_dir),
            "figures": [str(f) for f in figures], "card": card}  # fmt: skip


def _comparison_table(results: dict, best: str, metric: str) -> str:
    """Markdown table: baseline CV, tuned CV and held-out test scores for every model."""
    lines = [
        "# Model comparison",
        "",
        f"Selection metric: cross-validated {metric} on the training split.",
        f"Best model: **{best}**.",
        "CV columns are mean +/- std over the folds; test columns are on the held-out test split.",
        "",
        "| Model | CV AUC (default) | CV AUC (tuned) | CV F1 | Test acc. | Test prec. "
        "| Test recall | Test F1 | Test AUC |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, r in results.items():
        b, t, m = r["cv_baseline"]["roc_auc"], r["cv_tuned"], r["test_metrics"]
        star = " (best)" if name == best else ""
        lines.append(
            f"| {name}{star} | {b['mean']:.3f} +/- {b['std']:.3f} | "
            f"{t['roc_auc']['mean']:.3f} +/- {t['roc_auc']['std']:.3f} | "
            f"{t['f1']['mean']:.3f} +/- {t['f1']['std']:.3f} | "
            f"{m['accuracy']:.3f} | {m['precision']:.3f} | {m['recall']:.3f} | "
            f"{m['f1']:.3f} | {m['roc_auc']:.3f} |"
        )
    lines += ["", "## Best parameters", ""]
    for name, r in results.items():
        lines.append(f"- **{name}**: `{json.dumps(r['best_params'])}`")
    lines += [
        "",
        "Note: the tuned CV score is slightly optimistic because the parameters were chosen on",
        "the same folds. The test columns are the unbiased estimate.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit("Use: python -m mlops_project train")
