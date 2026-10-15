import json
from pathlib import Path

import joblib
import pytest

from mlops_project import train
from mlops_project.config import FEATURES
from mlops_project.datasets import load_xy
from mlops_project.evaluate import evaluate
from mlops_project.models import SPECS


def test_every_artefact_is_written(trained):
    result, _ = trained
    report = result["report_dir"]
    names = {p.name for p in Path(report).iterdir()}
    assert {
        "metrics.json",
        "model_comparison.md",
        "holdout_test.csv",
        "permutation_importance.csv",
        "roc_curves.png",
        "confusion_matrix.png",
        "permutation_importance.png",
        *(f"optuna_trials_{m}.csv" for m in SPECS),
    } <= names
    assert Path(result["model_path"]).exists()
    assert Path(result["model_path"]).with_name("model_card.json").exists()


def test_model_card_records_how_the_model_was_made(trained):
    result, _ = trained
    card = json.loads(open(result["model_path"].replace("model.joblib", "model_card.json")).read())
    assert card["best_model"] in SPECS
    assert card["features"] == FEATURES
    assert len(card["data"]["sha256"]) == 64
    assert {"python", "scikit-learn", "xgboost"} <= set(card["versions"])
    assert card["split"]["train_rows"] + card["split"]["test_rows"] == card["data"]["rows_used"]


def test_best_model_is_chosen_by_cross_validation_not_by_test_score(trained):
    result, _ = trained
    cv = {n: r["cv_tuned"]["roc_auc"]["mean"] for n, r in result["results"].items()}
    assert result["best_model"] == max(cv, key=cv.get)


def test_tuning_is_never_worse_than_the_default_parameters(trained):
    """The defaults are Optuna's first trial, evaluated on the same folds."""
    result, _ = trained
    for name, r in result["results"].items():
        tuned, baseline = r["cv_tuned"]["roc_auc"]["mean"], r["cv_baseline"]["roc_auc"]["mean"]
        assert tuned >= baseline - 1e-9, name


def test_models_learn_the_signal_in_the_data(trained):
    result, _ = trained
    assert result["results"][result["best_model"]]["test_metrics"]["roc_auc"] > 0.7


def test_saved_pipeline_predicts_probabilities_for_raw_rows(trained):
    result, tmp = trained
    model = joblib.load(result["model_path"])
    X, _, _ = load_xy(tmp / "data.csv")  # raw text rows, exactly what a CSV user would send
    proba = model.predict_proba(X)[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()


def test_evaluate_on_the_holdout_file_reproduces_the_training_test_scores(trained):
    result, _ = trained
    holdout = f"{result['report_dir']}/holdout_test.csv"
    outcome = evaluate(result["model_path"], holdout)
    for metric, change in outcome["change_vs_training"].items():
        assert change == pytest.approx(0, abs=1e-9), metric


def test_same_seed_gives_identical_results(tmp_path, learnable_df, make_settings):
    data = tmp_path / "d.csv"
    learnable_df.to_csv(data, index=False)
    runs = [
        train.run(make_settings(tmp_path / f"r{i}", data, ["logistic_regression"], n_trials=2))
        for i in (1, 2)
    ]
    a, b = (r["results"]["logistic_regression"] for r in runs)
    assert a["best_params"] == b["best_params"]
    assert a["test_metrics"] == b["test_metrics"]


def test_zero_trials_skips_tuning_and_uses_default_parameters(
    tmp_path, learnable_df, make_settings
):
    data = tmp_path / "d.csv"
    learnable_df.to_csv(data, index=False)
    result = train.run(make_settings(tmp_path, data, ["logistic_regression"], n_trials=0))
    r = result["results"]["logistic_regression"]
    assert r["optuna_trials"] == 0
    assert r["best_params"] == SPECS["logistic_regression"].defaults
    assert not list((tmp_path / "out" / "reports").glob("optuna_trials_*"))


def test_too_little_data_gives_a_clear_error(tmp_path, learnable_df, make_settings):
    data = tmp_path / "tiny.csv"
    learnable_df.head(12).to_csv(data, index=False)
    with pytest.raises(ValueError, match="Not enough data"):
        train.run(make_settings(tmp_path, data, ["logistic_regression"], folds=5))


def test_training_survives_a_messy_csv(tmp_path, learnable_df, make_settings):
    from mlops_project.corrupt import corrupt

    messy, _ = corrupt(learnable_df, rate=0.15, seed=3, messy_headers=True)
    data = tmp_path / "messy.csv"
    messy.to_csv(data, index=False)

    result = train.run(make_settings(tmp_path, data, ["logistic_regression"], n_trials=2))

    auc = result["results"]["logistic_regression"]["test_metrics"]["roc_auc"]
    assert 0.5 < auc <= 1


def test_text_reports_are_plain_ascii_so_they_print_on_a_windows_console(trained):
    """Windows consoles default to cp1252; characters outside it raise UnicodeEncodeError."""
    result, _ = trained
    text = (Path(result["report_dir"]) / "model_comparison.md").read_text(encoding="utf-8")
    text.encode("cp1252")
    assert text.isascii()
