import pytest

from mlops_project.corrupt import corrupt
from mlops_project.datasets import read_table
from mlops_project.evaluate import evaluate, format_result, score


def test_score_computes_the_five_metrics():
    metrics, cm = score([0, 1, 1, 0], [0.1, 0.9, 0.4, 0.2])
    assert metrics["accuracy"] == 0.75
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == pytest.approx(2 / 3)
    assert metrics["roc_auc"] == 1.0
    assert cm == [[2, 0], [1, 1]]


def test_roc_auc_is_none_when_only_one_class_is_present():
    metrics, _ = score([1, 1, 1], [0.9, 0.8, 0.7])
    assert metrics["roc_auc"] is None


def test_evaluating_a_messy_copy_still_works_and_reports_the_repairs(trained, tmp_path):
    result, _ = trained
    holdout = read_table(f"{result['report_dir']}/holdout_test.csv")
    messy, _ = corrupt(holdout, rate=0.2, seed=5, messy_headers=True)
    path = tmp_path / "messy.csv"
    messy.to_csv(path, index=False)

    outcome = evaluate(result["model_path"], path)

    assert sum(outcome["values_fixed_by_cleaner"].values()) > 0
    assert 0 <= outcome["metrics"]["roc_auc"] <= 1
    assert "values repaired" in format_result(outcome)


def test_more_damage_means_lower_scores(trained, tmp_path):
    """The cleaner keeps the model running, but information lost is still lost."""
    result, _ = trained
    holdout = read_table(f"{result['report_dir']}/holdout_test.csv")
    auc = {}
    for rate in (0.0, 0.6):
        messy, _ = corrupt(holdout, rate=rate, seed=2)
        path = tmp_path / f"m{rate}.csv"
        messy.to_csv(path, index=False)
        auc[rate] = evaluate(result["model_path"], path)["metrics"]["roc_auc"]
    assert auc[0.6] < auc[0.0]


def test_missing_model_error_says_how_to_create_one(tmp_path, learnable_csv):
    with pytest.raises(FileNotFoundError, match="train"):
        evaluate(tmp_path / "none.joblib", learnable_csv)


def test_data_without_a_required_column_is_rejected(trained, tmp_path, learnable_df):
    result, _ = trained
    path = tmp_path / "short.csv"
    learnable_df.drop(columns="thal").to_csv(path, index=False)
    with pytest.raises(ValueError, match="thal"):
        evaluate(result["model_path"], path)
