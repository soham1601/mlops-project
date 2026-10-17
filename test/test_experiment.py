import json

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from mlops_project import experiment
from mlops_project.cli import main
from mlops_project.config import FEATURES
from mlops_project.settings import Settings


def exp_settings(tmp_path, data_path, **experiment_options) -> Settings:
    settings = Settings(name="test", models=["logistic_regression"])
    settings.data.path = str(data_path)
    settings.cv.folds = 3
    settings.experiment.repeats = 1
    settings.experiment.top_k = [8]
    settings.experiment.output_dir = str(tmp_path / "exp")
    for key, value in experiment_options.items():
        setattr(settings.experiment, key, value)
    settings.validate()
    return settings


def test_corrected_t_test_matches_the_formula():
    diffs = np.array([0.02, 0.01, 0.03, 0.00, 0.02, 0.04])
    ratio = 1 / 4
    result = experiment.corrected_paired_ttest(diffs, ratio)
    se = np.sqrt((1 / 6 + ratio) * diffs.var(ddof=1))
    assert result["mean_diff"] == pytest.approx(diffs.mean())
    assert result["se"] == pytest.approx(se)
    assert result["ci_low"] < result["mean_diff"] < result["ci_high"]
    assert 0 < result["p"] < 1


def test_the_correction_makes_the_test_more_cautious_than_a_plain_paired_t_test():
    from scipy import stats

    diffs = np.array([0.02, 0.01, 0.03, 0.00, 0.02, 0.04, 0.01, 0.03])
    plain = stats.ttest_1samp(diffs, 0).pvalue
    assert experiment.corrected_paired_ttest(diffs, 1 / 4)["p"] > plain


def test_identical_scores_give_no_evidence_and_a_constant_gain_gives_certainty():
    assert experiment.corrected_paired_ttest(np.zeros(5), 0.25)["p"] == 1.0
    assert experiment.corrected_paired_ttest(np.full(5, 0.01), 0.25)["p"] == 0.0


def test_one_pair_of_scores_is_not_enough():
    with pytest.raises(ValueError, match="two"):
        experiment.corrected_paired_ttest([0.1], 0.25)


def test_holm_adjustment_keeps_order_and_never_exceeds_one():
    adjusted = experiment.holm_adjust([0.01, 0.04, 0.03])
    assert adjusted == pytest.approx([0.03, 0.06, 0.06])
    assert experiment.holm_adjust([0.6, 0.9]) == [1.0, 1.0]


def test_default_variants_are_baseline_selection_engineered_and_ablation(tmp_path):
    settings = Settings()
    settings.experiment.top_k = [10, 8, 6]
    variants = experiment.make_variants(settings)
    names = [v.name for v in variants]
    assert names[0] == experiment.BASELINE
    assert names[1:5] == ["top_10", "top_8", "top_6", "plus_hr_ratio"]
    assert [n for n in names if n.startswith("drop_")] == [f"drop_{f}" for f in FEATURES]
    assert len(names) == 1 + 3 + 1 + len(FEATURES)


def test_variants_can_be_switched_off():
    settings = Settings()
    settings.experiment.ablation = False
    settings.experiment.engineered = False
    settings.experiment.top_k = []
    assert [v.name for v in experiment.make_variants(settings)] == [experiment.BASELINE]


@pytest.fixture(scope="module")
def completed(tmp_path_factory, learnable_df):
    tmp = tmp_path_factory.mktemp("experiment")
    data = tmp / "data.csv"
    learnable_df.to_csv(data, index=False)
    return experiment.run(exp_settings(tmp, data)), tmp, learnable_df


def test_every_output_is_written(completed):
    result, _, _ = completed
    from pathlib import Path

    names = {p.name for p in Path(result["output_dir"]).iterdir()}
    assert {
        "feature_experiments.csv",
        "fold_scores.csv",
        "selection_frequency.csv",
        "experiment.json",
        "feature_comparison.md",
        "variant_comparison.png",
        "ablation.png",
    } <= names


def test_summary_has_one_row_per_variant_and_a_zero_baseline(completed):
    result, _, _ = completed
    summary = pd.DataFrame(result["summary"])
    assert len(summary) == 1 + 1 + 1 + len(FEATURES)  # baseline, top_8, engineered, drops
    base = summary[summary["variant"] == experiment.BASELINE].iloc[0]
    assert base["mean_diff"] == 0 and base["verdict"] == "reference"
    assert summary["p_holm"].between(0, 1).all()
    assert (summary["p_holm"] >= summary["p"] - 1e-12).all()  # adjustment never lowers a p-value


def test_scores_are_paired_over_the_same_resamples(completed):
    result, tmp, _ = completed
    folds = pd.read_csv(tmp / "exp" / "fold_scores.csv")
    counts = folds.groupby("variant")["resample"].nunique()
    assert (counts == result["resamples"]).all()  # 3 folds x 1 repeat for every variant


def test_selection_frequency_adds_up_to_k_features_per_fold(completed):
    result, _, _ = completed
    selection = result["selection"]
    assert selection.groupby("variant")["times_selected"].sum().iloc[0] == 8 * result["resamples"]


def test_report_names_every_feature_and_is_plain_ascii(completed):
    _, tmp, _ = completed
    text = (tmp / "exp" / "feature_comparison.md").read_text(encoding="utf-8")
    assert all(f in text for f in FEATURES)
    assert "Holm" in text
    text.encode("cp1252")  # Windows consoles choke on anything else
    assert text.isascii()


def test_the_held_out_test_rows_never_influence_the_experiment(completed, tmp_path):
    """Wreck the feature values of the test rows: the results must not change by a digit."""
    first, tmp, df = completed
    settings = exp_settings(tmp_path, tmp / "data.csv")
    _, test_idx = train_test_split(
        np.arange(len(df)), test_size=settings.split.test_size,
        stratify=df["target"], random_state=settings.split.seed,
    )  # fmt: skip
    wrecked = df.astype(object)
    wrecked.loc[test_idx, FEATURES] = "?"
    data = tmp_path / "wrecked.csv"
    wrecked.to_csv(data, index=False)

    second = experiment.run(exp_settings(tmp_path, data))

    pd.testing.assert_frame_equal(
        pd.DataFrame(first["summary"]), pd.DataFrame(second["summary"]), check_exact=True
    )
    assert second["train_rows"] == first["train_rows"]


def test_ablation_can_be_skipped(tmp_path, learnable_csv):
    result = experiment.run(exp_settings(tmp_path, learnable_csv, ablation=False, engineered=False))
    assert {r["variant"] for r in result["summary"]} == {experiment.BASELINE, "top_8"}
    assert not (tmp_path / "exp" / "ablation.png").exists()


def test_experiment_command_runs_from_the_command_line(learnable_csv, tmp_path, capsys):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        f"data: {{path: {json.dumps(str(learnable_csv))}}}\n"
        "cv: {folds: 3}\n"
        "models: [logistic_regression]\n"
        f"experiment: {{repeats: 1, top_k: [8], output_dir: {json.dumps(str(tmp_path / 'e'))}}}\n",
        encoding="utf-8",
    )
    assert main(["experiment", "--config", str(cfg), "--no-ablation"]) == 0
    assert (tmp_path / "e" / "feature_comparison.md").exists()
    assert "Feature experiments" in capsys.readouterr().out
