import pytest

from mlops_project.config import FEATURES
from mlops_project.corrupt import corrupt
from mlops_project.profiling import format_profile, profile_frame


def test_clean_data_reports_no_problems(learnable_df):
    report = profile_frame(learnable_df[FEATURES], learnable_df["target"])
    assert report["ready_for_model"]
    assert all(f["empty_after_cleaning"] == 0 for f in report["features"].values())
    assert sum(report["class_balance"].values()) == len(learnable_df)


def test_profile_counts_the_problems_that_were_injected(learnable_df):
    messy, damaged = corrupt(learnable_df[FEATURES], rate=0.1, seed=1)
    report = profile_frame(messy)
    for col, n in damaged.items():
        f = report["features"][col]
        # every damaged cell is a marker, junk text or an impossible value, and all end up empty
        assert f["markers"] + f["not_numeric"] + f["invalid"] == n
        assert f["empty_after_cleaning"] == n


def test_profile_flags_missing_columns_instead_of_crashing(learnable_df):
    report = profile_frame(learnable_df[FEATURES].drop(columns=["chol", "age"]))
    assert not report["ready_for_model"]
    assert report["missing_columns"] == ["age", "chol"]
    assert "NOT READY" in format_profile(report)


def test_profile_text_report_has_a_row_per_feature(learnable_df):
    text = format_profile(profile_frame(learnable_df[FEATURES]))
    assert all(col in text for col in FEATURES)
    assert "Verdict: READY" in text


def test_corrupt_never_touches_the_target(learnable_df):
    messy, _ = corrupt(learnable_df, rate=0.5, seed=0)
    assert (messy["target"].astype(int) == learnable_df["target"]).all()


def test_corrupt_damages_the_requested_fraction_per_column(learnable_df):
    _, damaged = corrupt(learnable_df, rate=0.1, seed=0)
    assert set(damaged) == set(FEATURES)
    assert all(n == round(0.1 * len(learnable_df)) for n in damaged.values())


def test_corrupt_is_repeatable_for_a_seed_and_different_for_another(learnable_df):
    a, _ = corrupt(learnable_df, rate=0.2, seed=1)
    b, _ = corrupt(learnable_df, rate=0.2, seed=1)
    c, _ = corrupt(learnable_df, rate=0.2, seed=2)
    assert a.equals(b)
    assert not a.equals(c)


def test_corrupt_with_rate_zero_changes_no_values(learnable_df):
    messy, damaged = corrupt(learnable_df, rate=0, seed=0)
    assert sum(damaged.values()) == 0
    assert (messy.astype(float) == learnable_df.astype(float)).all().all()


def test_messy_headers_are_padded_and_upper_case(learnable_df):
    messy, _ = corrupt(learnable_df, rate=0.1, messy_headers=True)
    assert " AGE " in messy.columns
    assert "target" in messy.columns


def test_invalid_rate_is_rejected(learnable_df):
    with pytest.raises(ValueError, match="rate"):
        corrupt(learnable_df, rate=1.5)
