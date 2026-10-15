import json

import pandas as pd
import pytest

from mlops_project.config import CATEGORICAL_FEATURES, FEATURES, RAW_TARGET, TARGET
from mlops_project.data import preprocess as pp


def test_target_is_binary_and_matches_the_original_label(synthetic_raw):
    df, _ = pp.clean(synthetic_raw, missing="keep")
    assert set(df[TARGET].unique()) <= {0, 1}
    assert (df[TARGET] == (synthetic_raw[RAW_TARGET] > 0).astype(int)).all()


def test_original_target_column_is_removed(synthetic_raw):
    df, _ = pp.clean(synthetic_raw)
    assert RAW_TARGET not in df.columns
    assert list(df.columns) == [*FEATURES, TARGET]


def test_drop_removes_rows_with_missing_values(synthetic_raw):
    df, report = pp.clean(synthetic_raw, missing="drop")
    assert df[FEATURES].isna().sum().sum() == 0
    assert len(df) == len(synthetic_raw) - 5  # 3 missing 'ca' + 2 missing 'thal', different rows
    assert report["rows_removed"] == 5
    assert report["missing_per_column"] == {"ca": 3, "thal": 2}


def test_keep_leaves_missing_values_for_the_model_pipeline(synthetic_raw):
    df, report = pp.clean(synthetic_raw, missing="keep")
    assert len(df) == len(synthetic_raw)
    assert df["ca"].isna().sum() == 3
    assert report["rows_removed"] == 0


def test_categorical_columns_become_integers_after_drop(synthetic_raw):
    df, _ = pp.clean(synthetic_raw, missing="drop")
    for col in CATEGORICAL_FEATURES:
        assert pd.api.types.is_integer_dtype(df[col]), col


def test_clean_does_not_modify_its_input(synthetic_raw):
    before = synthetic_raw.copy()
    pp.clean(synthetic_raw)
    pd.testing.assert_frame_equal(before, synthetic_raw)


def test_unknown_missing_strategy_is_rejected(synthetic_raw):
    with pytest.raises(ValueError, match="drop"):
        pp.clean(synthetic_raw, missing="median")


def test_missing_target_is_rejected(synthetic_raw):
    synthetic_raw.loc[0, RAW_TARGET] = None
    with pytest.raises(ValueError, match="Target"):
        pp.clean(synthetic_raw)


@pytest.mark.parametrize(
    ("column", "bad_value"),
    [("age", 250), ("chol", 5), ("sex", 2), ("cp", 9), ("thal", 5)],
)
def test_validation_catches_out_of_range_values(synthetic_raw, column, bad_value):
    synthetic_raw.loc[0, column] = bad_value
    with pytest.raises(ValueError, match=column):
        pp.clean(synthetic_raw, missing="keep")


def test_run_writes_clean_csv_and_report(synthetic_raw, tmp_path):
    raw_path, out_path, report_path = tmp_path / "raw.csv", tmp_path / "c.csv", tmp_path / "r.json"
    synthetic_raw.to_csv(raw_path, index=False)

    report = pp.run(raw_path, out_path, report_path)

    assert pp.load_clean(out_path).shape == (report["rows_clean"], len(FEATURES) + 1)
    assert json.loads(report_path.read_text())["rows_raw"] == len(synthetic_raw)


def test_run_explains_how_to_fix_a_missing_raw_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="download"):
        pp.run(tmp_path / "nope.csv", tmp_path / "c.csv", tmp_path / "r.json")
