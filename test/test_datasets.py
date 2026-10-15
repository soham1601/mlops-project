import pandas as pd
import pytest

from mlops_project.config import FEATURES, RAW_TARGET
from mlops_project.datasets import load_xy, read_table, split_xy

UCI_ROWS = [
    "28,1,2,130,132,0,2,185,0,0,?,?,?,0",  # first row has '?': must still be seen as data
    "63,1,1,145,233,1,2,150,0,2.3,3,0,6,0",
    "67,1,4,160,286,0,2,108,1,1.5,2,3,3,2",
]


def test_headerless_uci_file_is_detected_and_named(tmp_path):
    path = tmp_path / "processed.hungarian.data"
    path.write_text("\n".join(UCI_ROWS) + "\n")

    df = read_table(path)

    assert list(df.columns) == [*FEATURES, RAW_TARGET]
    assert len(df) == 3  # the first data row was not swallowed as a header
    assert df.loc[0, "slope"] == "?"  # raw text is kept for the cleaner to report


def test_headerless_file_with_the_wrong_width_is_rejected(tmp_path):
    path = tmp_path / "bad.data"
    path.write_text("1,2,3\n4,5,6\n")
    with pytest.raises(ValueError, match="no header"):
        read_table(path)


def test_raw_num_column_is_used_and_binarised(tmp_path):
    path = tmp_path / "raw.data"
    path.write_text("\n".join(UCI_ROWS) + "\n")

    X, y, info = load_xy(path, target="target")  # 'target' absent: falls back to 'num'

    assert list(y) == [0, 0, 1]  # num 0, 0, 2 -> 0, 0, 1
    assert RAW_TARGET not in X.columns
    assert info["target_column"] == "num"


def test_rows_without_a_readable_label_are_dropped_and_counted():
    df = pd.DataFrame({"age": [50, 60, 70], "target": ["1", "?", "0"]})
    X, y, info = split_xy(df)
    assert len(X) == len(y) == 2
    assert info["rows_dropped_no_label"] == 1


def test_missing_target_column_error_lists_the_columns():
    with pytest.raises(ValueError, match="age"):
        split_xy(pd.DataFrame({"age": [1, 2]}))


def test_headers_are_case_insensitive_and_can_be_renamed(tmp_path, learnable_df):
    df = learnable_df.rename(columns={"thalach": "THALCH", "target": " Target "})
    path = tmp_path / "other.csv"
    df.to_csv(path, index=False)

    X, y, _ = load_xy(path, target="target", rename={"thalch": "thalach"})

    assert "thalach" in X.columns
    assert len(y) == len(learnable_df)


def test_missing_file_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="nope.csv"):
        read_table(tmp_path / "nope.csv")
