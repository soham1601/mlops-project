import pandas as pd
import pytest

from mlops_project.config import FEATURES, RAW_TARGET
from mlops_project.data import download as dl


def _write_raw_format(df: pd.DataFrame, path):
    """Write a frame the way UCI's processed.cleveland.data looks: no header, '?' for missing."""
    df.to_csv(path, header=False, index=False, na_rep="?")


def test_from_file_reads_headerless_file_and_turns_question_marks_into_nan(synthetic_raw, tmp_path):
    path = tmp_path / "processed.cleveland.data"
    _write_raw_format(synthetic_raw, path)

    df = dl.from_file(path)

    assert list(df.columns) == [*FEATURES, RAW_TARGET]
    assert len(df) == len(synthetic_raw)
    assert df["ca"].isna().sum() == 3
    assert df["thal"].isna().sum() == 2


def test_standardise_rejects_data_with_missing_columns(synthetic_raw):
    with pytest.raises(ValueError, match="missing columns"):
        dl.standardise(synthetic_raw.drop(columns=["chol"]))


def test_standardise_rejects_non_numeric_values(synthetic_raw):
    bad = synthetic_raw.astype({"age": object})
    bad.loc[0, "age"] = "abc"
    with pytest.raises(ValueError):
        dl.standardise(bad)


def test_auto_falls_back_to_url_when_ucimlrepo_fails(synthetic_raw, tmp_path, monkeypatch):
    def broken():
        raise ConnectionError("no network")

    monkeypatch.setattr(dl, "from_ucimlrepo", broken)
    monkeypatch.setattr(dl, "from_url", lambda: synthetic_raw)

    out = tmp_path / "raw.csv"
    df = dl.download("auto", out=out)

    assert out.exists()
    assert len(pd.read_csv(out)) == len(df) == len(synthetic_raw)


def test_auto_reports_manual_fallback_when_every_source_fails(tmp_path, monkeypatch):
    def broken():
        raise ConnectionError("no network")

    monkeypatch.setattr(dl, "from_ucimlrepo", broken)
    monkeypatch.setattr(dl, "from_url", broken)

    with pytest.raises(RuntimeError, match="--source file"):
        dl.download("auto", out=tmp_path / "raw.csv")


def test_file_source_requires_a_path(tmp_path):
    with pytest.raises(ValueError, match="--file"):
        dl.download("file", out=tmp_path / "raw.csv")
