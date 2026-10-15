from mlops_project.data import eda
from mlops_project.data.preprocess import clean


def test_run_all_creates_every_figure(synthetic_raw, tmp_path):
    df, _ = clean(synthetic_raw)

    paths = eda.run_all(df, tmp_path)

    assert {p.name for p in paths} == {
        "class_balance.png",
        "numeric_histograms.png",
        "categorical_counts.png",
        "correlation_heatmap.png",
        "boxplots_by_target.png",
    }
    assert all(p.stat().st_size > 1000 for p in paths)


def test_summary_reports_class_balance_and_correlations(synthetic_raw, tmp_path):
    df, _ = clean(synthetic_raw)

    text = eda.write_summary(df, tmp_path / "summary.md").read_text()

    assert "## Class balance" in text
    assert "## Correlation with the target" in text
    assert f"Rows: {len(df)}" in text
