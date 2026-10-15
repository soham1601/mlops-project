import json

import pytest

from mlops_project.cli import main


def test_help_lists_every_command(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])
    assert exit_info.value.code == 0
    out = capsys.readouterr().out
    for command in ("download", "preprocess", "eda", "profile", "train", "evaluate", "corrupt"):
        assert command in out


def test_unknown_command_is_rejected():
    with pytest.raises(SystemExit) as exit_info:
        main(["fly"])
    assert exit_info.value.code == 2


def test_profile_command_prints_a_report(learnable_csv, capsys):
    assert main(["profile", "--data", str(learnable_csv)]) == 0
    assert "Verdict: READY" in capsys.readouterr().out


def test_corrupt_command_writes_a_messy_file(learnable_csv, tmp_path, capsys):
    out = tmp_path / "messy.csv"
    assert main(["corrupt", "--data", str(learnable_csv), "--out", str(out), "--rate", "0.2"]) == 0
    assert out.exists()
    assert main(["profile", "--data", str(out)]) == 0
    assert "invalid" in capsys.readouterr().out


def test_train_then_evaluate_through_the_command_line(learnable_csv, tmp_path, capsys):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        f"name: cli\n"
        f"data: {{path: {json.dumps(str(learnable_csv))}}}\n"
        f"cv: {{folds: 3}}\n"
        f"tuning: {{n_trials: 1}}\n"
        f"models: [logistic_regression]\n"
        f"output: {{model_path: {json.dumps(str(tmp_path / 'm' / 'model.joblib'))}, "
        f"report_dir: {json.dumps(str(tmp_path / 'm' / 'reports'))}}}\n"
    )
    assert main(["train", "--config", str(cfg)]) == 0
    assert (tmp_path / "m" / "model.joblib").exists()

    result_file = tmp_path / "result.json"
    code = main(
        ["evaluate", "--model", str(tmp_path / "m" / "model.joblib"),
         "--data", str(learnable_csv), "--json", str(result_file)]
    )  # fmt: skip
    assert code == 0
    assert 0 <= json.loads(result_file.read_text())["metrics"]["roc_auc"] <= 1


def test_errors_are_printed_without_a_traceback(tmp_path, capsys):
    code = main(
        ["evaluate", "--model", str(tmp_path / "none.joblib"), "--data", str(tmp_path / "x.csv")]
    )
    assert code == 1
    assert "ERROR" in capsys.readouterr().err
