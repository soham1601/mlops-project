import json

import pytest

from mlops_project.cli import main
from mlops_project.predict import SAMPLE_PATIENT, load_model, predict_records


@pytest.fixture(scope="module")
def model(trained):
    result, _ = trained
    return load_model(result["model_path"])


def test_prediction_has_label_probability_and_confidence(model):
    (out,) = predict_records(model, [SAMPLE_PATIENT])
    assert out["prediction"] in (0, 1)
    assert out["label"] == ("Disease" if out["prediction"] else "No disease")
    assert 0 <= out["probability"] <= 1
    assert out["confidence"] == pytest.approx(
        max(out["probability"], 1 - out["probability"]), abs=1e-4
    )


def test_batch_prediction_keeps_the_order(model):
    older = {**SAMPLE_PATIENT, "age": 29, "oldpeak": 0.0, "ca": 0, "thal": 3, "cp": 3}
    both = predict_records(model, [SAMPLE_PATIENT, older])
    assert [predict_records(model, [r])[0] for r in (SAMPLE_PATIENT, older)] == both


def test_missing_values_are_handled_by_the_pipeline(model):
    patient = {**SAMPLE_PATIENT, "ca": None, "thal": None}
    assert predict_records(model, [patient])[0]["prediction"] in (0, 1)


def test_missing_model_file_gives_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Train one"):
        load_model(tmp_path / "nope.joblib")


def test_predict_command_prints_json(trained, capsys):
    result, _ = trained
    assert main(["predict", "--model", result["model_path"], "--sample"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed[0]["label"] in ("Disease", "No disease")


def test_predict_command_reads_a_json_file(trained, tmp_path, capsys):
    result, _ = trained
    file = tmp_path / "patients.json"
    file.write_text(json.dumps([SAMPLE_PATIENT, SAMPLE_PATIENT]), encoding="utf-8")
    assert main(["predict", "--model", result["model_path"], "--file", str(file)]) == 0
    assert len(json.loads(capsys.readouterr().out)) == 2
