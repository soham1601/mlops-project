import pytest

from mlops_project.config import ROOT
from mlops_project.models import SPECS
from mlops_project.settings import load_settings, resolve


def test_defaults_are_valid():
    settings = load_settings()
    assert settings.models == list(SPECS)
    assert settings.cv.folds == 5


def test_the_shipped_cleveland_config_loads():
    settings = load_settings(ROOT / "configs" / "cleveland.yaml")
    assert settings.name == "cleveland"
    assert settings.tuning.metric == "roc_auc"


def test_yaml_values_override_the_defaults(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("name: demo\ntuning:\n  n_trials: 3\nmodels: [xgboost]\n")
    settings = load_settings(cfg)
    assert (settings.name, settings.tuning.n_trials, settings.models) == ("demo", 3, ["xgboost"])
    assert settings.split.seed == 42  # untouched defaults remain


@pytest.mark.parametrize(
    "text",
    [
        "tuning:\n  n_trails: 3\n",  # typo inside a section
        "colour: red\n",  # unknown top-level key
        "models: [svm]\n",  # unknown model
        "split:\n  test_size: 1.5\n",
        "cv:\n  folds: 1\n",
        "tuning:\n  metric: mcc\n",
        "- just\n- a list\n",
    ],
)
def test_bad_settings_are_rejected(tmp_path, text):
    cfg = tmp_path / "bad.yaml"
    cfg.write_text(text)
    with pytest.raises(ValueError):
        load_settings(cfg)


def test_missing_config_file_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="nope.yaml"):
        load_settings(tmp_path / "nope.yaml")


def test_relative_paths_resolve_against_the_project_root(tmp_path):
    assert resolve("data/x.csv") == ROOT / "data" / "x.csv"
    assert resolve(tmp_path) == tmp_path
