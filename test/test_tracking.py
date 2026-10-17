import pandas as pd
import pytest

from mlops_project import experiment, train
from mlops_project.settings import Settings, TrackingSettings
from mlops_project.tracking import ALIAS, Tracker

mlflow = pytest.importorskip("mlflow")


def tracked(tmp_path, csv, models=("logistic_regression", "random_forest")):
    settings = Settings(name="test", models=list(models))
    settings.data.path = str(csv)
    settings.tuning.n_trials = 2
    settings.cv.folds = 3
    settings.output.model_path = str(tmp_path / "out" / "model.joblib")
    settings.output.report_dir = str(tmp_path / "out" / "reports")
    settings.tracking = TrackingSettings(
        enabled=True,
        db=str(tmp_path / "mlflow.db"),
        artifact_dir=str(tmp_path / "mlruns"),
        experiment="test-exp",
        model_name="test-model",
    )
    settings.validate()
    return settings


@pytest.fixture(scope="module")
def tracked_run(tmp_path_factory, learnable_df):
    tmp = tmp_path_factory.mktemp("tracked")
    csv = tmp / "data.csv"
    learnable_df.to_csv(csv, index=False)
    settings = tracked(tmp, csv)
    result = train.run(settings)
    client = mlflow.MlflowClient(tracking_uri=mlflow.get_tracking_uri())
    exp = client.get_experiment_by_name("test-exp")
    runs = client.search_runs([exp.experiment_id])
    return result, client, runs, settings


def test_disabled_tracker_does_nothing(tmp_path):
    tracker = Tracker(TrackingSettings(db=str(tmp_path / "x.db")))
    with tracker.run("anything"):
        tracker.log_candidate("m", {}, None)
    assert not (tmp_path / "x.db").exists()


def test_one_parent_run_and_one_child_run_per_model(tracked_run):
    result, _, runs, _ = tracked_run
    parents = [r for r in runs if "mlflow.parentRunId" not in r.data.tags]
    children = [r for r in runs if "mlflow.parentRunId" in r.data.tags]
    assert len(parents) == 1
    assert sorted(r.data.params["model"] for r in children) == [
        "logistic_regression",
        "random_forest",
    ]
    assert parents[0].data.params["best_model"] == result["best_model"]


def test_child_run_holds_params_metrics_and_the_optuna_curve(tracked_run):
    result, client, runs, _ = tracked_run
    child = next(r for r in runs if r.data.params.get("model") == "logistic_regression")
    expected = result["results"]["logistic_regression"]
    assert child.data.metrics["test_roc_auc"] == pytest.approx(expected["test_metrics"]["roc_auc"])
    assert "cv_tuned_roc_auc_mean" in child.data.metrics
    assert "C" in child.data.params
    history = client.get_metric_history(child.info.run_id, "optuna_cv_score")
    assert len(history) == 2  # two trials in the test settings


def test_parent_run_has_data_fingerprint_figures_and_reports(tracked_run):
    result, client, runs, _ = tracked_run
    parent = next(r for r in runs if "mlflow.parentRunId" not in r.data.tags)
    assert parent.data.params["data_sha256"] == result["card"]["data"]["sha256"]
    assert "version_scikit-learn" in parent.data.params
    names = {f.path for f in client.list_artifacts(parent.info.run_id, "reports")}
    assert {"reports/roc_curves.png", "reports/metrics.json"} <= names


def test_best_model_is_registered_with_a_champion_alias(tracked_run):
    _, client, _, settings = tracked_run
    version = client.get_model_version_by_alias("test-model", ALIAS)
    assert int(version.version) == 1
    loaded = mlflow.sklearn.load_model(f"models:/test-model@{ALIAS}")
    holdout = pd.read_csv(settings.output.report_dir + "/holdout_test.csv")
    proba = loaded.predict_proba(holdout.drop(columns="target"))[:, 1]
    assert len(proba) == len(holdout)


def test_feature_experiment_is_logged_as_its_own_run(tmp_path, learnable_csv):
    settings = tracked(tmp_path, learnable_csv, models=("logistic_regression",))
    settings.experiment.repeats = 1
    settings.experiment.top_k = [8]
    settings.experiment.output_dir = str(tmp_path / "exp")
    settings.validate()
    experiment.run(settings)
    client = mlflow.MlflowClient(tracking_uri=mlflow.get_tracking_uri())
    exp = client.get_experiment_by_name("test-exp")
    (run,) = client.search_runs([exp.experiment_id])
    assert run.data.tags["kind"] == "feature_experiment"
    assert f"logistic_regression.{experiment.BASELINE}.roc_auc_mean" in run.data.metrics
    assert "logistic_regression.top_8.p_holm" in run.data.metrics


def test_unknown_tracking_option_is_rejected(tmp_path):
    from mlops_project.settings import load_settings

    cfg = tmp_path / "c.yaml"
    cfg.write_text("tracking: {enable: true}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown setting"):
        load_settings(cfg)


def test_default_settings_leave_tracking_off():
    assert Settings().tracking.enabled is False
