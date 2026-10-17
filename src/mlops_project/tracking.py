"""MLflow experiment tracking for training and the feature experiments."""

from __future__ import annotations

import logging
from contextlib import nullcontext
from pathlib import Path

import joblib

from mlops_project.settings import TrackingSettings, resolve

log = logging.getLogger(__name__)

ALIAS = "champion"


def tracking_uri(cfg: TrackingSettings) -> str:
    return "sqlite:///" + resolve(cfg.db).as_posix()


class Tracker:
    """Logs runs to MLflow. Every method does nothing when tracking is disabled."""

    def __init__(self, cfg: TrackingSettings):
        self.cfg = cfg
        self.enabled = cfg.enabled
        if not self.enabled:
            return
        import mlflow

        self.mlflow = mlflow
        mlflow.set_tracking_uri(tracking_uri(cfg))
        if mlflow.get_experiment_by_name(cfg.experiment) is None:
            mlflow.create_experiment(
                cfg.experiment, artifact_location=resolve(cfg.artifact_dir).as_uri()
            )
        mlflow.set_experiment(cfg.experiment)

    def run(self, name: str, nested: bool = False):
        if not self.enabled:
            return nullcontext()
        return self.mlflow.start_run(run_name=name, nested=nested)

    def log_candidate(self, name: str, result: dict, trials=None) -> None:
        """One model: its parameters, cross-validated and test metrics, Optuna score curve."""
        if not self.enabled:
            return
        mlflow = self.mlflow
        mlflow.set_tag("model", name)
        mlflow.log_param("model", name)
        mlflow.log_param("optuna_trials", result["optuna_trials"])
        mlflow.log_params(result["best_params"])
        metrics = {}
        for stage, key in (("default", "cv_baseline"), ("tuned", "cv_tuned")):
            for metric, value in result[key].items():
                metrics[f"cv_{stage}_{metric}_mean"] = value["mean"]
                metrics[f"cv_{stage}_{metric}_std"] = value["std"]
        for metric, value in result["test_metrics"].items():
            if value is not None:
                metrics[f"test_{metric}"] = value
        (tn, fp), (fn, tp) = result["test_confusion_matrix"]
        metrics |= {"test_tn": tn, "test_fp": fp, "test_fn": fn, "test_tp": tp}
        mlflow.log_metrics(metrics)
        if trials is not None:
            for step, value in enumerate(trials["value"]):
                mlflow.log_metric("optuna_cv_score", float(value), step=step)
            keep = [
                c for c in trials.columns if c in ("number", "value") or c.startswith("params_")
            ]
            mlflow.log_table(trials[keep], "optuna_trials.json")

    def log_training(self, result: dict, settings) -> None:
        """The winning model: settings, data fingerprint, scores, figures, reports, the model."""
        if not self.enabled:
            return
        mlflow = self.mlflow
        card, best = result["card"], result["best_model"]
        mlflow.set_tags({"best_model": best, "config": settings.name})
        mlflow.log_params(
            {
                "best_model": best,
                "selection_metric": card["selection_metric"],
                "cv_folds": card["cv_folds"],
                "optuna_trials": card["optuna_trials"],
                "test_size": card["split"]["test_size"],
                "seed": card["split"]["seed"],
                "train_rows": card["split"]["train_rows"],
                "data_file": card["data"]["file"],
                "data_sha256": card["data"]["sha256"],
            }
        )
        mlflow.log_params({f"version_{k}": v for k, v in card["versions"].items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in card["test_metrics"].items() if v})
        mlflow.log_metrics({f"cv_{k}_mean": v["mean"] for k, v in card["cv_metrics"].items()})
        report_dir = Path(result["report_dir"])
        mlflow.log_artifacts(str(report_dir), artifact_path="reports")
        model_path = Path(result["model_path"])
        mlflow.log_artifact(str(model_path.with_name("model_card.json")))
        mlflow.log_artifact(str(model_path))
        info = mlflow.sklearn.log_model(
            joblib.load(model_path),
            name="model",
            serialization_format="cloudpickle",  # our RawCleaner is not in skops' trusted types
            registered_model_name=self.cfg.model_name,
        )
        version = getattr(info, "registered_model_version", None)
        if version is not None:
            self.mlflow.MlflowClient().set_registered_model_alias(
                self.cfg.model_name, ALIAS, str(version)
            )
            log.info("Registered %s version %s (alias '%s')", self.cfg.model_name, version, ALIAS)

    def log_feature_experiment(self, result: dict, summary, settings) -> None:
        """The feature experiment: one metric per (model, feature set), plus all its files."""
        if not self.enabled:
            return
        mlflow = self.mlflow
        mlflow.set_tags({"kind": "feature_experiment", "config": settings.name})
        mlflow.log_params(
            {
                "folds": result["folds"],
                "repeats": result["repeats"],
                "seed": result["seed"],
                "metric": result["metric"],
                "train_rows": result["train_rows"],
                "models": ",".join(result["models"]),
                "data_file": result["data"]["file"],
            }
        )
        metric = result["metric"]
        metrics = {}
        for row in summary.itertuples():
            prefix = f"{row.model}.{row.variant}"
            metrics[f"{prefix}.{metric}_mean"] = getattr(row, f"{metric}_mean")
            metrics[f"{prefix}.mean_diff"] = row.mean_diff
            metrics[f"{prefix}.p_holm"] = row.p_holm
        mlflow.log_metrics(metrics)
        mlflow.log_artifacts(result["output_dir"], artifact_path="experiments")


def make_tracker(settings) -> Tracker:
    return Tracker(settings.tracking)
