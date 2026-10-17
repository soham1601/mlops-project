"""
download    fetch the raw UCI data
preprocess  clean it and write data/processed/heart_clean.csv
eda         write the EDA figures and summary
profile     data-quality report for any CSV
train       train, tune and compare models; save the best
experiment  compare feature sets: in-fold selection, an engineered column, ablation
evaluate    score a saved model on any CSV
corrupt     write a deliberately messy copy of a CSV, for stress tests
all         download -> preprocess -> train
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from mlops_project.settings import Settings, load_settings, resolve


def _add_data_options(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", help="YAML settings file (see configs/)")
    p.add_argument("--target", help="target column name (default: from config, else 'target')")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m mlops_project", description=__doc__.split("\n\n")[0]
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="command")

    p = sub.add_parser("download", help="fetch the raw UCI data")
    p.add_argument("--source", choices=["auto", "ucimlrepo", "url", "file"], default="auto")
    p.add_argument("--file", help="local file (with --source file)")

    p = sub.add_parser("preprocess", help="clean the raw data")
    p.add_argument("--missing", choices=["drop", "keep"], default="drop")

    sub.add_parser("eda", help="write EDA figures and summary")

    p = sub.add_parser("profile", help="data-quality report for a CSV")
    p.add_argument("--data", required=True, help="CSV file to inspect")
    _add_data_options(p)

    for name, help_text in (
        ("train", "train and compare models"),
        ("all", "download, preprocess, train"),
    ):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--config", help="YAML settings file (see configs/)")
        p.add_argument("--n-trials", type=int, help="Optuna trials per model (0 = no tuning)")
        if name == "train":
            p.add_argument("--data", help="train on this CSV instead of the one in the config")
            p.add_argument("--target", help="target column name")
            p.add_argument(
                "--models", help="comma-separated subset, e.g. logistic_regression,xgboost"
            )

    p = sub.add_parser("experiment", help="compare feature sets with paired statistics")
    p.add_argument("--config", help="YAML settings file (see configs/)")
    p.add_argument("--models", help="comma-separated subset, e.g. logistic_regression")
    p.add_argument("--repeats", type=int, help="repeats of the k-fold CV (default: from config)")
    p.add_argument("--no-ablation", action="store_true", help="skip the drop-one-feature runs")

    p = sub.add_parser("corrupt", help="write a deliberately messy copy of a CSV")
    p.add_argument("--data", required=True, help="clean CSV to damage")
    p.add_argument("--out", required=True, help="where to write the messy copy")
    p.add_argument("--rate", type=float, default=0.1, help="fraction of cells damaged per column")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--messy-headers", action="store_true", help="upper-case and pad the headers")

    p = sub.add_parser("evaluate", help="score a saved model on a CSV")
    p.add_argument("--model", default=str(resolve("models/model.joblib")), help="saved model file")
    p.add_argument("--data", required=True, help="CSV file to score")
    p.add_argument("--json", help="also write the result to this JSON file")
    _add_data_options(p)
    return parser


def _settings(args: argparse.Namespace) -> Settings:
    settings = load_settings(getattr(args, "config", None))
    if getattr(args, "data", None) and args.command == "train":
        settings.data.path = str(Path(args.data).resolve())
    if getattr(args, "target", None):
        settings.data.target = args.target
    if getattr(args, "n_trials", None) is not None:
        settings.tuning.n_trials = args.n_trials
    if getattr(args, "repeats", None) is not None:
        settings.experiment.repeats = args.repeats
    if getattr(args, "no_ablation", False):
        settings.experiment.ablation = False
    if getattr(args, "models", None):
        settings.models = [m.strip() for m in args.models.split(",") if m.strip()]
    settings.validate()
    return settings


def _train(args: argparse.Namespace) -> int:
    from mlops_project import train

    result = train.run(_settings(args))
    print(Path(result["report_dir"], "model_comparison.md").read_text(encoding="utf-8"))
    print(f"Best model: {result['best_model']}  ->  {result['model_path']}")
    return 0


def _experiment(args: argparse.Namespace) -> int:
    from mlops_project import experiment

    result = experiment.run(_settings(args))
    print(Path(result["output_dir"], "feature_comparison.md").read_text(encoding="utf-8"))
    print(f"Written to {result['output_dir']}")
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    from mlops_project.evaluate import evaluate, format_result

    settings = _settings(args)
    result = evaluate(
        args.model, args.data, target=settings.data.target,
        sep=settings.data.sep, rename=settings.data.rename,
    )  # fmt: skip
    print(format_result(result))
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


def _corrupt(args: argparse.Namespace) -> int:
    from mlops_project.corrupt import corrupt
    from mlops_project.datasets import read_table

    messy, damaged = corrupt(
        read_table(args.data), rate=args.rate, seed=args.seed, messy_headers=args.messy_headers
    )
    messy.to_csv(args.out, index=False)
    print(f"Wrote {args.out}: {len(messy)} rows, cells damaged per column {damaged}")
    return 0


def _profile(args: argparse.Namespace) -> int:
    from mlops_project.datasets import read_table, split_xy
    from mlops_project.profiling import format_profile, profile_frame

    settings = _settings(args)
    df = read_table(args.data, sep=settings.data.sep, rename=settings.data.rename)
    try:
        X, y, _ = split_xy(df, settings.data.target)
    except ValueError:  # no label column: still profile the features
        X, y = df, None
    print(format_profile(profile_frame(X, y)))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    try:
        if args.command == "download":
            from mlops_project.data import download

            extra = ["--file", args.file] if args.file else []
            return download.main(["--source", args.source, *extra])
        if args.command == "preprocess":
            from mlops_project.data import preprocess

            return preprocess.main(["--missing", args.missing])
        if args.command == "eda":
            from mlops_project.data import eda

            return eda.main()
        if args.command == "profile":
            return _profile(args)
        if args.command == "train":
            return _train(args)
        if args.command == "experiment":
            return _experiment(args)
        if args.command == "corrupt":
            return _corrupt(args)
        if args.command == "evaluate":
            return _evaluate(args)
        if args.command == "all":
            from mlops_project.data import download, preprocess

            for step in (lambda: download.main([]), lambda: preprocess.main([])):
                if step() != 0:
                    return 1
            return _train(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 2
