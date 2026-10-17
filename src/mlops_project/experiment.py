"""Feature experiments: does a smaller or richer feature set change the cross-validated score?"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate, train_test_split

from mlops_project import plots
from mlops_project.config import FEATURES
from mlops_project.datasets import load_xy
from mlops_project.evaluate import METRIC_NAMES
from mlops_project.features import build_pipeline
from mlops_project.models import SPECS
from mlops_project.settings import Settings, resolve
from mlops_project.tracking import make_tracker

log = logging.getLogger(__name__)

BASELINE = f"all_{len(FEATURES)}"
ALPHA = 0.05


@dataclass(frozen=True)
class Variant:
    """One feature set to evaluate."""

    name: str
    group: str  # baseline | selection | engineered | ablation
    options: dict  # keyword arguments for build_pipeline
    n_features: int


def make_variants(settings: Settings) -> list[Variant]:
    """The baseline first, then the variants the settings ask for."""
    n = len(FEATURES)
    exp = settings.experiment
    variants = [Variant(BASELINE, "baseline", {}, n)]
    for k in sorted(set(exp.top_k), reverse=True):
        variants.append(Variant(f"top_{k}", "selection", {"top_k": k}, k))
    if exp.engineered:
        variants.append(Variant("plus_hr_ratio", "engineered", {"engineer": True}, n + 1))
    if exp.ablation:
        variants += [Variant(f"drop_{f}", "ablation", {"drop": [f]}, n - 1) for f in FEATURES]
    return variants


def corrected_paired_ttest(diffs, ratio: float) -> dict:
    """Nadeau-Bengio corrected resampled t-test on paired score differences."""
    d = np.asarray(diffs, dtype=float)
    j = len(d)
    if j < 2:
        raise ValueError("Need at least two paired scores")
    mean = float(d.mean())
    se = float(np.sqrt((1.0 / j + ratio) * d.var(ddof=1)))
    if se == 0.0:  # identical differences everywhere
        return {"mean_diff": mean, "se": 0.0, "ci_low": mean, "ci_high": mean,
                "p": 1.0 if mean == 0.0 else 0.0}  # fmt: skip
    p = float(2 * stats.t.sf(abs(mean / se), df=j - 1))
    half = float(stats.t.ppf(0.975, df=j - 1) * se)
    return {"mean_diff": mean, "se": se, "ci_low": mean - half, "ci_high": mean + half, "p": p}


def holm_adjust(p_values) -> list[float]:
    """Holm's step-down adjustment: controls the chance of any false positive among many tests."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(np.argsort(p, kind="stable")):
        running = max(running, (m - rank) * p[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted.tolist()


def verdict(row: dict, baseline_n: int) -> str:
    """A one-phrase reading of a variant, based on the adjusted p-value."""
    if row["variant"] == BASELINE:
        return "reference"
    if row["p_holm"] < ALPHA:
        return "better" if row["mean_diff"] > 0 else "worse"
    return "no detectable difference" + (" (simpler)" if row["n_features"] < baseline_n else "")


def run(settings: Settings) -> dict:
    """Run every variant for every model and write the tables, figures and report."""
    exp = settings.experiment
    out_dir = resolve(exp.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_path = resolve(settings.data.path)
    metric = settings.tuning.metric

    X, y, info = load_xy(
        data_path, target=settings.data.target, sep=settings.data.sep, rename=settings.data.rename
    )
    # Same split as `train`. The test half is deliberately thrown away: never used here.
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=settings.split.test_size, stratify=y, random_state=settings.split.seed
    )
    folds = settings.cv.folds
    cv = RepeatedStratifiedKFold(
        n_splits=folds, n_repeats=exp.repeats, random_state=settings.split.seed
    )
    resamples = folds * exp.repeats
    ratio = 1.0 / (folds - 1)
    variants = make_variants(settings)
    log.info(
        "%d variants x %d models x %d folds on %d training rows (test rows untouched)",
        len(variants), len(settings.models), resamples, len(X_train),
    )  # fmt: skip

    fold_rows: list[dict] = []
    scores: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    chosen: dict[str, Counter] = {}  # selection variant -> how often each feature was kept

    for model in settings.models:
        spec = SPECS[model]
        for variant in variants:
            log.info("[%s] %s", model, variant.name)
            pipeline = build_pipeline(
                spec.build(spec.defaults), seed=settings.split.seed, **variant.options
            )
            result = cross_validate(
                pipeline, X_train, y_train, cv=cv, scoring=list(METRIC_NAMES),
                return_estimator="top_k" in variant.options, error_score="raise",
            )  # fmt: skip
            scores[(model, variant.name)] = {m: result[f"test_{m}"] for m in METRIC_NAMES}
            for i in range(resamples):
                fold_rows.append(
                    {"model": model, "variant": variant.name, "group": variant.group, "resample": i}
                    | {m: float(result[f"test_{m}"][i]) for m in METRIC_NAMES}
                )  # fmt: skip
            if "top_k" in variant.options and variant.name not in chosen:
                # Selection looks only at the data, not the model, so one model's folds suffice.
                chosen[variant.name] = Counter(
                    c for est in result["estimator"] for c in est.named_steps["select"].selected_
                )

    summary = _summarise(settings, variants, scores, ratio)
    folds_df = pd.DataFrame(fold_rows)
    selection_df = pd.DataFrame(
        [
            {
                "variant": v,
                "feature": f,
                "times_selected": chosen[v][f],
                "share": chosen[v][f] / resamples,
            }  # fmt: skip
            for v in chosen
            for f in FEATURES
        ]
    )

    summary.to_csv(out_dir / "feature_experiments.csv", index=False)
    folds_df.to_csv(out_dir / "fold_scores.csv", index=False)
    selection_df.to_csv(out_dir / "selection_frequency.csv", index=False)
    figures = [
        plots.plot_variant_scores(summary, metric, out_dir / "variant_comparison.png"),
    ]
    if exp.ablation:
        figures.append(plots.plot_ablation(summary, metric, out_dir / "ablation.png"))

    report = {
        "data": {"file": data_path.name, **info},
        "train_rows": len(X_train),
        "folds": folds,
        "repeats": exp.repeats,
        "resamples": resamples,
        "seed": settings.split.seed,
        "metric": metric,
        "models": settings.models,
        "summary": json.loads(summary.to_json(orient="records")),
    }
    (out_dir / "experiment.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (out_dir / "feature_comparison.md").write_text(
        _markdown(report, summary, selection_df), encoding="utf-8"
    )
    log.info("Wrote %s", out_dir)
    result = {**report, "output_dir": str(out_dir), "figures": [str(f) for f in figures],
              "selection": selection_df}  # fmt: skip
    tracker = make_tracker(settings)
    with tracker.run(f"features-{settings.name}"):
        tracker.log_feature_experiment(result, summary, settings)
    return result


def _summarise(settings, variants, scores, ratio) -> pd.DataFrame:
    """One row per (model, variant): scores, difference from the baseline, test result."""
    metric = settings.tuning.metric
    base_n = len(FEATURES)
    rows: list[dict] = []
    for model in settings.models:
        base = scores[(model, BASELINE)][metric]
        model_rows = []
        for v in variants:
            s = scores[(model, v.name)]
            row = {"model": model, "variant": v.name, "group": v.group, "n_features": v.n_features}
            for m in METRIC_NAMES:
                row[f"{m}_mean"] = float(s[m].mean())
                row[f"{m}_std"] = float(s[m].std(ddof=1))
            if v.name == BASELINE:
                row |= {"mean_diff": 0.0, "ci_low": 0.0, "ci_high": 0.0, "p": 1.0, "wins": np.nan}
            else:
                diffs = s[metric] - base
                test = corrected_paired_ttest(diffs, ratio)
                row |= {k: test[k] for k in ("mean_diff", "ci_low", "ci_high", "p")}
                row["wins"] = int((diffs > 0).sum())
            model_rows.append(row)
        compared = [r for r in model_rows if r["variant"] != BASELINE]
        for r, adj in zip(compared, holm_adjust([r["p"] for r in compared]), strict=True):
            r["p_holm"] = adj
        for r in model_rows:
            r.setdefault("p_holm", 1.0)
            r["verdict"] = verdict(r, base_n)
        rows += model_rows
    return pd.DataFrame(rows)


def _signed(value: float) -> str:
    """Format with an explicit sign, without a confusing '-0.000'."""
    text = f"{value:+.3f}"
    return text.replace("-0.000", "+0.000")


def _p(value: float) -> str:
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def _table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    return lines + ["| " + " | ".join(r) + " |" for r in rows]


def _markdown(report: dict, summary: pd.DataFrame, selection: pd.DataFrame) -> str:
    metric = report["metric"]
    mean, std = f"{metric}_mean", f"{metric}_std"
    lines = [
        "# Feature experiments",
        "",
        f"Data: {report['data']['file']}, {report['train_rows']} training rows "
        "(the held-out test rows were not used).",
        f"Protocol: {report['folds']}-fold stratified CV repeated {report['repeats']} times "
        f"= {report['resamples']} paired scores per variant, identical folds for all variants.",
        f"Primary metric: {metric}. Models use default parameters (not tuned).",
        "Differences: corrected resampled t-test, then Holm adjustment. "
        f"A variant counts as better or worse only when the adjusted p < {ALPHA}.",
        "",
    ]
    for model in report["models"]:
        part = summary[summary["model"] == model]
        lines += [f"## {model}", "", "### Feature sets", ""]
        rows = [
            [
                r.variant,
                str(r.n_features),
                f"{r[mean]:.3f} +/- {r[std]:.3f}",
                _signed(r.mean_diff),
                f"[{_signed(r.ci_low)}, {_signed(r.ci_high)}]",
                _p(r.p_holm),
                r.verdict,
            ]  # fmt: skip
            for _, r in part[part["group"] != "ablation"].iterrows()
        ]
        lines += _table(
            [
                "variant",
                "features",
                f"CV {metric}",
                "change",
                "95% interval",
                "p (Holm)",
                "reading",
            ],
            rows,
        )
        ablation = part[part["group"] == "ablation"].sort_values("mean_diff")
        if len(ablation):
            lines += ["", "### Dropping one feature at a time", ""]
            rows = [
                [
                    r.variant.removeprefix("drop_"),
                    f"{r[mean]:.3f}",
                    _signed(r.mean_diff),
                    f"[{_signed(r.ci_low)}, {_signed(r.ci_high)}]",
                    _p(r.p_holm),
                    r.verdict,
                ]  # fmt: skip
                for _, r in ablation.iterrows()
            ]
            lines += _table(
                [
                    "feature dropped",
                    f"CV {metric}",
                    "change",
                    "95% interval",
                    "p (Holm)",
                    "reading",
                ],
                rows,
            )
        lines.append("")

    if len(selection):
        share = selection.pivot(index="feature", columns="variant", values="share")
        share = share[sorted(share.columns, key=lambda c: -int(c.removeprefix("top_")))]
        share = share.loc[share.mean(axis=1).sort_values(ascending=False).index]
        lines += ["## Which features did selection keep?", "",
                  "Share of the folds in which each feature was among the top k "
                  "(1.00 = always kept, 0.00 = never kept):", ""]  # fmt: skip
        lines += _table(
            ["feature", *share.columns],
            [[f, *[f"{v:.2f}" for v in row]] for f, row in share.iterrows()],
        )
        lines.append("")

    lines += [
        "## How to read this",
        "",
        "- **change** is the variant's score minus the all-features score, so a negative number "
        "means the variant did worse. The interval is for that change; if it contains 0 the data "
        "cannot separate the two.",
        "- **no detectable difference** is not proof of equality. With a few hundred patients the "
        "test has little power, so small real effects can be missed.",
        "- When two feature sets score the same, the simpler one is usually preferred: fewer "
        "inputs to collect, validate and explain.",
        "- Dropping a feature that is correlated with another (for example oldpeak and slope) "
        "often changes nothing, because the other one carries the same information.",
        "- Many variants are compared at once, which is why the p-values are Holm-adjusted. "
        "Treat any single unadjusted p-value with suspicion.",
        "- The final model is still trained on the full feature set unless you change the "
        "pipeline yourself. These results inform that choice; they do not make it.",
        "",
    ]
    return "\n".join(lines)
