# Heart Disease MLOps Pipeline

End-to-end MLOps project for the UCI Heart Disease (Cleveland) dataset: data, modelling,
experiment tracking, packaging, CI/CD, containerisation, Kubernetes deployment and monitoring.

> Status: work in progress. See the task table at the bottom.

## Prerequisites

- Linux or WSL2 (Ubuntu 24.04), Python 3.12, [`uv`](https://docs.astral.sh/uv/), `make`, `git`
- Later tasks also need Docker, Minikube, kubectl and Helm

## Quick start (clean setup)

```bash
git clone https://github.com/soham1601/mlops-project.git
cd mlops-project
make setup     # creates .venv and installs requirements-dev.txt
make data      # downloads the dataset, writes data/processed/heart_clean.csv
make eda       # writes figures to reports/figures/ and reports/eda_summary.md
make test      # runs the unit tests
make lint      # style check (same check CI runs)
```

`make help` lists every command.

## Getting the data

`make data` tries the official `ucimlrepo` package (dataset id 45), then the plain file on the
UCI archive. If both fail (for example behind a firewall), download `processed.cleveland.data`
manually from <https://archive.ics.uci.edu/dataset/45/heart+disease> and run:

```bash
.venv/bin/python -m mlops_project.data.download --source file --file path/to/processed.cleveland.data
.venv/bin/python -m mlops_project.data.preprocess
```

## Data cleaning decisions

- **Target:** the original label `num` (0 to 4) becomes binary `target`: 0 = no disease, 1 = disease.
- **Missing values:** 6 patients have a missing `ca` or `thal`. By default they are dropped
  (`--missing drop`), which leaves 297 rows if the file matches UCI's description; the exact counts are
  recorded in `reports/preprocessing_report.json`. `--missing keep` retains them for the model pipeline to impute.
- **Validation:** every value is checked against the documented range of its column
  (`mlops_project.config.VALID_RANGES` and `VALID_CATEGORIES`); bad data stops the pipeline.
- Encoding and scaling happen inside the model pipeline (Task 2), fitted on training folds only.

## Project layout

```
src/mlops_project/      config.py, data/{download,preprocess,eda}.py
test/                 pytest unit tests
notebooks/            01_eda.ipynb
data/processed/       cleaned dataset (committed)
reports/              EDA figures and summaries
```

## Task progress

| # | Task | Status |
| --- | --- | --- |
| 1 | Data acquisition and EDA | done |
| 2 | Feature engineering and model development | todo |
| 3 | Experiment tracking (MLflow) | todo |
| 4 | Model packaging and reproducibility | todo |
| 5 | CI/CD pipeline and automated testing | todo |
| 6 | Model containerisation | todo |
| 7 | Production deployment (Kubernetes) | todo |
| 8 | Monitoring and logging | todo |
| 9 | Documentation and report | todo |
