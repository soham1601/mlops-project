# Heart Disease Risk — MLOps Project

End-to-end MLOps pipeline on the UCI Cleveland Heart Disease dataset.
XGBoost (test AUC 0.929) served from FastAPI on Kubernetes, with
MLflow tracking + Prometheus + Grafana monitoring.

## Run everything in 2 commands (fresh Prayogshala VM)

```bash
# 1. Bootstrap the VM — tools, repo, Python env (6 min)
bash <(curl -s https://raw.githubusercontent.com/soham1601/mlops-project/master/scripts/setup_prayogshala.sh)

# 2. Deploy to Minikube from GHCR + smoke-test the live API (4 min)
cd ~/workspace/mlops-project && bash scripts/run_all.sh
```

When the second command finishes it prints the URLs for the API,
Swagger, Prometheus and Grafana (with the Grafana password).

### Variants

```bash
# Also train fresh MLflow runs + start MLflow UI on :5000
TRAIN=yes MLFLOW=yes bash scripts/run_all.sh    # adds ~15 min

# Re-run just the smoke tests
bash scripts/smoke_test.sh

# Re-run just the deploy (after a chart change)
bash scripts/deploy_prayogshala.sh
```

## Scripts

| Script | What it does | Time |
| --- | --- | --- |
| `scripts/setup_prayogshala.sh` | Installs tools (dnf), clones repo, builds Python 3.12 venv | 6 min fresh, 10 s cached |
| `scripts/deploy_prayogshala.sh` | Starts Minikube, installs kube-prometheus-stack, deploys heart-api from GHCR, starts port-forwards | 3 min fresh, 30 s cached |
| `scripts/smoke_test.sh` | Hits `/ready`, two example `/predict` calls (Disease + No-disease), 100-request load | 20 s |
| `scripts/run_all.sh` | Orchestrator: optional train, deploy, port-forward API, smoke test | 4 min (deploy+test) |

All scripts are idempotent. Logs go to `logs/<script>_<timestamp>.log`
inside the repo (git-ignored).

## URLs (after `run_all.sh` finishes)

| Service | URL | Credentials |
| --- | --- | --- |
| FastAPI | http://localhost:8080 | — |
| Swagger / OpenAPI | http://localhost:8080/docs | — |
| MLflow UI (if `MLFLOW=yes`) | http://localhost:5000 | — |
| Prometheus | http://localhost:9090 | — |
| Prometheus heart-api targets | http://localhost:9090/targets?search=heart-api | — |
| Grafana | http://localhost:3000 | `admin` / see below |
| Grafana dashboard | Dashboards → "Heart disease API" | — |

Grafana password:
```bash
kubectl -n monitoring get secret monitoring-grafana \
  -o jsonpath='{.data.admin-password}' | base64 -d && echo
```

## Testing the API by hand

```bash
BASE=http://localhost:8080

# Disease case
curl -X POST $BASE/predict -H 'Content-Type: application/json' -d '{
  "age":57,"sex":1,"cp":4,"trestbps":140,"chol":241,"fbs":0,"restecg":0,
  "thalach":123,"exang":1,"oldpeak":0.2,"slope":2,"ca":0,"thal":7
}'
# {"prediction":1,"label":"Disease","probability":0.85,"confidence":0.85,"model":"xgboost"}

# No-disease case
curl -X POST $BASE/predict -H 'Content-Type: application/json' -d '{
  "age":63,"sex":1,"cp":1,"trestbps":145,"chol":233,"fbs":1,"restecg":2,
  "thalach":150,"exang":0,"oldpeak":2.3,"slope":3,"ca":0,"thal":6
}'
# {"prediction":0,"label":"No disease","probability":0.21,"confidence":0.79,"model":"xgboost"}
```

## Make targets (for development)

| Target | What it does |
| --- | --- |
| `make lint` | ruff check |
| `make test` | pytest (150 tests) |
| `make eda` | regenerate EDA plots in `reports/figures/` |
| `make train` | train 3 models with Optuna, log to MLflow, save best to `models/model.joblib` |
| `make experiment` | feature ablation + comparison (writes `reports/experiments/`) |
| `make evaluate DATA=path.csv` | score the saved model on a CSV |
| `make clean` | remove caches |

## Repo layout

```
.
├── src/mlops_project/        # training, cli, FastAPI app, preprocess
├── configs/cleveland.yaml    # dataset + model + MLflow config
├── Dockerfile                # multi-stage: trains inside the build
├── deploy/helm/heart-api/    # chart: Deployment, Service, ServiceMonitor, dashboard
├── scripts/                  # setup, deploy, smoke_test, run_all
├── .github/workflows/ci.yml  # lint → test → train → docker (pushes to GHCR on master)
├── test/                     # 150 pytest cases
├── notebooks/01_eda.ipynb    # EDA walkthrough
└── reports/                  # figures, metrics, experiment outputs
```

## CI → GHCR → cluster

Every push to `master` runs lint → test → train → docker, and the
`docker` job pushes `ghcr.io/soham1601/mlops-project:{sha,latest}` to
GitHub Container Registry. `scripts/deploy_prayogshala.sh` pulls
`:latest` from GHCR so the cluster always serves the newest image on a
re-deploy.

To deploy a different tag:
```bash
IMAGE_TAG=<sha> bash scripts/deploy_prayogshala.sh
```

## Links

- Repo: https://github.com/soham1601/mlops-project
- CI: https://github.com/soham1601/mlops-project/actions
- Image: https://github.com/users/soham1601/packages/container/package/mlops-project
