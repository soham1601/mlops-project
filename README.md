# Heart Disease Risk — MLOps Project

End-to-end MLOps pipeline on the UCI Cleveland Heart Disease dataset.
XGBoost model (test AUC 0.929) served from a FastAPI container on
Kubernetes with Prometheus + Grafana monitoring.

## Quick start on Prayogshala

```bash
# Fresh Rocky Linux VM — one command installs system tools, uv, kubectl,
# helm, clones the repo and builds the Python 3.12 venv.
bash <(curl -s https://raw.githubusercontent.com/soham1601/mlops-project/master/scripts/setup_prayogshala.sh)

cd ~/workspace/mlops-project
source .venv/bin/activate

# Train + log to MLflow (needs tracking.enabled: true in configs/cleveland.yaml)
make train

# Start MLflow UI in the background
nohup .venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000 \
  --host 0.0.0.0 > logs/mlflow.log 2>&1 &

# Start Minikube + install kube-prometheus-stack + deploy heart-api from GHCR
bash scripts/deploy_prayogshala.sh
```

## Local URLs

After `scripts/deploy_prayogshala.sh` finishes, the deploy script leaves
port-forwards running for monitoring:

| Service | URL | Credentials |
| --- | --- | --- |
| MLflow UI | http://localhost:5000 | — |
| Prometheus | http://localhost:9090 | — |
| Prometheus heart-api targets | http://localhost:9090/targets?search=heart-api | — |
| Grafana | http://localhost:3000 | `admin` / password below |
| Grafana dashboard | http://localhost:3000/dashboards | — |
| FastAPI docs | http://localhost:8080/docs (after `kubectl port-forward svc/heart-api 8080:80`) | — |

Grafana password:
```bash
kubectl -n monitoring get secret monitoring-grafana \
  -o jsonpath='{.data.admin-password}' | base64 -d && echo
```

## Testing the live API

Point `BASE` at either the Docker container (`localhost:8000`) or the
Kubernetes service via port-forward (`localhost:8080`).

```bash
BASE=http://localhost:8080     # or http://localhost:8000 for standalone docker

curl $BASE/ready
# {"status":"ready"}

curl $BASE/health
# {"status":"ok","model_loaded":true}
```

### Example 1 — predicts **Disease**
```bash
curl -X POST $BASE/predict -H 'Content-Type: application/json' -d '{
  "age":57,"sex":1,"cp":4,"trestbps":140,"chol":241,"fbs":0,"restecg":0,
  "thalach":123,"exang":1,"oldpeak":0.2,"slope":2,"ca":0,"thal":7
}'
# {"prediction":1,"label":"Disease","probability":0.85,"confidence":0.85,"model":"xgboost"}
```

### Example 2 — predicts **No disease**
```bash
curl -X POST $BASE/predict -H 'Content-Type: application/json' -d '{
  "age":63,"sex":1,"cp":1,"trestbps":145,"chol":233,"fbs":1,"restecg":2,
  "thalach":150,"exang":0,"oldpeak":2.3,"slope":3,"ca":0,"thal":6
}'
# {"prediction":0,"label":"No disease","probability":0.21,"confidence":0.79,"model":"xgboost"}
```

### Load-test (populates Grafana)
```bash
for i in $(seq 1 100); do
  curl -s -X POST $BASE/predict -H 'Content-Type: application/json' -d '{
    "age":57,"sex":1,"cp":4,"trestbps":140,"chol":241,"fbs":0,"restecg":0,
    "thalach":123,"exang":1,"oldpeak":0.2,"slope":2,"ca":0,"thal":7
  }' > /dev/null
done
echo "sent 100 requests"
```

Wait ~15 s, then reload Grafana. Panels for `predictions_total`, request
rate and p50/p95 latency will light up.

## Make targets

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
├── scripts/                  # setup + deploy automation
├── .github/workflows/ci.yml  # lint → test → train → docker (pushes to GHCR on master)
├── test/                     # 150 pytest cases
├── notebooks/01_eda.ipynb    # EDA walkthrough
└── reports/                  # figures, metrics, experiment outputs
```

## GitHub Container Registry

CI pushes `ghcr.io/soham1601/mlops-project:{sha,latest}` on every push to
`master`. The chart's `image.repository` defaults to `heart-api` (local
Minikube build via `minikube image load`); override to pull from GHCR:

```bash
helm upgrade --install heart-api ./deploy/helm/heart-api \
  --set image.repository=ghcr.io/soham1601/mlops-project \
  --set image.tag=latest \
  --set image.pullPolicy=Always
```

## Links

- Repo: https://github.com/soham1601/mlops-project
- CI: https://github.com/soham1601/mlops-project/actions
- Image: https://github.com/users/soham1601/packages/container/package/mlops-project
