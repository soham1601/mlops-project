#!/usr/bin/env bash
# One-command: deploy heart-api from GHCR to Minikube + run smoke tests.
# Assumes scripts/setup_prayogshala.sh has already been run on this VM.
# Usage:
#   bash scripts/run_all.sh                   # deploy + test (fast, ~3 min)
#   TRAIN=yes bash scripts/run_all.sh         # also run make train first (adds ~15 min)
#   MLFLOW=yes bash scripts/run_all.sh        # also start MLflow UI on :5000
set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# make sure the venv is active
if [ -z "$VIRTUAL_ENV" ] && [ -d .venv ]; then
  source .venv/bin/activate
fi

mkdir -p logs

# ---- 1. OPTIONAL: train (fresh MLflow runs) ----
if [ "${TRAIN:-no}" = "yes" ]; then
  echo "=== make train (15-20 min) ==="
  make train 2>&1 | tee logs/train_$(date +%Y%m%d_%H%M%S).log
fi

# ---- 2. OPTIONAL: start MLflow UI ----
if [ "${MLFLOW:-no}" = "yes" ] && ! ss -tln 2>/dev/null | grep -q ":5000"; then
  echo "=== starting MLflow UI on http://localhost:5000 ==="
  nohup .venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000 --host 0.0.0.0 > logs/mlflow.log 2>&1 &
  sleep 3
fi

# ---- 3. deploy ----
echo "=== deploy to Minikube ==="
bash scripts/deploy_prayogshala.sh

# ---- 4. port-forward the API so smoke_test can hit localhost:8080 ----
pkill -f "port-forward svc/heart-api" 2>/dev/null || true
kubectl port-forward svc/heart-api 8080:80 > logs/api_pf.log 2>&1 &
sleep 3

# ---- 5. smoke test ----
echo "=== smoke test ==="
BASE=http://localhost:8080 bash scripts/smoke_test.sh

# ---- 6. summary ----
GRAFANA_PASS=$(kubectl -n monitoring get secret monitoring-grafana -o jsonpath='{.data.admin-password}' | base64 -d 2>/dev/null || echo "N/A")
cat << END

================ ALL DONE ================
API:         curl http://localhost:8080/ready
Swagger:     http://localhost:8080/docs
Prometheus:  http://localhost:9090/targets?search=heart-api
Grafana:     http://localhost:3000      (admin / $GRAFANA_PASS)
MLflow:      http://localhost:5000      (if MLFLOW=yes was set)

Logs:        $REPO_ROOT/logs/
END
