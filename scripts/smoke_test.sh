#!/usr/bin/env bash
# Quick smoke-test of the heart-api — hits /ready, both predict cases, then
# runs a 100-request load so Grafana lights up.
# Usage:
#   BASE=http://localhost:8080 bash scripts/smoke_test.sh
#   (BASE defaults to http://localhost:8080)
set -e

BASE=${BASE:-http://localhost:8080}
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/smoke_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee "$LOG") 2>&1

echo "=== BASE = $BASE ==="

echo
echo "--- /ready ---"
curl -sf $BASE/ready || { echo "/ready failed"; exit 1; }
echo

echo
echo "--- /health ---"
curl -sf $BASE/health
echo

echo
echo "--- Example 1: Disease case ---"
curl -sf -X POST $BASE/predict -H 'Content-Type: application/json' -d '{
  "age":57,"sex":1,"cp":4,"trestbps":140,"chol":241,"fbs":0,"restecg":0,
  "thalach":123,"exang":1,"oldpeak":0.2,"slope":2,"ca":0,"thal":7
}'
echo

echo
echo "--- Example 2: No-disease case ---"
curl -sf -X POST $BASE/predict -H 'Content-Type: application/json' -d '{
  "age":63,"sex":1,"cp":1,"trestbps":145,"chol":233,"fbs":1,"restecg":2,
  "thalach":150,"exang":0,"oldpeak":2.3,"slope":3,"ca":0,"thal":6
}'
echo

echo
echo "--- Load test: 100 predict requests ---"
for i in $(seq 1 100); do
  curl -sf -X POST $BASE/predict -H 'Content-Type: application/json' -d '{
    "age":57,"sex":1,"cp":4,"trestbps":140,"chol":241,"fbs":0,"restecg":0,
    "thalach":123,"exang":1,"oldpeak":0.2,"slope":2,"ca":0,"thal":7
  }' > /dev/null
done
echo "sent 100 requests"

echo
echo "--- /metrics (predictions_total) ---"
curl -s $BASE/metrics | grep -E "^predictions_total" | head -5
echo

echo
echo "=== Smoke test DONE. Log: $LOG ==="
