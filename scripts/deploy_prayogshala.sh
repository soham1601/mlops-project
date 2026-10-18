#!/usr/bin/env bash
# Prayogshala deploy: start minikube, install kube-prometheus-stack, deploy heart-api from GHCR
# Safe to rerun (uses helm upgrade --install, kubectl create ... || true)
set -e

CLUSTER_CPUS=${CLUSTER_CPUS:-2}
CLUSTER_MEM=${CLUSTER_MEM:-4096}
IMAGE_REPO=${IMAGE_REPO:-ghcr.io/soham1601/mlops-project}
IMAGE_TAG=${IMAGE_TAG:-latest}

echo "=== 1. Minikube ==="
if ! minikube status 2>/dev/null | grep -q "Running"; then
  minikube start --driver=docker --cpus=$CLUSTER_CPUS --memory=$CLUSTER_MEM
else
  echo "Minikube already running."
fi
kubectl get nodes

echo
echo "=== 2. kube-prometheus-stack (monitoring namespace) ==="
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts 2>/dev/null || true
helm repo update >/dev/null
kubectl create namespace monitoring 2>/dev/null || true
helm upgrade --install monitoring prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --set grafana.service.type=NodePort \
  --set prometheus.service.type=NodePort \
  --wait --timeout 10m

echo
echo "=== 3. heart-api from GHCR ==="
cd "$(dirname "$0")/.."     # repo root
helm upgrade --install heart-api ./deploy/helm/heart-api \
  --set monitoring.serviceMonitor.enabled=true \
  --set monitoring.dashboard.enabled=true \
  --set image.repository=$IMAGE_REPO \
  --set image.tag=$IMAGE_TAG \
  --set image.pullPolicy=Always
kubectl rollout status deployment/heart-api --timeout=5m

echo
echo "=== 4. port-forwards for monitoring ==="
pkill -f "port-forward svc/monitoring" 2>/dev/null || true
kubectl -n monitoring port-forward svc/monitoring-kube-prometheus-prometheus 9090:9090 >/dev/null 2>&1 &
kubectl -n monitoring port-forward svc/monitoring-grafana 3000:80 >/dev/null 2>&1 &
sleep 3

echo
echo "=== 5. verify ==="
echo "--- pods ---"
kubectl get pods -l app.kubernetes.io/name=heart-api
echo
echo "--- image source ---"
kubectl describe pod -l app.kubernetes.io/name=heart-api | grep "Image:" | head -2
echo
echo "--- service ---"
kubectl get svc heart-api
echo
CLUSTER_IP=$(kubectl get svc heart-api -o jsonpath='{.spec.clusterIP}')
echo "--- /ready via ClusterIP $CLUSTER_IP ---"
curl -s --max-time 5 http://$CLUSTER_IP/ready || echo "(not reachable from host; use 'minikube tunnel' or 'kubectl port-forward svc/heart-api 8080:80' then curl localhost:8080/ready)"
echo

GRAFANA_PASS=$(kubectl -n monitoring get secret monitoring-grafana -o jsonpath='{.data.admin-password}' | base64 -d)

echo
echo "=== DONE ==="
echo "  API:        kubectl port-forward svc/heart-api 8080:80  &  curl http://localhost:8080/ready"
echo "  Prometheus: http://localhost:9090"
echo "  Grafana:    http://localhost:3000   (admin / $GRAFANA_PASS)"
