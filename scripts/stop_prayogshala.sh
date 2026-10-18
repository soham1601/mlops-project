#!/usr/bin/env bash
# Stop background processes started by run_all.sh / deploy_prayogshala.sh.
# Does NOT touch Minikube by default (its state is useful across sessions).
# Pass FULL=yes to also stop Minikube.
# Pass NUKE=yes to also delete Minikube (wipes cluster — slow next start).
set +e

echo "=== stopping port-forwards ==="
pkill -f "kubectl.*port-forward" 2>/dev/null && echo "  port-forwards killed" || echo "  none running"

echo "=== stopping MLflow UI ==="
pkill -f "mlflow ui" 2>/dev/null && echo "  mlflow killed" || echo "  none running"

sleep 1
echo "=== port status ==="
ss -tln 2>/dev/null | grep -E ":5000|:8080|:9090|:3000" || echo "  all ports clear"

if [ "${FULL:-no}" = "yes" ]; then
  echo "=== stopping Minikube (state preserved) ==="
  minikube stop
fi

if [ "${NUKE:-no}" = "yes" ]; then
  echo "=== DELETING Minikube (not reversible) ==="
  minikube delete
fi

echo
echo "Done. To resume: bash scripts/run_all.sh"
