#!/usr/bin/env bash
# Prayogshala Rocky Linux one-command setup
# Safe to rerun on any fresh VM (idempotent)
# Installs system tools + Python env + project, so `make train` works
set -e

echo "=== 1. system packages ==="
for pkg in make git tmux curl wget rsync jq conntrack; do
  which $pkg >/dev/null 2>&1 || sudo dnf install -y $pkg
done

echo "=== 2. uv (fast Python package manager) ==="
if ! which uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
grep -q '.local/bin' ~/.bashrc 2>/dev/null || echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc

echo "=== 3. kubectl ==="
if ! which kubectl >/dev/null 2>&1; then
  sudo tee /etc/yum.repos.d/kubernetes.repo >/dev/null << 'REPO_EOF'
[kubernetes]
name=Kubernetes
baseurl=https://pkgs.k8s.io/core:/stable:/v1.30/rpm/
enabled=1
gpgcheck=1
gpgkey=https://pkgs.k8s.io/core:/stable:/v1.30/rpm/repodata/repomd.xml.key
REPO_EOF
  sudo dnf install -y kubectl
fi

echo "=== 4. helm ==="
if ! which helm >/dev/null 2>&1; then
  curl -fsSL https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
fi

echo "=== 5. clone project (into workspace, persistent across Terminate) ==="
mkdir -p ~/workspace
cd ~/workspace
if [ ! -d mlops-project ]; then
  git clone https://github.com/soham1601/mlops-project.git
fi
cd mlops-project

echo "=== 6. python venv + deps + project ==="
# Fix: lockfile needs Python 3.12 (numpy 2.5.4). Rebuild venv if broken.
if ! .venv/bin/python -c "import mlflow, xgboost, mlops_project" 2>/dev/null; then
  echo "Rebuilding venv..."
  rm -rf .venv
  uv venv --python 3.12 .venv
  source .venv/bin/activate
  # Fix: uv sync alone doesn't install deps (lockfile-style project)
  uv pip install -r requirements.lock -r requirements-dev.txt
  uv pip install -e .
else
  echo "venv OK, skipping rebuild"
  source .venv/bin/activate
fi

echo "=== 7. verify ==="
.venv/bin/python -c "import mlflow, optuna, xgboost, sklearn, matplotlib, mlops_project; print('all deps + project OK')"

echo
echo "=== VERSIONS ==="
docker --version
kubectl version --client 2>/dev/null | head -1
minikube version | head -1
helm version --short
uv --version
.venv/bin/python --version

echo
echo "=== READY ==="
echo "Next:"
echo "  cd ~/workspace/mlops-project && source .venv/bin/activate"
echo "  make train              # train + log to MLflow (needs tracking: enabled: true in configs/cleveland.yaml)"
echo "  bash scripts/deploy_prayogshala.sh    # start minikube + deploy from GHCR"
