# Demo Video Script — Heart Disease MLOps (10 min)

## Pre-recording checklist (do all of this BEFORE hitting record)

**On the Prayogshala VM** (should already be done from today):
```bash
cd ~/workspace/mlops-project && git pull origin master
TRAIN=yes bash scripts/run_all.sh    # populates mlflow.db, deploys, starts port-forwards
```
Verify everything is up: `kubectl get pods -A | grep -E "heart-api|monitoring|grafana"` — all Running.

**Pre-open these Firefox tabs in this order** (bookmark bar makes switching clean):
1. `https://github.com/soham1601/mlops-project`
2. `https://github.com/soham1601/mlops-project/actions`
3. `https://github.com/users/soham1601/packages/container/mlops-project`
4. `http://localhost:8080/docs` (Swagger)
5. `http://localhost:5000` (MLflow)
6. `http://localhost:9090/targets?search=heart-api` (Prometheus)
7. `http://localhost:3000` (Grafana — logged in, dashboard open, time range "Last 15 minutes", auto-refresh 10s)

**Terminal ready** (one Terminal window, `cd ~/workspace/mlops-project`, prompt visible).

**Also open** on second monitor (not shared): this script, a timer, and the GitHub repo file tree.

**Grafana password handy**:
```bash
kubectl -n monitoring get secret monitoring-grafana -o jsonpath='{.data.admin-password}' | base64 -d
```

**Recording tool**: OBS Studio or Prayogshala's built-in screen share with a mic. 1280×720 is plenty.

---

## Script — 10 minutes total

### 0:00–0:30 · Intro (30s)

**Show**: Terminal with repo tree
```bash
tree -L 1 -I '.venv|__pycache__|logs|mlruns'
```

**Say**:
> "Hi, I'm Soham. This is an end-to-end MLOps pipeline I built for the UCI Heart Disease dataset.
> The goal was to train a classifier and ship it as a production-grade API — with CI, container registry,
> Kubernetes, and live monitoring. I'll walk through each layer in 10 minutes."

---

### 0:30–1:30 · Architecture (1 min)

**Show**: Open the architecture diagram in your report (Section 2, Figure 2.1), or draw it on screen briefly.

**Say**:
> "Four layers. One: train a scikit-learn pipeline — the preprocessing and the classifier are both inside
> the same Pipeline object, so training and serving can't drift apart. Three models — logistic regression,
> random forest, and XGBoost — are tuned with Optuna, cross-validated with 5-fold CV, and the best one wins.
> MLflow logs every run.
>
> Two: GitHub Actions runs lint, test, train, docker — all on every push to master. The docker job pushes
> the image to GitHub Container Registry.
>
> Three: Helm deploys the API to Kubernetes. Two replicas behind a LoadBalancer.
>
> Four: Prometheus scrapes a `/metrics` endpoint on every pod; Grafana visualizes it. JSON request logs
> go to stdout where kubectl captures them."

---

### 1:30–2:30 · Code and repo (1 min)

**Show**: GitHub tab 1 — repo home. Scroll through the file tree.

**Say** (while scrolling):
> "The code is in `src/mlops_project`. Training, CLI, FastAPI app, preprocessing. Config is in a YAML
> under `configs/`.
>
> Scripts in `scripts/` give one-command setup and deploy on a fresh VM. README has the quick start.
>
> One Dockerfile builds a multi-stage image — the model is trained inside the build, so the final image
> ships the pipeline and the trained artifact together."

**Zoom to** `README.md`. Scroll past the "Run in 2 commands" section at the top.

**Say**:
> "Someone cloning this repo can be on a running cluster in two commands — one bootstraps the VM, the other
> deploys from GHCR and runs the smoke tests."

---

### 2:30–4:00 · CI/CD pipeline (1.5 min)

**Show**: GitHub tab 2 — Actions.

**Say**:
> "Every push to master triggers CI. Here's the latest green run."

**Click** into a passing run. Scroll to show the four jobs:

**Say**:
> "Lint runs ruff. Test runs 150 pytest cases. Train runs the whole training pipeline with 10 trials —
> fast version for CI — and uploads the model and training outputs as artifacts.
>
> Docker builds the container, runs it, and does a smoke test — hits `/ready` 30 times until it's healthy,
> then POSTs to `/predict`, checks the response is valid JSON with a probability in the right range.
> If any step fails, the whole pipeline fails."

**Click** the "docker" job. Scroll to the smoke-test step. Expand it briefly to show the JSON response:

**Say**:
> "This is the smoke test proving the container actually works before we push the image."

**Switch to** a known failed run (`demo/break-lint` branch run from earlier):

**Say**:
> "Here's what a failure looks like. I pushed an intentional lint error on a branch — ruff caught the
> unused import in 7 seconds, and test, train, docker were all skipped. Fast feedback."

---

### 4:00–5:00 · GHCR + container image (1 min)

**Show**: GitHub tab 3 — Packages → mlops-project.

**Say**:
> "On every push to master, the docker job pushes the image to GitHub Container Registry — tagged by git
> SHA and `latest`. Here's the package. Public, so Kubernetes can pull it without a secret."

**Switch to Terminal**:
```bash
kubectl describe pod -l app.kubernetes.io/name=heart-api | grep "Image:" | head -2
```

**Say** (point at the `ghcr.io/soham1601/mlops-project:latest` line):
> "Here are the two pods — both running the image from GHCR. The deploy script in this repo overrides
> `image.repository` to GHCR, so the cluster pulls the registry image, not a locally-built one. Full
> CI-to-cluster loop."

---

### 5:00–6:30 · Live API demo (1.5 min)

**Show**: Firefox tab 4 — Swagger at `localhost:8080/docs`.

**Say**:
> "The API is FastAPI. Four endpoints — `health`, `ready`, `info`, `predict`, and `metrics`.
> `predict` takes 13 features from the patient record and returns the model's classification."

**Switch to Terminal**:
```bash
curl http://localhost:8080/ready
# {"status":"ready"}

curl -X POST http://localhost:8080/predict -H 'Content-Type: application/json' -d '{
  "age":57,"sex":1,"cp":4,"trestbps":140,"chol":241,"fbs":0,"restecg":0,
  "thalach":123,"exang":1,"oldpeak":0.2,"slope":2,"ca":0,"thal":7
}'
```

**Say** (pointing at the Disease response):
> "This patient is classified as Disease, with 85% probability. Model is xgboost — recorded in the response
> so we can trace which model served every prediction."

**Run the No-disease case**:
```bash
curl -X POST http://localhost:8080/predict -H 'Content-Type: application/json' -d '{
  "age":63,"sex":1,"cp":1,"trestbps":145,"chol":233,"fbs":1,"restecg":2,
  "thalach":150,"exang":0,"oldpeak":2.3,"slope":3,"ca":0,"thal":6
}'
```

**Say**:
> "Same API, different inputs — No disease, 21%. Live model serving two classes."

**Fire the 500 mixed-request burst** (keeps Grafana interesting for the next section):
```bash
for i in $(seq 1 500); do
  if [ $((i % 2)) -eq 0 ]; then
    P='{"age":57,"sex":1,"cp":4,"trestbps":140,"chol":241,"fbs":0,"restecg":0,"thalach":123,"exang":1,"oldpeak":0.2,"slope":2,"ca":0,"thal":7}'
  else
    P='{"age":63,"sex":1,"cp":1,"trestbps":145,"chol":233,"fbs":1,"restecg":2,"thalach":150,"exang":0,"oldpeak":2.3,"slope":3,"ca":0,"thal":6}'
  fi
  curl -s -X POST http://localhost:8080/predict -H 'Content-Type: application/json' -d "$P" > /dev/null
done &
```

**Say**:
> "Firing 500 mixed requests in the background. We'll see them on the dashboards in a moment."

---

### 6:30–8:00 · MLflow tracking (1.5 min)

**Show**: Firefox tab 5 — MLflow at `localhost:5000`.

**Say**:
> "This is MLflow. Every training run is logged — I'm looking at the latest.
> There's a parent run called `train-cleveland` and three child runs — one per model.
> Click into the parent."

**Click** `train-cleveland` → Overview tab.

**Say** (point at the metrics):
> "Ten metrics logged — test accuracy, test F1, test ROC-AUC, plus cross-validated means and standard
> deviations. XGBoost won with a test AUC of 0.929."

**Click** the Compare button after selecting the three child runs:

**Say**:
> "Compare view across the three models. You can see the parameters Optuna tried, the metrics they got,
> and which one came out on top. If I need to repro any run, every seed, every param, every artifact is here."

**Click** Models tab (left nav) → heart-disease-classifier → Version 1:

**Say**:
> "The best model is registered in the Model Registry with the alias `champion`. That alias is what the
> API loads on startup — so promoting a new model is just moving the alias."

**Click** back to the parent run → Artifacts tab:

**Say**:
> "Every run also stores artifacts — model card, saved pipeline, metrics JSON, reports. Full lineage."

---

### 8:00–9:30 · Prometheus + Grafana (1.5 min)

**Show**: Firefox tab 6 — Prometheus `localhost:9090/targets`.

**Say**:
> "Both API pods are discovered automatically. The Helm chart ships a ServiceMonitor and
> kube-prometheus-stack picks it up. Scrape every 15 seconds. Both green and up."

**Click** Query. Type `predictions_total`. Execute. Switch to Graph tab.

**Say**:
> "The 500 requests I fired a minute ago are visible as a step increase in `predictions_total`.
> One line per pod — load is balanced across them."

**Show**: Firefox tab 7 — Grafana dashboard.

**Say**:
> "Grafana has a custom dashboard shipped by the Helm chart.
>
> Top row: five stat tiles. Pods serving traffic — two.
> Model loaded — one means both pods have the model.
> Error ratio — zero percent, which is what we want.
> Predictions-per-hour and mean predicted probability — these move with the burst I just fired.
>
> Middle row: requests per second colored by HTTP status, and p50/p95 latency.
> Our p95 is around 60 milliseconds — plenty for a 13-feature classifier.
>
> Bottom: predictions by class — both classes represented because I mixed inputs.
> And requests-per-pod — the load balancer is doing its job, roughly 50/50 across the two pods."

**Hover** on a few data points to highlight the live tooltips.

**Say**:
> "Everything live. If a pod crashes, latency spikes, or the model stops loading, this dashboard would
> show it instantly."

---

### 9:30–10:00 · Wrap-up (30s)

**Switch to Terminal**:
```bash
ls scripts/
```

**Say**:
> "To summarize: 2 commands to go from a bare VM to a running cluster.
> Setup bootstraps the VM. Deploy brings up Minikube, monitoring, and the API from GHCR.
> Smoke test and run_all orchestrate the whole thing. Logs go to logs/ for debugging.
>
> Everything is in GitHub — repo, CI, container image are all linked from the README.
> Thank you."

---

## Backup slides (if a demo step fails)

If the API doesn't respond: the `kubectl get pods` output and the Grafana screenshot still prove the deployment. Keep the Appendix A screenshot register from your report handy as a fallback.

If MLflow isn't reachable: screenshots 6.1–6.5 from your report are the fallback.

If Prometheus/Grafana are down: Figures 11.2 and 11.3 from your report are the fallback.

## Timing cheat sheet

| Section | Minutes |
|---|---|
| Intro | 0.5 |
| Architecture | 1.0 |
| Code & repo | 1.0 |
| CI/CD pipeline | 1.5 |
| GHCR + image | 1.0 |
| Live API demo | 1.5 |
| MLflow | 1.5 |
| Prometheus + Grafana | 1.5 |
| Wrap | 0.5 |
| **Total** | **10.0** |

## Delivery tips

- Speak at ~140 words/min — slow enough to be understood, not boring
- No "um" or "uh" — pause silently if you need to think
- Point at things you're referring to (mouse hover is fine)
- Don't run anything that takes >10s on-camera (fire bursts in the background)
- Keep the mouse cursor still while narrating a static view
- Record in one take, trim dead air in post if needed
