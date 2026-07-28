---
title: Cardioplace BP Alerts
emoji: 🩺
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 5.36.2
app_file: app.py
# The Space defaults to Python 3.10, but the pinned numpy/scipy in
# requirements.txt need >= 3.11. Without this the build cannot resolve them.
python_version: "3.12"
pinned: false
short_description: BP forecasting, personalisation and early warning on HEMOBP
# cpu-basic is what this app actually needs -- the stack is scikit-learn end to
# end with no GPU path. It also runs on ZeroGPU: app.py registers the Gradio
# handler as a @spaces.GPU entry point when the `spaces` package is present,
# which is what ZeroGPU's supervisor looks for at startup. See "Hardware" below.
suggested_hardware: cpu-basic
---

# Cardioplace BP Alerts

Blood-pressure forecasting, personalisation and early warning for haemodialysis
patients, built on the HEMOBP corpus.

## What it does

Three models plus a deterministic floor:

| Layer | Role |
|---|---|
| Rule engine | The permanent floor. Never trained, never tuned, never consults the ML layer. |
| Model 1 — forecaster | SBP / DBP / IDWG at 1–3 sessions ahead, with a conformal 80% interval. |
| Model 2 — personalisation offset | A capped shrinkage blend of the patient's own band and a demographic cohort prior. |
| Model 3 — early warning | Detects that SBP will exceed the patient's own p95 within the next 3 sessions. |

The ML layer produces provider-visible **advisories**. It never writes a
DeviationAlert, and the emergency floor (SBP ≥ 180 / DBP ≥ 120) is never
personalised — safety gate 1 proves this by differential execution on every run.

## Running it

```bash
pip install -r requirements.txt

python main.py     # train: ingest -> validate -> transform -> model trainer
python app.py      # serve: dashboard + API on http://localhost:7860
```

Training takes tens of minutes: `data/vip.csv` is ~250 MB / 4.4M readings, streamed
in 1M-row chunks, and the run includes a forward-chained random search.

### Routes

| Route | Purpose |
|---|---|
| `GET /` | Dashboard (`templates/`, served by FastAPI) |
| `GET /gradio` | Gradio interface over the same predictor |
| `GET /api/health` | Liveness, loaded model, tier, training state |
| `GET /api/schema` | Symptom / condition / medication vocabulary the form renders |
| `GET /api/model` | Model version, selected families, governance parameters |
| `POST /api/predict` | Session history → all three models, rule engine, backtest |
| `POST /api/reload` | Reload the newest bundle from disk |
| `POST /api/train` | Kick off a training run in the background |
| `GET /api/train/status` | Progress and tailed log of that run |

### How it is hosted

The Space runs the **Gradio SDK**, which ignores the `Dockerfile` and simply executes
`app.py`. Gradio is therefore mounted onto the FastAPI app rather than the other way
round, so one uvicorn process serves the dashboard, the REST API and the Gradio
interface together. The `Dockerfile` is kept for running the same app anywhere else.

### Hardware

Nothing here uses a GPU, so **cpu-basic** is the natural tier. The app also runs on
**ZeroGPU**, which matters because a Space created on ZeroGPU cannot be downgraded
without a PRO subscription.

ZeroGPU's supervisor terminates any Space it finds without a GPU entry point, and a
merely-decorated function does not count — the entry point has to be the callable a
Gradio event invokes, since its model is *event → allocate GPU → run → release*.
`app.py` therefore wraps the Gradio handler in `_maybe_gpu`, which applies
`@spaces.GPU` only when the `spaces` package is importable. HuggingFace injects that
package on ZeroGPU alone, so on cpu-basic the decorator is the identity function and
nothing changes. `GET /api/health` reports the detected tier.

On ZeroGPU each prediction from the Gradio tab takes a GPU allocation it does not
need, so it queues and draws on the tier's quota. Requests to `/api/predict` are not
Gradio events and are unaffected.

```bash
curl -X POST localhost:7860/api/predict -H "Content-Type: application/json" -d '{
  "patient_id": "demo", "age": 68, "is_male": 1, "is_dm": 0,
  "readings": [{"ts": "2026-01-05", "sbp": 152, "dbp": 78}]
}'
```

## Configuration

Only MLflow is configured by environment; everything else lives in
`src/constants/training_pipeline/__init__.py`. Copy the keys into `.env`:

```
MLFLOW_TRACKING_URI=https://dagshub.com/<owner>/<repo>.mlflow
MLFLOW_TRACKING_USERNAME=<username>
MLFLOW_TRACKING_PASSWORD=<access token>
```

Unset, MLflow falls back to a local `mlruns/` store and the pipeline runs normally.

## Governance

Values under `GOVERNANCE_KEYS` are clinician/ops inputs. They are never searched,
never tuned, and asserted throughout:

| Parameter | Value |
|---|---|
| Population threshold | 140 mmHg |
| Emergency floor | 180 mmHg (never personalised) |
| Offset caps | +15 loosen / −25 tighten (asymmetric on purpose) |
| Alert budget | 5% of patient-steps |
| Warning window | 3 sessions |

## Data

`data/` holds the HEMOBP corpus — patient-level dialysis records. `vip.csv` is
tracked with Git LFS; run `git lfs pull` after cloning.
