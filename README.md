---
title: Cardioplace BP Alerts
emoji: 🩺
colorFrom: blue
colorTo: red
sdk: docker
app_port: 7860
pinned: false
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

### API

| Route | Purpose |
|---|---|
| `GET /` | Dashboard |
| `GET /api/health` | Liveness and whether a model is loaded |
| `GET /api/model` | Model version, selected families, governance parameters |
| `POST /api/predict` | A patient's session history → the full advisory |
| `POST /api/reload` | Reload the newest bundle from disk |
| `POST /api/train` | Kick off a training run in the background |

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
