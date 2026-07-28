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

### What actually ships

Every one of those layers is chosen by an evaluation, and the evaluation is **binding**:

- **Model 1** ships a learned model only if `(baseline MAE − learned MAE) > bootstrap CI
  width`, compared over the same horizons. On the current cohort no learned family clears
  that bar for any signal, so all three signals serve the **EWMA baseline**. `GET
  /api/model` reports what shipped per signal; the dashboard prints it on the chart.
- **Model 3** ships the highest-precision detector the bundle can evaluate, selected on
  **validation** at the alert budget. That is a forecast-relative detector — "is this
  patient heading above their *own* band" — not a population outlier model.
- **Model 2** is retained on a `BASELINE` verdict rather than swapped. Every candidate is
  capped to the legal band first — the raw personal band reaches 195 mmHg and would breach
  the emergency floor for 24 of 150 patients, so it was never a shippable rival. Once
  capped, cohort-only is a constant 155 for everyone, and personal-only has no shrinkage,
  so a patient with five readings gets a threshold set by five readings. The remaining
  separation is well inside the bootstrap CI. `BASELINE` here means "not proven better",
  and it is recorded as such in `offset_scorecard.csv` and the artifact.

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
`app.py`. Locally that means FastAPI owns the server and Gradio is mounted at
`/gradio`; on a Space it is the other way round, because the SDK may *import* the
module and launch `demo` itself rather than run it as `__main__`. The dashboard is
therefore built into the Gradio Blocks and rendered server-side — no `/static` fetch,
no `/api` call — so it stands up under either entry point. The `Dockerfile` is kept
for running the same app anywhere else.

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

The entry point is a **diagnostic button**, not the advisory handler. Decorating the
handler makes every "Get advisory" click queue for a GPU allocation and spend quota on
scikit-learn work that runs in ~46 ms and touches no GPU — which exhausts the ZeroGPU
runs limit in ordinary use. Normal use now costs no quota. If you hit the limit anyway,
adding `HF_TOKEN` as a Space secret raises it.

```bash
curl -X POST localhost:7860/api/predict -H "Content-Type: application/json" -d '{
  "patient_id": "demo",
  "profile": {"age": 68, "is_male": 1, "is_dm": 0},
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
MLFLOW_LOG_CANDIDATES=1        # optional; 0 logs only the models that ship
```

Note the `.mlflow` suffix — the DagsHub *clone* URL ends in `.git` and gives a 404 here.
Unset, MLflow falls back to a local `mlruns/` store and the pipeline runs normally.
On a Space these are **Space** secrets, not GitHub repository secrets: training runs
inside the Space, so the Actions runner's copies are not visible to it.

### What lands in MLflow

A parent run per pipeline run, and a nested child run per model fitted — 136 on this
cohort. Every candidate the pipeline chose between is recorded with its val *and* test
metrics, so the selection is auditable from the tracking store alone.

| Tag | Values |
|---|---|
| `category` | `forecaster` · `offset` · `offset-search` · `detector` · `classifier` |
| `selection` | `final` (it serves) · `candidate` (it lost) |
| `is_final` | `true` / `false` — same thing, for filtering |
| plus | `signal`, `horizon`, `family`, `model` / `detector`, `run_id` |

Only finals carry a serialised estimator and a registry name (`hemobp-forecaster-sbp-h1`,
`hemobp-detector`, `hemobp-offset`); logging ~130 artefacts nothing will load would
multiply storage for nothing. `MLFLOW_LOG_CANDIDATES=0` drops the candidate runs when the
tracking server is remote and 136 round trips per pipeline run is too many.

`final` follows the **ship decision**, not the sweep — so today the baselines are tagged
final, because they are what actually serves.

## Missed readings

Users skip days. Nothing is imputed: a skipped reading is an absent row, and
`days_since_last` carries the spacing. A daily grid would fabricate ~57% of its rows and
teach the forecaster to recover its own interpolator.

| | |
|---|---|
| One definition | `src/utils/ml_utils/feature/cadence.py` — six copies previously, one already drifted |
| Model input | `days_since_last`, clipped to [1, 30] |
| The truth | `days_since_last_raw`, unclipped, **never a feature**, asserted by `cadence_audit` |
| Contract | asserts on the raw gap: strictly positive, clip binding on <1% of rows, no forward-filled runs |
| Robustness | `missingness_robustness.csv` — deleting 35% of sessions costs 0.35 mmHg (gate 10) |
| Drift | `DriftMonitor.cadence_drift`, PSI on the raw gap, its own signal |

A reading older than `STALE_FORECAST_MAX_DAYS` (14, the same number the rule engine uses)
gets the personalised threshold but **no forecast and no early warning** — the lags and
windows describe a patient who has since been unobserved. Pass `as_of` to `/api/predict`
to control what "now" means; it defaults to request time.

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
| Stale forecast limit | 14 days (asserted equal to the engine's `STALE_GAP_DAYS`) |

Promotion is gated on them: a run with a critical gate failure keeps its own artifact and
leaves `final_model/model.pkl` untouched, rather than logging "PROMOTION BLOCKED" after
having already overwritten it.

## Data

`data/` holds the HEMOBP corpus — patient-level dialysis records. `vip.csv` is
tracked with Git LFS; run `git lfs pull` after cloning.
