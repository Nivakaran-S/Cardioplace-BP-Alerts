"""FastAPI serving layer for the Cardioplace BP alerts pipeline.

Loads the frozen BPPredictor bundle produced by the training pipeline and exposes
it over HTTP, plus the dashboard in templates/. The bundle is the only state; no
training-time globals are read, so this process runs anywhere the artifact goes.
"""

import glob
import os
import pickle
import subprocess
import sys
from contextlib import asynccontextmanager
from typing import List, Optional

import gradio as gr
import pandas as pd
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from src.constants.training_pipeline import (ALERT_BUDGET_PCT, EMERGENCY_FLOOR_MMHG,
                                             POPULATION_THRESHOLD_MMHG, WARN_WINDOW)
from src.logging.logger import logging
from src.utils.ml_utils.model.estimator import BPPredictor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")


# --------------------------------------------------------- ZeroGPU support
# Nothing in this pipeline uses a GPU -- it is scikit-learn end to end -- but a
# Space that cannot be downgraded to cpu-basic must still satisfy ZeroGPU, whose
# supervisor terminates any Space it finds without a GPU entry point.
#
# A bare decorated function is not enough: the entry point has to be the callable
# a Gradio event actually invokes, because ZeroGPU's model is event -> allocate
# GPU -> run -> release. `_maybe_gpu` therefore decorates the Gradio handler
# itself. The work inside stays on the CPU; the decoration exists to give ZeroGPU
# the entry point it requires.
#
# `spaces` is injected by HuggingFace only on a ZeroGPU Space, so on cpu-basic
# the import fails and the decorator is the identity function.
try:
    import spaces as _spaces
except Exception:
    _spaces = None

ZEROGPU = _spaces is not None


def _maybe_gpu(fn):
    """Register `fn` as ZeroGPU's entry point when running on that tier."""
    if _spaces is None:
        return fn
    logging.info("ZeroGPU tier: registering %s as the GPU entry point "
                 "(the work itself is CPU-only)", fn.__name__)
    return _spaces.GPU(duration=60)(fn)


@asynccontextmanager
async def lifespan(_: FastAPI):
    STORE.load()          # a missing model is a degraded state, never a failed boot
    yield


app = FastAPI(
    title="Cardioplace BP Alerts",
    description="Blood-pressure forecasting, personalisation and early warning on HEMOBP.",
    version="1.0.0",
    lifespan=lifespan,
)

templates = Jinja2Templates(directory=TEMPLATES_DIR)


# CSS and JS live alongside the template, per the project layout. This is an
# explicit route rather than app.mount(StaticFiles(...)) because a Mount cannot be
# transplanted onto Gradio's app the way a plain route can -- see _serve_on_space.
@app.get("/static/{filename}")
def static_file(filename: str):
    safe = os.path.basename(filename)          # no traversal out of templates/
    path = os.path.join(TEMPLATES_DIR, safe)
    if not os.path.isfile(path) or safe.endswith(".html"):
        raise HTTPException(404, "not found")
    media = {"css": "text/css", "js": "application/javascript",
             "svg": "image/svg+xml", "ico": "image/x-icon"}.get(safe.rsplit(".", 1)[-1],
                                                                "application/octet-stream")
    return FileResponse(path, media_type=media)


# ----------------------------------------------------------------- model loading

class ModelStore:
    """Holds the loaded predictor. Reload is explicit so a request never blocks on I/O."""

    def __init__(self):
        self.predictor: Optional[BPPredictor] = None
        self.source: Optional[str] = None
        self.error: Optional[str] = None

    @staticmethod
    def _candidates() -> List[str]:
        """Newest bundle first: the shipped final_model, then any local training run."""
        found = []
        pushed = os.path.join(BASE_DIR, "final_model", "model.pkl")
        if os.path.exists(pushed):
            found.append(pushed)
        runs = glob.glob(os.path.join(BASE_DIR, "Artifacts", "*", "model_trainer",
                                      "trained_model", "predictor.joblib"))
        found.extend(sorted(runs, reverse=True))
        return found

    def load(self) -> bool:
        self.error = None
        for path in self._candidates():
            try:
                if path.endswith(".joblib"):
                    self.predictor = BPPredictor.load(path)
                else:
                    # final_model/model.pkl is a pickled BPPredictor instance, not a bundle.
                    with open(path, "rb") as fh:
                        obj = pickle.load(fh)
                    self.predictor = obj if isinstance(obj, BPPredictor) else BPPredictor(obj)
                self.source = os.path.relpath(path, BASE_DIR)
                logging.info("model loaded from %s (%s)", self.source,
                             self.predictor.b.get("model_version"))
                return True
            except Exception as exc:
                self.error = f"{type(exc).__name__}: {exc}"
                logging.warning("could not load %s: %s", path, self.error)
        if not self.error:
            self.error = ("no trained model found -- run the training pipeline "
                          "(python main.py) or POST /api/train")
        logging.warning("no model available: %s", self.error)
        return False

    @property
    def ready(self) -> bool:
        return self.predictor is not None


STORE = ModelStore()


# --------------------------------------------------------------------- schemas

class Reading(BaseModel):
    ts: str = Field(..., description="Session date, ISO-8601 (YYYY-MM-DD)")
    sbp: float = Field(..., ge=40, le=300, description="Pre-dialysis systolic BP (mmHg)")
    dbp: float = Field(..., ge=20, le=200, description="Pre-dialysis diastolic BP (mmHg)")
    weight: Optional[float] = Field(None, description="Pre-dialysis weight (kg)")
    idwg: Optional[float] = Field(None, description="Interdialytic weight gain (kg)")


class PredictRequest(BaseModel):
    patient_id: str = Field("demo", description="Identifier, used only for labelling")
    age: float = Field(65, ge=18, le=110)
    is_male: int = Field(0, ge=0, le=1)
    is_dm: int = Field(0, ge=0, le=1, description="Diabetes mellitus")
    readings: List[Reading] = Field(..., min_length=1)


# --------------------------------------------------------------------- helpers

def _history_frame(req: PredictRequest) -> pd.DataFrame:
    """Build the frame the causal feature builder expects from an API payload.

    Every column the builder reads must exist even when unknown -- it indexes them
    directly, so a missing one is an AttributeError rather than a NaN.
    """
    rows = []
    for r in req.readings:
        rows.append(dict(patient_id=str(req.patient_id), ts=pd.to_datetime(r.ts, errors="coerce"),
                         sbp=float(r.sbp), dbp=float(r.dbp),
                         weight=r.weight, idwg=r.idwg,
                         sbp_drop=None, uf_total=None,
                         age=float(req.age), is_male=int(req.is_male),
                         is_dm=int(req.is_dm), DM=int(req.is_dm)))
    df = pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)
    if df.ts.isna().any():
        raise HTTPException(422, "one or more readings has an unparseable ts")
    for col in ("weight", "idwg", "sbp_drop", "uf_total"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _require_model() -> BPPredictor:
    if not STORE.ready:
        raise HTTPException(503, STORE.error or "model not loaded")
    return STORE.predictor


# ---------------------------------------------------------------------- routes

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request):
    """Same dashboard, reachable when Gradio owns / (the Spaces layout)."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model_loaded": STORE.ready,
        "model_source": STORE.source,
        "detail": STORE.error,
        "tier": "zerogpu" if ZEROGPU else "cpu",
    }


@app.get("/api/model")
def model_info():
    p = _require_model()
    b = p.b
    return {
        "model_version": b.get("model_version"),
        "run_id": b.get("run_id"),
        "source": STORE.source,
        "selected_family": b.get("selected_family"),
        "n_features": len(b.get("feature_names", [])),
        "forecast_horizons": sorted({h for _, h in b.get("forecasters", {}).keys()}),
        "forecast_signals": sorted({s for s, _ in b.get("forecasters", {}).keys()}),
        "interval": {
            "signal": b["interval"].get("signal"),
            "horizon": b["interval"].get("horizon"),
            "qhat": round(float(b["interval"].get("qhat", 0.0)), 2),
        },
        "offset": {k: b["offset"].get(k) for k in ("warm", "k", "q", "label")},
        "governance": {
            "population_threshold_mmHg": POPULATION_THRESHOLD_MMHG,
            "emergency_floor_mmHg": EMERGENCY_FLOOR_MMHG,
            "alert_budget_pct": ALERT_BUDGET_PCT,
            "warn_window_sessions": WARN_WINDOW,
        },
    }


@app.post("/api/predict")
def predict(req: PredictRequest):
    """One patient's session history -> the full advisory."""
    predictor = _require_model()
    history = _history_frame(req)
    try:
        advisory = predictor.predict(history)
    except Exception as exc:
        logging.exception("prediction failed")
        raise HTTPException(500, f"prediction failed: {type(exc).__name__}: {exc}")

    advisory["history"] = [
        {"ts": r.ts.strftime("%Y-%m-%d"), "sbp": float(r.sbp), "dbp": float(r.dbp)}
        for r in history.itertuples()
    ]
    return JSONResponse(advisory)


@app.post("/api/reload")
def reload_model():
    ok = STORE.load()
    if not ok:
        raise HTTPException(503, STORE.error)
    return {"reloaded": True, "source": STORE.source,
            "model_version": STORE.predictor.b.get("model_version")}


def _run_training():
    logging.info("training pipeline started from the API")
    try:
        subprocess.run([sys.executable, os.path.join(BASE_DIR, "main.py")],
                       cwd=BASE_DIR, check=True)
        STORE.load()
        logging.info("training pipeline finished; model reloaded")
    except Exception as exc:
        logging.exception("training pipeline failed: %s", exc)


@app.post("/api/train")
def train(background_tasks: BackgroundTasks):
    """Kick off a full training run. Long-running: poll /api/health for the new model."""
    background_tasks.add_task(_run_training)
    return {"started": True,
            "note": "a full run takes tens of minutes; poll /api/health for model_loaded"}


# ------------------------------------------------------------ gradio surface

# The Space runs on the Gradio SDK, which ignores the Dockerfile and simply executes
# `python app.py`. Mounting Gradio onto the FastAPI app keeps one uvicorn process
# serving everything: the dashboard at /, the REST API at /api/*, and Gradio at
# /gradio. A native Gradio-only app would have meant discarding templates/.

def _parse_readings(raw: str):
    """Parse the `YYYY-MM-DD, SBP, DBP` block the Gradio textbox accepts."""
    rows, errors = [], []
    for i, line in enumerate(raw.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.replace("\t", ",").replace(";", ",").split(",")]
        if len(parts) < 3:
            errors.append(f"line {i}: expected date, SBP, DBP")
            continue
        try:
            rows.append(Reading(ts=parts[0], sbp=float(parts[1]), dbp=float(parts[2])))
        except Exception as exc:
            errors.append(f"line {i}: {exc.__class__.__name__}")
    return rows, errors


@_maybe_gpu
def _gradio_predict(readings_text, patient_id, age, sex, dm):
    if not STORE.ready:
        return f"### No model loaded\n\n{STORE.error}", {}
    rows, errors = _parse_readings(readings_text or "")
    if errors:
        return "### Could not parse the readings\n\n- " + "\n- ".join(errors[:5]), {}
    if not rows:
        return "### Enter at least one reading", {}

    req = PredictRequest(patient_id=patient_id or "demo", age=float(age),
                         is_male=1 if sex == "Male" else 0,
                         is_dm=1 if dm == "Yes" else 0, readings=rows)
    advisory = STORE.predictor.predict(_history_frame(req))

    pers = advisory.get("personalisation", {})
    ew = advisory.get("early_warning") or {}
    lines = [
        f"### {advisory['patient_id']} — {advisory['confidence_tier'].replace('_', ' ')}",
        "",
        f"**Personalised threshold** {pers.get('threshold')} mmHg "
        f"(offset {pers.get('offset'):+} mmHg"
        + (", bound by the governance cap" if pers.get("capped") else "") + ")  ",
        f"**Emergency floor** {advisory.get('emergency_floor_mmHg')} mmHg — never personalised  ",
        f"**Observations** {advisory.get('n_observations')}",
    ]
    fc = (advisory.get("forecast") or {}).get("sbp") or {}
    if fc:
        lines += ["", "| Horizon | SBP | 80% interval |", "|---|---|---|"]
        for k in sorted(fc, key=lambda k: fc[k]["steps_ahead"]):
            f = fc[k]
            band = (f"{f['lo80']} – {f['hi80']}" if f.get("lo80") is not None else "—")
            lines.append(f"| +{f['steps_ahead']} sessions | {f['point']} mmHg | {band} |")
    else:
        lines += ["", f"_{advisory.get('note', 'no forecast issued')}_"]
    if ew:
        lines += ["", f"**Early warning** {'FLAGGED' if ew['flagged'] else 'not flagged'} — "
                      f"score {ew['score']} vs cut {ew['cut']}, est. lead {ew['est_lead_days']} d"]
    return "\n".join(lines), advisory


with gr.Blocks(title="Cardioplace BP Alerts", analytics_enabled=False) as demo:
    gr.Markdown(
        "## Cardioplace BP Alerts\n"
        "Blood-pressure forecasting, personalisation and early warning on HEMOBP. "
        "The full dashboard is at [/](/) and the REST API at `/api/predict`."
    )
    with gr.Row():
        with gr.Column(scale=2):
            g_readings = gr.Textbox(
                label="Session readings", lines=12,
                info="One per line: YYYY-MM-DD, SBP, DBP",
                value="\n".join(
                    f"2026-01-{d:02d}, {150 + (d % 7)}, {78 + (d % 5)}" for d in range(1, 26)))
        with gr.Column(scale=1):
            g_pid = gr.Textbox(label="Patient ID", value="demo-001")
            g_age = gr.Number(label="Age", value=68, minimum=18, maximum=110)
            g_sex = gr.Radio(["Female", "Male"], label="Sex", value="Male")
            g_dm = gr.Radio(["No", "Yes"], label="Diabetes", value="No")
            g_btn = gr.Button("Get advisory", variant="primary")
    g_summary = gr.Markdown()
    with gr.Accordion("Raw advisory", open=False):
        g_json = gr.JSON()

    g_btn.click(_gradio_predict, [g_readings, g_pid, g_age, g_sex, g_dm], [g_summary, g_json])

# Off-Space (local run, Docker) FastAPI owns the server and Gradio is mounted under
# it, which keeps the dashboard at /. On a Space that inversion is not available:
# see _serve_on_space below.
app = gr.mount_gradio_app(app, demo, path="/gradio")


def _server_port() -> int:
    """The port to bind.

    A Gradio Space is always proxied on 7860 -- that is why app_port is a
    Docker-only front-matter key. The Space also sets PORT to an internal value
    (7861) that something else already holds, so honouring PORT there makes the
    container exit with 'address already in use'. PORT is therefore only read
    when not running on a Space.
    """
    if os.getenv("SPACE_ID"):
        return int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    return int(os.getenv("PORT") or os.getenv("GRADIO_SERVER_PORT") or 7860)


def _serve_on_space(port: int) -> None:
    """Let Gradio own the server, then graft the FastAPI routes onto its app.

    A Space registers itself with the platform through Gradio's own launch path.
    Running a custom uvicorn with Gradio merely mounted bypasses that, and on
    ZeroGPU the supervisor then terminates the container shortly after startup --
    the app serves correctly right up to the SIGTERM, which is what made this so
    hard to read from the logs.

    Gradio therefore owns "/" here, and the dashboard moves to /dashboard.
    demo.app only exists once launch() has built it, so the routes are attached
    afterwards; Starlette resolves routes per request, so late registration works.
    """
    STORE.load()   # the FastAPI lifespan does not run when Gradio owns the server
    demo.launch(server_name="0.0.0.0", server_port=port,
                prevent_thread_lock=True, share=False)

    # Prepend, never append. Gradio's app ends in a catch-all that matches any
    # unclaimed path, and Starlette stops at the first match -- appended routes
    # are therefore unreachable and every one of them 404s.
    ours = [r for r in app.routes
            if getattr(r, "path", "").startswith(("/api", "/static", "/dashboard"))]
    for route in reversed(ours):
        demo.app.routes.insert(0, route)
    logging.info("grafted %d FastAPI routes ahead of Gradio's catch-all "
                 "(dashboard at /dashboard, API at /api/*)", len(ours))
    demo.block_thread()


if __name__ == "__main__":
    port = _server_port()
    if os.getenv("SPACE_ID"):
        _serve_on_space(port)
    else:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=port)
