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

import pandas as pd
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from src.constants.training_pipeline import (ALERT_BUDGET_PCT, EMERGENCY_FLOOR_MMHG,
                                             POPULATION_THRESHOLD_MMHG, WARN_WINDOW)
from src.logging.logger import logging
from src.utils.ml_utils.model.estimator import BPPredictor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

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
# CSS and JS live alongside the template, per the project layout.
app.mount("/static", StaticFiles(directory=TEMPLATES_DIR), name="static")


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


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model_loaded": STORE.ready,
        "model_source": STORE.source,
        "detail": STORE.error,
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


if __name__ == "__main__":
    import uvicorn
    # 7860 is the port HuggingFace Spaces expects; PORT overrides it elsewhere.
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", "7860")))
