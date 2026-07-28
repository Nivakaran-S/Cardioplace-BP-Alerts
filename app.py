"""FastAPI serving layer for the Cardioplace BP alerts pipeline.

Surfaces all three model jobs plus the deterministic rule engine:

  Model 1  forecaster              SBP/DBP/IDWG at +1..+3 sessions, conformal band
  Model 2  personalisation offset  capped per-patient threshold
  Model 3  early-warning detector  anomaly score at every session
  Engine   ExtendedRuleEngine      the permanent floor, fired on every reading

and two combined views the individual models cannot give on their own:

  predicted alert  the engine run on the FORECAST reading -- "what fires in ~2 days
                   if the forecast holds and symptoms/meds stay as entered"
  backtest         what the forecaster said h sessions ago vs what actually happened

The corpus (HEMOBP) carries no symptoms, medications or conditions, so 45 of the 57
rule slots are BLOCKED_ON_INPUTS at training time. They are not blocked at inference:
the engine is deterministic, so anything the user enters in the dashboard unlocks the
corresponding rules immediately. What cannot be done is FORECASTING a symptom -- there
is no symptom history to learn from, and the app never pretends otherwise.
"""

import glob
import os
import pickle
import subprocess
import sys
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, List, Optional

import gradio as gr
import numpy as np
import pandas as pd
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from src.constants.training_pipeline import (ALERT_BUDGET_PCT, EMERGENCY_FLOOR_MMHG,
                                             POPULATION_THRESHOLD_MMHG, WARN_WINDOW)
from src.logging.logger import logging
from src.utils.ml_utils.model.estimator import BPPredictor
from src.utils.ml_utils.rule_engine.registry import EMERGENCY_TIERS, build_registry
from src.utils.ml_utils.rule_engine.synthetic import (CONDITION_KEYS, MED_KEYS, SYMPTOM_KEYS,
                                                      ExtendedRuleEngine)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")


# --------------------------------------------------------- ZeroGPU support
# Nothing here uses a GPU, but a Space that cannot be downgraded to cpu-basic must
# still give ZeroGPU an entry point reachable from a Gradio event, or its supervisor
# terminates the container. `spaces` is injected only on that tier.
try:
    import spaces as _spaces
except Exception:
    _spaces = None

ZEROGPU = _spaces is not None


def _maybe_gpu(fn):
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
    description="Forecasting, personalisation, early warning and the rule engine.",
    version="2.0.0",
    lifespan=lifespan,
)

templates = Jinja2Templates(directory=TEMPLATES_DIR)


@app.get("/static/{filename}")
def static_file(filename: str):
    """CSS/JS live beside the template. An explicit route, not a StaticFiles mount --
    a Mount cannot be transplanted onto Gradio's app the way a plain route can."""
    safe = os.path.basename(filename)
    path = os.path.join(TEMPLATES_DIR, safe)
    if not os.path.isfile(path) or safe.endswith(".html"):
        raise HTTPException(404, "not found")
    media = {"css": "text/css", "js": "application/javascript",
             "svg": "image/svg+xml"}.get(safe.rsplit(".", 1)[-1], "application/octet-stream")
    return FileResponse(path, media_type=media)


# ----------------------------------------------------------------- model store

class ModelStore:
    """Holds the loaded predictor. Reload is explicit so a request never blocks on I/O."""

    def __init__(self):
        self.predictor: Optional[BPPredictor] = None
        self.source: Optional[str] = None
        self.error: Optional[str] = None
        self.registry: Optional[pd.DataFrame] = None

    @staticmethod
    def _candidates() -> List[str]:
        found = []
        pushed = os.path.join(BASE_DIR, "final_model", "model.pkl")
        if os.path.exists(pushed):
            found.append(pushed)
        found.extend(sorted(glob.glob(os.path.join(
            BASE_DIR, "Artifacts", "*", "model_trainer", "trained_model",
            "predictor.joblib")), reverse=True))
        return found

    def load(self) -> bool:
        self.error = None
        for path in self._candidates():
            try:
                if path.endswith(".joblib"):
                    self.predictor = BPPredictor.load(path)
                else:
                    with open(path, "rb") as fh:
                        obj = pickle.load(fh)
                    self.predictor = obj if isinstance(obj, BPPredictor) else BPPredictor(obj)
                self.source = os.path.relpath(path, BASE_DIR)
                try:
                    self.registry = build_registry()
                except Exception as exc:
                    logging.warning("rule registry unavailable: %s", exc)
                logging.info("model loaded from %s (%s)", self.source,
                             self.predictor.b.get("model_version"))
                return True
            except Exception as exc:
                self.error = f"{type(exc).__name__}: {exc}"
                logging.warning("could not load %s: %s", path, self.error)
        if not self.error:
            self.error = ("no trained model found -- run the training pipeline "
                          "(python main.py) or use Train model in the dashboard")
        return False

    @property
    def ready(self) -> bool:
        return self.predictor is not None


STORE = ModelStore()


# --------------------------------------------------------------------- schemas

class Reading(BaseModel):
    ts: str = Field(..., description="Session date, ISO-8601 (YYYY-MM-DD)")
    sbp: float = Field(..., ge=40, le=300)
    dbp: float = Field(..., ge=20, le=200)
    pulse: Optional[float] = Field(None, ge=20, le=250)
    weight: Optional[float] = Field(None, ge=20, le=300, description="kg")
    idwg: Optional[float] = Field(None, description="Interdialytic weight gain, kg")
    position: str = Field("SITTING", description="SITTING | STANDING | LYING")
    n_meas: int = Field(2, ge=1, description="Readings taken in the session")
    symptoms: List[str] = Field(default_factory=list,
                                description=f"Any of: {', '.join(SYMPTOM_KEYS)}")


class Profile(BaseModel):
    age: float = Field(65, ge=18, le=110)
    is_male: int = Field(0, ge=0, le=1)
    is_dm: int = Field(0, ge=0, le=1)
    is_pregnant: int = Field(0, ge=0, le=1)
    hf_type: str = Field("NONE", description="NONE | HFREF | HFPEF")
    conditions: List[str] = Field(default_factory=list,
                                  description=f"Any of: {', '.join(CONDITION_KEYS)}")
    medications: List[str] = Field(default_factory=list,
                                   description=f"Any of: {', '.join(MED_KEYS)}")
    provider_target: Optional[float] = Field(None, description="Clinician SBP target")
    missed_3d: int = Field(0, ge=0, le=3, description="Missed dose days in the last 3")
    adherence_7d: float = Field(1.0, ge=0, le=1)


class PredictRequest(BaseModel):
    patient_id: str = "demo"
    profile: Profile = Field(default_factory=Profile)
    readings: List[Reading] = Field(..., min_length=1)


# --------------------------------------------------------------------- helpers

def _history_frame(req: PredictRequest) -> pd.DataFrame:
    """The frame the causal feature builder expects.

    Every column the builder indexes must exist even when unknown -- a missing one
    is an AttributeError rather than a NaN.
    """
    p = req.profile
    rows = []
    for r in req.readings:
        rows.append(dict(patient_id=str(req.patient_id),
                         ts=pd.to_datetime(r.ts, errors="coerce"),
                         sbp=float(r.sbp), dbp=float(r.dbp),
                         weight=r.weight, idwg=r.idwg,
                         sbp_drop=None, uf_total=None,
                         age=float(p.age), is_male=int(p.is_male),
                         is_dm=int(p.is_dm), DM=int(p.is_dm)))
    df = pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)
    if df.ts.isna().any():
        raise HTTPException(422, "one or more readings has an unparseable ts")
    for col in ("weight", "idwg", "sbp_drop", "uf_total"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _engine_frame(req: PredictRequest) -> pd.DataFrame:
    """Panel the ExtendedRuleEngine evaluates: vitals plus everything the user entered.

    This is where the 45 BLOCKED_ON_INPUTS rules come back. They are blocked because
    HEMOBP has no symptom/medication/condition columns to TRAIN on -- but the engine
    is deterministic, so a symptom typed into the dashboard fires its rule immediately.
    """
    p = req.profile
    rows = []
    for i, r in enumerate(sorted(req.readings, key=lambda x: x.ts)):
        row = dict(series_id=str(req.patient_id), step=i,
                   ts=pd.to_datetime(r.ts, errors="coerce"),
                   sbp=float(r.sbp), dbp=float(r.dbp),
                   pulse=float(r.pulse) if r.pulse is not None else np.nan,
                   weight=float(r.weight) if r.weight is not None else np.nan,
                   idwg=float(r.idwg) if r.idwg is not None else np.nan,
                   n_meas=int(r.n_meas), position=r.position,
                   age=float(p.age), is_male=int(p.is_male), is_dm=int(p.is_dm),
                   is_pregnant=int(p.is_pregnant), hf_type=p.hf_type,
                   provider_target=(float(p.provider_target)
                                    if p.provider_target is not None else np.nan),
                   missed_3d=int(p.missed_3d), adherence_7d=float(p.adherence_7d),
                   missed_bb_today=int(p.missed_3d > 0 and "on_bb" in p.medications))
        for k in CONDITION_KEYS:
            row[k] = int(k in p.conditions)
        for k in MED_KEYS:
            row[k] = int(k in p.medications)
        for k in SYMPTOM_KEYS:
            row[k] = int(k in r.symptoms)
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("step").reset_index(drop=True)
    df["days_since_last"] = df.ts.diff().dt.days.fillna(2).clip(1, 30)
    df["weight_delta_24h"] = df.weight.diff().fillna(0.0)
    # prev_emergency reaches the predicates as a column: itertuples rows are immutable
    df["prev_emergency"] = (df.sbp.shift(1) >= 180).fillna(False)
    # brady run length and recent tachycardia, both causal
    run, runs, prev_high, highs = 0, [], -99, []
    for i, r in df.iterrows():
        hr = r.pulse
        run = run + 1 if (np.isfinite(hr) and 40 <= hr <= 49 and not r.dizziness) else 0
        runs.append(run)
        highs.append(int(i - prev_high <= 1))
        if np.isfinite(hr) and hr > 100:
            prev_high = i
    df["brady_run_len"] = runs
    df["hr_high_recent"] = highs
    return df


def _run_engine(frame: pd.DataFrame, personal_thr: Optional[float]) -> pd.DataFrame:
    thresholds = ({str(frame.series_id.iloc[0]): personal_thr}
                  if personal_thr is not None else {})
    engine = ExtendedRuleEngine(STORE.registry, thresholds)
    return engine.run(frame, use_personalisation=personal_thr is not None)


def _alert_payload(row) -> dict:
    return {
        "tier": row.get("tier"),
        "rule_id": row.get("rule_id"),
        "mode": row.get("mode"),
        "gate_reason": row.get("gate_reason"),
        "fired": bool(row.get("fired")),
        "is_emergency": bool(row.get("tier") in EMERGENCY_TIERS),
    }


def _require_model() -> BPPredictor:
    if not STORE.ready:
        raise HTTPException(503, STORE.error or "model not loaded")
    return STORE.predictor


def _feature_frame(predictor: BPPredictor, history: pd.DataFrame) -> pd.DataFrame:
    """Causal features at every session, built once (batch), not per-session."""
    g = history.sort_values("ts").copy()
    g["series_id"] = str(g.patient_id.iloc[0])
    g["days_since_last"] = g.ts.diff().dt.days.fillna(2).clip(1, 30)
    g["step"] = np.arange(len(g))
    g["is_weekend"] = (g.ts.dt.dayofweek >= 5).astype(int)
    return predictor.fb.transform(g)


def _anomaly_series(predictor: BPPredictor, F: pd.DataFrame) -> dict:
    """Model 3 across the whole history, so drift toward the cut is visible."""
    det = predictor.b["detector"]
    X = det["imputer"].transform(F.reindex(columns=det["cols"]))
    scores = -det["model"].score_samples(X)
    cut = float(det["cut"])
    warm = predictor.config.cold_start_min_readings
    points = [{"ts": t.strftime("%Y-%m-%d"), "score": round(float(s), 4),
               "flagged": bool(s >= cut), "warmup": i < warm}
              for i, (t, s) in enumerate(zip(F.ts, scores))]
    settled = [p for p in points if not p["warmup"]]
    return {"points": points, "cut": round(cut, 4), "budget_pct": det["budget_pct"],
            "n_flagged": sum(1 for p in settled if p["flagged"]),
            "n_settled": len(settled), "warmup_sessions": len(points) - len(settled),
            "event_definition": (f"SBP exceeds this patient's own p"
                                 f"{int(det['event_quantile'] * 100)} within the next "
                                 f"{det['warn_window']} sessions")}


def _backtest(predictor: BPPredictor, F: pd.DataFrame, signal: str = "sbp") -> dict:
    """What the forecaster said h sessions ago, against what actually happened.

    The forecaster maps features at step t to the reading at t+h, so replaying it
    over the history gives exactly the "BP we predicted two days prior" comparison,
    with no extra model and no leakage -- the features at t never see t+h.
    """
    out = {"horizons": [], "series": {}}
    names = predictor.b["feature_names"]
    med_gap = float(F.ts.diff().dt.days.median() or 2)
    for (sig, h), model in sorted(predictor.b["forecasters"].items()):
        if sig != signal:
            continue
        try:
            pred = model.predict(F[names])
        except Exception as exc:
            logging.warning("backtest unavailable for %s h%s: %s", sig, h, exc)
            continue
        rows, errs = [], []
        for t in range(len(F) - h):
            actual = float(F[signal].iloc[t + h])
            forecast = float(pred[t])
            if not (np.isfinite(actual) and np.isfinite(forecast)):
                continue
            rows.append({"ts": F.ts.iloc[t + h].strftime("%Y-%m-%d"),
                         "actual": round(actual, 1), "predicted": round(forecast, 1),
                         "error": round(forecast - actual, 1)})
            errs.append(abs(forecast - actual))
        if not rows:
            continue
        out["series"][f"h{h}"] = rows
        out["horizons"].append({
            "horizon": h, "days_ahead": round(h * med_gap, 1), "n": len(rows),
            "mae": round(float(np.mean(errs)), 2),
            "within_10": round(float(np.mean([e <= 10 for e in errs])), 3),
        })
    out["signal"] = signal
    return out


# ---------------------------------------------------------------- training job

class TrainingState:
    """Tracks the background pipeline run so the dashboard can show progress."""

    def __init__(self):
        self.lock = threading.Lock()
        self.running = False
        self.started_at: Optional[str] = None
        self.finished_at: Optional[str] = None
        self.returncode: Optional[int] = None
        self.tail: List[str] = []

    def snapshot(self) -> dict:
        with self.lock:
            return {"running": self.running, "started_at": self.started_at,
                    "finished_at": self.finished_at, "returncode": self.returncode,
                    "tail": list(self.tail[-25:])}


TRAINING = TrainingState()


def _run_training():
    with TRAINING.lock:
        if TRAINING.running:
            return
        TRAINING.running = True
        TRAINING.started_at = datetime.now().isoformat(timespec="seconds")
        TRAINING.finished_at = None
        TRAINING.returncode = None
        TRAINING.tail = ["starting main.py ..."]

    try:
        proc = subprocess.Popen(
            [sys.executable, os.path.join(BASE_DIR, "main.py")], cwd=BASE_DIR,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            line = line.rstrip()
            if not line:
                continue
            with TRAINING.lock:
                TRAINING.tail.append(line)
                TRAINING.tail = TRAINING.tail[-200:]
        proc.wait()
        rc = proc.returncode
        if rc == 0:
            STORE.load()
    except Exception as exc:
        rc = -1
        with TRAINING.lock:
            TRAINING.tail.append(f"FAILED: {type(exc).__name__}: {exc}")
        logging.exception("training pipeline failed")
    finally:
        with TRAINING.lock:
            TRAINING.running = False
            TRAINING.returncode = rc
            TRAINING.finished_at = datetime.now().isoformat(timespec="seconds")
        logging.info("training pipeline finished rc=%s", rc)


# ---------------------------------------------------------------------- routes

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/health")
def health():
    return {"status": "ok", "model_loaded": STORE.ready, "model_source": STORE.source,
            "detail": STORE.error, "tier": "zerogpu" if ZEROGPU else "cpu",
            "training": TRAINING.snapshot()}


@app.get("/api/schema")
def schema():
    """Field vocabulary the dashboard renders its inputs from."""
    reg = STORE.registry
    return {
        "symptoms": SYMPTOM_KEYS,
        "conditions": CONDITION_KEYS,
        "medications": MED_KEYS,
        "rules": {
            "total_slots": int(len(reg)) if reg is not None else None,
            "evaluable_on_corpus": (int((reg.status == "EVALUABLE").sum())
                                    if reg is not None else None),
            "blocked_on_corpus": (int((reg.status == "BLOCKED_ON_INPUTS").sum())
                                  if reg is not None else None),
            "note": ("BLOCKED_ON_INPUTS describes the TRAINING corpus, which has no "
                     "symptom/medication/condition columns. The engine is deterministic, "
                     "so entering those inputs here evaluates the rules immediately."),
        },
    }


@app.get("/api/model")
def model_info():
    p = _require_model()
    b = p.b
    return {
        "model_version": b.get("model_version"), "run_id": b.get("run_id"),
        "source": STORE.source, "selected_family": b.get("selected_family"),
        "n_features": len(b.get("feature_names", [])),
        "forecast_horizons": sorted({h for _, h in b.get("forecasters", {}).keys()}),
        "forecast_signals": sorted({s for s, _ in b.get("forecasters", {}).keys()}),
        "interval": {"signal": b["interval"].get("signal"),
                     "horizon": b["interval"].get("horizon"),
                     "qhat": round(float(b["interval"].get("qhat", 0.0)), 2)},
        "offset": {k: b["offset"].get(k) for k in ("warm", "k", "q", "label")},
        "governance": {"population_threshold_mmHg": POPULATION_THRESHOLD_MMHG,
                       "emergency_floor_mmHg": EMERGENCY_FLOOR_MMHG,
                       "alert_budget_pct": ALERT_BUDGET_PCT,
                       "warn_window_sessions": WARN_WINDOW},
    }


@app.post("/api/predict")
def predict(req: PredictRequest):
    """All three models, the rule engine, and the two combined views."""
    predictor = _require_model()
    history = _history_frame(req)

    try:
        advisory = predictor.predict(history)          # Models 1 + 2 + 3 at 'now'
    except Exception as exc:
        logging.exception("prediction failed")
        raise HTTPException(500, f"prediction failed: {type(exc).__name__}: {exc}")

    advisory["history"] = [
        {"ts": r.ts.strftime("%Y-%m-%d"), "sbp": float(r.sbp), "dbp": float(r.dbp)}
        for r in history.itertuples()]

    # Model 3 across the whole history, plus the backtest -- one feature build for both.
    try:
        F = _feature_frame(predictor, history)
        advisory["anomaly"] = _anomaly_series(predictor, F)
        advisory["backtest"] = _backtest(predictor, F)
    except Exception as exc:
        logging.warning("trend views unavailable: %s", exc)
        advisory["anomaly"] = None
        advisory["backtest"] = None

    # --- rule engine, on everything the user actually entered -------------------
    thr = (advisory.get("personalisation") or {}).get("threshold")
    try:
        frame = _engine_frame(req)
        alerts = _run_engine(frame, thr)
        current = alerts.iloc[-1].to_dict()
        fired = alerts[alerts.fired]
        advisory["rule_engine"] = {
            "current": _alert_payload(current),
            "personalised_threshold_used": thr,
            "history": [{"ts": pd.Timestamp(r.ts).strftime("%Y-%m-%d"), "tier": r.tier,
                         "rule_id": r.rule_id, "fired": bool(r.fired)}
                        for r in alerts.itertuples()],
            "fired_count": int(len(fired)),
            "tier_counts": (fired.tier.value_counts().to_dict() if len(fired) else {}),
        }
    except Exception as exc:
        logging.exception("rule engine failed")
        advisory["rule_engine"] = {"error": f"{type(exc).__name__}: {exc}"}

    # --- combined: run the engine on the FORECAST reading -----------------------
    # This is the "what fires in ~2 days" view. Vitals come from Model 1; symptoms,
    # medications and conditions are carried forward unchanged from the latest entry,
    # because nothing in the corpus supports forecasting them.
    try:
        fc = (advisory.get("forecast") or {}).get("sbp") or {}
        fc_d = (advisory.get("forecast") or {}).get("dbp") or {}
        predicted = []
        base = _engine_frame(req)
        for key in sorted(fc, key=lambda k: fc[k]["steps_ahead"]):
            node = fc[key]
            future = base.tail(1).copy()
            future["sbp"] = float(node["point"])
            if key in fc_d:
                future["dbp"] = float(fc_d[key]["point"])
            future["step"] = int(base.step.max()) + node["steps_ahead"]
            future["days_since_last"] = float(base.days_since_last.iloc[-1])
            future["prev_emergency"] = bool(base.sbp.iloc[-1] >= 180)
            res = _run_engine(future, thr).iloc[0].to_dict()
            predicted.append({
                "steps_ahead": node["steps_ahead"],
                "days_ahead": node["days_ahead_est"],
                "sbp": node["point"],
                "dbp": fc_d.get(key, {}).get("point"),
                "lo80": node.get("lo80"), "hi80": node.get("hi80"),
                **_alert_payload(res),
            })
        advisory["predicted_alert"] = {
            "horizons": predicted,
            "basis": ("rule engine evaluated on the forecast vitals; symptoms, "
                      "medications and conditions carried forward unchanged"),
            "symptom_forecast_supported": False,
            "symptom_note": ("The corpus has no symptom history, so no symptom is "
                             "forecast. Symptoms you enter are applied to the current "
                             "reading and carried forward into the predicted alert."),
        }
    except Exception as exc:
        logging.warning("predicted alert unavailable: %s", exc)
        advisory["predicted_alert"] = None

    return JSONResponse(advisory)


@app.post("/api/reload")
def reload_model():
    if not STORE.load():
        raise HTTPException(503, STORE.error)
    return {"reloaded": True, "source": STORE.source,
            "model_version": STORE.predictor.b.get("model_version")}


@app.post("/api/train")
def train(background_tasks: BackgroundTasks):
    """Run the full pipeline: ingest -> validate -> transform -> model trainer."""
    if TRAINING.snapshot()["running"]:
        raise HTTPException(409, "a training run is already in progress")
    background_tasks.add_task(_run_training)
    return {"started": True,
            "note": ("a full run streams 250 MB and does a forward-chained random "
                     "search; expect tens of minutes. Poll /api/train/status.")}


@app.get("/api/train/status")
def train_status():
    return TRAINING.snapshot()


# ------------------------------------------------------------ gradio surface

def _parse_readings(raw: str):
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
def _gradio_predict(readings_text, patient_id, age, sex, dm, symptoms):
    if not STORE.ready:
        return f"### No model loaded\n\n{STORE.error}", {}
    rows, errors = _parse_readings(readings_text or "")
    if errors:
        return "### Could not parse the readings\n\n- " + "\n- ".join(errors[:5]), {}
    if not rows:
        return "### Enter at least one reading", {}
    if symptoms:
        rows[-1].symptoms = list(symptoms)

    req = PredictRequest(patient_id=patient_id or "demo",
                         profile=Profile(age=float(age),
                                         is_male=1 if sex == "Male" else 0,
                                         is_dm=1 if dm == "Yes" else 0),
                         readings=rows)
    advisory = predict(req).body
    import json
    a = json.loads(advisory)

    pers = a.get("personalisation", {})
    ew = a.get("early_warning") or {}
    eng = (a.get("rule_engine") or {}).get("current") or {}
    lines = [f"### {a['patient_id']} — {a['confidence_tier'].replace('_', ' ')}", "",
             f"**Threshold** {pers.get('threshold')} mmHg · "
             f"**Floor** {a.get('emergency_floor_mmHg')} mmHg (never personalised)  ",
             f"**Engine now** {eng.get('tier') or 'no alert'}"
             + (f" — {eng.get('rule_id')}" if eng.get("rule_id") else "")]
    for p in (a.get("predicted_alert") or {}).get("horizons", []):
        lines.append(f"**+{p['steps_ahead']} sessions (~{p['days_ahead']} d)** "
                     f"SBP {p['sbp']} → {p.get('tier') or 'no alert'}"
                     + (f" ({p.get('rule_id')})" if p.get("rule_id") else ""))
    if ew:
        lines.append(f"**Early warning** {'FLAGGED' if ew['flagged'] else 'not flagged'} "
                     f"— score {ew['score']} vs cut {ew['cut']}")
    lines.append("")
    lines.append("The full dashboard, with all three models charted, is at "
                 "[/dashboard](/dashboard).")
    return "\n".join(lines), a


with gr.Blocks(title="Cardioplace BP Alerts", analytics_enabled=False) as demo:
    gr.Markdown("## Cardioplace BP Alerts\nQuick check. The full dashboard — three "
                "models, rule engine, backtest — is at [/dashboard](/dashboard).")
    with gr.Row():
        with gr.Column(scale=2):
            g_readings = gr.Textbox(label="Session readings", lines=10,
                                    info="One per line: YYYY-MM-DD, SBP, DBP",
                                    value="\n".join(f"2026-01-{d:02d}, {150 + (d % 7)}, "
                                                    f"{78 + (d % 5)}" for d in range(1, 26)))
            g_symp = gr.CheckboxGroup(SYMPTOM_KEYS[:10], label="Symptoms today")
        with gr.Column(scale=1):
            g_pid = gr.Textbox(label="Patient ID", value="demo-001")
            g_age = gr.Number(label="Age", value=68, minimum=18, maximum=110)
            g_sex = gr.Radio(["Female", "Male"], label="Sex", value="Male")
            g_dm = gr.Radio(["No", "Yes"], label="Diabetes", value="No")
            g_btn = gr.Button("Get advisory", variant="primary")
    g_summary = gr.Markdown()
    with gr.Accordion("Raw advisory", open=False):
        g_json = gr.JSON()
    g_btn.click(_gradio_predict, [g_readings, g_pid, g_age, g_sex, g_dm, g_symp],
                [g_summary, g_json])

# Off-Space, FastAPI owns the server and Gradio is mounted under it.
app = gr.mount_gradio_app(app, demo, path="/gradio")


def _server_port() -> int:
    """A Gradio Space is always proxied on 7860; it also sets PORT to an internal
    value that something else already holds, so PORT is read only off-Space."""
    if os.getenv("SPACE_ID"):
        return int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    return int(os.getenv("PORT") or os.getenv("GRADIO_SERVER_PORT") or 7860)


def _serve_on_space(port: int) -> None:
    """Gradio owns the server; the FastAPI routes are transplanted onto its app.

    A Space registers with the platform through Gradio's launch path -- a custom
    uvicorn with Gradio merely mounted is reaped shortly after startup. Routes are
    PREPENDED because Gradio's app ends in a catch-all and Starlette stops at the
    first match; appended routes silently 404. "/" is included, so the dashboard is
    the landing page rather than Gradio's own UI.
    """
    STORE.load()   # the FastAPI lifespan does not run when Gradio owns the server
    demo.launch(server_name="0.0.0.0", server_port=port,
                prevent_thread_lock=True, share=False)

    ours = [r for r in app.routes
            if getattr(r, "path", "") == "/"
            or getattr(r, "path", "").startswith(("/api", "/static", "/dashboard"))]
    for route in reversed(ours):
        demo.app.routes.insert(0, route)
    logging.info("grafted %d FastAPI routes ahead of Gradio's catch-all "
                 "(dashboard at /, Gradio at /gradio)", len(ours))
    demo.block_thread()


if __name__ == "__main__":
    port = _server_port()
    if os.getenv("SPACE_ID"):
        _serve_on_space(port)
    else:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=port)
