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
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import List, Optional

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
    """Model 3 across the whole history, so drift toward the cut is visible.

    Routed through the predictor's own scorer, so the chart and the advisory can never
    show two different detectors. This is a retrospective view: every session is scored
    against the patient's overall band, which is the same quantity the cut was
    calibrated against, so a point above the line means what the cut says it means.
    """
    det = predictor.b["detector"]
    scores = np.asarray(predictor.detector_score(F, F.sbp), float)
    cut = float(det["cut"])
    warm = predictor.config.cold_start_min_readings
    # Early sessions have no rolling window yet, so a column-based detector scores NaN
    # there. NaN is not valid JSON; those points carry a null score and never flag.
    points = [{"ts": t.strftime("%Y-%m-%d"),
               "score": (round(float(s), 4) if np.isfinite(s) else None),
               "flagged": bool(np.isfinite(s) and s >= cut), "warmup": i < warm}
              for i, (t, s) in enumerate(zip(F.ts, scores))]
    settled = [p for p in points if not p["warmup"] and p["score"] is not None]
    return {"points": points, "detector": det.get("name", "d_isoforest"),
            "cut": round(cut, 4), "budget_pct": det["budget_pct"],
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
        # What the ship decision actually sent to production, which is not always the
        # learned winner -- `selected_family` is only the best of the learned candidates.
        "shipped": {s: f"{fam}:{name}"
                    for s, (fam, name) in (b.get("shipped") or {}).items()},
        "detector": b.get("detector", {}).get("name"),
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
    # What the ship decision sent to production, per signal. A reader should not have to
    # assume a learned model produced the numbers on the page.
    advisory["shipped"] = {s: f"{fam}:{name}" for s, (fam, name)
                           in (predictor.b.get("shipped") or {}).items()}

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


def _sample_readings() -> str:
    seed, sbp, dbp, w = 20260728, 138.0, 76.0, 74.0
    out, d, gaps = [], datetime(2026, 1, 5), [2, 2, 3]
    for i in range(34):
        seed = (seed * 1103515245 + 12345) % 2147483648
        j = seed / 2147483648
        sbp = 0.72 * sbp + 0.28 * (138 + i * 0.75) + (j - 0.5) * 13
        dbp = 0.70 * dbp + 0.30 * (75 + i * 0.2) + (j - 0.5) * 7
        w += (j - 0.45) * 0.4
        out.append(f"{d:%Y-%m-%d}, {round(sbp)}, {round(dbp)}, "
                   f"{round(70 + (j - 0.5) * 16)}, {w:.1f}")
        d += timedelta(days=gaps[i % 3])
    return "\n".join(out)


# ---- server-side rendering -------------------------------------------------
# The dashboard is rendered here rather than by templates/app.js, because a Gradio
# Space may IMPORT this module instead of running it -- in which case nothing under
# `if __name__ == "__main__"` executes, no FastAPI route is ever grafted, and the
# browser would only ever see Gradio's own widgets. Everything below therefore
# travels inside the Gradio payload: inline CSS, inline SVG, no /static fetch and
# no /api call. templates/ still serves the identical view when FastAPI owns the
# server locally.

PALETTE = {  # validated categorical slots 1-2 + the reserved status ramp
    "light": {"s1": "#2a78d6", "s2": "#eb6834", "surface": "#fcfcfb", "plane": "#f9f9f7",
              "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781", "grid": "#e1e0d9",
              "axis": "#c3c2b7", "border": "rgba(11,11,11,.10)"},
}
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a",
          "critical": "#d03b3b"}

DASH_CSS = """
<style>
.cp{--s1:#2a78d6;--s2:#eb6834;--sf:#fcfcfb;--pl:#f9f9f7;--ink:#0b0b0b;--ink2:#52514e;
 --mut:#898781;--bd:rgba(11,11,11,.10);--gd:#0ca30c;--wn:#fab219;--cr:#d03b3b;
 font-family:system-ui,-apple-system,"Segoe UI",sans-serif;color:var(--ink);font-size:14px}
@media (prefers-color-scheme:dark){.cp{--sf:#1a1a19;--pl:#0d0d0d;--ink:#fff;--ink2:#c3c2b7;
 --bd:rgba(255,255,255,.12);--s1:#3987e5;--s2:#d95926}}
.cp *{box-sizing:border-box}
.cp .bn{display:flex;gap:12px;align-items:flex-start;background:var(--sf);
 border:1px solid var(--bd);border-left:3px solid var(--mut);border-radius:10px;
 padding:14px 16px;margin-bottom:16px}
.cp .bn.good{border-left-color:var(--gd)}.cp .bn.watch{border-left-color:var(--wn)}
.cp .bn.crit{border-left-color:var(--cr)}
.cp .bn h4{margin:0;font-size:.95rem}.cp .bn p{margin:3px 0 0;font-size:.83rem;color:var(--ink2)}
.cp .pnl{background:var(--sf);border:1px solid var(--bd);border-radius:10px;
 padding:16px;margin-bottom:16px}
.cp .pnl.acc{border-left:3px solid var(--s1)}
.cp h3{margin:0 0 4px;font-size:.95rem}
.cp .hint{margin:0;font-size:.76rem;color:var(--mut)}
.cp .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px}
.cp .card{border:1px solid var(--bd);border-radius:8px;padding:12px 14px;background:var(--pl)}
.cp .card .lbl{margin:0 0 5px;font-size:.68rem;font-weight:600;letter-spacing:.05em;
 text-transform:uppercase;color:var(--mut)}
.cp .card .val{margin:0;font-size:1.5rem;font-weight:600;letter-spacing:-.02em}
.cp .card .val .u{font-size:.72rem;font-weight:500;color:var(--mut);margin-left:4px}
.cp .card .sub{margin:5px 0 0;font-size:.74rem;color:var(--ink2)}
.cp .chip{display:inline-block;font-size:.72rem;font-weight:600;padding:3px 9px;
 border-radius:999px;border:1px solid var(--bd);color:var(--ink2)}
.cp .chip.good{color:var(--gd);border-color:var(--gd)}
.cp .chip.warn{color:var(--ink);border-color:var(--wn)}
.cp .chip.crit{color:var(--cr);border-color:var(--cr)}
.cp table{border-collapse:collapse;width:100%;font-size:.8rem;margin-top:10px}
.cp th,.cp td{text-align:left;padding:6px 9px;border-bottom:1px solid var(--bd);
 white-space:nowrap}
.cp th{color:var(--mut);font-size:.72rem;font-weight:600}
.cp td{font-variant-numeric:tabular-nums}
.cp .tl{display:flex;flex-wrap:wrap;gap:3px;margin:10px 0}
.cp .tl i{width:12px;height:20px;border-radius:3px;background:rgba(137,135,129,.28)}
.cp .lg{display:flex;gap:14px;flex-wrap:wrap;font-size:.75rem;color:var(--ink2);margin:6px 0}
.cp .lg b{display:inline-block;width:12px;height:3px;border-radius:2px;margin-right:5px;
 vertical-align:middle}
.cp .co{margin:10px 0 0;padding:9px 12px;border-radius:8px;background:var(--pl);
 border:1px solid var(--bd);font-size:.78rem;color:var(--ink2)}
.cp .sc{overflow-x:auto}.cp svg{display:block;max-width:100%;height:auto}
</style>
"""

_TIER_COLOUR = {
    "BP_LEVEL_2": STATUS["critical"], "BP_LEVEL_2_SYMPTOM_OVERRIDE": STATUS["critical"],
    "TIER_1_ANGIOEDEMA": STATUS["critical"], "TIER_1_CONTRAINDICATION": STATUS["serious"],
    "TIER_2_DISCREPANCY": STATUS["warning"], "BP_LEVEL_1_HIGH": "#eb6834",
    "BP_LEVEL_1_LOW": "#2a78d6", "TIER_3_INFO": "rgba(137,135,129,.55)",
}


def _esc(s) -> str:
    return (str("" if s is None else s).replace("&", "&amp;")
            .replace("<", "&lt;").replace(">", "&gt;"))


def _pretty(s) -> str:
    if not s:
        return ""
    t = str(s).replace("RULE_", "").replace("_", " ").lower()
    return t[:1].upper() + t[1:]


def _svg_series(series, refs=(), width=880, height=280, y_fmt="{:.0f}",
                x_labels=None, shade=None):
    """Minimal line chart. `series` = [(points, colour, dashed, marks)] where marks
    is a list of (index, colour) drawn as filled dots."""
    C = PALETTE["light"]
    ml, mr, mt, mb = 48, 76, 16, 30
    iw, ih = width - ml - mr, height - mt - mb
    vals = [v for pts, *_ in series for v in pts if v is not None]
    vals += [r[0] for r in refs if r[0] is not None]
    if len(vals) < 2:
        return "<p class='hint'>not enough data to plot</p>"
    lo, hi = min(vals), max(vals)
    pad = max((hi - lo) * 0.12, 0.01)
    y0, y1 = lo - pad, hi + pad
    n = max(len(p) for p, *_ in series)

    def X(i):
        return ml + (0 if n <= 1 else (i / (n - 1)) * iw)

    def Y(v):
        return mt + ih - ((v - y0) / (y1 - y0)) * ih

    out = [f'<svg viewBox="0 0 {width} {height}" preserveAspectRatio="xMidYMid meet" '
           f'role="img">']
    if shade:
        out.append(f'<rect x="{ml}" y="{mt}" width="{max(X(shade) - ml, 2):.1f}" '
                   f'height="{ih}" fill="{C["muted"]}" opacity=".09"/>'
                   f'<text x="{ml + 5}" y="{mt + 12}" fill="{C["muted"]}" '
                   f'font-size="10">warm-up</text>')
    for k in range(5):
        v = y0 + k / 4 * (y1 - y0)
        out.append(f'<line x1="{ml}" x2="{ml + iw}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" '
                   f'stroke="{C["grid"]}" stroke-width="1"/>'
                   f'<text x="{ml - 8}" y="{Y(v) + 4:.1f}" text-anchor="end" '
                   f'fill="{C["muted"]}" font-size="11">{y_fmt.format(v)}</text>')
    out.append(f'<line x1="{ml}" x2="{ml + iw}" y1="{mt + ih}" y2="{mt + ih}" '
               f'stroke="{C["axis"]}" stroke-width="1"/>')
    for v, colour, text in refs:
        if v is None or not (y0 <= v <= y1):
            continue
        out.append(f'<line x1="{ml}" x2="{ml + iw}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" '
                   f'stroke="{colour}" stroke-width="2" stroke-dasharray="5 4"/>'
                   f'<text x="{ml + iw + 6}" y="{Y(v) + 4:.1f}" fill="{C["ink2"]}" '
                   f'font-size="11">{_esc(text)}</text>')
    for i, lab in (x_labels or []):
        out.append(f'<text x="{X(i):.1f}" y="{mt + ih + 19}" text-anchor="middle" '
                   f'fill="{C["muted"]}" font-size="11">{_esc(lab)}</text>')
    for pts, colour, dashed, marks in series:
        d = " ".join(f'{"L" if k else "M"}{X(k):.1f} {Y(v):.1f}'
                     for k, v in enumerate(pts) if v is not None)
        if d:
            out.append(f'<path d="{d}" fill="none" stroke="{colour}" stroke-width="2" '
                       f'stroke-linejoin="round" stroke-linecap="round"'
                       + (' stroke-dasharray="6 4"' if dashed else "") + "/>")
        for idx, mcol in (marks or []):
            if 0 <= idx < len(pts) and pts[idx] is not None:
                out.append(f'<circle cx="{X(idx):.1f}" cy="{Y(pts[idx]):.1f}" r="5" '
                           f'fill="{mcol}" stroke="{C["surface"]}" stroke-width="2"/>')
    out.append("</svg>")
    return "".join(out)


def _legend(items):
    return ('<div class="lg">' + "".join(
        f'<span><b style="background:{c}"></b>{_esc(t)}</span>' for c, t in items)
        + "</div>")


def _render_dashboard(a: dict) -> str:
    C, S = PALETTE["light"], STATUS
    pers = a.get("personalisation") or {}
    ew = a.get("early_warning") or {}
    eng = (a.get("rule_engine") or {})
    cur = eng.get("current") or {}
    pa = (a.get("predicted_alert") or {})
    hz = pa.get("horizons") or []
    h = ['<div class="cp">', DASH_CSS]

    # ---- banner ----
    first_fire = next((x for x in hz if x.get("fired")), None)
    if cur.get("is_emergency"):
        cls, ic = "crit", "&#9888;"
        t = "Rule engine fired an emergency alert on the latest reading"
        d = (_pretty(cur.get("rule_id")) + " — the emergency floor is never personalised "
             "and the engine is authoritative here.")
    elif ew.get("flagged"):
        cls, ic = "crit", "&#9888;"
        t = "Early-warning detector flagged this patient"
        d = (f"Score {ew['score']} at or above the {ew['budget_pct']}% cut of {ew['cut']}, "
             f"about {ew['est_lead_days']} days of lead time.")
    elif first_fire:
        cls, ic = "watch", "&#9670;"
        t = (f"{_pretty(first_fire['tier'])} forecast in about "
             f"{first_fire['days_ahead']} days")
        d = (f"Predicted SBP {first_fire['sbp']} mmHg would fire "
             f"{_pretty(first_fire.get('rule_id'))} if the trend holds.")
    elif cur.get("fired"):
        cls, ic = "watch", "&#9670;"
        t = f"{_pretty(cur.get('tier'))} on the latest reading"
        d = _pretty(cur.get("rule_id")) + ". No further breach forecast."
    elif a.get("confidence_tier") == "cold_start":
        cls, ic, t, d = "", "&#9675;", "Cold start — no forecast issued", a.get("note", "")
    else:
        cls, ic, t = "good", "&#10003;", "Nothing due"
        d = ("The engine fired nothing on the latest reading, the forecast stays below "
             f"the personalised threshold of {pers.get('threshold')} mmHg, and the "
             "detector is below its cut.")
    h.append(f'<div class="bn {cls}"><span>{ic}</span><div><h4>{_esc(t)}</h4>'
             f'<p>{_esc(d)}</p></div></div>')

    # ---- combined: engine on the forecast ----
    h.append('<div class="pnl acc"><h3>Combined outlook</h3>'
             '<p class="hint">Forecast vitals passed through the rule engine — '
             'what fires if the trend holds</p><div class="grid" style="margin-top:12px">')
    if hz:
        for x in hz:
            cc = "crit" if x.get("is_emergency") else ("warn" if x.get("fired") else "good")
            h.append(
                f'<div class="card"><p class="lbl">in ~{_esc(x["days_ahead"])} days '
                f'· +{_esc(x["steps_ahead"])} sessions</p>'
                f'<p class="val">{_esc(x["sbp"])}'
                + (f'/{_esc(x["dbp"])}' if x.get("dbp") is not None else "")
                + '<span class="u">mmHg</span></p>'
                + (f'<p class="sub">80% {_esc(x["lo80"])} – {_esc(x["hi80"])}</p>'
                   if x.get("lo80") is not None else '<p class="sub">no interval</p>')
                + f'<span class="chip {cc}">'
                  f'{_esc(_pretty(x["tier"]) if x.get("tier") else "No alert")}</span>'
                + (f'<p class="sub">{_esc(_pretty(x.get("rule_id")))}</p>'
                   if x.get("rule_id") else "") + "</div>")
    else:
        h.append('<p class="hint">No forecast issued — below the cold-start floor.</p>')
    h.append("</div>")
    if pa.get("symptom_note"):
        h.append(f'<p class="co">{_esc(pa.get("basis", ""))}. '
                 f'{_esc(pa["symptom_note"])}</p>')
    h.append("</div>")

    # ---- tiles ----
    ec = ("crit" if cur.get("is_emergency") else "warn" if cur.get("fired") else "good")
    ew_chip = ("not issued" if not ew else
               ("&#9888; Flagged" if ew.get("flagged") else "&#10003; Not flagged"))
    ew_cls = "" if not ew else ("crit" if ew.get("flagged") else "good")
    h.append(
        '<div class="grid" style="margin-bottom:16px">'
        f'<div class="card"><p class="lbl">Engine now · rule engine</p>'
        f'<p class="val"><span class="chip {ec}">'
        f'{_esc(_pretty(cur.get("tier")) if cur.get("tier") else "No alert")}</span></p>'
        f'<p class="sub">{_esc(_pretty(cur.get("rule_id")) or "nothing fired")}</p></div>'
        f'<div class="card"><p class="lbl">Threshold · M2</p>'
        f'<p class="val">{_esc(pers.get("threshold", "–"))}<span class="u">mmHg</span></p>'
        f'<p class="sub">offset {_esc(pers.get("offset", "–"))} mmHg'
        + (" — capped" if pers.get("capped") else "") + "</p></div>"
        f'<div class="card"><p class="lbl">Early warning · M3</p>'
        f'<p class="val"><span class="chip {ew_cls}">{ew_chip}</span></p>'
        f'<p class="sub">'
        + (f'score {ew["score"]} vs cut {ew["cut"]}' if ew else "below cold-start floor")
        + "</p></div>"
        f'<div class="card"><p class="lbl">Confidence</p>'
        f'<p class="val" style="font-size:1.1rem">'
        f'{_esc(str(a.get("confidence_tier", "")).replace("_", " "))}</p>'
        f'<p class="sub">{_esc(a.get("n_observations", 0))} readings</p></div></div>')

    # ---- Model 1: forecast chart ----
    hist = a.get("history") or []
    fc = (a.get("forecast") or {}).get("sbp") or {}
    fpts = sorted(fc.values(), key=lambda v: v["steps_ahead"])
    if len(hist) >= 2:
        obs = [x["sbp"] for x in hist] + [None] * len(fpts)
        fut = [None] * (len(hist) - 1) + [hist[-1]["sbp"]] + [x["point"] for x in fpts]
        n = len(obs)
        xl = [(0, hist[0]["ts"][5:]), (len(hist) // 2, hist[len(hist) // 2]["ts"][5:]),
              (len(hist) - 1, hist[-1]["ts"][5:])]
        if fpts:
            xl.append((n - 1, f"+{fpts[-1]['steps_ahead']} sess"))
        refs = [(pers.get("threshold"), S["warning"],
                 f"threshold {pers.get('threshold')}"),
                (a.get("emergency_floor_mmHg"), S["critical"],
                 f"floor {a.get('emergency_floor_mmHg')}")]
        shipped_sbp = (a.get("shipped") or {}).get("sbp")
        h.append('<div class="pnl"><h3>Model 1 — forecaster</h3>'
                 '<p class="hint">Observed SBP and the forecast, per session'
                 + (f' · serving <b>{_esc(shipped_sbp)}</b>' if shipped_sbp else "")
                 + "</p>"
                 + _legend([(C["s1"], "Observed"), (C["s2"], "Forecast"),
                            (S["warning"], "Threshold"), (S["critical"], "Floor")])
                 + '<div class="sc">'
                 + _svg_series([(obs, C["s1"], False, [(len(hist) - 1, C["s1"])]),
                                (fut, C["s2"], True,
                                 [(len(hist) + i, C["s2"]) for i in range(len(fpts))])],
                               refs=refs, x_labels=xl)
                 + "</div>")
        if fpts:
            h.append('<table><thead><tr><th>Horizon</th><th>SBP</th><th>80% interval</th>'
                     "<th>Days</th><th>vs threshold</th></tr></thead><tbody>")
            for f in fpts:
                thr = pers.get("threshold")
                dl = (f"{'▲ ' if f['point'] >= thr else '▼ '}"
                      f"{abs(f['point'] - thr):.1f} "
                      f"{'over' if f['point'] >= thr else 'under'}") if thr else "–"
                band = (f"{f['lo80']} – {f['hi80']}" if f.get("lo80") is not None else "–")
                h.append(f"<tr><td>+{f['steps_ahead']} sessions</td><td>{f['point']}</td>"
                         f"<td>{band}</td><td>{f['days_ahead_est']}</td>"
                         f"<td>{_esc(dl)}</td></tr>")
            h.append("</tbody></table>")
        h.append("</div>")

    # ---- backtest ----
    bt = a.get("backtest") or {}
    if bt.get("horizons"):
        hh = bt["horizons"][0]
        rows = bt["series"].get(f"h{hh['horizon']}") or []
        if len(rows) >= 2:
            xl = [(0, rows[0]["ts"][5:]), (len(rows) // 2, rows[len(rows) // 2]["ts"][5:]),
                  (len(rows) - 1, rows[-1]["ts"][5:])]
            h.append('<div class="pnl"><h3>What we predicted, against what happened</h3>'
                     f'<p class="hint">Each point is the forecast made {hh["horizon"]} '
                     f'session(s) — about {hh["days_ahead"]} days — before that reading. '
                     "The features behind it never saw the day they predict.</p>"
                     + _legend([(C["s1"], "Actual"),
                                (C["s2"], f"Predicted {hh['days_ahead']} days earlier")])
                     + '<div class="sc">'
                     + _svg_series([([r["actual"] for r in rows], C["s1"], False, []),
                                    ([r["predicted"] for r in rows], C["s2"], True, [])],
                                   x_labels=xl, height=250)
                     + "</div><table><thead><tr><th>Horizon</th><th>Days ahead</th>"
                     "<th>n</th><th>MAE (mmHg)</th><th>within 10</th></tr></thead><tbody>")
            for x in bt["horizons"]:
                h.append(f"<tr><td>+{x['horizon']} sessions</td><td>{x['days_ahead']}</td>"
                         f"<td>{x['n']}</td><td>{x['mae']}</td>"
                         f"<td>{round(x['within_10'] * 100)}%</td></tr>")
            h.append("</tbody></table></div>")

    # ---- Model 3: anomaly ----
    an = a.get("anomaly") or {}
    scored = [p for p in (an.get("points") or []) if p["score"] is not None]
    if scored and len(an["points"]) >= 2:
        pts = an["points"]
        scores = [p["score"] for p in pts]     # None leaves a gap in the line
        marks = [(i, S["critical"]) for i, p in enumerate(pts) if p["flagged"]]
        xl = [(0, pts[0]["ts"][5:]), (len(pts) // 2, pts[len(pts) // 2]["ts"][5:]),
              (len(pts) - 1, pts[-1]["ts"][5:])]
        peak = max(scored, key=lambda p: p["score"])
        h.append('<div class="pnl"><h3>Model 3 — early-warning detector</h3>'
                 f'<p class="hint">{_esc(an["event_definition"])}'
                 + (f' · scorer <b>{_esc(an.get("detector"))}</b>'
                    if an.get("detector") else "") + "</p>"
                 + _legend([(C["s1"], "Detector score"), (S["warning"], "Alert cut"),
                            (S["critical"], "Flagged")])
                 + '<div class="sc">'
                 + _svg_series([(scores, C["s1"], False, marks)],
                               refs=[(an["cut"], S["warning"], f"cut {an['cut']}")],
                               x_labels=xl, height=250, y_fmt="{:.2f}",
                               shade=an.get("warmup_sessions") or None)
                 + "</div>"
                 + '<p class="co">'
                 + (f'{an["n_flagged"]} of {an["n_settled"]} settled sessions crossed the '
                    f'cut. Peak {peak["score"]:.3f} on {peak["ts"]}.'
                    if an["n_flagged"] else
                    f'No session crossed the cut. Peak {peak["score"]:.3f} on '
                    f'{peak["ts"]}, leaving {an["cut"] - peak["score"]:.3f} of headroom '
                    f'at the {an["budget_pct"]}% alert budget.')
                 + "</p></div>")

    # ---- rule engine ----
    if eng and not eng.get("error"):
        tl = "".join(
            f'<i style="background:{_TIER_COLOUR.get(x["tier"], "rgba(137,135,129,.28)")}" '
            f'title="{_esc(x["ts"])} {_esc(_pretty(x.get("rule_id")) or "no alert")}"></i>'
            for x in eng.get("history", []))
        fired = [x for x in eng.get("history", []) if x.get("fired")][::-1][:12]
        h.append('<div class="pnl"><h3>Rule engine</h3>'
                 '<p class="hint">The permanent floor — never trained, never tuned</p>'
                 f'<div class="tl">{tl}</div>'
                 f'<p class="hint">{eng.get("fired_count", 0)} of '
                 f'{len(eng.get("history", []))} readings fired</p>')
        if fired:
            h.append("<table><thead><tr><th>Session</th><th>Tier</th><th>Rule</th>"
                     "</tr></thead><tbody>")
            for x in fired:
                h.append(f'<tr><td>{_esc(x["ts"])}</td>'
                         f'<td>{_esc(_pretty(x["tier"]))}</td>'
                         f'<td>{_esc(_pretty(x["rule_id"]))}</td></tr>')
            h.append("</tbody></table>")
        reg = STORE.registry
        if reg is not None:
            blocked = int((reg.status == "BLOCKED_ON_INPUTS").sum())
            h.append(f'<p class="co">{blocked} of {len(reg)} rule slots are '
                     "BLOCKED_ON_INPUTS against the training corpus, which carries no "
                     "symptom, medication or condition columns. The engine is "
                     "deterministic, so whatever you enter above is evaluated here "
                     "immediately.</p>")
        h.append("</div>")
    elif eng.get("error"):
        h.append(f'<div class="pnl"><h3>Rule engine</h3>'
                 f'<p class="co">unavailable: {_esc(eng["error"])}</p></div>')

    # ---- governance ----
    h.append('<div class="pnl"><h3>Governance</h3><table><tbody>'
             f'<tr><td>Emergency floor</td><td>{_esc(a.get("emergency_floor_mmHg"))} mmHg'
             "</td><td>never personalised</td></tr>"
             f"<tr><td>Population threshold</td><td>{POPULATION_THRESHOLD_MMHG} mmHg</td>"
             "<td>the offset applies against this, within caps</td></tr>"
             f'<tr><td>Model version</td><td colspan="2">'
             f'{_esc(a.get("model_version"))}</td></tr></tbody></table>'
             '<p class="co">Provider-visible decision support. The ML layer never writes '
             "a DeviationAlert and never moves an emergency threshold.</p></div>")

    h.append("</div>")
    return "".join(h)


@_maybe_gpu
def _gradio_predict(readings_text, patient_id, age, sex, dm, pregnant, hf_type,
                    provider_target, conditions, medications, symptoms, position,
                    n_meas, missed_3d, adherence):
    if not STORE.ready:
        return (f'<div class="cp">{DASH_CSS}<div class="bn crit"><span>&#9888;</span>'
                f"<div><h4>No model loaded</h4><p>{_esc(STORE.error)}</p></div></div></div>",
                {})
    rows, errors = _parse_readings(readings_text or "")
    if errors:
        return (f'<div class="cp">{DASH_CSS}<div class="bn crit"><span>&#9888;</span>'
                "<div><h4>Could not parse the readings</h4><p>"
                + _esc(" · ".join(errors[:4])) + "</p></div></div></div>", {})
    if not rows:
        return (f'<div class="cp">{DASH_CSS}<div class="bn"><span>&#9675;</span>'
                "<div><h4>Enter at least one reading</h4></div></div></div>", {})

    rows[-1].symptoms = list(symptoms or [])
    rows[-1].position = position or "SITTING"
    rows[-1].n_meas = int(n_meas or 2)
    conds = list(conditions or [])
    if hf_type and hf_type != "NONE" and "has_hf" not in conds:
        conds.append("has_hf")

    req = PredictRequest(
        patient_id=patient_id or "demo",
        profile=Profile(age=float(age or 65), is_male=1 if sex == "Male" else 0,
                        is_dm=1 if dm == "Yes" else 0,
                        is_pregnant=1 if pregnant == "Yes" else 0,
                        hf_type=hf_type or "NONE", conditions=conds,
                        medications=list(medications or []),
                        provider_target=(float(provider_target)
                                         if provider_target else None),
                        missed_3d=int(missed_3d or 0),
                        adherence_7d=float(adherence if adherence is not None else 1.0)),
        readings=rows)

    import json
    a = json.loads(predict(req).body)
    return _render_dashboard(a), a


def _gradio_train():
    if TRAINING.snapshot()["running"]:
        return "A training run is already in progress."
    threading.Thread(target=_run_training, daemon=True).start()
    return ("Started. A full run streams the 250 MB corpus and does a forward-chained "
            "random search — expect tens of minutes. Press Refresh status for progress.")


def _gradio_train_status():
    s = TRAINING.snapshot()
    head = ("running" if s["running"] else
            f"finished (rc={s['returncode']})" if s["returncode"] is not None else "idle")
    return f"[{head}] started {s['started_at']}\n" + "\n".join(s["tail"][-20:])


with gr.Blocks(title="Cardioplace BP Alerts", analytics_enabled=False,
               css=".gradio-container{max-width:1500px!important}") as demo:
    gr.Markdown("## Cardioplace BP Alerts\n"
                "Forecasting · personalisation · early warning · deterministic rule engine")
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Patient")
            g_pid = gr.Textbox(label="Patient ID", value="demo-001")
            with gr.Row():
                g_age = gr.Number(label="Age", value=68, minimum=18, maximum=110)
                g_sex = gr.Radio(["Female", "Male"], label="Sex", value="Male")
            with gr.Row():
                g_dm = gr.Radio(["No", "Yes"], label="Diabetes", value="No")
                g_preg = gr.Radio(["No", "Yes"], label="Pregnant", value="No")
            with gr.Row():
                g_hf = gr.Dropdown(["NONE", "HFREF", "HFPEF"], label="Heart failure",
                                   value="NONE")
                g_pt = gr.Number(label="Provider SBP target", value=None)
            g_cond = gr.CheckboxGroup(CONDITION_KEYS, label="Conditions")
            g_med = gr.CheckboxGroup(MED_KEYS, label="Medications")
            with gr.Row():
                g_missed = gr.Number(label="Missed doses (3d)", value=0,
                                     minimum=0, maximum=3)
                g_adh = gr.Number(label="Adherence 7d", value=1.0, minimum=0, maximum=1)

            gr.Markdown("### Readings")
            g_readings = gr.Textbox(
                label="Session history", lines=10,
                info="One per line: date, SBP, DBP[, pulse, weight]",
                value=_sample_readings())
            gr.Markdown("### Today's symptoms  \n"
                        "<small>applied to the most recent reading — these are what "
                        "unlock the symptom-gated rules</small>")
            g_symp = gr.CheckboxGroup(SYMPTOM_KEYS, label="Symptoms")
            with gr.Row():
                g_pos = gr.Dropdown(["SITTING", "STANDING", "LYING"], label="Position",
                                    value="SITTING")
                g_nmeas = gr.Number(label="Readings in session", value=2,
                                    minimum=1, maximum=5)
            g_btn = gr.Button("Get advisory", variant="primary")

            with gr.Accordion("Train the model", open=False):
                gr.Markdown("Runs the full pipeline: ingest → validate → transform → "
                            "model trainer. Heavy for a free Space; training locally and "
                            "letting this serve the committed bundle is usually better.")
                g_train = gr.Button("Run training pipeline")
                g_refresh = gr.Button("Refresh status")
                g_tstatus = gr.Textbox(label="Pipeline status", lines=8,
                                       value="idle", interactive=False)

        with gr.Column(scale=2):
            g_out = gr.HTML(value=f'<div class="cp">{DASH_CSS}<div class="bn">'
                                  "<span>&#9675;</span><div><h4>No advisory yet</h4>"
                                  "<p>Enter the patient's history and select "
                                  "<b>Get advisory</b>. The forecaster needs 7 readings; "
                                  "the rule engine fires on every reading regardless.</p>"
                                  "</div></div></div>")
            with gr.Accordion("Raw advisory (JSON)", open=False):
                g_json = gr.JSON()

    g_btn.click(_gradio_predict,
                [g_readings, g_pid, g_age, g_sex, g_dm, g_preg, g_hf, g_pt,
                 g_cond, g_med, g_symp, g_pos, g_nmeas, g_missed, g_adh],
                [g_out, g_json])
    g_train.click(_gradio_train, None, g_tstatus)
    g_refresh.click(_gradio_train_status, None, g_tstatus)

# Off-Space, FastAPI owns the server and Gradio is mounted under it.
app = gr.mount_gradio_app(app, demo, path="/gradio")


def _server_port() -> int:
    """A Gradio Space is always proxied on 7860; it also sets PORT to an internal
    value that something else already holds, so PORT is read only off-Space."""
    if os.getenv("SPACE_ID"):
        return int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    return int(os.getenv("PORT") or os.getenv("GRADIO_SERVER_PORT") or 7860)


# A Gradio Space may IMPORT this module and launch `demo` itself rather than running
# it as __main__. Anything the dashboard depends on must therefore live at import
# time, in `demo` -- which it now does: the Blocks above renders the whole view from
# the server, with no /static fetch and no /api call. The model is loaded here for
# the same reason: the FastAPI lifespan does not run when Gradio owns the server.
if os.getenv("SPACE_ID") and not STORE.ready:
    STORE.load()


def _serve_on_space(port: int) -> None:
    """Run-as-__main__ path on a Space: Gradio owns the server, FastAPI routes are
    transplanted onto its app so the templates/ view and the REST API stay reachable.

    Routes are PREPENDED because Gradio's app ends in a catch-all and Starlette stops
    at the first match; appended routes silently 404. This is a convenience only --
    the Gradio Blocks alone is a complete dashboard if this never runs.
    """
    STORE.load()
    demo.launch(server_name="0.0.0.0", server_port=port,
                prevent_thread_lock=True, share=False)
    ours = [r for r in app.routes
            if getattr(r, "path", "").startswith(("/api", "/static"))]
    for route in reversed(ours):
        demo.app.routes.insert(0, route)
    logging.info("grafted %d FastAPI routes onto the Gradio app", len(ours))
    demo.block_thread()


if __name__ == "__main__":
    port = _server_port()
    if os.getenv("SPACE_ID"):
        _serve_on_space(port)
    else:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=port)
