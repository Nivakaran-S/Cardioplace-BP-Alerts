"""Panel construction, splits and strictly-causal feature engineering.

Port of notebook sections 1.3, 1.4, 3 and 3.2. The causal contract enforced here is the
single most consequential thing in the pipeline: a feature at step t may read only steps
<= t-1. `leakage_audit` checks it directly rather than assuming it.
"""

import numpy as np
import pandas as pd

from src.logging.logger import logging
from src.utils.main_utils.utils import deterministic_patient_split
from src.utils.ml_utils.feature.cadence import attach_cadence


def build_panel(sessions: pd.DataFrame, static: pd.DataFrame, config) -> pd.DataFrame:
    """Admit eligible patients, index them by session, and attach demographics."""
    g = sessions.groupby("patient_id")
    span = (g.ts.max() - g.ts.min()).dt.days
    sizes = g.size()
    eligible = [p for p in sizes[sizes >= config.min_sessions].index
                if span[p] >= config.min_span_days]
    df = sessions[sessions.patient_id.isin(eligible)]
    logging.info("%d patients meet >=%d sessions and >=%d days span",
                 len(eligible), config.min_sessions, config.min_span_days)

    if config.max_patients and len(eligible) > config.max_patients:
        # deterministic in the identifier, so the cohort does not change with row order
        ranked = sorted(eligible, key=lambda p: (-int(sizes[p]), str(p)))[:config.max_patients]
        df = df[df.patient_id.isin(ranked)]
        logging.info("capped to %d patients (set MAX_PATIENTS=None for all %d)",
                     config.max_patients, len(eligible))

    p = df.sort_values(["patient_id", "ts"]).copy()
    p["series_id"] = p.patient_id
    p = attach_cadence(p, by="series_id")
    p["step"] = p.groupby("series_id").cumcount()
    p = p.merge(static, on="patient_id", how="left")

    # derived clinical quantities used by both the rule engine and the features
    p["pulse_pressure"] = p.sbp - p.dbp
    p["map_est"] = p.dbp + (p.sbp - p.dbp) / 3.0
    p["dow"] = p.ts.dt.dayofweek
    p["is_weekend"] = (p.dow >= 5).astype(int)
    p["is_male"] = p.gender.astype(str).str.upper().str.startswith("M").astype(int)
    p["is_dm"] = p.DM.fillna(0).astype(int)
    p["age_band"] = pd.cut(p.age, [0, 50, 65, 75, 200],
                           labels=["<50", "50-64", "65-74", "75+"], right=False)
    return p


def add_splits(df: pd.DataFrame, config) -> pd.DataFrame:
    """Temporal tail per patient -> val/test, plus a hash-based patient-level holdout.

    A random row split would put a patient's future readings in train and their past in
    test, which inflates every metric downstream.
    """
    out = df.copy()
    last = out.groupby("series_id")["step"].transform("max")
    test_edge = (last * (1 - config.test_frac)).round()
    val_edge = (last * (1 - config.test_frac - config.val_frac)).round()
    out["split"] = np.where(out.step > test_edge, "test",
                            np.where(out.step > val_edge, "val", "train"))
    assign = deterministic_patient_split(sorted(out.series_id.unique()),
                                         config.holdout_patient_frac,
                                         salt=config.patient_split_salt)
    out["patient_split"] = out.series_id.map(assign)
    return out


class CausalFeatureBuilder:
    """Builds strictly-causal per-session features and the forecast targets.

    Contract: for a row at step t, every feature reads only steps <= t-1, and the target
    y_<signal>_h<h> is the observed value at step t+h. Violating this contract is the single
    most consequential bug available in this pipeline, so `leakage_audit` tests it directly.
    """

    STATIC_PASSTHROUGH = ["age", "is_male", "is_dm", "is_weekend", "days_since_last"]
    # `days_since_last_raw` is the unclipped truth and must never become a model input --
    # feature_names_ below admits every numeric column not listed here, so omitting it would
    # silently grow the feature list and change every estimator's input matrix.
    # `cadence_audit` asserts this rather than trusting the comment.
    DROP = {"ts", "series_id", "patient_id", "gender", "step", "age_band", "split",
            "patient_split", "sbp", "dbp", "idwg", "weight", "sbp_post", "sbp_min",
            "sbp_drop", "uf_total", "temperature", "weightstart", "weightend", "dryweight",
            "n_meas", "pulse_pressure", "map_est", "DM", "keyindate", "datatime", "dow",
            "days_since_last_raw", "reading_age_days"}

    def __init__(self, config):
        self.config = config
        self.feature_names_ = None

    @staticmethod
    def _rolling_slope(v: pd.Series, w: int) -> pd.Series:
        # Vectorised rolling OLS slope. x = 0..w-1 is fixed, so
        #   slope = [mean(x*y) - mean(x)*mean(y)] / var(x)
        # A .rolling().apply(np.polyfit) equivalent is ~200x slower and dominates runtime.
        y = v.astype(float)
        x_mean, x_var = (w - 1) / 2.0, (w * w - 1) / 12.0
        yv = y.values
        dot = np.full(len(yv), np.nan)
        if len(yv) >= w:
            dot[w - 1:] = np.convolve(np.nan_to_num(yv), np.arange(w, dtype=float)[::-1],
                                      mode="valid")
        return (pd.Series(dot, index=y.index) / w
                - x_mean * y.rolling(w, min_periods=w).mean()) / x_var

    def _one_series(self, g: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        g = g.sort_values("step").reset_index(drop=True).copy()
        for s in cfg.signals:
            v = g[s]
            for lag in cfg.lags:
                g[f"{s}_lag{lag}"] = v.shift(lag)
            for w in cfg.windows:
                r = v.shift(1).rolling(w, min_periods=max(2, w // 3))
                g[f"{s}_mean{w}"] = r.mean()
                g[f"{s}_std{w}"] = r.std()
                g[f"{s}_min{w}"] = r.min()
                g[f"{s}_max{w}"] = r.max()
                g[f"{s}_range{w}"] = g[f"{s}_max{w}"] - g[f"{s}_min{w}"]
                g[f"{s}_slope{w}"] = self._rolling_slope(v.shift(1), w)
            for a in cfg.ewm_alphas:
                g[f"{s}_ewm{a}"] = v.shift(1).ewm(alpha=a, adjust=False).mean()
            exp = v.shift(1).expanding(min_periods=5)
            g[f"{s}_base_mean"] = exp.mean()
            g[f"{s}_base_std"] = exp.std()
            g[f"{s}_z"] = (v.shift(1) - g[f"{s}_base_mean"]) / g[f"{s}_base_std"].replace(0, np.nan)
            g[f"{s}_d1"] = v.shift(1) - v.shift(2)

        if g.weight.notna().sum() > 3:
            g["weight_lag1"] = g.weight.shift(1)
            g["weight_slope24"] = self._rolling_slope(g.weight.shift(1), 24)
            g["weight_delta_base"] = (g.weight.shift(1)
                                      - g.weight.shift(1).expanding(min_periods=5).median())
        else:
            for c in ("weight_lag1", "weight_slope24", "weight_delta_base"):
                g[c] = np.nan

        g["gap_mean24"] = g.days_since_last.shift(1).rolling(24, min_periods=4).mean()
        g["sbp_drop_lag1"] = g.sbp_drop.shift(1)
        g["uf_lag1"] = g.uf_total.shift(1)
        for h in cfg.horizons:
            for s in cfg.signals:
                g[f"y_{s}_h{h}"] = g[s].shift(-h)
        return g

    def transform(self, panel: pd.DataFrame) -> pd.DataFrame:
        out = pd.concat([self._one_series(g) for _, g in panel.groupby("series_id", sort=False)],
                        ignore_index=True)
        for c in out.select_dtypes("float64").columns:
            out[c] = out[c].astype("float32")   # halves memory; mmHg/kg lose no usable precision
        self.feature_names_ = [c for c in out.columns
                               if c not in self.DROP and not c.startswith("y_")
                               and pd.api.types.is_numeric_dtype(out[c])]
        return out

    def transform_for_inference(self, history: pd.DataFrame,
                                as_of: pd.Timestamp = None) -> pd.DataFrame:
        """Feature row describing 'now' for one patient.

        Training pairs features at step t (which see <= t-1) with the reading at t+h. Applied
        naively at serving time that discards the newest reading. Appending one placeholder step
        puts the newest reading inside the .shift(1) window -- the exact training distribution --
        and makes horizon h mean h steps ahead of now. Getting this wrong shifts every forecast
        by one step and is invisible in offline metrics.

        `as_of` is the moment the advisory is being asked for. Without it the placeholder is
        parked one MEDIAN gap after the last reading, which silently asserts the patient is
        up to date: someone who last logged 30 days ago got a forecast built as though they
        had just logged. With it, the placeholder sits at the real "now", so the cadence
        features carry the true staleness. `as_of=None` reproduces the old behaviour exactly,
        which is what every offline caller relies on.
        """
        g = history.sort_values("ts").copy()
        med_gap = float(g.ts.diff().dt.days.median() or 2)
        ph = g.iloc[[-1]].copy()
        nominal = g.ts.max() + pd.Timedelta(days=med_gap)
        ph["ts"] = nominal if as_of is None else max(pd.Timestamp(as_of),
                                                     g.ts.max() + pd.Timedelta(days=1))
        for c in ("sbp", "dbp", "idwg", "weight", "sbp_drop", "uf_total"):
            if c in ph:
                ph[c] = np.nan
        gg = pd.concat([g, ph], ignore_index=True)
        gg["series_id"] = str(g.patient_id.iloc[0])
        gg = attach_cadence(gg, by=None)
        gg["step"] = np.arange(len(gg))
        gg["is_weekend"] = (gg.ts.dt.dayofweek >= 5).astype(int)
        for c, d in (("DM", 0), ("is_dm", 0), ("is_male", 0), ("age", 65.0)):
            if c not in gg:
                gg[c] = d
        return self._one_series(gg).iloc[[-1]]


def feature_group(name: str) -> str:
    """Human-readable family for a feature name, used by the feature dictionary."""
    if name.startswith(("sbp_", "dbp_", "idwg_")):
        base = name.split("_", 1)[1]
        if base.startswith("lag"):
            return "lag"
        if base.startswith(("mean", "std", "min", "max", "range")):
            return "rolling moment"
        if base.startswith("slope"):
            return "trend"
        if base.startswith("ewm"):
            return "smoother"
        if base.startswith("base"):
            return "personal baseline"
        if base == "z":
            return "personal z-score"
        if base == "d1":
            return "first difference"
    if name.startswith("weight"):
        return "fluid / weight"
    if name in ("age", "is_male", "is_dm"):
        return "static"
    if name in ("days_since_last", "gap_mean24", "is_weekend"):
        return "cadence"
    return "other"


def feature_dictionary(F: pd.DataFrame, features: list) -> pd.DataFrame:
    """Feature name -> group, signal and observed density."""
    d = pd.DataFrame({"feature": features})
    d["group"] = d.feature.map(feature_group)
    d["signal"] = d.feature.str.split("_").str[0].where(
        d.feature.str.startswith(("sbp", "dbp", "idwg", "weight")), "-")
    d["density"] = d.feature.map(F[features].notna().mean())
    return d


def leakage_audit(F: pd.DataFrame, features: list, config) -> pd.DataFrame:
    """Structural probes over the feature matrix and the split layout.

    Deliberately not a correlation screen: on a 3x/week dialysis cadence a lag-2 feature
    lands on the same weekday as the current reading, so correlation flags the cadence
    artefact and calls it leakage. These probes test the contract directly instead.
    """
    rows = []

    def add(name, ok, detail=""):
        rows.append(dict(probe=name, status="PASS" if ok else "FAIL", detail=detail))

    corr = F[features].corrwith(F.sbp).abs().sort_values(ascending=False)
    add("no feature is a copy of the current reading", corr.max() < 0.999,
        f"max |corr| {corr.max():.4f} ({corr.index[0]})")

    probe = F.sort_values(["series_id", "step"]).groupby("series_id").head(80)
    exp = probe.groupby("series_id").sbp.shift(1)
    m = exp.notna()
    add("sbp_lag1 == previous row's SBP, exactly",
        bool(np.allclose(exp[m].values, probe.loc[m, "sbp_lag1"].values, atol=1e-3)))

    add("no rolling feature precedes its first lag",
        int((F.sbp_mean3.notna() & F.sbp_lag1.isna()).sum()) == 0)

    bad = []
    for sid, g in F.groupby("series_id"):
        mx, mn = g.groupby("split").step.max(), g.groupby("split").step.min()
        if {"train", "val"} <= set(mx.index) and mx["train"] >= mn["val"]:
            bad.append(sid)
        elif {"val", "test"} <= set(mx.index) and mx["val"] >= mn["test"]:
            bad.append(sid)
    add("train precedes val precedes test within every patient", not bad,
        f"{len(bad)} patients out of order")

    add("patient split is disjoint",
        int((F.groupby("series_id").patient_split.nunique() > 1).sum()) == 0)

    inf_cols = [c for c in features if np.isinf(F[c].fillna(0)).any()]
    add("no infinite values in the feature matrix", not inf_cols, f"{inf_cols[:5]}")

    for h in config.horizons:
        cov = float(F[f"y_sbp_h{h}"].notna().mean())
        add(f"target y_sbp_h{h} present", cov > 0.5, f"{cov:.1%} of rows")

    sparse = [c for c in features if F[c].notna().mean() < 0.30]
    rows.append(dict(probe="every feature non-null on >30% of rows",
                     status="PASS" if not sparse else "WARN", detail=f"{len(sparse)} sparse"))
    return pd.DataFrame(rows)
