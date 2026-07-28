"""Point-forecast metrics, baselines and the ship decision.

Port of notebook sections 5 and 6B.5. The decision rule is the point of this module:
a learned model ships only when it beats the best baseline by more than the bootstrap
CI width. A smoother needs no registry, no drift monitor and no change-control plan.
"""

import sys

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.constants.training_pipeline import SEED
from src.entity.artifact_entity import ForecastMetricArtifact
from src.exception.custom_exception import CustomException


def bootstrap_ci(fn, y: np.ndarray, p: np.ndarray,
                 n_boot: int = 200, seed: int = SEED):
    """Point estimate plus a percentile bootstrap 95% interval for `fn`."""
    rng = np.random.default_rng(seed)
    m = np.isfinite(y) & np.isfinite(p)
    y, p = y[m], p[m]
    if len(y) < 10:
        return np.nan, (np.nan, np.nan)
    idx = rng.integers(0, len(y), size=(n_boot, len(y)))
    vals = np.array([fn(y[i], p[i]) for i in idx])
    return float(fn(y, p)), (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def evaluate(y, p, last=None, lo=None, hi=None, **meta) -> dict:
    """Point-forecast metrics with CI, bias, direction and (optionally) interval coverage."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    m = np.isfinite(y) & np.isfinite(p)
    y, p = y[m], p[m]
    if len(y) < 10:
        return {}
    mae, (lo_ci, hi_ci) = bootstrap_ci(mean_absolute_error, y, p)
    out = dict(meta, n=len(y), MAE=round(mae, 3), MAE_lo=round(lo_ci, 3), MAE_hi=round(hi_ci, 3),
               RMSE=round(float(np.sqrt(mean_squared_error(y, p))), 3),
               R2=round(float(r2_score(y, p)), 3), bias=round(float(np.mean(p - y)), 2),
               target_sd=round(float(np.std(y)), 2), unit="per patient-step")
    if last is not None:
        b = np.asarray(last, float)[m]
        ok = np.isfinite(b)
        dt, dp = y[ok] - b[ok], p[ok] - b[ok]
        nz = np.abs(dt) > 1e-9
        out["DirAcc"] = (round(float(np.mean(np.sign(dt[nz]) == np.sign(dp[nz]))), 3)
                         if nz.sum() and np.abs(dp).max() > 1e-9 else np.nan)
    if lo is not None:
        lo_a, hi_a = np.asarray(lo, float)[m], np.asarray(hi, float)[m]
        out["Cover80"] = round(float(np.mean((y >= lo_a) & (y <= hi_a))), 3)
        out["IntWidth"] = round(float(np.mean(hi_a - lo_a)), 1)
    return out


# Each baseline as a closed form over columns the feature matrix already carries. This is
# the single definition: `forecast_baselines` scores them and `BaselineForecaster` serves
# them, so a baseline that wins the ship decision cannot behave differently once shipped.
BASELINE_SPECS = {
    "persistence": lambda d, s, h: d[f"{s}_lag1"].to_numpy(float),
    "seasonal_naive_7": lambda d, s, h: (d[f"{s}_lag7"].to_numpy(float)
                                         if f"{s}_lag7" in d
                                         else np.full(len(d), np.nan)),
    "drift": lambda d, s, h: (d[f"{s}_lag1"].to_numpy(float)
                              + h * (d[f"{s}_lag1"].to_numpy(float)
                                     - d[f"{s}_lag2"].to_numpy(float))),
    "personal_mean": lambda d, s, h: d[f"{s}_base_mean"].to_numpy(float),
    "ewma": lambda d, s, h: d[f"{s}_ewm0.3"].to_numpy(float),
}


def forecast_baselines(df: pd.DataFrame, signal: str, h: int) -> dict:
    """The floor every learned model must clear, per signal and horizon."""
    return {name: fn(df, signal, h) for name, fn in BASELINE_SPECS.items()}


def ship_decision(learned_mae: float, baseline_mae: float, ci_width: float):
    """learned model ships iff (baseline MAE - learned MAE) > bootstrap CI width."""
    gain = baseline_mae - learned_mae
    return ("learned model" if gain > ci_width else "BASELINE"), gain


def paired_delta(err_a: np.ndarray, err_b: np.ndarray, n_boot: int = 200, seed: int = SEED):
    """Paired bootstrap on per-row error differences: mean delta, 95% CI, and n.

    Paired rather than independent because both models see the same rows; the unpaired
    interval is far wider than the evidence warrants.
    """
    a, b = np.asarray(err_a, float), np.asarray(err_b, float)
    m = np.isfinite(a) & np.isfinite(b)
    d = a[m] - b[m]
    if len(d) < 10:
        return np.nan, (np.nan, np.nan), int(len(d))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    return (float(d.mean()),
            (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))),
            int(len(d)))


def get_forecast_score(row: dict) -> ForecastMetricArtifact:
    """Turn one `evaluate` row into a typed artifact."""
    try:
        return ForecastMetricArtifact(
            signal=row.get("signal", ""),
            horizon=int(row.get("horizon", 0)),
            model=row.get("model", ""),
            n=int(row.get("n", 0)),
            mae=float(row.get("MAE", np.nan)),
            mae_lo=float(row.get("MAE_lo", np.nan)),
            mae_hi=float(row.get("MAE_hi", np.nan)),
            rmse=float(row.get("RMSE", np.nan)),
            r2=float(row.get("R2", np.nan)),
            bias=float(row.get("bias", np.nan)),
            target_sd=float(row.get("target_sd", np.nan)),
            dir_acc=(float(row["DirAcc"]) if row.get("DirAcc") is not None
                     and np.isfinite(row.get("DirAcc", np.nan)) else None),
            split=row.get("split", "test"),
        )
    except Exception as e:
        raise CustomException(e, sys)


def select_and_decide(R: pd.DataFrame, config):
    """Select on validation MAE, then apply the ship rule on test.

    Both sides of the comparison are averaged over the SAME horizons. Scoring the
    learned model across h1..h3 against the single best baseline cell -- which is
    always h1, because MAE grows with horizon -- makes the learned model answer for
    the hard horizons while the baseline is credited only with the easy one, and
    understates the gain by roughly the h1-to-h3 spread.
    """
    try:
        val = (R[(R.split == "val") & (R.family == "learned")]
               .groupby(["signal", "model"]).MAE.mean().unstack())
        if val.empty:
            return {}, pd.DataFrame()
        winner = {s: val.loc[s].idxmin() for s in val.index}

        test = R[R.split == "test"]
        rows = []
        for s in val.index:
            sel = test[(test.signal == s) & (test.model == winner[s])]
            base = test[(test.signal == s) & (test.family == "baseline")]
            if not len(sel) or not len(base):
                continue
            horizons = sorted(set(sel.horizon))
            base = base[base.horizon.isin(horizons)]
            # mean per baseline over the horizons the learned model is answering for
            per_base = base.groupby("model").MAE.mean()
            covers = base.groupby("model").horizon.nunique() == len(horizons)
            per_base = per_base[covers[covers].index] if covers.any() else per_base
            if per_base.empty:
                continue
            best_name = per_base.idxmin()
            learned_mae = float(sel.MAE.mean())
            ci_w = float(sel.MAE_hi.mean() - sel.MAE_lo.mean())
            verdict, gain = ship_decision(learned_mae, float(per_base.min()), ci_w)
            rows.append(dict(signal=s, selected=winner[s], learned_MAE=round(learned_mae, 3),
                             best_baseline=best_name,
                             baseline_MAE=round(float(per_base.min()), 3),
                             gain_mmHg=round(gain, 3), CI_width=round(ci_w, 3),
                             horizons=len(horizons), ship=verdict))
        return winner, pd.DataFrame(rows)
    except Exception as e:
        raise CustomException(e, sys)


def shipped_forecasters(decision: pd.DataFrame, winner: dict) -> dict:
    """signal -> ('learned', kind) or ('baseline', name), honouring the ship decision.

    Without this the verdict is a report nobody reads: `build` froze the learned winner
    whatever the rule said, so a model that never cleared the bar still served every
    request.
    """
    out = {s: ("learned", k) for s, k in winner.items()}
    for r in decision.itertuples() if len(decision) else []:
        if str(r.ship).upper() == "BASELINE":
            out[r.signal] = ("baseline", r.best_baseline)
    return out
