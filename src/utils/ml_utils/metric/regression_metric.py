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


def forecast_baselines(df: pd.DataFrame, signal: str, h: int) -> dict:
    """The floor every learned model must clear, per signal and horizon."""
    last = df[f"{signal}_lag1"].values
    return {
        "persistence": last,
        "seasonal_naive_7": df.get(f"{signal}_lag7", pd.Series(np.nan, index=df.index)).values,
        "drift": last + h * (df[f"{signal}_lag1"].values - df[f"{signal}_lag2"].values),
        "personal_mean": df[f"{signal}_base_mean"].values,
        "ewma": df[f"{signal}_ewm0.3"].values,
    }


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
    """Select on validation MAE, then apply the ship rule on test."""
    try:
        val = (R[(R.split == "val") & (R.family == "learned")]
               .groupby(["signal", "model"]).MAE.mean().unstack())
        if val.empty:
            return {}, pd.DataFrame()
        winner = {s: val.loc[s].idxmin() for s in val.index}

        test = R[R.split == "test"]
        rows = []
        for s in val.index:
            base = test[(test.signal == s) & (test.family == "baseline")]
            best_base = base.loc[base.MAE.idxmin()] if len(base) else None
            sel = test[(test.signal == s) & (test.model == winner[s])]
            if best_base is None or not len(sel):
                continue
            learned_mae = float(sel.MAE.mean())
            ci_w = float(sel.MAE_hi.mean() - sel.MAE_lo.mean())
            verdict, gain = ship_decision(learned_mae, float(best_base.MAE), ci_w)
            rows.append(dict(signal=s, selected=winner[s], learned_MAE=round(learned_mae, 3),
                             best_baseline=best_base.model,
                             baseline_MAE=round(float(best_base.MAE), 3),
                             gain_mmHg=round(gain, 3), CI_width=round(ci_w, 3), ship=verdict))
        return winner, pd.DataFrame(rows)
    except Exception as e:
        raise CustomException(e, sys)
