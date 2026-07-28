"""Model 3 -- the early-warning detector set.

Port of notebook sections 8, 8.1a, 8.1b and 8.1c. Every score is causal: it reads lagged
features only, never step t's own reading. Higher = more anomalous, so the scorecard's
ranking logic is the same for every family.
"""

import numpy as np
import pandas as pd
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

from src.constants.training_pipeline import SEED
from src.logging.logger import logging

BASE_DETECTORS = ["d_personal_z", "d_ewma_gap", "d_slope7", "d_mean7_excess",
                  "d_isoforest", "d_lof", "d_elliptic", "d_fixed_threshold"]

FUSION_DETECTORS = ["d_fuse_mean", "d_fuse_median", "d_fuse_max",
                    "d_fuse_trimmed", "d_fuse_weighted"]

DETECTOR_FAMILY = {
    "d_personal_z": "control chart", "d_ewma_gap": "control chart",
    "d_slope7": "trend", "d_mean7_excess": "trend",
    "d_fixed_threshold": "reference (rule engine)",
    "d_isoforest": "isolation", "d_lof": "neighbourhood", "d_elliptic": "covariance",
    "d_forecast_level": "forecast", "d_forecast_breach": "forecast",
    "d_cusum": "change detection", "d_page_hinkley": "change detection",
    "d_ocsvm": "boundary", "d_pca_recon": "subspace", "d_gmm_nll": "density",
}
DETECTOR_FAMILY.update({c: "fusion (ensemble)" for c in FUSION_DETECTORS})


class EarlyWarningDataset:
    """Builds the future-window event label and the causal detector score matrix."""

    def __init__(self, config):
        self.config = config
        self.event_threshold_ = None

    def build(self, F: pd.DataFrame, features: list) -> pd.DataFrame:
        c = self.config
        # event threshold fitted on TRAIN only, so the test-period definition is not fitted on test
        self.event_threshold_ = (F[F.split == "train"].groupby("series_id")
                                 .sbp.quantile(c.event_quantile).rename("event_threshold"))
        D = F.merge(self.event_threshold_, on="series_id", how="left")
        D = D.sort_values(["series_id", "step"]).reset_index(drop=True)

        parts = []
        for sid, g in D.groupby("series_id", sort=False):
            g = g.sort_values("step").copy()
            thr = g.event_threshold.iloc[0]
            fut = pd.concat([g.sbp.shift(-i) for i in range(1, c.warn_window + 1)], axis=1)
            g["event_next"] = (np.nan if not np.isfinite(thr)
                               else (fut.max(axis=1) >= thr).astype(float))
            sbp = g.sbp.values
            first = np.full(len(g), np.nan)
            if np.isfinite(thr):
                for i in range(len(g)):
                    for j in range(1, c.warn_window + 1):
                        if i + j < len(g) and sbp[i + j] >= thr:
                            first[i] = i + j
                            break
            g["event_step"] = first
            parts.append(g)
        D = pd.concat(parts, ignore_index=True)

        # statistical detectors, all causal
        D["d_personal_z"] = D["sbp_z"].abs()
        D["d_ewma_gap"] = ((D["sbp_lag1"] - D["sbp_ewm0.1"]).abs()
                           / D["sbp_base_std"].replace(0, np.nan))
        D["d_slope7"] = D["sbp_slope7"]
        D["d_mean7_excess"] = D["sbp_mean7"] - D["sbp_base_mean"]
        D["d_fixed_threshold"] = D["sbp_lag1"]     # the rule-engine analogue
        return D


# ------------------------------------------------------------------- helpers

def ap_lift(df: pd.DataFrame, score: np.ndarray) -> float:
    """Average precision divided by base rate.

    Comparable across splits with different prevalence in a way raw AP is not.
    """
    s = pd.DataFrame({"y": df.event_next.values, "s": score}).dropna()
    if s.y.nunique() < 2 or len(s) < 50:
        return np.nan
    return float(average_precision_score(s.y, s.s) / max(s.y.mean(), 1e-9))


def _per_patient_running(df: pd.DataFrame, col: str, fn) -> np.ndarray:
    """Apply a causal running statistic within each patient, in step order."""
    out = np.full(len(df), np.nan)
    for _, g in df.groupby("series_id", sort=False):
        idx = g.sort_values("step").index
        out[df.index.get_indexer(idx)] = fn(df.loc[idx, col].values)
    return out


def _cusum(x: np.ndarray, k: float = 0.5) -> np.ndarray:
    """One-sided CUSUM on standardised values. Accumulates evidence of an upward shift."""
    x = np.asarray(x, float)
    m = np.nanmean(x[:20]) if np.isfinite(x[:20]).any() else np.nanmean(x)
    s = np.nanstd(x[:20]) or (np.nanstd(x) or 1.0)
    z = (x - m) / (s if s > 0 else 1.0)
    out, acc = np.zeros(len(x)), 0.0
    for i, v in enumerate(z):
        acc = 0.0 if not np.isfinite(v) else max(0.0, acc + v - k)
        out[i] = acc
    return out


def _page_hinkley(x: np.ndarray, delta: float = 0.05) -> np.ndarray:
    """Page-Hinkley statistic: cumulative departure from the running mean, minus a tolerance."""
    x = np.asarray(x, float)
    out, mu, n, mT, MT = np.zeros(len(x)), 0.0, 0, 0.0, 0.0
    for i, v in enumerate(x):
        if not np.isfinite(v):
            out[i] = MT - mT if n else 0.0
            continue
        n += 1
        mu += (v - mu) / n
        mT += v - mu - delta
        MT = min(MT, mT) if n > 1 else mT
        out[i] = mT - MT
    return out


def build_score_matrix(D: pd.DataFrame, features: list):
    """Median-impute the dense feature columns; returns (imputer, dense_cols, X_train, X_all)."""
    dense_cols = [c for c in features if D[c].notna().mean() > .5]
    imputer = SimpleImputer(strategy="median").fit(D[D.split == "train"][dense_cols])
    X_train = imputer.transform(D[D.split == "train"][dense_cols])
    X_all = imputer.transform(D[dense_cols])
    return imputer, dense_cols, X_train, X_all


def fit_default_detectors(D: pd.DataFrame, X_train, X_all) -> pd.DataFrame:
    """The three model-based detectors at their defaults, before tuning."""
    D["d_isoforest"] = -IsolationForest(n_estimators=200, random_state=SEED,
                                        n_jobs=2).fit(X_train).score_samples(X_all)
    D["d_lof"] = -LocalOutlierFactor(n_neighbors=25,
                                     novelty=True).fit(X_train).score_samples(X_all)
    D["d_elliptic"] = EllipticEnvelope(support_fraction=.9,
                                       random_state=SEED).fit(X_train).mahalanobis(X_all)
    return D


def tune_detectors(D: pd.DataFrame, imputer, dense_cols, X_train, X_all) -> dict:
    """Search on val, then rebind the detector columns to the winners.

    Without the rebinding the search is decoration: it produces a table while every
    downstream number still comes from the default parameters.
    """
    D_val = D[(D.split == "val") & D.event_next.notna()]
    best_detector = {}
    if len(D_val) < 50:
        logging.warning("validation split too thin to tune detectors; keeping defaults")
        return best_detector

    # The EWMA baseline is computed on the FULL frame and then filtered to val. Computing it
    # on the val slice alone would shift across patient boundaries and across the removed
    # train rows -- patient B's first val row differenced against patient A's last.
    def _ewma_gap_score(alpha: float) -> pd.Series:
        ewm = D.groupby("series_id").sbp.transform(
            lambda x: x.shift(1).ewm(alpha=alpha, adjust=False).mean())
        return (D["sbp_lag1"] - ewm).abs() / D["sbp_base_std"].replace(0, np.nan)

    best, chosen = -np.inf, None
    for alpha in (0.05, 0.1, 0.2, 0.4):
        full = _ewma_gap_score(alpha)
        v = ap_lift(D_val, full.loc[D_val.index].values)
        if np.isfinite(v) and v > best:
            best, chosen = v, {"ewma_alpha": alpha}
    best_detector["statistical"] = {"params": chosen,
                                    "val_ap_lift": round(best, 3) if np.isfinite(best) else None}

    best, chosen = -np.inf, None
    for n_est in (100, 200, 400):
        for max_f in (0.5, 0.8, 1.0):
            m = IsolationForest(n_estimators=n_est, max_features=max_f,
                                random_state=SEED, n_jobs=2).fit(X_train)
            v = ap_lift(D_val, -m.score_samples(imputer.transform(D_val[dense_cols])))
            if np.isfinite(v) and v > best:
                best, chosen = v, {"n_estimators": n_est, "max_features": max_f}
    best_detector["isolation_forest"] = {"params": chosen,
                                         "val_ap_lift": round(best, 3) if np.isfinite(best) else None}

    best, chosen = -np.inf, None
    for nn in (10, 25, 50):
        m = LocalOutlierFactor(n_neighbors=nn, novelty=True).fit(X_train)
        v = ap_lift(D_val, -m.score_samples(imputer.transform(D_val[dense_cols])))
        if np.isfinite(v) and v > best:
            best, chosen = v, {"n_neighbors": nn}
    best_detector["lof"] = {"params": chosen,
                            "val_ap_lift": round(best, 3) if np.isfinite(best) else None}

    # Carry the winners to the detector columns.
    if best_detector["statistical"]["params"]:
        D["d_ewma_gap"] = _ewma_gap_score(best_detector["statistical"]["params"]["ewma_alpha"])
    if best_detector["isolation_forest"]["params"]:
        D["d_isoforest"] = -IsolationForest(
            random_state=SEED, n_jobs=2,
            **best_detector["isolation_forest"]["params"]).fit(X_train).score_samples(X_all)
    if best_detector["lof"]["params"]:
        D["d_lof"] = -LocalOutlierFactor(
            novelty=True, **best_detector["lof"]["params"]).fit(X_train).score_samples(X_all)

    logging.info("detector tuning: %s", best_detector)
    return best_detector


def widen_detector_set(D: pd.DataFrame, X_train, X_all, features: list,
                       forecaster=None, forecast_target: str = None) -> list:
    """Add six further families, each carrying a different bias about what 'anomalous' means.

    forecast residual  -- where the forecaster thinks the patient is heading, not where they are
    CUSUM / Page-Hinkley -- sequential CHANGE detection; BP drift is a sustained shift
    one-class SVM      -- boundary of the normal region
    PCA reconstruction -- subspace; normal rows lie near a low-rank manifold
    GMM likelihood     -- density; normal rows are where the mass is
    """
    tr_mask = (D.split == "train").values
    new = []

    if forecaster is not None and forecast_target and forecast_target in D.columns:
        try:
            pred = forecaster.predict(D[features])
            # standardise per patient against their own TRAIN distribution, so the score is
            # "unusually high FOR THIS PATIENT", not "high in the population"
            mu = D[tr_mask].groupby("series_id").sbp.mean()
            sd = D[tr_mask].groupby("series_id").sbp.std().replace(0, np.nan)
            D["d_forecast_level"] = (pred - D.series_id.map(mu)) / D.series_id.map(sd)
            D["d_forecast_breach"] = pred - D.event_threshold
            new += ["d_forecast_level", "d_forecast_breach"]
        except Exception as exc:
            logging.warning("forecast-residual detector unavailable: %s", exc)

    D["d_cusum"] = _per_patient_running(D, "sbp_lag1", _cusum)
    D["d_page_hinkley"] = _per_patient_running(D, "sbp_lag1", _page_hinkley)

    sub = X_train[np.random.default_rng(SEED).permutation(len(X_train))[:4000]]
    D["d_ocsvm"] = -OneClassSVM(nu=.05, gamma="scale").fit(sub).decision_function(X_all)

    pca = PCA(n_components=min(10, X_train.shape[1]), random_state=SEED).fit(X_train)
    D["d_pca_recon"] = np.sqrt(
        ((X_all - pca.inverse_transform(pca.transform(X_all))) ** 2).sum(1))

    gmm = GaussianMixture(n_components=4, covariance_type="diag", random_state=SEED,
                          reg_covar=1e-4).fit(X_train)
    D["d_gmm_nll"] = -gmm.score_samples(X_all)

    new += ["d_cusum", "d_page_hinkley", "d_ocsvm", "d_pca_recon", "d_gmm_nll"]
    return new


def fuse_detectors(D: pd.DataFrame, detectors: list) -> list:
    """Normalise on TRAIN only, then combine the diverse legs.

    Anomaly scores sit on wildly different scales (a Mahalanobis distance and a CUSUM
    statistic are not comparable), so fusion normalises first. `d_fixed_threshold` is
    excluded from every fusion: it is the rule engine, the thing fusion has to beat, not
    an ingredient of it.
    """
    legs = [d for d in detectors if d != "d_fixed_threshold"]
    stats = {}
    for c in legs:
        tr_vals = D.loc[D.split == "train", c].replace([np.inf, -np.inf], np.nan).dropna()
        if len(tr_vals) < 50:
            continue
        stats[c] = (float(tr_vals.mean()), float(tr_vals.std() or 1.0))
    if not stats:
        return []

    Z = pd.DataFrame({c: (D[c] - m) / (s if s > 0 else 1.0)
                      for c, (m, s) in stats.items()}, index=D.index)
    Z = Z.clip(-8, 8)   # one exploding leg must not dominate the average

    D["d_fuse_mean"] = Z.mean(axis=1, skipna=True)
    D["d_fuse_median"] = Z.median(axis=1, skipna=True)
    D["d_fuse_max"] = Z.max(axis=1, skipna=True)   # "any family is alarmed" -- high recall
    D["d_fuse_trimmed"] = Z.apply(
        lambda r: pd.Series(r).sort_values().iloc[1:-1].mean() if r.notna().sum() >= 4
        else r.mean(), axis=1)

    # Weighted fusion: weights from average precision on the VALIDATION split, never test.
    val = D[(D.split == "val") & D.event_next.notna()]
    W = {}
    if len(val) > 100 and val.event_next.nunique() > 1:
        for c in stats:
            s = val[[c, "event_next"]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(s) > 50 and s.event_next.nunique() > 1:
                ap = average_precision_score(s.event_next, s[c])
                base = float(s.event_next.mean())
                W[c] = max(ap - base, 0.0)   # lift over base rate, floored at zero
    if sum(W.values()) > 0:
        wv = pd.Series(W)
        wv = wv / wv.sum()
        D["d_fuse_weighted"] = (Z[list(wv.index)] * wv.values).sum(axis=1, skipna=True)
    else:
        D["d_fuse_weighted"] = D["d_fuse_mean"]
        logging.info("validation split too thin for weighted fusion -- using unweighted mean")

    return list(FUSION_DETECTORS)
