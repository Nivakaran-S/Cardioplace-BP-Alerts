"""Model factory, tuning, sweep and the frozen serving artifact.

Port of notebook sections 6, 6.1, 6.2, 6.4 and 10. `make_model` is the single place a
model is constructed, so training, tuning, evaluation and serving cannot drift apart.
"""

import sys
import time

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.constants.training_pipeline import GOVERNANCE_KEYS, SEED
from src.exception.custom_exception import CustomException
from src.logging.logger import logging
from src.utils.ml_utils.feature.causal_features import CausalFeatureBuilder
from src.utils.ml_utils.metric.regression_metric import evaluate, forecast_baselines
from src.utils.ml_utils.model.offset import OffsetModel

MODEL_KINDS = ("ridge", "elasticnet", "hgb")

SEARCH_SPACE = {
    "ridge": {"alpha": [0.1, 1.0, 10.0, 100.0, 300.0]},
    "elasticnet": {"alpha": [0.005, 0.02, 0.05, 0.2], "l1_ratio": [0.1, 0.5, 0.9]},
    "hgb": {"max_iter": [150, 250, 400], "learning_rate": [0.03, 0.07, 0.12],
            "max_leaf_nodes": [15, 31, 63], "min_samples_leaf": [20, 50, 100],
            "l2_regularization": [0.0, 0.5, 2.0], "max_features": [0.6, 0.8, 1.0]},
}

# A governance parameter that leaked into a hyperparameter grid would be silently tuned
# against the data, which is exactly what "never searched" is meant to prevent.
assert not (set(GOVERNANCE_KEYS) & {k for g in SEARCH_SPACE.values() for k in g}), \
    "a governance parameter leaked into a hyperparameter grid"


def make_model(kind: str, **kw) -> BaseEstimator:
    """Factory so that training, tuning, testing and serving all construct models identically."""
    if kind == "ridge":
        return Pipeline([("impute", SimpleImputer(strategy="median")),
                         ("scale", StandardScaler()),
                         ("model", Ridge(alpha=kw.get("alpha", 10.0)))])
    if kind == "elasticnet":
        return Pipeline([("impute", SimpleImputer(strategy="median")),
                         ("scale", StandardScaler()),
                         ("model", ElasticNet(alpha=kw.get("alpha", .05),
                                              l1_ratio=kw.get("l1_ratio", .5),
                                              random_state=SEED))])
    if kind == "hgb":
        return HistGradientBoostingRegressor(
            loss="absolute_error", random_state=SEED,
            max_iter=kw.get("max_iter", 250), learning_rate=kw.get("learning_rate", .07),
            max_leaf_nodes=kw.get("max_leaf_nodes", 31),
            min_samples_leaf=kw.get("min_samples_leaf", 20),
            l2_regularization=kw.get("l2_regularization", 0.0),
            max_features=kw.get("max_features", 1.0))
    raise ValueError(f"unknown model kind: {kind}")


# ------------------------------------------------------------------------------ sweep

def run_sweep(F: pd.DataFrame, features: list, config, params: dict = None, seed: int = SEED):
    """Fit on train; score on val (for selection) and test (for reporting), per signal x horizon."""
    params = params or {}
    rows, preds, fitted = [], {}, {}
    for signal in config.signals:
        for h in config.horizons:
            target = f"y_{signal}_h{h}"
            if target not in F.columns:
                continue
            d = F[F[target].notna()]
            tr, va, te = d[d.split == "train"], d[d.split == "val"], d[d.split == "test"]
            if len(tr) < 200 or len(te) < 40:
                continue
            if len(tr) > config.max_train_rows:
                tr = tr.sample(config.max_train_rows, random_state=seed)
            if len(va) > config.max_eval_rows:
                va = va.sample(config.max_eval_rows, random_state=seed)
            if len(te) > config.max_eval_rows:
                te = te.sample(config.max_eval_rows, random_state=seed)

            for name, pred in forecast_baselines(te, signal, h).items():
                r = evaluate(te[target].values, pred, te[f"{signal}_lag1"].values,
                             model=name, family="baseline", signal=signal, horizon=h,
                             split="test")
                if r:
                    rows.append(r)

            for kind in MODEL_KINDS:
                mdl = make_model(kind, **params.get((kind, signal), {}))
                mdl.fit(tr[features], tr[target].values)
                fitted[(kind, signal, h)] = mdl
                for label, part in (("val", va), ("test", te)):
                    if len(part) < 40:
                        continue
                    r = evaluate(part[target].values, mdl.predict(part[features]),
                                 part[f"{signal}_lag1"].values, model=kind,
                                 family="learned", signal=signal, horizon=h, split=label)
                    if r:
                        rows.append(r)
                if signal == "sbp" and h == config.horizons[min(1, len(config.horizons) - 1)]:
                    preds[kind] = (te[target].values, mdl.predict(te[features]),
                                   te[f"{signal}_lag1"].values)
    return pd.DataFrame(rows), preds, fitted


# ----------------------------------------------------------------------------- tuning

def forward_chain_folds(d: pd.DataFrame, n_folds: int):
    """Expanding-window folds inside each patient's own timeline."""
    q = d.groupby("series_id").step.transform(lambda s: s.rank(pct=True))
    edges = np.linspace(0.4, 1.0, n_folds + 1)
    for i in range(n_folds):
        yield (q <= edges[i]).values, ((q > edges[i]) & (q <= edges[i + 1])).values


def cv_score(kind: str, params: dict, d: pd.DataFrame, target: str,
             features: list, n_folds: int) -> float:
    errs = []
    for tr_m, va_m in forward_chain_folds(d, n_folds):
        tr, va = d[tr_m], d[va_m]
        if len(tr) < 200 or len(va) < 50:
            continue
        m = make_model(kind, **params).fit(tr[features], tr[target].values)
        errs.append(mean_absolute_error(va[target].values, m.predict(va[features])))
    return float(np.mean(errs)) if errs else np.nan


def random_search(F: pd.DataFrame, features: list, config, best_params: dict,
                  seed: int = SEED) -> pd.DataFrame:
    """Random draws from SEARCH_SPACE scored by forward-chained CV. Fills `best_params`."""
    rng = np.random.default_rng(seed)
    log = []
    for signal in config.signals:
        target = f"y_{signal}_h{config.horizons[0]}"
        if target not in F.columns:
            continue
        d = F[(F[target].notna()) & (F.split != "test")]
        if len(d) > 12_000:
            d = d.sample(12_000, random_state=seed).sort_values(["series_id", "step"])
        for kind in MODEL_KINDS:
            base = cv_score(kind, {}, d, target, features, config.tune_folds)
            best, best_p = base, {}
            for _ in range(config.tune_draws):
                grid = SEARCH_SPACE[kind]
                cand = {k: (float(rng.choice(v)) if isinstance(v[0], float)
                            else rng.choice(v).item())
                        for k, v in grid.items()}
                sc = cv_score(kind, cand, d, target, features, config.tune_folds)
                if np.isfinite(sc) and np.isfinite(best) and sc < best:
                    best, best_p = sc, cand
            best_params[(kind, signal)] = best_p
            log.append(dict(model=kind, signal=signal,
                            cv_mae_default=round(base, 4) if np.isfinite(base) else np.nan,
                            cv_mae_tuned=round(best, 4) if np.isfinite(best) else np.nan,
                            gain_pct=(round(100 * (base - best) / base, 2)
                                      if np.isfinite(base) and base else np.nan),
                            params=str(best_p)))
    return pd.DataFrame(log)


def fit_quantile_interval(F: pd.DataFrame, features: list, config, best_params: dict,
                          signal: str, horizon: int, seed: int = SEED):
    """Quantile GBM band fit on train, conformalised on val.

    The conformal widening is calibrated on val and deliberately NOT refit later --
    refitting it on data the band already saw would void the coverage guarantee.
    """
    target = f"y_{signal}_h{horizon}"
    if target not in F.columns:
        return {}, 0.0
    di = F[F[target].notna()]
    tr, va = di[di.split == "train"], di[di.split == "val"]
    if len(tr) < 200 or len(va) < 50:
        return {}, 0.0
    if len(tr) > config.max_train_rows:
        tr = tr.sample(config.max_train_rows, random_state=seed)

    hgb_p = {k: v for k, v in best_params.get(("hgb", signal), {}).items() if k != "loss"}
    qmodels = {}
    for q, name in [(.1, "lo"), (.5, "mid"), (.9, "hi")]:
        qmodels[name] = HistGradientBoostingRegressor(
            loss="quantile", quantile=q, random_state=seed,
            **{**dict(max_iter=250, learning_rate=.07), **hgb_p}
        ).fit(tr[features], tr[target].values)

    va_lo = qmodels["lo"].predict(va[features])
    va_hi = qmodels["hi"].predict(va[features])
    conformity = np.maximum(va_lo - va[target].values, va[target].values - va_hi)
    finite = conformity[np.isfinite(conformity)]
    qhat = float(np.percentile(finite, 80)) if len(finite) else 0.0
    logging.info("conformal widening qhat = %.2f mmHg", qhat)
    return qmodels, qhat


# ---------------------------------------------------------------------------- serving

class BPPredictor:
    """Frozen serving artifact. The only state is the bundle; no module globals are read."""

    def __init__(self, bundle: dict):
        # The bundle stores plain data plus sklearn estimators only. Project-defined classes
        # are reconstructed here rather than pickled, so a service process can load the
        # bundle without carrying the training package's class graph.
        self.b = bundle
        self.config = bundle["config"]
        self.fb = CausalFeatureBuilder(self.config)
        o = bundle["offset"]
        self.offset = OffsetModel(
            self.config, warm=o["warm"], k=o["k"], q=o["q"],
            cohort_prior=o["cohort_prior"], global_prior=o["global_prior"])

    # ---------- construction ----------
    @classmethod
    def build(cls, F, features, config, winner, best_params, offset_model,
              qmodels, qhat, interval_signal, interval_horizon,
              detector, imputer, dense_cols, detector_cut) -> "BPPredictor":
        """Refit the forecasters on train+val, then freeze everything into one bundle."""
        fit_mask = F.split.isin(["train", "val"])

        forecasters = {}
        for s in config.signals:
            kind = winner.get(s, "hgb")
            for h in config.horizons:
                tgt = f"y_{s}_h{h}"
                if tgt not in F.columns:
                    continue
                d = F[fit_mask & F[tgt].notna()]
                if len(d) < 200:
                    continue
                if len(d) > config.max_train_rows:
                    d = d.sample(config.max_train_rows, random_state=config.seed)
                forecasters[(s, h)] = make_model(kind, **best_params.get((kind, s), {})).fit(
                    d[features], d[tgt].values)

        bundle = dict(
            model_version=f"hemobp-bp-{config.run_id}", run_id=config.run_id,
            config=config, feature_names=features,
            selected_family=winner, forecasters=forecasters,
            interval=dict(models=qmodels, qhat=qhat, signal=interval_signal,
                          horizon=interval_horizon, fit_on="train", calibrated_on="val"),
            offset=dict(warm=offset_model.warm, k=offset_model.k, q=offset_model.q,
                        cohort_prior=offset_model.cohort_prior,
                        global_prior=offset_model.global_prior,
                        label="capped shrinkage blend"),
            detector=dict(model=detector, imputer=imputer, cols=dense_cols, cut=detector_cut,
                          budget_pct=config.alert_budget_pct, warn_window=config.warn_window,
                          event_quantile=config.event_quantile),
        )
        return cls(bundle)

    # ---------- persistence ----------
    def save(self, path: str) -> str:
        import joblib
        joblib.dump(self.b, path)
        return path

    @classmethod
    def load(cls, path: str) -> "BPPredictor":
        import joblib
        return cls(joblib.load(path))

    # ---------- inference ----------
    def _tier(self, n: int) -> str:
        c = self.config
        if n < c.cold_start_min_readings:
            return "cold_start"
        return "bootstrapping" if n < c.steady_state_readings else "steady"

    def predict(self, history: pd.DataFrame, as_of=None) -> dict:
        """One patient's raw session history -> a serialisable advisory."""
        try:
            t0 = time.perf_counter()
            c, b = self.config, self.b
            g = history.sort_values("ts").copy()
            pid = str(g.patient_id.iloc[0])
            n_obs = int(g.sbp.notna().sum())
            tier = self._tier(n_obs)
            age = float(g.age.iloc[0]) if "age" in g and pd.notna(g.age.iloc[0]) else 65.0
            male = int(g.is_male.iloc[0]) if "is_male" in g else 0

            pers = self.offset.threshold_for(g.sbp, age, male)
            out = dict(patient_id=pid, as_of=str(as_of or g.ts.max()),
                       model_version=b["model_version"], n_observations=n_obs,
                       confidence_tier=tier, personalisation=pers, forecast={},
                       early_warning=None, emergency_floor_mmHg=c.emergency_floor_mmHg)

            if tier == "cold_start":
                out["note"] = (f"fewer than {c.cold_start_min_readings} readings; cohort "
                               f"threshold only, no forecast issued")
                out["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
                return out

            row = self.fb.transform_for_inference(g)
            X = row.reindex(columns=b["feature_names"])
            med_gap = float(g.ts.diff().dt.days.median() or 2)

            for (sig, h), mdl in b["forecasters"].items():
                try:
                    p = float(mdl.predict(X)[0])
                except Exception:
                    continue
                out["forecast"].setdefault(sig, {})[f"h{h}"] = dict(
                    point=round(p, 1), steps_ahead=h, days_ahead_est=round(h * med_gap, 1))

            iv = b["interval"]
            node = out["forecast"].get(iv["signal"], {}).get(f"h{iv['horizon']}")
            if node is not None and iv["models"]:
                lo = float(iv["models"]["lo"].predict(X)[0]) - iv["qhat"]
                hi = float(iv["models"]["hi"].predict(X)[0]) + iv["qhat"]
                node.update(lo80=round(lo, 1), hi80=round(hi, 1),
                            interval_basis=f"quantile GBM fit on {iv['fit_on']}, "
                                           f"conformal on {iv['calibrated_on']}")

            det = b["detector"]
            score = float(-det["model"].score_samples(
                det["imputer"].transform(row.reindex(columns=det["cols"])))[0])
            out["early_warning"] = dict(
                score=round(score, 4), cut=round(det["cut"], 4),
                flagged=bool(score >= det["cut"]), budget_pct=det["budget_pct"],
                event_definition=(f"SBP exceeds this patient's own p"
                                  f"{int(det['event_quantile'] * 100)} within the next "
                                  f"{det['warn_window']} sessions"),
                est_lead_days=round(det["warn_window"] * med_gap, 1))

            if tier == "bootstrapping":
                out["note"] = (f"{n_obs} readings, below the {c.steady_state_readings}-reading "
                               f"steady state; cohort-weighted, render a low-confidence badge")
            out["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            return out
        except Exception as e:
            raise CustomException(e, sys)


def explain_prediction(model, X_row, feature_names, reference, top_n: int = 5) -> dict:
    """Top-N contributing features for a single prediction.

    Local perturbation attribution: how much does the prediction move when each feature is
    replaced by its TRAINING median? Model-agnostic, cheap enough to run per advisory, and
    honest about being an approximation of a Shapley value rather than one.

    `reference` must be the training-set medians. Using the row's own values as the reference
    makes every perturbation a no-op and every contribution exactly zero -- which looks like a
    working explainer right up until someone reads the output.
    """
    base = float(model.predict(X_row)[0])
    contrib = {}
    for f in feature_names:
        ref = reference.get(f, np.nan)
        pert = X_row.copy()
        pert[f] = ref if np.isfinite(ref) else 0.0
        contrib[f] = base - float(model.predict(pert)[0])
    s = pd.Series(contrib)
    s = s.reindex(s.abs().sort_values(ascending=False).index)
    return s.head(top_n).round(3).to_dict()


def champion_challenger(F, features, config, champ_kind: str, chall_kind: str,
                        signal: str = "sbp") -> pd.DataFrame:
    """Score an incumbent and a candidate on identical rows, so the delta is paired."""
    target = f"y_{signal}_h{config.horizons[0]}"
    if target not in F.columns:
        return pd.DataFrame()
    d = F[F[target].notna()]
    tr, te = d[d.split == "train"], d[d.split == "test"]
    if len(tr) < 200 or len(te) < 40:
        return pd.DataFrame()
    if len(tr) > config.max_train_rows:
        tr = tr.sample(config.max_train_rows, random_state=config.seed)
    rows = []
    for label, kind in (("champion", champ_kind), ("challenger", chall_kind)):
        m = make_model(kind).fit(tr[features], tr[target].values)
        r = evaluate(te[target].values, m.predict(te[features]), te[f"{signal}_lag1"].values,
                     model=kind, role=label, signal=signal)
        if r:
            rows.append(r)
    return pd.DataFrame(rows)
