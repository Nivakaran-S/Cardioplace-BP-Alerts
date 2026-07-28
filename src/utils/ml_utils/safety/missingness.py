"""Does the shipped forecaster survive users skipping readings?

Port of notebook Part 26. The pipeline never imputes a missed session -- the row simply
does not exist -- which is the right choice, but it leaves a question unanswered: how much
does the forecast degrade as sessions disappear? Nothing in the production pipeline measured
that, so "it handles missing readings" was an architectural claim with no evidence behind it.

Deletion, not imputation, is the honest experiment: real sessions are removed and the panel
is rebuilt exactly as it would be for a patient who logged that sparsely.

Three deliberate departures from the notebook:

1. The masked panel goes through `attach_cadence`, so its cadence feature is on the same
   scale the model trained on. The notebook re-clips at 60 days instead of 30, which means
   part of the degradation it measures is a train/serve skew the experiment introduced.
2. It scores what SHIPS. The notebook fits the learned winner; since the ship decision
   became binding, that would measure the robustness of a model no request will ever see.
3. It reports how many patients the deletion pushes below the cold-start floor -- a patient
   who stops getting a forecast at all is a worse outcome than one whose forecast worsens,
   and averaging MAE over survivors hides it completely.
"""

import numpy as np
import pandas as pd

from src.constants.training_pipeline import SEED
from src.logging.logger import logging
from src.utils.ml_utils.feature.cadence import attach_cadence
from src.utils.ml_utils.feature.causal_features import CausalFeatureBuilder
from src.utils.ml_utils.metric.regression_metric import BASELINE_SPECS
from src.utils.ml_utils.model.estimator import BaselineForecaster, make_model

DEFAULT_GRID = ((0.0, "none"), (0.15, "uniform"), (0.35, "uniform"), (0.35, "runs"))


def mask_panel(panel: pd.DataFrame, rate: float, mode: str = "uniform",
               seed: int = SEED, run_len: int = 4, min_rows: int = 12) -> pd.DataFrame:
    """Delete real sessions. Nothing is imputed; the rows simply cease to exist.

    `uniform` drops sessions independently -- a user who forgets at random. `runs` drops
    contiguous blocks -- a holiday, an illness, a hospital stay -- which is the harder and
    more realistic pattern, because it removes the recent history every lag depends on.
    """
    if rate <= 0:
        # Still rebuilt through attach_cadence rather than returned as-is, so the unmasked
        # control is constructed exactly like every masked regime. A baseline built by a
        # different path is not a baseline.
        out = panel.sort_values(["series_id", "ts"]).copy()
        out = attach_cadence(out, by="series_id")
        out["step"] = out.groupby("series_id").cumcount()
        return out.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    keep_parts = []
    for _, g in panel.groupby("series_id", sort=False):
        g = g.sort_values("step")
        n = len(g)
        if mode == "runs":
            drop = np.zeros(n, dtype=bool)
            target = int(round(rate * n))
            while drop.sum() < target and n > run_len:
                s = int(rng.integers(0, max(n - run_len, 1)))
                drop[s:s + run_len] = True
            m = ~drop
        else:
            m = rng.random(n) >= rate
        if m.sum() >= min_rows:
            keep_parts.append(g[m])
    if not keep_parts:
        return panel.iloc[0:0].copy()

    out = pd.concat(keep_parts, ignore_index=True).sort_values(["series_id", "ts"])
    out = attach_cadence(out, by="series_id")          # production bounds, not the notebook's
    out["step"] = out.groupby("series_id").cumcount()
    return out.reset_index(drop=True)


def missingness_sweep(panel: pd.DataFrame, features: list, config, shipped: dict,
                      best_params: dict = None, grid=None, max_patients: int = 60,
                      seed: int = SEED) -> pd.DataFrame:
    """Degradation of the SHIPPED sbp forecaster as sessions are deleted."""
    grid = grid or DEFAULT_GRID
    best_params = best_params or {}
    target = f"y_sbp_h{config.horizons[0]}"

    ids = sorted(panel.series_id.unique())
    if max_patients and len(ids) > max_patients:
        # The question is whether degradation happens, not its third decimal place.
        ids = ids[:max_patients]
        logging.info("missingness sweep capped to %d patients of %d", len(ids),
                     len(sorted(panel.series_id.unique())))
    sub = panel[panel.series_id.isin(ids)]

    family, name = shipped.get("sbp", ("learned", "hgb"))
    rows = []
    for rate, mode in grid:
        masked = mask_panel(sub, rate, mode, seed=seed)
        if masked.empty:
            continue
        F = CausalFeatureBuilder(config).transform(masked)
        d = F[F[target].notna()]
        tr, te = d[d.split == "train"], d[d.split == "test"]
        if len(tr) < 200 or len(te) < 40:
            logging.warning("missingness %s@%.0f%%: too little data left to score",
                            mode, rate * 100)
            continue

        if family == "baseline":
            model = BaselineForecaster(name, "sbp", config.horizons[0])
        else:
            fit = tr.sample(min(config.max_train_rows, len(tr)), random_state=seed)
            model = make_model(name, **best_params.get((name, "sbp"), {})).fit(
                fit[features], fit[target].values)

        pred = np.asarray(model.predict(te[features]), float)
        y = te[target].to_numpy(float)
        pers = BASELINE_SPECS["persistence"](te, "sbp", config.horizons[0])
        ok = np.isfinite(pred) & np.isfinite(y)
        ok_p = np.isfinite(pers) & np.isfinite(y)
        mae = float(np.abs(pred[ok] - y[ok]).mean()) if ok.any() else np.nan
        pmae = float(np.abs(pers[ok_p] - y[ok_p]).mean()) if ok_p.any() else np.nan

        raw = masked.days_since_last_raw.dropna()
        n_short = int((masked.groupby("series_id").size()
                       < config.cold_start_min_readings).sum())
        rows.append(dict(
            rate=rate, mode=mode, model=f"{family}:{name}",
            n_patients=int(masked.series_id.nunique()), n_rows=int(len(masked)),
            n_scored=int(ok.sum()), mae=round(mae, 3), persistence_mae=round(pmae, 3),
            lift_over_persistence=round(pmae - mae, 3),
            median_gap_days=round(float(raw.median()), 2) if len(raw) else np.nan,
            p95_gap_days=round(float(raw.quantile(0.95)), 2) if len(raw) else np.nan,
            pct_gap_over_clip=round(float((raw > 30).mean()), 5) if len(raw) else np.nan,
            n_patients_below_cold_start=n_short))
    return pd.DataFrame(rows)


def missingness_gate(sweep: pd.DataFrame, min_lift: float = 0.0):
    """Does the shipped forecaster still beat persistence once sessions go missing?

    Persistence is the right yardstick: it needs no history beyond the last reading, so it
    degrades most gracefully under deletion. A model that falls below it under realistic
    missingness is worse than doing nothing, whatever its clean-data score.
    """
    if sweep is None or sweep.empty:
        return True, "sweep produced no scorable regime"
    worst = sweep.loc[sweep.lift_over_persistence.idxmin()]
    ok = bool(worst.lift_over_persistence > min_lift)
    return ok, (f"worst regime {worst['mode']}@{worst.rate:.0%}: MAE {worst.mae:.2f} vs "
                f"persistence {worst.persistence_mae:.2f} "
                f"(lift {worst.lift_over_persistence:+.2f} mmHg); "
                f"{int(worst.n_patients_below_cold_start)} patients fell below the "
                f"cold-start floor")
