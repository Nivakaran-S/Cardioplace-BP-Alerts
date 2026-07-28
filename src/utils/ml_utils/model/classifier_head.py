"""The hybrid ML layer: features at t -> the tier the rule engine fires at t+h.

Port of notebook section 16. The engine's output is the ground truth this layer forecasts
and the vocabulary it translates into -- never something the ML layer produces.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from src.constants.training_pipeline import SEED
from src.logging.logger import logging


def build_classifier_frame(F: pd.DataFrame, alerts: pd.DataFrame, h: int) -> pd.DataFrame:
    """Features at t -> the tier the engine fires at t+h. Causal, gate-aware."""
    a = alerts[["series_id", "step", "tier", "rule_id", "gate_reason"]].copy()
    a["step"] = a.step - h                      # shift the label back to the decision point
    a = a.rename(columns={"tier": "tier_future", "rule_id": "rule_future",
                          "gate_reason": "gate_future"})
    X = F.merge(a, on=["series_id", "step"], how="left")
    X["tier_future"] = X.tier_future.fillna("NO_ALERT")
    X["gate_future"] = X.gate_future.fillna("NONE")
    # Rows the engine could not evaluate are not clean negatives. Including them would
    # teach the model that "gated" looks like "healthy".
    X["clean_negative"] = (X.tier_future == "NO_ALERT") & (X.gate_future == "NONE")
    X["trainable"] = (X.tier_future != "NO_ALERT") | X.clean_negative
    return X


def train_tier_head(CLS: pd.DataFrame, features: list, config, seed: int = SEED):
    """8-way primary head over tiers, plus a per-tier secondary head over rules.

    Tier probability and rule probability are calibrated and thresholded independently.
    """
    tr_c = CLS[(CLS.split == "train") & CLS.trainable]
    te_c = CLS[(CLS.split == "test") & CLS.trainable]
    if len(tr_c) < 100 or tr_c.tier_future.nunique() < 2:
        logging.warning("not enough trainable rows for the tier head; skipping")
        return None, {}, pd.DataFrame(), te_c, None

    if len(tr_c) > 40_000:
        tr_c = tr_c.sample(40_000, random_state=seed)

    tier_clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=.08,
                                              random_state=seed, class_weight="balanced")
    tier_clf.fit(tr_c[features], tr_c.tier_future.values)
    classes = list(tier_clf.classes_)
    proba_te = tier_clf.predict_proba(te_c[features]) if len(te_c) else None
    logging.info("tier head trained on %d rows, %d classes: %s", len(tr_c), len(classes), classes)

    # Rule-secondary head: within each fired tier, which rule?
    rule_clf, rule_report = {}, []
    for tier in [t for t in classes if t != "NO_ALERT"]:
        sub = tr_c[(tr_c.tier_future == tier) & tr_c.rule_future.notna()]
        if sub.rule_future.nunique() < 2 or len(sub) < 100:
            rule_report.append(dict(tier=tier, n_rules=int(sub.rule_future.nunique()),
                                    status="single rule or too few rows -- head not required"))
            continue
        clf = HistGradientBoostingClassifier(max_iter=120, learning_rate=.1, random_state=seed)
        clf.fit(sub[features], sub.rule_future.values)
        rule_clf[tier] = clf
        sub_te = te_c[(te_c.tier_future == tier) & te_c.rule_future.notna()]
        acc = (float((clf.predict(sub_te[features]) == sub_te.rule_future.values).mean())
               if len(sub_te) else np.nan)
        rule_report.append(dict(tier=tier, n_rules=int(sub.rule_future.nunique()),
                                status=(f"trained; test top-1 accuracy {acc:.3f}"
                                        if np.isfinite(acc) else "trained; no test rows")))

    return tier_clf, rule_clf, pd.DataFrame(rule_report), te_c, proba_te
