"""SHAP-based feature selection, fitted strictly inside a training window.

The panel carries roughly 200 features whose strongest univariate correlation
with the label is about 0.03. Most of them are noise, and noise is not free:
with column subsampling, every junk feature is a chance for a split to be
spent on nothing. The proposal's own plan is to cut to 60-80, and this is where
that happens.

Selection runs on the fit slice of a fold only. Running it once on the whole
panel and reusing the result would be a subtle and very common form of
leakage - the feature *set* would then encode which columns happened to work
during the test period.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def shap_rank(model, X: pd.DataFrame, sample: int = 6000, seed: int = 0) -> pd.Series:
    """Mean |SHAP| per feature. TreeExplainer on a subsample - exact values on
    a slice beat approximate values on everything for a ranking task."""
    import shap

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(sample, len(X)), replace=False)
    Xs = X.iloc[idx]
    expl = shap.TreeExplainer(model.model_)
    vals = expl.shap_values(Xs, check_additivity=False)
    if isinstance(vals, list):          # older shap returns one array per class
        vals = vals[1]
    if vals.ndim == 3:                  # (n, features, classes)
        vals = vals[:, :, -1]
    return pd.Series(np.abs(vals).mean(axis=0), index=X.columns).sort_values(ascending=False)


def select_features(model, X: pd.DataFrame, top_k: int = 70,
                    always_keep: tuple[str, ...] = ("ret_1d",)) -> list[str]:
    rank = shap_rank(model, X)
    chosen = list(rank.head(top_k).index)
    for col in always_keep:                       # baselines depend on these
        if col in X.columns and col not in chosen:
            chosen.append(col)
    log.info("selected %d/%d features; top 5: %s",
             len(chosen), X.shape[1], ", ".join(rank.head(5).index))
    return chosen
