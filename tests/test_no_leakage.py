"""Leak-detection suite. Non-negotiable.

A directional equity model reporting anything much above the base rate is far
more likely to be leaking than to be working, so these tests exist to make the
modest numbers this project reports believable. The shuffled-label test is the
single most valuable one: if a model trained on destroyed labels still scores
above chance, information is reaching it through a channel other than the
features, and every other number in the project is void.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cef.config import DATA_DIR, WALK_FORWARD
from cef.features.build import feature_columns
from cef.features.macro import macro_features
from cef.features.technical import technical_features
from cef.universe import MACRO_BY_KEY
from cef.validation.walk_forward import generate_folds, split_fold, three_way_split

ROOT = Path(__file__).resolve().parent.parent
PANEL = DATA_DIR / "panel.parquet"


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    if not PANEL.exists():
        pytest.skip("panel.parquet missing - run scripts/build_features.py")
    return pd.read_parquet(PANEL)


# --------------------------------------------------------------------------
# 1. Features must not move when future data is appended.
# --------------------------------------------------------------------------
def test_features_use_no_future_data(panel):
    """The feature vector for date D must be identical whether it is computed
    on D or recomputed once six more months of data exist."""
    from cef.db import read_ohlcv

    df = read_ohlcv(["RELIANCE"]).set_index("date").sort_index()
    cutoff = df.index[-130]

    full = technical_features(df)
    truncated = technical_features(df.loc[:cutoff])

    common = truncated.index[-60:]
    a = full.loc[common].astype(float)
    b = truncated.loc[common].astype(float)
    delta = (a - b).abs().max().max()
    assert delta < 1e-9, f"features shifted by {delta} when future bars were added"


# --------------------------------------------------------------------------
# 2. Transforms are fitted inside the fold, never across it.
# --------------------------------------------------------------------------
def test_scaler_statistics_differ_across_folds(panel):
    """A scaler fitted per fold must produce different statistics per fold. If
    they match, it was fitted once on the pooled panel."""
    from cef.models.direction import LogisticDirection

    feats = feature_columns(panel)
    p = panel[panel["y_rel"].notna()]
    folds = generate_folds(p["date"], WALK_FORWARD)[:3]

    means = []
    for f in folds:
        train, _ = split_fold(p, f)
        fit, _, _ = three_way_split(train)
        mdl = LogisticDirection().fit(fit[feats], fit["y_rel"])
        means.append(mdl.pipe["scale"].mean_)

    for i in range(len(means) - 1):
        assert not np.allclose(means[i], means[i + 1]), \
            "scaler statistics identical across folds - fitted on pooled data"


# --------------------------------------------------------------------------
# 3. No feature may be a restatement of the label.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("target", ["y_dir", "y_rel", "fwd_ret", "fwd_ret_rel"])
def test_no_feature_correlates_with_target(panel, target):
    feats = feature_columns(panel)
    sub = panel[panel[target].notna()]
    corr = sub[feats].corrwith(sub[target]).abs().dropna()
    worst = corr.sort_values(ascending=False).head(3)
    assert corr.max() < 0.95, f"near-perfect predictor of {target}:\n{worst}"
    # A directional label is close to unpredictable; anything above 0.25 is a
    # leak long before it is a discovery.
    if target.startswith("y_"):
        assert corr.max() < 0.25, f"implausibly strong feature for {target}:\n{worst}"


# --------------------------------------------------------------------------
# 4. The shuffled-label test. The important one.
# --------------------------------------------------------------------------
def test_shuffled_labels_collapse_to_chance(panel):
    """Destroy the label, keep everything else, and the model must learn
    nothing. Any real out-of-sample skill here means information is arriving
    through a path other than the features."""
    from cef.models.direction import LightGBMDirection
    from sklearn.metrics import roc_auc_score

    feats = feature_columns(panel)
    p = panel[panel["y_rel"].notna()].copy()
    f0 = generate_folds(p["date"], WALK_FORWARD)[3]
    train, test = split_fold(p, f0)
    fit, val, _ = three_way_split(train)

    rng = np.random.default_rng(7)
    y_fit = pd.Series(rng.permutation(fit["y_rel"].to_numpy()), index=fit.index)
    y_val = pd.Series(rng.permutation(val["y_rel"].to_numpy()), index=val.index)

    mdl = LightGBMDirection().fit(fit[feats], y_fit, val[feats], y_val)
    auc = roc_auc_score(test["y_rel"], mdl.predict_proba(test[feats]))
    assert 0.45 < auc < 0.55, f"shuffled-label AUC {auc:.4f} is not chance - leakage"


# --------------------------------------------------------------------------
# 5. Macro availability lags are actually applied.
# --------------------------------------------------------------------------
def test_macro_lag_enforced():
    """A lag-1 series must expose its date t-1 value at date t, and a lag-0
    series its date-t value. This is what stops a US close that lands at
    01:30 IST from being used to forecast the session it precedes."""
    from cef.db import read_macro, read_ohlcv

    days = pd.DatetimeIndex(sorted(read_ohlcv(["RELIANCE"])["date"].unique()))
    mw = read_macro()
    feat = macro_features(days, mw)

    for key, expected_lag in (("sp500", 1), ("nifty50", 0)):
        assert MACRO_BY_KEY[key].lag == expected_lag
        raw = mw[key].reindex(mw.index.union(days)).ffill().reindex(days)
        want = np.log(raw).diff().shift(expected_lag)
        got = feat[f"{key}_chg1"]
        ok = pd.concat([want, got], axis=1).dropna()
        assert np.allclose(ok.iloc[:, 0], ok.iloc[:, 1], atol=1e-9), \
            f"{key} does not respect its declared lag of {expected_lag}"


# --------------------------------------------------------------------------
# 6. Every prediction is dated before the session it forecasts.
# --------------------------------------------------------------------------
def test_target_date_strictly_after_feature_date(panel):
    bad = panel[panel["target_date"] <= panel["date"]]
    assert bad.empty, f"{len(bad)} rows forecast a session at or before their own"


# --------------------------------------------------------------------------
# 7. The embargo must cover the label's forward span.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("horizon", [1, 5, 20])
def test_embargo_covers_label_span(panel, horizon):
    p = panel[panel["y_rel"].notna()]
    for f in generate_folds(p["date"], WALK_FORWARD, horizon):
        gap = (f.test_start - f.train_end).days
        assert gap > horizon, \
            f"h={horizon}: embargo of {gap} days cannot cover a {horizon}-session label"


# --------------------------------------------------------------------------
# 8. Source-level guard: one forward shift, in one place.
# --------------------------------------------------------------------------
def test_only_the_target_module_shifts_forward():
    """Scan the feature code for negative shifts. Trailing windows are safe by
    construction; a ``shift(-n)`` is the one way a feature can reach forward,
    so it is confined to the target functions in build.py and asserted here."""
    pattern = re.compile(r"shift\(\s*-")
    offenders = []
    for path in (ROOT / "cef").rglob("*.py"):
        text = path.read_text()
        for i, line in enumerate(text.splitlines(), 1):
            if pattern.search(line) and not line.strip().startswith("#"):
                offenders.append(f"{path.relative_to(ROOT)}:{i}: {line.strip()}")

    allowed_file = "cef/features/build.py"
    unexpected = [o for o in offenders if not o.startswith(allowed_file)]
    assert not unexpected, "forward shift outside the target module:\n" + "\n".join(unexpected)


# --------------------------------------------------------------------------
# 9. Cross-sectional statistics must use one date at a time.
# --------------------------------------------------------------------------
def test_cross_sectional_ranks_are_within_date(panel):
    """A within-date percentile rank has a near-uniform distribution on every
    date. If ranks had been computed across the pooled panel, the per-date
    spread would collapse."""
    for col in ("xs_rank_mom20", "xs_rank_rs20"):
        per_date = panel.groupby("date")[col].agg(["min", "max", "count"])
        busy = per_date[per_date["count"] >= 40]
        assert (busy["min"] < 0.10).mean() > 0.95, f"{col} not ranked within date"
        assert (busy["max"] > 0.90).mean() > 0.95, f"{col} not ranked within date"


# --------------------------------------------------------------------------
# 10. The relative label must be centred by construction.
# --------------------------------------------------------------------------
def test_relative_target_is_balanced(panel):
    rate = panel["y_rel"].mean()
    assert 0.48 < rate < 0.52, f"market-relative base rate {rate:.4f} is not ~0.50"
