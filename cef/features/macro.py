"""Market-wide macro features, with availability lags enforced.

This is the single place in the codebase where the lag rule declared in
``cef.universe.MACRO_SERIES`` is applied. Everything upstream stores raw
as-reported values; everything downstream sees only lag-corrected ones.

Order of operations matters and is deliberate:

1. Reindex each series onto the NSE trading calendar and forward-fill. A
   foreign market that was shut on an NSE trading day contributes no new
   information, so carrying its last close forward is the correct treatment -
   it yields a change of zero rather than a spurious jump.
2. Shift the *level* series by its lag. Shifting the level rather than the
   derived change makes the semantics plain: this frame holds what was
   knowable at the Indian close on date t, and nothing else.
3. Derive changes and z-scores from the already-shifted levels.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cef.universe import MACRO_SERIES

EPS = 1e-9

# Series whose *level* carries meaning and belongs in the feature set
# alongside its change (a VIX of 30 means something; an S&P level of 5000
# means nothing without context).
LEVEL_SERIES = {"vix", "indiavix", "us10y"}

# Rates and volatility indices are quoted in points; differencing them is
# correct. Everything else is a price, where log returns are correct.
DIFF_NOT_RETURN = {"us10y", "vix", "indiavix"}


def macro_features(trading_days: pd.DatetimeIndex, macro_wide: pd.DataFrame) -> pd.DataFrame:
    """Build the market-wide feature frame indexed by NSE trading day."""
    trading_days = pd.DatetimeIndex(sorted(set(trading_days)))
    f = pd.DataFrame(index=trading_days)

    known: dict[str, pd.Series] = {}
    for spec in MACRO_SERIES:
        if spec.key not in macro_wide.columns:
            continue
        # Steps 1 and 2: onto the NSE calendar, forward-filled, then lagged.
        lvl = macro_wide[spec.key].reindex(
            macro_wide.index.union(trading_days)).ffill().reindex(trading_days)
        known[spec.key] = lvl.shift(spec.lag)

    # Step 3: derive only from lag-corrected levels.
    for key, lvl in known.items():
        use_diff = key in DIFF_NOT_RETURN
        base = lvl if use_diff else np.log(lvl.where(lvl > 0))
        for n in (1, 5, 20):
            f[f"{key}_chg{n}"] = base.diff(n)
        if key in LEVEL_SERIES:
            f[f"{key}_level"] = lvl
            f[f"{key}_z60"] = (lvl - lvl.rolling(60).mean()) / (lvl.rolling(60).std() + EPS)
        f[f"{key}_vs_ma20"] = lvl / (lvl.rolling(20).mean() + EPS) - 1

    # --- composite risk appetite --------------------------------------------
    if {"sp500", "vix"} <= known.keys():
        f["risk_on_5d"] = f["sp500_chg5"] - 0.01 * f["vix_chg5"]
    if {"nifty50", "sp500"} <= known.keys():
        # Has India been leading or lagging the US over the last month?
        f["india_us_spread_20d"] = f["nifty50_chg20"] - f["sp500_chg20"]
    if {"nikkei", "hangseng", "kospi", "shanghai"} <= known.keys():
        f["asia_breadth_1d"] = pd.concat(
            [f[f"{k}_chg1"] for k in ("nikkei", "hangseng", "kospi", "shanghai")],
            axis=1).gt(0).mean(axis=1)
        f["asia_mean_1d"] = pd.concat(
            [f[f"{k}_chg1"] for k in ("nikkei", "hangseng", "kospi", "shanghai")],
            axis=1).mean(axis=1)
    if {"gold", "silver"} <= known.keys():
        f["gold_silver_ratio_chg5"] = f["gold_chg5"] - f["silver_chg5"]

    # --- calendar (no leak: known years in advance) --------------------------
    f["dow"] = trading_days.dayofweek
    f["month"] = trading_days.month
    f["is_month_end_wk"] = (trading_days.day >= 24).astype(int)
    f["is_expiry_wk"] = ((trading_days.day >= 22) & (trading_days.day <= 28) &
                         (trading_days.dayofweek == 3)).astype(int)
    # Sessions since the previous trading day: flags weekends and holiday gaps,
    # over which more foreign information accumulates.
    f["gap_days"] = pd.Series(trading_days, index=trading_days).diff().dt.days.fillna(1)

    return f.replace([np.inf, -np.inf], np.nan)
