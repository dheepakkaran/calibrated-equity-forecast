"""Bucketing per-feature SHAP values into the seven reader-facing aspects.

A 70-feature SHAP waterfall is not an explanation for anyone who is not
already an ML practitioner. What a reader can act on is "global cues pushed
this down, fundamentals pushed back" - so feature-level attributions are
summed into a small fixed set of buckets, and the buckets are what the
interface and the narrator see.

Summing signed SHAP values within a bucket is the right operation and worth
being explicit about: two features that disagree should cancel, because the
aspect genuinely contributed little. Summing absolute values would make every
aspect look influential and would defeat the purpose.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

# Order is the display order, and also the resolution order - the first
# pattern that matches a feature name claims it.
ASPECT_RULES: list[tuple[str, str]] = [
    # Calendar first: the day-of-week feature is literally named `dow`, which
    # the Dow Jones pattern below would otherwise claim, and the expiry and
    # month-end flags start with `is_`, which Technical would claim.
    ("Event & calendar", r"^(dow$|month$|is_month_end|is_expiry|gap_days$)"),
    ("Global cues", r"^(sp500|nasdaq|dow_|vix|us10y|dxy|brent|gold|silver|copper|"
                    r"nikkei|hangseng|kospi|shanghai|usdinr|risk_on|asia_|"
                    r"india_us_spread|gold_silver)"),
    ("Driver linkage", r"^driver_"),
    ("Sector & peers", r"^(rs_sector|corr_sector|xs_sector|niftybank)"),
    ("Market & breadth", r"^(nifty50|sensex|indiavix|breadth|beta|corr_nifty|"
                         r"rs_nifty|xs_rank_rs)"),
    ("Technical", r"^(px_|sma|rsi|macd|stoch|williams|roc_|adx|choppiness|hurst|"
                  r"trend_persistence|bb_|keltner|atr|realvol|vol_of_vol|hl_range|"
                  r"ret_\d|dist_|range_pos|higher_|up_days|close_loc|gap_|"
                  r"body_pct|upper_wick|lower_wick|is_|xs_rank_mom|xs_rank_idiovol|"
                  r"idio_)"),
    ("Liquidity & flow", r"^(vol_ratio|vol_trend|obv|vwap|dollar_vol)"),
]

_COMPILED = [(name, re.compile(p)) for name, p in ASPECT_RULES]
FALLBACK = "Other"


def aspect_of(feature: str) -> str:
    for name, pattern in _COMPILED:
        if pattern.match(feature):
            return name
    return FALLBACK


# Raw SHAP contributions are log-odds and land around 0.02, which is
# meaningless to a reader and, handed to a language model, gets quoted verbatim
# as "contributing 0.02205". They are therefore also expressed as integer
# points scaled so the largest absolute contribution in a forecast is 100 -
# an ordinal presentation that carries the same ranking without implying a
# precision the number does not have.
POINTS_SCALE = 100


def bucket_shap(shap_row: pd.Series, feature_values: pd.Series | None = None,
                top_evidence: int = 2) -> list[dict]:
    """Collapse one row of SHAP values into aspect contributions.

    Returns aspects ordered by absolute contribution, each carrying the
    features that drove it so the interface can show a reason rather than a
    number alone.
    """
    df = pd.DataFrame({"feature": shap_row.index, "shap": shap_row.to_numpy(float)})
    df["aspect"] = [aspect_of(f) for f in df["feature"]]

    out = []
    total_abs = df["shap"].abs().sum() or 1.0
    for aspect, g in df.groupby("aspect", observed=True):
        contribution = float(g["shap"].sum())
        drivers = g.reindex(g["shap"].abs().sort_values(ascending=False).index)
        evidence = []
        for r in drivers.head(top_evidence).itertuples(index=False):
            item = {"feature": r.feature, "shap": round(float(r.shap), 5)}
            if feature_values is not None and r.feature in feature_values.index:
                v = feature_values[r.feature]
                if np.isfinite(v):
                    item["value"] = round(float(v), 4)
            evidence.append(item)
        out.append({
            "aspect": aspect,
            "contribution": round(contribution, 5),
            "share_of_signal": round(float(g["shap"].abs().sum() / total_abs), 3),
            "direction": "up" if contribution > 0 else "down" if contribution < 0 else "flat",
            "evidence": evidence,
        })
    out = sorted(out, key=lambda d: -abs(d["contribution"]))
    biggest = max((abs(a["contribution"]) for a in out), default=0.0) or 1.0
    for a in out:
        a["points"] = int(round(POINTS_SCALE * a["contribution"] / biggest))
    return out
