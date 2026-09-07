"""Significant-move detection.

Attribution runs only on moves worth explaining. Trying to explain every
session would produce a story for noise, which is the failure mode of most
market commentary: something always happened yesterday, so something can
always be blamed.

A move is significant when it exceeds ``k`` trailing standard deviations. The
volatility estimate uses only sessions strictly before the move, so a large
move never inflates the threshold it is being judged against - the same
causality rule as the rest of the pipeline.

Both an absolute and a market-relative measure are kept. A stock that fell 3%
on a day the index fell 3% has a large absolute move and no relative move at
all; the first calls for a market explanation, the second for none.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-9
DEFAULT_SIGMA = 1.5
VOL_WINDOW = 60


def detect_moves(panel: pd.DataFrame, k: float = DEFAULT_SIGMA) -> pd.DataFrame:
    """Return one row per significant session, ranked by relative severity.

    ``panel`` needs symbol, date, close and the market-relative forward return
    machinery from ``cef.features.build``; only realised (backward-looking)
    returns are used here.
    """
    p = panel[["symbol", "date", "close"]].copy().sort_values(["symbol", "date"])
    g = p.groupby("symbol", observed=True)
    p["ret"] = np.log(p["close"] / g["close"].shift(1))

    # Market return for the session: the cross-sectional median, so a single
    # blown-up name cannot define "the market".
    mkt = p.groupby("date", observed=True)["ret"].transform("median")
    p["ret_mkt"] = mkt
    p["ret_rel"] = p["ret"] - mkt

    g = p.groupby("symbol", observed=True)
    # shift(1) before rolling: the threshold is built from history only.
    for col in ("ret", "ret_rel"):
        vol = g[col].transform(lambda s: s.shift(1).rolling(VOL_WINDOW).std())
        p[f"{col}_sigma"] = p[col] / (vol + EPS)

    sig = p[(p["ret_sigma"].abs() >= k) | (p["ret_rel_sigma"].abs() >= k)].copy()
    sig["severity"] = sig[["ret_sigma", "ret_rel_sigma"]].abs().max(axis=1)
    # A move the whole market shared needs a market explanation, not a
    # company-specific one; flagging it here keeps attribution honest.
    sig["market_wide"] = (sig["ret_rel_sigma"].abs() < 1.0) & (sig["ret_sigma"].abs() >= k)
    return sig.sort_values("severity", ascending=False).reset_index(drop=True)


def move_summary(moves: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    n_sessions = len(panel)
    return pd.DataFrame([{
        "sessions": n_sessions,
        "significant": len(moves),
        "pct_of_sessions": round(100 * len(moves) / max(n_sessions, 1), 2),
        "market_wide": int(moves["market_wide"].sum()),
        "idiosyncratic": int((~moves["market_wide"]).sum()),
        "median_severity": round(float(moves["severity"].median()), 2),
        "max_severity": round(float(moves["severity"].max()), 2),
    }])
