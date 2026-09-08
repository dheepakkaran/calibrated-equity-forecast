"""Data assembly for the dashboard view.

The dashboard shows more panels than the guided flow does, and every one of
them has to come from something the pipeline actually computed. Where the
original design called for a panel this system cannot fill - a next-day price
range, a gap-open call, detected chart patterns, FII/DII flows, fundamentals -
the panel is reported as unbuilt rather than populated with a plausible
number. A dashboard whose whole argument is calibrated honesty cannot have
fabricated tiles on it.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from cef.db import connect, read_macro
from cef.universe import MACRO_BY_KEY, SECTOR, drivers_for

# Panels the original mockups specified that this system does not produce.
# Surfaced in the API so the interface can state the gap instead of hiding it.
NOT_BUILT = [
    {"panel": "Expected price range",
     "reason": "No quantile head. Predicting a next-day high/low band is a "
               "different model, and an interval invented from ATR would look "
               "like a forecast without being one."},
    {"panel": "Gap-open call",
     "reason": "Measured and dropped. Overnight returns are positive 60.2% of "
               "the time on this universe, so a gap model scores 63.6% against "
               "a 63.5% base rate - AUC 0.505. The number would impress and "
               "mean nothing."},
    {"panel": "Detected chart patterns",
     "reason": "Candlestick flags exist as model features, but a confidence "
               "score per named pattern was never validated, so none is shown."},
    {"panel": "FII / DII flows",
     "reason": "Not ingested. NSE publishes it daily and it is a genuine "
               "India-specific signal; it is simply not wired up yet."},
    {"panel": "Fundamentals",
     "reason": "Not ingested. Quarterly financials move on a different clock "
               "than a one-session forecast and were deferred."},
]

BOARD_KEYS = ["nifty50", "sensex", "niftybank", "indiavix", "silver", "gold",
              "copper", "brent", "usdinr", "us10y", "dxy", "sp500"]

PRETTY = {
    "nifty50": "Nifty 50", "sensex": "Sensex", "niftybank": "Bank Nifty",
    "indiavix": "India VIX", "silver": "Silver", "gold": "Gold",
    "copper": "Copper", "brent": "Brent crude", "usdinr": "USD / INR",
    "us10y": "US 10Y", "dxy": "Dollar index", "sp500": "S&P 500",
}


def board() -> list[dict]:
    """The macro strip. Each tile carries the lag it was read at, because a US
    close and an Indian close are not knowable at the same moment and the
    dashboard should not imply they are."""
    mw = read_macro()
    if mw.empty:
        return []
    out = []
    for key in BOARD_KEYS:
        if key not in mw.columns:
            continue
        s = mw[key].dropna()
        if len(s) < 6:
            continue
        spec = MACRO_BY_KEY[key]
        last, prev = float(s.iloc[-1]), float(s.iloc[-2])
        wk = float(s.iloc[-6])
        pct = (last / prev - 1) * 100 if prev else 0.0
        wk_pct = (last / wk - 1) * 100 if wk else 0.0
        out.append({
            "key": key, "label": PRETTY.get(key, key),
            # Twelve sessions, for the sparkline. Shape only - the interface
            # draws it axis-free, because 60 pixels cannot carry a level.
            "spark": [round(float(v), 4) for v in s.iloc[-12:]],
            "value": round(last, 2),
            "chg_pct": round(pct, 2),
            "chg_week_pct": round(wk_pct, 2),
            "as_of": s.index[-1].strftime("%Y-%m-%d"),
            "lag_sessions": spec.lag,
            "availability": ("known at the Indian close" if spec.lag == 0
                             else "lagged one session - lands after the Indian close"),
        })
    return out


def price_series(symbol: str, sessions: int = 260) -> dict:
    """Closes plus any attributed moves, for the annotated chart."""
    with connect() as conn:
        px = pd.read_sql(
            "SELECT date, open, high, low, close FROM ohlcv WHERE symbol=? "
            "ORDER BY date", conn, params=[symbol.upper()], parse_dates=["date"])
        att = pd.read_sql(
            "SELECT date, ret, sigma, kind, headline, source_url, confidence "
            "FROM attributions WHERE symbol=? ORDER BY date",
            conn, params=[symbol.upper()], parse_dates=["date"])
    if px.empty:
        return {"points": [], "markers": []}
    px = px.tail(sessions)
    lo, hi = float(px["low"].min()), float(px["high"].max())

    marks = []
    if not att.empty:
        window = att[att["date"] >= px["date"].min()]
        for r in window.itertuples(index=False):
            row = px[px["date"] == r.date]
            if row.empty:
                continue
            marks.append({
                "date": r.date.strftime("%Y-%m-%d"),
                "close": round(float(row["close"].iloc[0]), 2),
                "ret_pct": round(float(r.ret) * 100, 2),
                "sigma": float(r.sigma),
                "kind": r.kind,
                "headline": (r.headline or "")[:220] or None,
                "source_url": r.source_url,
                "confidence": float(r.confidence or 0.0),
            })
    return {
        "points": [{"date": d.strftime("%Y-%m-%d"), "close": round(float(c), 2)}
                   for d, c in zip(px["date"], px["close"])],
        "markers": marks[-14:],
        "low": round(lo, 2), "high": round(hi, 2),
        "sessions": len(px),
    }


def key_levels(symbol: str) -> dict:
    """Classic floor-trader pivots from the last completed session, plus the
    rolling structural extremes.

    These are arithmetic on prices that have already printed, not predictions -
    which is exactly why they are safe to show next to a 51% model. They are
    labelled as reference levels rather than targets.
    """
    with connect() as conn:
        px = pd.read_sql(
            "SELECT date, high, low, close FROM ohlcv WHERE symbol=? "
            "ORDER BY date DESC LIMIT 220", conn, params=[symbol.upper()],
            parse_dates=["date"])
    if px.empty:
        return {}
    px = px.iloc[::-1]
    h, l, c = (float(px["high"].iloc[-1]), float(px["low"].iloc[-1]),
               float(px["close"].iloc[-1]))
    pivot = (h + l + c) / 3
    rng = h - l
    return {
        "pivot": round(pivot, 2),
        "r1": round(2 * pivot - l, 2), "r2": round(pivot + rng, 2),
        "s1": round(2 * pivot - h, 2), "s2": round(pivot - rng, 2),
        "high_20": round(float(px["high"].tail(20).max()), 2),
        "low_20": round(float(px["low"].tail(20).min()), 2),
        "high_200": round(float(px["high"].tail(200).max()), 2),
        "low_200": round(float(px["low"].tail(200).min()), 2),
        "basis": f"floor-trader pivots from the {px['date'].iloc[-1].date()} session",
        "note": ("Arithmetic on prices that have already printed, not forecasts. "
                 "Shown as reference levels only."),
    }


def driver_linkage(symbol: str) -> list[dict]:
    """Rolling correlation of the symbol's returns to its mapped drivers.

    Correlations are computed against the *lagged* driver series, which is why
    they read lower than the contemporaneous figures a data terminal would
    show. The lagged number is the only one that could inform a forecast.
    """
    from cef.config import DATA_DIR

    panel = pd.read_parquet(DATA_DIR / "panel.parquet",
                            columns=["symbol", "date", "close"])
    mw = read_macro()
    sub = panel[panel["symbol"] == symbol.upper()].sort_values("date")
    if sub.empty:
        return []
    days = pd.DatetimeIndex(sub["date"])
    own = np.log(sub.set_index("date")["close"]).diff()

    out = []
    for key in drivers_for(symbol.upper()):
        if key not in mw.columns:
            continue
        spec = MACRO_BY_KEY[key]
        lvl = (mw[key].reindex(mw.index.union(days)).ffill()
               .reindex(days).shift(spec.lag))
        dret = np.log(lvl.where(lvl > 0)).diff()
        corr = own.tail(120).corr(dret.tail(120))
        last = dret.dropna()
        out.append({
            "driver": PRETTY.get(key, key),
            "corr_120d": round(float(corr), 3) if pd.notna(corr) else None,
            "last_chg_pct": round(float(last.iloc[-1]) * 100, 2) if len(last) else None,
            "lag_sessions": spec.lag,
        })
    return sorted(out, key=lambda d: -(abs(d["corr_120d"] or 0)))


def timeline(symbol: str, limit: int = 8) -> list[dict]:
    with connect() as conn:
        df = pd.read_sql(
            "SELECT date, ret, sigma, kind, headline, source_url, confidence, rationale "
            "FROM attributions WHERE symbol=? ORDER BY date DESC LIMIT ?",
            conn, params=[symbol.upper(), limit])
    if df.empty:
        return []
    return json.loads(df.to_json(orient="records"))


def source_coverage(symbol: str) -> dict:
    with connect() as conn:
        total = conn.execute(
            "SELECT COUNT(*) FROM announcements WHERE symbol=?",
            (symbol.upper(),)).fetchone()[0]
        recent = conn.execute(
            "SELECT COUNT(*) FROM announcements WHERE symbol=? AND date >= "
            "date((SELECT MAX(date) FROM announcements), '-30 day')",
            (symbol.upper(),)).fetchone()[0]
        moves = conn.execute(
            "SELECT kind, COUNT(*) FROM attributions WHERE symbol=? GROUP BY kind",
            (symbol.upper(),)).fetchall()
        actions = conn.execute(
            "SELECT COUNT(*) FROM corporate_actions WHERE symbol=?",
            (symbol.upper(),)).fetchone()[0]
    by_kind = {k: n for k, n in moves}
    n_moves = sum(by_kind.values())
    return {
        "filings_total": total,
        "filings_30d": recent,
        "corporate_actions": actions,
        "significant_moves": n_moves,
        "explained": n_moves - by_kind.get("unexplained", 0),
        "unexplained": by_kind.get("unexplained", 0),
        "explained_share": (round((n_moves - by_kind.get("unexplained", 0)) / n_moves, 3)
                            if n_moves else None),
    }
