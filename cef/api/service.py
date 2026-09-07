"""Service layer: everything the HTTP handlers need, and no HTTP in it.

The ensemble takes several seconds to fit, which is far too slow to do inside a
request, so it is fitted once per as-of date and held in a module-level cache.
Forecasts are also cached per (symbol, session) because they are deterministic
given the panel - the same inputs cannot produce a different answer later, and
a reader refreshing the page should not see the number move.
"""
from __future__ import annotations

import json
import logging
import threading

import numpy as np
import pandas as pd

from cef.config import DATA_DIR
from cef.db import connect
from cef.evidence.attribution import confidence_band
from cef.feedback import bandit
from cef.feedback.ensemble import Ensemble
from cef.feedback.predict import ABSTAIN_BELOW, make_prediction
from cef.feedback.resolver import performance
from cef.models.calibration import reliability_curve
from cef.universe import SECTOR, drivers_for

log = logging.getLogger(__name__)

_lock = threading.Lock()
_panel: pd.DataFrame | None = None
_ensembles: dict[str, Ensemble] = {}
_forecasts: dict[tuple[str, str], dict] = {}

# Held-out figures from scripts/report.py. Shown beside every live number so a
# reader always sees the track record next to the claim.
HELD_OUT = {
    "accuracy": 0.5102, "coin_flip_baseline": 0.5001, "edge_pp": 1.01,
    "fold_t_stat": 3.01, "folds_won": "5 of 5", "auc": 0.515,
    "ece_pooled_pp": 0.69, "window": "2024-03 to 2026-08",
    "accuracy_at_10pct_coverage": 0.5362,
}


def panel() -> pd.DataFrame:
    global _panel
    if _panel is None:
        with _lock:
            if _panel is None:
                _panel = pd.read_parquet(DATA_DIR / "panel.parquet")
                log.info("panel loaded: %d rows", len(_panel))
    return _panel


def ensemble_for(as_of: pd.Timestamp) -> Ensemble:
    key = as_of.strftime("%Y-%m-%d")
    if key not in _ensembles:
        with _lock:
            if key not in _ensembles:
                _ensembles[key] = Ensemble().fit(panel(), as_of)
    return _ensembles[key]


def search_symbols(q: str, limit: int = 8) -> list[dict]:
    p = panel()
    syms = sorted(p["symbol"].unique())
    ql = (q or "").strip().upper()
    hits = [s for s in syms if s.startswith(ql)] + \
           [s for s in syms if ql in s and not s.startswith(ql)]
    seen, out = set(), []
    for s in hits or syms:
        if s in seen:
            continue
        seen.add(s)
        out.append({"symbol": s, "sector": SECTOR.get(s, "-")})
        if len(out) >= limit:
            break
    return out


def latest_session() -> pd.Timestamp:
    return panel()["date"].max()


def forecast(symbol: str, persist: bool = True) -> dict:
    symbol = symbol.upper()
    p = panel()
    if symbol not in set(p["symbol"]):
        raise KeyError(symbol)
    as_of = p[p["symbol"] == symbol]["date"].max()
    key = (symbol, as_of.strftime("%Y-%m-%d"))
    if key in _forecasts:
        return _forecasts[key]

    ens = ensemble_for(as_of)
    rec = make_prediction(p, symbol, ens, as_of=as_of)
    row = p[(p["symbol"] == symbol) & (p["date"] == as_of)].iloc[0]

    proba = rec["proba"]
    close = float(row["close"])
    aspects = json.loads(rec["aspects"])
    arm_probas = json.loads(rec["arm_probas"])
    weights = json.loads(rec["arm_weights"])

    out = {
        "symbol": symbol,
        "sector": SECTOR.get(symbol, "-"),
        "as_of": rec["as_of_date"],
        "target_date": rec["target_date"],
        "prediction_id": rec["id"],
        "question": ("Will this share out-perform or under-perform the median "
                     "NSE large-cap in the next session?"),
        "direction": rec["direction"],
        "proba_outperform": round(proba, 4),
        "confidence": round(rec["confidence"], 4),
        "conviction": round(rec["conviction"], 5),
        "acted": bool(rec["acted"]),
        "abstain_threshold": ABSTAIN_BELOW,
        "coin_flip_zone": rec["conviction"] < ABSTAIN_BELOW,
        "last_close": round(close, 2),
        "typical_daily_range": round(float(row["atr_pct"]) * close, 2),
        "regime": {"label": rec["regime"], "volatility": rec["vol_bucket"],
                   "trend": rec["trend_bucket"]},
        "aspects": aspects,
        "arms": {"probas": arm_probas, "weights": weights,
                 "mode": rec["bandit_mode"], "n_pulls": rec["bandit_n_pulls"]},
        "drivers": drivers_for(symbol),
        "track_record": HELD_OUT,
        "model_version": rec["model_version"],
        "disclaimer": ("Educational and research use only. Not investment advice. "
                       "Next-session equity direction is close to unpredictable; "
                       "this system reports its own accuracy, including the "
                       "forecasts it got wrong."),
    }
    if persist:
        _forecasts[key] = out
    return out


def attribution_for(symbol: str, limit: int = 8) -> list[dict]:
    with connect() as conn:
        df = pd.read_sql(
            "SELECT date, ret, ret_rel, sigma, kind, headline, source_url, "
            "confidence, rationale, market_wide FROM attributions "
            "WHERE symbol=? ORDER BY date DESC LIMIT ?", conn, params=[symbol.upper(), limit])
    if df.empty:
        return []
    df["band"] = df["confidence"].map(confidence_band)
    return json.loads(df.to_json(orient="records"))


def history(symbol: str | None = None, limit: int = 60) -> list[dict]:
    q = ("SELECT p.id, p.symbol, p.as_of_date, p.target_date, p.direction, p.proba, "
         "p.confidence, p.acted, p.regime, p.status, o.actual_ret_rel, o.actual_dir, "
         "o.correct, o.reward FROM predictions p "
         "LEFT JOIN outcomes o ON o.prediction_id = p.id")
    params: list = []
    if symbol:
        q += " WHERE p.symbol = ?"
        params.append(symbol.upper())
    q += " ORDER BY p.target_date DESC, p.symbol LIMIT ?"
    params.append(limit)
    with connect() as conn:
        df = pd.read_sql(q, conn, params=params)
    return json.loads(df.to_json(orient="records")) if not df.empty else []


def live_performance() -> dict:
    perf = performance()
    perf["held_out"] = HELD_OUT

    with connect() as conn:
        df = pd.read_sql(
            "SELECT p.proba, o.correct FROM outcomes o "
            "JOIN predictions p ON p.id = o.prediction_id WHERE p.acted = 1", conn)
    if len(df) >= 50:
        # The observed rate is P(out-performs), so a call of "under-perform"
        # at 0.45 has to be flipped to sit on the same axis as its own claim.
        proba = df["proba"].to_numpy(float)
        y = np.where(proba > 0.5, df["correct"].to_numpy(float),
                     1 - df["correct"].to_numpy(float))
        perf["reliability"] = reliability_curve(proba, y, n_bins=8)
    else:
        perf["reliability"] = []
        perf["reliability_note"] = (
            f"needs 50 resolved forecasts, have {len(df)}")
    return perf


def bandit_state() -> dict:
    rows = bandit.state_table()
    with connect() as conn:
        events = conn.execute(
            "SELECT at, kind, regime, detail FROM guardrail_log "
            "ORDER BY id DESC LIMIT 10").fetchall()
    return {
        "arms": list(bandit.ARMS),
        "state": rows,
        "guardrails": {
            "min_pulls_per_regime": bandit.MIN_PULLS_PER_REGIME,
            "weight_floor": bandit.WEIGHT_FLOOR,
            "weight_ceiling": bandit.WEIGHT_CEILING,
            "drift_window": bandit.DRIFT_WINDOW,
            "drift_threshold": bandit.DRIFT_THRESHOLD,
        },
        "recent_guardrail_events": [
            {"at": e[0], "kind": e[1], "regime": e[2],
             "detail": json.loads(e[3]) if e[3] else None} for e in events],
    }


def _attribution_summary(symbol: str, limit: int = 3) -> dict:
    moves = attribution_for(symbol, limit)
    explained = [m for m in moves if m["kind"] != "unexplained"]
    return {
        "recent_large_moves": [
            {"date": m["date"],
             "move_pct": round(m["ret"] * 100, 2),
             "cause_found": m["kind"] != "unexplained",
             "cause": (m["headline"] or "")[:200] if m["kind"] != "unexplained" else None,
             "evidence_strength": m["band"]}
            for m in moves],
        "how_many_had_an_identifiable_cause": f"{len(explained)} of {len(moves)}",
        "attribution_kind": ("mixed" if explained and len(explained) < len(moves)
                             else "all explained" if explained else "unexplained"),
    }


def narration_for(symbol: str) -> dict:
    from cef.evidence.narrate import build_evidence, narrate

    f = forecast(symbol)
    ev = build_evidence(
        symbol=f["symbol"], forecast_date=f["target_date"],
        direction=f["direction"], confidence=f["confidence"],
        aspects=[{"aspect": a["aspect"], "points": a["points"],
                  "direction": a["direction"]} for a in f["aspects"] if a["points"]],
        # Field names reach the reader if they are shaped like prose: an
        # earlier version passed {"kind": "recent"} and the narrator dutifully
        # wrote "the attribution is recent". The summary is phrased instead.
        attribution=_attribution_summary(symbol),
        levels={"last_close": f["last_close"],
                "typical_daily_range": f["typical_daily_range"]})
    ev["question"] = f["question"]
    ev["model_track_record"] = f["track_record"]
    return narrate(ev)
