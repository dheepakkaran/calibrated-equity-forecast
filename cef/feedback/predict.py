"""Writing live forecasts to durable storage.

A forecast is written once, before the session it describes, and afterwards
only an outcome row is attached. Nothing about the claim itself is ever
rewritten, which is what makes the history screen a record rather than a
narrative - including the wrong calls.

The feature snapshot is stored with every row. Without it a past forecast can
be read but not checked; with it, the exact inputs can be replayed years later
and the explanation regenerated, which is the difference between logging and
auditability.
"""
from __future__ import annotations

import json
import logging
import uuid

import numpy as np
import pandas as pd

from cef.db import connect, upsert, utcnow
from cef.evidence.aspects import bucket_shap
from cef.feedback import bandit
from cef.feedback.ensemble import Ensemble

log = logging.getLogger(__name__)

# Below this conviction the system declines to make a call. Set from the M1
# coverage curve: the edge is concentrated in the extremes, and forecasting
# every session at 51% is worth less than forecasting a third of them at 52.5%.
ABSTAIN_BELOW = 0.008          # |proba - 0.5|
MODEL_VERSION = "m3.0.0"


def next_session(panel: pd.DataFrame, as_of: pd.Timestamp) -> str:
    """The session being forecast: the next one on the exchange calendar."""
    days = pd.DatetimeIndex(sorted(panel["date"].unique()))
    later = days[days > as_of]
    if len(later):
        return later[0].strftime("%Y-%m-%d")
    return (as_of + pd.tseries.offsets.BDay(1)).strftime("%Y-%m-%d")


def existing(symbol: str, as_of_date: str, target: str) -> dict | None:
    """A forecast already written for this (symbol, session, target).

    Forecasts must be idempotent per session. Without this check every page
    view wrote a fresh row, so the history screen would show one symbol
    forecast three times for the same session - which quietly contradicts the
    claim that a forecast is written once and never rewritten. The stored row
    is authoritative; a later view returns it rather than recomputing.
    """
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM predictions WHERE symbol=? AND as_of_date=? AND target=? "
            "ORDER BY created_at LIMIT 1", (symbol, as_of_date, target)).fetchone()
        if row is None:
            return None
        cols = [c[0] for c in conn.execute("SELECT * FROM predictions LIMIT 0").description]
    return dict(zip(cols, row))


def make_prediction(panel: pd.DataFrame, symbol: str, ens: Ensemble,
                    as_of: pd.Timestamp | None = None,
                    rng: np.random.Generator | None = None) -> dict:
    sub = panel[panel["symbol"] == symbol]
    if sub.empty:
        raise ValueError(f"{symbol} not in panel")
    as_of = as_of or sub["date"].max()
    row = sub[sub["date"] == as_of]
    if row.empty:
        raise ValueError(f"no row for {symbol} at {as_of.date()}")

    prior = existing(symbol, as_of.strftime("%Y-%m-%d"), ens.target)
    if prior is not None:
        return {**prior, "bandit_mode": "replayed from the stored forecast",
                "bandit_n_pulls": None}

    probas = ens.arm_probas(row)
    regime = str(row["regime"].iloc[0])
    wsel = bandit.weights_for(regime, rng)
    proba = bandit.blend(probas, wsel["weights"])

    conviction = abs(proba - 0.5)
    acted = conviction >= ABSTAIN_BELOW
    direction = ("abstain" if not acted else
                 "outperform" if proba > 0.5 else "underperform")

    aspects = bucket_shap(ens.shap_row(row), row[ens.features_].iloc[0])
    snapshot = {f: (None if not np.isfinite(v) else round(float(v), 6))
                for f, v in row[ens.features_].iloc[0].items()}

    rec = {
        "id": uuid.uuid4().hex[:16],
        "symbol": symbol,
        "as_of_date": as_of.strftime("%Y-%m-%d"),
        "target_date": next_session(panel, as_of),
        "target": ens.target,
        "direction": direction,
        "proba": round(proba, 6),
        "confidence": round(max(proba, 1 - proba), 6),
        "conviction": round(conviction, 6),
        "acted": int(acted),
        "regime": regime,
        "vol_bucket": str(row["vol_bucket"].iloc[0]),
        "trend_bucket": str(row["trend_bucket"].iloc[0]),
        "arm_weights": json.dumps(wsel["weights"]),
        "arm_probas": json.dumps({k: round(v, 6) for k, v in probas.items()}),
        "aspects": json.dumps(aspects[:6]),
        "feature_snapshot": json.dumps(snapshot),
        "model_version": MODEL_VERSION,
        "status": "PENDING",
        "created_at": utcnow(),
    }
    upsert("predictions", pd.DataFrame([rec]), ["id"])
    return {**rec, "bandit_mode": wsel["mode"], "bandit_n_pulls": wsel["n_pulls"]}


def pending(target_date: str | None = None) -> pd.DataFrame:
    q = "SELECT * FROM predictions WHERE status='PENDING'"
    params: list = []
    if target_date:
        q += " AND target_date <= ?"
        params.append(target_date)
    with connect() as conn:
        return pd.read_sql(q, conn, params=params)
