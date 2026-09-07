"""Macro, global-index, FX and commodity ingest.

Availability lags are *not* applied here. The raw series is stored as reported,
with the lag applied later during feature construction. Storing the raw truth
and shifting at read time keeps the database honest and makes the lag auditable
in one place (``cef.features.macro``).
"""
from __future__ import annotations

import logging

import pandas as pd
import yfinance as yf

from cef.config import HISTORY_END, HISTORY_START
from cef.db import upsert, utcnow
from cef.universe import MACRO_SERIES

log = logging.getLogger(__name__)
SOURCE = "yfinance"


def ingest_macro() -> int:
    total = 0
    for series in MACRO_SERIES:
        try:
            hist = yf.Ticker(series.yf_ticker).history(
                start=HISTORY_START.isoformat(),
                end=HISTORY_END.isoformat(),
                auto_adjust=False,
            )
        except Exception as exc:  # noqa: BLE001 - one bad series must not stop the run
            log.warning("%-10s FAILED %s", series.key, exc)
            continue
        if hist.empty:
            log.warning("%-10s empty", series.key)
            continue
        df = pd.DataFrame({
            "key": series.key,
            "date": pd.to_datetime(hist.index).strftime("%Y-%m-%d"),
            "close": hist["Close"].values,
            "source": SOURCE,
            "ingested_at": utcnow(),
        }).dropna(subset=["close"])
        n = upsert("macro", df, ["key", "date"])
        total += n
        log.info("%-10s %5d rows  %s -> %s", series.key, n,
                 df["date"].min(), df["date"].max())
    return total
