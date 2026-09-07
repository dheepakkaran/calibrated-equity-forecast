"""OHLCV ingest for the NSE universe via Yahoo Finance."""
from __future__ import annotations

import logging

import pandas as pd
import yfinance as yf

from cef.config import HISTORY_END, HISTORY_START
from cef.db import upsert, utcnow
from cef.universe import UNIVERSE, yf_symbol

log = logging.getLogger(__name__)
SOURCE = "yfinance"
CHUNK = 12


def _tidy(raw: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    """Flatten yfinance's MultiIndex frame into long rows."""
    frames = []
    for sym in symbols:
        tick = yf_symbol(sym)
        try:
            sub = raw[tick] if isinstance(raw.columns, pd.MultiIndex) else raw
        except KeyError:
            log.warning("no data returned for %s", sym)
            continue
        sub = sub.dropna(how="all")
        if sub.empty:
            continue
        out = pd.DataFrame({
            "symbol": sym,
            "date": pd.to_datetime(sub.index).strftime("%Y-%m-%d"),
            "open": sub["Open"].values,
            "high": sub["High"].values,
            "low": sub["Low"].values,
            "close": sub["Close"].values,
            "adj_close": sub.get("Adj Close", sub["Close"]).values,
            "volume": sub["Volume"].values,
        })
        frames.append(out)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    # A zero-volume, zero-range bar is an exchange holiday leaking through.
    df = df[~(df["volume"].fillna(0).eq(0) & df["high"].eq(df["low"]))]
    df = df.dropna(subset=["close"])
    df["source"] = SOURCE
    df["ingested_at"] = utcnow()
    return df


def ingest_equities(symbols: list[str] | None = None) -> int:
    symbols = symbols or UNIVERSE
    total = 0
    for i in range(0, len(symbols), CHUNK):
        batch = symbols[i:i + CHUNK]
        raw = yf.download(
            [yf_symbol(s) for s in batch],
            start=HISTORY_START.isoformat(),
            end=HISTORY_END.isoformat(),
            auto_adjust=False,
            group_by="ticker",
            progress=False,
            threads=True,
        )
        df = _tidy(raw, batch)
        n = upsert("ohlcv", df, ["symbol", "date"])
        total += n
        log.info("ingested %5d rows for %s", n, ", ".join(batch))
    return total
