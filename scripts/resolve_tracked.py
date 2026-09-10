#!/usr/bin/env python
"""Resolve tracked guesses against what the market actually did.

Runs once a day, after the Indian close. Fetches the closes and grades each
pending guess into okay / okayish / not okay with its two numbers.

An earlier version also had a 09:20 IST mode that read the opening gap. It was
removed because GitHub's scheduler is best-effort and fired it 4.5 hours late,
by which point a "gap read" described nothing. This job is immune to that: the
closing price stops changing at 15:30 IST, so reading it late still reads it
correctly.

Prices come from Yahoo Finance at run time. The market-relative move is taken
against the median of the whole universe on the same day, which is the
definition the model was trained on — grading against a different benchmark
than the one it was fitted to would be meaningless.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
log = logging.getLogger("resolve")


def fetch(symbols: list[str], session: str) -> pd.DataFrame:
    import yfinance as yf

    from cef.universe import yf_symbol

    start = (pd.Timestamp(session) - pd.Timedelta(days=8)).strftime("%Y-%m-%d")
    end = (pd.Timestamp(session) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    raw = yf.download([yf_symbol(s) for s in symbols], start=start, end=end,
                      auto_adjust=False, group_by="ticker", progress=False, threads=True)
    rows = []
    for s in symbols:
        try:
            sub = raw[yf_symbol(s)] if isinstance(raw.columns, pd.MultiIndex) else raw
        except KeyError:
            continue
        sub = sub.dropna(how="all")
        if sub.empty:
            continue
        sub.index = pd.to_datetime(sub.index).tz_localize(None)
        rows.append(pd.DataFrame({
            "symbol": s, "date": sub.index.strftime("%Y-%m-%d"),
            "open": sub["Open"].values, "close": sub["Close"].values,
        }))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--session", default=None,
                    help="target session to resolve; defaults to every pending one")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from cef.tracking import load, resolve, save, summarise

    rows = load()
    # EARLY_READ is a status only historical rows carry; it is still resolvable.
    todo = [r for r in rows
            if r["status"] in ("PENDING", "EARLY_READ")
            and (args.session is None or r["target_session"] == args.session)]
    if not todo:
        log.info("nothing to resolve")
        summarise(rows)
        return 0

    sessions = sorted({r["target_session"] for r in todo})
    # Universe median, not just the tracked names: the model's benchmark is the
    # median of all 52, so that is what it has to be graded against.
    from cef.universe import UNIVERSE
    px = fetch(UNIVERSE, sessions[-1])
    if px.empty:
        log.error("no price data returned")
        return 1

    changed = 0
    for r in todo:
        session = r["target_session"]
        day = px[px["date"] == session]
        if day.empty:
            log.info("%s: session %s has not traded yet", r["symbol"], session)
            continue
        prior = px[px["date"] < session]
        if prior.empty:
            continue
        prev_session = prior["date"].max()
        prev = px[px["date"] == prev_session].set_index("symbol")["close"]

        me = day[day["symbol"] == r["symbol"]]
        if me.empty or r["symbol"] not in prev.index:
            log.info("%s: not priced for %s", r["symbol"], session)
            continue
        prev_close = float(prev[r["symbol"]])

        joined = day.set_index("symbol").join(prev.rename("prev"), how="inner")
        mkt = float(np.nanmedian(joined["close"] / joined["prev"] - 1) * 100)
        r["outcome"] = resolve(r, float(me["close"].iloc[0]), prev_close, mkt)
        r["status"] = "RESOLVED"
        o = r["outcome"]
        log.info("%-12s %-9s rel %+.2f%% (%.2f x typical swing) reward %+.3f",
                 r["symbol"], o["category"], o["relative_move_pct"],
                 o["move_vs_typical_swing"], o["reward"])
        changed += 1

    save(rows)
    s = summarise(rows)
    print()
    print(json.dumps(s, indent=2))
    log.info("updated %d row(s)", changed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
