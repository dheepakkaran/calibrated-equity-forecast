#!/usr/bin/env python
"""Replay the live loop over history to exercise persistence, resolution and
the bandit end to end.

This is not a performance claim - the ensemble is refitted on an expanding
window and the numbers it produces are simply the M1 result arriving one
session at a time. What it tests is that the machinery works: forecasts get
written before their session, resolved after it, and the bandit posteriors move
in response to real outcomes rather than to a fixture.
"""
from __future__ import annotations

import argparse
import logging
import sys

import numpy as np
import pandas as pd

from cef.config import DATA_DIR
from cef.db import init_db
from cef.feedback import bandit
from cef.feedback.ensemble import Ensemble
from cef.feedback.predict import make_prediction
from cef.feedback.resolver import performance, resolve


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--from-date", default="2026-03-02")
    ap.add_argument("--symbols", nargs="*")
    ap.add_argument("--refit-every", type=int, default=10,
                    help="sessions between ensemble refits")
    ap.add_argument("--n-symbols", type=int, default=12)
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING)
    init_db()
    panel = pd.read_parquet(DATA_DIR / "panel.parquet")
    # A random sample, not the alphabetical head: sorted()[:12] is
    # Adani-and-banks heavy and would make the result a sector bet.
    all_syms = sorted(panel["symbol"].unique())
    symbols = args.symbols or list(
        np.random.default_rng(0).choice(all_syms, size=min(args.n_symbols, len(all_syms)),
                                        replace=False))
    panel_s = panel[panel["symbol"].isin(symbols)]

    days = pd.DatetimeIndex(sorted(panel["date"].unique()))
    days = days[days >= pd.Timestamp(args.from_date)]
    rng = np.random.default_rng(11)

    ens = None
    print(f"replaying {len(days)} sessions x {len(symbols)} symbols "
          f"(refit every {args.refit_every})")
    for i, day in enumerate(days):
        if ens is None or i % args.refit_every == 0:
            ens = Ensemble().fit(panel, day)
        for sym in symbols:
            if not ((panel_s["symbol"] == sym) & (panel_s["date"] == day)).any():
                continue
            try:
                make_prediction(panel_s, sym, ens, as_of=day, rng=rng)
            except Exception as exc:                       # noqa: BLE001
                print(f"  {sym} {day.date()}: {str(exc)[:70]}")
        resolve(panel, as_of=day.strftime("%Y-%m-%d"))
        if i % 20 == 0:
            print(f"  {day.date()}  ...", flush=True)

    resolve(panel)
    perf = performance()
    print()
    print("=" * 72)
    print("LIVE LOOP RESULT")
    print("=" * 72)
    for k, v in perf.items():
        if k != "by_regime":
            print(f"  {k:22s} {v}")
    print("\n  by regime:")
    for regime, m in sorted(perf["by_regime"].items(), key=lambda kv: -kv[1]["n"]):
        print(f"    {regime:26s} n={m['n']:5d}  acc={m['accuracy']:.4f}  "
              f"reward={m['mean_reward']:+.4f}")

    print()
    print("=== bandit posteriors ===")
    st = pd.DataFrame(bandit.state_table())
    if not st.empty:
        piv = st.pivot_table(index="regime", columns="arm", values="posterior_mean")
        npulls = st.groupby("regime")["n_pulls"].max()
        piv["n"] = npulls
        print(piv.round(4).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
