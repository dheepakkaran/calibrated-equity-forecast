#!/usr/bin/env python
"""Detect significant moves and attach the evidence that explains them."""
from __future__ import annotations

import argparse
import logging
import sys

import numpy as np
import pandas as pd

from cef.config import DATA_DIR
from cef.db import connect, init_db, read_macro
from cef.evidence.attribution import attribute, confidence_band
from cef.evidence.moves import detect_moves, move_summary
from cef.universe import MACRO_SERIES


def driver_context(days: pd.DatetimeIndex, panel: pd.DataFrame):
    """Per-session driver moves in sigma, plus symbol-to-driver correlations.

    Driver levels are lagged exactly as in the feature pipeline, so a US
    settlement that lands after the Indian close cannot explain that close.
    """
    macro = read_macro()
    lagged = {}
    for spec in MACRO_SERIES:
        if spec.key in macro.columns:
            lagged[spec.key] = (macro[spec.key]
                                .reindex(macro.index.union(days)).ffill()
                                .reindex(days).shift(spec.lag))
    lv = pd.DataFrame(lagged, index=days)
    ret = np.log(lv.where(lv > 0)).diff()
    # shift(1) before rolling: the sigma scale uses history only.
    sigma = ret / ret.shift(1).rolling(120).std()

    corrs = {}
    px = panel.pivot_table(index="date", columns="symbol", values="close")
    sret = np.log(px).diff()
    for sym in px.columns:
        for key in ret.columns:
            c = sret[sym].rolling(250).corr(ret[key]).median()
            if np.isfinite(c):
                corrs[(sym, key)] = float(c)
    return sigma, corrs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sigma", type=float, default=2.0)
    ap.add_argument("--symbols", nargs="*")
    ap.add_argument("--lookback", type=int, default=1,
                    help="sessions of lookback for candidate filings")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                        datefmt="%H:%M:%S")
    init_db()
    panel = pd.read_parquet(DATA_DIR / "panel.parquet")
    if args.symbols:
        panel = panel[panel["symbol"].isin(args.symbols)]

    moves = detect_moves(panel, args.sigma)
    print(move_summary(moves, panel).to_string(index=False))

    with connect() as conn:
        ann = pd.read_sql(
            "SELECT seq_id, symbol, announced_at, date, category, body, "
            "attachment, after_close FROM announcements", conn)
    logging.info("announcements available: %d rows, %d symbols",
                 len(ann), ann["symbol"].nunique())

    days = pd.DatetimeIndex(sorted(panel["date"].unique()))
    sigma, corrs = driver_context(days, panel)
    logging.info("driver context: %d series, %d symbol-driver correlations",
                 sigma.shape[1], len(corrs))

    att = attribute(moves, ann, sigma, corrs,
                    lookback=args.lookback, sessions=days)
    att["band"] = att["confidence"].map(confidence_band)

    print()
    print("=" * 78)
    print(f"ATTRIBUTION  |  {len(att):,} moves at {args.sigma} sigma, lookback {args.lookback} session(s)")
    print("=" * 78)
    breakdown = (att.groupby(["kind", "band"]).size().rename("n").reset_index()
                 .sort_values("n", ascending=False))
    print(breakdown.to_string(index=False))
    print()
    print(f"explained          {(att['kind'] != 'unexplained').mean():.1%}")
    print(f"  by filing        {(att['kind'] == 'announcement').mean():.1%}")
    print(f"  by driver        {(att['kind'] == 'driver').mean():.1%}")
    print(f"unexplained        {(att['kind'] == 'unexplained').mean():.1%}")
    print(f"mean confidence    {att.loc[att['confidence'] > 0, 'confidence'].mean():.3f}")

    att.to_csv(DATA_DIR / "attributions.csv", index=False)
    print(f"\nwritten -> {DATA_DIR / 'attributions.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
