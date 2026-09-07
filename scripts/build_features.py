#!/usr/bin/env python
"""Build the modelling panel and cache it to parquet."""
from __future__ import annotations

import logging
import sys
import time

import pandas as pd

from cef.config import DATA_DIR
from cef.features.build import build_panel, feature_columns

PANEL_PATH = DATA_DIR / "panel.parquet"


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                        datefmt="%H:%M:%S")
    t0 = time.time()
    panel = build_panel()
    cols = feature_columns(panel)

    panel.to_parquet(PANEL_PATH, index=False)

    print()
    print(f"panel        {len(panel):,} rows x {len(cols)} features")
    print(f"symbols      {panel['symbol'].nunique()}")
    print(f"date range   {panel['date'].min().date()} -> {panel['date'].max().date()}")
    print(f"base rate    {panel['y_dir'].mean():.4f}  (fraction of up sessions)")
    print(f"gap up rate  {panel['y_gap'].mean():.4f}")
    print(f"written      {PANEL_PATH}  ({PANEL_PATH.stat().st_size / 1e6:.1f} MB)")
    print(f"elapsed      {time.time() - t0:.1f}s")

    nan = panel[cols].isna().mean().sort_values(ascending=False)
    bad = nan[nan > 0.02]
    print(f"\nfeatures with >2% NaN: {len(bad)}")
    if len(bad):
        print(bad.head(12).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
