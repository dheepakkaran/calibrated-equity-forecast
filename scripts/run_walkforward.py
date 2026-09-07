#!/usr/bin/env python
"""Run walk-forward validation and print the honest report."""
from __future__ import annotations

import argparse
import logging
import sys
import time

import pandas as pd

from cef.config import DATA_DIR
from cef.db import init_db
from cef.validation.runner import run_walk_forward, summarise


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--no-calibrate", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                        datefmt="%H:%M:%S")
    init_db()
    panel = pd.read_parquet(DATA_DIR / "panel.parquet")

    t0 = time.time()
    metrics, preds = run_walk_forward(panel, args.run_id, calibrate=not args.no_calibrate)
    elapsed = time.time() - t0

    print()
    print("=" * 78)
    print("WALK-FORWARD RESULT  (mean across folds, equally weighted)")
    print("=" * 78)
    print(summarise(metrics).to_string())
    print()
    print(f"run_id {metrics['run_id'].iloc[0]}   folds {metrics['fold'].nunique()}"
          f"   elapsed {elapsed:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
