#!/usr/bin/env python
"""Run the full ingest. Idempotent - safe to re-run any time."""
from __future__ import annotations

import argparse
import logging
import sys
import time

from cef.db import init_db, table_summary
from cef.ingest.equities import ingest_equities
from cef.ingest.macro import ingest_macro


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", choices=["equities", "macro"], help="run one stage only")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                        datefmt="%H:%M:%S")
    init_db()
    t0 = time.time()

    if args.only in (None, "macro"):
        logging.info("=== macro ===")
        ingest_macro()
    if args.only in (None, "equities"):
        logging.info("=== equities ===")
        ingest_equities()

    logging.info("done in %.1fs", time.time() - t0)
    print()
    print(table_summary().to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
