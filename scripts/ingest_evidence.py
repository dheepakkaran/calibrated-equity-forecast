#!/usr/bin/env python
"""Ingest the evidence layer: corporate filings and (forward-only) news."""
from __future__ import annotations

import argparse
import logging
import sys
import time

from cef.db import init_db, table_summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", choices=["announcements", "news"])
    ap.add_argument("--symbols", nargs="*")
    ap.add_argument("--start-year", type=int, default=2018)
    ap.add_argument("--end-year", type=int, default=2026)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                        datefmt="%H:%M:%S")
    init_db()
    t0 = time.time()

    if args.only in (None, "announcements"):
        from cef.ingest.announcements import ingest_announcements
        logging.info("=== NSE corporate filings ===")
        ingest_announcements(args.symbols, args.start_year, args.end_year)

    if args.only in (None, "news"):
        from cef.ingest.news import ingest_news
        logging.info("=== news (RSS + per-ticker) ===")
        ingest_news(args.symbols)

    logging.info("done in %.0fs", time.time() - t0)
    print()
    print(table_summary().to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
