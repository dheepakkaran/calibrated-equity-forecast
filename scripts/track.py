#!/usr/bin/env python
"""Record a guess for tomorrow. Used by the interface and by GitHub Actions."""
from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings

warnings.filterwarnings("ignore")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("symbol")
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING)
    from cef.api import service
    from cef.tracking import summarise, track

    try:
        f = service.forecast(args.symbol)
    except KeyError:
        print(f"{args.symbol.upper()} is not in the universe", file=sys.stderr)
        return 1

    row = track(f, args.note)
    s = summarise()
    print(json.dumps({"tracked": row["symbol"], "target": row["target_session"],
                      "already": row["already_tracked"],
                      "call": row["guess"]["direction"],
                      "confidence": row["guess"]["confidence"],
                      "ledger_size": s["tracked"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
