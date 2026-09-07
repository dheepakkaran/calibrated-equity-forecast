"""NSE corporate-filings ingest.

This is the backbone of the evidence layer, and it is a better attribution
source than news scraping for the events that actually move a stock. Filings
are authoritative rather than reported, categorised by the exchange, carry a
link to the original document, and - crucially - are timestamped to the second.

That timestamp is what makes causal attribution possible. A filing released at
11:04 IST can move the session it lands in. One released at 18:20 IST cannot;
its first tradeable session is the next one. Attributing an after-hours filing
to the day it was filed would be look-ahead bias of exactly the kind the rest
of this project is built to avoid, so ``after_close`` is resolved at ingest
time and the ``date`` column already points at the first session the filing
could have acted on.

The API needs a browser User-Agent and a cookie handshake. Note this is the
opposite of some other financial APIs, which reject browser agents - so the
header choice here is deliberate, not copied boilerplate.
"""
from __future__ import annotations

import logging
import re
import time
import unicodedata
from datetime import datetime, timedelta

import pandas as pd
import requests

from cef.db import upsert, utcnow
from cef.universe import UNIVERSE

log = logging.getLogger(__name__)

SOURCE = "nse_announcements"
BASE = "https://www.nseindia.com"
API = f"{BASE}/api/corporate-announcements"

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"),
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": f"{BASE}/companies-listing/corporate-filings-announcements",
}

# NSE closes at 15:30 IST. A filing at or after that cannot affect the session
# it was filed in.
MARKET_CLOSE = (15, 30)
REQUEST_GAP = 1.5          # seconds between calls; NSE throttles aggressively
COOKIE_REFRESH = 40        # re-handshake every N requests, cookies expire


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update(HEADERS)
    s.get(BASE, timeout=30)
    s.get(f"{BASE}/companies-listing/corporate-filings-announcements", timeout=30)
    return s


# The NSE feed mixes encodings: non-breaking spaces, smart quotes and stray
# replacement characters arrive inside body text and render as mojibake
# ("\ufffdVedanta Limited\ufffdhas Submitted..."). Cleaning at ingest keeps the
# damage out of the database and out of anything shown to a reader.
_CTRL = re.compile(r"[\u0000-\u0008\u000b-\u001f\u007f\ufffd]")
_WS = re.compile(r"\s+")


def clean_text(raw: str | None) -> str:
    if not raw:
        return ""
    t = unicodedata.normalize("NFKC", raw)
    t = _CTRL.sub(" ", t)
    return _WS.sub(" ", t).strip()


def _next_session_date(ts: pd.Timestamp, trading_days: pd.DatetimeIndex) -> str:
    """First trading session on which a filing made at ``ts`` could be acted on."""
    after = (ts.hour, ts.minute) >= MARKET_CLOSE
    cutoff = ts.normalize() + (pd.Timedelta(days=1) if after else pd.Timedelta(0))
    later = trading_days[trading_days >= cutoff]
    if len(later):
        return later[0].strftime("%Y-%m-%d")
    return cutoff.strftime("%Y-%m-%d")


def _parse(rows: list[dict], trading_days: pd.DatetimeIndex) -> pd.DataFrame:
    out = []
    for a in rows:
        raw = a.get("sort_date") or a.get("an_dt")
        if not raw:
            continue
        ts = pd.to_datetime(raw, errors="coerce", dayfirst=False)
        if pd.isna(ts):
            ts = pd.to_datetime(a.get("an_dt"), format="%d-%b-%Y %H:%M:%S", errors="coerce")
        if pd.isna(ts):
            continue
        out.append({
            "seq_id": str(a.get("seq_id") or f"{a.get('symbol')}-{raw}"),
            "symbol": a.get("symbol"),
            "announced_at": ts.isoformat(),
            "date": _next_session_date(ts, trading_days),
            "category": a.get("desc"),
            "body": clean_text(a.get("attchmntText"))[:2000],
            "attachment": a.get("attchmntFile"),
            "industry": a.get("smIndustry"),
            "after_close": int((ts.hour, ts.minute) >= MARKET_CLOSE),
            "source": SOURCE,
            "ingested_at": utcnow(),
        })
    return pd.DataFrame(out)


def ingest_announcements(symbols: list[str] | None = None,
                         start_year: int = 2018, end_year: int = 2026) -> int:
    from cef.db import read_ohlcv

    symbols = symbols or UNIVERSE
    trading_days = pd.DatetimeIndex(sorted(read_ohlcv()["date"].unique()))

    sess = _session()
    calls = 0
    total = 0

    for sym in symbols:
        got = 0
        for year in range(start_year, end_year + 1):
            if calls and calls % COOKIE_REFRESH == 0:
                sess = _session()                     # cookies expire silently
            params = {"index": "equities", "symbol": sym,
                      "from_date": f"01-01-{year}", "to_date": f"31-12-{year}"}
            try:
                r = sess.get(API, params=params, timeout=40)
                calls += 1
                if r.status_code != 200 or not r.content:
                    log.warning("%s %d: HTTP %s", sym, year, r.status_code)
                    time.sleep(REQUEST_GAP * 2)
                    continue
                payload = r.json()
                rows = payload if isinstance(payload, list) else payload.get("data", [])
                df = _parse(rows, trading_days)
                if not df.empty:
                    df = df.drop_duplicates(subset=["seq_id"])
                    got += upsert("announcements", df, ["seq_id"])
            except Exception as exc:                  # noqa: BLE001
                log.warning("%s %d: %s", sym, year, str(exc)[:90])
                sess = _session()
            time.sleep(REQUEST_GAP)
        total += got
        log.info("%-12s %5d filings", sym, got)
    return total
