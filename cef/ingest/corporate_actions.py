"""Corporate-action ex-dates from the NSE.

Yahoo's ``Adj Close`` corrects for splits and dividends but not for
demergers. VEDL's 2026-04-30 demerger appears as a -65% single-session return,
byte-identical in ``close`` and ``adj_close``, because value left the entity
rather than the share being subdivided. Some bonus issues also arrive
unadjusted: TRENT's 2026-01-01 session shows a price ratio of 0.670, which is
2/3 to within half a percent - a 1:2 bonus, not a 33% fall.

Left alone these rows are not merely noisy, they are wrong in a specific and
damaging way: the label says "down 65%" on a day no holder lost anything, and
the trailing 60-session volatility that every threshold in this project is
scaled by stays inflated for three months afterwards.

So ex-dates are ingested from the exchange and used to confirm breaks detected
in the price series. Confirmation matters in both directions - the point is to
mask capital-structure changes while leaving genuine crashes alone. The COVID
bottom on 2020-03-23 and the Adani selloff of February 2023 must survive.
"""
from __future__ import annotations

import logging
import re
import time

import pandas as pd
import requests

from cef.db import upsert, utcnow
from cef.universe import UNIVERSE

log = logging.getLogger(__name__)

SOURCE = "nse_corporate_actions"
BASE = "https://www.nseindia.com"
API = f"{BASE}/api/corporates-corporateActions"
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"),
    "Accept": "*/*", "Accept-Language": "en-US,en;q=0.9",
    "Referer": f"{BASE}/companies-listing/corporate-filings-actions",
}
REQUEST_GAP = 1.6
COOKIE_REFRESH = 40

# Actions that change the share count or the assets behind it, and therefore
# break the price series.
_STRUCTURAL = re.compile(
    r"demerger|de-merger|spin[- ]?off|split|sub[- ]?division|bonus|"
    r"consolidation of share|scheme of arrangement|amalgamat|reduction of capital", re.I)
_DIVIDEND = re.compile(r"dividend", re.I)
_MEETING = re.compile(r"meeting|agm|egm|postal ballot", re.I)


def classify_action(subject: str | None) -> str:
    t = subject or ""
    if _STRUCTURAL.search(t):
        return "structural"
    if _DIVIDEND.search(t):
        return "dividend"
    if _MEETING.search(t):
        return "meeting"
    return "other"


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update(HEADERS)
    s.get(BASE, timeout=30)
    s.get(f"{BASE}/companies-listing/corporate-filings-actions", timeout=30)
    return s


def ingest_corporate_actions(symbols: list[str] | None = None,
                             start_year: int = 2018, end_year: int = 2026) -> int:
    symbols = symbols or UNIVERSE
    sess = _session()
    calls = total = 0

    for sym in symbols:
        rows_out = []
        for year in range(start_year, end_year + 1):
            if calls and calls % COOKIE_REFRESH == 0:
                sess = _session()
            try:
                r = sess.get(API, params={"index": "equities", "symbol": sym,
                                          "from_date": f"01-01-{year}",
                                          "to_date": f"31-12-{year}"}, timeout=40)
                calls += 1
                if r.status_code != 200 or not r.content:
                    time.sleep(REQUEST_GAP * 2)
                    continue
                payload = r.json()
                rows = payload if isinstance(payload, list) else payload.get("data", [])
                for a in rows:
                    ex = pd.to_datetime(a.get("exDate"), errors="coerce", dayfirst=True)
                    if pd.isna(ex):
                        continue
                    subject = (a.get("subject") or "").strip()[:300]
                    rows_out.append({
                        "symbol": sym, "ex_date": ex.strftime("%Y-%m-%d"),
                        "subject": subject, "kind": classify_action(subject),
                        "source": SOURCE, "ingested_at": utcnow(),
                    })
            except Exception as exc:                     # noqa: BLE001
                log.warning("%s %d: %s", sym, year, str(exc)[:80])
                sess = _session()
            time.sleep(REQUEST_GAP)

        if rows_out:
            df = pd.DataFrame(rows_out).drop_duplicates(subset=["symbol", "ex_date", "subject"])
            n = upsert("corporate_actions", df, ["symbol", "ex_date", "subject"])
            total += n
            structural = int((df["kind"] == "structural").sum())
            log.info("%-12s %4d actions (%d structural)", sym, n, structural)
    return total
