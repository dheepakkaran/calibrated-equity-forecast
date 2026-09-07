"""Headline ingest: market-wide RSS plus per-ticker items.

Unlike ``announcements``, this table cannot be backfilled. Indian financial RSS
feeds expose between one and three days of history, and GDELT - the obvious
free archive - rate-limits to roughly one request every five seconds with
loose entity relevance, which makes a multi-year backfill across 52 symbols
both slow and low quality. So news is forward-only: it starts accumulating the
day ingest first runs, and historical attribution leans on corporate filings
and driver moves instead. That is a real limitation and is documented rather
than papered over.

A browser User-Agent is required: Moneycontrol and Business Standard return
403 to anything else. This is the opposite of several other financial
endpoints, which reject browser agents - so the header here is a deliberate
per-source choice.
"""
from __future__ import annotations

import hashlib
import logging
import time
from datetime import timezone

import pandas as pd
import requests

from cef.db import upsert, utcnow
from cef.universe import UNIVERSE, yf_symbol

log = logging.getLogger(__name__)

BROWSER_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
              "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36")

FEEDS = {
    "economic_times_markets": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "economic_times_stocks": "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms",
    "livemint_markets": "https://www.livemint.com/rss/markets",
    "hindu_businessline": "https://www.thehindubusinessline.com/markets/feeder/default.rss",
    "moneycontrol_markets": "https://www.moneycontrol.com/rss/marketreports.xml",
    "business_standard": "https://www.business-standard.com/rss/markets-106.rss",
}


def _hash(url: str, headline: str) -> str:
    return hashlib.sha1(f"{url}|{headline}".encode()).hexdigest()[:20]


def _to_ist_date(ts) -> tuple[str, str]:
    t = pd.to_datetime(ts, errors="coerce", utc=True)
    if pd.isna(t):
        t = pd.Timestamp.now(tz="UTC")
    ist = t.tz_convert("Asia/Kolkata")
    return ist.isoformat(), ist.strftime("%Y-%m-%d")


def fetch_rss() -> pd.DataFrame:
    import feedparser

    rows = []
    for outlet, url in FEEDS.items():
        try:
            r = requests.get(url, timeout=25, headers={"User-Agent": BROWSER_UA})
            if r.status_code != 200:
                log.warning("%-24s HTTP %s", outlet, r.status_code)
                continue
            parsed = feedparser.parse(r.content)
            for e in parsed.entries:
                head = (e.get("title") or "").strip()
                if not head:
                    continue
                pub, date = _to_ist_date(e.get("published") or e.get("updated"))
                rows.append({
                    "url_hash": _hash(e.get("link", ""), head),
                    "symbol": None,                 # market-wide until matched
                    "published_at": pub, "date": date,
                    "headline": head[:500],
                    "summary": (e.get("summary") or "")[:1000],
                    "outlet": outlet, "url": e.get("link", ""),
                    "finbert_label": None, "finbert_score": None,
                    "source": "rss", "ingested_at": utcnow(),
                })
            log.info("%-24s %3d items", outlet, len(parsed.entries))
        except Exception as exc:                    # noqa: BLE001
            log.warning("%-24s %s", outlet, str(exc)[:70])
        time.sleep(0.4)
    return pd.DataFrame(rows)


def fetch_ticker_news(symbols: list[str]) -> pd.DataFrame:
    """Per-symbol headlines from Yahoo. Sparse, but already entity-resolved -
    no name matching needed, which makes these the highest-precision rows."""
    import yfinance as yf

    rows = []
    for sym in symbols:
        try:
            items = yf.Ticker(yf_symbol(sym)).news or []
        except Exception as exc:                    # noqa: BLE001
            log.warning("%s: %s", sym, str(exc)[:60])
            continue
        for a in items:
            c = a.get("content", a)
            head = (c.get("title") or "").strip()
            if not head:
                continue
            link = ((c.get("canonicalUrl") or {}).get("url")
                    if isinstance(c.get("canonicalUrl"), dict) else c.get("link", "")) or ""
            pub, date = _to_ist_date(c.get("pubDate") or c.get("providerPublishTime"))
            provider = c.get("provider")
            rows.append({
                "url_hash": _hash(link, head),
                "symbol": sym,
                "published_at": pub, "date": date,
                "headline": head[:500],
                "summary": (c.get("summary") or c.get("description") or "")[:1000],
                "outlet": (provider or {}).get("displayName") if isinstance(provider, dict) else "yahoo",
                "url": link,
                "finbert_label": None, "finbert_score": None,
                "source": "yfinance", "ingested_at": utcnow(),
            })
        time.sleep(0.25)
    return pd.DataFrame(rows)


def ingest_news(symbols: list[str] | None = None) -> int:
    symbols = symbols or UNIVERSE
    frames = [fetch_rss(), fetch_ticker_news(symbols)]
    df = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    if df.empty:
        return 0
    df = df.drop_duplicates(subset=["url_hash"])
    n = upsert("news", df, ["url_hash"])
    log.info("news: %d rows (%d market-wide, %d symbol-tagged)",
             n, int(df["symbol"].isna().sum()), int(df["symbol"].notna().sum()))
    return n
