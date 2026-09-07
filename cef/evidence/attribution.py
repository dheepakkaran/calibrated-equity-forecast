"""Matching significant price moves to the events that plausibly caused them.

The output of this module is the "why it moved" table: for each significant
session, the most likely explanation, a confidence, and a link to a source a
reader can open. Four design commitments keep it honest.

**Attribution is correlational, and says so.** A high score means a plausible
event was found close in time, pointing the same way, from an authoritative
source. It does not establish cause. The confidence number is the strength of
that circumstantial case, not a probability that the event caused the move.

**Silence beats invention.** A move with no qualifying evidence is labelled
unexplained and kept in the table. Roughly half of all NSE filings are
administrative - lost share certificates, trading-window notices - so a
system willing to attribute anything will always find something to blame, and
that is the failure mode of ordinary market commentary. Low-materiality
filings are never offered as explanations.

**Causality is respected.** Candidate filings are matched on
``announcements.date``, which the ingest layer has already set to the first
session a filing could act on: a release at 18:20 IST belongs to the next
session, not the one it was filed after. Over half of all filings arrive after
the close, so this is the difference between attribution and hindsight.

**The market gets credit for market moves.** A stock that fell with its index
did not need a company-specific reason. Market-wide moves are attributed to
drivers rather than to filings, and if neither fits, to nothing.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from cef.db import upsert, utcnow
from cef.evidence.event_map import classify
from cef.universe import drivers_for

log = logging.getLogger(__name__)

MIN_MATERIALITY = 0.55        # below this a filing is never an explanation
MIN_CONFIDENCE = 0.20         # below this the move is reported as unexplained
DRIVER_MIN_SIGMA = 1.5        # a driver must itself have moved to explain anything
DRIVER_MIN_CORR = 0.20

# Sessions of lookback when searching for a candidate filing. The ingest layer
# has already rolled after-close filings to the next session, so 0 would be
# defensible; 1 is allowed because an event sometimes takes a session to
# propagate through analyst notes and media pickup.
#
# Widening this is the easiest way to inflate the explained rate, and the
# measurements say exactly how easy. Of the idiosyncratic moves in this panel,
# the share with *any* filing nearby runs 47.8% at zero lookback, 65.4% at one
# session and 80.7% at three. The share with a filing above the materiality
# threshold runs 12.6%, 19.2% and 27.7%. A system reporting "81% explained"
# would be citing lost share certificates and analyst-meet notices, so the
# window stays short and the materiality gate does the work.
LOOKBACK_SESSIONS = 1


def _direction_factor(event_dir: int, move_sign: int) -> tuple[float, str]:
    """How much a filing's own directional prior supports this move."""
    if event_dir == 0:
        return 0.75, "direction not implied by filing type"
    if event_dir == move_sign:
        return 1.00, "filing direction agrees with the move"
    return 0.35, "filing direction disagrees with the move"


def _best_announcement(cands: pd.DataFrame, move_sign: int) -> dict | None:
    best = None
    for a in cands.itertuples(index=False):
        ec = classify(a.category, a.body)
        if ec.materiality < MIN_MATERIALITY:
            continue
        factor, note = _direction_factor(ec.direction, move_sign)
        conf = ec.materiality * factor
        if best is None or conf > best["confidence"]:
            best = {
                "evidence_id": a.seq_id,
                "headline": (a.body or ec.label)[:280],
                "source_url": a.attachment,
                "confidence": round(float(conf), 3),
                "rationale": (f"{ec.label} filed {a.announced_at[:16]} IST"
                              f"{' (after close, first acts next session)' if a.after_close else ''}"
                              f"; materiality {ec.materiality:.2f}; {note}"),
                "n_candidates": len(cands),
            }
    return best


def _driver_explanation(symbol: str, date: pd.Timestamp, move_sign: int,
                        driver_moves: pd.DataFrame, corrs: dict) -> dict | None:
    """Fall back to the commodities and indices this symbol actually tracks."""
    if date not in driver_moves.index:
        return None
    row = driver_moves.loc[date]
    best = None
    for key in drivers_for(symbol):
        if key not in row.index:
            continue
        sigma = row[key]
        corr = corrs.get((symbol, key), np.nan)
        if not np.isfinite(sigma) or not np.isfinite(corr):
            continue
        if abs(sigma) < DRIVER_MIN_SIGMA or abs(corr) < DRIVER_MIN_CORR:
            continue
        # Does the driver move, transmitted through its correlation, point the
        # same way the stock went?
        implied = np.sign(sigma) * np.sign(corr)
        if implied != move_sign:
            continue
        conf = float(min(0.65, abs(corr) * min(1.0, abs(sigma) / 3.0) * 1.6))
        if best is None or conf > best["confidence"]:
            best = {
                "evidence_id": f"driver:{key}",
                "headline": f"{key} moved {sigma:+.1f} sigma",
                "source_url": None,
                "confidence": round(conf, 3),
                "rationale": (f"mapped driver {key} moved {sigma:+.1f} sigma; "
                              f"120-day correlation {corr:+.2f}; direction consistent"),
                "n_candidates": 0,
            }
    return best


def attribute(moves: pd.DataFrame, announcements: pd.DataFrame,
              driver_moves: pd.DataFrame, corrs: dict,
              lookback: int = LOOKBACK_SESSIONS,
              sessions: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    """One row per significant move, explained or explicitly not."""
    ann = announcements.copy()
    ann["date"] = pd.to_datetime(ann["date"])
    by_key = {k: g for k, g in ann.groupby(["symbol", "date"], observed=True)}

    # Lookback is counted in trading sessions, not calendar days, so a Monday
    # move can reach Friday's filings without also reaching into a holiday gap.
    sessions = (pd.DatetimeIndex(sorted(driver_moves.index))
                if sessions is None else pd.DatetimeIndex(sorted(sessions)))
    pos = {d: i for i, d in enumerate(sessions)}

    def candidates(symbol: str, date: pd.Timestamp) -> pd.DataFrame | None:
        i = pos.get(date)
        window = ([sessions[j] for j in range(max(0, i - lookback), i + 1)]
                  if i is not None else [date])
        parts = [by_key[(symbol, d)] for d in window if (symbol, d) in by_key]
        return pd.concat(parts) if parts else None

    out = []
    for m in moves.itertuples(index=False):
        move_sign = int(np.sign(m.ret_rel if not m.market_wide else m.ret))
        if move_sign == 0:
            continue

        pick, kind = None, "unexplained"

        # A move the whole market made needs a market reason, not a filing.
        if not m.market_wide:
            cands = candidates(m.symbol, m.date)
            if cands is not None and len(cands):
                pick = _best_announcement(cands, move_sign)
                if pick:
                    kind = "announcement"

        if pick is None:
            pick = _driver_explanation(m.symbol, m.date, int(np.sign(m.ret)),
                                       driver_moves, corrs)
            if pick:
                kind = "driver"

        if pick is None or pick["confidence"] < MIN_CONFIDENCE:
            pick = {"evidence_id": None, "headline": None, "source_url": None,
                    "confidence": 0.0, "n_candidates": 0,
                    "rationale": ("no filing above the materiality threshold and no "
                                  "mapped driver moved consistently; left unexplained")}
            kind = "unexplained"

        out.append({
            "symbol": m.symbol, "date": m.date.strftime("%Y-%m-%d"),
            "ret": round(float(m.ret), 5),
            # Both returns are recorded because the directional test uses the
            # relative one. A stock down 1% on a day the market fell 3% moved
            # *up* relative to its peers, and a reader shown only the absolute
            # figure would think "direction agrees" was a bug.
            "ret_rel": round(float(m.ret_rel), 5),
            "sigma": round(float(m.severity), 2),
            "market_wide": int(bool(m.market_wide)),
            "kind": kind, "evidence_id": pick["evidence_id"],
            "headline": pick["headline"], "source_url": pick["source_url"],
            "confidence": pick["confidence"], "rationale": pick["rationale"],
            "created_at": utcnow(),
        })

    df = pd.DataFrame(out)
    if not df.empty:
        upsert("attributions", df, ["symbol", "date"])
    return df


def confidence_band(c: float) -> str:
    """The label the interface shows. Deliberately coarse - a two-decimal
    confidence on a correlational match would imply precision that is not
    there."""
    if c >= 0.75:
        return "high"
    if c >= 0.45:
        return "medium"
    if c > 0.0:
        return "low"
    return "none"
