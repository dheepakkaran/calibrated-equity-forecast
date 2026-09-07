"""Detecting capital-structure breaks in the price series.

A single-session -65% "return" on a demerger ex-date is not a return, and the
job here is to separate those from genuine crashes without discarding either
by accident. Two independent confirmations are used, and a candidate needs one
of them plus an idiosyncrasy test.

*Idiosyncrasy.* A capital-structure change affects one company. Genuine large
moves cluster: fifteen symbols fell more than 18% on 2020-03-23, five on
2024-06-04. If many names moved together on a date, the cause is the market
and the row is real.

*Exchange confirmation.* A structural corporate action from the NSE within a
short window of the break. This is what catches demergers, whose price ratio
depends on the value of the demerged entity and lands on no particular number.
It is the reason ADANIENT's 2018-09-06 scheme of arrangement is caught.

*Ratio confirmation, tied to a specific action.* Splits and bonuses divide the
share by a stated amount, so a 1:2 bonus must produce a price ratio of exactly
2/3. TRENT breaks on 2026-01-01 at a ratio of 0.6695 while the exchange
records its 1:2 bonus with an ex-date of 2026-06-04 - the ratio is 2/3 to
within 0.4%, so the break is that bonus and the recorded ex-date is simply
misaligned in the feed. This path therefore still requires exchange evidence:
the symbol must have a structural action whose *own* stated ratio matches.

An earlier version of this detector matched any simple fraction with a
denominator up to ten. That set, widened by a tolerance, very nearly covers
the whole interval below 1 - 5/7 is 0.714 and 7/9 is 0.778 - so it masked the
Adani selloff of February 2023 as a "split" and would have quietly deleted the
largest genuine event in the panel. Unanchored numerology is not evidence.

What must survive: 2020-03-23, the Adani selloff of February 2023, and
INDUSINDBK's 2025-03-11 accounting disclosure. All are real, and all are
among the largest moves in the panel.
"""
from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

RET_THRESHOLD = 0.18          # ~20%; below this a large-cap move is plausible
RATIO_TOL = 0.006             # 0.6%: a stated split ratio is exact, not approximate
# A structural action explains the break on its ex-date, not one a fortnight
# later. The window exists only to absorb ex-date misalignment in the feed, so
# it is deliberately short: at ten days a genuine crash landing shortly after a
# demerger would be masked as part of it.
ACTION_WINDOW_DAYS = 3
MAX_PEERS_MOVING = 2          # more than this and the cause is the market

# "Bonus 1:2" -> one new share per two held -> price ratio 2/3.
# "Split ... 1:5" / face-value subdivision -> ratio 1/5.
_BONUS_RE = re.compile(r"bonus\s*(\d+)\s*[:/]\s*(\d+)", re.I)
_SPLIT_RE = re.compile(r"(?:split|sub[- ]?division).*?(\d+)\s*[:/]\s*(\d+)", re.I)
# "Face Value Split From Rs 10 To Rs 2" -> ratio 2/10
_FACE_RE = re.compile(r"face value.*?from\s*(?:rs\.?\s*)?(\d+(?:\.\d+)?)"
                      r".*?to\s*(?:rs\.?\s*)?(\d+(?:\.\d+)?)", re.I)


def action_implied_ratio(subject: str) -> float | None:
    """Price ratio a stated corporate action must produce, or None."""
    m = _BONUS_RE.search(subject or "")
    if m:
        new, held = int(m.group(1)), int(m.group(2))
        if new > 0 and held > 0:
            return held / (held + new)
    m = _SPLIT_RE.search(subject or "")
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        if a > 0 and b > 0:
            return min(a, b) / max(a, b)
    m = _FACE_RE.search(subject or "")
    if m:
        before, after = float(m.group(1)), float(m.group(2))
        if before > 0 and 0 < after < before:
            return after / before
    return None


def detect_breaks(ohlcv: pd.DataFrame, actions: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return one row per confirmed capital-structure break."""
    p = ohlcv[["symbol", "date", "close"]].copy().sort_values(["symbol", "date"])
    p["ret"] = np.log(p["close"] / p.groupby("symbol", observed=True)["close"].shift(1))
    p["ratio"] = np.exp(p["ret"])

    cand = p[p["ret"].abs() > RET_THRESHOLD].copy()
    if cand.empty:
        return pd.DataFrame(columns=["symbol", "date", "ret", "implied_ratio",
                                     "reason", "confirmed_by"])

    # How many other symbols moved hard on the same date?
    peers = cand.groupby("date", observed=True)["symbol"].transform("size") - 1
    cand["peers_moving"] = peers

    structural = pd.DataFrame(columns=["symbol", "ex_date", "subject"])
    if actions is not None and not actions.empty:
        structural = actions[actions["kind"] == "structural"].copy()
        structural["ex_date"] = pd.to_datetime(structural["ex_date"])

    out = []
    for row in cand.itertuples(index=False):
        if row.peers_moving > MAX_PEERS_MOVING:
            continue                                  # market-wide, therefore real

        confirmed_by = reason = None

        if not structural.empty:
            near = structural[
                (structural["symbol"] == row.symbol)
                & ((structural["ex_date"] - row.date).abs() <= pd.Timedelta(days=ACTION_WINDOW_DAYS))]
            if not near.empty:
                confirmed_by = "nse_corporate_action"
                reason = str(near.iloc[0]["subject"])[:120]

        if confirmed_by is None and not structural.empty:
            # Fall back to matching the break against the ratio a recorded
            # action must produce, ignoring its ex-date. Still exchange-
            # anchored: an unexplained number that merely looks tidy is not
            # accepted.
            same = structural[structural["symbol"] == row.symbol]
            for act in same.itertuples(index=False):
                expected = action_implied_ratio(act.subject)
                if expected and abs(row.ratio - expected) <= RATIO_TOL * expected:
                    confirmed_by = "action_ratio_match"
                    reason = (f"{str(act.subject)[:60]} implies ratio {expected:.4f}; "
                              f"observed {row.ratio:.4f}")
                    break

        if confirmed_by:
            out.append({
                "symbol": row.symbol, "date": row.date.strftime("%Y-%m-%d"),
                "ret": float(row.ret), "implied_ratio": float(row.ratio),
                "reason": reason, "confirmed_by": confirmed_by,
            })

    df = pd.DataFrame(out)
    if df.empty:
        return df

    # One action explains one break. Where several candidates matched the same
    # corporate action, keep the closest in time and release the rest back to
    # being ordinary price moves - otherwise the rebound out of a demerger gap
    # gets masked alongside the gap itself.
    df["_key"] = df["symbol"] + "|" + df["reason"].astype(str)
    df["_dist"] = [
        min((abs((pd.Timestamp(d) - ex).days)
             for ex in structural.loc[structural["symbol"] == sym, "ex_date"]),
            default=0)
        for sym, d in zip(df["symbol"], df["date"])
    ] if not structural.empty else 0
    df = (df.sort_values(["_key", "_dist", "date"])
            .groupby("_key", as_index=False, sort=False).head(1)
            .drop(columns=["_key", "_dist"])
            .sort_values("date").reset_index(drop=True))
    return df


def report_candidates(ohlcv: pd.DataFrame, actions: pd.DataFrame | None = None) -> pd.DataFrame:
    """Every large move with the verdict attached - for auditing what was
    masked and, more importantly, what was not."""
    p = ohlcv[["symbol", "date", "close"]].copy().sort_values(["symbol", "date"])
    p["ret"] = np.log(p["close"] / p.groupby("symbol", observed=True)["close"].shift(1))
    p["ratio"] = np.exp(p["ret"])
    cand = p[p["ret"].abs() > RET_THRESHOLD].copy()
    cand["peers_moving"] = cand.groupby("date", observed=True)["symbol"].transform("size") - 1

    breaks = detect_breaks(ohlcv, actions)
    key = set(zip(breaks["symbol"], breaks["date"])) if not breaks.empty else set()
    cand["date_s"] = cand["date"].dt.strftime("%Y-%m-%d")
    cand["masked"] = [(s, d) in key for s, d in zip(cand["symbol"], cand["date_s"])]
    cand["verdict"] = np.where(cand["masked"], "MASKED - capital structure",
                       np.where(cand["peers_moving"] > MAX_PEERS_MOVING,
                                "kept - market-wide", "kept - idiosyncratic price move"))
    return cand[["symbol", "date_s", "ret", "ratio", "peers_moving", "verdict"]]
