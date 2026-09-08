"""Tracked guesses: a plain JSON ledger in the repository.

Deliberately a file in git rather than a database row. The point of tracking a
forecast is that the claim is fixed *before* the outcome is known, and a git
commit is the strongest cheap proof of that available — the timestamp and the
content are both in the history, and neither can be quietly edited later
without it showing up in a diff.

Resolution happens twice, because a next-session close-to-close forecast cannot
honestly be graded at the opening bell:

    09:20 IST  early read   the gap only. Provisional, and labelled as such.
    15:45 IST  resolved     the actual close-to-close outcome. Final.

Each resolved guess gets a category and two numbers, which is what a reader
needs to judge it: was the direction right, and did the share move enough for
that to mean anything.
"""
from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

from cef.config import ROOT

log = logging.getLogger(__name__)

LEDGER = ROOT / "tracking" / "predictions.json"
SUMMARY = ROOT / "tracking" / "summary.json"

# A relative move smaller than this fraction of the share's own typical daily
# swing is noise. Being right about noise is luck, and being wrong about it
# costs nothing — both land in the middle category rather than at an extreme.
NOISE_BAND_SIGMA = 0.25


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load() -> list[dict]:
    if not LEDGER.exists():
        return []
    try:
        return json.loads(LEDGER.read_text())
    except json.JSONDecodeError:
        log.error("ledger is not valid JSON; refusing to overwrite it")
        raise


def save(rows: list[dict]) -> None:
    LEDGER.parent.mkdir(exist_ok=True, parents=True)
    LEDGER.write_text(json.dumps(rows, indent=2) + "\n")


def track(forecast: dict, note: str = "") -> dict:
    """Record a guess for tomorrow. Idempotent per (symbol, target session)."""
    rows = load()
    for r in rows:
        if (r["symbol"] == forecast["symbol"]
                and r["target_session"] == forecast["target_date"]):
            return {**r, "already_tracked": True}

    row = {
        "id": uuid.uuid4().hex[:12],
        "symbol": forecast["symbol"],
        "sector": forecast.get("sector"),
        "tracked_at": _now(),
        "as_of_session": forecast["as_of"],
        "target_session": forecast["target_date"],
        "prediction_id": forecast.get("prediction_id"),
        "model_version": forecast.get("model_version"),
        "guess": {
            "direction": forecast["direction"],
            "proba_outperform": forecast["proba_outperform"],
            "confidence": forecast["confidence"],
            "conviction": forecast["conviction"],
            "acted": forecast["acted"],
            "regime": forecast["regime"]["label"],
            "last_close": forecast["last_close"],
            "typical_daily_swing": forecast["typical_daily_range"],
        },
        "note": note[:280],
        "status": "PENDING",
        "early_read": None,
        "outcome": None,
    }
    rows.append(row)
    save(rows)
    return {**row, "already_tracked": False}


def categorise(direction_hit: bool | None, move_vs_typical: float,
               acted: bool) -> tuple[str, str]:
    """Category plus the one-line reason it landed there.

    ``move_vs_typical`` is the realised market-relative move divided by the
    share's own typical daily swing, so it is comparable across a ₹500 stock
    and a ₹5,000 one.
    """
    noise = abs(move_vs_typical) < NOISE_BAND_SIGMA

    if not acted:
        # An abstention is graded on whether staying quiet was the right call.
        if noise:
            return "okay", ("Declined to call it, and the share barely moved — "
                            "staying quiet was right.")
        return "okayish", ("Declined to call it and the share did move. Nothing "
                           "was lost, but nothing was caught either.")

    if direction_hit and not noise:
        return "okay", "Called the direction correctly on a move that mattered."
    if direction_hit and noise:
        return "okayish", ("Direction was right, but the share hardly moved — "
                           "closer to luck than to skill.")
    if not direction_hit and noise:
        return "okayish", ("Direction was wrong, but on a move too small to "
                           "matter either way.")
    return "not okay", "Called the direction wrong on a move that mattered."


def early_read(row: dict, open_px: float, prev_close: float,
               market_open_move_pct: float) -> dict:
    """The 09:20 IST provisional read, from the opening gap only.

    Explicitly not a verdict. The forecast is close-to-close and the closing
    price does not exist yet; a gap agreeing with the call is weak evidence and
    is labelled that way.
    """
    gap_pct = (open_px / prev_close - 1) * 100
    rel_gap = gap_pct - market_open_move_pct
    called_up = row["guess"]["proba_outperform"] > 0.5
    leaning = "with the call" if (rel_gap > 0) == called_up else "against the call"
    return {
        "read_at": _now(),
        "open": round(open_px, 2),
        "gap_pct": round(gap_pct, 3),
        "market_gap_pct": round(market_open_move_pct, 3),
        "relative_gap_pct": round(rel_gap, 3),
        "leaning": leaning,
        "note": ("Provisional. The forecast is close-to-close, so the opening gap "
                 "is weak evidence and settles nothing — the outcome is graded "
                 "after the close."),
    }


def resolve(row: dict, close_px: float, prev_close: float,
            market_close_move_pct: float) -> dict:
    """The 15:45 IST final resolution."""
    from cef.feedback.reward import continuous_reward

    move_pct = (close_px / prev_close - 1) * 100
    rel_move_pct = move_pct - market_close_move_pct
    actual_up = rel_move_pct > 0
    called_up = row["guess"]["proba_outperform"] > 0.5
    acted = bool(row["guess"]["acted"])
    hit = (called_up == actual_up) if acted else None

    swing = row["guess"].get("typical_daily_swing") or 0.0
    last_close = row["guess"].get("last_close") or close_px
    typical_pct = (swing / last_close * 100) if last_close else 1.0
    move_vs_typical = rel_move_pct / typical_pct if typical_pct else 0.0

    category, why = categorise(hit, move_vs_typical, acted)
    reward = continuous_reward(row["guess"]["proba_outperform"], actual_up, acted)

    return {
        "resolved_at": _now(),
        "close": round(close_px, 2),
        "move_pct": round(move_pct, 3),
        "market_move_pct": round(market_close_move_pct, 3),
        "relative_move_pct": round(rel_move_pct, 3),
        "actual": "outperformed" if actual_up else "lagged",
        "category": category,
        "why": why,
        # The two values a reader is owed: was it right, and did it matter.
        "direction_hit": hit,
        "move_vs_typical_swing": round(move_vs_typical, 2),
        "reward": reward,
    }


def summarise(rows: list[dict] | None = None) -> dict:
    rows = rows if rows is not None else load()
    resolved = [r for r in rows if r.get("outcome")]
    counts = {"okay": 0, "okayish": 0, "not okay": 0}
    for r in resolved:
        counts[r["outcome"]["category"]] = counts.get(r["outcome"]["category"], 0) + 1

    acted = [r for r in resolved if r["guess"]["acted"]]
    hits = [r for r in acted if r["outcome"]["direction_hit"]]
    out = {
        "generated_at": _now(),
        "tracked": len(rows),
        "pending": sum(1 for r in rows if r["status"] == "PENDING"),
        "resolved": len(resolved),
        "categories": counts,
        "with_a_call": len(acted),
        "abstained": len(resolved) - len(acted),
        "direction_accuracy": (round(len(hits) / len(acted), 4) if acted else None),
        "mean_reward": (round(sum(r["outcome"]["reward"] for r in resolved) / len(resolved), 4)
                        if resolved else None),
        "note": ("Direction accuracy counts only the guesses that carried a call. "
                 "An abstention is neither right nor wrong, and folding it either "
                 "way would flatter the number. This ledger is small — treat it as "
                 "a live log, not as evidence; the held-out assessment in the "
                 "README is the measured result."),
    }
    SUMMARY.parent.mkdir(exist_ok=True, parents=True)
    SUMMARY.write_text(json.dumps(out, indent=2) + "\n")
    return out
