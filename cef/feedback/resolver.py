"""The nightly job that closes the loop.

For every forecast whose target session has now traded: fetch what actually
happened, score the ensemble call and each arm separately, update the bandit
posterior for that (arm, regime), and mark the forecast resolved.

Scoring each arm independently is what makes the bandit contextual rather than
decorative. The ensemble's own outcome says whether the blend worked; only the
per-arm outcomes say *which* arm deserved the weight in *that* regime, and the
per-regime posterior is where that accumulates.

A target session that never traded - a holiday the calendar did not
anticipate, or a suspension - is marked VOID rather than resolved. Scoring a
forecast against a session that did not happen would quietly reward or punish
arms for nothing.
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from cef.db import connect, utcnow
from cef.feedback import bandit
from cef.feedback.reward import arm_bernoulli, continuous_reward

log = logging.getLogger(__name__)


def _actuals(panel: pd.DataFrame) -> pd.DataFrame:
    """Realised relative return per (symbol, session), from the panel."""
    p = panel[["symbol", "date", "close"]].copy().sort_values(["symbol", "date"])
    p["ret"] = np.log(p["close"] / p.groupby("symbol", observed=True)["close"].shift(1))
    p["ret_mkt"] = p.groupby("date", observed=True)["ret"].transform("median")
    p["ret_rel"] = p["ret"] - p["ret_mkt"]
    p["date_s"] = p["date"].dt.strftime("%Y-%m-%d")
    return p.set_index(["symbol", "date_s"])[["ret", "ret_rel"]]


def resolve(panel: pd.DataFrame, as_of: str | None = None) -> dict:
    with connect() as conn:
        preds = pd.read_sql(
            "SELECT id, symbol, target_date, direction, proba, acted, regime, arm_probas "
            "FROM predictions WHERE status='PENDING'" +
            (" AND target_date <= ?" if as_of else ""),
            conn, params=[as_of] if as_of else [])
    if preds.empty:
        log.info("nothing pending")
        return {"resolved": 0, "void": 0}

    actuals = _actuals(panel)
    resolved = void = 0
    out_rows, updates = [], []

    for p in preds.itertuples(index=False):
        key = (p.symbol, p.target_date)
        if key not in actuals.index:
            updates.append(("VOID", p.id))
            void += 1
            continue

        a = actuals.loc[key]
        ret, ret_rel = float(a["ret"]), float(a["ret_rel"])
        if not np.isfinite(ret_rel):
            updates.append(("VOID", p.id))
            void += 1
            continue

        actual_up = ret_rel > 0
        called_up = p.proba > 0.5
        correct = int(called_up == actual_up) if p.acted else None

        probas = json.loads(p.arm_probas or "{}")
        scores = [arm_bernoulli(arm, pr, actual_up)
                  for arm, pr in probas.items() if arm in bandit.ARMS]
        bandit.update(p.regime, scores)

        out_rows.append({
            "prediction_id": p.id,
            "actual_ret": round(ret, 6),
            "actual_ret_rel": round(ret_rel, 6),
            "actual_dir": "outperform" if actual_up else "underperform",
            "correct": correct,
            "reward": continuous_reward(p.proba, actual_up, bool(p.acted)),
            "arm_rewards": json.dumps({s.arm: round(s.reward, 4) for s in scores}),
            "resolved_at": utcnow(),
        })
        updates.append(("RESOLVED", p.id))
        resolved += 1

    with connect() as conn:
        if out_rows:
            cols = list(out_rows[0])
            conn.executemany(
                f"INSERT OR REPLACE INTO outcomes ({','.join(cols)}) "
                f"VALUES ({','.join('?' * len(cols))})",
                [tuple(r[c] for c in cols) for r in out_rows])
        conn.executemany("UPDATE predictions SET status=? WHERE id=?", updates)

    log.info("resolved %d, void %d", resolved, void)
    return {"resolved": resolved, "void": void}


def performance(window: int | None = None) -> dict:
    """Live track record. Abstentions are excluded from accuracy and counted
    separately - a forecast the system declined to make is neither right nor
    wrong, and folding it in either direction would flatter the number."""
    q = ("SELECT p.symbol, p.as_of_date, p.target_date, p.regime, p.acted, "
         "p.proba, p.conviction, o.correct, o.reward, o.actual_ret_rel "
         "FROM outcomes o JOIN predictions p ON p.id = o.prediction_id "
         "ORDER BY p.target_date DESC")
    if window:
        q += f" LIMIT {int(window)}"
    with connect() as conn:
        df = pd.read_sql(q, conn)
    if df.empty:
        return {"n": 0}

    acted = df[df["acted"] == 1]
    # Two baselines, and the second is the one that matters. `base_all` is the
    # out-performance rate over every forecast; `base_acted` is the rate on the
    # rows the system actually chose to act on. Measuring the edge against 0.5,
    # or against the full-sample rate, credits the model for having selected
    # sessions that were easier - which is timing, not direction.
    base_all = float((df["actual_ret_rel"] > 0).mean())
    base_acted = (float((acted["actual_ret_rel"] > 0).mean()) if len(acted)
                  else float("nan"))
    majority = max(base_acted, 1 - base_acted) if len(acted) else float("nan")
    out = {
        "n": len(df),
        "n_acted": len(acted),
        "abstention_rate": round(1 - len(acted) / len(df), 4),
        "accuracy": round(float(acted["correct"].mean()), 4) if len(acted) else None,
        "base_rate_all": round(base_all, 4),
        "base_rate_on_acted": round(base_acted, 4) if len(acted) else None,
        "majority_baseline_on_acted": round(majority, 4) if len(acted) else None,
        "edge_vs_majority_pp": (round((acted["correct"].mean() - majority) * 100, 2)
                                if len(acted) else None),
        "mean_reward": round(float(df["reward"].mean()), 4),
        "by_regime": {},
    }
    for regime, g in acted.groupby("regime"):
        if len(g) >= 20:
            out["by_regime"][regime] = {
                "n": len(g),
                "accuracy": round(float(g["correct"].mean()), 4),
                "mean_reward": round(float(g["reward"].mean()), 4),
            }
    return out
