"""Contextual Thompson sampling over ensemble arms.

Framing first, because mislabelling this matters. There is no sequential
decision environment here, no state the system acts upon, and no multi-step
credit assignment - so it is not reinforcement learning, and calling it that
would be the kind of overclaim this project exists to avoid. What it is: a
**contextual multi-armed bandit with delayed reward.**

    arms     ensemble members (GBM, calibrated GBM, logistic, reversal rule)
    context  market regime - volatility bucket x trend state
    reward   realised directional quality, computed at T+1
    method   Thompson sampling on a Beta-Bernoulli posterior per (arm, regime)

Regime is the right context because M1 measured real structure there: on the
held-out folds the model's edge was +1.77 pp in trending-up markets and
**-1.20 pp in elevated volatility.** An arm that helps in calm trends and hurts
in turbulence should be weighted accordingly, and that is precisely what a
per-regime posterior encodes.

Three guardrails, because a bandit on noisy financial data will otherwise
collapse onto whichever arm got lucky first:

*Minimum sample gate.* Weights stay uniform in a regime until it has enough
resolved forecasts. With a true edge near one percentage point, a handful of
outcomes is indistinguishable from noise.

*Floor and ceiling.* No arm below 5% or above 60%. This deliberately caps how
much the bandit can ever help - and that is the correct trade, because the
downside it prevents is a single arm taking the whole weight on a run of luck.

*Drift detection.* If rolling accuracy in a regime falls below the coin flip
for long enough, the regime is flagged and reverts to uniform weights. The
event is logged so a later reader can see why the weights moved.
"""
from __future__ import annotations

import json
import logging

import numpy as np

from cef.db import connect, utcnow

log = logging.getLogger(__name__)

ARMS = ("lightgbm_cal", "lightgbm", "logistic_cal", "reversal")

PRIOR_ALPHA = 1.0
PRIOR_BETA = 1.0
MIN_PULLS_PER_REGIME = 100
WEIGHT_FLOOR = 0.05
WEIGHT_CEILING = 0.60
DRIFT_WINDOW = 60          # resolved forecasts
DRIFT_THRESHOLD = 0.48     # accuracy below this is worse than a coin flip
THOMPSON_DRAWS = 400       # draws used to turn posteriors into weights


def _load_state(regime: str) -> dict[str, tuple[float, float, int]]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT arm, alpha, beta, n_pulls FROM bandit_state WHERE regime=?",
            (regime,)).fetchall()
    state = {a: (PRIOR_ALPHA, PRIOR_BETA, 0) for a in ARMS}
    for arm, alpha, beta, n in rows:
        if arm in state:
            state[arm] = (float(alpha), float(beta), int(n))
    return state


def _clip_and_renormalise(weights: dict[str, float]) -> dict[str, float]:
    """Project weights onto {sum = 1, floor <= w <= ceiling} by water-filling.

    Clipping and then rescaling does not work, and the failure is not subtle:
    rescaling to restore the sum pushes a value that was just pinned at the
    ceiling straight back above it. So the residual is instead poured only into
    the arms that still have room, repeatedly, until the sum closes.

    The constraint set is feasible by construction here - one arm at the 60%
    ceiling leaves 40% for three arms with a 5% floor - but the degenerate case
    where every arm is pinned is handled by falling back to uniform rather
    than returning something that violates the bounds.
    """
    keys = list(weights)
    total = sum(weights.values()) or 1.0
    w = {k: weights[k] / total for k in keys}

    for _ in range(64):
        w = {k: float(np.clip(v, WEIGHT_FLOOR, WEIGHT_CEILING)) for k, v in w.items()}
        residual = 1.0 - sum(w.values())
        if abs(residual) < 1e-12:
            break
        # Only arms strictly inside their bounds can absorb the residual, and
        # only in the direction that does not immediately re-pin them.
        free = [k for k in keys
                if (residual > 0 and w[k] < WEIGHT_CEILING - 1e-12)
                or (residual < 0 and w[k] > WEIGHT_FLOOR + 1e-12)]
        if not free:
            return {k: round(1.0 / len(keys), 6) for k in keys}
        share = residual / len(free)
        for k in free:
            w[k] += share

    # Round last, and give any rounding residue to the arm furthest from a
    # bound, so the published weights both sum to 1 and respect the limits.
    out = {k: round(v, 6) for k, v in w.items()}
    drift = round(1.0 - sum(out.values()), 6)
    if drift:
        slack = {k: min(WEIGHT_CEILING - out[k], out[k] - WEIGHT_FLOOR) for k in keys}
        out[max(slack, key=slack.get)] += drift
    return {k: round(v, 6) for k, v in out.items()}


def weights_for(regime: str, rng: np.random.Generator | None = None) -> dict:
    """Thompson-sampled weights for a regime, with guardrails applied."""
    rng = rng or np.random.default_rng()
    state = _load_state(regime)
    n_total = sum(n for _, _, n in state.values())

    uniform = {a: 1.0 / len(ARMS) for a in ARMS}
    if n_total < MIN_PULLS_PER_REGIME:
        return {"weights": uniform, "regime": regime, "n_pulls": n_total,
                "mode": "uniform (below minimum sample gate)",
                "posteriors": {a: {"alpha": state[a][0], "beta": state[a][1],
                                   "n": state[a][2]} for a in ARMS}}

    if _regime_is_drifting(regime):
        return {"weights": uniform, "regime": regime, "n_pulls": n_total,
                "mode": "uniform (drift fallback)",
                "posteriors": {a: {"alpha": state[a][0], "beta": state[a][1],
                                   "n": state[a][2]} for a in ARMS}}

    # Probability each arm is best, estimated by repeated posterior draws.
    wins = {a: 0 for a in ARMS}
    for _ in range(THOMPSON_DRAWS):
        draw = {a: rng.beta(state[a][0], state[a][1]) for a in ARMS}
        wins[max(draw, key=draw.get)] += 1
    raw = {a: wins[a] / THOMPSON_DRAWS for a in ARMS}

    return {"weights": _clip_and_renormalise(raw), "regime": regime,
            "n_pulls": n_total, "mode": "thompson",
            "posteriors": {a: {"alpha": round(state[a][0], 3),
                               "beta": round(state[a][1], 3),
                               "n": state[a][2],
                               "mean": round(state[a][0] / (state[a][0] + state[a][1]), 4)}
                           for a in ARMS}}


def _regime_is_drifting(regime: str) -> bool:
    with connect() as conn:
        rows = conn.execute(
            "SELECT o.correct FROM outcomes o JOIN predictions p ON p.id = o.prediction_id "
            "WHERE p.regime = ? AND p.acted = 1 ORDER BY o.resolved_at DESC LIMIT ?",
            (regime, DRIFT_WINDOW)).fetchall()
    if len(rows) < DRIFT_WINDOW:
        return False
    acc = float(np.mean([r[0] for r in rows]))
    if acc < DRIFT_THRESHOLD:
        with connect() as conn:
            conn.execute(
                "INSERT INTO guardrail_log (at, kind, regime, detail) VALUES (?,?,?,?)",
                (utcnow(), "drift_fallback", regime,
                 json.dumps({"rolling_accuracy": round(acc, 4),
                             "window": DRIFT_WINDOW,
                             "threshold": DRIFT_THRESHOLD})))
        log.warning("regime %s drifting: rolling accuracy %.4f - reverting to uniform",
                    regime, acc)
        return True
    return False


def update(regime: str, arm_scores: list) -> None:
    """Apply Beta updates for one resolved forecast."""
    with connect() as conn:
        for s in arm_scores:
            cur = conn.execute(
                "SELECT alpha, beta, n_pulls FROM bandit_state WHERE arm=? AND regime=?",
                (s.arm, regime)).fetchone()
            alpha, beta, n = cur if cur else (PRIOR_ALPHA, PRIOR_BETA, 0)
            conn.execute(
                "INSERT OR REPLACE INTO bandit_state (arm, regime, alpha, beta, n_pulls, updated_at) "
                "VALUES (?,?,?,?,?,?)",
                (s.arm, regime, float(alpha) + s.alpha_add, float(beta) + s.beta_add,
                 int(n) + 1, utcnow()))


def blend(arm_probas: dict[str, float], weights: dict[str, float]) -> float:
    """Weighted blend of arm probabilities, over the arms actually present."""
    usable = {a: p for a, p in arm_probas.items() if p is not None and np.isfinite(p)}
    if not usable:
        return 0.5
    total = sum(weights.get(a, 0.0) for a in usable) or 1.0
    return float(sum(p * weights.get(a, 0.0) for a, p in usable.items()) / total)


def state_table() -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT arm, regime, alpha, beta, n_pulls FROM bandit_state "
            "ORDER BY regime, arm").fetchall()
    return [{"arm": r[0], "regime": r[1], "alpha": round(r[2], 3),
             "beta": round(r[3], 3), "n_pulls": r[4],
             "posterior_mean": round(r[2] / (r[2] + r[3]), 4)} for r in rows]
