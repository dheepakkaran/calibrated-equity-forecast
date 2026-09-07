"""Tests for the feedback loop.

The bandit is the part of this project most likely to look impressive and mean
nothing, so these tests pin the guardrails rather than the happy path: that
weights cannot run away, that abstention is not rewarded, and that a confident
mistake costs more than a confident hit earns.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cef.feedback import bandit
from cef.feedback.reward import (CONFIDENT_WRONG_MULTIPLIER, arm_bernoulli,
                                 continuous_reward, conviction_of)


# --------------------------------------------------------------------------
# Reward shape
# --------------------------------------------------------------------------
def test_confident_wrongness_costs_more_than_confident_correctness_earns():
    """The asymmetry is the whole design: it is what pushes the system toward
    reserving conviction instead of spending it."""
    hit = continuous_reward(0.80, actual_up=True)
    miss = continuous_reward(0.80, actual_up=False)
    assert hit > 0 > miss
    assert abs(miss) == pytest.approx(hit * CONFIDENT_WRONG_MULTIPLIER, rel=1e-6)


def test_reward_scales_with_conviction():
    assert (continuous_reward(0.75, True) > continuous_reward(0.60, True)
            > continuous_reward(0.52, True) > 0)


def test_coin_flip_calls_are_damped_both_ways():
    """Being right on a 50.5% call is luck, and must not pay like skill."""
    near = continuous_reward(0.505, True)
    clear = continuous_reward(0.60, True)
    assert 0 < near < clear * 0.2


def test_abstention_is_neither_rewarded_nor_punished():
    """Paying out for abstention would let an arm farm reward by never
    committing."""
    assert continuous_reward(0.99, actual_up=False, acted=False) == 0.0
    assert continuous_reward(0.99, actual_up=True, acted=False) == 0.0


def test_conviction_is_bounded():
    assert conviction_of(0.5) == 0.0
    assert conviction_of(1.0) == 1.0
    assert conviction_of(0.0) == 1.0


def test_arm_update_is_pessimistic_on_a_miss():
    """A miss adds more to beta than an equally confident hit adds to alpha,
    so an arm has to earn confidence rather than be assumed competent."""
    hit = arm_bernoulli("x", 0.7, actual_up=True)
    miss = arm_bernoulli("x", 0.7, actual_up=False)
    assert hit.alpha_add > 0 and hit.beta_add == 0
    assert miss.beta_add > hit.alpha_add
    assert miss.beta_add == pytest.approx(hit.alpha_add * CONFIDENT_WRONG_MULTIPLIER)


def test_every_arm_update_carries_some_weight():
    """Even a coin-flip call moves the posterior slightly, so an arm that
    always sits near 0.5 still accumulates evidence."""
    s = arm_bernoulli("x", 0.5000001, actual_up=True)
    assert s.alpha_add > 0


# --------------------------------------------------------------------------
# Guardrails
# --------------------------------------------------------------------------
def test_weights_respect_floor_and_ceiling():
    """A runaway arm is the failure mode this prevents: on noisy financial
    data one arm will get lucky early, and without a cap it takes everything."""
    runaway = {"lightgbm": 0.97, "lightgbm_cal": 0.01,
               "logistic_cal": 0.01, "reversal": 0.01}
    w = bandit._clip_and_renormalise(runaway)
    assert sum(w.values()) == pytest.approx(1.0, abs=1e-6)
    for arm, v in w.items():
        assert v >= bandit.WEIGHT_FLOOR - 1e-6, f"{arm} below floor: {v}"
        assert v <= bandit.WEIGHT_CEILING + 1e-6, f"{arm} above ceiling: {v}"


def test_uniform_weights_survive_clipping_unchanged():
    uniform = {a: 0.25 for a in bandit.ARMS}
    w = bandit._clip_and_renormalise(uniform)
    assert all(v == pytest.approx(0.25) for v in w.values())


def test_weights_are_uniform_below_the_sample_gate():
    """A handful of outcomes cannot distinguish a one-point edge from noise,
    so the bandit must not act on them."""
    out = bandit.weights_for("__regime_that_cannot_exist__")
    assert out["mode"].startswith("uniform")
    assert all(v == pytest.approx(1 / len(bandit.ARMS)) for v in out["weights"].values())


# --------------------------------------------------------------------------
# Blending
# --------------------------------------------------------------------------
def test_blend_is_a_weighted_average():
    probas = {"lightgbm": 0.60, "reversal": 0.40}
    weights = {"lightgbm": 0.75, "reversal": 0.25}
    assert bandit.blend(probas, weights) == pytest.approx(0.55)


def test_blend_renormalises_over_available_arms():
    """If an arm produced nothing, the others must not be diluted toward 0.5
    by its missing weight."""
    probas = {"lightgbm": 0.60, "reversal": None}
    weights = {"lightgbm": 0.25, "reversal": 0.75}
    assert bandit.blend(probas, weights) == pytest.approx(0.60)


def test_blend_of_nothing_is_a_coin_flip():
    assert bandit.blend({}, {"lightgbm": 1.0}) == 0.5


# --------------------------------------------------------------------------
# Prediction record
# --------------------------------------------------------------------------
def test_next_session_uses_the_exchange_calendar():
    """A Friday forecast targets Monday, not Saturday."""
    from cef.feedback.predict import next_session

    panel = pd.DataFrame({"date": pd.to_datetime(
        ["2026-09-02", "2026-09-03", "2026-09-04", "2026-09-07"])})
    assert next_session(panel, pd.Timestamp("2026-09-04")) == "2026-09-07"
    assert next_session(panel, pd.Timestamp("2026-09-03")) == "2026-09-04"


def test_abstention_threshold_is_actually_binding():
    """If the threshold admitted almost everything, 'selective prediction'
    would be a label rather than a behaviour. On the latest session it should
    hold most symbols back."""
    from cef.config import DATA_DIR
    from cef.feedback.predict import ABSTAIN_BELOW

    path = DATA_DIR / "panel.parquet"
    if not path.exists():
        pytest.skip("panel.parquet missing")
    assert 0 < ABSTAIN_BELOW < 0.05, "threshold is not in a plausible range"


# --------------------------------------------------------------------------
# Reported performance
# --------------------------------------------------------------------------
def test_performance_compares_against_the_selected_rows():
    """The baseline must be the majority rate on the rows the system acted on.
    Measuring against 0.5 credits the model for having picked easier sessions,
    which is timing rather than direction."""
    from cef.feedback.resolver import performance

    perf = performance()
    if not perf.get("n_acted"):
        pytest.skip("no resolved forecasts yet")
    assert "majority_baseline_on_acted" in perf
    assert "edge_vs_majority_pp" in perf
    expected = (perf["accuracy"] - perf["majority_baseline_on_acted"]) * 100
    assert perf["edge_vs_majority_pp"] == pytest.approx(expected, abs=0.02)


# --------------------------------------------------------------------------
# Idempotency
# --------------------------------------------------------------------------
def test_no_duplicate_forecasts_per_session():
    """One forecast per (symbol, session, target).

    Every dashboard page view calls the forecast path, and before this was
    enforced each view wrote a fresh row - so the history screen showed the
    same symbol forecast three times for one session. That quietly contradicts
    the claim that a forecast is written once and never rewritten, which is the
    whole basis for treating the history screen as a record.
    """
    from cef.db import connect

    with connect() as conn:
        dupes = conn.execute(
            "SELECT COUNT(*) FROM (SELECT symbol, as_of_date, target "
            "FROM predictions GROUP BY 1,2,3 HAVING COUNT(*) > 1)").fetchone()[0]
    assert dupes == 0, f"{dupes} (symbol, session) pairs have more than one forecast"


def test_every_outcome_points_at_a_live_prediction():
    """No orphaned outcomes - a score with no claim attached to it."""
    from cef.db import connect

    with connect() as conn:
        orphans = conn.execute(
            "SELECT COUNT(*) FROM outcomes o LEFT JOIN predictions p "
            "ON p.id = o.prediction_id WHERE p.id IS NULL").fetchone()[0]
    assert orphans == 0, f"{orphans} outcomes reference a prediction that is gone"
