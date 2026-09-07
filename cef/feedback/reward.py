"""Reward computation for resolved forecasts.

Two numbers come out of a resolved forecast and they are used for different
things, so they are computed separately.

``continuous_reward`` is the reportable score - a signed quantity shown on the
history screen. It is scaled by conviction and penalises confident wrongness
harder than it rewards confident correctness. That asymmetry is the point: a
system that is punished 1.5x for being loudly wrong learns to reserve
conviction, and a project whose entire claim is honest confidence cannot use a
symmetric loss.

``arm_bernoulli`` is what the bandit consumes. Thompson sampling over a
Beta-Bernoulli posterior needs successes and failures, not a signed score, so
each arm's call is scored as a conviction-weighted hit or miss. The same 1.5x
asymmetry is applied to the failure weight, which biases every posterior
slightly pessimistic - arms have to earn confidence rather than be assumed
competent.

Abstention earns nothing and costs nothing. A forecast the system declined to
make is not evidence about any arm, and paying out for abstention would let a
model farm reward by never committing.
"""
from __future__ import annotations

from dataclasses import dataclass

CONFIDENT_WRONG_MULTIPLIER = 1.5

# Below this the call is a coin toss and is treated as such: the reward is
# damped so that being right by luck on a 50.5% call cannot look like skill.
COIN_FLIP_BAND = 0.02          # |proba - 0.5| below this is noise


@dataclass(frozen=True)
class ArmScore:
    arm: str
    proba: float
    correct: bool
    conviction: float
    alpha_add: float
    beta_add: float
    reward: float


def conviction_of(proba: float) -> float:
    """Distance from the coin flip, rescaled to [0, 1]."""
    return min(1.0, abs(float(proba) - 0.5) * 2.0)


def continuous_reward(proba: float, actual_up: bool, acted: bool = True) -> float:
    """Signed, reportable score for one forecast."""
    if not acted:
        return 0.0
    conv = conviction_of(proba)
    called_up = proba > 0.5
    correct = called_up == bool(actual_up)

    r = conv if correct else -CONFIDENT_WRONG_MULTIPLIER * conv
    if abs(proba - 0.5) <= COIN_FLIP_BAND:
        # Low stakes both ways. Explicitly *not* zero: a system should still
        # prefer being right near the boundary, just not by much.
        r *= 0.5
    return round(float(r), 5)


def arm_bernoulli(arm: str, proba: float, actual_up: bool) -> ArmScore:
    """Conviction-weighted Beta update for one arm."""
    conv = conviction_of(proba)
    correct = (proba > 0.5) == bool(actual_up)
    # A near-coin-flip call carries almost no information about the arm, so it
    # barely moves the posterior. The floor keeps abstention-heavy arms from
    # never being updated at all.
    weight = max(0.05, conv)
    return ArmScore(
        arm=arm, proba=float(proba), correct=correct, conviction=conv,
        alpha_add=weight if correct else 0.0,
        beta_add=0.0 if correct else weight * CONFIDENT_WRONG_MULTIPLIER,
        reward=continuous_reward(proba, actual_up),
    )
