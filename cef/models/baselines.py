"""Naive baselines, logged permanently alongside every model.

The point of this module is defensive. Directional equity forecasting has a
base rate near 51%, so a model reporting 54% accuracy sounds impressive and
means very little until it is placed next to the number a coin, or a constant,
or yesterday's sign would have produced on the identical test slice. These
baselines are therefore not a warm-up exercise to be deleted once the real
model works - they are refit on every fold, written to the same metrics table,
and rendered beside every accuracy figure the system displays.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class Baseline:
    name: str = "baseline"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "Baseline":
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError


class AlwaysUp(Baseline):
    """Predict 'up' every session, with the train-window base rate as its
    confidence. On Indian equities this is the baseline that is hardest to
    beat, because the unconditional drift is genuinely positive."""

    name = "always_up"

    def fit(self, X, y):
        self.p_ = float(y.mean())
        return self

    def predict_proba(self, X):
        return np.full(len(X), self.p_)


class Random(Baseline):
    """Zero information. Brier 0.25, accuracy 50%, AUC 0.5 by construction."""

    name = "random"

    def predict_proba(self, X):
        return np.full(len(X), 0.5)


class Persistence(Baseline):
    """Tomorrow repeats today's sign - the momentum straw man.

    Maps today's return to a probability using the train-window frequency of
    'the sign repeats', so it produces calibrated probabilities rather than
    hard 0/1 calls and can be scored on Brier alongside everything else.
    """

    name = "persistence"

    def fit(self, X, y):
        up_today = X["ret_1d"] > 0
        self.p_up_ = float(y[up_today].mean()) if up_today.any() else 0.5
        self.p_dn_ = float(y[~up_today].mean()) if (~up_today).any() else 0.5
        return self

    def predict_proba(self, X):
        return np.where(X["ret_1d"] > 0, self.p_up_, self.p_dn_)


class MeanReversion(Baseline):
    """The mirror of persistence: today's sign inverts tomorrow. Included
    because on daily equity data it is often the better of the two, and which
    one wins is itself a regime diagnostic."""

    name = "mean_reversion"

    def fit(self, X, y):
        up_today = X["ret_1d"] > 0
        self.p_up_ = float(y[~up_today].mean()) if (~up_today).any() else 0.5
        self.p_dn_ = float(y[up_today].mean()) if up_today.any() else 0.5
        return self

    def predict_proba(self, X):
        return np.where(X["ret_1d"] > 0, self.p_dn_, self.p_up_)


def all_baselines() -> list[Baseline]:
    return [AlwaysUp(), Random(), Persistence(), MeanReversion()]
