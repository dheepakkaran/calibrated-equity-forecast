"""Probability calibration.

Gradient-boosted classifiers on noisy financial data produce scores that rank
reasonably but are badly scaled - they push mass toward the extremes, so a raw
output of 0.70 might be right only 55% of the time. Since this project displays
confidence to a user as a headline number, that scaling error is the difference
between an honest interface and a misleading one.

Isotonic regression is used rather than Platt scaling because the distortion is
not reliably sigmoid-shaped, and the calibration fold here is large enough
(thousands of rows) that isotonic's extra flexibility is not a liability.
"""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression


class IsotonicCalibrator:
    def __init__(self, clip: float = 0.02) -> None:
        # Never assert more certainty than the data can support, and never
        # emit a hard 0 or 1 - log-loss is unbounded there.
        self.clip = clip
        self.iso_ = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)

    def fit(self, proba: np.ndarray, y: np.ndarray) -> "IsotonicCalibrator":
        self.iso_.fit(np.asarray(proba, dtype=float), np.asarray(y, dtype=float))
        return self

    def transform(self, proba: np.ndarray) -> np.ndarray:
        out = self.iso_.transform(np.asarray(proba, dtype=float))
        return np.clip(out, self.clip, 1 - self.clip)


def reliability_curve(proba: np.ndarray, y: np.ndarray, n_bins: int = 10):
    """Equal-count bins rather than equal-width. Predictions cluster tightly
    around 0.5 on this problem, so fixed-width bins leave the tails empty and
    the curve unreadable."""
    proba, y = np.asarray(proba, float), np.asarray(y, float)
    order = np.argsort(proba)
    bins = np.array_split(order, n_bins)
    rows = []
    for b in bins:
        if len(b) == 0:
            continue
        rows.append({
            "n": len(b),
            "pred_mean": float(proba[b].mean()),
            "actual_rate": float(y[b].mean()),
        })
    return rows


def expected_calibration_error(proba: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """ECE: mean |predicted - actual| across bins, weighted by bin size.
    Reported in percentage points because that is how it is read."""
    rows = reliability_curve(proba, y, n_bins)
    total = sum(r["n"] for r in rows)
    if not total:
        return float("nan")
    return 100.0 * sum(r["n"] * abs(r["pred_mean"] - r["actual_rate"]) for r in rows) / total
