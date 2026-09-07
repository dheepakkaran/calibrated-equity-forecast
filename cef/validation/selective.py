"""Selective-prediction metrics: accuracy as a function of coverage.

Accuracy at a 0.5 threshold is the wrong headline for this problem and the
decile analysis shows why. The model's probabilities cluster tightly around the
base rate, so the 0.5 cut lands in the region where it has no information, and
a genuine edge in the tails is averaged away against noise in the middle.

What the model actually produces is a *ranking*. The metric that respects that
is accuracy at a chosen coverage: sort predictions by distance from the base
rate, act on the most extreme ``c`` fraction, abstain on the rest, and measure
the hit rate on what was acted upon. A system that declines to forecast on 90%
of sessions and is right 55% of the time on the other 10% is more useful, and
far easier to defend, than one that forecasts every session at 51.7%.

Every number here is computed within a fold and then averaged across folds, so
a fold with an unusual base rate cannot distort the aggregate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

COVERAGES = (0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.00)


def selective_accuracy(proba: np.ndarray, y: np.ndarray, coverage: float,
                       reference: float | None = None) -> dict:
    """Act on the most extreme ``coverage`` fraction of predictions.

    ``reference`` is the point from which 'extremeness' is measured. It
    defaults to the mean prediction rather than 0.5, because a model whose
    outputs all sit near a base rate of 0.51 would otherwise have its entire
    ranking read as bullish.
    """
    proba, y = np.asarray(proba, float), np.asarray(y, float)
    ref = float(np.mean(proba)) if reference is None else reference
    k = max(1, int(round(len(proba) * coverage)))
    conviction = np.abs(proba - ref)
    take = np.argsort(-conviction)[:k]

    p_take, y_take = proba[take], y[take]
    called_up = p_take > ref
    hits = np.where(called_up, y_take, 1 - y_take)

    majority_up = y.mean() > 0.5
    return {
        "coverage": coverage,
        "n": int(k),
        "accuracy": float(hits.mean()),
        # Two comparators, because they answer different questions and only
        # reporting the flattering one is how 55% gets oversold.
        #
        # `baseline_global` is always-predict-majority over the *whole* test
        # slice. Beating it means the model found a subset of sessions where
        # being right is easier - real, useful, but it is timing, not
        # direction.
        #
        # `baseline_selected` is always-predict-majority on the *same selected
        # rows*. Beating it means the model's directional call added something
        # beyond the drift on those rows. This is the strict test of skill, and
        # it is the one that matters for a product that shows a bull/bear
        # verdict.
        "baseline_global": float(y.mean() if majority_up else 1 - y.mean()),
        "baseline_selected": float(np.where(majority_up, y_take, 1 - y_take).mean()),
        "frac_up_calls": float(called_up.mean()),
    }


def coverage_curve(preds: pd.DataFrame, coverages=COVERAGES) -> pd.DataFrame:
    """Coverage curve averaged across folds. ``preds`` needs fold/proba/y_true."""
    rows = []
    for cov in coverages:
        per_fold = [
            selective_accuracy(g["proba"].to_numpy(), g["y_true"].to_numpy(), cov)
            for _, g in preds.groupby("fold")
        ]
        acc = np.array([r["accuracy"] for r in per_fold])
        base_g = np.array([r["baseline_global"] for r in per_fold])
        base_s = np.array([r["baseline_selected"] for r in per_fold])
        edge = acc - base_s          # the strict edge drives the t-statistic
        edge_g = acc - base_g
        rows.append({
            "coverage": cov,
            "n_per_fold": int(np.mean([r["n"] for r in per_fold])),
            "accuracy": acc.mean(),
            "vs_global": edge_g.mean() * 100,
            "vs_selected": edge.mean() * 100,
            "frac_up": float(np.mean([r["frac_up_calls"] for r in per_fold])),
            "edge_pp": edge.mean() * 100,
            "edge_std_pp": edge.std(ddof=1) * 100,
            # Fold-level t-statistic on the edge. With 13 folds this is a
            # blunt instrument, but it is the right blunt instrument: it asks
            # whether the edge recurs across regimes rather than whether it
            # exists once when all rows are pooled.
            "t_stat": edge.mean() / (edge.std(ddof=1) / np.sqrt(len(edge))) if edge.std(ddof=1) > 0 else np.nan,
            "folds_pos": int((edge > 0).sum()),
            "n_folds": len(edge),
        })
    return pd.DataFrame(rows).round(4)
