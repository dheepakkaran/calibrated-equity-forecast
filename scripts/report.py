#!/usr/bin/env python
"""Final held-out assessment. Reads folds 8+ and writes the report.

Configuration was frozen on the tuning folds before this script was first run:

    target       y_rel   (1-session market-relative direction)
    features     top 70 by mean |SHAP|, reselected inside every fold
    model        LightGBM, params in cef.models.direction.LGB_PARAMS
    calibration  isotonic, fitted on the last 15% of each train window

Whatever comes out is the number. There is no branch in this file that
depends on the result.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings

import numpy as np
import pandas as pd

from cef.config import ARTIFACT_DIR, DATA_DIR
from cef.models.calibration import expected_calibration_error, reliability_curve
from cef.validation.runner import run_walk_forward, summarise
from cef.validation.selective import coverage_curve

warnings.filterwarnings("ignore")

CONFIG = dict(target="y_rel", n_features=70, calibrate=True)


def regime_breakdown(preds: pd.DataFrame, panel: pd.DataFrame, model: str) -> pd.DataFrame:
    p = preds[preds["model"] == model].copy()
    p["date"] = pd.to_datetime(p["date"])
    j = p.merge(panel[["symbol", "date", "vol_bucket", "trend_bucket"]],
                on=["symbol", "date"], how="left")
    rows = []
    for col in ("vol_bucket", "trend_bucket"):
        for val, g in j.groupby(col, observed=True):
            if len(g) < 500:
                continue
            rows.append({
                "dimension": col, "regime": val, "n": len(g),
                "base_rate": g["y_true"].mean(),
                "accuracy": ((g["proba"] > 0.5) == g["y_true"].astype(bool)).mean(),
                "ece": expected_calibration_error(g["proba"].to_numpy(), g["y_true"].to_numpy()),
            })
    d = pd.DataFrame(rows)
    d["edge_pp"] = (d["accuracy"] - d["base_rate"].clip(lower=0.5)) * 100
    return d.round(4)


def plot_reliability(preds: pd.DataFrame, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), facecolor="#16202B")
    for ax, model, title in ((axes[0], "lightgbm", "Raw model output"),
                             (axes[1], "lightgbm_cal", "After isotonic calibration")):
        g = preds[preds["model"] == model]
        rows = reliability_curve(g["proba"].to_numpy(), g["y_true"].to_numpy(), 10)
        x = [r["pred_mean"] for r in rows]
        y = [r["actual_rate"] for r in rows]
        # Pooled across folds, which is the curve a reader wants to see. It
        # is milder than the per-fold mean ECE in the summary table, because
        # errors in opposite directions cancel when folds are combined - the
        # summary's figure is the conservative one and is the headline.
        ece = expected_calibration_error(g["proba"].to_numpy(), g["y_true"].to_numpy())

        ax.set_facecolor("#16202B")
        lo, hi = min(x + y) - 0.02, max(x + y) + 0.02
        ax.plot([lo, hi], [lo, hi], "--", color="#647889", lw=1, label="perfect")
        ax.plot(x, y, "o-", color="#5FA8A0", lw=1.8, ms=5, label=model)
        ax.axhline(0.5, color="#2E3F50", lw=0.8)
        ax.axvline(0.5, color="#2E3F50", lw=0.8)
        ax.set_title(f"{title}\npooled ECE {ece:.2f} pp", color="#E8EFF5", fontsize=11)
        ax.set_xlabel("predicted probability", color="#9BAEC0", fontsize=9)
        ax.set_ylabel("observed frequency", color="#9BAEC0", fontsize=9)
        ax.tick_params(colors="#647889", labelsize=8)
        for s in ax.spines.values():
            s.set_color("#2E3F50")
        ax.legend(fontsize=8, facecolor="#1C2833", edgecolor="#2E3F50",
                  labelcolor="#9BAEC0")
    fig.suptitle("Reliability — held-out folds, 1-session market-relative direction",
                 color="#E8EFF5", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, facecolor="#16202B")
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="holdout", choices=["holdout", "tune", "all"])
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                        datefmt="%H:%M:%S")
    panel = pd.read_parquet(DATA_DIR / "panel.parquet")

    metrics, preds = run_walk_forward(panel, split=args.split, **CONFIG)
    summary = summarise(metrics)
    lg = preds[preds["model"] == "lightgbm"]
    cov = coverage_curve(lg)
    reg = regime_breakdown(preds, panel, "lightgbm")
    png = plot_reliability(preds, ARTIFACT_DIR / f"reliability_{args.split}.png")

    print()
    print("=" * 92)
    print(f"HELD-OUT ASSESSMENT   split={args.split}   target={CONFIG['target']}   "
          f"features={CONFIG['n_features']}")
    print("=" * 92)
    print(summary.to_string())
    print()
    print("--- accuracy vs coverage (selective prediction) ---")
    print(cov.to_string(index=False))
    print()
    print("--- per-regime, lightgbm ---")
    print(reg.to_string(index=False))
    print()
    print(f"reliability diagram -> {png}")

    out = {
        "split": args.split, "config": CONFIG,
        "folds": sorted(metrics["fold"].unique().tolist()),
        "summary": json.loads(summary.reset_index().to_json(orient="records")),
        "coverage": json.loads(cov.to_json(orient="records")),
        "regimes": json.loads(reg.to_json(orient="records")),
    }
    path = ARTIFACT_DIR / f"report_{args.split}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"machine-readable  -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
