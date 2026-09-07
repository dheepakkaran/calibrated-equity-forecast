#!/usr/bin/env python
"""Hyperparameter and design search. Runs on the TUNE folds only.

Every configuration here was evaluated against folds 0-7. The held-out folds
are read once, at the end, by scripts/report.py - so the numbers in this file
are development numbers and are labelled as such.
"""
from __future__ import annotations

import argparse
import logging
import sys
import warnings

import pandas as pd

from cef.config import DATA_DIR
from cef.validation.runner import run_walk_forward, summarise
from cef.validation.selective import coverage_curve

warnings.filterwarnings("ignore")

# --- the search space --------------------------------------------------------
EXPERIMENTS: dict[str, dict] = {
    "base_200feat":      dict(),
    "sel_100feat":       dict(n_features=100),
    "sel_70feat":        dict(n_features=70),
    "sel_40feat":        dict(n_features=40),
    "looser_trees":      dict(lgb_params=dict(num_leaves=31, max_depth=6, min_child_samples=100)),
    "tighter_trees":     dict(lgb_params=dict(num_leaves=7, max_depth=3, min_child_samples=600,
                                              reg_lambda=20.0)),
    "slower_more":       dict(lgb_params=dict(learning_rate=0.008, n_estimators=3000)),
    "sel70_tighter":     dict(n_features=70,
                              lgb_params=dict(num_leaves=7, max_depth=3,
                                              min_child_samples=600, reg_lambda=20.0)),
    "sel70_lowcols":     dict(n_features=70, lgb_params=dict(colsample_bytree=0.3)),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", default="y_rel")
    ap.add_argument("--only", nargs="*", help="run a subset of experiment names")
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING)
    panel = pd.read_parquet(DATA_DIR / "panel.parquet")

    names = args.only or list(EXPERIMENTS)
    rows, curves = [], {}
    for name in names:
        kw = EXPERIMENTS[name]
        m, p = run_walk_forward(panel, target=args.target, split="tune",
                                persist=False, **kw)
        s = summarise(m)
        lg = p[p["model"] == "lightgbm"]
        cc = coverage_curve(lg)
        curves[name] = cc
        rows.append({
            "experiment": name,
            "accuracy": s.loc["lightgbm", "accuracy"],
            "auc": s.loc["lightgbm", "auc"],
            "edge_pp": s.loc["lightgbm", "edge_pp"],
            "edge_t": s.loc["lightgbm", "edge_t"],
            "folds_won": s.loc["lightgbm", "folds_won"],
            "ece": s.loc["lightgbm", "ece"],
            "acc@10%": cc.loc[cc["coverage"] == 0.10, "accuracy"].iloc[0],
            "edge@10%": cc.loc[cc["coverage"] == 0.10, "vs_selected"].iloc[0],
            "t@10%": cc.loc[cc["coverage"] == 0.10, "t_stat"].iloc[0],
        })
        print(f"  {name:18s} acc {rows[-1]['accuracy']:.4f}  auc {rows[-1]['auc']:.4f}  "
              f"edge {rows[-1]['edge_pp']:+.3f}pp (t={rows[-1]['edge_t']:.2f})  "
              f"acc@10% {rows[-1]['acc@10%']:.4f}", flush=True)

    res = pd.DataFrame(rows).sort_values("auc", ascending=False)
    print()
    print("=" * 100)
    print(f"TUNING RESULTS  target={args.target}  folds 0-7 (development set)")
    print("=" * 100)
    print(res.round(4).to_string(index=False))
    res.to_csv(DATA_DIR / f"tuning_{args.target}.csv", index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
