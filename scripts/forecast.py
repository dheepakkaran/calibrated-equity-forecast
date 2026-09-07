#!/usr/bin/env python
"""Produce a forecast for one symbol, with evidence and plain-language narration.

The forecast target is **market-relative** direction: does this symbol
out-perform the cross-section of the NSE next session? That is a deliberate
product decision, not a technicality. M1 established that absolute
close-to-close direction is not learnable from this feature set - the label is
mostly a restatement of "was the index up", and a model trained on it settles
on emitting the base rate. The relative question is the one the features can
answer, so it is the one the system asks, and every reader-facing string says
so explicitly.
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
from cef.db import connect
from cef.evidence.aspects import bucket_shap
from cef.evidence.attribution import confidence_band
from cef.features.build import feature_columns
from cef.models.calibration import IsotonicCalibrator
from cef.models.direction import LightGBMDirection
from cef.models.selection import select_features
from cef.validation.walk_forward import three_way_split

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)


def fit_production(panel: pd.DataFrame, target: str, n_features: int, as_of: pd.Timestamp):
    """Train exactly as a fold is trained, on everything up to ``as_of``."""
    train = panel[(panel["date"] < as_of) & panel[target].notna()]
    feats_all = feature_columns(panel)
    fit, val, cal = three_way_split(train)

    probe = LightGBMDirection().fit(fit[feats_all], fit[target], val[feats_all], val[target])
    feats = select_features(probe, fit[feats_all], top_k=n_features)

    model = LightGBMDirection().fit(fit[feats], fit[target], val[feats], val[target])
    calib = IsotonicCalibrator().fit(model.predict_proba(cal[feats]), cal[target])
    return model, calib, feats, len(train)


def recent_attributions(symbol: str, limit: int = 4) -> list[dict]:
    with connect() as conn:
        df = pd.read_sql(
            "SELECT date, ret, ret_rel, sigma, kind, headline, source_url, confidence, "
            "rationale FROM attributions WHERE symbol=? ORDER BY date DESC LIMIT ?",
            conn, params=[symbol, limit])
    df["band"] = df["confidence"].map(confidence_band)
    return json.loads(df.to_json(orient="records"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("symbol")
    ap.add_argument("--target", default="y_rel")
    ap.add_argument("--n-features", type=int, default=70)
    ap.add_argument("--no-narrate", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                        datefmt="%H:%M:%S")
    panel = pd.read_parquet(DATA_DIR / "panel.parquet")
    sym = args.symbol.upper()
    if sym not in set(panel["symbol"]):
        print(f"{sym} is not in the panel")
        return 1

    latest = panel[panel["symbol"] == sym]["date"].max()
    row = panel[(panel["symbol"] == sym) & (panel["date"] == latest)]

    model, calib, feats, n_train = fit_production(panel, args.target, args.n_features, latest)
    raw = float(model.predict_proba(row[feats])[0])
    prob = float(calib.transform(np.array([raw]))[0])
    log.info("trained on %d rows up to %s", n_train, latest.date())

    # --- SHAP -> aspects ----------------------------------------------------
    import shap
    expl = shap.TreeExplainer(model.model_)
    vals = expl.shap_values(row[feats], check_additivity=False)
    if isinstance(vals, list):
        vals = vals[1]
    if np.asarray(vals).ndim == 3:
        vals = np.asarray(vals)[:, :, -1]
    shap_row = pd.Series(np.asarray(vals)[0], index=feats)
    aspects = bucket_shap(shap_row, row[feats].iloc[0])

    direction = "outperform" if prob > 0.5 else "underperform"
    close = float(row["close"].iloc[0])
    atr = float(row["atr_pct"].iloc[0]) * close

    evidence = {
        "symbol": sym,
        "forecast_for": "next trading session",
        "as_of_close": latest.strftime("%Y-%m-%d"),
        "last_close": round(close, 2),
        "question": ("Will this share out-perform or under-perform the median "
                     "NSE large-cap next session?"),
        "direction": direction,
        "confidence": round(max(prob, 1 - prob), 3),
        "confidence_note": ("near coin-flip" if max(prob, 1 - prob) < 0.55
                            else "modest" if max(prob, 1 - prob) < 0.65 else "firm"),
        "model_track_record": {
            "held_out_accuracy": 0.5102, "coin_flip_baseline": 0.5001,
            "edge_pp": 1.01, "fold_t_stat": 3.01, "folds_won": "5 of 5",
        },
        # Points, not raw SHAP: see cef/evidence/aspects.POINTS_SCALE.
        "aspects": [{"aspect": a["aspect"], "points": a["points"],
                     "direction": a["direction"]}
                    for a in aspects[:5] if a["points"] != 0],
        "typical_daily_range_rupees": round(atr, 2),
        "recent_explained_moves": recent_attributions(sym),
    }

    print()
    print("=" * 76)
    print(f"{sym}  ·  forecast for the session after {latest.date()}")
    print("=" * 76)
    print(f"question    {evidence['question']}")
    print(f"call        {direction.upper()}   confidence {evidence['confidence']:.1%} "
          f"({evidence['confidence_note']})")
    print(f"raw -> cal  {raw:.4f} -> {prob:.4f}")
    print(f"last close  Rs {close:,.2f}   typical daily range Rs {atr:,.2f}")
    print()
    print("aspect contributions (SHAP, bucketed):")
    for a in aspects:
        if abs(a["contribution"]) < 1e-6:
            continue
        top = ", ".join(f"{e['feature']}={e.get('value', '?')}" for e in a["evidence"])
        print(f"  {a['aspect']:18s} {a['points']:+4d} pts   (raw {a['contribution']:+.4f}, "
              f"{a['share_of_signal']:.0%} of signal)")
        print(f"    {top}")
    print()
    print("recent significant moves, with evidence:")
    for m in evidence["recent_explained_moves"]:
        print(f"  {m['date']}  ret {m['ret']:+.3f} ({m['sigma']:.1f}s)  "
              f"{m['kind']}/{m['band']}  {str(m['headline'] or '-')[:70]}")

    if not args.no_narrate:
        from cef.evidence.narrate import narrate
        out = narrate(evidence)
        print()
        print("-" * 76)
        print("PLAIN LANGUAGE" + ("  (cached)" if out["cached"] else
                                  f"  ({out['model']}, {out['completion_tokens']} tokens)"))
        print("-" * 76)
        print(out["text"])
        if out["unverified_numbers"]:
            print(f"\n!! numbers not present in the evidence: {out['unverified_numbers']}")
        else:
            print("\n[verified: every number in the text appears in the evidence]")

    path = ARTIFACT_DIR / f"forecast_{sym}_{latest.strftime('%Y%m%d')}.json"
    path.write_text(json.dumps(evidence, indent=2, default=str))
    print(f"\nevidence -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
