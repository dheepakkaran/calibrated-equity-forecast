"""The walk-forward experiment runner - where every reported number is made.

One rule governs this file: **tuning folds and reporting folds never overlap.**

Choosing hyperparameters, a feature count, a calibrator or a target by looking
at walk-forward output and then quoting that same output as the result is the
most common way a project like this ends up reporting a number it cannot
reproduce. So the fold sequence is cut in two. Early folds are the development
set - anything may be tried there, as often as needed. Late folds are touched
only to produce a final figure, and every time they are read the count is
logged. The split is chronological, which also means the held-out assessment is
the most recent market, not a random slice of it.
"""
from __future__ import annotations

import logging
import re
import uuid

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

from cef.config import WALK_FORWARD
from cef.db import upsert, utcnow
from cef.features.build import feature_columns
from cef.models.baselines import all_baselines
from cef.models.calibration import IsotonicCalibrator, expected_calibration_error
from cef.models.direction import LightGBMDirection, LogisticDirection
from cef.models.selection import select_features
from cef.validation.walk_forward import generate_folds, split_fold, three_way_split

log = logging.getLogger(__name__)

MIN_TEST_ROWS = 1000
N_TUNE_FOLDS = 8          # folds 0-7 are the development set
DEFAULT_TARGET = "y_rel"


def score(y: np.ndarray, p: np.ndarray) -> dict:
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    out = {
        "accuracy": float(accuracy_score(y, (p > 0.5).astype(int))),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
    }
    out["auc"] = float(roc_auc_score(y, p)) if len(np.unique(p)) > 1 else float("nan")
    return out


def target_horizon(target: str) -> int:
    """Forward-session span implied by a target name, for the embargo."""
    if target in ("y_dir", "y_rel", "y_gap"):
        return 1
    m = re.search(r"_(\d+)d$", target)
    return int(m.group(1)) if m else 1


def usable_folds(panel: pd.DataFrame, horizon: int = 1) -> list:
    return [f for f in generate_folds(panel["date"], WALK_FORWARD, horizon)
            if len(split_fold(panel, f)[1]) >= MIN_TEST_ROWS]


def run_walk_forward(
    panel: pd.DataFrame,
    run_id: str | None = None,
    target: str = DEFAULT_TARGET,
    split: str = "tune",              # "tune" | "holdout" | "all"
    calibrate: bool = True,
    n_features: int | None = None,    # None = use every feature
    lgb_params: dict | None = None,
    persist: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_id = run_id or uuid.uuid4().hex[:12]
    feats_all = feature_columns(panel)
    panel = panel[panel[target].notna()]

    horizon = target_horizon(target)
    folds = usable_folds(panel, horizon)
    if split == "tune":
        folds = folds[:N_TUNE_FOLDS]
    elif split == "holdout":
        folds = folds[N_TUNE_FOLDS:]
        log.warning("READING THE HELD-OUT FOLDS (%d) - this is a reportable run",
                    len(folds))

    log.info("run %s | target=%s (h=%d) | split=%s | %d folds | %d features%s",
             run_id, target, horizon, split, len(folds), len(feats_all),
             f" -> top {n_features}" if n_features else "")

    metric_rows, pred_rows = [], []

    for fold in folds:
        train, test = split_fold(panel, fold)
        fit, val, cal = three_way_split(train)
        y_test = test[target].to_numpy(float)
        feats = feats_all
        models: dict[str, np.ndarray] = {}

        for b in all_baselines():
            b.fit(train[feats], train[target])
            models[b.name] = np.asarray(b.predict_proba(test[feats]), float)

        # --- feature selection, inside this fold only ------------------------
        if n_features:
            probe = LightGBMDirection(lgb_params).fit(
                fit[feats_all], fit[target], val[feats_all], val[target])
            feats = select_features(probe, fit[feats_all], top_k=n_features)

        for mdl in (LogisticDirection(), LightGBMDirection(lgb_params)):
            mdl.fit(fit[feats], fit[target], val[feats], val[target])
            raw = mdl.predict_proba(test[feats])
            models[mdl.name] = raw
            if calibrate:
                c = IsotonicCalibrator().fit(mdl.predict_proba(cal[feats]), cal[target])
                models[f"{mdl.name}_cal"] = c.transform(raw)

        for name, p in models.items():
            m = score(y_test, p)
            m["ece"] = expected_calibration_error(p, y_test)
            metric_rows.append({
                "run_id": run_id, "fold": fold.index, "model": name,
                "train_start": str(fold.train_start.date()), "train_end": str(fold.train_end.date()),
                "test_start": str(fold.test_start.date()), "test_end": str(fold.test_end.date()),
                "n_train": len(fit), "n_test": len(test), "created_at": utcnow(), **m,
            })
            pred_rows.append(pd.DataFrame({
                "run_id": run_id, "model": name, "symbol": test["symbol"].values,
                "date": test["date"].dt.strftime("%Y-%m-%d").values,
                "fold": fold.index, "proba": p, "y_true": y_test.astype(int),
            }))

        log.info("fold %2d  test %s->%s  base %.4f | lgbm %.4f | logit %.4f | n_feat %d",
                 fold.index, fold.test_start.date(), (fold.test_end - pd.Timedelta(days=1)).date(),
                 y_test.mean(), score(y_test, models["lightgbm"])["accuracy"],
                 score(y_test, models["logistic"])["accuracy"], len(feats))

    metrics = pd.DataFrame(metric_rows)
    preds = pd.concat(pred_rows, ignore_index=True)

    if persist:
        db_cols = ["run_id", "fold", "model", "train_start", "train_end", "test_start",
                   "test_end", "n_train", "n_test", "accuracy", "auc", "brier",
                   "log_loss", "created_at"]
        upsert("wf_metrics", metrics[db_cols], ["run_id", "fold", "model"])
        upsert("wf_predictions", preds, ["run_id", "model", "symbol", "date"])
    return metrics, preds


def summarise(metrics: pd.DataFrame) -> pd.DataFrame:
    """Equal-weighted across folds, with a fold-level t-statistic on the edge
    over the always-up baseline. Pooling rows instead would let one large or
    one unusually easy test slice carry the headline."""
    agg = (metrics.groupby("model")
           .agg(accuracy=("accuracy", "mean"), acc_std=("accuracy", "std"),
                auc=("auc", "mean"), brier=("brier", "mean"),
                log_loss=("log_loss", "mean"), ece=("ece", "mean"),
                folds=("fold", "nunique"))
           .sort_values("accuracy", ascending=False))

    wide = metrics.pivot_table(index="fold", columns="model", values="accuracy")
    if "always_up" in wide.columns:
        for model in agg.index:
            d = (wide[model] - wide["always_up"]).dropna()
            agg.loc[model, "edge_pp"] = d.mean() * 100
            agg.loc[model, "edge_t"] = (d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
                                        if len(d) > 1 and d.std(ddof=1) > 0 else np.nan)
            agg.loc[model, "folds_won"] = int((d > 0).sum())
    return agg.round(4)
