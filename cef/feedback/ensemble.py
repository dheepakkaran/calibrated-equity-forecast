"""The ensemble the bandit weights over.

Four arms, chosen because M1 measured them as genuinely different rather than
because four sounds like a good number:

``lightgbm``      raw GBM output. Best AUC on the held-out folds (0.515).
``lightgbm_cal``  the same model through isotonic calibration. Nearly identical
                  accuracy, materially better ECE - a different trade, not a
                  strict improvement, which is why both are arms.
``logistic_cal``  L2 logistic regression, calibrated. The interpretable floor.
                  Lower AUC, but it fails differently, and an ensemble wants
                  arms whose errors are not the same errors.
``reversal``      the two-parameter rule: P(out-performs tomorrow | out-
                  performed today) = 0.485 versus 0.508 the other way. On the
                  held-out folds this matched the GBM's accuracy. Including it
                  is a standing honesty check - if the bandit ends up loading
                  onto it, the machine learning is not earning its place.

The whole ensemble is fitted once per as-of date on a single training window,
using the same chronological fit/validate/calibrate split as a walk-forward
fold, so nothing here sees data it would not have had in production.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from cef.features.build import feature_columns
from cef.models.baselines import Persistence
from cef.models.calibration import IsotonicCalibrator
from cef.models.direction import LightGBMDirection, LogisticDirection
from cef.models.selection import select_features
from cef.validation.walk_forward import three_way_split

log = logging.getLogger(__name__)


@dataclass
class Ensemble:
    target: str = "y_rel"
    n_features: int = 70
    features_: list[str] = field(default_factory=list)
    n_train_: int = 0
    trained_through_: str = ""

    def fit(self, panel: pd.DataFrame, as_of: pd.Timestamp) -> "Ensemble":
        train = panel[(panel["date"] < as_of) & panel[self.target].notna()]
        if train.empty:
            raise ValueError(f"no training rows before {as_of}")
        feats_all = feature_columns(panel)
        fit, val, cal = three_way_split(train)

        probe = LightGBMDirection().fit(fit[feats_all], fit[self.target],
                                        val[feats_all], val[self.target])
        self.features_ = select_features(probe, fit[feats_all], top_k=self.n_features)

        self.gbm_ = LightGBMDirection().fit(fit[self.features_], fit[self.target],
                                            val[self.features_], val[self.target])
        self.gbm_cal_ = IsotonicCalibrator().fit(
            self.gbm_.predict_proba(cal[self.features_]), cal[self.target])

        self.logit_ = LogisticDirection().fit(fit[self.features_], fit[self.target])
        self.logit_cal_ = IsotonicCalibrator().fit(
            self.logit_.predict_proba(cal[self.features_]), cal[self.target])

        # Fitted on the whole train window: it has two parameters and no
        # capacity to overfit, so withholding data from it would only make the
        # honesty check weaker.
        self.reversal_ = Persistence().fit(train[feats_all], train[self.target])

        self.n_train_ = len(train)
        self.trained_through_ = str(train["date"].max().date())
        log.info("ensemble fitted on %d rows through %s, %d features",
                 self.n_train_, self.trained_through_, len(self.features_))
        return self

    def arm_probas(self, row: pd.DataFrame) -> dict[str, float]:
        raw_gbm = float(self.gbm_.predict_proba(row[self.features_])[0])
        raw_logit = float(self.logit_.predict_proba(row[self.features_])[0])
        return {
            "lightgbm": raw_gbm,
            "lightgbm_cal": float(self.gbm_cal_.transform(np.array([raw_gbm]))[0]),
            "logistic_cal": float(self.logit_cal_.transform(np.array([raw_logit]))[0]),
            "reversal": float(np.asarray(self.reversal_.predict_proba(row)).ravel()[0]),
        }

    def shap_row(self, row: pd.DataFrame) -> pd.Series:
        import shap

        expl = shap.TreeExplainer(self.gbm_.model_)
        vals = expl.shap_values(row[self.features_], check_additivity=False)
        if isinstance(vals, list):
            vals = vals[1]
        arr = np.asarray(vals)
        if arr.ndim == 3:
            arr = arr[:, :, -1]
        return pd.Series(arr[0], index=self.features_)
