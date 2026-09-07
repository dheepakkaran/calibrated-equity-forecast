"""Direction-head models: an interpretable floor and the GBM workhorse."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)


# Hyperparameters chosen for a low signal-to-noise tabular problem. The
# defaults would overfit badly here: ~100k rows carrying maybe a few percent of
# genuine directional signal rewards shallow trees, heavy row and column
# subsampling, large leaf minimums and strong L2. Depth is capped rather than
# left to num_leaves alone so a single fold cannot grow a deep memoriser.
LGB_PARAMS = dict(
    objective="binary",
    n_estimators=1200,
    learning_rate=0.015,
    num_leaves=15,
    max_depth=4,
    min_child_samples=300,
    subsample=0.7,
    subsample_freq=1,
    colsample_bytree=0.5,
    reg_lambda=5.0,
    reg_alpha=0.5,
    max_bin=127,
    verbose=-1,
    n_jobs=-1,
)


class LightGBMDirection:
    """Gradient-boosted direction classifier with early stopping on a
    chronologically later validation slice."""

    name = "lightgbm"

    def __init__(self, params: dict | None = None, seed: int = 42) -> None:
        self.params = {**LGB_PARAMS, **(params or {})}
        self.params["random_state"] = seed
        self.features_: list[str] = []
        self.best_iteration_: int | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series,
            X_val: pd.DataFrame | None = None, y_val: pd.Series | None = None):
        import lightgbm as lgb

        self.features_ = list(X.columns)
        self.model_ = lgb.LGBMClassifier(**self.params)
        kw = {}
        if X_val is not None and len(X_val):
            kw = dict(
                eval_X=X_val[self.features_], eval_y=y_val,
                eval_metric="binary_logloss",
                callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)],
            )
        self.model_.fit(X, y, **kw)
        self.best_iteration_ = getattr(self.model_, "best_iteration_", None)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict_proba(X[self.features_])[:, 1]

    def feature_importance(self) -> pd.Series:
        return pd.Series(
            self.model_.booster_.feature_importance(importance_type="gain"),
            index=self.features_).sort_values(ascending=False)


class LogisticDirection:
    """L2 logistic regression - the interpretable floor.

    Imputer and scaler live inside the pipeline, so ``fit`` touches only the
    rows it is handed. That is what keeps 'never fit a scaler on the full
    dataset' true by construction rather than by discipline.
    """

    name = "logistic"

    def __init__(self, C: float = 0.05) -> None:
        # VarianceThreshold guards the scaler: a feature that is constant
        # inside one fold's fit slice divides by ~0 and sends the solver to
        # inf. Dropping it is correct - a constant column carries no signal.
        self.pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("drop_const", VarianceThreshold(threshold=1e-12)),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=C, penalty="l2", max_iter=3000, solver="lbfgs")),
        ])
        self.features_: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series, X_val=None, y_val=None):
        self.features_ = list(X.columns)
        self.pipe.fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipe.predict_proba(X[self.features_])[:, 1]

    def coefficients(self) -> pd.Series:
        kept = self.pipe["drop_const"].get_support()
        return pd.Series(self.pipe["clf"].coef_[0],
                         index=[f for f, k in zip(self.features_, kept) if k])
