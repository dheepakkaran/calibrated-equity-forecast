"""Expanding-window walk-forward validation.

Three properties make this harness correct, and each maps to a rule from the
proposal's section 7.4:

*Chronology.* Folds are cut by date. No shuffled split exists anywhere in this
file, so a training row is always older than every row it is evaluated on.

*Embargo.* The direction label at date t is drawn from close[t+1]. A training
row dated one session before a test window would therefore carry a label taken
from inside that window. Samples within ``embargo_days`` of the test start are
purged, which is the minimal correct version of Lopez de Prado's purging for a
one-step-ahead label.

*In-fold fitting.* The train window is split chronologically three ways -
fit / early-stop / calibrate. Imputers, scalers, tree ensembles and the
isotonic calibrator are all fitted strictly inside their own slice. Nothing is
fitted on the pooled dataset, so no test-period statistic can reach a
transform.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from cef.config import WalkForwardConfig

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Fold:
    index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp     # inclusive, already embargoed
    test_start: pd.Timestamp
    test_end: pd.Timestamp      # exclusive

    def label(self) -> str:
        return (f"fold {self.index}  train {self.train_start.date()}->{self.train_end.date()}"
                f"  test {self.test_start.date()}->{self.test_end.date()}")


def generate_folds(dates: pd.Series, cfg: WalkForwardConfig,
                   horizon: int = 1) -> list[Fold]:
    """Cut the panel into expanding-window folds with fixed test slices.

    ``horizon`` sets the embargo floor and must match the label being used. A
    label measured over h forward sessions at date t spans [t, t+h], so a
    training row within h sessions of the test start carries an outcome drawn
    from inside the test window. With h=20 and a 2-day embargo, four weeks of
    the test period leak into training - enough to manufacture a result out of
    nothing. The embargo is therefore max(configured, horizon + 1), in calendar
    days scaled for weekends.
    """
    dmin, dmax = pd.Timestamp(dates.min()), pd.Timestamp(dates.max())
    folds: list[Fold] = []
    test_start = pd.Timestamp(cfg.first_test_start)
    idx = 0
    # ~1.45 calendar days per trading session once weekends and holidays are
    # counted; rounding up keeps the embargo conservative.
    embargo = max(cfg.embargo_days, int(np.ceil((horizon + 1) * 1.45)))
    while test_start < dmax:
        test_end = test_start + relativedelta(months=cfg.test_months)
        train_end = test_start - pd.Timedelta(days=embargo + 1)
        n_train_days = (train_end - dmin).days
        if n_train_days >= cfg.min_train_days:
            folds.append(Fold(idx, dmin, train_end, test_start, min(test_end, dmax + pd.Timedelta(days=1))))
            idx += 1
        test_start = test_end
    return folds


def three_way_split(train: pd.DataFrame, val_frac: float = 0.15, cal_frac: float = 0.15):
    """Chronological fit / early-stop / calibrate split of one train window.

    Split points are chosen on the *date* axis, not the row axis, so all 52
    symbols move between slices together. Splitting by row would place the same
    session in two different slices for different symbols and let a
    cross-sectional feature carry information across the boundary.
    """
    days = np.sort(train["date"].unique())
    n = len(days)
    i_val = int(n * (1 - val_frac - cal_frac))
    i_cal = int(n * (1 - cal_frac))
    d_val, d_cal = days[i_val], days[i_cal]
    fit = train[train["date"] < d_val]
    val = train[(train["date"] >= d_val) & (train["date"] < d_cal)]
    cal = train[train["date"] >= d_cal]
    return fit, val, cal


def split_fold(panel: pd.DataFrame, fold: Fold):
    train = panel[(panel["date"] >= fold.train_start) & (panel["date"] <= fold.train_end)]
    test = panel[(panel["date"] >= fold.test_start) & (panel["date"] < fold.test_end)]
    return train, test


def describe_folds(panel: pd.DataFrame, folds: list[Fold]) -> pd.DataFrame:
    rows = []
    for f in folds:
        tr, te = split_fold(panel, f)
        rows.append({
            "fold": f.index,
            "train_end": f.train_end.date(),
            "test_start": f.test_start.date(),
            "test_end": (f.test_end - pd.Timedelta(days=1)).date(),
            "n_train": len(tr),
            "n_test": len(te),
            "test_base_rate": round(te["y_dir"].mean(), 4) if len(te) else np.nan,
            "test_vol": round(te["ret_1d"].std(), 4) if len(te) else np.nan,
        })
    return pd.DataFrame(rows)
