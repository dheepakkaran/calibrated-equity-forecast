"""Panel assembly and target construction.

The target is the only place in this codebase where a negative shift appears:

    y[t] = 1 if close[t+1] > close[t] else 0

Everything in the feature blocks is a trailing window. That asymmetry is the
design - one forward shift, in one function, on the label only - and
``tests/test_no_leakage.py`` asserts it holds by scanning for stray negative
shifts elsewhere.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from cef.db import read_macro, read_ohlcv
from cef.features.breaks import detect_breaks
from cef.features.cross_sectional import cross_sectional_features
from cef.features.macro import macro_features
from cef.features.technical import technical_features
from cef.universe import MACRO_SERIES, SECTOR

log = logging.getLogger(__name__)

WARMUP_ROWS = 200      # sma200 and beta120 need this much history per symbol
HORIZONS = (1, 3, 5, 10, 20)

META_COLS = (["symbol", "date", "sector", "close", "target_date",
              "fwd_ret", "fwd_ret_rel", "y_dir", "y_gap", "y_rel",
              "regime", "vol_bucket", "trend_bucket"]
             + [f"fwd_ret_{h}d" for h in HORIZONS]
             + [f"fwd_ret_rel_{h}d" for h in HORIZONS]
             + [f"y_rel_{h}d" for h in HORIZONS]
             + [f"y_dir_{h}d" for h in HORIZONS])


def _lagged_macro_levels(days: pd.DatetimeIndex, macro_wide: pd.DataFrame) -> pd.DataFrame:
    cols = {}
    for spec in MACRO_SERIES:
        if spec.key in macro_wide.columns:
            cols[spec.key] = (macro_wide[spec.key]
                              .reindex(macro_wide.index.union(days))
                              .ffill().reindex(days).shift(spec.lag))
    return pd.DataFrame(cols, index=days)


def _targets(oh: pd.DataFrame) -> pd.DataFrame:
    """Attach next-session targets. The single forward-looking operation."""
    g = oh.sort_values(["symbol", "date"]).groupby("symbol", observed=True)
    out = oh.sort_values(["symbol", "date"]).copy()
    out["next_close"] = g["close"].shift(-1)
    out["next_open"] = g["open"].shift(-1)
    out["target_date"] = g["date"].shift(-1)
    out["fwd_ret"] = np.log(out["next_close"] / out["close"])
    out["y_dir"] = (out["next_close"] > out["close"]).astype("float")
    out["y_gap"] = (out["next_open"] > out["close"]).astype("float")
    # The final session of each symbol has no next bar; label undefined.
    out.loc[out["next_close"].isna(), ["y_dir", "y_gap", "fwd_ret"]] = np.nan
    return out.drop(columns=["next_close", "next_open"])


def _relative_target(panel: pd.DataFrame) -> pd.DataFrame:
    """Market-relative direction: does this symbol beat the cross-section?

    The absolute close-to-close label is dominated by a market-wide drift that
    no per-symbol feature can explain - on any given day roughly all 52 names
    move together, so `y_dir` is mostly a restatement of "was the index up".
    A model trained on it learns to predict the index, cannot, and settles on
    emitting the base rate.

    Subtracting the cross-sectional median forward return removes that common
    factor and leaves the part the features describe: relative strength,
    sector rotation, driver sensitivity, idiosyncratic momentum. The label is
    ~50/50 by construction, so there is no drift left to hide behind - any
    accuracy above 50% has to come from ranking the cross-section correctly.

    The median is taken across symbols within a single target date, so it uses
    no information from outside that session.
    """
    med = panel.groupby("target_date", observed=True)["fwd_ret"].transform("median")
    panel["fwd_ret_rel"] = panel["fwd_ret"] - med
    panel["y_rel"] = (panel["fwd_ret_rel"] > 0).astype(float)
    panel.loc[panel["fwd_ret"].isna(), ["fwd_ret_rel", "y_rel"]] = np.nan

    # Multi-horizon variants. A one-session relative return is almost all
    # noise; the same signal integrated over 5 or 20 sessions may be far more
    # legible. Measuring where the signal-to-noise ratio actually peaks is a
    # question about the problem, not about the model, so it belongs here.
    logc = np.log(panel["close"])
    panel = panel.sort_values(["symbol", "date"])
    g = panel.groupby("symbol", observed=True)
    for h in HORIZONS:
        fwd = g["close"].shift(-h)
        r = np.log(fwd / panel["close"])
        panel[f"fwd_ret_{h}d"] = r
        panel[f"y_dir_{h}d"] = (r > 0).astype(float)
        m = panel.groupby("date", observed=True)[f"fwd_ret_{h}d"].transform("median")
        panel[f"fwd_ret_rel_{h}d"] = r - m
        panel[f"y_rel_{h}d"] = (r - m > 0).astype(float)
        panel.loc[r.isna(), [f"fwd_ret_rel_{h}d", f"y_rel_{h}d", f"y_dir_{h}d"]] = np.nan
    return panel


def _regimes(feat: pd.DataFrame) -> pd.DataFrame:
    """Regime labels from already-lagged features, used for the per-regime
    breakdown in M1 and as bandit context in M3."""
    f = feat
    vol = pd.cut(f["atr_ratio"], [-np.inf, 0.85, 1.15, np.inf],
                 labels=["compressed", "normal", "elevated"])
    trend = pd.Series(
        np.where(f["adx14"] > 0.25,
                 np.where(f["px_sma50"] > 0, "trend_up", "trend_down"),
                 "range"),
        index=f.index)
    return pd.DataFrame({
        "vol_bucket": vol.astype(object),
        "trend_bucket": trend,
        "regime": vol.astype(str) + "|" + trend,
    }, index=f.index)


def _rebase_price_breaks(oh: pd.DataFrame) -> tuple[pd.DataFrame, set]:
    """Make the price series continuous across capital-structure ex-dates.

    Blanking the ex-date is not enough. Every trailing statistic denominated in
    rupees keeps its pre-event scale while the price does not, so the ratios
    built from them are wrong long after the gap: masking VEDL's demerger left
    ATR at pre-demerger rupee levels divided by a post-demerger price, and
    ``atr_pct`` jumped from 0.034 to 0.089 - a fabricated volatility regime
    lasting weeks.

    So the series is rebased instead. All prices strictly before a break are
    multiplied by (post / pre), which is the standard adjustment-factor
    treatment: the discontinuity disappears, every within-regime price
    relationship is preserved, and rolling windows can span the event without
    picking up a jump. Breaks are applied newest-first so that a symbol with
    several of them compounds correctly.

    The ex-date's own label is still discarded - the series is continuous now,
    but no holder earned that session's "return", so it must not be trained on.
    """
    from cef.db import connect

    try:
        with connect() as conn:
            actions = pd.read_sql(
                "SELECT symbol, ex_date, subject, kind FROM corporate_actions", conn)
    except Exception:                                    # noqa: BLE001
        actions = pd.DataFrame()

    breaks = detect_breaks(oh, actions if not actions.empty else None)
    if breaks.empty:
        log.info("no capital-structure breaks detected")
        return oh, set()

    oh = oh.sort_values(["symbol", "date"]).copy()
    price_cols = ["open", "high", "low", "close", "adj_close"]
    break_keys = set()

    for b in breaks.sort_values("date", ascending=False).itertuples(index=False):
        d = pd.Timestamp(b.date)
        rows = oh[oh["symbol"] == b.symbol]
        at = rows[rows["date"] == d]
        prev = rows[rows["date"] < d]
        if at.empty or prev.empty:
            continue
        post = float(at["close"].iloc[0])
        pre = float(prev["close"].iloc[-1])
        if not (np.isfinite(post) and np.isfinite(pre)) or pre <= 0:
            continue
        factor = post / pre
        mask = (oh["symbol"] == b.symbol) & (oh["date"] < d)
        oh.loc[mask, price_cols] = oh.loc[mask, price_cols] * factor
        break_keys.add((b.symbol, b.date))
        log.info("rebased %s before %s by x%.4f  (%s)",
                 b.symbol, b.date, factor, str(b.reason)[:60])

    return oh, break_keys


def build_panel(symbols: list[str] | None = None) -> pd.DataFrame:
    """Assemble the modelling panel: one row per (symbol, session)."""
    oh = read_ohlcv(symbols)
    if oh.empty:
        raise RuntimeError("ohlcv table is empty - run scripts/ingest.py first")
    oh, break_keys = _rebase_price_breaks(oh)

    days = pd.DatetimeIndex(sorted(oh["date"].unique()))
    macro_wide = read_macro()
    mac_feat = macro_features(days, macro_wide)
    mac_lagged = _lagged_macro_levels(days, macro_wide)
    log.info("macro block: %d features over %d sessions", mac_feat.shape[1], len(days))

    # --- per-symbol technical block -----------------------------------------
    tech_parts = []
    for sym, g in oh.groupby("symbol", observed=True, sort=False):
        g = g.sort_values("date").set_index("date")
        if len(g) < WARMUP_ROWS + 60:
            log.warning("%s: only %d sessions, skipping", sym, len(g))
            continue
        t = technical_features(g)
        t["symbol"] = sym
        tech_parts.append(t.reset_index())
    tech = pd.concat(tech_parts, ignore_index=True)
    log.info("technical block: %d features", tech.shape[1] - 2)

    kept = set(tech["symbol"])
    oh = oh[oh["symbol"].isin(kept)]

    # --- cross-sectional block ----------------------------------------------
    xs = cross_sectional_features(oh[["symbol", "date", "close"]], mac_feat, mac_lagged)
    log.info("cross-sectional block: %d features", xs.shape[1] - 2)

    # --- join ----------------------------------------------------------------
    panel = _targets(oh)[["symbol", "date", "close", "target_date", "fwd_ret", "y_dir", "y_gap"]]
    panel = panel.merge(tech, on=["symbol", "date"], how="left")
    panel = panel.merge(xs, on=["symbol", "date"], how="left")
    panel = panel.merge(mac_feat.reset_index(names="date"), on="date", how="left")
    panel["sector"] = panel["symbol"].map(SECTOR)
    panel = _relative_target(panel)
    panel = pd.concat([panel, _regimes(panel)], axis=1)

    # --- warmup + label trim -------------------------------------------------
    before = len(panel)
    panel = panel.sort_values(["symbol", "date"])
    panel = panel[panel.groupby("symbol", observed=True).cumcount() >= WARMUP_ROWS]
    # The ex-date session is continuous in price now but is still not a
    # tradeable return, so its label is dropped.
    if break_keys:
        ds = panel["date"].dt.strftime("%Y-%m-%d")
        is_break = pd.Series(list(zip(panel["symbol"], ds)), index=panel.index).isin(break_keys)
        panel = panel[~is_break]
        log.info("dropped %d capital-structure ex-date rows", int(is_break.sum()))
    panel = panel[panel["y_dir"].notna()]
    log.info("trimmed %d -> %d rows (warmup %d/symbol + undefined labels)",
             before, len(panel), WARMUP_ROWS)

    return panel.sort_values(["date", "symbol"]).reset_index(drop=True)


def feature_columns(panel: pd.DataFrame) -> list[str]:
    """Model input columns: everything that is not metadata or a label."""
    return [c for c in panel.columns if c not in META_COLS]
