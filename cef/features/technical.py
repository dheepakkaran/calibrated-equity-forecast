"""Per-symbol technical features, hand-written rather than library-sourced.

Two reasons for hand-rolling instead of pulling TA-Lib or pandas-ta:

1. Causality is the whole point of this project. Every window below is a
   trailing ``rolling`` or ``ewm`` with no centring and no negative shift, so
   the value at row t is a function of bars <= t by construction. That is a
   property we can assert in tests rather than trust in a dependency.
2. Every feature is scale-free - a ratio, a z-score, or a bounded oscillator.
   A single model is trained across all 52 symbols, so a raw rupee price or a
   raw share volume would be worse than useless. Nothing here carries units.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-9


# --- primitives --------------------------------------------------------------
def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = gain / (loss + EPS)
    return 100 - 100 / (1 + rs)


def _true_range(h: pd.Series, l: pd.Series, c: pd.Series) -> pd.Series:
    prev = c.shift(1)
    return pd.concat([h - l, (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)


def _atr(h: pd.Series, l: pd.Series, c: pd.Series, n: int = 14) -> pd.Series:
    return _true_range(h, l, c).ewm(alpha=1 / n, adjust=False).mean()


def _adx(h: pd.Series, l: pd.Series, c: pd.Series, n: int = 14) -> pd.Series:
    up, dn = h.diff(), -l.diff()
    plus = np.where((up > dn) & (up > 0), up, 0.0)
    minus = np.where((dn > up) & (dn > 0), dn, 0.0)
    atr = _atr(h, l, c, n)
    pdi = 100 * pd.Series(plus, index=h.index).ewm(alpha=1 / n, adjust=False).mean() / (atr + EPS)
    mdi = 100 * pd.Series(minus, index=h.index).ewm(alpha=1 / n, adjust=False).mean() / (atr + EPS)
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi + EPS)
    return dx.ewm(alpha=1 / n, adjust=False).mean()


def _hurst(x: pd.Series, window: int = 60) -> pd.Series:
    """Rolling Hurst exponent via rescaled-range on log returns.

    >0.5 trending, <0.5 mean-reverting. Cheap approximation: R/S over the
    window rather than the full multi-lag regression, which is enough to
    separate regimes and keeps the rolling cost linear.
    """
    r = np.log(x).diff()

    def rs(v: np.ndarray) -> float:
        if np.isnan(v).any() or len(v) < 8:
            return np.nan
        dev = np.cumsum(v - v.mean())
        rng = dev.max() - dev.min()
        sd = v.std(ddof=0)
        if sd < EPS or rng < EPS:
            return np.nan
        return float(np.log(rng / sd) / np.log(len(v)))

    return r.rolling(window).apply(rs, raw=True)


# --- feature block -----------------------------------------------------------
def technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """``df`` is one symbol's OHLCV sorted ascending by date.

    Returns a frame indexed like ``df`` holding only scale-free columns.
    """
    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]
    f = pd.DataFrame(index=df.index)

    logc = np.log(c)
    ret1 = logc.diff()

    # --- returns -------------------------------------------------------------
    for n in (1, 2, 3, 5, 10, 20, 60):
        f[f"ret_{n}d"] = logc.diff(n)

    # --- trend: price relative to its own moving averages --------------------
    for n in (5, 10, 20, 50, 200):
        sma = c.rolling(n).mean()
        ema = c.ewm(span=n, adjust=False).mean()
        f[f"px_sma{n}"] = c / (sma + EPS) - 1
        f[f"px_ema{n}"] = c / (ema + EPS) - 1
        f[f"sma{n}_slope"] = sma.pct_change(5)
    f["sma20_50"] = c.rolling(20).mean() / (c.rolling(50).mean() + EPS) - 1
    f["sma50_200"] = c.rolling(50).mean() / (c.rolling(200).mean() + EPS) - 1

    # --- momentum ------------------------------------------------------------
    f["rsi14"] = _rsi(c, 14) / 100
    f["rsi7"] = _rsi(c, 7) / 100
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    f["macd"] = macd / (c + EPS)
    f["macd_hist"] = (macd - signal) / (c + EPS)
    f["macd_cross"] = np.sign(macd - signal)

    lo14, hi14 = l.rolling(14).min(), h.rolling(14).max()
    stoch_k = (c - lo14) / (hi14 - lo14 + EPS)
    f["stoch_k"] = stoch_k
    f["stoch_d"] = stoch_k.rolling(3).mean()
    f["williams_r"] = (hi14 - c) / (hi14 - lo14 + EPS)
    for n in (5, 10, 20):
        f[f"roc_{n}"] = c.pct_change(n)

    # --- volatility ----------------------------------------------------------
    atr14 = _atr(h, l, c, 14)
    f["atr_pct"] = atr14 / (c + EPS)
    f["atr_ratio"] = atr14 / (atr14.rolling(60).mean() + EPS)
    for n in (5, 10, 20):
        f[f"realvol_{n}"] = ret1.rolling(n).std()
    f["vol_of_vol"] = ret1.rolling(20).std().rolling(20).std()
    sma20, sd20 = c.rolling(20).mean(), c.rolling(20).std()
    f["bb_width"] = 4 * sd20 / (sma20 + EPS)
    f["bb_pos"] = (c - (sma20 - 2 * sd20)) / (4 * sd20 + EPS)
    kel = c.ewm(span=20, adjust=False).mean()
    f["keltner_pos"] = (c - kel) / (2 * atr14 + EPS)
    f["hl_range"] = (h - l) / (c + EPS)
    f["hl_range_z"] = (f["hl_range"] - f["hl_range"].rolling(60).mean()) / (
        f["hl_range"].rolling(60).std() + EPS)

    # --- volume --------------------------------------------------------------
    f["vol_ratio_20"] = v / (v.rolling(20).mean() + EPS)
    f["vol_ratio_60"] = v / (v.rolling(60).mean() + EPS)
    f["vol_trend"] = v.rolling(5).mean() / (v.rolling(20).mean() + EPS)
    obv = (np.sign(ret1).fillna(0) * v).cumsum()
    f["obv_slope"] = obv.diff(10) / (v.rolling(20).mean() * 10 + EPS)
    tp = (h + l + c) / 3
    vwap20 = (tp * v).rolling(20).sum() / (v.rolling(20).sum() + EPS)
    f["vwap_dist"] = c / (vwap20 + EPS) - 1
    f["dollar_vol_z"] = np.log1p(c * v).pipe(
        lambda s: (s - s.rolling(60).mean()) / (s.rolling(60).std() + EPS))

    # --- structure -----------------------------------------------------------
    for n in (20, 50, 200):
        hh, ll = h.rolling(n).max(), l.rolling(n).min()
        f[f"dist_high{n}"] = c / (hh + EPS) - 1
        f[f"dist_low{n}"] = c / (ll + EPS) - 1
        f[f"range_pos{n}"] = (c - ll) / (hh - ll + EPS)
    f["gap_open"] = o / (c.shift(1) + EPS) - 1
    f["gap_abs_mean5"] = f["gap_open"].abs().rolling(5).mean()
    f["higher_high_10"] = (h > h.shift(1)).rolling(10).sum() / 10
    f["higher_low_10"] = (l > l.shift(1)).rolling(10).sum() / 10
    f["up_days_10"] = (ret1 > 0).rolling(10).sum() / 10
    f["up_days_20"] = (ret1 > 0).rolling(20).sum() / 20
    f["close_loc"] = (c - l) / (h - l + EPS)          # where in the bar it closed
    f["close_loc_mean5"] = f["close_loc"].rolling(5).mean()

    # --- regime --------------------------------------------------------------
    f["adx14"] = _adx(h, l, c, 14) / 100
    tr_sum = _true_range(h, l, c).rolling(14).sum()
    hi_lo = h.rolling(14).max() - l.rolling(14).min()
    f["choppiness"] = 100 * np.log10((tr_sum + EPS) / (hi_lo + EPS)) / np.log10(14)
    f["hurst60"] = _hurst(c, 60)
    f["trend_persistence"] = ret1.rolling(20).apply(
        lambda s: np.corrcoef(s[:-1], s[1:])[0, 1] if np.isfinite(s).all() else np.nan, raw=True)

    # --- candlestick patterns (hand-coded, all bar-local) --------------------
    body = (c - o).abs()
    rng = (h - l) + EPS
    upper_wick = h - np.maximum(c, o)
    lower_wick = np.minimum(c, o) - l
    f["body_pct"] = body / rng
    f["upper_wick_pct"] = upper_wick / rng
    f["lower_wick_pct"] = lower_wick / rng
    f["is_doji"] = (body / rng < 0.1).astype(float)
    f["is_hammer"] = ((lower_wick > 2 * body) & (upper_wick < body)).astype(float)
    f["is_shooting_star"] = ((upper_wick > 2 * body) & (lower_wick < body)).astype(float)
    prev_body_lo = np.minimum(c.shift(1), o.shift(1))
    prev_body_hi = np.maximum(c.shift(1), o.shift(1))
    f["is_bull_engulf"] = ((c > o) & (o < prev_body_lo) & (c > prev_body_hi)).astype(float)
    f["is_bear_engulf"] = ((c < o) & (o > prev_body_hi) & (c < prev_body_lo)).astype(float)

    return f.replace([np.inf, -np.inf], np.nan)
