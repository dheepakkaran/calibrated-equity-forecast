"""Cross-sectional and driver-linkage features.

Everything above this module is per-symbol or market-wide. The market-wide
macro block is identical for all 52 symbols on a given date, which means on its
own it can only teach the model about *days*, not about *stocks*. This module
supplies the interaction: how this symbol behaves relative to its index, its
sector, and the commodities its earnings actually depend on.

All cross-sectional statistics are computed within a single date across
symbols. That uses no future information - every symbol's close for date t is
known at date t's close - so date-wise ranking is leak-free.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cef.universe import SECTOR, drivers_for

EPS = 1e-9


def _rolling_beta(sym_ret: pd.Series, mkt_ret: pd.Series, n: int = 60) -> pd.Series:
    cov = sym_ret.rolling(n).cov(mkt_ret)
    var = mkt_ret.rolling(n).var()
    return cov / (var + EPS)


def cross_sectional_features(
    panel: pd.DataFrame,
    macro_feat: pd.DataFrame,
    macro_wide_lagged: pd.DataFrame,
) -> pd.DataFrame:
    """
    ``panel``  long frame with columns [symbol, date, close], sorted.
    ``macro_feat``  output of ``macro_features`` (indexed by trading day).
    ``macro_wide_lagged``  raw macro levels already reindexed and lag-shifted.

    Returns a long frame keyed by (symbol, date).
    """
    p = panel.copy()
    p["sector"] = p["symbol"].map(SECTOR)
    p["logc"] = np.log(p["close"])
    p["ret1"] = p.groupby("symbol", observed=True)["logc"].diff()

    nifty_ret = macro_feat["nifty50_chg1"]

    # --- sector index: equal-weight mean return of that sector's members -----
    sector_ret = (p.groupby(["date", "sector"], observed=True)["ret1"]
                   .mean().rename("sector_ret1").reset_index())
    p = p.merge(sector_ret, on=["date", "sector"], how="left")

    out = []
    for sym, g in p.groupby("symbol", observed=True, sort=False):
        g = g.sort_values("date").set_index("date")
        f = pd.DataFrame(index=g.index)

        mkt = nifty_ret.reindex(g.index)
        sec = g["sector_ret1"]
        own = g["ret1"]

        # --- relative strength ----------------------------------------------
        for n in (5, 20, 60):
            f[f"rs_nifty_{n}d"] = own.rolling(n).sum() - mkt.rolling(n).sum()
            f[f"rs_sector_{n}d"] = own.rolling(n).sum() - sec.rolling(n).sum()

        # --- market sensitivity ---------------------------------------------
        f["beta60"] = _rolling_beta(own, mkt, 60)
        f["beta120"] = _rolling_beta(own, mkt, 120)
        f["beta_shift"] = f["beta60"] - f["beta120"]
        f["corr_nifty60"] = own.rolling(60).corr(mkt)
        f["corr_sector60"] = own.rolling(60).corr(sec)
        # Residual return: today's move net of what beta says the index explains.
        f["idio_ret1"] = own - f["beta60"] * mkt
        f["idio_vol20"] = f["idio_ret1"].rolling(20).std()
        f["idio_share"] = f["idio_vol20"] / (own.rolling(20).std() + EPS)

        # --- commodity / macro driver linkage (section 6.4) ------------------
        drivers = [d for d in drivers_for(sym) if d in macro_wide_lagged.columns]
        if drivers:
            dret = np.log(macro_wide_lagged[drivers].where(lambda x: x > 0)).diff()
            dret = dret.reindex(g.index)
            f["driver_ret1"] = dret.mean(axis=1)
            f["driver_ret5"] = dret.rolling(5).sum().mean(axis=1)
            f["driver_ret20"] = dret.rolling(20).sum().mean(axis=1)
            primary = drivers[0]
            f["driver_corr120"] = own.rolling(120).corr(dret[primary])
            f["driver_beta60"] = _rolling_beta(own, dret[primary].fillna(0.0), 60)
            # Only counts the driver move the stock is actually sensitive to.
            f["driver_ret1_wtd"] = f["driver_ret1"] * f["driver_corr120"]

        f["symbol"] = sym
        out.append(f.reset_index())

    feat = pd.concat(out, ignore_index=True)

    # --- within-date cross-sectional ranks (uniform in [0, 1]) --------------
    p_idx = p.set_index(["symbol", "date"])
    feat = feat.set_index(["symbol", "date"])
    feat["mom20_raw"] = p_idx["logc"].groupby(level=0).diff(20)
    feat["mom5_raw"] = p_idx["logc"].groupby(level=0).diff(5)
    feat = feat.reset_index()

    for col, name in (("mom20_raw", "xs_rank_mom20"), ("mom5_raw", "xs_rank_mom5"),
                      ("rs_nifty_20d", "xs_rank_rs20"), ("idio_vol20", "xs_rank_idiovol")):
        feat[name] = feat.groupby("date", observed=True)[col].rank(pct=True)

    feat["xs_sector_rank_mom20"] = feat.merge(
        p[["symbol", "date", "sector"]], on=["symbol", "date"], how="left"
    ).groupby(["date", "sector"], observed=True)["mom20_raw"].rank(pct=True).values

    # Breadth: fraction of the universe that rose today. Same for all symbols
    # on a date, but cheap and genuinely informative about regime.
    breadth = p.assign(up=p["ret1"] > 0).groupby("date", observed=True)["up"].mean()
    feat["breadth_1d"] = feat["date"].map(breadth)
    feat["breadth_ma5"] = feat["date"].map(breadth.rolling(5).mean())

    return feat.drop(columns=["mom20_raw", "mom5_raw"]).replace([np.inf, -np.inf], np.nan)
