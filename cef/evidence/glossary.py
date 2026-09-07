"""Plain-English glossary for model features.

The simple view exists to explain a forecast to someone who has never read a
candlestick chart. That cannot be done by handing a language model a list like
``dollar_vol_z = -2.41`` and hoping — it will invent a confident-sounding
meaning, and the whole page becomes untrustworthy in exactly the way this
project is trying to avoid.

So each feature carries three things written by hand:

``plain``  what it measures, in a sentence a non-specialist can follow.
``high``   what an unusually high reading means.
``low``    what an unusually low reading means.

Patterns are matched most-specific-first. Anything unmatched is described
generically rather than guessed at, and the generation prompt forbids the model
from characterising a feature the glossary does not cover.
"""
from __future__ import annotations

import re

# (regex, plain, meaning when high, meaning when low)
GLOSSARY: list[tuple[str, str, str, str]] = [
    # --- driver linkage -----------------------------------------------------
    (r"^driver_ret5$", "the recent move in the commodities and currencies this company's earnings actually depend on, over the last week",
     "those inputs have been rising, which usually helps earnings",
     "those inputs have been falling, which usually squeezes earnings"),
    (r"^driver_ret20$", "the move in this company's key commodities and currencies over the last month",
     "a month of rising input prices", "a month of falling input prices"),
    (r"^driver_beta60$", "how strongly this share has followed its main commodity over three months",
     "it follows that commodity closely", "it has been moving independently of it"),
    (r"^driver_corr120$", "how tightly this share tracks its main commodity over six months",
     "very tightly", "barely at all"),
    (r"^driver_ret1_wtd$", "yesterday's commodity move, scaled by how much this share normally cares about it",
     "a commodity move this share tends to respond to", "a commodity move it tends to ignore"),

    # --- relative strength / sector ----------------------------------------
    (r"^rs_sector_(\d+)d$", "how this share has done against its own sector over the stated period",
     "it has been beating its sector", "it has been lagging its sector"),
    (r"^rs_nifty_(\d+)d$", "how this share has done against the Nifty over the stated period",
     "it has been beating the index", "it has been lagging the index"),
    (r"^xs_rank_", "where this share sits against all 52 in the universe on that measure, as a percentile",
     "near the top of the pack", "near the bottom of the pack"),
    (r"^xs_sector_rank_", "where it ranks inside its own sector, as a percentile",
     "leading its sector", "trailing its sector"),

    # --- market / breadth ---------------------------------------------------
    (r"^sensex_vs_ma20$", "where the Sensex sits relative to its own 20-day average",
     "the wider market is trading above its recent average", "the wider market is trading below its recent average"),
    (r"^nifty50_vs_ma20$", "where the Nifty sits relative to its 20-day average",
     "the index is above its recent average", "the index is below its recent average"),
    (r"^nifty50_chg(\d+)$", "the Nifty's move over the stated number of sessions",
     "the index has been rising", "the index has been falling"),
    (r"^sensex_chg(\d+)$", "the Sensex's move over the stated number of sessions",
     "the market has been rising", "the market has been falling"),
    (r"^niftybank_chg(\d+)$", "the move in Bank Nifty, which leads the Indian market more often than not",
     "banks have been strong", "banks have been weak"),
    (r"^breadth_1d$", "the share of the 52 large-caps that rose in the last session",
     "most of the market rose together", "most of the market fell together"),
    (r"^breadth_ma5$", "the share of the market rising, averaged over the last week",
     "broad participation in the rally", "broad weakness across the market"),
    (r"^corr_nifty60$", "how closely this share has moved with the index over three months",
     "it is moving almost in lockstep with the index", "it has been moving on its own"),
    (r"^corr_sector60$", "how closely it has moved with its sector over three months",
     "in step with its sector", "detached from its sector"),
    (r"^beta(60|120)$", "how much this share amplifies an index move",
     "it exaggerates index moves", "it dampens index moves"),
    (r"^beta_shift$", "whether the share has recently become more or less sensitive to the index",
     "it has become more index-driven lately", "it has become less index-driven lately"),
    (r"^india_us_spread_20d$", "whether India has been outperforming or lagging the US market over a month",
     "India has been leading", "India has been lagging"),

    # --- global cues --------------------------------------------------------
    (r"^us10y_(level|z60)$", "the US ten-year government bond yield, the world's benchmark interest rate",
     "high rates, which usually pressure share valuations and gold",
     "low rates, which usually support share valuations"),
    (r"^us10y_chg(\d+)$", "the change in the US ten-year yield",
     "rates rising, generally unhelpful for shares", "rates falling, generally helpful for shares"),
    (r"^vix_(level|z60)$", "the VIX, Wall Street's fear gauge",
     "elevated fear in global markets", "calm global markets"),
    (r"^indiavix_(level|z60)$", "India VIX, the local fear gauge",
     "nervous local markets", "calm local markets"),
    (r"^(vix|indiavix)_chg(\d+)$", "the change in the fear gauge",
     "fear rising", "fear subsiding"),
    (r"^brent_chg(\d+)$", "the move in Brent crude. India imports most of its oil, so expensive crude squeezes the whole economy",
     "crude has been rising, a headwind for India", "crude has been falling, a tailwind for India"),
    (r"^gold_chg(\d+)$", "the move in gold", "gold has been rising", "gold has been falling"),
    (r"^silver_chg(\d+)$", "the move in silver", "silver has been rising", "silver has been falling"),
    (r"^copper_chg(\d+)$", "the move in copper, a proxy for global industrial demand",
     "industrial demand looks firm", "industrial demand looks soft"),
    (r"^dxy_chg(\d+)$", "the move in the US dollar index",
     "a strengthening dollar, which usually pressures commodities and emerging markets",
     "a weakening dollar, which usually helps commodities and emerging markets"),
    (r"^usdinr_chg(\d+)$", "the move in the rupee against the dollar",
     "a weakening rupee", "a strengthening rupee"),
    (r"^(sp500|nasdaq|dow)_chg(\d+)$", "the move in US equities",
     "US markets have been rising", "US markets have been falling"),
    (r"^(nikkei|hangseng|kospi|shanghai)_chg(\d+)$", "the move in an Asian market that closes before India does",
     "Asian markets were strong", "Asian markets were weak"),
    (r"^asia_(breadth|mean)_1d$", "how Asian markets did in the session before India closed",
     "Asia was broadly up", "Asia was broadly down"),
    (r"^risk_on_5d$", "a combined read on global risk appetite over the past week",
     "investors have been willing to take risk", "investors have been avoiding risk"),
    (r"^gold_silver_ratio_chg5$", "whether gold has been outpacing silver",
     "gold outpacing silver, typically a defensive signal", "silver outpacing gold, typically a risk-on signal"),

    # --- price action / technical ------------------------------------------
    (r"^ret_(\d+)d$", "this share's own return over the stated number of sessions",
     "it has been climbing", "it has been sliding"),
    (r"^idio_ret1$", "the part of yesterday's move that was specific to this company rather than the market",
     "it outperformed what the market alone would explain",
     "it underperformed what the market alone would explain"),
    (r"^idio_vol20$", "how much this share moves for reasons of its own rather than the market's",
     "a lot of company-specific movement", "movement mostly explained by the market"),
    (r"^idio_share$", "the share of its volatility that is company-specific",
     "mostly its own story", "mostly the market's story"),
    (r"^px_(sma|ema)(\d+)$", "where the price sits relative to its own moving average",
     "trading above its own average, a sign of momentum",
     "trading below its own average, a sign momentum has gone"),
    (r"^sma(\d+)_slope$", "whether the share's trend line is turning up or down",
     "the trend is turning up", "the trend is turning down"),
    (r"^sma20_50$|^sma50_200$", "whether the short-term trend sits above or below the long-term one",
     "short-term strength above the longer trend", "short-term weakness below the longer trend"),
    (r"^rsi(\d+)$", "a standard 0-to-1 gauge of whether a share looks overbought or oversold",
     "looking overbought after a run up", "looking oversold after a fall"),
    (r"^macd", "a standard momentum indicator built from two moving averages",
     "momentum building", "momentum fading"),
    (r"^stoch_[kd]$|^williams_r$", "where the price sits inside its recent trading range",
     "near the top of its recent range", "near the bottom of its recent range"),
    (r"^roc_(\d+)$", "the rate of change in price over the stated period",
     "accelerating upward", "accelerating downward"),
    (r"^range_pos(\d+)$", "where today's close sits between the period's high and low",
     "closing near the highs of the period", "closing near the lows of the period"),
    (r"^dist_high(\d+)$", "how far below the period's high the price is",
     "close to its recent high", "well below its recent high"),
    (r"^dist_low(\d+)$", "how far above the period's low the price is",
     "well above its recent low", "close to its recent low"),
    (r"^up_days_(\d+)$", "the share of recent sessions that closed higher",
     "more up days than down", "more down days than up"),
    (r"^higher_(high|low)_10$", "whether the share has been making progressively higher highs or lows",
     "a rising staircase of highs and lows", "a falling staircase"),
    (r"^close_loc(_mean5)?$", "where in the day's range the share closed — near the high or near the low",
     "buyers held control into the close", "sellers held control into the close"),
    (r"^(upper|lower)_wick_pct$", "how long the candle's upper or lower shadow was — how far price went before being pushed back",
     "price was pushed back from an extreme", "little rejection at the extremes"),
    (r"^body_pct$", "how decisive the session was — a big body means one side dominated",
     "a decisive session", "an indecisive session"),
    (r"^is_(doji|hammer|shooting_star|bull_engulf|bear_engulf)$", "whether a named candlestick shape appeared",
     "the pattern is present", "the pattern is absent"),
    (r"^gap_open$|^gap_abs_mean5$", "how far the share opened away from the previous close",
     "opening well away from the prior close", "opening close to the prior close"),
    (r"^trend_persistence$", "whether recent daily moves have tended to continue or reverse",
     "moves have been continuing", "moves have been reversing"),
    (r"^hurst60$", "a statistical read on whether the share has been trending or mean-reverting",
     "behaving like a trending share", "behaving like a mean-reverting share"),
    (r"^adx14$|^choppiness$", "how directional or how choppy recent trading has been",
     "strongly directional", "choppy and directionless"),

    # --- volatility ---------------------------------------------------------
    (r"^atr_pct$|^atr_ratio$", "how much this share typically moves in a day, relative to its own history",
     "moving more than it usually does", "moving less than it usually does"),
    (r"^realvol_(\d+)$", "how volatile the share has actually been recently",
     "unusually volatile", "unusually quiet"),
    (r"^vol_of_vol$", "how unstable the share's own volatility has been",
     "even its volatility is unsettled", "steady, predictable volatility"),
    (r"^bb_(width|pos)$|^keltner_pos$", "where the price sits inside its statistical bands, and how wide those bands are",
     "stretched toward the top of its band", "stretched toward the bottom of its band"),
    (r"^hl_range(_z)?$", "how wide the day's high-to-low range was against its norm",
     "a much wider day than usual", "a much narrower day than usual"),

    # --- liquidity / volume -------------------------------------------------
    (r"^vol_ratio_(\d+)$", "today's traded volume against its recent average",
     "unusually heavy trading", "unusually thin trading"),
    (r"^vol_trend$", "whether trading activity has been picking up or drying up",
     "activity picking up", "activity drying up"),
    (r"^dollar_vol_z$", "how unusual the rupee value traded was, against the last three months",
     "an unusually large amount of money changed hands",
     "an unusually small amount of money changed hands — thin, easily-moved trading"),
    (r"^obv_slope$", "whether volume has been flowing into the share or out of it",
     "volume flowing in", "volume flowing out"),
    (r"^vwap_dist$", "where the price sits against the average price people actually paid recently",
     "above what recent buyers paid", "below what recent buyers paid"),

    # --- calendar -----------------------------------------------------------
    (r"^dow$", "the day of the week", "later in the week", "earlier in the week"),
    (r"^month$", "the month of the year", "later in the year", "earlier in the year"),
    (r"^is_month_end_wk$|^is_expiry_wk$", "whether this falls in a month-end or derivatives-expiry week",
     "an expiry or month-end week, when flows can distort prices", "an ordinary week"),
    (r"^gap_days$", "how many calendar days since the last trading session",
     "a long weekend or holiday break, over which more foreign news accumulates",
     "a normal overnight gap"),

    # --- generic catch-alls, matched last -----------------------------------
    (r"^driver_ret1$", "yesterday's move in the commodities and currencies this company's earnings depend on",
     "those inputs rose", "those inputs fell"),
    (r"_vs_ma20$", "where that market or commodity sits relative to its own 20-day average",
     "above its recent average", "below its recent average"),
]

_COMPILED = [(re.compile(p), plain, hi, lo) for p, plain, hi, lo in GLOSSARY]


def describe(feature: str) -> dict | None:
    for rx, plain, hi, lo in _COMPILED:
        if rx.search(feature):
            return {"feature": feature, "means": plain, "when_high": hi, "when_low": lo}
    return None


def coverage(features: list[str]) -> tuple[list[str], list[str]]:
    """Which features the glossary can and cannot describe."""
    known = [f for f in features if describe(f)]
    return known, [f for f in features if f not in known]
