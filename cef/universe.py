"""Trading universe and macro-series specifications.

The ``lag`` field on each macro series is the core leak-prevention mechanism in
this project, so it is worth stating the reasoning explicitly.

A forecast for session t+1 is generated at the close of session t, i.e. 15:30
IST on date t. A macro series may only contribute its date-t value if that
value was already published by 15:30 IST on date t.

    Asian sessions   close 11:30-13:00 IST  ->  date-t value IS available  (lag 0)
    Indian sessions  close 15:30    IST     ->  date-t value IS available  (lag 0)
    US sessions      close 01:30    IST t+1 ->  date-t value is NOT        (lag 1)
    US macro prints  released 18:00 IST t   ->  treated as lag 1 for safety

Using a US date-t close to predict session t+1 is a real and legitimate signal
in live trading (it lands hours before the Indian open), but only if the
forecast is generated after it. Backtesting it against a 15:30 IST generation
time is look-ahead bias. We therefore lag it, and accept the weaker signal in
exchange for a number we can defend.
"""
from __future__ import annotations

from dataclasses import dataclass

# --- Nifty 50 constituents ---------------------------------------------------
# Current membership. This introduces survivorship bias: names that were
# dropped from the index over the sample period are absent. Documented as a
# known limitation rather than silently ignored; see README.
NIFTY50 = [
    "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK",
    "BAJAJ-AUTO", "BAJFINANCE", "BAJAJFINSV", "BEL", "BHARTIARTL",
    "CIPLA", "COALINDIA", "DRREDDY", "EICHERMOT", "ETERNAL",
    "GRASIM", "HCLTECH", "HDFCBANK", "HDFCLIFE", "HEROMOTOCO",
    "HINDALCO", "HINDUNILVR", "ICICIBANK", "INDUSINDBK", "INFY",
    "ITC", "JIOFIN", "JSWSTEEL", "KOTAKBANK", "LT",
    "M&M", "MARUTI", "NESTLEIND", "NTPC", "ONGC",
    "POWERGRID", "RELIANCE", "SBILIFE", "SBIN", "SHRIRAMFIN",
    "SUNPHARMA", "TATACONSUM", "TATASTEEL", "TCS",
    "TECHM", "TITAN", "TRENT", "ULTRACEMCO", "WIPRO",
]

# Excluded, with reasons. Kept in code rather than dropped silently so the
# panel's composition is self-documenting.
EXCLUDED = {
    "TATAMOTORS": (
        "Demerged in 2025 into commercial- and passenger-vehicle entities. The "
        "old ticker no longer resolves and the successor (TMPV.NS) has ~24 "
        "sessions of history, so no continuous 5-year series exists."
    ),
}

# Genuinely short histories - correct data, not gaps. Excluded from any
# cross-sectional feature that needs a long lookback, but kept for training.
LATE_LISTINGS = {
    "JIOFIN": "2023-08-21 demerger listing from Reliance",
    "ETERNAL": "2021-07-23 IPO as Zomato; renamed Eternal in 2025",
}

# Extra names carried for the metals thesis in the mockups.
EXTRA_SYMBOLS = ["HINDZINC", "VEDL", "NATIONALUM"]

UNIVERSE = NIFTY50 + EXTRA_SYMBOLS


def yf_symbol(symbol: str) -> str:
    """NSE ticker -> Yahoo Finance ticker."""
    return f"{symbol}.NS"


# --- macro / global series ---------------------------------------------------
@dataclass(frozen=True)
class MacroSeries:
    key: str          # column name in the macro table
    yf_ticker: str    # Yahoo Finance ticker
    lag: int          # sessions of lag before the value may be used
    kind: str         # "index" | "fx" | "rate" | "commodity" | "vol"


MACRO_SERIES: list[MacroSeries] = [
    # India — available at our generation time
    MacroSeries("nifty50",   "^NSEI",     0, "index"),
    MacroSeries("sensex",    "^BSESN",    0, "index"),
    MacroSeries("niftybank", "^NSEBANK",  0, "index"),
    MacroSeries("indiavix",  "^INDIAVIX", 0, "vol"),

    # Asia — sessions close before the Indian close
    MacroSeries("nikkei",    "^N225",     0, "index"),
    MacroSeries("hangseng",  "^HSI",      0, "index"),
    MacroSeries("kospi",     "^KS11",     0, "index"),
    MacroSeries("shanghai",  "000001.SS", 0, "index"),

    # US / global — date-t close lands after the Indian close
    MacroSeries("sp500",     "^GSPC",     1, "index"),
    MacroSeries("nasdaq",    "^IXIC",     1, "index"),
    MacroSeries("dow",       "^DJI",      1, "index"),
    MacroSeries("vix",       "^VIX",      1, "vol"),
    MacroSeries("us10y",     "^TNX",      1, "rate"),
    MacroSeries("dxy",       "DX-Y.NYB",  1, "fx"),
    MacroSeries("usdinr",    "USDINR=X",  1, "fx"),
    MacroSeries("brent",     "BZ=F",      1, "commodity"),
    MacroSeries("gold",      "GC=F",      1, "commodity"),
    MacroSeries("silver",    "SI=F",      1, "commodity"),
    MacroSeries("copper",    "HG=F",      1, "commodity"),
]

MACRO_BY_KEY = {m.key: m for m in MACRO_SERIES}


# --- sector mapping ----------------------------------------------------------
SECTOR = {
    "ADANIENT": "conglomerate", "ADANIPORTS": "infra", "APOLLOHOSP": "healthcare",
    "ASIANPAINT": "consumer", "AXISBANK": "bank", "BAJAJ-AUTO": "auto",
    "BAJFINANCE": "nbfc", "BAJAJFINSV": "nbfc", "BEL": "defence",
    "BHARTIARTL": "telecom", "CIPLA": "pharma", "COALINDIA": "energy",
    "DRREDDY": "pharma", "EICHERMOT": "auto", "ETERNAL": "consumer",
    "GRASIM": "materials", "HCLTECH": "it", "HDFCBANK": "bank",
    "HDFCLIFE": "insurance", "HEROMOTOCO": "auto", "HINDALCO": "metals",
    "HINDUNILVR": "consumer", "ICICIBANK": "bank", "INDUSINDBK": "bank",
    "INFY": "it", "ITC": "consumer", "JIOFIN": "nbfc", "JSWSTEEL": "metals",
    "KOTAKBANK": "bank", "LT": "infra", "M&M": "auto", "MARUTI": "auto",
    "NESTLEIND": "consumer", "NTPC": "power", "ONGC": "energy",
    "POWERGRID": "power", "RELIANCE": "conglomerate", "SBILIFE": "insurance",
    "SBIN": "bank", "SHRIRAMFIN": "nbfc", "SUNPHARMA": "pharma",
    "TATACONSUM": "consumer", "TATASTEEL": "metals",
    "TCS": "it", "TECHM": "it", "TITAN": "consumer", "TRENT": "consumer",
    "ULTRACEMCO": "materials", "WIPRO": "it",
    "HINDZINC": "metals", "VEDL": "metals", "NATIONALUM": "metals",
}

# --- symbol -> commodity/macro driver mapping --------------------------------
# Section 6.4 of the proposal. A metals name is scored against metals, an IT
# name against USD/INR and the Nasdaq.
DRIVERS: dict[str, list[str]] = {
    "metals":     ["copper", "silver", "gold", "dxy"],
    "it":         ["usdinr", "nasdaq", "sp500"],
    "bank":       ["us10y", "niftybank", "indiavix"],
    "nbfc":       ["us10y", "niftybank"],
    "insurance":  ["us10y", "niftybank"],
    "energy":     ["brent", "dxy"],
    "auto":       ["brent", "usdinr"],
    "pharma":     ["usdinr", "nasdaq"],
    "consumer":   ["brent", "nifty50"],
    "materials":  ["copper", "brent"],
    "power":      ["brent", "us10y"],
    "infra":      ["brent", "us10y"],
    "telecom":    ["nifty50", "us10y"],
    "defence":    ["nifty50"],
    "healthcare": ["nifty50"],
    "conglomerate": ["brent", "nifty50", "dxy"],
}


def drivers_for(symbol: str) -> list[str]:
    return DRIVERS.get(SECTOR.get(symbol, ""), ["nifty50"])
