"""Hand-labelled probe set for the headline scorer.

Labels are the *directional implication for the named stock*, not the tone of
the sentence - which is precisely the distinction a general financial
sentiment model is not trained to make. Grouped by the kind of reasoning each
headline demands.
"""

# (headline, label in {+1 bullish, 0 neutral, -1 bearish}, reasoning class)
PROBES = [
    # --- plain tone: sentiment and direction agree -----------------------
    ("Hindustan Zinc posts record Q1 profit, up 145% year-on-year", +1, "plain"),
    ("Tata Steel reports wider-than-expected quarterly loss", -1, "plain"),
    ("Infosys raises full-year revenue growth guidance", +1, "plain"),
    ("Nifty extended its decline for a fourth straight week", -1, "plain"),
    ("Profit taking drags metal stocks lower after seven-session rally", -1, "plain"),
    ("Reliance shares hit a fresh 52-week high on retail demerger buzz", +1, "plain"),

    # --- supply/ownership: needs market-structure knowledge --------------
    ("Government plans to sell 1.5% stake in Hindustan Zinc via offer for sale", -1, "supply"),
    ("DIPAM rules out immediate plans for a government offer for sale", +1, "supply"),
    ("Promoter pledges additional 3% of shareholding to lenders", -1, "supply"),
    ("Company announces Rs 3,000 crore share buyback at a premium", +1, "supply"),

    # --- shareholder returns: tone is dry, direction is not --------------
    ("Board declares interim dividend of Rs 11 per share", +1, "returns"),
    ("Board defers dividend decision to the next quarter", -1, "returns"),

    # --- commodity linkage: direction depends on the stock, not the tone -
    ("Silver eased as a stronger-than-expected jobs report lifted Treasury yields", -1, "commodity"),
    ("Crude oil surges past $95 a barrel on supply disruption fears", -1, "commodity"),
    ("LME zinc prices rally to a 14-month high on smelter shutdowns", +1, "commodity"),
    ("Rupee weakens past 94 against the dollar", +1, "commodity"),

    # --- rates and policy ------------------------------------------------
    ("RBI holds repo rate steady, signals prolonged pause", 0, "policy"),
    ("Fed rate-hike bets return after hot inflation print", -1, "policy"),

    # --- governance ------------------------------------------------------
    ("Chief financial officer resigns with immediate effect, cites personal reasons", -1, "governance"),
    ("Amarendu Prakash takes charge as chief executive from 1 August", 0, "governance"),
]
