"""The simple view: the same forecast, same evidence, no jargon.

Written for someone who has never read a candlestick chart and does not intend
to start. The structure walks a reader from the answer to the reasons and back
to the answer:

    headline    what the model thinks, in one sentence
    confidence  how sure it is, and what that percentage means in practice
    tally       points pushing up versus points pulling down
    reasons     one card per aspect: what it is, why it matters, its points,
                and a source where one exists
    explainer   why a good company's share can still fall
    levels      prices worth watching, with what each would mean
    closing     the points added up, and the resulting call at its percentage

Every number is produced by the pipeline before the language model is called.
The model receives structured evidence plus a hand-written glossary entry for
each feature, and its only job is to turn that into prose. It is forbidden from
computing anything, from characterising a feature the glossary does not cover,
and from sounding more certain than the confidence number allows. Output is
checked against the evidence afterwards, and any number that does not appear
there is surfaced on the page rather than quietly trusted.
"""
from __future__ import annotations

import hashlib
import json
import logging

from cef.config import ARTIFACT_DIR
from cef.evidence.glossary import describe
from cef.evidence.narrate import _extract_numbers
from cef.llm import generate_json

log = logging.getLogger(__name__)

CACHE_DIR = ARTIFACT_DIR / "simple_cache"
CACHE_DIR.mkdir(exist_ok=True, parents=True)

SYSTEM_PROMPT = """\
You rewrite a stock forecast for a general reader in India who has never read a \
candlestick chart. Aim at a reading level a bright twelve-year-old could follow.

Hard rules. Breaking any of them makes the output unusable:

1. Never calculate anything. Every number you may use is already in the \
evidence. Do not add, average, convert or infer a new one.
2. Never introduce a fact that is not in the evidence. No company history, no \
news you recall, no sector opinion. If the evidence says a move is \
unexplained, say it is unexplained; do not supply a cause.
3. Never sound more confident than the `confidence_pct` field allows. Around \
50-53% means "close to a coin toss". Above 55% means "a lean, not a \
conviction". There is no case here that justifies certainty.
4. Describe a feature only using the glossary entry supplied with it. If a \
value is positive use `when_high`, if negative use `when_low`. Do not \
characterise anything the glossary does not cover.
5. Points are already reader-facing and already scaled. Quote them as points. \
Never re-scale them and never call them percentages.
5a. THE SIGN OF `points` IS AUTHORITATIVE AND DECIDES THE WHOLE CARD. Negative \
points mean this factor is working against the share; positive means it is \
working for it. The title and the body must both read that way. Never write a \
hopeful title over negative points, and never explain positive points with a \
reason that sounds bad. If you cannot see why the readings net out the way the \
points say, write about the direction the points give and say the underlying \
readings are mixed - do not argue with the number.
5b. When the readings inside one card disagree with each other, say so in \
plain words: "the last week points one way and the last month the other". Never \
state one and then contradict it in the next sentence.
5d. A GOOD-LOOKING READING WITH NEGATIVE POINTS IS NOT A MISTAKE. It is the \
single most important thing this model has learned, so explain it rather than \
apologising for it: a share that has just run ahead of its sector or the market \
tends to give a little of it back the next day. Write it that way - "it has \
been beating its sector, and the model reads a run like that as something to \
be given back tomorrow." The mirror case is equally real: a share that has just \
fallen behind often makes a little of it up. Whenever `is_reversal` is true on \
a card, this is the explanation to give.
5c. Do not merely restate what a reading measures. Say why it matters for THIS \
company, using `sector`, `what_this_company_depends_on` and \
`why_this_matters_here`. A card that only paraphrases the glossary is a failure.
5e. Never quote the evidence text. No quotation marks around phrases lifted \
from `meaning` or `measures`, and no "one reading says…". Absorb the meaning \
and write it in your own plain register, as if you understood the thing and \
were explaining it to a friend. Do not mention "readings", "features", \
"the model's inputs" or any other machinery by name.
6. Express probabilities as percentages, never bare decimals: "about 51%", not \
"0.51".
7. Attach the `source` string verbatim to any reason that carries one. Invent \
no sources. Where a reason has no source, leave the field empty.
8. Plain words over jargon. Write "how much the price swings about" rather than \
"volatility"; "the wider market" rather than "benchmark index". Short \
sentences. No exclamation marks, no hype, no advice on whether to buy or sell.
9. `explainer` is a fixed teaching box and must NOT re-explain the confidence \
number. Its subject is why a single day's share price has so little to do with \
whether the company is any good: over one day a price is mostly a popularity \
contest, over five years it mostly follows the business; news that is already \
public is already in the price. Use `explainer_brief` as the topic. Three short \
paragraphs.
10. Each `levels` entry needs a human label - "the encouraging line", "the line \
it hovers around", "the discouraging line" - never a field name from the \
evidence. `meaning` says what it would tell a reader if the price got there.
11. `closing` must do three things in order: state the points that came in for \
and against, state the resulting call, and give the percentage in plain words. \
End on the honest note that this is a weak edge.

Tone: calm, direct, quietly honest about how little is known. The reader should \
finish understanding both what the model thinks and why that is weak evidence.
"""

SCHEMA = {
    "type": "object",
    "properties": {
        "headline": {"type": "string",
                     "description": "One sentence starting 'Tomorrow, this share is…'"},
        "opening": {"type": "string", "description": "2-4 sentences of framing"},
        "confidence_line": {"type": "string",
                            "description": "What the confidence percentage means in practice"},
        "reasons": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "aspect": {"type": "string"},
                    "points": {"type": "number"},
                    "title": {"type": "string", "description": "A short plain headline"},
                    "body": {"type": "string", "description": "2-4 plain sentences"},
                    "source": {"type": "string"},
                },
                "required": ["aspect", "points", "title", "body"],
            },
        },
        "explainer": {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "paragraphs": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["question", "paragraphs"],
        },
        "levels": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "price": {"type": "number"},
                    "meaning": {"type": "string"},
                },
                "required": ["label", "price", "meaning"],
            },
        },
        "closing": {"type": "string",
                    "description": "The points added up, the resulting call, and its percentage"},
    },
    "required": ["headline", "opening", "confidence_line", "reasons",
                 "explainer", "levels", "closing"],
}


# What each aspect bucket actually means for a specific company, so the model
# has something to explain rather than a glossary line to paraphrase.
_WHY = {
    "Driver linkage": ("This company's profits rise and fall with the price of "
                       "{drivers}. When those move, the share usually follows."),
    "Global cues": ("Money moves between countries. When American interest rates "
                    "or the dollar shift, foreign investors reprice Indian shares "
                    "regardless of how this company is doing."),
    "Market & breadth": ("On most days a share mainly does what the whole market "
                         "does. This measures how much of the market's mood is "
                         "likely to carry through to this one."),
    "Sector & peers": ("Companies in the same business tend to move together, "
                       "because the same news hits all of them. This compares it "
                       "against the rest of {sector}."),
    "Technical": ("This is the share's own recent behaviour - whether it has been "
                  "climbing or sliding, and whether buyers or sellers finished "
                  "each day in control."),
    "Liquidity & flow": ("How much money is actually changing hands. A share that "
                         "few people are trading moves further on the same amount "
                         "of buying or selling."),
    "Volatility": ("How much this share normally swings about. Wide swings make "
                   "any single-day call less reliable."),
    "Calendar": ("Some weeks distort prices for reasons unrelated to any company - "
                 "month-end and derivatives-expiry weeks especially."),
}


def _why_matters(aspect: str, forecast: dict) -> str:
    template = _WHY.get(aspect)
    if not template:
        return ""
    return template.format(
        drivers=", ".join(forecast.get("drivers", [])) or "its key inputs",
        sector=forecast.get("sector", "its sector"))


def build_evidence(forecast: dict, attribution: list[dict], drivers: list[dict],
                   levels: dict, coverage: dict) -> dict:
    """Assemble the structured evidence the model is allowed to draw on."""
    f = forecast
    acted = f["acted"]
    up = f["proba_outperform"] > 0.5

    reasons = []
    for a in f.get("aspects", []):
        if not a.get("points"):
            continue
        described = []
        for e in a.get("evidence", []):
            g = describe(e["feature"])
            if not g:
                continue
            described.append({
                "measures": g["means"],
                "reading": ("high" if (e.get("value") or 0) >= 0 else "low"),
                "meaning": g["when_high"] if (e.get("value") or 0) >= 0 else g["when_low"],
                "value": e.get("value"),
            })
        # The net sign can disagree with individual readings, because the
        # points are a sum of contributions that may pull opposite ways. Saying
        # so explicitly stops the model from writing a cheerful title over a
        # negative number - the failure mode of the first version.
        readings_up = sum(1 for d in described if d["reading"] == "high")
        mixed = 0 < readings_up < len(described)
        # A card whose readings all look favourable but whose points are
        # negative (or the reverse) is the reversal effect showing through -
        # the one real edge this model has. Flagged so the prose can explain it
        # instead of tying itself in knots.
        is_reversal = (described and not mixed
                       and ((readings_up == len(described)) == (a["points"] < 0)))
        reasons.append({
            "aspect": a["aspect"],
            "points": a["points"],
            "works_for_or_against": ("against the share" if a["points"] < 0
                                     else "for the share"),
            "readings_disagree_with_each_other": mixed,
            "is_reversal": bool(is_reversal),
            "share_of_signal_pct": round(a.get("share_of_signal", 0) * 100, 1),
            "why_this_matters_here": _why_matters(a["aspect"], forecast),
            "what_it_looked_at": described,
        })

    # Real filings and unexplained moves, kept separate so the model cannot
    # blur "we found a cause" into "we did not".
    moves = [{
        "date": m["date"],
        "move_pct": round(m["ret"] * 100, 2),
        "sigma": m["sigma"],
        "cause_found": m["kind"] != "unexplained",
        "what_was_filed": (m["headline"] or "")[:260] if m["kind"] != "unexplained" else None,
        "source": (f"NSE filing · {m['source_url']}" if m.get("source_url") else ""),
        "evidence_strength": m.get("band"),
    } for m in (attribution or [])[:5]]

    up_pts = sum(r["points"] for r in reasons if r["points"] > 0)
    dn_pts = sum(r["points"] for r in reasons if r["points"] < 0)

    return {
        "symbol": f["symbol"],
        "sector": f["sector"],
        "question": ("Will this share do better or worse than the middle-of-the-pack "
                     "large Indian company in the next trading session?"),
        "last_close_rupees": f["last_close"],
        "typical_daily_swing_rupees": f["typical_daily_range"],
        "target_session": f["target_date"],
        "call": ("no call — the model is not confident enough to take a side" if not acted
                 else "expected to do better than the pack" if up
                 else "expected to do worse than the pack"),
        "confidence_pct": round(f["confidence"] * 100, 1),
        "probability_it_outperforms_pct": round(f["proba_outperform"] * 100, 1),
        "distance_from_coin_toss_pp": round(f["conviction"] * 100, 2),
        "threshold_needed_to_commit_pp": round(f["abstain_threshold"] * 100, 1),
        "tally": {
            "points_pushing_up": up_pts,
            "points_pulling_down": dn_pts,
            "checks_pushing_up": sum(1 for r in reasons if r["points"] > 0),
            "checks_pulling_down": sum(1 for r in reasons if r["points"] < 0),
            "net_points": up_pts + dn_pts,
        },
        "reasons": reasons,
        "recent_big_moves": moves,
        "how_many_moves_had_a_cause": coverage.get("explained"),
        "how_many_moves_were_unexplained": coverage.get("unexplained"),
        "commodity_and_currency_drivers": [{
            "driver": d["driver"],
            "how_closely_this_share_follows_it": d["corr_120d"],
            "its_last_move_pct": d["last_chg_pct"],
        } for d in (drivers or [])],
        "what_this_company_depends_on": forecast.get("drivers", []),
        "explainer_brief": ("Why a single day's share price has little to do with "
                           "whether the company is any good. Over one day a price "
                           "is mostly a popularity contest; over five years it "
                           "mostly follows the business. Results everyone has "
                           "already read are already in the price, so a fine "
                           "company can drift down for weeks while nothing about "
                           "the company changes."),
        "prices_worth_watching": [
            {"label": "the encouraging line", "price": levels.get("r1"),
             "if_it_gets_here": "the weak patch is probably over"},
            {"label": "the line it hovers around", "price": levels.get("pivot"),
             "if_it_gets_here": "above it is mildly encouraging, below it mildly discouraging"},
            {"label": "the discouraging line", "price": levels.get("s1"),
             "if_it_gets_here": "expect more falling before any recovery"},
        ],
        "how_these_prices_were_worked_out": levels.get("basis"),
        "market_conditions": {
            "how_much_it_swings": f["regime"]["volatility"],
            "trend": f["regime"]["trend"].replace("_", " "),
        },
        "model_track_record": {
            "right_this_often_pct": round(f["track_record"]["accuracy"] * 100, 1),
            "coin_flip_would_be_pct": round(f["track_record"]["coin_flip_baseline"] * 100, 1),
            "tested_over": f["track_record"]["window"],
            "declines_to_call_share": "about three sessions in five",
        },
    }


def _cache_key(ev: dict) -> str:
    blob = json.dumps(ev, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:20]


def verify(out: dict, ev: dict) -> list[str]:
    """Numbers in the prose must trace back to the evidence.

    A language model asked to write plainly will occasionally round, convert or
    helpfully invent a figure. Rather than trust it, every number in the output
    is checked against the evidence and anything unmatched is reported to the
    page, which shows it as a warning instead of hiding it.
    """
    # Compare magnitudes. The evidence stores points_pulling_down as -260 and
    # good prose says "260 points against"; a signed string comparison calls
    # that a fabrication when it is exactly faithful.
    def mag(ns: set[str]) -> set[str]:
        out = set()
        for n in ns:
            try:
                out.add(f"{abs(float(n)):g}")
            except ValueError:
                out.add(n)
        return out

    allowed = mag(_extract_numbers(json.dumps(ev, default=str)))
    prose = " ".join([
        out.get("headline", ""), out.get("opening", ""), out.get("confidence_line", ""),
        out.get("closing", ""),
        " ".join(r.get("title", "") + " " + r.get("body", "") for r in out.get("reasons", [])),
        " ".join(" ".join(p for p in out.get("explainer", {}).get("paragraphs", []))),
        " ".join(l.get("meaning", "") for l in out.get("levels", [])),
    ])
    used = mag(_extract_numbers(prose))
    # Small integers are ordinary prose ("two reasons", "five sessions") and
    # flagging them would bury the real problems in noise.
    return sorted(n for n in used - allowed if abs(float(n)) >= 10)


def generate(forecast: dict, attribution: list[dict], drivers: list[dict],
             levels: dict, coverage: dict, use_cache: bool = True,
             prefer: str = "gemini") -> dict:
    ev = build_evidence(forecast, attribution, drivers, levels, coverage)
    key = _cache_key(ev)
    path = CACHE_DIR / f"{ev['symbol']}_{ev['target_session']}_{key}.json"
    if use_cache and path.exists():
        cached = json.loads(path.read_text())
        cached["cached"] = True
        return cached

    user = ("Here is everything known about this forecast. Write the simple view "
            "from it and nothing else.\n\n" + json.dumps(ev, indent=2, default=str))
    out, meta = generate_json(SYSTEM_PROMPT, user, SCHEMA, prefer=prefer)

    out["evidence"] = ev
    out["provider"] = meta["provider"]
    out["model"] = meta["model"]
    out["tokens"] = {"in": meta.get("tokens_in"), "out": meta.get("tokens_out")}
    out["unverified_numbers"] = verify(out, ev)
    out["cached"] = False
    path.write_text(json.dumps(out, indent=2, default=str))
    return out
