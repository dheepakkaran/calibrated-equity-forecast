"""Materiality and direction priors for NSE filing categories.

Two facts about the filings table shape this module.

First, **most filings are noise.** Of the 143 distinct categories the exchange
uses, the largest are administrative: "Analysts/Institutional Investor Meet
Updates" and generic "Updates" together account for a third of all rows, and
"Loss of Share Certificates" alone is nearly 5%. Attributing a 3-sigma price
move to a lost share certificate would discredit the whole evidence layer, so
every category carries a materiality weight and the inert ones are scored to
zero and never attributed.

Second, **the category often fixes the sign where a text model cannot.** The
probe set in `tests/fixtures/sentiment_probe.py` shows FinBERT scoring 0/4 on
market-structure headlines - it reads a government stake sale and a share
buyback as equally neutral. But the exchange has already told us the filing is
a buyback. A hand-written mapping is exact, auditable and free, and it beats
inference on precisely the class where inference fails.

Matching is on normalised substrings rather than exact equality, because the
raw feed is inconsistent: "General Updates" and "General updates" appear as
separate categories with 261 and 242 rows.

Third, and most importantly: **the exchange's own category field is unreliable
for exactly the events that matter most.** IndusInd Bank's accounting-
discrepancy disclosure - which took 27% off the stock in a single session, the
largest idiosyncratic move in the panel - was filed under "General Updates",
a category that appears 500+ times and is almost always administrative. A gate
that reads only the category cannot catch it, and a gate loose enough to admit
"General Updates" would admit everything.

So materiality is escalated from the body text, and the strongest single cue is
a reference to Regulation 30 of SEBI's Listing Obligations and Disclosure
Requirements. Regulation 30 is the rule that *defines* a material event and
compels its disclosure, so a filing citing it has already been judged material
by the issuer's own compliance officer. That is better evidence than a
free-text category, and it is what recovers the IndusInd disclosure.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class EventClass:
    materiality: float      # 0 = never attribute, 1 = highly material
    direction: int          # -1 bearish, 0 sign unknown, +1 bullish
    label: str              # human-readable, used in the evidence text


# Order matters: the first matching pattern wins, so specific patterns are
# listed before the generic ones they would otherwise be swallowed by.
RULES: list[tuple[str, EventClass]] = [
    # --- inert: routine compliance and clerical filings ------------------
    (r"loss of share|duplicate share|share certificate", EventClass(0.0, 0, "share certificate admin")),
    (r"copy of newspaper|newspaper publication", EventClass(0.0, 0, "newspaper copy")),
    (r"trading window", EventClass(0.0, 0, "trading window closure")),
    (r"esop|esos|esps|employee stock", EventClass(0.0, 0, "employee stock routine")),
    (r"certificate under sebi|reg\. 74|reconciliation of share", EventClass(0.0, 0, "SEBI certificate")),
    (r"schedule of analyst|intimation of.*meet|postal ballot", EventClass(0.0, 0, "meeting schedule")),
    (r"^record date", EventClass(0.0, 0, "record date")),
    (r"shareholders meeting|annual general|agm|egm", EventClass(0.05, 0, "shareholder meeting")),
    (r"news verification", EventClass(0.0, 0, "news verification")),
    (r"newspaper advertisement", EventClass(0.0, 0, "newspaper advertisement")),
    (r"date of payment|book closure", EventClass(0.0, 0, "dividend mechanics")),
    (r"allotment of securit", EventClass(0.20, 0, "allotment follow-through")),
    (r"incorporation", EventClass(0.20, 0, "subsidiary incorporation")),

    # --- high materiality, sign fixed by the category itself -------------
    (r"buy[- ]?back", EventClass(0.95, +1, "share buyback")),
    (r"^dividend|interim dividend|final dividend", EventClass(0.75, +1, "dividend declared")),
    (r"issue of securit|preferential issue|qip|rights issue|fund rais",
     EventClass(0.85, -1, "equity issuance (dilution)")),
    (r"action\(s\) taken|orders passed|penalt|show cause|adjudicat",
     EventClass(0.90, -1, "regulatory action")),
    (r"pledge|encumbran", EventClass(0.80, -1, "promoter pledge")),
    (r"resignation|cessation", EventClass(0.65, -1, "senior departure")),
    # Narrowly worded: "insolvency" appears in filings where the company is the
    # *acquirer* in a resolution process, which is not distress. JSW Steel's
    # letter of intent from a committee of creditors was being read as
    # financial distress before this was tightened.
    (r"\bdefault(ed|ing)?\b|winding up|liquidation|admitted to nclt|"
     r"insolvency proceedings against", EventClass(0.95, -1, "financial distress")),

    # --- high materiality, sign must come from the text or the price -----
    (r"financial result|quarterly result|audited result|unaudited",
     EventClass(1.00, 0, "financial results")),
    (r"outcome of board", EventClass(0.85, 0, "board meeting outcome")),
    (r"takeover regulation|substantial acquisition|open offer",
     EventClass(0.85, 0, "takeover-code disclosure")),
    (r"acquisition|divest|stake sale|slump sale", EventClass(0.80, 0, "acquisition or divestment")),
    (r"scheme of arrangement|amalgamat|demerger|restructur",
     EventClass(0.90, 0, "corporate restructuring")),
    (r"credit rating", EventClass(0.70, 0, "credit rating action")),
    (r"capacity|expansion|new plant|commission", EventClass(0.65, 0, "capacity change")),
    (r"order (win|receipt)|bagging|bags order|receiving of order|contract win|letter of award",
     EventClass(0.75, +1, "order win")),
    # An auditor change is a governance signal well out of proportion to how
    # dull the filing reads.
    (r"change in auditor|auditor resign", EventClass(0.75, -1, "auditor change")),
    (r"diversification|disinvestment", EventClass(0.55, 0, "diversification")),
    (r"retirement", EventClass(0.40, 0, "planned retirement")),
    (r"reg\.\s*30 of sebi \(sast\)|sast reg", EventClass(0.85, 0, "takeover-code disclosure")),

    # --- medium materiality ----------------------------------------------
    (r"change in (director|management|kmp)|appointment|takes charge",
     EventClass(0.45, 0, "leadership change")),
    (r"agreement|mou|joint venture|partnership", EventClass(0.55, 0, "agreement signed")),
    (r"investor presentation|monthly business|operational update",
     EventClass(0.40, 0, "business update")),
    (r"related party", EventClass(0.35, 0, "related-party transaction")),
    (r"news clarification|clarification", EventClass(0.40, 0, "clarification")),

    # --- low materiality catch-alls (must stay last) ---------------------
    (r"analyst|institutional investor|con\.? call", EventClass(0.15, 0, "analyst interaction")),
    (r"press release", EventClass(0.20, 0, "press release")),
    (r"general update|^updates?$|update", EventClass(0.10, 0, "generic update")),
]

_COMPILED = [(re.compile(p, re.I), ec) for p, ec in RULES]
UNKNOWN = EventClass(0.25, 0, "uncategorised filing")

# Clerical categories are never escalated from body text. A lost-share-
# certificate notice that happens to cite a regulation is still clerical.
_NEVER_ESCALATE = {
    "share certificate admin", "newspaper copy", "newspaper advertisement",
    "trading window closure", "employee stock routine", "SEBI certificate",
    "meeting schedule", "record date", "dividend mechanics", "news verification",
}

# Body-text cues that resolve the sign where the category leaves it open.
# Applied only to filings whose category direction is 0.
_TEXT_BULLISH = re.compile(
    r"\b(record (profit|revenue|quarter)|highest ever|beat|exceed|"
    r"upgrade[ds]?|rating upgraded|profit (rose|jump|surge|up)|"
    r"revenue (rose|grew|up)|order win|bags|secured|approval granted)\b", re.I)
_TEXT_BEARISH = re.compile(
    r"\b(loss|decline[ds]?|fell|drop(ped)?|downgrade[ds]?|rating downgraded|"
    r"profit (fell|declined|down)|shortfall|impairment|write[- ]off|"
    r"suspend|terminat|resign|rejected|withdraw)\b", re.I)

# Body cues that raise materiality regardless of how the filing was
# categorised. Each maps to a floor, not an increment: the filing is at least
# this material.
_BODY_ESCALATION: list[tuple[re.Pattern, float, str]] = [
    # Regulation 30 of SEBI LODR is the material-events rule itself.
    (re.compile(r"regulation\s*30|reg\.?\s*30\b|listing obligations and disclosure", re.I),
     0.75, "Reg 30 material-event disclosure"),
    (re.compile(r"discrepanc|restat(e|ement)|mis-?statement|accounting (issue|lapse)|"
                r"derivative portfolio|net worth impact", re.I),
     0.95, "accounting discrepancy"),
    (re.compile(r"managing director|chief executive|\bmd & ceo\b|whole[- ]time director",
                re.I), 0.60, "MD/CEO change"),
    (re.compile(r"forensic audit|whistle[- ]?blower|fraud|siphon", re.I),
     0.95, "governance investigation"),
    (re.compile(r"resolution plan|committee of creditors|letter of intent|successful bidder",
                re.I), 0.75, "insolvency-process outcome"),
    (re.compile(r"credit rating.*(downgrad|upgrad)|rating (revised|placed on watch)", re.I),
     0.75, "rating change"),
]


def classify(category: str | None, body: str | None = None) -> EventClass:
    """Map a filing to its materiality and directional prior."""
    text = (category or "").strip()
    ec = UNKNOWN
    for pattern, candidate in _COMPILED:
        if pattern.search(text):
            ec = candidate
            break

    # Body-text escalation. Applied even to categories scored inert, because
    # that is precisely where the misfiled material events hide - but never to
    # the clerical categories, where a stray keyword means nothing.
    if body and ec.label not in _NEVER_ESCALATE:
        for pattern, floor, label in _BODY_ESCALATION:
            if floor > ec.materiality and pattern.search(body):
                ec = EventClass(floor, ec.direction, f"{label} (filed as: {ec.label})")

    if ec.direction == 0 and ec.materiality > 0 and body:
        bull = len(_TEXT_BULLISH.findall(body))
        bear = len(_TEXT_BEARISH.findall(body))
        if bull != bear:
            return EventClass(ec.materiality, 1 if bull > bear else -1, ec.label)
    return ec
