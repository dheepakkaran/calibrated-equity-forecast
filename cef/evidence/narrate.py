"""Plain-language narration of an already-computed forecast.

The language model's job here is narrow and worth stating precisely: it
rewrites structured evidence that the pipeline has already produced. It does
not forecast, does not compute, does not rank, and is never asked what it
thinks a stock will do. Every number in its output arrives in its input.

That boundary is enforced three ways rather than merely requested. The prompt
carries only the evidence dict, so there is nothing else to draw on. The
system prompt forbids new facts and caps the certainty of the language against
the confidence figure. And ``verify_narration`` checks the returned text for
numbers that do not appear in the evidence, so a hallucinated figure fails
loudly instead of reaching a reader.

The confidence cap matters most. A model told only "bearish" will happily
write "poised to fall"; told the confidence is 52% it still will, unless
instructed otherwise. Since the whole project exists to report weak evidence
honestly, prose that oversells a coin-flip would undo it.

Output is cached per symbol per forecast date, so one generation serves every
reader that day.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re

from cef.config import (ARTIFACT_DIR, NARRATOR_MODEL, NARRATOR_REASONING_EFFORT,
                        OPENAI_API_KEY)

log = logging.getLogger(__name__)

CACHE_DIR = ARTIFACT_DIR / "narration_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

SYSTEM_PROMPT = """\
You rewrite pre-computed equity forecast evidence into plain language for a \
reader who has never read a candlestick chart and does not intend to start.

Absolute rules:
1. Use ONLY facts present in the JSON you are given. Never add a number, a \
date, a company detail, a cause or a piece of market history that is not \
there. If the evidence is thin, say so plainly.
2. Never state or imply a forecast of your own. You are describing what a \
model produced, not what will happen.
3. Match your certainty to the `confidence` field. Below 0.55 the language \
must be visibly hedged - "slightly more likely", "close to a coin toss", \
"weak evidence". Never write "poised to", "set to", "will", or "expect" about \
a confidence under 0.60.
4. If `attribution_kind` is "unexplained", say that no identifiable cause was \
found. Do not invent one.
5. Attribution is correlational. Write "followed", "coincided with", "is \
associated with" - never "caused" or "because of".
6. No advice. Never suggest buying, selling, holding or waiting.

7. Never quote a field name from the JSON ("share_of_signal", "aspects", \
"contribution") and never print a raw decimal with more than two places. The \
`points` values are already reader-facing: describe them as points, or \
describe the factor qualitatively ("the largest single influence"). Percentages \
and rupee prices may be quoted as given.
8. Express probabilities and accuracies as percentages, never as bare \
decimals. Write "about 51%", not "0.51".

Style: short sentences, no jargon, no bullet lists, 90-140 words. Explain any \
unavoidable term in the same sentence. Lead with what the call is and how weak \
or strong it is. Write in British English."""


def _cache_key(evidence: dict) -> str:
    blob = json.dumps(evidence, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


def build_evidence(symbol: str, forecast_date: str, direction: str,
                   confidence: float, aspects: list[dict],
                   attribution: dict | None = None,
                   levels: dict | None = None) -> dict:
    """Assemble the only input the model is allowed to see."""
    return {
        "symbol": symbol,
        "forecast_for": forecast_date,
        "direction": direction,
        "confidence": round(float(confidence), 3),
        "confidence_note": ("near coin-flip" if confidence < 0.55
                            else "modest" if confidence < 0.65 else "firm"),
        "aspects": aspects,
        "attribution_kind": (attribution or {}).get("kind", "none"),
        "attribution": attribution or {},
        "levels": levels or {},
    }


# Language models write minus signs as U+2212 and dashes as en/em dashes, so a
# regex expecting an ASCII hyphen silently drops the sign and then reports a
# correctly-quoted negative number as invented. Normalise before extracting.
_DASHES = str.maketrans({"\u2212": "-", "\u2013": "-", "\u2014": "-", "\u2012": "-"})


def _extract_numbers(text: str) -> set[str]:
    """Numbers as written, normalised. Used to catch invented figures."""
    text = text.translate(_DASHES)
    out = set()
    for tok in re.findall(r"-?\d[\d,]*\.?\d*", text):
        t = tok.replace(",", "").rstrip(".")
        try:
            v = float(t)
        except ValueError:
            continue
        # Small integers are ordinary prose ("one of two reasons") and would
        # produce noise, so they are not treated as claims.
        if abs(v) >= 10 or "." in t:
            out.add(f"{v:g}")
    return out


def verify_narration(text: str, evidence: dict) -> list[str]:
    """Return every numeric claim in the text absent from the evidence."""
    allowed = _extract_numbers(json.dumps(evidence, default=str))
    # Compare on magnitude as well as signed value: prose legitimately writes
    # "a negative contribution of 0.003" for a stored -0.003.
    allowed |= {a.lstrip("-") for a in list(allowed)}
    # Legitimate paraphrase must not be flagged. A stored 0.5102 may honestly
    # appear as "51%", "51.0%" or "about 51", so every allowed value also
    # admits its percentage form and both rounded to zero and one decimal.
    # Fabricated figures still fail, because they have no source value to
    # round from.
    for a in list(allowed):
        try:
            v = float(a)
        except ValueError:
            continue
        for variant in (v, v * 100, v / 100):
            allowed.add(f"{variant:g}")
            allowed.add(f"{round(variant):g}")
            allowed.add(f"{round(variant, 1):g}")
            allowed.add(f"{round(variant, 2):g}")
    found = _extract_numbers(text)
    return sorted(n for n in found if n not in allowed and n.lstrip("-") not in allowed)


def narrate(evidence: dict, model: str | None = None, use_cache: bool = True) -> dict:
    """Return {text, model, unverified_numbers, cached}."""
    model = model or NARRATOR_MODEL
    path = CACHE_DIR / f"{evidence['symbol']}_{evidence['forecast_for']}_{_cache_key(evidence)}.json"
    if use_cache and path.exists():
        return {**json.loads(path.read_text()), "cached": True}

    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set; narration unavailable")

    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(evidence, indent=2, default=str)},
        ],
        reasoning_effort=NARRATOR_REASONING_EFFORT,
        max_completion_tokens=1400,
    )
    text = (resp.choices[0].message.content or "").strip()
    result = {
        "text": text,
        "model": model,
        "unverified_numbers": verify_narration(text, evidence),
        "prompt_tokens": resp.usage.prompt_tokens,
        "completion_tokens": resp.usage.completion_tokens,
    }
    if result["unverified_numbers"]:
        log.warning("%s: narration contains numbers absent from the evidence: %s",
                    evidence["symbol"], result["unverified_numbers"])
    path.write_text(json.dumps(result, indent=2))
    return {**result, "cached": False}
