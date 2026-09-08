"""Tests for the plain-language layer.

The generated prose cannot be asserted on directly, so these tests pin the
deterministic evidence the model is handed — which is where the real bugs were.
A headline that says "more likely to do better" over a 49.9% probability is the
worst error this interface can make, and it came from letting the model infer
the direction rather than being told it.
"""
from __future__ import annotations

import pytest

from cef.evidence.glossary import coverage, describe
from cef.evidence.simple import build_evidence, verify


def _forecast(proba: float, acted: bool = True) -> dict:
    return {
        "symbol": "TESTCO", "sector": "metals",
        "as_of": "2026-09-03", "target_date": "2026-09-04",
        "direction": "outperform" if proba > 0.5 else "underperform",
        "proba_outperform": proba,
        "confidence": max(proba, 1 - proba),
        "conviction": abs(proba - 0.5),
        "acted": acted, "abstain_threshold": 0.008,
        "last_close": 500.0, "typical_daily_range": 10.0,
        "regime": {"label": "normal|range", "volatility": "normal", "trend": "range"},
        "drivers": ["copper", "silver"],
        "aspects": [
            {"aspect": "Technical", "points": 100, "direction": "up",
             "share_of_signal": 0.4,
             "evidence": [{"feature": "ret_3d", "shap": 0.01, "value": 0.02}]},
            {"aspect": "Global cues", "points": -40, "direction": "down",
             "share_of_signal": 0.2,
             "evidence": [{"feature": "us10y_level", "shap": -0.004, "value": 4.7}]},
        ],
        "track_record": {"accuracy": 0.5102, "coin_flip_baseline": 0.5001,
                         "window": "2024-03 to 2026-08"},
    }


# --------------------------------------------------------------------------
# The direction must be stated, not inferred.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("proba,expected", [
    (0.4991, "worse"), (0.4700, "worse"), (0.5009, "better"), (0.5600, "better"),
])
def test_lean_direction_matches_the_probability(proba, expected):
    ev = build_evidence(_forecast(proba), [], [], {}, {})
    assert expected in ev["which_way_it_leans"], (
        f"p={proba} described as {ev['which_way_it_leans']!r}")


def test_lean_is_stated_even_when_abstaining():
    """An abstaining forecast still has a tilt, and the page describes it. The
    direction must still be right."""
    ev = build_evidence(_forecast(0.4991, acted=False), [], [], {}, {})
    assert "worse" in ev["which_way_it_leans"]
    assert "refuses to act" in ev["lean_is_this_slight"]


def test_lean_is_never_shouted():
    ev = build_evidence(_forecast(0.47), [], [], {}, {})
    assert ev["which_way_it_leans"] == ev["which_way_it_leans"].lower()


# --------------------------------------------------------------------------
# Points and probability can genuinely disagree; the page must know.
# --------------------------------------------------------------------------
def test_disagreement_between_points_and_probability_is_flagged():
    """Aspect points are the boosted model's attribution; the headline
    probability blends four models. They can point opposite ways, and the page
    has to say so rather than read as a contradiction."""
    ev = build_evidence(_forecast(0.4991), [], [], {}, {})
    assert ev["tally"]["net_points"] > 0            # points lean up
    assert ev["probability_it_outperforms_pct"] < 50  # probability leans down
    assert ev["points_and_probability_agree"] is False


def test_agreement_is_flagged_when_they_do_agree():
    ev = build_evidence(_forecast(0.56), [], [], {}, {})
    assert ev["points_and_probability_agree"] is True


# --------------------------------------------------------------------------
# Reversal is the real edge, so it has to be detectable.
# --------------------------------------------------------------------------
def test_reversal_is_flagged_when_a_good_reading_scores_negative():
    f = _forecast(0.49)
    f["aspects"] = [{
        "aspect": "Sector & peers", "points": -44, "direction": "down",
        "share_of_signal": 0.3,
        "evidence": [{"feature": "rs_sector_5d", "shap": -0.005, "value": 0.0138}],
    }]
    ev = build_evidence(f, [], [], {}, {})
    assert ev["reasons"][0]["is_reversal"] is True, (
        "a favourable reading with negative points is the reversal effect and "
        "must be flagged, or the prose apologises for it instead of explaining it")


# --------------------------------------------------------------------------
# Glossary coverage: an undescribed feature would be guessed at.
# --------------------------------------------------------------------------
def test_glossary_covers_every_model_feature():
    import pandas as pd

    from cef.config import DATA_DIR
    from cef.features.build import feature_columns

    path = DATA_DIR / "panel.parquet"
    if not path.exists():
        pytest.skip("panel.parquet missing")
    feats = feature_columns(pd.read_parquet(path))
    known, unknown = coverage(feats)
    assert not unknown, f"{len(unknown)} features have no glossary entry: {unknown[:8]}"
    assert len(known) == len(feats)


def test_glossary_entries_have_both_directions():
    for f in ("dollar_vol_z", "driver_ret5", "us10y_level", "rs_sector_5d"):
        d = describe(f)
        assert d and d["when_high"] and d["when_low"] and d["means"]
        assert d["when_high"] != d["when_low"]


# --------------------------------------------------------------------------
# Number verification compares magnitudes, not signed strings.
# --------------------------------------------------------------------------
def test_verifier_accepts_a_faithfully_unsigned_number():
    """The evidence stores points_pulling_down as -70 and good prose says
    "70 points against". A signed comparison calls that a fabrication."""
    ev = build_evidence(_forecast(0.49), [], [], {}, {})
    out = {"headline": "", "opening": "", "confidence_line": "",
           "closing": "100 points for and 40 points against.",
           "reasons": [], "explainer": {"paragraphs": []}, "levels": []}
    assert verify(out, ev) == []


def test_verifier_catches_an_invented_number():
    ev = build_evidence(_forecast(0.49), [], [], {}, {})
    out = {"headline": "", "opening": "", "confidence_line": "",
           "closing": "Profit rose 145% last quarter.",
           "reasons": [], "explainer": {"paragraphs": []}, "levels": []}
    assert "145" in verify(out, ev)


# --------------------------------------------------------------------------
# Field names are plumbing and must never reach the reader.
# --------------------------------------------------------------------------
def test_prompt_does_not_name_internal_fields_the_reader_could_see():
    """The prompt used to say "if positive use `when_high`, if negative use
    `when_low`" — and the model dutifully wrote "when_low it has been sliding"
    into a reason card. Naming a plumbing key in the instructions invites it
    into the prose, so the evidence resolves the reading itself and the prompt
    never mentions the key.

    `measures` and `meaning` are deliberately excluded from this check: they
    are ordinary English words a good sentence may legitimately contain
    ("both measures of the market show weakness").
    """
    from cef.evidence.simple import SYSTEM_PROMPT

    for key in ("when_high", "when_low", "is_reversal",
                "what_it_looked_at", "points_and_probability_agree"):
        assert f"`{key}`" not in SYSTEM_PROMPT or "NEVER WRITE A FIELD NAME" in SYSTEM_PROMPT, (
            f"the prompt names {key} in a way that invites it into the output")
    assert "NEVER WRITE A FIELD NAME" in SYSTEM_PROMPT


def test_described_readings_carry_resolved_meaning_not_keys():
    """Each reading handed to the model already holds the correct meaning for
    its value, so the model never has to choose between two glossary keys."""
    ev = build_evidence(_forecast(0.49), [], [], {}, {})
    looked = ev["reasons"][0]["what_it_looked_at"]
    assert looked, "a reason with evidence should describe what it looked at"
    for r in looked:
        assert set(r) == {"measures", "reading", "meaning", "value"}
        assert r["reading"] in {"high", "low"}
        assert r["meaning"] and "when_" not in r["meaning"]
