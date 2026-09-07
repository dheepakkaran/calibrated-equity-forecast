"""Tests for the evidence layer.

The evidence layer's failure mode is not a crash, it is a plausible story
attached to the wrong thing. These tests therefore pin down the judgement
calls: what counts as material, when a filing may be attributed to a session,
and what the narrator is allowed to say.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cef.evidence.aspects import aspect_of, bucket_shap
from cef.evidence.event_map import classify
from cef.evidence.moves import detect_moves
from cef.evidence.narrate import verify_narration
from cef.features.breaks import action_implied_ratio, detect_breaks
from cef.ingest.announcements import clean_text


# --------------------------------------------------------------------------
# Materiality
# --------------------------------------------------------------------------
@pytest.mark.parametrize("category", [
    "Loss of Share Certificates", "Copy of Newspaper Publication",
    "Trading Window", "ESOP/ESOS/ESPS", "Newspaper Advertisements",
])
def test_clerical_filings_are_never_material(category):
    """A price move must never be explained by a lost share certificate."""
    assert classify(category).materiality == 0.0


def test_clerical_filings_resist_body_escalation():
    """A clerical notice that happens to cite a regulation is still clerical."""
    ec = classify("Loss of Share Certificates",
                  "Intimation under Regulation 30 of SEBI (LODR) Regulations, 2015")
    assert ec.materiality == 0.0


def test_reg30_rescues_a_miscategorised_material_filing():
    """The IndusInd case: the largest idiosyncratic move in the panel was
    filed under 'General Updates'. Regulation 30 is the material-events rule,
    so citing it must lift the filing over the attribution threshold."""
    ec = classify("General Updates",
                  "Indusind Bank Limited has informed the Exchange about disclosure "
                  "under Regulation 30 of SEBI (Listing Obligations and Disclosure "
                  "Requirements) Regulations, 2015")
    assert ec.materiality >= 0.55
    assert "Reg 30" in ec.label


def test_direction_priors_where_the_category_fixes_the_sign():
    assert classify("Buy-back of Equity Shares").direction == 1
    assert classify("Dividend").direction == 1
    assert classify("Issue of Securities").direction == -1
    assert classify("Action(s) taken or orders passed").direction == -1


def test_acquirer_in_an_insolvency_is_not_distress():
    """JSW Steel receiving a letter of intent from a committee of creditors is
    an acquisition, not financial distress."""
    ec = classify("Acquisition",
                  "Acceptance of Letter of Intent issued to JSW Steel by the "
                  "Committee of Creditors")
    assert ec.direction >= 0, f"read as bearish: {ec.label}"


# --------------------------------------------------------------------------
# Causality of filing timestamps
# --------------------------------------------------------------------------
def test_after_close_filings_roll_to_the_next_session():
    from cef.ingest.announcements import _next_session_date

    days = pd.DatetimeIndex(["2026-09-03", "2026-09-04", "2026-09-07"])
    before = _next_session_date(pd.Timestamp("2026-09-03 15:23:00"), days)
    at_close = _next_session_date(pd.Timestamp("2026-09-03 15:30:01"), days)
    friday_night = _next_session_date(pd.Timestamp("2026-09-04 19:45:00"), days)

    assert before == "2026-09-03", "a filing before the close acts the same session"
    assert at_close == "2026-09-04", "a filing at the close cannot act that session"
    assert friday_night == "2026-09-07", "a Friday-evening filing waits for Monday"


# --------------------------------------------------------------------------
# Capital-structure breaks
# --------------------------------------------------------------------------
@pytest.mark.parametrize("subject,expected", [
    ("Bonus 1:2", 2 / 3), ("Bonus 1:1", 0.5), ("Bonus 2:1", 1 / 3),
    ("Split 1:5", 0.2), ("Face Value Split From Rs 10 To Rs 2", 0.2),
    ("Demerger", None), ("Interim Dividend - Rs 11 Per Share", None),
])
def test_action_implied_ratio(subject, expected):
    got = action_implied_ratio(subject)
    if expected is None:
        assert got is None
    else:
        assert got == pytest.approx(expected, abs=1e-9)


def test_break_detection_requires_exchange_evidence():
    """A 40% single-session fall with no corporate action behind it is a price
    move and must survive. An earlier version matched any tidy-looking
    fraction and masked the Adani selloff of February 2023."""
    dates = pd.bdate_range("2026-01-01", periods=12)
    px = np.linspace(1000, 1050, 12)
    px[6:] *= 0.60                                 # a step down, as a demerger looks
    oh = pd.DataFrame({"symbol": "TESTCO", "date": dates, "close": px})
    assert detect_breaks(oh, pd.DataFrame()).empty

    actions = pd.DataFrame([{"symbol": "TESTCO", "ex_date": dates[6].strftime("%Y-%m-%d"),
                             "subject": "Demerger", "kind": "structural"}])
    found = detect_breaks(oh, actions)
    assert len(found) == 1 and found.iloc[0]["confirmed_by"] == "nse_corporate_action"


def test_only_the_closest_break_is_attributed_to_one_action():
    """A genuine crash a week after a demerger is not part of the demerger."""
    dates = pd.bdate_range("2026-01-01", periods=20)
    px = np.full(20, 1000.0)
    px[5:] *= 0.60                                 # demerger step
    px[12:] *= 0.75                                # unrelated crash, 7 sessions later
    oh = pd.DataFrame({"symbol": "TESTCO", "date": dates, "close": px})
    actions = pd.DataFrame([{"symbol": "TESTCO", "ex_date": dates[5].strftime("%Y-%m-%d"),
                             "subject": "Demerger", "kind": "structural"}])
    found = detect_breaks(oh, actions)
    assert len(found) == 1
    assert found.iloc[0]["date"] == dates[5].strftime("%Y-%m-%d")


def test_market_wide_moves_are_not_treated_as_breaks():
    """Fifteen names fell together on 2020-03-23. A shared move is the market,
    never a capital-structure change."""
    dates = pd.bdate_range("2026-01-01", periods=10)
    frames = []
    for i in range(6):
        px = np.full(10, 1000.0)
        px[5:] = 700.0                             # every symbol gaps the same day
        frames.append(pd.DataFrame({"symbol": f"S{i}", "date": dates, "close": px}))
    oh = pd.concat(frames, ignore_index=True)
    assert detect_breaks(oh, pd.DataFrame()).empty


# --------------------------------------------------------------------------
# Move detection
# --------------------------------------------------------------------------
def test_move_thresholds_use_only_prior_history():
    """A large move must not inflate the volatility it is measured against."""
    dates = pd.bdate_range("2024-01-01", periods=140)
    rng = np.random.default_rng(3)
    px = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(dates))))
    px[130:] *= 1.12
    panel = pd.DataFrame({"symbol": "A", "date": dates, "close": px})
    moves = detect_moves(panel, k=2.0)
    assert (moves["date"] == dates[130]).any(), "the jump session should be flagged"


# --------------------------------------------------------------------------
# Aspect bucketing
# --------------------------------------------------------------------------
def test_every_feature_maps_to_an_aspect():
    from cef.config import DATA_DIR
    from cef.features.build import feature_columns

    path = DATA_DIR / "panel.parquet"
    if not path.exists():
        pytest.skip("panel.parquet missing")
    feats = feature_columns(pd.read_parquet(path))
    unmapped = [f for f in feats if aspect_of(f) == "Other"]
    assert not unmapped, f"unmapped features: {unmapped}"


def test_day_of_week_is_calendar_not_dow_jones():
    """`dow` is the weekday feature; `dow_chg1` is the Dow Jones index."""
    assert aspect_of("dow") == "Event & calendar"
    assert aspect_of("dow_chg1") == "Global cues"


def test_opposing_features_cancel_within_an_aspect():
    """Signed summation, not absolute: an aspect whose features disagree
    contributed little and must read that way."""
    row = pd.Series({"rsi14": 0.05, "macd": -0.05, "driver_ret5": 0.02})
    buckets = {b["aspect"]: b for b in bucket_shap(row)}
    assert buckets["Technical"]["contribution"] == pytest.approx(0.0, abs=1e-9)
    assert buckets["Driver linkage"]["contribution"] == pytest.approx(0.02)


# --------------------------------------------------------------------------
# Narration guardrails
# --------------------------------------------------------------------------
def test_narration_verifier_allows_honest_rounding():
    ev = {"confidence": 0.512, "model_track_record": {"held_out_accuracy": 0.5102}}
    text = "Confidence is 51.2%, and past accuracy is about 51%."
    assert verify_narration(text, ev) == []


def test_narration_verifier_handles_unicode_minus():
    """Language models write U+2212, not a hyphen."""
    ev = {"aspects": [{"aspect": "Driver linkage", "points": -16}]}
    assert verify_narration("Driver linkage contributed −16 points.", ev) == []


@pytest.mark.parametrize("text,expected", [
    ("The model sees this reaching Rs 640 next session.", "640"),
    ("The model is right 74% of the time.", "74"),
    ("It sits 23% below its 52-week high.", "23"),
])
def test_narration_verifier_catches_invented_numbers(text, expected):
    ev = {"confidence": 0.512, "last_close": 587.95}
    assert expected in verify_narration(text, ev)


# --------------------------------------------------------------------------
# Ingest hygiene
# --------------------------------------------------------------------------
def test_clean_text_strips_mojibake_and_collapses_space():
    raw = "�Vedanta Limited�has   Submitted to the Exchange\n\n"
    assert clean_text(raw) == "Vedanta Limited has Submitted to the Exchange"
