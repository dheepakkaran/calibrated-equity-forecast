"""Central configuration. Everything path- or date-related lives here."""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts"
DB_PATH = DATA_DIR / "cef.sqlite"

for _d in (DATA_DIR, ARTIFACT_DIR):
    _d.mkdir(exist_ok=True)

# --- data window -------------------------------------------------------------
HISTORY_START = date(2018, 1, 1)
HISTORY_END = date(2026, 9, 6)

# --- API keys (optional at ingest time; each source degrades gracefully) -----
FRED_API_KEY = os.getenv("FRED_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")


@dataclass(frozen=True)
class WalkForwardConfig:
    """Expanding-window walk-forward schedule.

    ``embargo_days`` purges samples between train and test. The direction target
    at date t depends on close[t+1], so a train sample dated one day before the
    test window would have its label drawn from inside that window. One day is
    the minimum correct embargo; we use two for safety around holidays.
    """

    first_test_start: date = date(2019, 9, 1)
    test_months: int = 6
    min_train_days: int = 400
    embargo_days: int = 2


WALK_FORWARD = WalkForwardConfig()
