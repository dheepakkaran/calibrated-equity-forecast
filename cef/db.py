"""SQLite persistence.

Deliberately plain ``sqlite3`` rather than an ORM. The workload is a research
panel of roughly 80k rows written by batch jobs and read wholesale into pandas;
an ORM's unit-of-work machinery buys nothing here. All reads and writes go
through pandas, so swapping the backend later means changing ``connect`` and
the DDL only.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterator

import pandas as pd

from cef.config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS ohlcv (
    symbol      TEXT    NOT NULL,
    date        TEXT    NOT NULL,
    open        REAL,
    high        REAL,
    low         REAL,
    close       REAL,
    adj_close   REAL,
    volume      REAL,
    source      TEXT    NOT NULL,
    ingested_at TEXT    NOT NULL,
    PRIMARY KEY (symbol, date)
);
CREATE INDEX IF NOT EXISTS ix_ohlcv_date ON ohlcv(date);

CREATE TABLE IF NOT EXISTS macro (
    key         TEXT    NOT NULL,
    date        TEXT    NOT NULL,
    close       REAL,
    source      TEXT    NOT NULL,
    ingested_at TEXT    NOT NULL,
    PRIMARY KEY (key, date)
);
CREATE INDEX IF NOT EXISTS ix_macro_date ON macro(date);

-- Corporate actions with ex-dates. Needed because Yahoo's adjusted close does
-- not correct for demergers: VEDL shows a -65% "return" on its 2026-04-30
-- demerger ex-date, identical in `close` and `adj_close`. Left uncorrected
-- that single row would corrupt a label and pollute 60 sessions of trailing
-- volatility.
CREATE TABLE IF NOT EXISTS corporate_actions (
    symbol      TEXT    NOT NULL,
    ex_date     TEXT    NOT NULL,
    subject     TEXT,
    kind        TEXT,                   -- structural | dividend | meeting | other
    source      TEXT    NOT NULL,
    ingested_at TEXT    NOT NULL,
    PRIMARY KEY (symbol, ex_date, subject)
);
CREATE INDEX IF NOT EXISTS ix_ca_symbol ON corporate_actions(symbol, ex_date);

-- Price-series breaks that are capital-structure changes rather than returns.
CREATE TABLE IF NOT EXISTS price_breaks (
    symbol      TEXT    NOT NULL,
    date        TEXT    NOT NULL,
    ret         REAL,
    implied_ratio REAL,
    reason      TEXT,
    confirmed_by TEXT,
    created_at  TEXT    NOT NULL,
    PRIMARY KEY (symbol, date)
);

-- Corporate filings from the NSE announcements API. The backbone of the
-- evidence layer: authoritative, precisely timestamped, categorised, and each
-- row carries the URL of the actual filing so any claim can be checked.
CREATE TABLE IF NOT EXISTS announcements (
    seq_id      TEXT    NOT NULL PRIMARY KEY,
    symbol      TEXT    NOT NULL,
    announced_at TEXT   NOT NULL,       -- ISO datetime, IST
    date        TEXT    NOT NULL,       -- trading date the filing can first act on
    category    TEXT,
    body        TEXT,
    attachment  TEXT,                   -- source URL, for traceability
    industry    TEXT,
    after_close INTEGER NOT NULL,       -- 1 if filed at/after 15:30 IST
    source      TEXT    NOT NULL,
    ingested_at TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_ann_symbol_date ON announcements(symbol, date);

-- Headlines with FinBERT sentiment. Forward-looking only: RSS exposes a few
-- days of history at most, so unlike `announcements` this table cannot be
-- backfilled and grows from the day ingest starts.
CREATE TABLE IF NOT EXISTS news (
    url_hash    TEXT    NOT NULL PRIMARY KEY,
    symbol      TEXT,                   -- NULL for market-wide items
    published_at TEXT   NOT NULL,
    date        TEXT    NOT NULL,
    headline    TEXT    NOT NULL,
    summary     TEXT,
    outlet      TEXT,
    url         TEXT,
    finbert_label TEXT,
    finbert_score REAL,                 -- signed: positive minus negative
    source      TEXT    NOT NULL,
    ingested_at TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_news_symbol_date ON news(symbol, date);

-- Significant moves matched to the event most likely to explain them.
CREATE TABLE IF NOT EXISTS attributions (
    symbol      TEXT    NOT NULL,
    date        TEXT    NOT NULL,
    ret         REAL,                   -- absolute log return
    ret_rel     REAL,                   -- return net of the cross-sectional median
    sigma       REAL,                   -- move size in trailing standard deviations
    market_wide INTEGER,
    kind        TEXT,                   -- announcement | driver | unexplained
    evidence_id TEXT,                   -- announcements.seq_id or a driver key
    headline    TEXT,
    source_url  TEXT,
    confidence  REAL,
    rationale   TEXT,
    created_at  TEXT    NOT NULL,
    PRIMARY KEY (symbol, date)
);

-- Live forecasts. Written before the target session opens and never touched
-- again except to attach an outcome, so the record of what was claimed cannot
-- drift after the fact. `feature_snapshot` stores the exact inputs, which is
-- what makes a past forecast auditable rather than merely logged.
CREATE TABLE IF NOT EXISTS predictions (
    id            TEXT    NOT NULL PRIMARY KEY,
    symbol        TEXT    NOT NULL,
    as_of_date    TEXT    NOT NULL,      -- session whose close produced it
    target_date   TEXT    NOT NULL,      -- session being forecast
    target        TEXT    NOT NULL,      -- y_rel etc
    direction     TEXT    NOT NULL,      -- outperform | underperform | abstain
    proba         REAL    NOT NULL,      -- calibrated P(outperform)
    confidence    REAL    NOT NULL,      -- max(p, 1-p)
    conviction    REAL,                  -- |proba - 0.5|, drives selection
    acted         INTEGER NOT NULL,      -- 1 if above the abstention threshold
    regime        TEXT,
    vol_bucket    TEXT,
    trend_bucket  TEXT,
    arm_weights   TEXT,                  -- JSON: bandit weights used
    arm_probas    TEXT,                  -- JSON: each arm's raw output
    aspects       TEXT,                  -- JSON: bucketed SHAP
    feature_snapshot TEXT,               -- JSON
    model_version TEXT,
    status        TEXT    NOT NULL,      -- PENDING | RESOLVED | VOID
    created_at    TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_pred_target ON predictions(target_date, status);
CREATE INDEX IF NOT EXISTS ix_pred_symbol ON predictions(symbol, as_of_date);

CREATE TABLE IF NOT EXISTS outcomes (
    prediction_id TEXT NOT NULL PRIMARY KEY,
    actual_ret    REAL,
    actual_ret_rel REAL,
    actual_dir    TEXT,
    correct       INTEGER,
    reward        REAL,
    arm_rewards   TEXT,                  -- JSON: per-arm reward, for the bandit
    resolved_at   TEXT NOT NULL
);

-- Beta-Bernoulli posterior per (arm, regime). Thompson sampling draws from
-- these; the resolver updates them.
CREATE TABLE IF NOT EXISTS bandit_state (
    arm        TEXT    NOT NULL,
    regime     TEXT    NOT NULL,
    alpha      REAL    NOT NULL,
    beta       REAL    NOT NULL,
    n_pulls    INTEGER NOT NULL,
    updated_at TEXT    NOT NULL,
    PRIMARY KEY (arm, regime)
);

-- Drift and guardrail events, kept so a fallback can be explained later.
CREATE TABLE IF NOT EXISTS guardrail_log (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    at         TEXT NOT NULL,
    kind       TEXT NOT NULL,
    regime     TEXT,
    detail     TEXT
);

-- One row per (walk-forward run, fold, model). The permanent record of every
-- number this project has ever reported.
CREATE TABLE IF NOT EXISTS wf_metrics (
    run_id      TEXT    NOT NULL,
    fold        INTEGER NOT NULL,
    model       TEXT    NOT NULL,
    train_start TEXT,
    train_end   TEXT,
    test_start  TEXT,
    test_end    TEXT,
    n_train     INTEGER,
    n_test      INTEGER,
    accuracy    REAL,
    auc         REAL,
    brier       REAL,
    log_loss    REAL,
    created_at  TEXT    NOT NULL,
    PRIMARY KEY (run_id, fold, model)
);

-- Per-row out-of-sample predictions, kept so calibration curves and per-regime
-- breakdowns can be recomputed without refitting.
CREATE TABLE IF NOT EXISTS wf_predictions (
    run_id     TEXT    NOT NULL,
    model      TEXT    NOT NULL,
    symbol     TEXT    NOT NULL,
    date       TEXT    NOT NULL,
    fold       INTEGER NOT NULL,
    proba      REAL,
    y_true     INTEGER,
    PRIMARY KEY (run_id, model, symbol, date)
);
"""


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with connect() as conn:
        conn.executescript(SCHEMA)


def upsert(table: str, df: pd.DataFrame, keys: list[str]) -> int:
    """Idempotent insert-or-replace. Re-running any ingest job is safe."""
    if df.empty:
        return 0
    cols = list(df.columns)
    placeholders = ",".join("?" * len(cols))
    sql = f"INSERT OR REPLACE INTO {table} ({','.join(cols)}) VALUES ({placeholders})"
    rows = [tuple(None if pd.isna(v) else v for v in rec) for rec in df.itertuples(index=False, name=None)]
    with connect() as conn:
        conn.executemany(sql, rows)
    return len(rows)


def read_ohlcv(symbols: list[str] | None = None) -> pd.DataFrame:
    q = "SELECT symbol, date, open, high, low, close, adj_close, volume FROM ohlcv"
    params: list = []
    if symbols:
        q += f" WHERE symbol IN ({','.join('?' * len(symbols))})"
        params = symbols
    q += " ORDER BY symbol, date"
    with connect() as conn:
        df = pd.read_sql(q, conn, params=params, parse_dates=["date"])
    return df


def read_macro() -> pd.DataFrame:
    """Returns the macro table pivoted wide: index=date, columns=series key."""
    with connect() as conn:
        df = pd.read_sql("SELECT key, date, close FROM macro", conn, parse_dates=["date"])
    if df.empty:
        return pd.DataFrame()
    return df.pivot(index="date", columns="key", values="close").sort_index()


def table_summary() -> pd.DataFrame:
    with connect() as conn:
        names = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
        rows = []
        for t in names:
            n = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            span = ("", "")
            if n:
                try:
                    span = conn.execute(f"SELECT MIN(date), MAX(date) FROM {t}").fetchone()
                except sqlite3.OperationalError:
                    pass
            rows.append({"table": t, "rows": n, "from": span[0] or "-", "to": span[1] or "-"})
    return pd.DataFrame(rows)
