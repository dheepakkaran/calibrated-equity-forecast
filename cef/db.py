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
