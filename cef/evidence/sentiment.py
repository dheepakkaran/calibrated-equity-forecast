"""FinBERT sentiment scoring for headlines.

``ProsusAI/finbert`` rather than a general-purpose sentiment model or an LLM
call, for three reasons. It is tuned on financial text, where the vocabulary
inverts ordinary sentiment - "profit taking", "cooling inflation" and "rate cut
hopes" all read differently in a market context. It runs locally, so scoring
tens of thousands of headlines costs nothing and needs no network. And it is
deterministic, which matters in a research pipeline: a re-run has to reproduce
the same numbers, and a hosted model behind a moving version string does not
guarantee that.

Scores are collapsed to a single signed number, ``positive - negative``, in
[-1, 1]. The neutral mass is deliberately discarded rather than treated as a
third state: for attribution what matters is direction and strength, and a
headline that is 80% neutral / 20% positive should read as weakly positive
rather than as its own category.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

MODEL = "ProsusAI/finbert"
_pipe = None


def _load():
    global _pipe
    if _pipe is None:
        from transformers import pipeline
        log.info("loading %s (first run downloads ~440MB)", MODEL)
        _pipe = pipeline("text-classification", model=MODEL,
                         top_k=None, truncation=True, max_length=256)
    return _pipe


def score_texts(texts: list[str], batch_size: int = 32) -> pd.DataFrame:
    """Return one row per text with a signed score and the dominant label."""
    if not texts:
        return pd.DataFrame(columns=["finbert_label", "finbert_score"])
    pipe = _load()
    out = []
    for i in range(0, len(texts), batch_size):
        chunk = [t[:1000] for t in texts[i:i + batch_size]]
        for scores in pipe(chunk):
            d = {s["label"].lower(): s["score"] for s in scores}
            signed = d.get("positive", 0.0) - d.get("negative", 0.0)
            out.append({
                "finbert_label": max(d, key=d.get),
                "finbert_score": float(np.clip(signed, -1.0, 1.0)),
            })
    return pd.DataFrame(out)


def score_unscored_news(limit: int | None = None) -> int:
    """Score any news rows that do not yet carry a sentiment. Idempotent."""
    from cef.db import connect

    q = "SELECT url_hash, headline FROM news WHERE finbert_score IS NULL"
    if limit:
        q += f" LIMIT {int(limit)}"
    with connect() as conn:
        todo = pd.read_sql(q, conn)
    if todo.empty:
        log.info("no unscored headlines")
        return 0

    scored = score_texts(todo["headline"].tolist())
    todo = pd.concat([todo.reset_index(drop=True), scored], axis=1)

    with connect() as conn:
        conn.executemany(
            "UPDATE news SET finbert_label=?, finbert_score=? WHERE url_hash=?",
            list(zip(todo["finbert_label"], todo["finbert_score"], todo["url_hash"])))
    log.info("scored %d headlines  mean %+.3f", len(todo), todo["finbert_score"].mean())
    return len(todo)
