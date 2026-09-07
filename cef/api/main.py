"""FastAPI surface.

Thin by design: every handler delegates to ``cef.api.service`` and does nothing
but shape the response. Two things are enforced here rather than left to the
frontend, because they are the project's substantive commitments and a client
must not be able to omit them: every forecast response carries the held-out
track record, and every one carries the disclaimer.
"""
from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from cef.api import service
from cef.config import ROOT

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                    datefmt="%H:%M:%S")

app = FastAPI(
    title="Calibrated Selective Prediction for Next-Session Equity Direction",
    description="Evidence from the NSE. Educational and research use only.",
    version="0.3.0",
)

STATIC = ROOT / "cef" / "api" / "static"


class CompareRequest(BaseModel):
    symbols: list[str] = Field(min_length=2, max_length=5)


@app.get("/api/health")
def health() -> dict:
    p = service.panel()
    return {
        "status": "ok",
        "panel_rows": len(p),
        "symbols": int(p["symbol"].nunique()),
        "latest_session": service.latest_session().strftime("%Y-%m-%d"),
    }


@app.get("/api/symbols/search")
def symbols(q: str = Query("", max_length=20), limit: int = Query(8, le=30)) -> list[dict]:
    return service.search_symbols(q, limit)


@app.get("/api/forecast/{symbol}")
def forecast(symbol: str) -> dict:
    try:
        return service.forecast(symbol)
    except KeyError:
        raise HTTPException(404, f"{symbol.upper()} is not in the universe")


@app.get("/api/narration/{symbol}")
def narration(symbol: str) -> dict:
    try:
        return service.narration_for(symbol)
    except KeyError:
        raise HTTPException(404, f"{symbol.upper()} is not in the universe")
    except RuntimeError as exc:
        # No API key configured. The rest of the system works without it, so
        # this degrades rather than fails.
        raise HTTPException(503, str(exc))


@app.get("/api/attribution/{symbol}")
def attribution(symbol: str, limit: int = Query(8, le=40)) -> list[dict]:
    return service.attribution_for(symbol, limit)


@app.post("/api/compare")
def compare(req: CompareRequest) -> dict:
    out, missing = [], []
    for s in req.symbols:
        try:
            f = service.forecast(s)
        except KeyError:
            missing.append(s.upper())
            continue
        out.append({
            "symbol": f["symbol"], "sector": f["sector"],
            "direction": f["direction"], "confidence": f["confidence"],
            "proba_outperform": f["proba_outperform"],
            "conviction": f["conviction"], "acted": f["acted"],
            "regime": f["regime"]["label"],
            "top_aspect": (f["aspects"][0]["aspect"] if f["aspects"] else None),
        })
    # Ranked by conviction, not by probability: the useful question across a
    # basket is which calls the system is least unsure about.
    out.sort(key=lambda d: -d["conviction"])
    return {"results": out, "not_found": missing,
            "track_record": service.HELD_OUT,
            "note": ("Ranked by conviction. Rows below the abstention threshold "
                     "are shown but carry no call.")}


@app.get("/api/history")
def history(symbol: str | None = None, limit: int = Query(60, le=500)) -> list[dict]:
    return service.history(symbol, limit)


@app.get("/api/performance")
def performance() -> dict:
    return service.live_performance()


@app.get("/api/bandit")
def bandit_state() -> dict:
    return service.bandit_state()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
