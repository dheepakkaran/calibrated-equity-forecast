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
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from cef.api import dashboard as dash
from cef.api import service
from cef.api import stream as stream_mod
from cef.config import ROOT

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                    datefmt="%H:%M:%S")

app = FastAPI(
    title="Calibrated Selective Prediction for Next-Session Equity Direction",
    description="Evidence from the NSE. Educational and research use only.",
    version="0.3.0",
)

STATIC = ROOT / "cef" / "api" / "static"
WEBDIST = ROOT / "cef" / "api" / "webdist"


class TrackRequest(BaseModel):
    symbol: str
    note: str = Field("", max_length=280)


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


@app.get("/api/analyse/{symbol}")
async def analyse(symbol: str) -> StreamingResponse:
    """Server-sent events for the live analysis run.

    ``X-Accel-Buffering: no`` matters: without it a reverse proxy will hold the
    whole stream and deliver it in one burst at the end, which defeats the
    entire point of streaming the stages.
    """
    return StreamingResponse(
        stream_mod.analyse(symbol),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no",
                 "Connection": "keep-alive"},
    )


@app.post("/api/track")
def track(req: TrackRequest) -> dict:
    """Record a guess for tomorrow morning."""
    from cef.tracking import summarise, track as do_track

    try:
        f = service.forecast(req.symbol)
    except KeyError:
        raise HTTPException(404, f"{req.symbol.upper()} is not in the universe")
    row = do_track(f, req.note)
    return {"tracked": row, "summary": summarise()}


@app.get("/api/tracking")
def tracking(limit: int = Query(50, le=500)) -> dict:
    """The ledger and its running summary."""
    from cef.tracking import load, summarise

    rows = load()
    rows.sort(key=lambda r: r["tracked_at"], reverse=True)
    return {"summary": summarise(), "rows": rows[:limit]}


@app.get("/api/simple/{symbol}")
def simple(symbol: str, prefer: str = Query("gemini", pattern="^(gemini|openai)$")) -> dict:
    """Plain-language view. Gemini first (free tier), OpenAI as fallback."""
    try:
        return service.simple_for(symbol, prefer)
    except KeyError:
        raise HTTPException(404, f"{symbol.upper()} is not in the universe")
    except RuntimeError as exc:
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


@app.get("/api/board")
def board() -> list[dict]:
    """Macro strip. Each tile carries its availability lag."""
    return dash.board()


@app.get("/api/dashboard/{symbol}")
def dashboard(symbol: str) -> dict:
    """Everything the dashboard needs for one symbol, in one round trip.

    Bundled deliberately: the panels are read together and six separate
    requests would each pay the same panel-load cost for no benefit.
    """
    try:
        fc = service.forecast(symbol)
    except KeyError:
        raise HTTPException(404, f"{symbol.upper()} is not in the universe")
    return {
        "forecast": fc,
        "price": dash.price_series(symbol),
        "levels": dash.key_levels(symbol),
        "drivers": dash.driver_linkage(symbol),
        "timeline": dash.timeline(symbol),
        "coverage": dash.source_coverage(symbol),
        "attribution": service.attribution_for(symbol, 8),
        "not_built": dash.NOT_BUILT,
    }


@app.get("/")
def index() -> FileResponse:
    """The React app: streaming analysis, three tabs, tracking.

    Falls back to the dashboard when the bundle has not been built, so a fresh
    clone that has not run `npm run build` still serves something useful.
    """
    built = WEBDIST / "index.html"
    return FileResponse(built if built.exists() else STATIC / "dashboard.html")


@app.get("/dashboard")
def dashboard_page() -> FileResponse:
    """The multi-panel dashboard."""
    return FileResponse(STATIC / "dashboard.html")


@app.get("/flow")
def flow() -> FileResponse:
    """The guided single-question flow."""
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
if (WEBDIST / "assets").exists():
    app.mount("/assets", StaticFiles(directory=WEBDIST / "assets"), name="assets")
