"""Server-sent event stream for the live analysis run.

The interface shows the pipeline working rather than a spinner, so every stage
that genuinely takes time emits an event as it starts and again as it finishes:
what it is doing, what it found, and how long it took. Nothing here is
theatre — if a stage is fast it reports fast, and the elapsed milliseconds are
measured rather than invented. A progress bar that lies about what a system is
doing is worse than no progress bar.

The pipeline stages are CPU-bound and synchronous (a gradient-boosted fit, a
SHAP pass), so each is pushed to a worker thread. Without that the event loop
would block and the very events describing the work would arrive in one burst
at the end.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import AsyncIterator, Callable

from cef.api import dashboard as dash
from cef.api import service

log = logging.getLogger(__name__)


def sse(event: dict) -> str:
    return f"data: {json.dumps(event, default=str)}\n\n"


class Stage:
    """One pipeline stage: a label, a worker, and a plain-English summariser.

    ``summarise`` receives the worker's result and returns the sentence shown
    on the right-hand side of the screen. Keeping it beside the work means the
    explanation cannot drift away from what was actually computed.
    """

    def __init__(self, key: str, label: str, doing: str,
                 work: Callable[[dict], object],
                 summarise: Callable[[object, dict], str],
                 weight: float = 1.0) -> None:
        self.key, self.label, self.doing = key, label, doing
        self.work, self.summarise, self.weight = work, summarise, weight


# ── stage definitions ────────────────────────────────────────────────────────
def _stages(symbol: str) -> list[Stage]:
    def load(ctx):
        p = service.panel()
        sub = p[p["symbol"] == symbol]
        if sub.empty:
            raise KeyError(symbol)
        ctx["as_of"] = sub["date"].max()
        return {"rows": len(sub), "sessions": int(sub["date"].nunique()),
                "from": str(sub["date"].min().date()), "to": str(sub["date"].max().date()),
                "universe": int(p["symbol"].nunique())}

    def fit(ctx):
        t = time.perf_counter()
        ens = service.ensemble_for(ctx["as_of"])
        ctx["ens"] = ens
        return {"n_train": ens.n_train_, "through": ens.trained_through_,
                "n_features": len(ens.features_), "cached": (time.perf_counter() - t) < 0.5,
                "top_features": ens.features_[:6]}

    def forecast(ctx):
        f = service.forecast(symbol)
        ctx["f"] = f
        return f

    def levels(ctx):
        r = dash.key_levels(symbol)
        ctx["levels"] = r
        return r

    def drivers(ctx):
        r = dash.driver_linkage(symbol)
        ctx["drivers"] = r
        return r

    def moves(ctx):
        r = service.attribution_for(symbol, 8)
        ctx["attribution"] = r
        return r

    def coverage(ctx):
        r = dash.source_coverage(symbol)
        ctx["coverage"] = r
        return r

    def board(ctx):
        r = dash.board()
        ctx["board"] = r
        return r

    def price(ctx):
        r = dash.price_series(symbol)
        ctx["price"] = r
        return r

    def simple(ctx):
        r = service.simple_for(symbol)
        ctx["simple"] = r
        return r

    # --- plain-English summarisers ------------------------------------------
    def s_load(r, ctx):
        return (f"Loaded {r['sessions']} trading sessions for {symbol}, "
                f"{r['from']} to {r['to']}, alongside {r['universe'] - 1} other "
                f"large-caps to compare it against.")

    def s_fit(r, ctx):
        if r["cached"]:
            return (f"Reused the model already fitted for this session — trained on "
                    f"{r['n_train']:,} rows up to {r['through']}, using "
                    f"{r['n_features']} of 200 available measures.")
        return (f"Trained four models on {r['n_train']:,} rows of history ending "
                f"{r['through']} — nothing after that date. Narrowed 200 measures "
                f"down to the {r['n_features']} that carried signal.")

    def s_forecast(r, ctx):
        p = r["proba_outperform"] * 100
        if not r["acted"]:
            return (f"The four models land at {p:.1f}% — {r['conviction'] * 100:.2f} "
                    f"points from a coin toss, inside the band where this system "
                    f"declines to take a side.")
        side = "beat" if p > 50 else "lag"
        return (f"The four models agree closely enough to call it: {p:.1f}% that "
                f"{symbol} will {side} the middle of the pack, at "
                f"{r['confidence'] * 100:.1f}% confidence.")

    def s_levels(r, ctx):
        return (f"Worked out the prices that matter from the last session's high, "
                f"low and close: ₹{r['s1']:,.2f} on the downside, ₹{r['pivot']:,.2f} "
                f"as the line it hovers around, ₹{r['r1']:,.2f} on the upside.")

    def s_drivers(r, ctx):
        if not r:
            return "No commodity or currency drivers are mapped to this name."
        top = r[0]
        return (f"Checked what this share actually tracks. The closest link is "
                f"{top['driver']} at {top['corr_120d']:+.2f} over six months, "
                f"which last moved {top['last_chg_pct']:+.2f}%.")

    def s_moves(r, ctx):
        if not r:
            return "No move beyond two standard deviations in the recorded window."
        explained = sum(1 for m in r if m["kind"] != "unexplained")
        return (f"Found {len(r)} unusually large moves and searched exchange filings "
                f"around each. {explained} could be matched to something concrete; "
                f"{len(r) - explained} are left unexplained rather than given a story.")

    def s_coverage(r, ctx):
        share = r.get("explained_share")
        return (f"Searched {r['filings_total']:,} exchange filings for this company. "
                f"Across its {r['significant_moves']} big moves, "
                f"{int((share or 0) * 100)}% have an identifiable cause.")

    def s_board(r, ctx):
        movers = sorted(r, key=lambda t: -abs(t["chg_pct"]))[:2]
        return ("Read the wider market and the commodity board: "
                + ", ".join(f"{m['label']} {m['chg_pct']:+.2f}%" for m in movers)
                + ". Anything that closes after India does is held back a session.")

    def s_price(r, ctx):
        return (f"Charted {r['sessions']} sessions of price and marked the moves "
                f"that had evidence attached to them.")

    def s_simple(r, ctx):
        n = len(r.get("reasons", []))
        bad = len(r.get("unverified_numbers", []))
        return (f"Wrote the plain-language explanation from that evidence — {n} "
                f"factors, each with its reason. "
                + ("Every number in it traced back to the evidence."
                   if not bad else f"{bad} number(s) could not be traced and are flagged."))

    return [
        Stage("load", "Loading price history", "reading the exchange data", load, s_load, 1),
        Stage("board", "Reading the market board", "global and local markets", board, s_board, 1),
        Stage("fit", "Fitting the models", "training on history only", fit, s_fit, 4),
        Stage("forecast", "Asking the models", "blending four opinions", forecast, s_forecast, 3),
        Stage("levels", "Working out key prices", "from the last session", levels, s_levels, 1),
        Stage("drivers", "Checking what it tracks", "commodities and currencies", drivers, s_drivers, 1),
        Stage("moves", "Searching filings", "matching news to big moves", moves, s_moves, 2),
        Stage("coverage", "Counting the evidence", "how much can be explained", coverage, s_coverage, 1),
        Stage("price", "Charting the history", "price with event markers", price, s_price, 1),
        Stage("simple", "Writing it in plain English", "no jargon", simple, s_simple, 5),
    ]


async def analyse(symbol: str) -> AsyncIterator[str]:
    symbol = symbol.upper()
    stages = _stages(symbol)
    total_weight = sum(s.weight for s in stages)
    ctx: dict = {}
    done_weight = 0.0
    t0 = time.perf_counter()

    yield sse({"type": "start", "symbol": symbol,
               "stages": [{"key": s.key, "label": s.label, "doing": s.doing}
                          for s in stages]})

    for stage in stages:
        yield sse({"type": "stage", "key": stage.key, "status": "running",
                   "label": stage.label, "doing": stage.doing,
                   "pct": round(done_weight / total_weight * 100)})
        started = time.perf_counter()
        try:
            result = await asyncio.to_thread(stage.work, ctx)
        except KeyError:
            yield sse({"type": "error", "key": stage.key,
                       "message": f"{symbol} is not in the universe of 52 symbols."})
            return
        except Exception as exc:                        # noqa: BLE001
            log.exception("stage %s failed", stage.key)
            # A failed stage is reported and the run continues. The simple view
            # needs a language-model key, and losing it should not take the
            # numbers down with it.
            yield sse({"type": "stage", "key": stage.key, "status": "failed",
                       "label": stage.label, "summary": f"Could not complete: {str(exc)[:180]}",
                       "ms": int((time.perf_counter() - started) * 1000),
                       "pct": round((done_weight + stage.weight) / total_weight * 100)})
            done_weight += stage.weight
            continue

        try:
            summary = stage.summarise(result, ctx)
        except Exception:                               # noqa: BLE001
            summary = "Completed."
        done_weight += stage.weight
        yield sse({"type": "stage", "key": stage.key, "status": "done",
                   "label": stage.label, "summary": summary,
                   "ms": int((time.perf_counter() - started) * 1000),
                   "pct": round(done_weight / total_weight * 100)})

    f = ctx.get("f") or {}
    yield sse({
        "type": "final",
        "elapsed_ms": int((time.perf_counter() - t0) * 1000),
        "payload": {
            "forecast": f,
            "simple": ctx.get("simple"),
            "levels": ctx.get("levels"),
            "drivers": ctx.get("drivers"),
            "attribution": ctx.get("attribution"),
            "coverage": ctx.get("coverage"),
            "board": ctx.get("board"),
            "price": ctx.get("price"),
            "not_built": dash.NOT_BUILT,
        },
    })
