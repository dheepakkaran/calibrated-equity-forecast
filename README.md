# Calibrated Selective Prediction for Next-Session Equity Direction

### Evidence from the NSE

Next-session directional forecasting for Indian equities, built so that the
reported accuracy can be trusted. The engineering claim is point-in-time
correctness, honest validation and calibrated confidence — not alpha.

> Educational and research use only. Not investment advice. Directional
> forecasts for equities are inherently low-confidence, and this system
> reports its own accuracy openly, including the parts that do not work.

---

## Headline result — M1 (modelling)

Held-out folds (2024-03 → 2026-08), never read during development:

| | accuracy | edge vs baseline | fold t-stat | folds won | AUC |
|---|---|---|---|---|---|
| LightGBM (raw) | **51.02%** | **+1.01 pp** | 3.01 | 5/5 | 0.515 |
| LightGBM + isotonic | 51.03% | +1.02 pp | 2.65 | 5/5 | 0.514 |
| Logistic + isotonic | 51.00% | +0.99 pp | 2.25 | 4/5 | 0.510 |
| 1-day reversal rule | 50.91% | +0.90 pp | 2.81 | 5/5 | 0.509 |
| Coin flip / always-up | 50.01% | — | — | — | — |

Under selective prediction — forecasting only when conviction is high:

| coverage | accuracy | edge vs baseline on the same rows | t-stat | folds won |
|---|---|---|---|---|
| 5% | **54.80%** | +0.76 pp | 0.97 | 3/5 |
| 10% | 53.62% | +1.64 pp | 1.27 | 3/5 |
| 20% | 52.52% | +1.91 pp | 1.65 | 4/5 |
| 50% | 51.80% | +1.72 pp | 2.03 | 4/5 |
| 100% | 51.04% | +1.02 pp | 3.42 | 5/5 |

Calibration, held-out: **ECE 0.69 pp pooled**, 1.88 pp as a per-fold mean.
The reliability curve sits within about one percentage point of the diagonal
across the range the model actually uses. Predicted probabilities span roughly
0.46 to 0.53 — the model never claims high confidence, which is the intended
behaviour and constrains what the interface is allowed to show.

**Read this next to the numbers above:** a 200-feature gradient-boosted model
is matched on raw accuracy by a two-parameter reversal rule. That is the
honest summary of M1 and it is reported here rather than buried.

---

## What was actually learned

Five findings, each of which changed the design or contradicts the original
proposal.

**1. The absolute direction target is a trap.** Predicting `sign(close[t+1] -
close[t])` gives a model nothing to learn. On any Indian session roughly all
52 names move together, so the label is mostly a restatement of "was the index
up", which no per-symbol feature can predict. The first model trained this way
reached 51.72% against a 51.11% always-up baseline — and a coverage analysis
showed the entire apparent edge was drift-riding: on its own top-conviction
rows, simply predicting "up" scored the same. Subtracting the cross-sectional
median forward return removes the common factor and leaves a label that is
50/50 by construction, with nowhere to hide.

**2. Longer horizons are worse, not better.** Signal-to-noise reasoning
predicts that a 5- or 20-session label should be easier than a 1-session one.
Measured, with the embargo correctly widened to cover each label's forward
span, the opposite holds:

| target | AUC | edge | t |
|---|---|---|---|
| 1-session relative | **0.5150** | +0.65 pp | +1.98 |
| 3-session | 0.5000 | −0.47 pp | −1.54 |
| 5-session | 0.4987 | −0.34 pp | −0.60 |
| 10-session | 0.4846 | −1.14 pp | −2.91 |
| 20-session | 0.4930 | −1.25 pp | −2.11 |

The feature set is built from short-horizon descriptions — oscillators,
one-day macro changes, gap structure. They describe days, not weeks. Multi-
session labels also overlap, so widening the embargo to 32 calendar days for
the 20-session target leaves far fewer independent observations than the row
count suggests.

**3. The gap head does not do what the proposal expected.** The proposal
argues gap direction is more predictable than full-day direction. It is not —
it merely has a higher base rate. Overnight returns on this universe are
positive 60.2% of the time (+13.5 bps mean) while intraday returns are
positive only 46.9% of the time (−6.7 bps): the well-documented overnight
drift. A gap model scores 63.56% accuracy against a 63.45% base rate, an edge
of +0.11 pp, at AUC 0.5048 — worse ranking than the 1-session relative head.
The impressive-looking 63% is the base rate, and reporting it without the
comparator would be the single most misleading number this project could
publish.

**4. Hyperparameters barely matter; the target formulation is everything.**
Nine configurations spanning tree depth, learning rate, regularisation and
column sampling produced accuracies from 0.5078 to 0.5114 — a spread narrower
than the fold-to-fold standard deviation. Changing the target moved AUC by
three times as much as every hyperparameter combined. Cutting 200 features to
the top 70 by mean |SHAP| was the one tuning change that helped consistently.

**5. The edge is a one-day relative reversal, and the model is mostly
re-deriving it.** Measured on the held-out window:

```
P(outperforms tomorrow | outperformed today)  = 0.4854
P(outperforms tomorrow | underperformed today) = 0.5084
```

That ~2.3 pp asymmetry is the entire signal. The GBM agrees with this
two-parameter rule on 72.2% of its hard calls.

So why keep the GBM? Because the reversal rule emits exactly two
probabilities and therefore cannot rank. Selective prediction — the 53.48% at
10% coverage above, and the "no useful opinion today" state the interface
needs — requires a continuous, calibrated conviction score. **The GBM's
contribution is conviction, not accuracy.** That is a defensible reason to
keep it and a poor reason to claim it beat the baseline.

---

## Headline result — M2 (evidence and attribution)

Every significant price move is matched, where possible, to a dated corporate
filing or a mapped market driver, and each match carries the source document.

Corpus: **69,747 NSE corporate filings** across all 52 symbols, 2018–2026,
plus corporate-action ex-dates.

| | share of 7,769 significant moves (≥2σ) |
|---|---|
| Explained by a corporate filing | **24.3%** |
| Explained by a mapped driver | 13.6% |
| **Left explicitly unexplained** | **62.1%** |

That last row is the point. Widening the search window until everything has a
story is trivially easy, and the measurements say exactly how easy: the share
of moves with *any* filing nearby runs 47.8% at zero lookback, 65.4% at one
session and 80.7% at three. A system reporting "81% explained" would be citing
lost share certificates and analyst-meet notices. The materiality gate is what
makes the 24.3% mean something, and an unexplained move is reported as
unexplained rather than given a plausible cause.

### Five things the evidence layer turned up

**1. Two thirds of corporate filings arrive after the close.** Of 69,747
filings, 46,260 (66.3%) are timestamped at or after 15:30 IST. Their first
tradeable session is the *next* one. Attribution that matches filings to the
calendar day they were filed therefore misattributes two filings in every
three, so `announcements.date` is resolved at ingest to the first session the
filing could act on — rolling a Friday-evening release to Monday using the
trading calendar.

**2. The exchange's own category field is unreliable for the events that
matter most.** IndusInd Bank's accounting-discrepancy disclosure — a 27%
single-session fall, the largest idiosyncratic move in the panel — was filed
under "General Updates", a category that is otherwise almost entirely
administrative. No category-based materiality gate can catch that, and one
loose enough to admit "General Updates" would admit everything.

The fix is to escalate materiality from the body text, and the strongest single
cue is a reference to **Regulation 30 of SEBI's Listing Obligations and
Disclosure Requirements** — the rule that *defines* a material event and
compels its disclosure. A filing citing it has already been judged material by
the issuer's own compliance officer, which is better evidence than a free-text
category. This recovered the IndusInd disclosure and, in total, **513
attributions (27% of all filing-based matches) come from filings the exchange
had categorised as generic.**

**3. Yahoo's adjusted close does not correct for demergers.** VEDL's
2026-04-30 demerger appears as a −65% single-session return, byte-identical in
`close` and `adj_close`, because value left the entity rather than the share
being subdivided. TRENT's 2026-01-01 session shows a ratio of 0.670 — 2/3 to
within half a percent, an unadjusted 1:2 bonus.

Three such breaks exist in 109,105 rows, and fixing them mattered more than the
count suggests. Masking alone was not enough: every trailing statistic
denominated in rupees kept its pre-event scale while the price did not, so
`atr_pct` jumped from 0.034 to 0.089 and stayed there for weeks — a fabricated
volatility regime. Rebasing the series across each break removed it. **The
correction moved held-out accuracy by only +0.15 pp but lifted the fold
t-statistic from 1.88 to 3.01**, because the spurious rows were contributing
variance rather than bias.

**4. Deciding which breaks to mask is where this could have gone badly wrong.**
An early version of the detector matched any price ratio near a simple
fraction with a denominator up to ten. That set, widened by a tolerance, very
nearly covers the whole interval below 1 — 5/7 is 0.714, 7/9 is 0.778 — so it
masked the Adani selloff of February 2023 as a "split" and would have silently
deleted the largest genuine event in the panel. Unanchored numerology is not
evidence. Detection is now anchored to exchange records: a structural corporate
action within three sessions, or a ratio matching the one a *recorded* action
must produce. What survives is checked as explicitly as what is masked —
2020-03-23, the Adani selloff and the IndusInd disclosure are all still there.

**5. FinBERT is excellent at tone and useless at market structure.** Scored
against a hand-labelled probe set of the *directional implication for the named
stock* (`tests/fixtures/sentiment_probe.py`), it gets 12/20 overall — but the
breakdown is the finding:

| reasoning required | FinBERT |
|---|---|
| Plain tone (sentiment and direction agree) | 6/6 |
| Governance | 2/2 |
| Commodity linkage | 2/4 |
| Policy | 1/2 |
| Shareholder returns | 1/2 |
| **Market structure (stake sales, buybacks, pledges)** | **0/4** |

All four market-structure headlines score below 0.15 and collapse to neutral:
it reads a government stake sale and a share buyback as equally uninformative.
Three outright sign flips, the worst being *"Silver eased as a
stronger-than-expected jobs report lifted Treasury yields"* → **+0.71
positive**, for a company whose earnings depend on silver.

So the architecture follows the measurement rather than the plan. FinBERT
scores headline tone, where it is perfect and free. Commodity and macro
direction comes from **prices**, never from text — silver's direction is a fact
about silver, not a reading of a sentence about it. And the market-structure
class, where FinBERT scores zero, is handled by a deterministic map over
exchange filing categories, which is exact and auditable.

### Narration

Plain-language output is generated by `gpt-5-mini` and is confined to
rewriting: it receives only the structured evidence the pipeline computed, and
never forecasts, computes or ranks. The boundary is enforced rather than
requested — the prompt carries nothing but the evidence, the system prompt caps
the certainty of the language against the confidence figure, and
`verify_narration` scans the output for numbers absent from the input.

The verifier tolerates honest paraphrase (a stored 0.5102 may appear as
"about 51%") and still catches fabrication:

| narration | verdict |
|---|---|
| "Confidence is 51.2%, past accuracy about 51%" | clean |
| "The model sees this reaching Rs 640" | flagged: 640 |
| "The model is right 74% of the time" | flagged: 74 |
| "It sits 23% below its 52-week high" | flagged: 23, 52 |

Two revisions came out of running it. Raw SHAP contributions (~0.02 in
log-odds) were quoted verbatim as *"contributing 0.02205, with
share_of_signal 0.678"*, so aspects are now also expressed as integer points
scaled to the largest contribution — ordinal, and readable. And the verifier
initially flagged its own correct output, because language models write minus
signs as U+2212 rather than an ASCII hyphen.

---

## Headline result — M3 (interface and feedback loop)

The loop runs: forecasts are written before the session they describe, resolved
against what happened, and the outcomes update a contextual bandit that
reweights the ensemble by market regime.

Replayed over 12 months and 30 randomly chosen symbols — 7,464 forecasts
written and resolved:

| | |
|---|---|
| Forecasts made | 7,464 |
| Forecasts carrying an actual call | 3,308 |
| **Abstention rate** | **55.7%** |
| Accuracy on the calls it made | **52.33%** |
| Majority baseline *on those same rows* | 50.45% |
| **Edge** | **+1.87 pp** |
| Mean reward | −0.0017 |

The edge is larger than M1's +1.01 pp at full coverage, and that is not a new
discovery — it is the coverage curve arriving one session at a time. Acting on
44% of sessions is where M1 already predicted roughly +1.7 pp, so the live loop
reproducing it is the machinery working, not the model improving.

**The negative mean reward is by design and worth stating plainly.** The reward
function penalises a confident miss at 1.5× a confident hit, so its expectation
is `conviction × (2.5p − 1.5)`, which only turns positive above **60%**
accuracy. At 52% it is slightly negative and always will be. A reward function
that paid out at 52% would be measuring something other than skill.

A first, shorter replay looked much better — 56.1% over three months and 12
symbols — and it was wrong to believe. The symbols were the alphabetical head
of the universe (Adani-and-banks heavy) and the window was the single best fold
in the whole M1 assessment. Both were fixed: the replay now samples symbols
randomly and runs a year.

### Where the model works, and where it does not

The regime breakdown is the most useful output of the whole loop:

| regime | n | accuracy |
|---|---|---|
| compressed vol · trending down | 100 | **58.00%** |
| compressed vol · range-bound | 371 | **57.14%** |
| normal vol · range-bound | 1,346 | 53.71% |
| elevated vol · trending down | 129 | 52.71% |
| normal vol · trending down | 259 | 49.81% |
| elevated vol · range-bound | 280 | 49.64% |
| **normal vol · trending up** | 476 | **49.16%** |
| **elevated vol · trending up** | 324 | **48.77%** |

It works in quiet, directionless or falling markets and fails outright in
rising and volatile ones. That is consistent with M1's held-out per-regime
table, where elevated volatility was the one regime with a negative edge, and
it is exactly the structure a per-regime bandit exists to exploit.

### The feedback layer, framed honestly

This is **not** reinforcement learning, and calling it that would be the kind of
overclaim the rest of the project is built to avoid. There is no environment the
system acts upon, no state transition it causes, and no multi-step credit
assignment. What it is: a **contextual multi-armed bandit with delayed reward.**

```
arms     lightgbm · lightgbm_cal · logistic_cal · reversal rule
context  volatility bucket x trend state  (9 regimes)
reward   conviction-weighted hit/miss, computed at T+1
method   Thompson sampling on a Beta-Bernoulli posterior per (arm, regime)
```

The two-parameter reversal rule is deliberately included as a standing honesty
check: M1 found it matched the GBM's accuracy, so if the bandit ever loads onto
it, the machine learning is not earning its place.

Three guardrails, each of which caps how much the bandit can help — which is
the correct trade, because what they prevent is worse than what they cost:

- **Minimum sample gate.** Weights stay uniform in a regime below 100 resolved
  forecasts. With a true edge near one percentage point, a handful of outcomes
  is indistinguishable from noise.
- **Floor and ceiling.** No arm below 5% or above 60%, enforced by projecting
  onto the constraint set. Clipping and rescaling — the obvious implementation —
  is wrong: rescaling to restore the sum pushes a just-capped arm back over its
  ceiling. A test asserts the bound holds for a 0.97 runaway input.
- **Drift detection.** If rolling accuracy in a regime falls below 48% over 60
  resolved forecasts, that regime reverts to uniform weights and the event is
  logged. This fired during the replay, as it should have.

### Interface

Typeform-style single-question flow, served by FastAPI as two static files. No
Node, no build step, no framework — the interaction is a linear sequence of
steps, which is not what a framework earns its weight on.

M1 and M2 dictated the design more than taste did:

- **Model confidence never exceeds about 55%,** so there is no state in which
  the interface shows a bold green arrow. Confidence renders amber by default
  and only gains saturation above 60%, which is rare.
- **"No useful opinion" is a first-class screen,** not an empty state. It is the
  *common* case: on the latest session the system declines to call 39 of 52
  symbols, and the copy says that is normal rather than apologising for it.
- **"We don't know why this moved" is a designed state too,** because 62% of
  significant moves have no identifiable cause.
- **No number appears without its comparator.** Every accuracy figure renders
  beside the coin flip; the live accuracy renders beside the majority baseline
  on the same rows.

Verified in-browser end to end: the abstain path, the directional path
(BHARTIARTL at 55.3%, rendered amber), aspect attribution, evidence with source
links, and the live reliability curve. Three defects were found and fixed by
running it — a conviction reported in "basis points" that was actually
percentage points, a live accuracy shown without its baseline, and an internal
field name (`kind: "recent"`) leaking into the narrator's prose as "the
attribution is recent".

```bash
.venv/bin/uvicorn cef.api.main:app --port 8123     # then open localhost:8123
```

---

## Why the numbers are believable

Every result above rests on the validation design, so that is where the effort
went.

**Availability lags, per series.** A forecast for session t+1 is generated at
the Indian close on date t. A US index closes at 01:30 IST on t+1, so its
date-t value is *not* knowable at generation time and is lagged one session.
Asian sessions close before the Indian close and are not. Each series in
`cef/universe.py` declares its own lag, applied in exactly one place. Using an
unlagged US close would be look-ahead bias that flatters every downstream
number.

**Horizon-aware embargo.** A label spanning h forward sessions at date t
reaches to t+h, so a training row within h sessions of the test start carries
an outcome from inside the test window. The embargo is
`max(2, ceil((h+1) × 1.45))` calendar days — 4 days at h=1, 32 at h=20. With a
fixed 2-day embargo the 20-session results would have been fabricated out of
four weeks of overlap.

**Tuning and reporting folds never overlap.** Folds 0–7 are the development
set and were read freely. Folds 8–12 were read once, by `scripts/report.py`,
after the configuration was frozen. The held-out numbers came out *better*
than the tuning numbers, which is the outcome that suggests the tuning did not
overfit.

**One forward shift, in one place.** Every feature is a trailing `rolling` or
`ewm` with no centring. The only `shift(-n)` in `cef/` is in the target
functions, and a test scans the source tree to keep it that way.

**Transforms fitted inside the fold.** Imputer, variance filter, scaler, tree
ensemble and isotonic calibrator are each fitted on their own chronological
slice of a single training window. A test asserts that scaler statistics
differ across folds, which they cannot if anything was fitted on pooled data.

**The shuffled-label test.** Permute the label, keep everything else, and
out-of-sample AUC must collapse to chance. Across five seeds: 0.4887, 0.4975,
0.4986, 0.5015, 0.5037. Nothing reaches the model except through the features.

```bash
python -m pytest tests/ -v
```

15 tests, all passing.

---

## Known limitations

- **Survivorship bias.** The universe is current Nifty 50 membership, so names
  dropped over the sample are absent. `TATAMOTORS` is excluded outright (2025
  demerger left no continuous series); `JIOFIN` and `ETERNAL` have genuinely
  short histories. All three are recorded in `cef/universe.py` rather than
  dropped silently.
- **EOD only.** Daily bars from Yahoo Finance. No intraday, no tick data.
- **No transaction costs.** A 51% hit rate is very unlikely to be profitable
  after costs, and profitability is deliberately not a success criterion.
- **The edge is weak and regime-dependent.** In elevated-volatility regimes the
  held-out edge is *negative* (−0.95 pp). It is strongest in trending-up
  regimes (+1.75 pp).
- **13 folds is a small sample** for fold-level t-statistics. They test whether
  an edge recurs across regimes, which is the right question, but they are a
  blunt instrument.
- **News has no history.** Attribution over 2018–2026 uses corporate filings
  and driver moves. Headline sentiment only accumulates from the day ingest
  first runs.
- **Attribution is correlational.** A high confidence means a material,
  correctly-timed, directionally-consistent filing was found — not that it
  caused the move. External events leave no filing at all: the Adani selloff of
  February 2023 was a short-seller report, so the system correctly reports it
  as unexplained.
- **Not yet built:** the flow (FII/DII) and fundamentals analyzers, the
  delayed-reward bandit, and the interface.

---

## Architecture

```
scripts/ingest.py           yfinance -> SQLite   (109k OHLCV rows, 19 macro series)
scripts/ingest_evidence.py  NSE filings + news   (69,747 filings, 2018-2026)
scripts/build_features.py   -> panel.parquet     (98,651 rows x 200 features)
scripts/tune.py             search, tune folds only
scripts/report.py           held-out assessment + reliability diagram
scripts/attribute.py        significant moves -> evidence
scripts/forecast.py         one symbol: call, aspects, evidence, narration
scripts/replay.py           replay the live loop over history

cef/
  universe.py              52 symbols, sector map, macro series + availability lags
  db.py                    SQLite schema and I/O
  ingest/                  equities, macro, announcements, corporate_actions, news
  features/
    technical.py           81 features, hand-written, trailing windows only
    macro.py               92 features, lag enforcement lives here
    cross_sectional.py     27 features, relative strength / beta / driver linkage
    breaks.py              capital-structure break detection, exchange-anchored
    build.py               panel assembly, rebasing, every target definition
  feedback/
    ensemble.py            the four arms the bandit weights over
    bandit.py              contextual Thompson sampling + guardrails
    reward.py              conviction-weighted, asymmetric reward
    predict.py             writes forecasts with full feature snapshots
    resolver.py            nightly resolution, per-arm scoring
  api/
    main.py                FastAPI surface
    service.py             everything the handlers need, no HTTP in it
    static/                Typeform-style flow: index.html, app.css, app.js
  evidence/
    event_map.py           filing materiality + direction priors
    moves.py               significant-move detection
    attribution.py         move -> filing or driver, with sources
    aspects.py             SHAP -> seven reader-facing buckets
    sentiment.py           FinBERT headline scoring
    narrate.py             OpenAI rewriting, with a hallucination verifier
  models/
    baselines.py           always-up, random, persistence, mean-reversion
    direction.py           LightGBM and logistic heads
    calibration.py         isotonic + reliability + ECE
    selection.py           SHAP selection, refitted inside each fold
  validation/
    walk_forward.py        fold generation, embargo, three-way split
    selective.py           accuracy-vs-coverage with two comparators
    runner.py              the experiment loop; enforces tune/holdout separation
tests/test_no_leakage.py   15 leakage tests
tests/test_evidence.py     30 evidence-layer tests
tests/test_feedback.py     16 bandit and reward tests
```

Technical indicators are hand-written rather than taken from TA-Lib or
pandas-ta. The point of this project is provable causality, and a trailing
window we control is a property that can be asserted in a test rather than
trusted in a dependency.

---

## Setup

No Docker, no Node, no Homebrew required.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -e .
.venv/bin/python scripts/postinstall_macos.py   # macOS arm64 only, see below
cp .env.example .env                            # add FRED_API_KEY / OPENAI_API_KEY
```

On Apple Silicon, LightGBM's wheel links `@rpath/libomp.dylib` but resolves it
only under Homebrew paths. `scripts/postinstall_macos.py` copies the `libomp`
that scikit-learn already vendors into LightGBM's lib directory, adds an
`@loader_path` rpath and re-signs the dylib — so LightGBM works without
installing Homebrew.

```bash
.venv/bin/python scripts/ingest.py                        # ~15s
.venv/bin/python scripts/ingest_evidence.py               # ~25 min (NSE throttles)
.venv/bin/python scripts/build_features.py                # ~5s
.venv/bin/python -m pytest tests/ -v                      # ~10s, 45 tests
.venv/bin/python scripts/report.py --split holdout        # the held-out number
.venv/bin/python scripts/attribute.py --sigma 2.0         # move attribution
.venv/bin/python scripts/forecast.py HINDZINC             # one full forecast
.venv/bin/python scripts/replay.py --n-symbols 30         # exercise the loop
.venv/bin/uvicorn cef.api.main:app --port 8123            # the interface
```

### Data sources

| layer | source | key |
|---|---|---|
| OHLCV, 52 NSE symbols | yfinance (`.NS`) | none |
| Global indices, FX, commodities, rates | yfinance | none |
| Corporate filings, corporate actions | NSE APIs | none (browser UA + cookie handshake) |
| Headlines | 6 RSS feeds + per-ticker Yahoo | none |
| Headline sentiment | FinBERT, local | none |
| US macro releases | FRED | free, `.env` |
| Narration | OpenAI (`gpt-5-mini`) | `.env` |

Two source notes worth recording. The NSE endpoints require a *browser*
User-Agent and a cookie handshake, and return 403 without them — which is the
opposite of several other financial APIs that reject browser agents, so the
headers are a deliberate per-source choice rather than copied boilerplate. And
news is **forward-only**: Indian financial RSS exposes one to three days of
history, and GDELT — the obvious free archive — rate-limits to roughly one
request every five seconds with loose entity matching, so a multi-year backfill
across 52 symbols is neither fast nor accurate. Historical attribution
therefore rests on corporate filings and driver moves, which do have full
history.

---

## Roadmap

- **M1 — the honest number.** ✅ Ingest, point-in-time features, baselines,
  walk-forward with embargo, calibration, leak suite, held-out report.
- **M2 — evidence and attribution.** ✅ NSE filings corpus, causal filing
  timestamps, materiality gating, move attribution with sources, driver
  linkage, SHAP-to-aspect bucketing, verified narration.
- **M3 — interface and feedback loop.** ✅ Prediction persistence with feature
  snapshots, nightly resolver, contextual Thompson-sampling bandit with
  guardrails, FastAPI surface, Typeform-style interface.

Still open: the flow (FII/DII) and fundamentals analyzers, a range/volatility
head, multi-symbol comparison in the interface, and scheduling the resolver as
a real nightly job rather than a script.
