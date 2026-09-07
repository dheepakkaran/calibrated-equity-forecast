# Calibrated Directional Forecasting with Delayed-Reward Ensemble Reweighting

### An Evidence-Grounded System for NSE Equities

Next-session directional forecasting for Indian equities, built so that the
reported accuracy can be trusted. The engineering claim is point-in-time
correctness, honest validation and calibrated confidence — not alpha.

> Educational and research use only. Not investment advice. Directional
> forecasts for equities are inherently low-confidence, and this system
> reports its own accuracy openly, including the parts that do not work.

---

## Headline result — M1

Held-out folds (2024-03 → 2026-08), never read during development:

| | accuracy | edge vs baseline | fold t-stat | folds won | AUC |
|---|---|---|---|---|---|
| LightGBM + isotonic | **50.91%** | **+0.90 pp** | 2.41 | 5/5 | 0.512 |
| Logistic + isotonic | 51.02% | +1.02 pp | 2.35 | 4/5 | 0.513 |
| 1-day reversal rule | 50.92% | +0.91 pp | 2.81 | 5/5 | 0.509 |
| Coin flip / always-up | 50.01% | — | — | — | — |

Under selective prediction — forecasting only when conviction is high:

| coverage | accuracy | edge vs baseline on the same rows | t-stat | folds won |
|---|---|---|---|---|
| 10% | **53.48%** | +2.24 pp | 1.37 | 4/5 |
| 30% | 51.74% | +1.70 pp | 2.24 | 4/5 |
| 50% | 51.40% | +1.43 pp | 2.02 | 5/5 |
| 100% | 50.93% | +0.91 pp | 2.26 | 5/5 |

Calibration, held-out: **ECE 0.69 pp pooled**, 1.78 pp as a per-fold mean.
The reliability curve sits within about one percentage point of the diagonal
across the range the model actually uses.

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
- **Not yet built:** the flow (FII/DII), sentiment and fundamentals analyzers,
  the evidence/attribution layer, the delayed-reward bandit, and the interface.

---

## Architecture

```
scripts/ingest.py          yfinance -> SQLite    (109k OHLCV rows, 19 macro series)
scripts/build_features.py  -> panel.parquet      (98,653 rows x 200 features)
scripts/tune.py            search, tune folds only
scripts/report.py          held-out assessment + reliability diagram

cef/
  universe.py              52 symbols, sector map, macro series + availability lags
  db.py                    SQLite schema and I/O
  ingest/                  equities, macro
  features/
    technical.py           81 features, hand-written, trailing windows only
    macro.py               92 features, lag enforcement lives here
    cross_sectional.py     27 features, relative strength / beta / driver linkage
    build.py               panel assembly and every target definition
  models/
    baselines.py           always-up, random, persistence, mean-reversion
    direction.py           LightGBM and logistic heads
    calibration.py         isotonic + reliability + ECE
    selection.py           SHAP selection, refitted inside each fold
  validation/
    walk_forward.py        fold generation, embargo, three-way split
    selective.py           accuracy-vs-coverage with two comparators
    runner.py              the experiment loop; enforces tune/holdout separation
tests/test_no_leakage.py   15 tests
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
cp .env.example .env                            # add FRED_API_KEY when needed
```

On Apple Silicon, LightGBM's wheel links `@rpath/libomp.dylib` but resolves it
only under Homebrew paths. `scripts/postinstall_macos.py` copies the `libomp`
that scikit-learn already vendors into LightGBM's lib directory, adds an
`@loader_path` rpath and re-signs the dylib — so LightGBM works without
installing Homebrew.

```bash
.venv/bin/python scripts/ingest.py           # ~15s
.venv/bin/python scripts/build_features.py   # ~5s
.venv/bin/python -m pytest tests/ -v         # ~3s
.venv/bin/python scripts/report.py --split holdout
```

### Data sources

| layer | source | key |
|---|---|---|
| OHLCV, 52 NSE symbols | yfinance (`.NS`) | none |
| Global indices, FX, commodities, rates | yfinance | none |
| US macro releases | FRED | free, `.env` |
| Narration (M2) | Anthropic API | `.env` |

---

## Roadmap

- **M1 — the honest number.** ✅ Ingest, point-in-time features, baselines,
  walk-forward with embargo, calibration, leak suite, held-out report.
- **M2 — evidence and attribution.** Move attribution against news, driver
  linkage, source-traceable evidence, plain-language narration.
- **M3 — interface and feedback loop.** Typeform-style UI, prediction
  persistence, nightly resolver, delayed-reward Thompson-sampling bandit over
  ensemble members.

M1's result shapes M2 and M3 directly: with a maximum model confidence of
about 53%, the interface can never show a bold directional arrow, and
"no useful opinion today" has to be a first-class state rather than an edge
case.
