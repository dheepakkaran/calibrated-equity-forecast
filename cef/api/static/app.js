/* ────────────────────────────────────────────────────────────────────
   Flow controller.

   Deliberately dependency-free. There is no build step, no bundler and no
   framework, which keeps the whole interface two static files the API serves
   directly - and the interaction here is a linear sequence of steps, which
   is not what a framework earns its weight on.

   The one rule the rendering enforces everywhere: a number is never shown
   without the thing it should be compared against. Accuracy always carries
   the coin flip beside it, and a verdict always carries its confidence.
   ──────────────────────────────────────────────────────────────────── */
'use strict';

const STEPS = ['intro', 'symbol', 'working', 'verdict', 'why', 'evidence', 'record'];
const $ = (s) => document.querySelector(s);
const $$ = (s) => Array.from(document.querySelectorAll(s));

const state = { step: 0, symbol: null, forecast: null, moves: [], perf: null, bandit: null };

/* ── navigation ─────────────────────────────────────────────────── */
function show(name) {
  const target = $(`.step[data-step="${name}"]`);
  if (!target) return;
  $$('.step').forEach((el) => {
    const active = el === target;
    el.classList.toggle('is-active', active);
    el.classList.toggle('is-past', !active && STEPS.indexOf(el.dataset.step) < STEPS.indexOf(name));
  });
  const idx = STEPS.indexOf(name);
  if (idx >= 0) {
    state.step = idx;
    $('#progress').style.width = `${(idx / (STEPS.length - 1)) * 100}%`;
  }
  $('#restart').hidden = name === 'intro';
  if (name === 'symbol') setTimeout(() => $('#symbol-input').focus(), 380);
  if (name === 'why') loadNarration();
}

function next() {
  const cur = STEPS[state.step];
  if (cur === 'intro') return show('symbol');
  if (cur === 'symbol') return runForecast();
  if (cur === 'verdict') return show('why');
  if (cur === 'why') return show('evidence');
  if (cur === 'evidence') return show('record');
}

/* ── typeahead ──────────────────────────────────────────────────── */
let taItems = [], taSel = -1, taTimer = null;

async function refreshTypeahead() {
  const q = $('#symbol-input').value.trim();
  const box = $('#typeahead');
  try {
    const r = await fetch(`/api/symbols/search?q=${encodeURIComponent(q)}&limit=7`);
    taItems = await r.json();
  } catch { taItems = []; }
  if (!taItems.length) { box.classList.remove('open'); return; }
  taSel = 0;
  box.innerHTML = taItems.map((it, i) => `
    <div class="ta-item ${i === 0 ? 'sel' : ''}" data-sym="${it.symbol}" role="option">
      <span class="ta-sym">${it.symbol}</span>
      <span class="ta-sec">${it.sector}</span>
    </div>`).join('');
  box.classList.add('open');
  $$('#typeahead .ta-item').forEach((el) => {
    el.addEventListener('click', () => { pick(el.dataset.sym); });
  });
}

function moveSel(delta) {
  if (!taItems.length) return;
  taSel = (taSel + delta + taItems.length) % taItems.length;
  $$('#typeahead .ta-item').forEach((el, i) => el.classList.toggle('sel', i === taSel));
}

function pick(sym) {
  $('#symbol-input').value = sym;
  $('#typeahead').classList.remove('open');
  runForecast();
}

/* ── the forecast ───────────────────────────────────────────────── */
const WORKING_BEATS = [
  ['Fitting the ensemble…', 'Four arms, trained on everything up to the last close. Nothing after it.'],
  ['Selecting features…', 'Top 70 of 200 by mean |SHAP|, chosen inside the training window only.'],
  ['Calibrating…', 'Isotonic regression, so a stated 52% means 52%.'],
  ['Weighting arms…', 'Thompson sampling over the current market regime.'],
];

async function runForecast() {
  const sym = ($('#symbol-input').value || '').trim().toUpperCase();
  if (!sym) { $('#symbol-input').focus(); return; }
  state.symbol = sym;
  show('working');

  let beat = 0;
  const tick = setInterval(() => {
    beat = (beat + 1) % WORKING_BEATS.length;
    $('#working-label').textContent = WORKING_BEATS[beat][0];
    $('#working-sub').textContent = WORKING_BEATS[beat][1];
  }, 1500);

  try {
    const r = await fetch(`/api/forecast/${encodeURIComponent(sym)}`);
    if (!r.ok) throw new Error((await r.json()).detail || 'not found');
    state.forecast = await r.json();
    const [moves, perf, band] = await Promise.all([
      fetch(`/api/attribution/${encodeURIComponent(sym)}?limit=6`).then((x) => x.json()),
      fetch('/api/performance').then((x) => x.json()),
      fetch('/api/bandit').then((x) => x.json()),
    ]);
    state.moves = moves; state.perf = perf; state.bandit = band;
    clearInterval(tick);
    renderVerdict(); renderAspects(); renderMoves(); renderRecord(); renderTechnical();
    show('verdict');
  } catch (err) {
    clearInterval(tick);
    $('#working-label').textContent = `Could not forecast ${sym}`;
    $('#working-sub').textContent = `${err.message}. Press Escape to pick another symbol.`;
  }
}

/* ── renderers ──────────────────────────────────────────────────── */
function renderVerdict() {
  const f = state.forecast;
  const abstain = !f.acted;
  const up = f.proba_outperform > 0.5;

  $('#verdict-eyebrow').textContent =
    `${f.symbol} · ${f.sector} · for the session after ${f.as_of}`;

  // The wording carries the uncertainty, not just the number beside it.
  let line;
  if (abstain) {
    line = 'The model has <em class="none">no useful opinion</em> on this share today.';
  } else if (up) {
    line = `Slightly more likely to <em class="up">out-perform</em> the median large-cap than to lag it.`;
  } else {
    line = `Slightly more likely to <em class="down">lag</em> the median large-cap than to beat it.`;
  }
  $('#verdict-line').innerHTML = line;

  const pct = (f.confidence * 100).toFixed(1);
  const num = $('#conf-num');
  num.textContent = abstain ? '—' : `${pct}%`;
  num.className = 'conf-num' + (abstain ? ' none' : f.confidence >= 0.6 ? ' firm' : '');

  $('#conf-note').textContent = abstain
    // Percentage points, not basis points: a conviction of 0.0059 is 0.59 pp.
    // Getting that label wrong by a factor of 100 in a financial tool is worse
    // than saying nothing.
    ? `Conviction is ${(f.conviction * 100).toFixed(2)} percentage points from a coin toss, `
      + `below the ${(f.abstain_threshold * 100).toFixed(1)} pp this system needs before it will `
      + `commit. That is the common case, not a malfunction.`
    : `Out of every 100 forecasts at this confidence, about ${Math.round(f.confidence * 100)} `
      + `come out right and ${100 - Math.round(f.confidence * 100)} come out wrong. `
      + `This is weak evidence, not a signal.`;

  // Needle: probability mapped across the track, clamped to stay visible.
  const posPct = Math.min(96, Math.max(4, 50 + (f.proba_outperform - 0.5) * 900));
  const needle = $('#gauge-needle');
  needle.style.left = `${posPct}%`;
  needle.style.background = abstain ? 'var(--warn)' : up ? 'var(--bull)' : 'var(--bear)';

  const t = f.track_record;
  $('#verdict-facts').innerHTML = [
    ['Last close', `₹${f.last_close.toLocaleString('en-IN')}`],
    ['Typical daily range', `₹${f.typical_daily_range.toLocaleString('en-IN')}`],
    ['Market regime', `${f.regime.volatility} <small>vol</small> · ${f.regime.trend.replace('_', ' ')}`],
    ['Held-out accuracy', `${(t.accuracy * 100).toFixed(1)}% <small>vs ${(t.coin_flip_baseline * 100).toFixed(1)}% coin flip</small>`],
  ].map(([k, v]) => `<div class="fact"><div class="fact-k">${k}</div><div class="fact-v">${v}</div></div>`).join('');
}

function renderAspects() {
  const aspects = (state.forecast.aspects || []).filter((a) => a.points !== 0);
  if (!aspects.length) {
    $('#aspects').innerHTML = `<div class="aspect"><div class="aspect-name">No aspect
      contributed measurably.</div></div>`;
    return;
  }
  const max = Math.max(...aspects.map((a) => Math.abs(a.points))) || 1;
  $('#aspects').innerHTML = aspects.map((a) => {
    const pos = a.points > 0;
    const w = (Math.abs(a.points) / max) * 46;
    const ev = (a.evidence || []).map((e) => `${e.feature}${e.value !== undefined ? `=${e.value}` : ''}`).join('  ·  ');
    return `<div class="aspect">
      <div>
        <div class="aspect-name">${a.aspect}</div>
        ${ev ? `<div class="aspect-ev">${ev}</div>` : ''}
      </div>
      <div class="aspect-bar">
        <i style="background:var(--${pos ? 'bull' : 'bear'});${pos ? 'left' : 'right'}:50%;width:${w}%"></i>
        <div class="mid"></div>
      </div>
      <div class="aspect-pts ${pos ? 'p' : 'n'}">${pos ? '+' : ''}${a.points}</div>
    </div>`;
  }).join('');
}

async function loadNarration() {
  if (!state.symbol || $('#narration').dataset.loaded === state.symbol) return;
  try {
    const r = await fetch(`/api/narration/${encodeURIComponent(state.symbol)}`);
    if (!r.ok) return;
    const n = await r.json();
    $('#narration-text').textContent = n.text;
    const bad = (n.unverified_numbers || []).length;
    $('#narration-foot').textContent = bad
      ? `⚠ ${bad} number(s) in this text do not appear in the evidence: ${n.unverified_numbers.join(', ')}`
      : `Written by ${n.model} from the evidence above. Every number in it was checked against that evidence.`;
    $('#narration').hidden = false;
    $('#narration').dataset.loaded = state.symbol;
  } catch { /* narration is optional; the rest of the flow does not depend on it */ }
}

function renderMoves() {
  const moves = state.moves || [];
  if (!moves.length) {
    $('#moves').innerHTML = `<div class="move unexplained">
      <div class="move-none">No move larger than two standard deviations in the
      recorded window. Quiet is the normal state.</div></div>`;
    return;
  }
  // The unexplained rationale is stated in full once and abbreviated after
  // that. Five identical paragraphs read as a template, which undercuts the
  // point they are making.
  let saidUnexplained = false;
  $('#moves').innerHTML = moves.map((m) => {
    const cls = m.kind === 'announcement' ? 'filing' : m.kind === 'driver' ? 'driver' : 'unexplained';
    const pos = m.ret > 0;
    let body;
    if (m.kind === 'unexplained') {
      body = saidUnexplained
        ? `<div class="move-none">No qualifying evidence for this one either.</div>`
        : `<div class="move-none">No filing above the materiality threshold, and no
           mapped driver moved consistently. This one is unexplained — which is the
           honest answer for about 62% of large moves.</div>`;
      saidUnexplained = true;
    } else {
      body = `<div class="move-body">${(m.headline || '').slice(0, 260)}</div>`;
    }
    const src = m.source_url
      ? `<a class="move-src" href="${m.source_url}" target="_blank" rel="noopener">Open the filing</a>`
      : (m.kind === 'driver' ? `<div class="move-src">${m.rationale || ''}</div>` : '');
    return `<div class="move ${cls}">
      <div class="move-top">
        <span class="move-date">${m.date}</span>
        <span class="move-ret ${pos ? 'p' : 'n'}">${pos ? '+' : ''}${(m.ret * 100).toFixed(2)}%</span>
        <span class="move-sigma">${m.sigma}σ${m.market_wide ? ' · market-wide' : ''}</span>
        <span class="pill ${m.band}">${m.kind === 'unexplained' ? 'unexplained' : `${m.kind} · ${m.band}`}</span>
      </div>
      ${body}${src}</div>`;
  }).join('');
}

function renderRecord() {
  const p = state.perf || {}, t = (p.held_out || state.forecast.track_record);
  const cards = [
    [`${(t.accuracy * 100).toFixed(1)}%`, 'Held-out accuracy<br>2024-03 → 2026-08', ''],
    [`${(t.coin_flip_baseline * 100).toFixed(1)}%`, 'Coin-flip baseline<br>what beating nothing looks like', 'dim'],
    [`+${t.edge_pp} pp`, `Edge, t = ${t.fold_t_stat}<br>${t.folds_won} folds won`, ''],
    [`${t.ece_pooled_pp} pp`, 'Calibration error<br>a stated 52% means 52%', 'warn'],
  ];
  if (p.n_acted) {
    cards.push([`${(p.accuracy * 100).toFixed(1)}%`,
      `Live accuracy<br>${p.n_acted} resolved calls`, '']);
    // The same rule as everywhere else: never a number without its
    // comparator. This is the majority-class rate on the rows the system
    // actually acted on, which is the strict baseline - not 50%.
    if (p.majority_baseline_on_acted != null) {
      cards.push([`${(p.majority_baseline_on_acted * 100).toFixed(1)}%`,
        `Baseline on those same rows<br>edge is +${p.edge_vs_majority_pp} pp`, 'dim']);
    }
    cards.push([`${(p.abstention_rate * 100).toFixed(0)}%`,
      'Sessions with no opinion<br>silence is the default', 'dim']);
  }
  $('#scorecards').innerHTML = cards.map(([n, l, c]) =>
    `<div class="card"><div class="card-n ${c}">${n}</div><div class="card-l">${l}</div></div>`).join('');

  const rel = p.reliability || [];
  if (rel.length >= 4) {
    const W = 420, H = 170, pad = 34;
    const xs = rel.map((r) => r.pred_mean), ys = rel.map((r) => r.actual_rate);
    const lo = Math.min(...xs, ...ys) - 0.02, hi = Math.max(...xs, ...ys) + 0.02;
    const sx = (v) => pad + ((v - lo) / (hi - lo)) * (W - pad - 12);
    const sy = (v) => H - pad - ((v - lo) / (hi - lo)) * (H - pad - 14);
    const path = rel.map((r, i) => `${i ? 'L' : 'M'}${sx(r.pred_mean).toFixed(1)},${sy(r.actual_rate).toFixed(1)}`).join(' ');
    const dots = rel.map((r) => `<circle cx="${sx(r.pred_mean).toFixed(1)}" cy="${sy(r.actual_rate).toFixed(1)}" r="3" fill="#5fa8a0"/>`).join('');
    $('#reliability').innerHTML = `
      <div class="rel-head">Reliability — live resolved forecasts</div>
      <svg viewBox="0 0 ${W} ${H}" style="width:100%;height:auto">
        <line x1="${sx(lo)}" y1="${sy(lo)}" x2="${sx(hi)}" y2="${sy(hi)}"
              stroke="#647889" stroke-width="1" stroke-dasharray="4 4"/>
        <path d="${path}" fill="none" stroke="#5fa8a0" stroke-width="2"/>
        ${dots}
        <text x="${pad}" y="${H - 8}" fill="#647889" font-size="10">predicted →</text>
        <text x="${W - 150}" y="${sy(hi) + 16}" fill="#647889" font-size="10">perfect calibration</text>
      </svg>
      <div class="rel-note">If the teal line tracks the dashed diagonal, a stated
        52% is right about 52% of the time. This is the screen that decides
        whether the confidence numbers elsewhere mean anything.</div>`;
  } else {
    $('#reliability').innerHTML = `<div class="rel-head">Reliability — live forecasts</div>
      <div class="rel-note">${p.reliability_note || 'Not enough resolved forecasts yet.'}
      The held-out reliability, measured on 31,905 out-of-sample rows, is
      ${t.ece_pooled_pp} pp from the diagonal.</div>`;
  }

  $('#record-note').textContent = p.n
    ? `${p.n} forecasts written and resolved so far, of which ${p.n_acted} carried a call. `
      + `Nothing is deleted, including the wrong ones.`
    : 'No live forecasts resolved yet — the numbers above are the held-out assessment.';
}

function renderTechnical() {
  const f = state.forecast, b = state.bandit || {};
  const arms = f.arms || {};
  const armRows = Object.keys(arms.probas || {}).map((a) =>
    `<tr><td>${a}</td><td>${(arms.probas[a]).toFixed(4)} · w ${(arms.weights[a] ?? 0).toFixed(3)}</td></tr>`).join('');
  const g = b.guardrails || {};
  const regimeRows = (b.state || []).filter((r) => r.regime === f.regime.label)
    .map((r) => `<tr><td>${r.arm}</td><td>${r.posterior_mean} <small>(n=${r.n_pulls})</small></td></tr>`).join('')
    || '<tr><td colspan="2">no resolved forecasts in this regime yet</td></tr>';

  $('#tech-grid').innerHTML = `
    <div class="tech-card"><h3>Ensemble arms</h3>
      <table class="kv">${armRows}</table>
      <div class="rel-note">Blend mode: ${arms.mode}. ${arms.n_pulls} resolved
        forecasts in this regime.</div></div>
    <div class="tech-card"><h3>Bandit posterior · ${f.regime.label}</h3>
      <table class="kv">${regimeRows}</table>
      <div class="rel-note">Beta-Bernoulli mean per arm. Weights stay uniform
        below ${g.min_pulls_per_regime} pulls, and no arm may exceed
        ${(g.weight_ceiling * 100).toFixed(0)}% or fall below
        ${(g.weight_floor * 100).toFixed(0)}%.</div></div>
    <div class="tech-card"><h3>Forecast metadata</h3>
      <table class="kv">
        <tr><td>target</td><td>market-relative direction</td></tr>
        <tr><td>as-of close</td><td>${f.as_of}</td></tr>
        <tr><td>target session</td><td>${f.target_date}</td></tr>
        <tr><td>P(out-perform)</td><td>${f.proba_outperform.toFixed(4)}</td></tr>
        <tr><td>conviction</td><td>${f.conviction.toFixed(5)}</td></tr>
        <tr><td>model version</td><td>${f.model_version}</td></tr>
      </table>
      <div class="rel-note">Mapped drivers: ${(f.drivers || []).join(', ')}</div></div>
    <div class="tech-card"><h3>Prediction id</h3>
      <div class="mono">${f.prediction_id}</div>
      <div class="rel-note">Stored with the full feature snapshot, so this
        forecast can be replayed and re-explained later without recomputing
        anything from today's data.</div></div>`;
}

/* ── events ─────────────────────────────────────────────────────── */
$$('[data-next]').forEach((b) => b.addEventListener('click', next));
$('#restart').addEventListener('click', () => { $('#symbol-input').value = ''; show('intro'); });
$('#again').addEventListener('click', () => { $('#symbol-input').value = ''; show('symbol'); });
$('#technical').addEventListener('click', () => show('technical'));
$('#back-to-record').addEventListener('click', () => show('record'));

$('#symbol-input').addEventListener('input', () => {
  clearTimeout(taTimer);
  taTimer = setTimeout(refreshTypeahead, 110);
});
$('#symbol-input').addEventListener('focus', refreshTypeahead);

document.addEventListener('keydown', (e) => {
  const onSymbol = STEPS[state.step] === 'symbol';
  if (e.key === 'Enter') {
    if (onSymbol && taItems.length && taSel >= 0 && document.activeElement === $('#symbol-input')
        && $('#typeahead').classList.contains('open')) {
      e.preventDefault(); pick(taItems[taSel].symbol); return;
    }
    const btn = $('.step.is-active [data-next]') || $('.step.is-active .cta');
    if (btn) { e.preventDefault(); btn.click(); }
    return;
  }
  if (onSymbol && (e.key === 'ArrowDown' || e.key === 'ArrowUp')) {
    e.preventDefault(); moveSel(e.key === 'ArrowDown' ? 1 : -1); return;
  }
  if (e.key === 'Escape') { $('#typeahead').classList.remove('open'); show('symbol'); }
});

show('intro');
