/* ────────────────────────────────────────────────────────────────────
   Dashboard controller. Dependency-free, no build step.

   Charts are hand-drawn SVG rather than a charting library. Four small
   plots do not justify 200kB of Recharts, and drawing them here means
   the axis ranges are chosen from the actual data rather than by a
   library guessing at nice round numbers.

   The invariant enforced throughout: every rendered number carries its
   comparator. Accuracy renders beside the coin flip, confidence beside
   the coin-flip band, driver correlations beside their lag.
   ──────────────────────────────────────────────────────────────────── */
'use strict';

const $ = (s) => document.querySelector(s);
const $$ = (s) => Array.from(document.querySelectorAll(s));
const fmt = (n, d = 2) => (n === null || n === undefined || Number.isNaN(n))
  ? '—' : Number(n).toLocaleString('en-IN', { minimumFractionDigits: d, maximumFractionDigits: d });
const pct = (n, d = 2) => (n === null || n === undefined) ? '—' : `${n > 0 ? '+' : ''}${fmt(n, d)}%`;
const esc = (s) => String(s ?? '').replace(/[&<>"]/g, (c) =>
  ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

const S = { symbol: 'HINDZINC', data: null, perf: null, bandit: null,
            simple: null, tab: 'simple' };

/* ── SVG helpers ─────────────────────────────────────────────────── */
function lineChart({ points, width = 700, height = 210, pad = 34, levels = [], markers = [] }) {
  if (!points.length) return '<div class="micro">No price history.</div>';
  const ys = points.map((p) => p.close);
  const extra = levels.map((l) => l.y).filter((v) => Number.isFinite(v));
  let lo = Math.min(...ys, ...extra), hi = Math.max(...ys, ...extra);
  const padY = (hi - lo) * 0.08 || 1;
  lo -= padY; hi += padY;
  const sx = (i) => pad + (i / Math.max(1, points.length - 1)) * (width - pad - 46);
  const sy = (v) => height - 26 - ((v - lo) / (hi - lo)) * (height - 40);

  const path = points.map((p, i) => `${i ? 'L' : 'M'}${sx(i).toFixed(1)},${sy(p.close).toFixed(1)}`).join(' ');

  const grid = [0, 0.25, 0.5, 0.75, 1].map((f) => {
    const v = lo + f * (hi - lo);
    return `<line x1="${pad}" y1="${sy(v).toFixed(1)}" x2="${width - 46}" y2="${sy(v).toFixed(1)}"
              stroke="#26343F" stroke-width="1"/>
            <text x="${width - 42}" y="${(sy(v) + 3.5).toFixed(1)}" fill="#647889"
              font-size="9.5">${fmt(v, 0)}</text>`;
  }).join('');

  // Reference levels are pivots from a single session, so they sit within a
  // rupee or two of each other while the y-axis spans a year. Stacked at the
  // same x they render as one illegible smudge, so labels are laid out with a
  // simple collision pass: anything within 11px of an already-placed label
  // steps right by one slot.
  const placed = [];
  const lvl = levels.filter((l) => Number.isFinite(l.y))
    .sort((a, b) => a.y - b.y)
    .map((l) => {
      const y = sy(l.y);
      let slot = 0;
      while (placed.some((q) => q.slot === slot && Math.abs(q.y - y) < 11)) slot += 1;
      placed.push({ slot, y });
      const x = pad + 4 + slot * 62;
      return `<line x1="${pad}" y1="${y.toFixed(1)}" x2="${width - 46}" y2="${y.toFixed(1)}"
         stroke="${l.color}" stroke-width="1" stroke-dasharray="4 4" opacity=".75"/>
       <text x="${x}" y="${(y - 4).toFixed(1)}" fill="${l.color}"
         font-size="9.5">${esc(l.label)}</text>`;
    }).join('');

  const idxByDate = new Map(points.map((p, i) => [p.date, i]));
  const mk = markers.map((m, k) => {
    const i = idxByDate.get(m.date);
    if (i === undefined) return '';
    const col = m.kind === 'announcement' ? '#7B95C4'
      : m.kind === 'driver' ? '#5FA8A0' : '#647889';
    return `<g class="marker-hit" data-marker="${k}">
      <circle cx="${sx(i).toFixed(1)}" cy="${sy(m.close).toFixed(1)}" r="4.5"
        fill="#16202B" stroke="${col}" stroke-width="2"/>
      <title>${esc(m.date)} · ${pct(m.ret_pct)} · ${m.sigma}σ · ${esc(m.kind)}</title>
    </g>`;
  }).join('');

  const first = points[0].date, last = points[points.length - 1].date;
  return `<svg viewBox="0 0 ${width} ${height}">
    ${grid}${lvl}
    <path d="${path}" fill="none" stroke="#9BAEC0" stroke-width="1.6"/>
    <circle cx="${sx(points.length - 1).toFixed(1)}" cy="${sy(ys[ys.length - 1]).toFixed(1)}"
      r="3" fill="#E8EFF5"/>
    ${mk}
    <text x="${pad}" y="${height - 6}" fill="#647889" font-size="9.5">${esc(first)}</text>
    <text x="${width - 46}" y="${height - 6}" fill="#647889" font-size="9.5"
      text-anchor="end">${esc(last)}</text>
  </svg>`;
}

function reliabilityChart(rows, width = 400, height = 200, pad = 40) {
  if (!rows || rows.length < 4) return '';
  const xs = rows.map((r) => r.pred_mean), ysv = rows.map((r) => r.actual_rate);
  let lo = Math.min(...xs, ...ysv), hi = Math.max(...xs, ...ysv);
  const p = (hi - lo) * 0.12 || 0.01; lo -= p; hi += p;
  const sx = (v) => pad + ((v - lo) / (hi - lo)) * (width - pad - 14);
  const sy = (v) => height - pad - ((v - lo) / (hi - lo)) * (height - pad - 16);
  const path = rows.map((r, i) => `${i ? 'L' : 'M'}${sx(r.pred_mean).toFixed(1)},${sy(r.actual_rate).toFixed(1)}`).join(' ');
  const dots = rows.map((r) => `<circle cx="${sx(r.pred_mean).toFixed(1)}"
    cy="${sy(r.actual_rate).toFixed(1)}" r="3" fill="#5FA8A0"><title>predicted ${(r.pred_mean * 100).toFixed(1)}% · observed ${(r.actual_rate * 100).toFixed(1)}% · n=${r.n}</title></circle>`).join('');
  return `<svg viewBox="0 0 ${width} ${height}">
    <line x1="${sx(lo)}" y1="${sy(lo)}" x2="${sx(hi)}" y2="${sy(hi)}"
      stroke="#647889" stroke-width="1" stroke-dasharray="4 4"/>
    <path d="${path}" fill="none" stroke="#5FA8A0" stroke-width="2"/>${dots}
    <text x="${pad}" y="${height - 10}" fill="#647889" font-size="9.5">predicted probability →</text>
    <text x="${width - 12}" y="${sy(hi) + 12}" fill="#647889" font-size="9.5"
      text-anchor="end">perfect calibration</text>
  </svg>`;
}

/* ── board ───────────────────────────────────────────────────────── */
async function renderBoard() {
  const tiles = await fetch('/api/board').then((r) => r.json());
  $('#board').innerHTML = tiles.map((t) => {
    const cls = t.chg_pct > 0.02 ? 'up' : t.chg_pct < -0.02 ? 'dn' : 'fl';
    return `<div class="tile" title="${esc(t.availability)} · as of ${esc(t.as_of)}">
      <div class="n">${esc(t.label)}</div>
      <div class="v">${fmt(t.value, t.value > 1000 ? 0 : 2)}</div>
      <div class="c ${cls}">${pct(t.chg_pct)}</div>
      <div class="lag">${t.lag_sessions ? 'lag 1' : 'live'}</div>
    </div>`;
  }).join('');
}

/* ── forecast tab ────────────────────────────────────────────────── */
function renderForecast() {
  const d = S.data, f = d.forecast;
  const abstain = !f.acted;
  const up = f.proba_outperform > 0.5;

  $('#symbar').innerHTML = `
    <div class="sym">${esc(f.symbol)}</div>
    <div class="co">${esc(f.sector)} · last close ₹${fmt(f.last_close)}</div>
    <div class="stamp">forecast for the session after ${esc(f.as_of)} ·
      target ${esc(f.target_date)} · model ${esc(f.model_version)}</div>`;

  const dirLabel = abstain ? 'No useful opinion'
    : up ? 'Leans out-performer' : 'Leans laggard';
  const dirCls = abstain ? 'none' : up ? 'bull' : 'bear';
  const confCls = abstain ? 'none' : f.confidence >= 0.6 ? 'firm' : '';
  const needle = Math.min(96, Math.max(4, 50 + (f.proba_outperform - 0.5) * 900));

  $('#verdict').innerHTML = `
    <div class="vtop">
      <div class="callout">
        <div class="k">Directional call</div>
        <div class="v ${dirCls}">${dirLabel}</div>
        <div class="sub">vs the median NSE large-cap, next session</div>
      </div>
      <div class="callout">
        <div class="k">P(out-performs)</div>
        <div class="v">${(f.proba_outperform * 100).toFixed(1)}%</div>
        <div class="sub">conviction ${(f.conviction * 100).toFixed(2)} pp from a coin toss</div>
      </div>
      <div class="callout right">
        <div class="k">Regime</div>
        <div class="v">${esc(f.regime.volatility)}</div>
        <div class="sub">${esc(f.regime.trend.replace('_', ' '))}</div>
      </div>
    </div>
    <div class="confwrap">
      <div class="confhead">
        <div class="lbl">Model confidence</div>
        <div class="num ${confCls}">${abstain ? '—' : (f.confidence * 100).toFixed(1) + '%'}</div>
      </div>
      <div class="band">
        <div class="track"></div><div class="dead"></div>
        <div class="needle" style="left:${needle}%;background:var(--${abstain ? 'warn' : up ? 'bull' : 'bear'})"></div>
      </div>
      <div class="bandkeys"><span>lags</span><span>coin-flip zone</span><span>out-performs</span></div>
      <div class="deadnote">${abstain
        ? `Inside the coin-flip zone. Conviction is ${(f.conviction * 100).toFixed(2)} percentage
           points from 50/50, below the ${(f.abstain_threshold * 100).toFixed(1)} pp this system
           requires before it will commit — so it does not. On the held-out window it
           declined roughly three sessions in five.`
        : `Outside the coin-flip zone, but only just. Held out, forecasts from this
           model were right ${(f.track_record.accuracy * 100).toFixed(1)}% of the time against a
           ${(f.track_record.coin_flip_baseline * 100).toFixed(1)}% coin flip. Treat this as weak
           evidence, not a signal.`}</div>
    </div>`;

  // price + reference levels
  const L = d.levels || {};
  $('#px-note').textContent = `${d.price.sessions} sessions`;
  $('#px-chart').innerHTML = lineChart({
    points: d.price.points,
    levels: [
      { y: L.r1, label: `${fmt(L.r1, 0)} R1`, color: '#C97A82' },
      { y: L.pivot, label: `${fmt(L.pivot, 0)} pivot`, color: '#D9A441' },
      { y: L.s1, label: `${fmt(L.s1, 0)} S1`, color: '#5FA8A0' },
    ],
  });
  $('#px-basis').textContent = `${L.basis || ''}. ${L.note || ''}`;

  // aspects
  const asp = (f.aspects || []).filter((a) => a.points !== 0);
  const max = Math.max(...asp.map((a) => Math.abs(a.points)), 1);
  const net = asp.reduce((s, a) => s + a.points, 0);
  $('#aspects').innerHTML = `
    <tr><th style="width:44%">Aspect</th><th style="width:26%">Signal</th>
      <th style="width:30%">Points</th></tr>
    ${asp.map((a) => {
      const p = a.points > 0, w = (Math.abs(a.points) / max) * 46;
      const ev = (a.evidence || []).map((e) =>
        `${e.feature}${e.value !== undefined ? '=' + e.value : ''}`).join('  ·  ');
      return `<tr>
        <td><div class="aname">${esc(a.aspect)}</div>
          ${ev ? `<div class="aevid">${esc(ev)}</div>` : ''}</td>
        <td><div class="bar-mini">
          <em style="background:var(--${p ? 'bull' : 'bear'});${p ? 'left' : 'right'}:50%;width:${w}%"></em>
          <div class="mid"></div></div></td>
        <td class="contrib ${p ? 'p' : 'n'}">${p ? '+' : ''}${a.points}</td></tr>`;
    }).join('')}
    <tr class="nettotal"><td>Net</td><td></td>
      <td class="contrib ${net > 0 ? 'p' : 'n'}">${net > 0 ? '+' : ''}${net}</td></tr>`;

  // rail
  $('#levels-card').innerHTML = `<h3>Reference levels</h3>
    <div class="lvlrow"><span>R2</span><b>₹${fmt(L.r2)}</b></div>
    <div class="lvlrow"><span>R1</span><b>₹${fmt(L.r1)}</b></div>
    <div class="lvlrow pivot"><span>Pivot</span><b>₹${fmt(L.pivot)}</b></div>
    <div class="lvlrow"><span>S1</span><b>₹${fmt(L.s1)}</b></div>
    <div class="lvlrow"><span>S2</span><b>₹${fmt(L.s2)}</b></div>
    <div class="lvlrow"><span>20-session range</span><b>₹${fmt(L.low_20, 0)}–${fmt(L.high_20, 0)}</b></div>
    <div class="lvlrow"><span>200-session range</span><b>₹${fmt(L.low_200, 0)}–${fmt(L.high_200, 0)}</b></div>
    <div class="micro">Arithmetic on prices that already printed — not forecasts.</div>`;

  $('#regime-card').innerHTML = `<h3>Regime</h3>
    <div class="metric"><span>Volatility</span>
      <div class="pill ${f.regime.volatility === 'elevated' ? 'warn' : 'flat'}">${esc(f.regime.volatility)}</div></div>
    <div class="metric"><span>Trend state</span>
      <div class="pill flat">${esc(f.regime.trend.replace('_', ' '))}</div></div>
    <div class="metric"><span>Typical daily range</span><b>₹${fmt(f.typical_daily_range)}</b></div>
    <div class="metric"><span>Mapped drivers</span><b>${(f.drivers || []).length}</b></div>
    <div class="micro">Regime is the bandit's context — arm weights are held
      separately for each of the nine buckets.</div>`;

  const arms = f.arms || {};
  $('#arms-card').innerHTML = `<h3>Ensemble arms</h3>
    ${Object.keys(arms.probas || {}).map((a) => `
      <div class="metric"><span>${esc(a)}</span>
        <b>${(arms.probas[a]).toFixed(3)} <span style="color:var(--ink-faint);font-weight:400">
        · w ${(arms.weights[a] ?? 0).toFixed(2)}</span></b></div>`).join('')}
    <div class="micro">Blend: ${esc(arms.mode)}. The two-parameter reversal rule is
      kept as an honesty check — it matched the GBM's accuracy on held-out data.</div>`;

  const t = f.track_record;
  $('#baseline-card').innerHTML = `<h3>Model vs baseline</h3>
    <div class="metric"><span>Held out, ${esc(t.window)}</span><b>${(t.accuracy * 100).toFixed(1)}%</b></div>
    <div class="metric"><span>Coin flip</span><b>${(t.coin_flip_baseline * 100).toFixed(1)}%</b></div>
    <div class="metric"><span>Edge</span><div class="pill bull">+${t.edge_pp} pp</div></div>
    <div class="metric"><span>Fold t-stat</span><b>${t.fold_t_stat}</b></div>
    <div class="metric"><span>Folds won</span><b>${esc(t.folds_won)}</b></div>
    <div class="metric"><span>Calibration error</span><b>${t.ece_pooled_pp} pp</b></div>
    <div class="metric"><span>At 10% coverage</span><b>${(t.accuracy_at_10pct_coverage * 100).toFixed(1)}%</b></div>`;

  $('#notbuilt').innerHTML = `<div class="chead"><h3>Panels this system does not have</h3>
      <div class="note">stated rather than filled with something plausible</div></div>
    ${(d.not_built || []).map((n) => `<div class="nb-row"><b>${esc(n.panel)}</b>
      <p>${esc(n.reason)}</p></div>`).join('')}`;

  $('#disc').textContent = f.disclaimer;
  loadNarration();
}

async function loadNarration() {
  const sym = S.symbol;
  $('#narr').textContent = 'Generating…';
  $('#narr-foot').textContent = '';
  try {
    const r = await fetch(`/api/narration/${encodeURIComponent(sym)}`);
    if (!r.ok) throw new Error('unavailable');
    const n = await r.json();
    if (S.symbol !== sym) return;              // a newer symbol won the race
    $('#narr').textContent = n.text;
    const bad = (n.unverified_numbers || []).length;
    $('#narr-foot').textContent = bad
      ? `⚠ ${bad} number(s) not found in the evidence: ${n.unverified_numbers.join(', ')}`
      : `Written by ${n.model} from the evidence on this page. Every number was checked against it; the model supplies no facts of its own.`;
  } catch {
    $('#narr').textContent = 'Narration unavailable — the rest of the page does not depend on it.';
  }
}

/* ── simple tab ──────────────────────────────────────────────────── */
async function renderSimple() {
  const d = S.data;
  if (!d) return;
  const f = d.forecast;
  const sym = S.symbol;

  $('#simplebar').innerHTML = `<div class="sym">Simple view</div>
    <div class="co">${esc(f.symbol)} · for the session after ${esc(f.as_of)} —
      same forecast, same evidence, no jargon</div>`;

  if (S.simple?.forSymbol !== sym) {
    $('#s-answer').innerHTML = '<div class="micro">Writing the plain-language version…</div>';
    ['#s-tally', '#s-reasons', '#s-levels'].forEach((k) => { $(k).innerHTML = ''; });
    ['#s-means', '#s-closing', '#s-genfoot'].forEach((k) => { $(k).innerHTML = ''; });
    try {
      const r = await fetch(`/api/simple/${encodeURIComponent(sym)}`);
      if (!r.ok) throw new Error((await r.json()).detail || 'unavailable');
      const j = await r.json();
      j.forSymbol = sym;
      S.simple = j;
    } catch (e) {
      $('#s-answer').innerHTML = `<div class="micro">The plain-language view could not be
        written — ${esc(e.message)}. Set GEMINI_API_KEY (free tier) or OPENAI_API_KEY
        in .env. Every other tab works without it.</div>`;
      return;
    }
  }
  if (S.symbol !== sym) return;            // a newer symbol won the race

  const v = S.simple;
  const ev = v.evidence || {};
  const abstain = !f.acted;
  const up = f.proba_outperform > 0.5;
  const dirCls = abstain ? 'none' : up ? 'up' : 'dn';

  // The verdict phrase inside the headline is highlighted rather than the whole
  // sentence, so the colour marks the claim and not the framing around it.
  //
  // The direction words are matched whichever way the forecast went, because
  // an abstaining forecast still has a tilt and the model describes it - an
  // earlier version only looked for abstention words, so a "no call" headline
  // reading "more likely to do better" got no highlight at all. On an abstain
  // the tilt is coloured amber rather than teal or rose, which is the whole
  // point: it is a lean the system declined to act on, not a call.
  const phrase = /(slightly more likely to [a-z ]*?(?:better|worse|beat|lag|rise|fall)[a-z]*|no useful opinion|not confident enough|do better than[a-z \-]*|do worse than[a-z \-]*|out-?perform\w*|under-?perform\w*|beat the pack|lag the pack)/i;
  let headline = esc(v.headline);
  const m = headline.match(phrase);
  if (m) headline = headline.replace(m[0], `<em class="${dirCls}">${m[0]}</em>`);

  $('#s-answer').innerHTML = `
    <div class="who">${esc(f.symbol)} · ${esc(f.sector)} ·
      <b>closed at ₹${fmt(f.last_close)}</b> on ${esc(f.as_of)}</div>
    <div class="big">${headline}</div>
    <p class="plain">${esc(v.opening)}</p>
    <div class="sureline">
      <div class="sureno ${abstain ? 'none' : ''}">${abstain ? 'no call' : (f.confidence * 100).toFixed(1) + '%'}</div>
      <div class="suretext">${esc(v.confidence_line)}</div>
    </div>`;

  const t = ev.tally || {};
  $('#s-tallysub').textContent = `${(v.reasons || []).length} things were checked. `
    + 'Each one either pushed the price up or pulled it down.';
  $('#s-tally').innerHTML = `
    <div class="side up">
      <div class="lbl">Points pushing it up</div>
      <div class="pts p">+${t.points_pushing_up ?? 0}</div>
      <div class="cnt">from ${t.checks_pushing_up ?? 0} of ${(v.reasons || []).length} checks</div>
    </div>
    <div class="side">
      <div class="lbl">Points pulling it down</div>
      <div class="pts n">${t.points_pulling_down ?? 0}</div>
      <div class="cnt">from ${t.checks_pulling_down ?? 0} of ${(v.reasons || []).length} checks</div>
    </div>`;

  // Biggest movers first, so the argument reads in order of importance.
  const evReasons = new Map((ev.reasons || []).map((r) => [r.aspect, r]));
  const reasons = [...(v.reasons || [])].sort((a, b) => Math.abs(b.points) - Math.abs(a.points));
  $('#s-reasons').innerHTML = reasons.map((r) => {
    const p = r.points > 0;
    const meta = evReasons.get(r.aspect);
    const isRev = meta?.is_reversal;
    const url = (r.source || '').match(/https?:\/\/\S+/);
    const cite = r.source
      ? (url ? `<a class="cite" href="${esc(url[0])}" target="_blank" rel="noopener">${esc(r.source.replace(url[0], '').trim() || 'Open the filing')}</a>`
             : `<div class="cite">${esc(r.source)}</div>`)
      : '';
    return `<div class="reason-card ${p ? 'pos' : 'neg'}">
      <div class="score"><div class="v ${p ? 'p' : 'n'}">${p ? '+' : ''}${r.points}</div>
        <div class="u">points</div></div>
      <div class="txt">
        <h4>${esc(r.title)}</h4>
        <p>${esc(r.body)}</p>
        ${isRev ? '<span class="rev">reversal effect</span>' : ''}${cite}
      </div></div>`;
  }).join('');

  const ex = v.explainer || {};
  $('#s-means').innerHTML = `<h3>${esc(ex.question || '')}</h3>
    ${(ex.paragraphs || []).map((x) => `<p>${esc(x)}</p>`).join('')}`;

  const lvlCls = ['up', 'mid', 'dn'];
  $('#s-levels').innerHTML = (v.levels || []).map((l, i) => `
    <div class="lvl">
      <div class="k">${esc(l.label)}</div>
      <div class="n ${lvlCls[i] || 'mid'}">₹${fmt(l.price)}</div>
      <div class="d">${esc(l.meaning)}</div>
    </div>`).join('');

  $('#s-closing').innerHTML = `<b>So, adding it up.</b> ${esc(v.closing)}`;
  $('#s-disc').textContent = f.disclaimer;

  const bad = (v.unverified_numbers || []).length;
  $('#s-genfoot').innerHTML = `Every number above came from the model and the
    exchange data, not from the language model — it receives the evidence already
    computed and rewrites it in plain language, and is instructed never to add a
    fact or sound more certain than the confidence allows. Written by
    <b>${esc(v.provider)} · ${esc(v.model)}</b>${v.cached ? ' (cached for this session)' : ''}.
    ${bad ? `<span class="warn">⚠ ${bad} number(s) could not be traced to the evidence:
      ${esc((v.unverified_numbers || []).join(', '))}.</span>`
      : 'Every number in the text was checked against that evidence and matched.'}`;
}

/* ── context tab ─────────────────────────────────────────────────── */
function renderContext() {
  const d = S.data, f = d.forecast;
  $('#ctxbar').innerHTML = `<div class="sym">${esc(f.symbol)}</div>
    <div class="co">what moved it, and whether we can tell</div>
    <div class="stamp">${d.coverage.significant_moves} moves beyond 2σ ·
      ${d.coverage.explained} matched to a filing ·
      ${Math.round((1 - (d.coverage.explained_share ?? 0)) * 100)}% unexplained</div>`;

  $('#ctx-chart').innerHTML = lineChart({
    points: d.price.points, markers: d.price.markers, height: 250,
  });

  const rows = d.attribution || [];
  $('#attr').innerHTML = rows.length ? `
    <tr><th style="width:84px">Date</th><th style="width:78px">Move</th>
      <th>What was found</th><th style="width:96px">Attribution</th></tr>
    ${rows.map((m) => {
      const p = m.ret > 0;
      const body = m.kind === 'unexplained'
        ? `<b>No qualifying evidence.</b> <div class="why">${esc(m.rationale || '')}</div>`
        : `<b>${esc((m.headline || '').slice(0, 200))}</b>
           <div class="why">${esc(m.rationale || '')}</div>`;
      const src = m.source_url
        ? `<a class="src" href="${esc(m.source_url)}" target="_blank" rel="noopener">Open the filing</a>` : '';
      return `<tr>
        <td class="num">${esc(m.date)}<div style="font-size:10.5px;color:var(--ink-faint);margin-top:2px">
          ${m.sigma}σ${m.market_wide ? ' · market-wide' : ''}</div></td>
        <td class="mv ${p ? 'p' : 'n'}">${pct(m.ret * 100)}</td>
        <td class="reason">${body}${src}</td>
        <td><span class="cb cb-${esc(m.band)}">${m.kind === 'unexplained'
          ? 'none' : esc(m.band) + ' ' + (m.confidence ?? 0).toFixed(2)}</span></td></tr>`;
    }).join('')}` : '<tr><td class="micro">No moves beyond 2σ recorded.</td></tr>';

  $('#drivers').innerHTML = (d.drivers || []).map((v) => {
    const c = v.corr_120d ?? 0;
    const col = c >= 0 ? 'var(--info)' : 'var(--bear)';
    return `<div class="drv">
      <div class="dn2">${esc(v.driver)}
        <div class="lagtag">${v.lag_sessions ? 'lagged one session' : 'same session'}</div></div>
      <div class="corrbar"><em style="width:${Math.min(100, Math.abs(c) * 100)}%;background:${col}"></em></div>
      <div class="corr">${c >= 0 ? '+' : ''}${fmt(c, 3)}</div>
      <div class="chg ${(v.last_chg_pct ?? 0) > 0 ? 'up' : (v.last_chg_pct ?? 0) < 0 ? 'dn' : 'fl'}">
        ${pct(v.last_chg_pct)}</div></div>`;
  }).join('') || '<div class="micro">No mapped drivers.</div>';

  $('#tl').innerHTML = (d.timeline || []).map((e) => {
    const cls = e.kind === 'announcement' ? 'ev' : e.ret > 0 ? 'up' : 'dn';
    return `<div class="tlitem ${cls}">
      <div class="tldate">${esc(e.date)}</div>
      <div class="tlhead">${e.kind === 'unexplained' ? 'Unexplained move' : 'Filing matched'}
        <span class="tlmove ${e.ret > 0 ? 'up' : 'dn'}">${pct(e.ret * 100)}</span></div>
      <div class="tlbody">${esc((e.headline || e.rationale || '').slice(0, 150))}</div></div>`;
  }).join('') || '<div class="micro">Nothing recorded.</div>';

  const c = d.coverage;
  $('#cov-card').innerHTML = `<h3>Source coverage</h3>
    <div class="metric"><span>NSE filings, all time</span><b>${c.filings_total}</b></div>
    <div class="metric"><span>Filings, last 30 days</span><b>${c.filings_30d}</b></div>
    <div class="metric"><span>Corporate actions</span><b>${c.corporate_actions}</b></div>
    <div class="metric"><span>Moves beyond 2σ</span><b>${c.significant_moves}</b></div>
    <div class="metric"><span>Matched to a filing</span><b>${c.explained}</b></div>
    <div class="metric"><span>Unexplained</span>
      <div class="pill flat">${c.unexplained}</div></div>
    <div class="micro">Only ${Math.round((c.explained_share ?? 0) * 100)}% of this
      symbol's large moves have an identifiable cause. Across the universe it is
      38%. The rest are labelled unexplained rather than given a story.</div>`;
}

/* ── compare ─────────────────────────────────────────────────────── */
const BASKETS = {
  metals: ['HINDZINC', 'VEDL', 'NATIONALUM', 'HINDALCO', 'JSWSTEEL'],
  it: ['TCS', 'INFY', 'HCLTECH', 'WIPRO', 'TECHM'],
  bank: ['HDFCBANK', 'ICICIBANK', 'SBIN', 'AXISBANK', 'KOTAKBANK'],
};

async function renderCompare(which = 'metals') {
  $('#cmp').innerHTML = '<tr><td class="micro">Fitting…</td></tr>';
  const r = await fetch('/api/compare', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ symbols: BASKETS[which] }),
  }).then((x) => x.json());

  $('#cmp').innerHTML = `
    <tr><th>Symbol</th><th>Sector</th><th>Call</th><th>P(out-perform)</th>
      <th>Confidence</th><th>Conviction</th><th>Regime</th><th>Top aspect</th></tr>
    ${r.results.map((s) => {
      const cls = !s.acted ? 'flat' : s.proba_outperform > 0.5 ? 'bull' : 'bear';
      const label = !s.acted ? 'no call' : s.proba_outperform > 0.5 ? 'out-perform' : 'lag';
      return `<tr class="clickable" data-sym="${esc(s.symbol)}">
        <td><b>${esc(s.symbol)}</b></td>
        <td style="color:var(--ink-dim)">${esc(s.sector)}</td>
        <td><span class="pill ${cls}">${label}</span></td>
        <td class="num">${(s.proba_outperform * 100).toFixed(1)}%</td>
        <td class="num">${s.acted ? (s.confidence * 100).toFixed(1) + '%' : '—'}</td>
        <td class="num">${(s.conviction * 100).toFixed(2)} pp</td>
        <td style="color:var(--ink-dim)">${esc(s.regime)}</td>
        <td style="color:var(--ink-dim)">${esc(s.top_aspect || '—')}</td></tr>`;
    }).join('')}`;
  $('#cmp-note').textContent = r.note;
  $$('#cmp tr.clickable').forEach((tr) => tr.addEventListener('click', () => {
    load(tr.dataset.sym); switchTab('forecast');
  }));
}

/* ── history ─────────────────────────────────────────────────────── */
async function renderHistory() {
  const rows = await fetch(`/api/history?symbol=${encodeURIComponent(S.symbol)}&limit=40`)
    .then((r) => r.json());
  $('#histbar').innerHTML = `<div class="sym">${esc(S.symbol)}</div>
    <div class="co">every forecast written for this symbol, resolved and scored</div>
    <div class="stamp">${rows.length} rows</div>`;
  if (!rows.length) {
    $('#hist').innerHTML = `<tr><td class="micro">No forecasts stored for
      ${esc(S.symbol)} yet. The replay covered 30 randomly sampled symbols.</td></tr>`;
    return;
  }
  $('#hist').innerHTML = `
    <tr><th>Target session</th><th>Call</th><th>P(out)</th><th>Actual</th>
      <th>Result</th><th>Reward</th><th>Regime</th></tr>
    ${rows.map((h) => {
      const acted = h.acted === 1;
      const res = h.correct === null || h.correct === undefined ? '—'
        : h.correct ? '<span class="tick">✓</span>' : '<span class="cross">✗</span>';
      return `<tr>
        <td class="num">${esc(h.target_date)}</td>
        <td>${acted ? (h.proba > 0.5 ? 'out-perform' : 'lag')
          : '<span style="color:var(--ink-faint)">no call</span>'}</td>
        <td class="num">${(h.proba * 100).toFixed(1)}%</td>
        <td class="num ${(h.actual_ret_rel ?? 0) > 0 ? 'up' : 'dn'}">
          ${h.actual_ret_rel === null ? '—' : pct(h.actual_ret_rel * 100)}</td>
        <td>${res}</td>
        <td class="rw ${(h.reward ?? 0) > 0 ? 'p' : (h.reward ?? 0) < 0 ? 'n' : ''}">
          ${h.reward === null ? '—' : (h.reward > 0 ? '+' : '') + fmt(h.reward, 3)}</td>
        <td style="color:var(--ink-dim)">${esc(h.regime || '—')}</td></tr>`;
    }).join('')}`;
}

/* ── performance ─────────────────────────────────────────────────── */
async function renderPerformance() {
  if (!S.perf) S.perf = await fetch('/api/performance').then((r) => r.json());
  if (!S.bandit) S.bandit = await fetch('/api/bandit').then((r) => r.json());
  const p = S.perf, t = p.held_out;

  const cards = [
    [`${(t.accuracy * 100).toFixed(1)}%`, `Held-out accuracy<br>${t.window}`, ''],
    [`${(t.coin_flip_baseline * 100).toFixed(1)}%`, 'Coin-flip baseline<br>what beating nothing looks like', 'dim'],
    [`+${t.edge_pp} pp`, `Held-out edge<br>t = ${t.fold_t_stat} · ${t.folds_won} folds`, ''],
    [`${t.ece_pooled_pp} pp`, 'Calibration error<br>a stated 52% means 52%', 'warn'],
    [`${(t.accuracy_at_10pct_coverage * 100).toFixed(1)}%`, 'At 10% coverage<br>the edge lives in the tails', ''],
  ];
  if (p.n_acted) {
    cards.push([`${(p.accuracy * 100).toFixed(1)}%`, `Live accuracy<br>${p.n_acted} resolved calls`, '']);
    cards.push([`${(p.majority_baseline_on_acted * 100).toFixed(1)}%`,
      `Baseline on those rows<br>edge +${p.edge_vs_majority_pp} pp`, 'dim']);
    cards.push([`${(p.abstention_rate * 100).toFixed(0)}%`, 'No opinion<br>silence is the default', 'dim']);
    cards.push([`${fmt(p.mean_reward, 4)}`, 'Mean reward<br>negative below 60% by design', 'warn']);
  }
  $('#stats').innerHTML = cards.map(([n, l, c]) =>
    `<div class="stat"><div class="n ${c}">${n}</div><div class="l">${l}</div></div>`).join('');

  const rel = reliabilityChart(p.reliability);
  $('#rel-chart').innerHTML = rel || `<div class="micro">${esc(p.reliability_note
    || 'Not enough resolved forecasts.')} Held-out pooled ECE is ${t.ece_pooled_pp} pp.</div>`;
  $('#rel-note').textContent = rel ? `${p.n_acted} live resolved forecasts` : 'held-out only';

  const reg = Object.entries(p.by_regime || {}).sort((a, b) => b[1].n - a[1].n);
  $('#regimes').innerHTML = reg.length ? `
    <tr><th>Regime</th><th>n</th><th>Accuracy</th><th>Mean reward</th></tr>
    ${reg.map(([k, v]) => {
      const good = v.accuracy >= 0.52, bad = v.accuracy < 0.50;
      return `<tr><td>${esc(k)}</td><td class="num">${v.n}</td>
        <td class="num" style="color:var(--${good ? 'bull' : bad ? 'bear' : 'ink'})">
          ${(v.accuracy * 100).toFixed(2)}%</td>
        <td class="rw ${v.mean_reward > 0 ? 'p' : 'n'}">${fmt(v.mean_reward, 4)}</td></tr>`;
    }).join('')}` : '<tr><td class="micro">No resolved forecasts yet.</td></tr>';

  const st = S.bandit.state || [];
  const byRegime = {};
  st.forEach((r) => { (byRegime[r.regime] ??= {})[r.arm] = r; });
  const armNames = S.bandit.arms;
  $('#bandit').innerHTML = `
    <tr><th>Regime</th>${armNames.map((a) => `<th>${esc(a)}</th>`).join('')}<th>n</th></tr>
    ${Object.entries(byRegime).sort((a, b) =>
      (Math.max(...Object.values(b[1]).map((x) => x.n_pulls)))
      - (Math.max(...Object.values(a[1]).map((x) => x.n_pulls))))
      .map(([regime, arms]) => {
        const n = Math.max(...Object.values(arms).map((x) => x.n_pulls));
        const best = Math.max(...armNames.map((a) => arms[a]?.posterior_mean ?? 0));
        return `<tr><td>${esc(regime)}</td>
          ${armNames.map((a) => {
            const v = arms[a]?.posterior_mean;
            const lead = v && v === best;
            return `<td class="num" style="${lead ? 'color:var(--bull);font-weight:500' : ''}">
              ${v === undefined ? '—' : fmt(v, 3)}</td>`;
          }).join('')}
          <td class="num">${n}</td></tr>`;
      }).join('')}`;
  const g = S.bandit.guardrails;
  const ev = S.bandit.recent_guardrail_events || [];
  $('#bandit-note').innerHTML = `Weights stay uniform below ${g.min_pulls_per_regime}
    resolved forecasts per regime; no arm may exceed ${(g.weight_ceiling * 100).toFixed(0)}%
    or fall below ${(g.weight_floor * 100).toFixed(0)}%. A regime whose rolling accuracy
    over ${g.drift_window} forecasts falls under ${(g.drift_threshold * 100).toFixed(0)}%
    reverts to uniform — that has fired ${ev.length ? `${ev.length}+ times` : 'not yet'}.
    Highlighted cell is the leading arm in that regime.`;
}

/* ── shell ───────────────────────────────────────────────────────── */
function switchTab(name) {
  S.tab = name;
  $$('#tabs div').forEach((d) => d.classList.toggle('on', d.dataset.tab === name));
  $$('.tab').forEach((t) => t.classList.toggle('is-on', t.dataset.panel === name));
  if (name === 'simple') renderSimple();
  if (name === 'compare') renderCompare();
  if (name === 'history') renderHistory();
  if (name === 'performance') renderPerformance();
  if (name === 'context' && S.data) renderContext();
}

async function load(symbol) {
  S.symbol = symbol.toUpperCase();
  $('#sym').value = S.symbol;
  $('#verdict').innerHTML = '<div class="micro">Fitting the ensemble on everything up to the last close…</div>';
  try {
    S.data = await fetch(`/api/dashboard/${encodeURIComponent(S.symbol)}`).then((r) => {
      if (!r.ok) throw new Error('not found');
      return r.json();
    });
    renderForecast();
    if (S.tab === 'simple') renderSimple();
    if (S.tab === 'context') renderContext();
    if (S.tab === 'history') renderHistory();
  } catch (e) {
    $('#verdict').innerHTML = `<div class="micro">Could not load ${esc(S.symbol)} — ${esc(e.message)}.</div>`;
  }
}

/* typeahead */
let taItems = [], taSel = 0, taTimer = null;
async function refreshTa() {
  const q = $('#sym').value.trim();
  taItems = await fetch(`/api/symbols/search?q=${encodeURIComponent(q)}&limit=8`)
    .then((r) => r.json()).catch(() => []);
  if (!taItems.length) { $('#ta').classList.remove('open'); return; }
  taSel = 0;
  $('#ta').innerHTML = taItems.map((it, i) =>
    `<div class="${i === 0 ? 'sel' : ''}" data-sym="${esc(it.symbol)}">
      <span>${esc(it.symbol)}</span><em>${esc(it.sector)}</em></div>`).join('');
  $('#ta').classList.add('open');
  $$('#ta div').forEach((el) => el.addEventListener('mousedown', (e) => {
    e.preventDefault(); $('#ta').classList.remove('open'); load(el.dataset.sym);
  }));
}

$('#sym').addEventListener('input', () => { clearTimeout(taTimer); taTimer = setTimeout(refreshTa, 110); });
$('#sym').addEventListener('focus', refreshTa);
$('#sym').addEventListener('blur', () => setTimeout(() => $('#ta').classList.remove('open'), 150));
$('#sym').addEventListener('keydown', (e) => {
  if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
    e.preventDefault();
    if (!taItems.length) return;
    taSel = (taSel + (e.key === 'ArrowDown' ? 1 : -1) + taItems.length) % taItems.length;
    $$('#ta div').forEach((el, i) => el.classList.toggle('sel', i === taSel));
  } else if (e.key === 'Enter') {
    e.preventDefault();
    $('#ta').classList.remove('open');
    load(taItems[taSel]?.symbol || $('#sym').value);
  } else if (e.key === 'Escape') {
    $('#ta').classList.remove('open');
  }
});

$$('#tabs div').forEach((d) => d.addEventListener('click', () => switchTab(d.dataset.tab)));
$$('[data-goto]').forEach((el) => el.addEventListener('click', () => switchTab(el.dataset.goto)));
$('#cmp-metals').addEventListener('click', () => renderCompare('metals'));
$('#cmp-it').addEventListener('click', () => renderCompare('it'));
$('#cmp-bank').addEventListener('click', () => renderCompare('bank'));

renderBoard();
load('HINDZINC');
