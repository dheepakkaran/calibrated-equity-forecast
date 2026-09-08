import { motion } from 'framer-motion'
import type { Payload } from '../../lib/types'
import { Panel } from '../ui'
import './Sources.css'

const EASE = [0.22, 0.61, 0.36, 1] as const

export default function Sources({ data }: { data: Payload }) {
  const f = data.forecast
  const c = data.coverage
  const filings = data.attribution.filter((m) => m.kind !== 'unexplained')
  const unexplained = data.attribution.filter((m) => m.kind === 'unexplained')

  return (
    <div className="src">
      {/* ── primary: exchange filings ───────────────────────────── */}
      <div className="sect">
        <div className="sect-head">Primary — exchange filings</div>
        <div className="sect-sub">
          Company disclosures filed with the NSE, matched to unusually large moves
          by date, materiality under SEBI LODR Regulation 30, and directional fit.
          Each one links to the original PDF, so the reasoning can be checked
          rather than trusted.
        </div>
        {filings.length ? filings.map((m, i) => (
          <motion.div key={m.date + i} className="src-row"
            initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}
            transition={{ delay: Math.min(i * 0.05, 0.25), duration: 0.36, ease: EASE }}>
            <div className="src-meta">
              <div className="src-date">{m.date}</div>
              <div className={`src-move ${m.ret > 0 ? 'up' : 'dn'}`}>
                {Math.abs(m.ret * 100).toFixed(2)}%
              </div>
              <div className="src-sig">{m.sigma}σ</div>
            </div>
            <div className="src-body">
              <p className="src-head">{m.headline}</p>
              <p className="src-why">{m.rationale}</p>
              {m.source_url && (
                <a className="src-link" href={m.source_url} target="_blank" rel="noopener noreferrer">
                  Open the original filing
                </a>
              )}
            </div>
            <div className={m.band === 'high' ? 'chip solid' : m.band === 'medium' ? 'chip strong' : 'chip hatch'}>
              {m.band} {m.confidence.toFixed(2)}
            </div>
          </motion.div>
        )) : <div className="card micro">No filing above the materiality threshold matched a large move in this window.</div>}
      </div>

      {/* ── what has no source ──────────────────────────────────── */}
      <div className="sect">
        <div className="sect-head">What has no source</div>
        <div className="sect-sub">
          Listed rather than hidden. A large move nobody can explain is itself
          information, and a system that only showed the moves it could account
          for would be describing a tidier market than the real one.
        </div>
        <Panel>
          <table className="d">
            <thead><tr><th>Date</th><th>Move</th><th>Size</th><th>Searched</th></tr></thead>
            <tbody>
              {unexplained.map((m, i) => (
                <tr key={m.date + i}>
                  <td className="num">{m.date}</td>
                  <td className={`num an-tc ${m.ret > 0 ? 'up' : 'dn'}`}>
                    {Math.abs(m.ret * 100).toFixed(2)}%
                  </td>
                  <td className="num">{m.sigma}σ</td>
                  <td className="micro">{m.rationale}</td>
                </tr>
              ))}
              {!unexplained.length && <tr><td className="micro">Everything in this window had a match.</td></tr>}
            </tbody>
          </table>
        </Panel>
      </div>

      {/* ── market data provenance ──────────────────────────────── */}
      <div className="sect">
        <div className="sect-head">Market and commodity data</div>
        <div className="sect-sub">
          Every series carries the lag at which it was read. This is the single
          place look-ahead bias would enter, so it is stated per series rather
          than assumed.
        </div>
        <Panel>
          <table className="d">
            <thead><tr><th>Series</th><th>Source</th><th>Read at</th><th>Last value</th></tr></thead>
            <tbody>
              {data.board.map((b) => (
                <tr key={b.key}>
                  <td>{b.label}</td>
                  <td className="micro">Yahoo Finance · {b.as_of}</td>
                  <td className="micro">{b.lag_sessions ? 'previous session' : 'same session'}</td>
                  <td className="num">{b.value.toLocaleString('en-IN')}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Panel>
      </div>

      {/* ── coverage ────────────────────────────────────────────── */}
      <div className="sect">
        <div className="sect-head">How much was searched</div>
        <div className="tiles">
          <div className="tile"><div className="tile-n">{(c.filings_total ?? 0).toLocaleString('en-IN')}</div>
            <div className="tile-l">NSE filings for this company<br />all time</div></div>
          <div className="tile"><div className="tile-n">{c.filings_30d ?? 0}</div>
            <div className="tile-l">Filings in the last 30 days</div></div>
          <div className="tile"><div className="tile-n">{c.corporate_actions ?? 0}</div>
            <div className="tile-l">Corporate actions<br />splits, dividends, bonuses</div></div>
          <div className="tile"><div className="tile-n">{c.significant_moves ?? 0}</div>
            <div className="tile-l">Moves beyond 2σ</div></div>
          <div className="tile"><div className="tile-n">{c.explained ?? 0}</div>
            <div className="tile-l">Matched to a cause</div></div>
          <div className="tile"><div className="tile-n dim">{c.unexplained ?? 0}</div>
            <div className="tile-l">Left unexplained</div></div>
        </div>
        <div className="note">
          <b>Attribution is correlational, not causal.</b> A high score means a
          filing was found close in time whose materiality and direction fit the
          move — not that it caused it. Two thirds of NSE filings arrive after the
          15:30 close, so a disclosure is treated as first acting on the following
          session.
        </div>
      </div>

      {/* ── model provenance ────────────────────────────────────── */}
      <div className="sect">
        <div className="sect-head">The model itself</div>
        <div className="grid-2">
          <div className="card">
            <table className="d">
              <tbody>
                <tr><td>Target</td><td>next-session market-relative direction</td></tr>
                <tr><td>Benchmark</td><td>median of 52 NSE large-caps</td></tr>
                <tr><td>Trained through</td><td className="num">{f.as_of}</td></tr>
                <tr><td>Forecast session</td><td className="num">{f.target_date}</td></tr>
                <tr><td>Version</td><td className="num">{f.model_version}</td></tr>
                <tr><td>Forecast id</td><td className="mono" style={{ fontSize: 11 }}>{f.prediction_id}</td></tr>
              </tbody>
            </table>
          </div>
          <div className="card">
            <table className="d">
              <tbody>
                <tr><td>Validation</td><td>expanding walk-forward, 13 folds</td></tr>
                <tr><td>Embargo</td><td>scaled to the label's forward span</td></tr>
                <tr><td>Tuning folds</td><td className="num">0–7 (development)</td></tr>
                <tr><td>Reported folds</td><td className="num">8–12 (read once)</td></tr>
                <tr><td>Leakage tests</td><td className="num">15, all passing</td></tr>
                <tr><td>Shuffled-label AUC</td><td className="num">0.489–0.504</td></tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* ── gaps ────────────────────────────────────────────────── */}
      <div className="sect">
        <div className="sect-head">What this system does not have</div>
        <div className="sect-sub">
          Stated rather than filled with something plausible. A dashboard whose
          argument is calibrated honesty cannot carry invented tiles.
        </div>
        <Panel className="src-nb">
          {data.not_built.map((n) => (
            <div className="src-nbrow" key={n.panel}>
              <b>{n.panel}</b><p>{n.reason}</p>
            </div>
          ))}
        </Panel>
      </div>
    </div>
  )
}
