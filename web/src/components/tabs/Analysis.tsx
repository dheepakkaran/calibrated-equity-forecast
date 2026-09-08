import { motion } from 'framer-motion'
import type { Payload } from '../../lib/types'
import PriceChart from '../PriceChart'
import './Analysis.css'

const EASE = [0.22, 0.61, 0.36, 1] as const
const pct = (n: number | null | undefined, d = 2) =>
  n === null || n === undefined ? '—' : `${n > 0 ? '+' : ''}${n.toFixed(d)}%`
const num = (n: number, d = 2) =>
  n.toLocaleString('en-IN', { minimumFractionDigits: d, maximumFractionDigits: d })

export default function Analysis({ data }: { data: Payload }) {
  const f = data.forecast
  const t = f.track_record
  const aspects = f.aspects.filter((a) => a.points !== 0)
  const maxPts = Math.max(...aspects.map((a) => Math.abs(a.points)), 1)
  const net = aspects.reduce((s, a) => s + a.points, 0)

  return (
    <div className="an">
      {/* ── the market it is trading in ─────────────────────────── */}
      <div className="sect">
        <div className="sect-head">What the markets were doing</div>
        <div className="sect-sub">
          Each tile says whether the value was knowable at the Indian close. A US
          index closes at 01:30 IST, so its reading is held back a session — using
          today's figure to forecast the session it precedes would be look-ahead bias.
        </div>
        <div className="an-board">
          {data.board.map((b, i) => (
            <motion.div key={b.key} className="an-tile" title={b.availability}
              initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}
              transition={{ delay: Math.min(i * 0.02, 0.22), duration: 0.3, ease: EASE }}>
              <div className="an-tl">{b.label}</div>
              <div className="an-tv">{num(b.value, b.value > 1000 ? 0 : 2)}</div>
              <div className={`an-tc ${b.chg_pct > 0.02 ? 'up' : b.chg_pct < -0.02 ? 'dn' : 'fl'}`}>
                {pct(b.chg_pct)}
              </div>
              <div className="an-tlag">{b.lag_sessions ? 'held back 1 session' : 'live'}</div>
            </motion.div>
          ))}
        </div>
      </div>

      {/* ── why ─────────────────────────────────────────────────── */}
      <div className="sect">
        <div className="sect-head">Why — what actually drove the number</div>
        <div className="sect-sub">
          The model's own attributions, grouped and scaled so the largest is 100.
          Positive pushes toward out-performing. A favourable reading with negative
          points is the reversal effect, not an error: a share that has just run
          ahead tends to give a little back.
        </div>
        <div className="card">
          <table className="d an-asp">
            <thead>
              <tr><th style={{ width: '40%' }}>Aspect</th><th style={{ width: '14%' }}>Share of signal</th>
                <th style={{ width: '26%' }}>Direction</th><th>Points</th></tr>
            </thead>
            <tbody>
              {aspects.map((a) => {
                const pos = a.points > 0
                const w = (Math.abs(a.points) / maxPts) * 46
                return (
                  <tr key={a.aspect}>
                    <td>
                      <div className="an-aname">{a.aspect}</div>
                      <div className="an-aev mono">
                        {a.evidence.map((e) => `${e.feature}${e.value !== undefined ? `=${e.value}` : ''}`).join('  ·  ')}
                      </div>
                    </td>
                    <td className="num">{(a.share_of_signal * 100).toFixed(1)}%</td>
                    <td>
                      <div className="an-bar">
                        <motion.i
                          initial={{ width: 0 }} animate={{ width: `${w}%` }}
                          transition={{ duration: 0.55, ease: EASE }}
                          style={{ background: `var(--${pos ? 'bull' : 'bear'})`, [pos ? 'left' : 'right']: '50%' }} />
                        <span className="an-mid" />
                      </div>
                    </td>
                    <td className={`num an-pts ${pos ? 'p' : 'n'}`}>{pos ? '+' : ''}{a.points}</td>
                  </tr>
                )
              })}
              <tr className="an-net">
                <td>Net</td><td /><td />
                <td className={`num an-pts ${net > 0 ? 'p' : 'n'}`}>{net > 0 ? '+' : ''}{net}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* ── history ─────────────────────────────────────────────── */}
      <div className="sect">
        <div className="sect-head">History — the trend it sits inside</div>
        <div className="sect-sub">
          {data.price.sessions} sessions, with the moves that had evidence attached
          marked. Hover any marker for what was filed.
        </div>
        <div className="card">
          <PriceChart price={data.price} levels={data.levels as Record<string, number>} />
        </div>
      </div>

      {/* ── drivers + regime ───────────────────────────────────── */}
      <div className="grid-2">
        <div className="card">
          <div className="sect-head" style={{ fontSize: 14 }}>What this share tracks</div>
          <div className="micro" style={{ marginBottom: 14 }}>
            Rolling 120-session correlation against the <em>lagged</em> driver — the
            value that was knowable at the close. Same-day correlations read far
            higher and cannot inform a forecast.
          </div>
          {data.drivers.map((d) => {
            const c = d.corr_120d ?? 0
            return (
              <div className="an-drv" key={d.driver}>
                <div className="an-dname">{d.driver}</div>
                <div className="an-dbar">
                  <motion.i initial={{ width: 0 }}
                    animate={{ width: `${Math.min(100, Math.abs(c) * 100)}%` }}
                    transition={{ duration: 0.5, ease: EASE }}
                    style={{ background: c >= 0 ? 'var(--info)' : 'var(--bear)' }} />
                </div>
                <div className="an-dcorr num">{c >= 0 ? '+' : ''}{c.toFixed(3)}</div>
                <div className={`an-dchg num ${(d.last_chg_pct ?? 0) > 0 ? 'up' : (d.last_chg_pct ?? 0) < 0 ? 'dn' : 'fl'}`}>
                  {pct(d.last_chg_pct)}
                </div>
              </div>
            )
          })}
        </div>

        <div className="card">
          <div className="sect-head" style={{ fontSize: 14 }}>Conditions and the four models</div>
          <div className="micro" style={{ marginBottom: 14 }}>
            Regime is the context the weighting is held against — nine buckets,
            each with its own record.
          </div>
          <div className="an-kv"><span>How much it swings</span>
            <span className={`pill ${f.regime.volatility === 'elevated' ? 'warn' : 'flat'}`}>{f.regime.volatility}</span></div>
          <div className="an-kv"><span>Trend</span>
            <span className="pill flat">{f.regime.trend.replace('_', ' ')}</span></div>
          <div className="an-kv"><span>Typical daily swing</span><b>₹{num(f.typical_daily_range)}</b></div>
          <div className="an-sep" />
          {Object.keys(f.arms.probas).map((a) => (
            <div className="an-kv" key={a}>
              <span>{a}</span>
              <b>{f.arms.probas[a].toFixed(3)}
                <span className="an-w"> · weight {(f.arms.weights[a] ?? 0).toFixed(2)}</span></b>
            </div>
          ))}
          <div className="micro" style={{ marginTop: 12 }}>
            Blend: {f.arms.mode}. The two-parameter reversal rule is kept as a
            standing honesty check — it matched the gradient-boosted model's
            accuracy on held-out data, so if the weighting ever loads onto it, the
            machine learning is not earning its place.
          </div>
        </div>
      </div>

      {/* ── track record ───────────────────────────────────────── */}
      <div className="sect" style={{ marginTop: 30 }}>
        <div className="sect-head">Is any of this any good?</div>
        <div className="sect-sub">
          Measured on folds the model never saw during development, read once
          after the configuration was frozen.
        </div>
        <div className="tiles">
          <div className="tile"><div className="tile-n">{(t.accuracy * 100).toFixed(1)}%</div>
            <div className="tile-l">Held-out accuracy<br />{t.window}</div></div>
          <div className="tile"><div className="tile-n dim">{(t.coin_flip_baseline * 100).toFixed(1)}%</div>
            <div className="tile-l">Coin flip<br />what beating nothing looks like</div></div>
          <div className="tile"><div className="tile-n">+{t.edge_pp} pp</div>
            <div className="tile-l">Edge<br />t = {t.fold_t_stat} · {t.folds_won} folds</div></div>
          <div className="tile"><div className="tile-n warn">{t.ece_pooled_pp} pp</div>
            <div className="tile-l">Calibration error<br />a stated 52% means 52%</div></div>
          <div className="tile"><div className="tile-n">{(t.accuracy_at_10pct_coverage * 100).toFixed(1)}%</div>
            <div className="tile-l">At 10% coverage<br />the edge lives in the tails</div></div>
        </div>
        <div className="verify">
          <b>What the edge actually is.</b> A two-parameter rule — a share that
          out-performed today tends to give a little back tomorrow — matches this
          model's accuracy. The models earn their place by producing a calibrated
          conviction score, which is what makes declining to forecast possible;
          they do not beat that rule on raw accuracy, and this system does not
          claim they do.
        </div>
      </div>
    </div>
  )
}
