import { motion } from 'framer-motion'
import type { Payload } from '../../lib/types'
import { ConvictionBar, Num, Panel } from '../ui'
import './Overview.css'

const EASE = [0.22, 0.61, 0.36, 1] as const
const inr = (n: number) => n.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })

/** Highlights the verdict phrase, not the whole sentence, so the colour marks
 *  the claim rather than the framing. On an abstention the tilt is amber: a
 *  lean the system declined to act on is neither teal nor rose. */
function Headline({ text, cls }: { text: string; cls: string }) {
  const rx = /(slightly more likely to [a-z ]*?(?:better|worse|beat|lag|rise|fall)[a-z]*|no useful opinion|not confident enough|do better than[a-z -]*|do worse than[a-z -]*|out-?perform\w*|under-?perform\w*)/i
  const m = text.match(rx)
  if (!m) return <>{text}</>
  const [before, after] = text.split(m[0])
  return <>{before}<em className={cls}>{m[0]}</em>{after}</>
}

export default function Overview({ data }: { data: Payload }) {
  const f = data.forecast
  const v = data.simple
  const abstain = !f.acted
  const up = f.proba_outperform > 0.5
  const cls = abstain ? 'none' : up ? 'up' : 'dn'

  if (!v) {
    return (
      <div className="card">
        <div className="micro">
          The plain-language view needs a language-model key — set GEMINI_API_KEY
          (free tier) or OPENAI_API_KEY. Every number and every other tab works
          without it; open Analysis for the full breakdown.
        </div>
      </div>
    )
  }

  const t = v.evidence?.tally ?? {}
  const reasons = [...v.reasons].sort((a, b) => Math.abs(b.points) - Math.abs(a.points))
  const evReasons: Record<string, any> = Object.fromEntries(
    (v.evidence?.reasons ?? []).map((r: any) => [r.aspect, r]),
  )

  return (
    <div className="ov">
      {/* ── the answer ──────────────────────────────────────────── */}
      <motion.div className="ov-answer"
        initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, ease: EASE }}>
        <div className="ov-who">
          {f.symbol} · {f.sector} · <b>closed at ₹{inr(f.last_close)}</b> on {f.as_of}
        </div>
        <h3 className="ov-big"><Headline text={v.headline} cls={cls} /></h3>
        <p className="ov-plain">{v.opening}</p>
        <div className="ov-sure">
          <div>
            <div className={abstain ? 'ov-no none' : 'ov-no'}>
              {abstain ? 'no call' : <Num value={f.confidence * 100} decimals={1} suffix="%" delay={0.3} />}
            </div>
            <div className="ov-no-l">{abstain ? 'below the acting threshold' : 'confidence'}</div>
          </div>
          <p>{v.confidence_line}</p>
        </div>
        {/* The coin-flip band, drawn. Without colour this is what tells a
            reader at a glance that the needle sits in the region where the
            model has nothing to say. */}
        <ConvictionBar proba={f.proba_outperform}
          threshold={f.abstain_threshold} acted={f.acted} />
      </motion.div>

      {/* ── the tally ───────────────────────────────────────────── */}
      <div className="sect">
        <div className="sect-head">How the score added up</div>
        <div className="sect-sub">
          {reasons.length} things were checked. Each one either pushed the price up
          or pulled it down.
        </div>
        {(() => {
          const up_ = Math.abs(t.points_pushing_up ?? 0)
          const dn_ = Math.abs(t.points_pulling_down ?? 0)
          const tot = up_ + dn_ || 1
          return (
            <div className="ov-tally">
              <div className="ov-side">
                <div className="ov-lbl"><span className="dir for" /> pushing it up</div>
                <div className="ov-pts"><Num value={up_} signed delay={0.1} /></div>
                <div className="ov-cnt">from {t.checks_pushing_up ?? 0} of {reasons.length} checks</div>
                <motion.div className="ov-side-bar" initial={{ width: 0 }}
                  animate={{ width: `${(up_ / tot) * 100}%` }}
                  transition={{ duration: 0.8, ease: EASE, delay: 0.2 }} />
              </div>
              <div className="ov-side">
                <div className="ov-lbl"><span className="dir against" /> pulling it down</div>
                <div className="ov-pts"><Num value={-dn_} signed delay={0.15} /></div>
                <div className="ov-cnt">from {t.checks_pulling_down ?? 0} of {reasons.length} checks</div>
                <motion.div className="ov-side-bar" initial={{ width: 0 }}
                  animate={{ width: `${(dn_ / tot) * 100}%` }}
                  transition={{ duration: 0.8, ease: EASE, delay: 0.25 }} />
              </div>
            </div>
          )
        })()}

        {reasons.map((r, i) => {
          const pos = r.points > 0
          const rev = evReasons[r.aspect]?.is_reversal
          const url = r.source?.match(/https?:\/\/\S+/)
          return (
            <motion.div key={r.aspect} className="ov-reason"
              initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }}
              transition={{ delay: Math.min(i * 0.05, 0.3), duration: 0.45, ease: EASE }}>
              <div className="ov-score">
                <div className={pos ? 'v' : 'v n'}>{pos ? '+' : '−'}{Math.abs(r.points)}</div>
                <div className="u">points</div>
                <div className="g">{pos ? '▲ for' : '▽ against'}</div>
              </div>
              <div className="ov-txt">
                <h4>{r.title}</h4>
                <p>{r.body}</p>
                <div className="ov-tags">
                  {rev && <span className="chip hatch">reversal effect</span>}
                  {r.source && (url
                    ? <a className="ov-cite" href={url[0]} target="_blank" rel="noopener noreferrer">
                        {r.source.replace(url[0], '').trim() || 'Open the filing'}
                      </a>
                    : <span className="ov-cite">{r.source}</span>)}
                </div>
              </div>
            </motion.div>
          )
        })}
      </div>

      {/* ── teaching box ────────────────────────────────────────── */}
      <Panel className="ov-means">
        <h3>{v.explainer.question}</h3>
        {v.explainer.paragraphs.map((p, i) => <p key={i}>{p}</p>)}
      </Panel>

      {/* ── levels ──────────────────────────────────────────────── */}
      <div className="sect">
        <div className="sect-head">What to watch tomorrow</div>
        <div className="sect-sub">Three prices worth knowing, if you plan to follow along.</div>
        <div className="grid-3">
          {v.levels.map((l) => (
            <div className="ov-lvl" key={l.label}>
              <div className="k">{l.label}</div>
              <div className="n">₹{inr(l.price)}</div>
              <div className="d">{l.meaning}</div>
            </div>
          ))}
        </div>
        <div className="micro" style={{ marginTop: 11 }}>
          {String(data.levels.basis ?? '')}. Arithmetic on prices that already
          printed — not forecasts.
        </div>
      </div>

      {/* ── the closing guess ───────────────────────────────────── */}
      <motion.div className="ov-closing"
        initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.15, duration: 0.45, ease: EASE }}>
        <div className="eyebrow">So, adding it up</div>
        <p>{v.closing}</p>
      </motion.div>

      <div className="micro ov-disc">{f.disclaimer}</div>
      <div className="micro ov-prov">
        Every number above came from the model and the exchange data. The language
        model receives evidence already computed and rewrites it in plain English;
        it is instructed never to add a fact or sound more certain than the
        confidence allows. Written by <b>{v.provider} · {v.model}</b>
        {v.cached ? ' (cached for this session)' : ''}.{' '}
        {v.unverified_numbers.length
          ? <span className="ov-warn">⚠ {v.unverified_numbers.length} number(s) could not be
              traced to the evidence: {v.unverified_numbers.join(', ')}.</span>
          : 'Every number in the text was checked against that evidence and matched.'}
      </div>
    </div>
  )
}
