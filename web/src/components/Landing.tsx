import { motion } from 'framer-motion'
import { useEffect, useRef, useState } from 'react'
import { searchSymbols } from '../lib/api'
import { Num } from './ui'
import './Landing.css'

const EASE = [0.16, 1, 0.3, 1] as const
type Props = { onRun: (symbol: string) => void; onOpenLedger: () => void }

// Staggered reveal. The children declare their own timing off the parent so the
// order is a property of the layout rather than a list of hand-tuned delays.
const container = { hidden: {}, show: { transition: { staggerChildren: 0.07, delayChildren: 0.06 } } }
const item = {
  hidden: { opacity: 0, y: 16 },
  show: { opacity: 1, y: 0, transition: { duration: 0.62, ease: EASE } },
}

export default function Landing({ onRun, onOpenLedger }: Props) {
  const [q, setQ] = useState('')
  const [hits, setHits] = useState<{ symbol: string; sector: string }[]>([])
  const [sel, setSel] = useState(0)
  const input = useRef<HTMLInputElement>(null)

  useEffect(() => { input.current?.focus() }, [])

  useEffect(() => {
    if (!q.trim()) { setHits([]); return }
    let live = true
    const t = setTimeout(async () => {
      const r = await searchSymbols(q).catch(() => [])
      if (live) { setHits(r); setSel(0) }
    }, 110)
    return () => { live = false; clearTimeout(t) }
  }, [q])

  const go = (s?: string) => {
    const sym = (s ?? hits[sel]?.symbol ?? q).trim().toUpperCase()
    if (sym) onRun(sym)
  }

  return (
    <motion.div className="lp"
      initial={{ opacity: 0 }} animate={{ opacity: 1 }}
      exit={{ opacity: 0, filter: 'blur(6px)' }}
      transition={{ duration: 0.4, ease: EASE }}>

      <div className="lp-rule lp-rule-t" />
      <div className="lp-rule lp-rule-b" />

      <button className="lp-ledger" onClick={onOpenLedger}>
        <svg viewBox="0 0 24 24" width="15" height="15" fill="none"
          stroke="currentColor" strokeWidth="1.7" strokeLinecap="round">
          <path d="M3 12a9 9 0 1 0 3-6.7" /><path d="M3 4v4h4" /><path d="M12 8v4l3 2" />
        </svg>
        <span>Tracked guesses</span>
      </button>

      <motion.div className="lp-inner" variants={container} initial="hidden" animate="show">
        <motion.div className="lp-brow" variants={item}>
          <span className="eyebrow">Calibrated selective prediction</span>
          <span className="lp-brow-sep" />
          <span className="eyebrow">NSE · research only</span>
        </motion.div>

        <motion.h1 className="balance" variants={item}>
          Which share should I<br />look at?
        </motion.h1>

        <motion.div className="lp-field" variants={item}>
          <input ref={input} value={q} placeholder="HINDZINC" spellCheck={false}
            autoComplete="off" aria-label="NSE symbol"
            onChange={(e) => setQ(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') { e.preventDefault(); go() }
              else if (e.key === 'ArrowDown') { e.preventDefault(); setSel((s) => (s + 1) % Math.max(1, hits.length)) }
              else if (e.key === 'ArrowUp') { e.preventDefault(); setSel((s) => (s - 1 + hits.length) % Math.max(1, hits.length)) }
            }} />
          <motion.button className="lp-go" onClick={() => go()}
            animate={{ opacity: q.trim() ? 1 : 0.28 }} whileTap={{ scale: 0.96 }}>
            Analyse <kbd>↵</kbd>
          </motion.button>
          {hits.length > 0 && (
            <motion.div className="lp-ta"
              initial={{ opacity: 0, y: -8 }} animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.2, ease: EASE }}>
              {hits.map((h, i) => (
                <div key={h.symbol} className={i === sel ? 'lp-ta-row sel' : 'lp-ta-row'}
                  onMouseDown={(e) => { e.preventDefault(); go(h.symbol) }}
                  onMouseEnter={() => setSel(i)}>
                  <b>{h.symbol}</b><em>{h.sector}</em>
                </div>
              ))}
            </motion.div>
          )}
        </motion.div>

        <motion.p className="lp-hint" variants={item}>
          52 NSE large-caps · <kbd>↑</kbd><kbd>↓</kbd> to pick ·{' '}
          <kbd>⌘</kbd><kbd>K</kbd> from anywhere
        </motion.p>

        {/* The claim, with the comparator beside it. This is the first thing a
            reader should see and the only place on the landing page a number
            is set large. */}
        <motion.div className="lp-claim" variants={item}>
          <div className="lp-stat">
            <div className="lp-stat-n"><Num value={51.0} decimals={1} suffix="%" delay={0.55} /></div>
            <div className="lp-stat-l">right, held out<br />2024-03 → 2026-08</div>
          </div>
          <div className="lp-stat-vs">against</div>
          <div className="lp-stat">
            <div className="lp-stat-n dim"><Num value={50.0} decimals={1} suffix="%" delay={0.7} /></div>
            <div className="lp-stat-l">a coin flip<br />what beating nothing looks like</div>
          </div>
          <div className="lp-stat lp-stat-wide">
            <div className="lp-stat-n"><Num value={56} suffix="%" delay={0.85} /></div>
            <div className="lp-stat-l">of sessions it says<br />nothing useful at all</div>
          </div>
        </motion.div>

        <motion.p className="lp-foot pretty" variants={item}>
          One question, asked honestly: will this share out-perform or lag the
          median NSE large-cap in the next session? That edge is real and it is
          small. Most days the model has nothing worth saying, and on those days
          it says so rather than manufacturing a view.
        </motion.p>
      </motion.div>
    </motion.div>
  )
}
