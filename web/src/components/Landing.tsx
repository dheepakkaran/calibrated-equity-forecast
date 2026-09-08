import { AnimatePresence, motion } from 'framer-motion'
import { useEffect, useRef, useState } from 'react'
import { searchSymbols } from '../lib/api'
import './Landing.css'

type Props = { onRun: (symbol: string) => void; onOpenLedger: () => void }

export default function Landing({ onRun, onOpenLedger }: Props) {
  const [q, setQ] = useState('')
  const [hits, setHits] = useState<{ symbol: string; sector: string }[]>([])
  const [sel, setSel] = useState(0)
  const input = useRef<HTMLInputElement>(null)

  useEffect(() => { input.current?.focus() }, [])

  useEffect(() => {
    // Only once something has been typed. Opening the list on mount covers the
    // paragraph that explains what this system is and what its accuracy
    // actually is, which is the one thing a first-time reader should see.
    if (!q.trim()) { setHits([]); return }
    // Debounced, and the result of a stale keystroke is discarded rather than
    // allowed to overwrite a newer one.
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
    <motion.div
      className="landing"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0, y: -18 }}
      transition={{ duration: 0.4, ease: [0.22, 0.61, 0.36, 1] }}
    >
      <button className="ledger-icon" onClick={onOpenLedger} title="Tracked guesses and their outcomes">
        <svg viewBox="0 0 24 24" width="17" height="17" fill="none"
             stroke="currentColor" strokeWidth="1.7" strokeLinecap="round">
          <path d="M3 12a9 9 0 1 0 3-6.7" /><path d="M3 4v4h4" /><path d="M12 8v4l3 2" />
        </svg>
        <span>Tracked</span>
      </button>

      <div className="landing-inner">
        <motion.p className="eyebrow"
          initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.05 }}>
          Research system · educational use only
        </motion.p>

        <motion.h1
          initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1, duration: 0.5, ease: [0.22, 0.61, 0.36, 1] }}>
          Which share should I look at?
        </motion.h1>

        <motion.div className="field"
          initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.18, duration: 0.5, ease: [0.22, 0.61, 0.36, 1] }}>
          <input
            ref={input}
            value={q}
            placeholder="HINDZINC"
            spellCheck={false}
            autoComplete="off"
            onChange={(e) => setQ(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') { e.preventDefault(); go() }
              else if (e.key === 'ArrowDown') { e.preventDefault(); setSel((s) => (s + 1) % Math.max(1, hits.length)) }
              else if (e.key === 'ArrowUp') { e.preventDefault(); setSel((s) => (s - 1 + hits.length) % Math.max(1, hits.length)) }
            }}
          />
          <AnimatePresence>
            {hits.length > 0 && (
              <motion.div className="typeahead"
                initial={{ opacity: 0, y: -6 }} animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -6 }} transition={{ duration: 0.16 }}>
                {hits.map((h, i) => (
                  <div key={h.symbol}
                    className={i === sel ? 'ta-row sel' : 'ta-row'}
                    onMouseDown={(e) => { e.preventDefault(); go(h.symbol) }}
                    onMouseEnter={() => setSel(i)}>
                    <b>{h.symbol}</b><em>{h.sector}</em>
                  </div>
                ))}
              </motion.div>
            )}
          </AnimatePresence>
        </motion.div>

        <motion.p className="hint"
          initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.3 }}>
          52 NSE large-caps. <kbd>↑</kbd><kbd>↓</kbd> to pick, <kbd>↵</kbd> to run the analysis.
        </motion.p>

        <motion.p className="landing-foot"
          initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.42 }}>
          One question, asked honestly: will this share out-perform or lag the
          median NSE large-cap in the next session? On held-out data the model is
          right <b>51.0%</b> of the time against a <b>50.0%</b> coin flip — a real
          edge, and a small one. Most days it has nothing useful to say, and on
          those days it says so.
        </motion.p>
      </div>
    </motion.div>
  )
}
