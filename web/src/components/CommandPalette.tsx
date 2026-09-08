import { AnimatePresence, motion } from 'framer-motion'
import { useEffect, useRef, useState } from 'react'
import { searchSymbols } from '../lib/api'

const EASE = [0.16, 1, 0.3, 1] as const

export default function CommandPalette({
  open, onClose, onPick,
}: { open: boolean; onClose: () => void; onPick: (s: string) => void }) {
  const [q, setQ] = useState('')
  const [hits, setHits] = useState<{ symbol: string; sector: string }[]>([])
  const [sel, setSel] = useState(0)
  const input = useRef<HTMLInputElement>(null)

  useEffect(() => {
    if (!open) return
    setQ(''); setSel(0)
    // Focus after the entry transition, or the browser scrolls the panel to
    // where it started rather than where it lands.
    const t = setTimeout(() => input.current?.focus(), 90)
    return () => clearTimeout(t)
  }, [open])

  useEffect(() => {
    if (!open) return
    let live = true
    const t = setTimeout(async () => {
      const r = await searchSymbols(q).catch(() => [])
      if (live) { setHits(r); setSel(0) }
    }, 90)
    return () => { live = false; clearTimeout(t) }
  }, [q, open])

  const take = (s?: string) => {
    const sym = (s ?? hits[sel]?.symbol ?? q).trim().toUpperCase()
    if (sym) { onPick(sym); onClose() }
  }

  return (
    <AnimatePresence>
      {open && (
        <>
          <motion.div className="cmd-scrim" onClick={onClose}
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }} />
          <motion.div className="cmd" role="dialog" aria-label="Find a symbol"
            initial={{ opacity: 0, y: -14, scale: 0.985, x: '-50%' }}
            animate={{ opacity: 1, y: 0, scale: 1, x: '-50%' }}
            exit={{ opacity: 0, y: -10, scale: 0.99, x: '-50%' }}
            transition={{ duration: 0.26, ease: EASE }}
            onKeyDown={(e) => {
              if (e.key === 'Escape') onClose()
              else if (e.key === 'Enter') { e.preventDefault(); take() }
              else if (e.key === 'ArrowDown') {
                e.preventDefault(); setSel((s) => (s + 1) % Math.max(1, hits.length))
              } else if (e.key === 'ArrowUp') {
                e.preventDefault(); setSel((s) => (s - 1 + hits.length) % Math.max(1, hits.length))
              }
            }}>
            <div className="cmd-field">
              <svg viewBox="0 0 24 24" width="15" height="15" fill="none"
                stroke="currentColor" strokeWidth="1.8" style={{ color: 'var(--ink-4)' }}>
                <circle cx="11" cy="11" r="7" /><path d="M20 20l-4.2-4.2" />
              </svg>
              <input ref={input} value={q} placeholder="Find a symbol…"
                spellCheck={false} autoComplete="off"
                onChange={(e) => setQ(e.target.value)} />
              <kbd>esc</kbd>
            </div>
            <div className="cmd-list">
              {hits.map((h, i) => (
                <div key={h.symbol} className={i === sel ? 'cmd-row sel' : 'cmd-row'}
                  onMouseDown={(e) => { e.preventDefault(); take(h.symbol) }}
                  onMouseEnter={() => setSel(i)}>
                  <b>{h.symbol}</b><em>{h.sector}</em>
                </div>
              ))}
              {!hits.length && (
                <div className="cmd-row" style={{ cursor: 'default' }}>
                  <span className="micro">No symbol matches that.</span>
                </div>
              )}
            </div>
            <div className="cmd-foot">
              <span><kbd>↑</kbd><kbd>↓</kbd> move</span>
              <span><kbd>↵</kbd> analyse</span>
              <span style={{ marginLeft: 'auto' }}>52 NSE large-caps</span>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  )
}
