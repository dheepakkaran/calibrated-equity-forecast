import { AnimatePresence, motion } from 'framer-motion'
import { useEffect, useState } from 'react'
import { getTracking } from '../lib/api'
import type { Tracked, TrackingSummary } from '../lib/types'
import './LedgerDrawer.css'

const EASE = [0.22, 0.61, 0.36, 1] as const

const CAT: Record<string, string> = {
  okay: 'bull', okayish: 'warn', 'not okay': 'bear',
}

export default function LedgerDrawer({ open, onClose }: { open: boolean; onClose: () => void }) {
  const [rows, setRows] = useState<Tracked[]>([])
  const [sum, setSum] = useState<TrackingSummary | null>(null)
  const [err, setErr] = useState('')

  useEffect(() => {
    if (!open) return
    getTracking()
      .then((r) => { setRows(r.rows); setSum(r.summary) })
      .catch((e) => setErr((e as Error).message))
  }, [open])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  return (
    <AnimatePresence>
      {open && (
        <>
          <motion.div className="ld-scrim" onClick={onClose}
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
            transition={{ duration: 0.22 }} />
          <motion.aside className="ld"
            initial={{ x: '100%' }} animate={{ x: 0 }} exit={{ x: '100%' }}
            transition={{ duration: 0.4, ease: EASE }}>
            <header className="ld-head">
              <div>
                <div className="eyebrow">Tracked guesses</div>
                <h3>What it said, and what happened</h3>
              </div>
              <button className="ghost" onClick={onClose}>Close <kbd>esc</kbd></button>
            </header>

            {sum && (
              <div className="ld-sum">
                <div className="ld-stat"><b>{sum.tracked}</b><span>tracked</span></div>
                <div className="ld-stat"><b>{sum.pending}</b><span>awaiting</span></div>
                <div className="ld-stat"><b className="bull">{sum.categories.okay ?? 0}</b><span>okay</span></div>
                <div className="ld-stat"><b className="warn">{sum.categories.okayish ?? 0}</b><span>okayish</span></div>
                <div className="ld-stat"><b className="bear">{sum.categories['not okay'] ?? 0}</b><span>not okay</span></div>
                <div className="ld-stat">
                  <b>{sum.direction_accuracy === null ? '—' : `${(sum.direction_accuracy * 100).toFixed(0)}%`}</b>
                  <span>direction, on calls only</span>
                </div>
              </div>
            )}

            <div className="ld-list">
              {err && <div className="micro">{err}</div>}
              {!rows.length && !err && (
                <div className="micro">
                  Nothing tracked yet. Run an analysis and use the button in the
                  corner to fix a guess before the session it describes.
                </div>
              )}
              {rows.map((r, i) => (
                <motion.div key={r.id} className="ld-row"
                  initial={{ opacity: 0, x: 14 }} animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: Math.min(i * 0.03, 0.25), duration: 0.3, ease: EASE }}>
                  <div className="ld-rtop">
                    <b>{r.symbol}</b>
                    <span className="micro">for {r.target_session}</span>
                    {r.outcome
                      ? <span className={`pill ${CAT[r.outcome.category] ?? 'flat'}`}>{r.outcome.category}</span>
                      : <span className="pill flat">{r.status === 'EARLY_READ' ? 'early read' : 'awaiting'}</span>}
                  </div>

                  <div className="ld-guess">
                    Said: <b>{r.guess.acted
                      ? (r.guess.proba_outperform > 0.5 ? 'will out-perform' : 'will lag')
                      : 'no call'}</b>
                    {r.guess.acted && ` at ${(r.guess.confidence * 100).toFixed(1)}%`}
                    {' · '}<span className="micro">{r.guess.regime}</span>
                  </div>

                  {r.early_read && !r.outcome && (
                    <div className="ld-early">
                      Opening gap {r.early_read.relative_gap_pct > 0 ? '+' : ''}
                      {r.early_read.relative_gap_pct}% against the market —
                      leaning <b>{r.early_read.leaning}</b>. {r.early_read.note}
                    </div>
                  )}

                  {r.outcome && (
                    <div className="ld-out">
                      <div className="ld-two">
                        <div>
                          <span className="micro">direction</span>
                          <b className={r.outcome.direction_hit === null ? ''
                            : r.outcome.direction_hit ? 'bull' : 'bear'}>
                            {r.outcome.direction_hit === null ? 'abstained'
                              : r.outcome.direction_hit ? 'right' : 'wrong'}
                          </b>
                        </div>
                        <div>
                          <span className="micro">move vs its typical swing</span>
                          <b>{r.outcome.move_vs_typical_swing.toFixed(2)}×</b>
                        </div>
                        <div>
                          <span className="micro">reward</span>
                          <b className={r.outcome.reward > 0 ? 'bull' : r.outcome.reward < 0 ? 'bear' : ''}>
                            {r.outcome.reward > 0 ? '+' : ''}{r.outcome.reward.toFixed(3)}
                          </b>
                        </div>
                      </div>
                      <p className="ld-why">{r.outcome.why}</p>
                    </div>
                  )}
                </motion.div>
              ))}
            </div>

            {sum && <div className="ld-note micro">{sum.note}</div>}
          </motion.aside>
        </>
      )}
    </AnimatePresence>
  )
}
