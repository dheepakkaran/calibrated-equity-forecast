import { AnimatePresence, motion } from 'framer-motion'
import { useEffect, useRef } from 'react'
import type { StageMeta } from '../lib/types'
import './RunView.css'

export type RunStage = StageMeta & {
  status: 'waiting' | 'running' | 'done' | 'failed'
  summary?: string
  ms?: number
}

type Props = { symbol: string; stages: RunStage[]; pct: number; error?: string }

const EASE = [0.22, 0.61, 0.36, 1] as const

export default function RunView({ symbol, stages, pct, error }: Props) {
  const feed = useRef<HTMLDivElement>(null)
  const spoken = stages.filter((s) => s.summary)

  useEffect(() => {
    // Keep the newest sentence in view. `smooth` on every append fights itself
    // when several stages land in the same tick, so the scroll is instant and
    // the motion comes from the entering card instead.
    feed.current?.scrollTo({ top: feed.current.scrollHeight })
  }, [spoken.length])

  return (
    <motion.div className="run"
      initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
      transition={{ duration: 0.35, ease: EASE }}>

      <div className="run-top">
        <div className="run-sym">
          <span className="eyebrow">Analysing</span>
          <h2>{symbol}</h2>
        </div>
        <div className="run-pct">
          <span>{pct}%</span>
          <div className="run-track">
            <motion.div className="run-fill"
              animate={{ width: `${pct}%` }}
              transition={{ duration: 0.5, ease: EASE }} />
          </div>
        </div>
      </div>

      <div className="run-split">
        {/* ── left: the operations ────────────────────────────────── */}
        <div className="run-ops">
          <div className="run-colhead">Operations</div>
          <div className="ops-list">
            {stages.map((s, i) => (
              <motion.div key={s.key}
                className={`op op-${s.status}`}
                initial={{ opacity: 0, x: -12 }}
                animate={{ opacity: s.status === 'waiting' ? 0.4 : 1, x: 0 }}
                transition={{ delay: Math.min(i * 0.035, 0.3), duration: 0.3, ease: EASE }}>
                <div className="op-mark">
                  {s.status === 'running' && (
                    <motion.span className="op-spin"
                      animate={{ rotate: 360 }}
                      transition={{ duration: 0.9, repeat: Infinity, ease: 'linear' }} />
                  )}
                  {s.status === 'done' && (
                    <motion.svg viewBox="0 0 16 16" width="13" height="13"
                      initial={{ scale: 0.4, opacity: 0 }} animate={{ scale: 1, opacity: 1 }}
                      transition={{ type: 'spring', stiffness: 420, damping: 24 }}>
                      <path d="M3 8.4 6.2 11.6 13 4.8" fill="none" stroke="currentColor"
                        strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                    </motion.svg>
                  )}
                  {s.status === 'failed' && <span className="op-x">×</span>}
                  {s.status === 'waiting' && <span className="op-dot" />}
                </div>
                <div className="op-body">
                  <div className="op-label">{s.label}</div>
                  <div className="op-doing">{s.doing}</div>
                </div>
                {s.ms !== undefined && (
                  <motion.div className="op-ms"
                    initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                    {s.ms < 1000 ? `${s.ms} ms` : `${(s.ms / 1000).toFixed(1)} s`}
                  </motion.div>
                )}
              </motion.div>
            ))}
          </div>
        </div>

        {/* ── right: what it means ────────────────────────────────── */}
        <div className="run-feed" ref={feed}>
          <div className="run-colhead">What that means</div>
          <AnimatePresence initial={false}>
            {spoken.map((s) => (
              <motion.div key={s.key}
                className={s.status === 'failed' ? 'said said-failed' : 'said'}
                initial={{ opacity: 0, y: 14 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.42, ease: EASE }}>
                <div className="said-key">{s.label}</div>
                <p>{s.summary}</p>
              </motion.div>
            ))}
          </AnimatePresence>
          {!spoken.length && !error && (
            <div className="said-empty">
              Each step will explain itself here as it finishes.
            </div>
          )}
          {error && (
            <motion.div className="said said-failed"
              initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>
              <div className="said-key">Stopped</div>
              <p>{error}</p>
            </motion.div>
          )}
        </div>
      </div>
    </motion.div>
  )
}
