import { AnimatePresence, motion } from 'framer-motion'
import { useState } from 'react'
import type { Payload } from '../lib/types'
import Analysis from './tabs/Analysis'
import Overview from './tabs/Overview'
import Sources from './tabs/Sources'
import TrackButton from './TrackButton'
import './Result.css'

const TABS = [
  { key: 'overview', label: 'Overview', sub: 'the answer, in plain English' },
  { key: 'analysis', label: 'Analysis', sub: 'markets, history, and why' },
  { key: 'sources', label: 'Sources', sub: 'where every claim came from' },
] as const

type TabKey = (typeof TABS)[number]['key']
const EASE = [0.22, 0.61, 0.36, 1] as const

type Props = {
  data: Payload
  elapsedMs: number
  onRestart: () => void
  onOpenLedger: () => void
}

export default function Result({ data, elapsedMs, onRestart, onOpenLedger }: Props) {
  const [tab, setTab] = useState<TabKey>('overview')
  const f = data.forecast

  return (
    <motion.div className="result"
      initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.45, ease: EASE }}>

      <header className="res-head">
        <div className="res-id">
          <h2>{f.symbol}</h2>
          <span className="res-meta">
            {f.sector} · closed ₹{f.last_close.toLocaleString('en-IN')} on {f.as_of}
            {' · '}forecast for {f.target_date}
          </span>
        </div>
        <div className="res-actions">
          <button className="ghost" onClick={onOpenLedger}>Tracked</button>
          <button className="ghost" onClick={onRestart}>Another share</button>
        </div>
      </header>

      <nav className="res-tabs">
        {TABS.map((t) => (
          <button key={t.key}
            className={t.key === tab ? 'res-tab on' : 'res-tab'}
            onClick={() => setTab(t.key)}>
            <span className="rt-label">{t.label}</span>
            <span className="rt-sub">{t.sub}</span>
            {t.key === tab && (
              <motion.div className="rt-underline" layoutId="rt-underline"
                transition={{ duration: 0.32, ease: EASE }} />
            )}
          </button>
        ))}
        <div className="res-elapsed">analysed in {(elapsedMs / 1000).toFixed(1)}s</div>
      </nav>

      <div className="res-body">
        <AnimatePresence mode="wait">
          <motion.div key={tab}
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ duration: 0.28, ease: EASE }}>
            {tab === 'overview' && <Overview data={data} />}
            {tab === 'analysis' && <Analysis data={data} />}
            {tab === 'sources' && <Sources data={data} />}
          </motion.div>
        </AnimatePresence>
      </div>

      <TrackButton forecast={f} />
    </motion.div>
  )
}
