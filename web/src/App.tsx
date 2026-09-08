import { AnimatePresence } from 'framer-motion'
import { useCallback, useRef, useState } from 'react'
import Landing from './components/Landing'
import LedgerDrawer from './components/LedgerDrawer'
import Result from './components/Result'
import RunView, { type RunStage } from './components/RunView'
import { streamAnalysis } from './lib/api'
import type { Payload, StreamEvent } from './lib/types'

type Phase = 'landing' | 'running' | 'result'

export default function App() {
  const [phase, setPhase] = useState<Phase>('landing')
  const [symbol, setSymbol] = useState('')
  const [stages, setStages] = useState<RunStage[]>([])
  const [pct, setPct] = useState(0)
  const [error, setError] = useState('')
  const [data, setData] = useState<Payload | null>(null)
  const [elapsed, setElapsed] = useState(0)
  const [ledger, setLedger] = useState(false)
  const cancel = useRef<(() => void) | null>(null)

  const run = useCallback((sym: string) => {
    cancel.current?.()
    setSymbol(sym); setStages([]); setPct(0); setError(''); setData(null)
    setPhase('running')

    cancel.current = streamAnalysis(sym, (e: StreamEvent) => {
      if (e.type === 'start') {
        setStages(e.stages.map((s) => ({ ...s, status: 'waiting' })))
      } else if (e.type === 'stage') {
        setPct(e.pct)
        setStages((prev) => prev.map((s) =>
          s.key === e.key
            ? { ...s, status: e.status, summary: e.summary ?? s.summary, ms: e.ms ?? s.ms }
            : s))
      } else if (e.type === 'final') {
        setElapsed(e.elapsed_ms)
        setData(e.payload)
        setPct(100)
        // A beat before the swap, so the last stage is seen to finish rather
        // than being yanked off screen the instant it lands.
        setTimeout(() => setPhase('result'), 620)
      } else if (e.type === 'error') {
        setError(e.message)
      }
    })
  }, [])

  const restart = useCallback(() => {
    cancel.current?.()
    setPhase('landing'); setData(null); setStages([]); setPct(0); setError('')
  }, [])

  return (
    <>
      <AnimatePresence mode="wait">
        {phase === 'landing' && (
          <Landing key="landing" onRun={run} onOpenLedger={() => setLedger(true)} />
        )}
        {phase === 'running' && (
          <RunView key="running" symbol={symbol} stages={stages} pct={pct} error={error} />
        )}
        {phase === 'result' && data && (
          <Result key="result" data={data} elapsedMs={elapsed}
            onRestart={restart} onOpenLedger={() => setLedger(true)} />
        )}
      </AnimatePresence>
      <LedgerDrawer open={ledger} onClose={() => setLedger(false)} />
    </>
  )
}
