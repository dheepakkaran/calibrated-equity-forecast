import { AnimatePresence, motion } from 'framer-motion'
import { useState } from 'react'
import { trackSymbol } from '../lib/api'
import type { Forecast } from '../lib/types'
import './TrackButton.css'

const EASE = [0.22, 0.61, 0.36, 1] as const

const REPO = 'dheepakkaran/calibrated-equity-forecast'

/** Where the button goes when there is no backend to POST to.
 *
 *  On a static host the tracking write has to happen in CI, and a token with
 *  write access cannot be shipped to a web page. So the button opens a
 *  pre-filled issue instead: an Action reads the title, runs the forecast
 *  server-side and appends to the ledger. The only credential involved is the
 *  Actions token, which never leaves CI.
 */
function issueUrl(symbol: string, target: string) {
  const body = [
    `Track \`${symbol}\` for the session on ${target}.`,
    '',
    'Opened from the interface. An Action will run the forecast, append the',
    'guess to `tracking/predictions.json`, and close this issue.',
  ].join('\n')
  return `https://github.com/${REPO}/issues/new?title=${
    encodeURIComponent(`track: ${symbol}`)}&body=${encodeURIComponent(body)}`
}

export default function TrackButton({ forecast }: { forecast: Forecast }) {
  const [state, setState] = useState<'idle' | 'busy' | 'done' | 'already' | 'issue' | 'error'>('idle')
  const [msg, setMsg] = useState('')

  const go = async () => {
    setState('busy')
    try {
      const r = await trackSymbol(forecast.symbol)
      setState(r.tracked.already_tracked ? 'already' : 'done')
      setMsg(`${r.summary.tracked} tracked · ${r.summary.pending} awaiting tomorrow`)
    } catch (e) {
      // No backend reachable — almost certainly the static deploy. Hand off to
      // the issue-triggered path rather than reporting a failure the reader
      // cannot do anything about.
      const url = issueUrl(forecast.symbol, forecast.target_date)
      const win = window.open(url, '_blank', 'noopener')
      if (win) {
        setState('issue')
        setMsg((e as Error).message)
      } else {
        setState('error')
        setMsg((e as Error).message)
      }
    }
  }

  return (
    <motion.div className="track"
      initial={{ opacity: 0, y: 22 }} animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.5, duration: 0.45, ease: EASE }}>
      <AnimatePresence mode="wait">
        {state === 'idle' && (
          <motion.button key="idle" className="track-btn" onClick={go}
            whileHover={{ y: -1 }} whileTap={{ scale: 0.98 }}
            exit={{ opacity: 0, scale: 0.96 }} transition={{ duration: 0.18 }}>
            <span className="track-dot" />
            Track this for tomorrow morning?
          </motion.button>
        )}
        {state === 'busy' && (
          <motion.div key="busy" className="track-btn is-busy"
            initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
            <motion.span className="track-spin"
              animate={{ rotate: 360 }}
              transition={{ duration: 0.9, repeat: Infinity, ease: 'linear' }} />
            Saving the guess…
          </motion.div>
        )}
        {(state === 'done' || state === 'already' || state === 'issue') && (
          <motion.div key="done" className="track-btn is-done"
            initial={{ opacity: 0, scale: 0.96 }} animate={{ opacity: 1, scale: 1 }}
            transition={{ type: 'spring', stiffness: 380, damping: 26 }}>
            <svg viewBox="0 0 16 16" width="13" height="13">
              <path d="M3 8.4 6.2 11.6 13 4.8" fill="none" stroke="currentColor"
                strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
            <span>
              {state === 'already' ? 'Already tracked' : 'Locked in'} for {forecast.target_date}
              <em>{msg}</em>
            </span>
          </motion.div>
        )}
        {state === 'issue' && (
          <motion.a key="issue" className="track-btn is-done"
            href={issueUrl(forecast.symbol, forecast.target_date)}
            target="_blank" rel="noopener noreferrer"
            initial={{ opacity: 0, scale: 0.96 }} animate={{ opacity: 1, scale: 1 }}
            transition={{ type: 'spring', stiffness: 380, damping: 26 }}>
            <span>Opened a tracking request<em>finish it on GitHub — an Action records the guess</em></span>
          </motion.a>
        )}
        {state === 'error' && (
          <motion.div key="err" className="track-btn is-error"
            initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
            Could not save — {msg}
          </motion.div>
        )}
      </AnimatePresence>

      {(state === 'done' || state === 'already' || state === 'issue') && (
        <motion.div className="track-note"
          initial={{ opacity: 0, y: -6 }} animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.15 }}>
          The guess is fixed now, before the outcome exists. At 09:20 IST tomorrow
          a job reads the opening gap for a provisional look; after the 15:30 close
          it grades the call properly and writes the result back.
        </motion.div>
      )}
    </motion.div>
  )
}
