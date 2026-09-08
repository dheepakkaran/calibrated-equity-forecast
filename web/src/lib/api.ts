import type { Payload, StreamEvent, Tracked, TrackingSummary } from './types'

/** Streams the analysis run. Returns a cancel function.
 *
 *  fetch + a ReadableStream reader rather than EventSource, because
 *  EventSource cannot be cancelled cleanly mid-flight and gives no access to
 *  an HTTP error body — both of which matter when a run takes half a minute
 *  and the user may type a different symbol partway through.
 */
export function streamAnalysis(
  symbol: string,
  onEvent: (e: StreamEvent) => void,
): () => void {
  const ctrl = new AbortController()
  void (async () => {
    try {
      const res = await fetch(`/api/analyse/${encodeURIComponent(symbol)}`, {
        signal: ctrl.signal,
        headers: { Accept: 'text/event-stream' },
      })
      if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`)
      const reader = res.body.getReader()
      const dec = new TextDecoder()
      let buf = ''
      for (;;) {
        const { done, value } = await reader.read()
        if (done) break
        buf += dec.decode(value, { stream: true })
        const parts = buf.split('\n\n')
        buf = parts.pop() ?? ''
        for (const part of parts) {
          const line = part.split('\n').find((l) => l.startsWith('data: '))
          if (!line) continue
          try {
            onEvent(JSON.parse(line.slice(6)) as StreamEvent)
          } catch {
            /* a partial frame; the next read completes it */
          }
        }
      }
    } catch (err) {
      if ((err as Error).name === 'AbortError') return
      onEvent({ type: 'error', message: (err as Error).message })
    }
  })()
  return () => ctrl.abort()
}

export async function searchSymbols(q: string) {
  const r = await fetch(`/api/symbols/search?q=${encodeURIComponent(q)}&limit=8`)
  return (await r.json()) as { symbol: string; sector: string }[]
}

export async function trackSymbol(symbol: string, note = '') {
  const r = await fetch('/api/track', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ symbol, note }),
  })
  if (!r.ok) throw new Error(((await r.json()) as { detail?: string }).detail ?? 'could not track')
  return (await r.json()) as {
    tracked: Tracked & { already_tracked: boolean }
    summary: TrackingSummary
  }
}

export async function getTracking() {
  const r = await fetch('/api/tracking?limit=100')
  return (await r.json()) as { summary: TrackingSummary; rows: Tracked[] }
}

export type { Payload }
