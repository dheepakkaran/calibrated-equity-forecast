import { useMemo, useState } from 'react'
import { motion } from 'framer-motion'
import type { Payload } from '../lib/types'

/** Hand-drawn SVG rather than a charting library.
 *
 *  One plot does not justify the weight of Recharts, and drawing it here means
 *  the axis range comes from the data instead of a library rounding to numbers
 *  it finds tidy. Reference-level labels run a collision pass: pivots are
 *  computed from a single session, so they sit within a rupee or two of each
 *  other while the vertical axis spans a year, and stacked at one x they render
 *  as an illegible smudge.
 */
export default function PriceChart({
  price, levels,
}: { price: Payload['price']; levels: Record<string, number> }) {
  const [hover, setHover] = useState<number | null>(null)
  const W = 860, H = 280, PAD = 40, RIGHT = 52

  const geom = useMemo(() => {
    const pts = price.points
    if (!pts.length) return null
    const lv = [
      { y: levels.r1, label: `${Math.round(levels.r1)} R1`, color: '#c97a82' },
      { y: levels.pivot, label: `${Math.round(levels.pivot)} pivot`, color: '#d9a441' },
      { y: levels.s1, label: `${Math.round(levels.s1)} S1`, color: '#5fa8a0' },
    ].filter((l) => Number.isFinite(l.y))

    const ys = pts.map((p) => p.close)
    let lo = Math.min(...ys, ...lv.map((l) => l.y))
    let hi = Math.max(...ys, ...lv.map((l) => l.y))
    const pad = (hi - lo) * 0.08 || 1
    lo -= pad; hi += pad

    const sx = (i: number) => PAD + (i / Math.max(1, pts.length - 1)) * (W - PAD - RIGHT)
    const sy = (v: number) => H - 28 - ((v - lo) / (hi - lo)) * (H - 46)
    const path = pts.map((p, i) => `${i ? 'L' : 'M'}${sx(i).toFixed(1)},${sy(p.close).toFixed(1)}`).join(' ')

    const placed: { slot: number; y: number }[] = []
    const labelled = [...lv].sort((a, b) => a.y - b.y).map((l) => {
      const y = sy(l.y)
      let slot = 0
      while (placed.some((q) => q.slot === slot && Math.abs(q.y - y) < 12)) slot += 1
      placed.push({ slot, y })
      return { ...l, y, x: PAD + 4 + slot * 66 }
    })

    const idx = new Map(pts.map((p, i) => [p.date, i]))
    const marks = price.markers
      .map((m) => ({ ...m, i: idx.get(m.date) }))
      .filter((m) => m.i !== undefined)
      .map((m) => ({ ...m, cx: sx(m.i as number), cy: sy(m.close) }))

    const grid = [0, 0.25, 0.5, 0.75, 1].map((fr) => {
      const v = lo + fr * (hi - lo)
      return { y: sy(v), v }
    })
    return { pts, path, labelled, marks, grid, sx, sy, lo, hi }
  }, [price, levels])

  if (!geom) return <div className="micro">No price history.</div>
  const hovered = hover !== null ? geom.marks[hover] : null

  return (
    <div style={{ position: 'relative' }}>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: 'auto', display: 'block' }}>
        {geom.grid.map((g, i) => (
          <g key={i}>
            <line x1={PAD} y1={g.y} x2={W - RIGHT} y2={g.y} stroke="#24323d" strokeWidth="1" />
            <text x={W - RIGHT + 6} y={g.y + 3.5} fill="#647889" fontSize="9.5">
              {Math.round(g.v).toLocaleString('en-IN')}
            </text>
          </g>
        ))}
        {geom.labelled.map((l) => (
          <g key={l.label}>
            <line x1={PAD} y1={l.y} x2={W - RIGHT} y2={l.y}
              stroke={l.color} strokeWidth="1" strokeDasharray="4 4" opacity="0.75" />
            <text x={l.x} y={l.y - 5} fill={l.color} fontSize="9.5">{l.label}</text>
          </g>
        ))}
        <motion.path d={geom.path} fill="none" stroke="#9cafc1" strokeWidth="1.6"
          initial={{ pathLength: 0 }} animate={{ pathLength: 1 }}
          transition={{ duration: 1.1, ease: [0.22, 0.61, 0.36, 1] }} />
        {geom.marks.map((m, i) => (
          <motion.circle key={`${m.date}-${i}`} cx={m.cx} cy={m.cy} r={hover === i ? 6 : 4.5}
            fill="#131c25" strokeWidth="2"
            stroke={m.kind === 'announcement' ? '#7b95c4' : m.kind === 'driver' ? '#5fa8a0' : '#647889'}
            style={{ cursor: 'pointer' }}
            initial={{ opacity: 0, scale: 0.4 }} animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.7 + i * 0.03, type: 'spring', stiffness: 380, damping: 22 }}
            onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(null)} />
        ))}
        <text x={PAD} y={H - 6} fill="#647889" fontSize="9.5">{geom.pts[0].date}</text>
        <text x={W - RIGHT} y={H - 6} fill="#647889" fontSize="9.5" textAnchor="end">
          {geom.pts[geom.pts.length - 1].date}
        </text>
      </svg>

      {hovered && (
        <motion.div className="pc-tip"
          initial={{ opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.16 }}>
          <div className="pc-tip-top">
            <b>{hovered.date}</b>
            <span className={hovered.ret_pct > 0 ? 'up' : 'dn'}>
              {hovered.ret_pct > 0 ? '+' : ''}{hovered.ret_pct}%
            </span>
            <span className="micro">{hovered.sigma}σ</span>
          </div>
          <p>{hovered.kind === 'unexplained'
            ? 'No filing above the materiality threshold, and no mapped driver moved consistently — left unexplained.'
            : hovered.headline}</p>
        </motion.div>
      )}

      <div className="pc-key">
        <span><i style={{ background: '#7b95c4' }} />matched to a filing</span>
        <span><i style={{ background: '#5fa8a0' }} />matched to a driver</span>
        <span><i style={{ background: '#647889' }} />unexplained</span>
      </div>
    </div>
  )
}
