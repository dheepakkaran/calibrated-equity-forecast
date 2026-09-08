import { animate, motion, useMotionValue, useReducedMotion, useTransform } from 'framer-motion'
import { useEffect, useRef, useState, type ReactNode } from 'react'

/* ── animated number ─────────────────────────────────────────────────
   A figure that counts up reads as a figure being *measured*, which is
   what these are. Spring-driven rather than linear so it settles the way
   a physical readout would, and skipped entirely when the reader has
   asked for reduced motion.
   ─────────────────────────────────────────────────────────────────── */
export function Num({
  value, decimals = 0, prefix = '', suffix = '', signed = false, delay = 0,
}: {
  value: number; decimals?: number; prefix?: string; suffix?: string
  signed?: boolean; delay?: number
}) {
  const reduce = useReducedMotion()
  const mv = useMotionValue(reduce ? value : 0)
  const text = useTransform(mv, (v) => {
    const s = Math.abs(v).toLocaleString('en-IN', {
      minimumFractionDigits: decimals, maximumFractionDigits: decimals,
    })
    const sign = signed && v > 0 ? '+' : v < 0 ? '−' : ''
    return `${sign}${prefix}${s}${suffix}`
  })

  useEffect(() => {
    if (reduce) { mv.set(value); return }
    const c = animate(mv, value, {
      type: 'spring', stiffness: 60, damping: 18, delay, restDelta: 0.001,
    })
    return () => c.stop()
  }, [value, delay, reduce, mv])

  return <motion.span className="num">{text}</motion.span>
}

/* ── sparkline ───────────────────────────────────────────────────────
   Drawn as a path so it inherits currentColor and needs no palette.
   Deliberately axis-free: it shows shape, not level, and putting numbers
   on it would imply a precision 40 pixels cannot carry.
   ─────────────────────────────────────────────────────────────────── */
export function Spark({
  points, w = 62, h = 18, strokeWidth = 1.2,
}: { points: number[]; w?: number; h?: number; strokeWidth?: number }) {
  if (points.length < 2) return null
  const lo = Math.min(...points), hi = Math.max(...points)
  const span = hi - lo || 1
  const d = points
    .map((p, i) => {
      const x = (i / (points.length - 1)) * (w - 2) + 1
      const y = h - 2 - ((p - lo) / span) * (h - 4)
      return `${i ? 'L' : 'M'}${x.toFixed(1)},${y.toFixed(1)}`
    })
    .join(' ')
  const rising = points[points.length - 1] >= points[0]
  return (
    <svg viewBox={`0 0 ${w} ${h}`} width={w} height={h} aria-hidden
      style={{ opacity: rising ? 0.85 : 0.5, overflow: 'visible' }}>
      <path d={d} fill="none" stroke="currentColor" strokeWidth={strokeWidth}
        strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  )
}

/* ── spotlight ───────────────────────────────────────────────────────
   Writes the cursor position into the registered custom properties the
   .spot class reads. Attached per-card rather than globally so a page
   full of panels does not run one listener per pixel of movement.
   ─────────────────────────────────────────────────────────────────── */
export function useSpotlight<T extends HTMLElement>() {
  const ref = useRef<T>(null)
  useEffect(() => {
    const el = ref.current
    if (!el || window.matchMedia('(prefers-reduced-motion: reduce)').matches) return
    let frame = 0
    const move = (e: PointerEvent) => {
      if (frame) return
      frame = requestAnimationFrame(() => {
        frame = 0
        const r = el.getBoundingClientRect()
        el.style.setProperty('--spot-x', `${((e.clientX - r.left) / r.width) * 100}%`)
        el.style.setProperty('--spot-y', `${((e.clientY - r.top) / r.height) * 100}%`)
      })
    }
    el.addEventListener('pointermove', move)
    return () => { el.removeEventListener('pointermove', move); cancelAnimationFrame(frame) }
  }, [])
  return ref
}

export function Panel({
  children, className = '', ...rest
}: { children: ReactNode; className?: string } & React.HTMLAttributes<HTMLDivElement>) {
  const ref = useSpotlight<HTMLDivElement>()
  return (
    <div ref={ref} className={`card spot ${className}`} {...rest}>{children}</div>
  )
}

/* ── hatched band ────────────────────────────────────────────────────
   The coin-flip zone, drawn rather than described. Without colour this
   is what tells a reader at a glance that a needle is sitting in the
   region where the model has nothing to say.
   ─────────────────────────────────────────────────────────────────── */
export function ConvictionBar({
  proba, threshold, acted,
}: { proba: number; threshold: number; acted: boolean }) {
  // 900x amplification: the model's real output spans roughly 0.46–0.54, so a
  // literal 0–1 axis would render every forecast as the same dot in the middle.
  const pos = Math.min(96, Math.max(4, 50 + (proba - 0.5) * 900))
  const band = threshold * 900
  return (
    <div className="cbar">
      <div className="cbar-track" />
      <div className="cbar-band" style={{ left: `${50 - band}%`, width: `${band * 2}%` }} />
      <motion.div className={acted ? 'cbar-needle' : 'cbar-needle is-in'}
        initial={{ left: '50%' }} animate={{ left: `${pos}%` }}
        transition={{ duration: 0.85, ease: [0.16, 1, 0.3, 1], delay: 0.15 }} />
      <div className="cbar-keys">
        <span>lags the pack</span>
        <span>no useful opinion</span>
        <span>beats the pack</span>
      </div>
    </div>
  )
}

/* ── command palette ─────────────────────────────────────────────────
   ⌘K. Cheap to add and the fastest path to the one thing this app does,
   which is: look at a different symbol.
   ─────────────────────────────────────────────────────────────────── */
export function useHotkey(key: string, fn: () => void) {
  useEffect(() => {
    const h = (e: KeyboardEvent) => {
      if (e.key.toLowerCase() === key && (e.metaKey || e.ctrlKey)) {
        e.preventDefault(); fn()
      }
    }
    window.addEventListener('keydown', h)
    return () => window.removeEventListener('keydown', h)
  }, [key, fn])
}

export function useScrollProgress() {
  const [p, setP] = useState(0)
  useEffect(() => {
    let frame = 0
    const on = () => {
      if (frame) return
      frame = requestAnimationFrame(() => {
        frame = 0
        const h = document.documentElement.scrollHeight - window.innerHeight
        setP(h > 0 ? Math.min(1, window.scrollY / h) : 0)
      })
    }
    on()
    window.addEventListener('scroll', on, { passive: true })
    window.addEventListener('resize', on)
    return () => {
      window.removeEventListener('scroll', on); window.removeEventListener('resize', on)
      cancelAnimationFrame(frame)
    }
  }, [])
  return p
}
