export type StageMeta = { key: string; label: string; doing: string }

export type StageEvent = {
  type: 'stage'
  key: string
  status: 'running' | 'done' | 'failed'
  label: string
  doing?: string
  summary?: string
  ms?: number
  pct: number
}

export type StreamEvent =
  | { type: 'start'; symbol: string; stages: StageMeta[] }
  | StageEvent
  | { type: 'final'; elapsed_ms: number; payload: Payload }
  | { type: 'error'; key?: string; message: string }

export type Aspect = {
  aspect: string
  points: number
  direction: string
  contribution: number
  share_of_signal: number
  evidence: { feature: string; shap: number; value?: number }[]
}

export type Forecast = {
  symbol: string
  sector: string
  as_of: string
  target_date: string
  prediction_id: string
  direction: string
  proba_outperform: number
  confidence: number
  conviction: number
  acted: boolean
  abstain_threshold: number
  last_close: number
  typical_daily_range: number
  regime: { label: string; volatility: string; trend: string }
  aspects: Aspect[]
  arms: { probas: Record<string, number>; weights: Record<string, number>; mode: string }
  drivers: string[]
  track_record: {
    accuracy: number; coin_flip_baseline: number; edge_pp: number
    fold_t_stat: number; folds_won: string; auc: number
    ece_pooled_pp: number; window: string; accuracy_at_10pct_coverage: number
  }
  model_version: string
  disclaimer: string
}

export type SimpleView = {
  headline: string
  opening: string
  confidence_line: string
  reasons: { aspect: string; points: number; title: string; body: string; source?: string }[]
  explainer: { question: string; paragraphs: string[] }
  levels: { label: string; price: number; meaning: string }[]
  closing: string
  provider: string
  model: string
  cached?: boolean
  unverified_numbers: string[]
  evidence?: any
}

export type Move = {
  date: string; ret: number; ret_rel: number; sigma: number
  kind: string; headline: string | null; source_url: string | null
  confidence: number; rationale: string; band: string; market_wide: number
}

export type BoardTile = {
  key: string; label: string; value: number; spark: number[]; chg_pct: number
  chg_week_pct: number; as_of: string; lag_sessions: number; availability: string
}

export type Payload = {
  forecast: Forecast
  simple: SimpleView | null
  levels: Record<string, number | string>
  drivers: { driver: string; corr_120d: number | null; last_chg_pct: number | null; lag_sessions: number }[]
  attribution: Move[]
  coverage: Record<string, number | null>
  board: BoardTile[]
  price: { points: { date: string; close: number }[]; markers: any[]; sessions: number }
  not_built: { panel: string; reason: string }[]
}

export type Tracked = {
  id: string; symbol: string; sector: string; tracked_at: string
  as_of_session: string; target_session: string; status: string
  guess: { direction: string; proba_outperform: number; confidence: number
           conviction: number; acted: boolean; regime: string; last_close: number }
  early_read: { gap_pct: number; relative_gap_pct: number; leaning: string; note: string } | null
  outcome: {
    category: string; why: string; direction_hit: boolean | null
    move_vs_typical_swing: number; relative_move_pct: number; reward: number
    actual: string; close: number
  } | null
}

export type TrackingSummary = {
  tracked: number; pending: number; resolved: number
  categories: Record<string, number>
  with_a_call: number; abstained: number
  direction_accuracy: number | null; mean_reward: number | null
  note: string
}
