/**
 * API response types.
 *
 * The backend wraps every payload in an availability envelope so the UI can
 * distinguish "no model trained yet" from "request failed" - the two need very
 * different empty states.
 */

export interface ApiEnvelope<T> {
  available: boolean;
  demo_mode: boolean;
  detail?: string;
  detail_fa?: string;
  data: T | null;
  [key: string]: unknown;
}

export type Level = 'listing' | 'destination' | 'category' | 'market';

export interface HealthResponse {
  status: string;
  demo_mode: boolean;
  sync_tasks: boolean;
  default_horizon: number;
  primary_metric: string;
  training_profile: string;
  has_trained_model: boolean;
  latest_run: string | null;
}

export interface ConfidenceScore {
  score: number;
  label: 'high' | 'medium' | 'low';
  interval_penalty: number;
  accuracy_penalty: number;
  horizon_penalty: number;
  history_penalty: number;
  formula: string;
}

export interface EntityInsight {
  entity_id: string;
  label: string;
  forecast_total: number;
  forecast_mean: number;
  recent_total: number | null;
  change_pct: number | null;
  lower_total: number;
  upper_total: number;
  relative_interval_width: number | null;
  periods: number;
}

export interface DashboardSummary {
  horizon: number;
  level: Level;
  forecast_total: number;
  forecast_lower: number;
  forecast_upper: number;
  relative_interval_width: number | null;
  previous_period_total: number | null;
  change_pct: number | null;
  top_growth: EntityInsight | null;
  confidence: ConfidenceScore;
  model: {
    champion: string | null;
    primary_metric: string | null;
    primary_value: number | null;
    baseline_model: string | null;
    baseline_value: number | null;
    improvement_pct: number | null;
    last_trained: string | null;
    training_window: { start: string; end: string } | null;
    uncertainty_method: string | null;
    n_features: number | null;
  };
  anomalies: { total: number; important: number; spikes?: number; drops?: number; high_severity?: number };
  data_quality: { health_score: number | null; grade: string | null };
  run_id: string;
  levels: Level[];
  available_horizons: number[];
  insights: {
    decision_opportunities: Opportunity[];
    fastest_growing: EntityInsight | null;
    fastest_declining: EntityInsight | null;
    highest_uncertainty: EntityInsight | null;
  };
}

export interface Opportunity {
  kind: 'growth' | 'decline' | 'peak' | 'uncertainty';
  entity_id: string;
  label: string;
  headline_fa: string;
  change_pct: number | null;
  evidence_fa: string;
  uncertainty?: number | null;
  confidence?: string;
}

export interface SeriesPoint {
  ds: string;
  actual?: number;
  backtest?: number;
  backtest_lower?: number;
  backtest_upper?: number;
  forecast?: number;
  lower?: number;
  upper?: number;
}

export interface TimeseriesResponse {
  level: Level;
  entity_id: string | null;
  label: string;
  horizon: number;
  forecast_start: string | null;
  series: SeriesPoint[];
  totals: { forecast: number; lower: number; upper: number; history_mean: number | null };
  members: Member[];
  levels: Level[];
}

export interface Member {
  id: string;
  label: string;
  forecast_total: number;
}

export interface DriverGroup {
  group: string;
  label_fa: string;
  contribution_share: number;
  contribution_score: number;
  signed_effect: number;
  direction: 'positive' | 'negative' | 'neutral';
}

export interface DriversResponse {
  method: string;
  base_value: number;
  groups: DriverGroup[];
  positive: DriverGroup[];
  negative: DriverGroup[];
  features: { feature: string; importance: number; direction: string | null; group?: string }[];
  note: string;
  price_dependence: { feature: string; curve: DependencePoint[] }[];
}

export interface DependencePoint {
  bin: number;
  feature_value: number;
  feature_min: number;
  feature_max: number;
  mean_contribution: number;
  n: number;
}

export interface OverviewRow {
  entity_id: string;
  label: string;
  current_demand: number | null;
  forecast_total: number;
  lower_total: number;
  upper_total: number;
  change_pct: number | null;
  relative_interval_width: number | null;
  confidence: string;
  status: string;
  status_fa: string;
}

export interface PeakPeriod {
  entity_id: string;
  label: string;
  type: 'peak' | 'trough';
  start: string;
  end: string;
  days: number;
  expected_change_pct: number;
  baseline_level: number;
  forecast_level: number;
  total_forecast: number;
  confidence: string;
  level?: string;
}

export interface Anomaly {
  entity_id: string;
  label: string;
  ds: string;
  score: number;
  type: 'spike' | 'drop';
  type_fa: string;
  severity: 'high' | 'medium' | 'low';
  deviation: number | null;
  source: string;
  actual?: number | null;
  expected?: number | null;
  forecast?: number | null;
}

export interface LeaderboardRow {
  rank: number;
  model: string;
  is_baseline: boolean;
  is_champion: boolean;
  primary_metric: string;
  primary_value: number | null;
  metrics: Record<string, number | null>;
  by_horizon: HorizonScore[];
  improvement_vs_baseline?: number | null;
  baseline_model?: string;
  ensemble_weight?: number | null;
  fit_seconds?: number | null;
}

export interface HorizonScore {
  bucket: string;
  horizon_min: number;
  horizon_max: number;
  n: number;
  [metric: string]: number | string;
}

export interface LeaderboardResponse {
  leaderboard: LeaderboardRow[];
  champion: string;
  primary_metric: string;
  horizon_buckets: string[];
  folds: Fold[];
}

export interface Fold {
  fold: number;
  train_end: string;
  valid_start: string;
  valid_end: string;
}

export interface ModelInfo {
  name: string;
  label: string;
  kind: string;
  optional: boolean;
  available: boolean;
  status: string;
}

export interface TuningRun {
  model: string;
  best_params: Record<string, number | string>;
  best_score: number | null;
  default_score: number | null;
  n_trials: number;
  seconds: number;
  improved: boolean;
  note: string;
}

export interface CensoringReport {
  enabled: boolean;
  reason: string;
  capacity_column: string | null;
  censored_share: number;
  censored_periods: number;
  affected_entities: number;
  mean_uplift: number;
  method: string;
}

export interface ModelsResponse {
  registry: ModelInfo[];
  trained: {
    run_id: string;
    champion: string;
    primary_metric: string;
    ensemble_weights: Record<string, number>;
    last_trained: string;
    profile: string;
    horizon: number;
    frequency: string;
    n_features: number;
    training_window: { start: string; end: string };
    dataset: string;
    target: string;
    uncertainty_method: string;
    environment: Record<string, unknown>;
    warnings: string[];
    tuning: TuningRun[];
    censoring: CensoringReport | null;
  } | null;
}

export interface BacktestMetrics {
  champion: string;
  primary_metric: string;
  folds: Fold[];
  fold_scores: Record<string, number | string>[];
  by_horizon: HorizonScore[];
  segments: {
    by_destination?: SegmentScore[];
    by_weekday?: SegmentScore[];
    by_month?: SegmentScore[];
    by_horizon?: SegmentScore[];
  };
  intervals: Record<string, number | null>;
  coverage_by_horizon: CoverageRow[];
  uncertainty_method: string;
}

export interface SegmentScore {
  key: string | number;
  n: number;
  [metric: string]: number | string | null;
}

export interface CoverageRow {
  bucket: string;
  horizon_min: number;
  horizon_max: number;
  n: number;
  nominal_coverage: number;
  observed_coverage: number;
  mean_width: number;
}

export interface BacktestSeries {
  series: { ds: string; actual: number; prediction: number; lower: number | null; upper: number | null }[];
  rows: Record<string, unknown>[];
  count: number;
  level: Level;
}

export interface ScenarioAdjustable {
  column: string;
  kind: string;
  mode: 'relative' | 'absolute';
  unit: string;
  label_fa: string;
  current_mean: number | null;
  is_binary: boolean;
}

export interface ScenarioOptions {
  run_id: string;
  model: string;
  horizon: number;
  adjustable: ScenarioAdjustable[];
  window: { start: string; end: string };
  levels: Level[];
}

export interface ScenarioResult {
  level: Level;
  entity_id: string | null;
  label: string;
  horizon: number;
  window: { start: string; end: string; adjusted_start: string | null; adjusted_end: string | null };
  baseline_total: number;
  scenario_total: number;
  delta_total: number;
  impact_pct: number | null;
  applied: { column: string; kind: string; label_fa: string; change: string; mean_before: number; mean_after: number }[];
  rejected: { column: string; reason: string }[];
  model: string;
  series: { ds: string; baseline: number; scenario: number; delta: number }[];
  note_fa: string;
}

export interface ColumnProfile {
  name: string;
  dtype: string;
  kind: string;
  null_count: number;
  null_pct: number;
  unique_count: number;
  unique_pct: number;
  constant: boolean;
  sample_values: (string | number | null)[];
  min?: number | string;
  max?: number | string;
  mean?: number;
  std?: number;
  zero_pct?: number;
  negative_pct?: number;
  integer_like?: boolean;
}

export interface DatasetProfile {
  file: string;
  rows: number;
  columns: ColumnProfile[];
  n_columns: number;
  candidates: Record<string, { column: string; score: number }[]>;
  suggested_schema: SuggestedSchema;
  frequency: FrequencyInfo | null;
  sample_rows: Record<string, unknown>[];
  memory_mb: number;
}

export interface SuggestedSchema {
  timestamp: string | null;
  target: string | null;
  entity_id: string | null;
  destination: string | null;
  category: string | null;
  future_features: string[];
  historical_features: string[];
  static_features: string[];
  non_negative: boolean;
  integer: boolean;
}

export interface FrequencyInfo {
  frequency: string;
  pandas_inferred: string | null;
  confidence: number;
  regular: boolean;
  median_delta_days: number | null;
  n_timestamps: number;
  reason: string;
}

export interface DatasetRecord {
  id: number;
  name: string;
  slug: string;
  source: string;
  path: string;
  original_filename: string;
  size_bytes: number;
  n_rows: number;
  n_columns: number;
  mapping: Record<string, unknown>;
  contract_path: string;
  created_at: string;
  updated_at: string;
}

export interface ValidationFinding {
  code: string;
  severity: 'critical' | 'high' | 'medium' | 'low' | 'info';
  title: string;
  detail: string;
  recommendation: string;
  context: Record<string, unknown>;
}

export interface ValidationReport {
  health_score: number;
  grade: string;
  n_findings: number;
  findings: ValidationFinding[];
  by_severity: Record<string, number>;
  summary: Record<string, string | number>;
  panel?: Record<string, unknown>;
  adapter_notes?: string[];
}

export interface TrainingRun {
  run_id: string;
  status: 'pending' | 'running' | 'succeeded' | 'failed';
  stage: string;
  progress: number;
  profile: string;
  target: string;
  primary_metric: string;
  horizon: number;
  champion_model: string;
  champion_score: number | null;
  improvement: number | null;
  training_seconds: number | null;
  created_at: string;
  finished_at: string | null;
  warnings?: string[];
  error?: string;
}

export interface TrainingListResponse {
  runs: TrainingRun[];
  profiles: Record<string, { description: string; models: string[]; cv_folds: number; max_train_rows: number }>;
  metrics: string[];
  default_profile: string;
  sync_tasks: boolean;
}

export interface NarrativeResponse {
  text: string;
  source: string;
  grounded: boolean;
  facts: Record<string, unknown>;
}

export interface HeatmapResponse {
  dates: string[];
  rows: { entity_id: string; label: string; values: (number | null)[]; intensity: (number | null)[]; total: number }[];
  level: Level;
}
