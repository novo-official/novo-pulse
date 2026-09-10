/**
 * Shapes returned by /api/v1/pol4/*.
 *
 * Every one of these is read from a generated artefact - nothing here is
 * computed in the browser, and nothing is a placeholder. `city_code` is the
 * competition identifier and stays numeric everywhere; `city` and `province`
 * are display names from city_code_mapping.csv.
 */

export interface CityRef {
  city_code: number;
  city: string;
  province: string;
}

export interface DayPoint {
  checkin: string;
  predicted_demand: number;
  observed_so_far: number;
  predicted_remaining: number;
}

export interface CityRow extends CityRef {
  predicted_demand: number;
  observed_so_far: number;
  predicted_remaining: number;
  pickup_ratio: number | null;
  curve_source: string | null;
}

export interface ProvinceRow {
  province: string;
  predicted_demand: number;
  observed_so_far: number;
  predicted_remaining: number;
  cities: number;
  share: number;
}

export interface MomentumRow extends CityRef {
  observed: number;
  recent_pickup: number;
  expected_pickup: number;
  pickup_ratio: number;
  curve_source: string;
}

export interface OverviewKpis {
  total_predicted_demand: number;
  observed_so_far: number;
  predicted_remaining: number;
  peak_date: string;
  peak_demand: number;
  top_city: string;
  top_city_demand: number;
  fastest_pickup_city: string | null;
  fastest_pickup_ratio: number | null;
  backtest_wape: number;
  baseline_wape: number;
  stability_score: number | null;
  cities: number;
  dates: number;
}

export interface Overview {
  cutoff: string;
  target_window: [string, string];
  kpis: OverviewKpis;
  series: DayPoint[];
  top_cities: CityRow[];
  provinces: ProvinceRow[];
  momentum: MomentumRow[];
}

export interface HeatmapRow extends CityRef {
  values: number[];
}

export interface Heatmap {
  dates: string[];
  rows: HeatmapRow[];
  max: number;
}

export interface CityDetail {
  city: CityRef;
  totals: { predicted_demand: number; observed_so_far: number; predicted_remaining: number };
  momentum: MomentumRow | null;
  series: DayPoint[];
  peaks: { checkin: string; predicted_demand: number; observed_so_far: number }[];
  history: { checkin: string; demand: number }[];
}

export interface PickupCurve {
  city: CityRef;
  checkin: string;
  horizon: number;
  curve_level: string;
  observed_so_far: number;
  predicted_demand: number;
  predicted_remaining: number;
  observed: { days_to_checkin: number; searches: number; observed_cumulative: number }[];
  expected: { days_to_checkin: number; completion_fraction: number; expected_cumulative: number }[];
  available_checkins: string[];
}

export interface StabilitySnapshot {
  horizon: number;
  observed: number;
  prediction: number;
  actual: number;
  absolute_error: number;
  relative_error: number;
}

export interface Stability {
  city: CityRef;
  checkin: string;
  actual: number;
  snapshots: StabilitySnapshot[];
  available_checkins: string[];
  window: { target_start: string | null; anchor_cutoff: string | null };
  aggregate: {
    stability_score: number | null;
    mean_absolute_revision: number | null;
    mean_relative_revision: number | null;
    convergence_rate: number | null;
    by_step: { step: string; mean_absolute_revision: number; mean_relative_revision: number; convergence_rate: number; n: number }[];
    by_horizon: { horizon: number; wape: number; mean_prediction: number }[];
  };
}

export interface Segment {
  key: string | number;
  n: number;
  wape: number;
  mae: number;
  normalised_bias: number;
  actual_total: number;
  predicted_total: number;
}

export interface ModelPerformance {
  champion: {
    model: string;
    horizon_bands: number[][];
    n_features: number;
    log1p_target: boolean;
    training_rows: number;
    wape: number;
    normalised_bias: number;
  };
  baseline: { name: string; wape: number; normalised_bias: number };
  improvement: number;
  folds: { cutoff: string; champion_wape: number; baseline_wape: number; actual_total: number }[];
  hardest_fold: string;
  by_horizon_bucket: Segment[];
  baseline_by_horizon_bucket: Segment[];
  by_province: Segment[];
  by_weekday: Segment[];
  by_demand_bucket: Segment[];
  by_observation_state: Segment[];
  high_demand: Record<string, Segment>;
  feature_importance: { feature: string; importance: number; share: number }[];
  experiments: { name: string; stage: string; pooled: { wape: number; normalised_bias: number } }[];
  config: Record<string, unknown>;
}

export interface ReportKind {
  kind: string;
  description: string;
}

export interface ReportPreview {
  kind: string;
  rows: number;
  columns: string[];
  preview: Record<string, unknown>[];
}

export type ModelVariant = 'raw' | 'calibrated';

export interface DashboardModel {
  key: ModelVariant;
  label: string;
  calibration: string;
  wape: number;
  normalised_bias: number;
  forecast_total: number;
}

export interface DashboardCity extends CityRow {
  share: number;
  cumulative_share: number;
  observed_share: number;
  remaining_share: number;
  historical_mean: number | null;
  historical_cv: number | null;
}

export interface DashboardDay extends DayPoint {
  deviation_from_mean: number;
  demand_band: 'high' | 'normal' | 'low';
}

export interface Dashboard {
  shock: { all_folds_wape: number; without_fold_wape: number | null; error_share: number | null };
  selected_model: ModelVariant;
  models: DashboardModel[];
  cutoff: string;
  target_window: [string, string];
  summary: DashboardModel & {
    observed_so_far: number;
    predicted_remaining: number;
    baseline_wape: number;
    peak_date: string;
    peak_demand: number;
    peak_deviation: number;
    low_date: string;
    low_demand: number;
    low_deviation: number;
    cities: number;
    dates: number;
  };
  national_series: DashboardDay[];
  model_comparison: {
    checkin: string;
    predicted_demand_raw: number;
    predicted_demand_calibrated: number;
    delta: number;
  }[];
  cities: DashboardCity[];
  provinces: ProvinceRow[];
  heatmap: Heatmap;
  thresholds: {
    demand_median: number;
    pickup_median: number;
    remaining_share_median: number;
  };
  lead_time: (Segment & { observed_share: number })[];
  lead_time_daily: (Segment & { observed_share: number })[];
  demand_buckets: Segment[];
  high_demand: Segment[];
  evaluation_dimensions: {
    province: Segment[];
    weekday: Segment[];
    observation: Segment[];
  };
  folds: {
    cutoff: string;
    model_wape: number;
    model_bias: number;
    baseline_wape: number;
  }[];
  stability: Stability['aggregate'] & Record<string, unknown>;
}

export interface JuryEvidence {
  uncertainty: null | { nominal_coverage: number; folds: { cutoff: string; eligible_rows: number; observed_coverage: number | null; mean_width: number | null }[] };
  event_study: null | { summary: { clock: string; period: string; days: number; weekday_adjusted_ratio: number; matched_jalali_year_ratio: number }[] };
  generated_at: string;
  cutoff: string;
  mode: string;
  target_window: string[];
  scope: { cities: number; provinces: string[]; dates: number; target: string };
  performance: {
    champion: { wape: number }; baseline: { wape: number }; relative_improvement: number;
    confidence_interval: { lower: number; upper: number }; prequential_wape: number;
    folds: { cutoff: string; champion: number; baseline: number; shock_overlap: boolean }[];
  };
  shock: { all_folds_wape: number; without_fold_wape: number | null; error_share: number | null; source: string };
  raw_shock: { all_folds_wape: number; without_fold_wape: number | null };
  clustering: null | { selected_arm: string; warning: string; levels: {
    mean_groups: number; clustered_wape: number; control_wape: number;
    mechanical_gain: number; modelling_gain: number;
  }[] };
  experiments: null | { status: string; limitations: string[]; arms: Record<string, {
    raw: { pooled: { wape: number }; interval: { lower?: number; upper?: number } };
    calibrated: { pooled: { wape: number } };
  }> };
  submission: { rows: number; finite: boolean; nonnegative: boolean; observed_floor: boolean; duplicate_keys: number; sha256: string };
  trainset_recovery: null | { rows: number; file_hash_matches: Record<string, boolean> };
  decisions: { city_code: number; city: string; province: string; predicted_demand: number;
    observed_share: number; forecast_share: number; pickup_ratio: number | null;
    peak_checkin: string; action: string; owner: string; evidence_status: string; required_before_spend: string }[];
  limits: string[];
  sources: Record<string, string>;
}
