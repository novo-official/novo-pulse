/** One typed function per API endpoint. Nothing else calls `fetch` directly. */
import { getEnvelope, getRaw, postEnvelope, postForm } from './client';
import type {
  Anomaly,
  BacktestMetrics,
  BacktestSeries,
  DashboardSummary,
  DatasetProfile,
  DatasetRecord,
  DriversResponse,
  HealthResponse,
  HeatmapResponse,
  LeaderboardResponse,
  Level,
  ModelsResponse,
  NarrativeResponse,
  Opportunity,
  OverviewRow,
  PeakPeriod,
  ScenarioOptions,
  ScenarioResult,
  SuggestedSchema,
  TimeseriesResponse,
  TrainingListResponse,
  TrainingRun,
  ValidationReport,
} from '@/lib/types/api';

export interface ForecastFilters {
  level?: Level;
  id?: string | null;
  horizon?: number;
  run_id?: string | null;
}

export const api = {
  health: () => getRaw<HealthResponse>('/health/'),

  system: () => getEnvelope<Record<string, unknown>>('/system/'),

  dashboardSummary: (filters: ForecastFilters) =>
    getEnvelope<DashboardSummary>('/dashboard/summary/', { ...filters }),

  timeseries: (filters: ForecastFilters) =>
    getEnvelope<TimeseriesResponse>('/forecasts/timeseries/', { ...filters }),

  drivers: (filters: ForecastFilters = {}) =>
    getEnvelope<DriversResponse>('/forecasts/drivers/', { ...filters }),

  peaks: (limit = 8) =>
    getEnvelope<{ peaks: PeakPeriod[]; troughs: PeakPeriod[] }>('/forecasts/peaks/', { limit }),

  overview: (filters: ForecastFilters) =>
    getEnvelope<OverviewRow[]>('/forecasts/overview/', { ...filters }),

  heatmap: (filters: ForecastFilters & { top_n?: number }) =>
    getEnvelope<HeatmapResponse>('/forecasts/heatmap/', { ...filters }),

  narrative: (filters: ForecastFilters) =>
    getEnvelope<NarrativeResponse>('/forecasts/narrative/', { ...filters }),

  anomalies: (params: { id?: string | null; severity?: string; limit?: number } = {}) =>
    getEnvelope<Anomaly[]>('/anomalies/', { ...params }),

  models: () => getEnvelope<ModelsResponse>('/models/'),

  leaderboard: () => getEnvelope<LeaderboardResponse>('/models/leaderboard/'),

  backtests: (filters: ForecastFilters) =>
    getEnvelope<BacktestSeries>('/backtests/', { ...filters }),

  backtestMetrics: () => getEnvelope<BacktestMetrics>('/backtests/metrics/'),

  insights: () => getEnvelope<Record<string, unknown>>('/insights/'),

  opportunities: () => getEnvelope<Opportunity[]>('/insights/opportunities/'),

  scenarioOptions: () => getEnvelope<ScenarioOptions>('/scenarios/options/'),

  simulate: (body: {
    adjustments: { column: string; change_pct?: number; value?: number; mode?: string }[];
    level?: Level;
    entity_id?: string | null;
    start_date?: string | null;
    end_date?: string | null;
    horizon?: number;
  }) => postEnvelope<ScenarioResult>('/scenarios/simulate/', body),

  datasets: () =>
    getEnvelope<{
      datasets: DatasetRecord[];
      active_contract: Record<string, unknown>;
      metrics: string[];
    }>('/datasets/'),

  uploadDataset: (form: FormData) =>
    postForm<{ dataset: DatasetRecord; profile: DatasetProfile }>('/datasets/upload/', form),

  profileDataset: (body: { dataset_id?: number; path?: string }) =>
    postEnvelope<{ profile: DatasetProfile; dataset_id: number | null }>('/datasets/profile/', body),

  mapDataset: (body: Record<string, unknown>) =>
    postEnvelope<{ contract: Record<string, unknown>; contract_path: string; dataset: DatasetRecord }>(
      '/datasets/map/',
      body,
    ),

  validateDataset: (body: { dataset_id?: number; mapping?: Record<string, unknown> }) =>
    postEnvelope<ValidationReport>('/datasets/validate/', body),

  contract: () => getEnvelope<Record<string, unknown>>('/datasets/contract/'),

  training: () => getEnvelope<TrainingListResponse>('/training/'),

  startTraining: (body: {
    dataset_id?: number;
    profile?: string;
    horizon?: number;
    metric?: string;
    experiment?: string;
    /** Sent so the run uses the mapping currently on screen. */
    mapping?: Record<string, unknown>;
  }) => postEnvelope<TrainingRun>('/training/run/', body),

  trainingRun: (runId: string) => getEnvelope<TrainingRun>(`/training/${runId}/`),
};

export type { SuggestedSchema };
