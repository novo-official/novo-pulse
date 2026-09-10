/** One typed function per Pol 4 endpoint. Nothing else calls fetch directly. */
import { getEnvelope } from '@/lib/api/client';

import type {
  CityDetail,
  CityRow,
  Dashboard,
  Heatmap,
  ModelPerformance,
  Overview,
  PickupCurve,
  ReportKind,
  ReportPreview,
  Stability,
  ModelVariant,
  JuryEvidence,
} from './types';

const BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? 'http://localhost:8000/api/v1';

export const pol4 = {
  jury: () => getEnvelope<JuryEvidence>('/pol4/jury/'),
  dashboard: (model: ModelVariant) =>
    getEnvelope<Dashboard>('/pol4/dashboard/', { model }),
  overview: () => getEnvelope<Overview>('/pol4/overview/'),

  heatmap: (topN: number) => getEnvelope<Heatmap>('/pol4/heatmap/', { top_n: topN }),

  cities: () => getEnvelope<{ cities: CityRow[] }>('/pol4/cities/'),

  city: (cityCode: number | string) => getEnvelope<CityDetail>(`/pol4/cities/${cityCode}/`),

  pickup: (cityCode: number | string, checkin?: string) =>
    getEnvelope<PickupCurve>(`/pol4/cities/${cityCode}/pickup/`, checkin ? { checkin } : undefined),

  stability: (cityCode?: number | string, checkin?: string) =>
    getEnvelope<Stability>('/pol4/stability/', {
      ...(cityCode ? { city_code: cityCode } : {}),
      ...(checkin ? { checkin } : {}),
    }),

  modelPerformance: () => getEnvelope<ModelPerformance>('/pol4/model-performance/'),

  reports: () => getEnvelope<{ reports: ReportKind[] }>('/pol4/reports/'),

  reportPreview: (kind: string, filters: Record<string, string | undefined>) =>
    getEnvelope<ReportPreview>(`/pol4/reports/${kind}/`, filters),

  /** Direct CSV download URL - the browser fetches it, not the app. */
  reportUrl: (kind: string, filters: Record<string, string | undefined>) => {
    const query = new URLSearchParams();
    Object.entries(filters).forEach(([key, value]) => {
      if (value) query.set(key, value);
    });
    const suffix = query.toString();
    return `${BASE}/pol4/reports/${kind}.csv${suffix ? `?${suffix}` : ''}`;
  },
};
