'use client';

import { useQuery } from '@tanstack/react-query';
import { LineChart } from 'lucide-react';

import { DemandHeatmap } from '@/components/charts/heatmap';
import { ForecastChart } from '@/components/charts/forecast-chart';
import { AnomalyPanel } from '@/components/dashboard/anomaly-panel';
import { DriversCard } from '@/components/dashboard/drivers-card';
import { KpiCards } from '@/components/dashboard/kpi-cards';
import { NarrativeCard } from '@/components/dashboard/narrative-card';
import { OpportunitiesCard } from '@/components/dashboard/opportunities-card';
import { OverviewTable } from '@/components/dashboard/overview-table';
import { PeaksCard } from '@/components/dashboard/peaks-card';
import { FilterBar } from '@/components/filters/forecast-filters';
import { Card, CardBody, CardHeader } from '@/components/ui/card';
import {
  AsyncBoundary,
  CardSkeleton,
  ChartSkeleton,
  NoModelState,
  Skeleton,
} from '@/components/ui/states';
import { useForecastFilters } from '@/hooks/useForecastFilters';
import { api } from '@/lib/api/endpoints';
import { LEVEL_FA } from '@/lib/utils';

export default function DashboardPage() {
  const { level, entityId, horizon, setEntityId } = useForecastFilters();
  const filters = { level, id: entityId, horizon };

  const summaryQuery = useQuery({
    queryKey: ['dashboard-summary', level, horizon],
    queryFn: () => api.dashboardSummary({ level, horizon }),
  });
  const timeseriesQuery = useQuery({
    queryKey: ['timeseries', level, entityId, horizon],
    queryFn: () => api.timeseries(filters),
  });
  const driversQuery = useQuery({ queryKey: ['drivers'], queryFn: () => api.drivers() });
  const overviewQuery = useQuery({
    queryKey: ['overview', level, horizon],
    queryFn: () => api.overview({ level, horizon }),
  });
  const peaksQuery = useQuery({ queryKey: ['peaks'], queryFn: () => api.peaks(6) });
  const anomalyQuery = useQuery({
    queryKey: ['anomalies', entityId],
    queryFn: () => api.anomalies({ id: entityId, limit: 12 }),
  });
  const narrativeQuery = useQuery({
    queryKey: ['narrative', level, entityId, horizon],
    queryFn: () => api.narrative(filters),
  });
  const heatmapQuery = useQuery({
    queryKey: ['heatmap', level, horizon],
    queryFn: () => api.heatmap({ level, horizon, top_n: 12 }),
  });

  const summary = summaryQuery.data;
  const timeseries = timeseriesQuery.data?.data;

  if (summaryQuery.isSuccess && !summary?.available) {
    return (
      <Card>
        <NoModelState detail={summary?.detailFa} />
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      {/* --------------------------------------------------------- KPIs */}
      <AsyncBoundary
        isLoading={summaryQuery.isLoading}
        error={summaryQuery.error}
        onRetry={() => summaryQuery.refetch()}
        skeleton={
          <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-6">
            {Array.from({ length: 6 }).map((_, index) => (
              <Card key={index}>
                <CardSkeleton lines={2} />
              </Card>
            ))}
          </div>
        }
      >
        {summary?.data ? <KpiCards summary={summary.data} /> : null}
      </AsyncBoundary>

      {/* ------------------------------------------------------ filters */}
      {timeseries ? (
        <FilterBar
          levels={timeseries.levels}
          members={timeseries.members}
          horizons={summary?.data?.available_horizons ?? [7, 14, 30]}
        />
      ) : (
        <Skeleton className="h-24 w-full rounded-2xl" />
      )}

      {/* ------------------------------------------------- forecast chart */}
      <Card>
        <CardHeader
          icon={<LineChart className="h-4.5 w-4.5" />}
          title={`روند تقاضا — ${timeseries?.label ?? LEVEL_FA[level]}`}
          subtitle="تاریخچه واقعی، پیش‌بینی گذشته‌نگر و پیش‌بینی آینده به همراه بازه اطمینان ۸۰٪"
        />
        <AsyncBoundary
          isLoading={timeseriesQuery.isLoading}
          error={timeseriesQuery.error}
          isEmpty={!timeseries?.series?.length}
          onRetry={() => timeseriesQuery.refetch()}
          skeleton={<ChartSkeleton height={340} />}
        >
          <CardBody>
            {timeseries ? (
              <ForecastChart
                series={timeseries.series}
                forecastStart={timeseries.forecast_start}
                height={380}
              />
            ) : null}
          </CardBody>
        </AsyncBoundary>
      </Card>

      {/* ------------------------------------- narrative + drivers + opps */}
      <div className="grid gap-6 lg:grid-cols-3">
        <NarrativeCard
          narrative={narrativeQuery.data?.data ?? null}
          isLoading={narrativeQuery.isLoading}
        />
        <AsyncBoundary
          isLoading={driversQuery.isLoading}
          error={driversQuery.error}
          onRetry={() => driversQuery.refetch()}
          skeleton={
            <Card>
              <CardSkeleton lines={6} />
            </Card>
          }
        >
          {driversQuery.data?.data ? <DriversCard drivers={driversQuery.data.data} /> : null}
        </AsyncBoundary>
        <AsyncBoundary
          isLoading={summaryQuery.isLoading}
          error={summaryQuery.error}
          skeleton={
            <Card>
              <CardSkeleton lines={5} />
            </Card>
          }
        >
          <OpportunitiesCard
            opportunities={summary?.data?.insights.decision_opportunities ?? []}
          />
        </AsyncBoundary>
      </div>

      {/* ------------------------------------------- peaks + anomalies */}
      <div className="grid gap-6 lg:grid-cols-2">
        <AsyncBoundary
          isLoading={peaksQuery.isLoading}
          error={peaksQuery.error}
          onRetry={() => peaksQuery.refetch()}
          skeleton={
            <Card>
              <CardSkeleton lines={5} />
            </Card>
          }
        >
          <PeaksCard
            peaks={peaksQuery.data?.data?.peaks ?? []}
            troughs={peaksQuery.data?.data?.troughs ?? []}
          />
        </AsyncBoundary>
        <AsyncBoundary
          isLoading={anomalyQuery.isLoading}
          error={anomalyQuery.error}
          onRetry={() => anomalyQuery.refetch()}
          skeleton={
            <Card>
              <CardSkeleton lines={5} />
            </Card>
          }
        >
          <AnomalyPanel anomalies={anomalyQuery.data?.data ?? []} />
        </AsyncBoundary>
      </div>

      {/* -------------------------------------------------- overview table */}
      <AsyncBoundary
        isLoading={overviewQuery.isLoading}
        error={overviewQuery.error}
        onRetry={() => overviewQuery.refetch()}
        skeleton={
          <Card>
            <CardSkeleton lines={7} />
          </Card>
        }
      >
        <OverviewTable
          rows={overviewQuery.data?.data ?? []}
          onSelect={setEntityId}
          title={`نمای کلی — ${LEVEL_FA[level]}`}
        />
      </AsyncBoundary>

      {/* ------------------------------------------------------- heatmap */}
      <AsyncBoundary
        isLoading={heatmapQuery.isLoading}
        error={heatmapQuery.error}
        onRetry={() => heatmapQuery.refetch()}
        skeleton={
          <Card>
            <ChartSkeleton height={260} />
          </Card>
        }
      >
        {heatmapQuery.data?.data ? <DemandHeatmap data={heatmapQuery.data.data} /> : null}
      </AsyncBoundary>
    </div>
  );
}
