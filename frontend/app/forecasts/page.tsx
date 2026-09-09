'use client';

import { useQuery } from '@tanstack/react-query';
import { LineChart } from 'lucide-react';

import { ForecastChart } from '@/components/charts/forecast-chart';
import { DemandHeatmap } from '@/components/charts/heatmap';
import { OverviewTable } from '@/components/dashboard/overview-table';
import { ForecastHierarchyCard } from '@/components/dashboard/forecast-hierarchy-card';
import { PeaksCard } from '@/components/dashboard/peaks-card';
import { FilterBar } from '@/components/filters/forecast-filters';
import { Badge } from '@/components/ui/badge';
import { Card, CardBody, CardHeader } from '@/components/ui/card';
import { PageHeader } from '@/components/ui/page-header';
import {
  AsyncBoundary,
  CardSkeleton,
  ChartSkeleton,
  NoModelState,
  Skeleton,
} from '@/components/ui/states';
import { useForecastFilters } from '@/hooks/useForecastFilters';
import { api } from '@/lib/api/endpoints';
import { LEVEL_FA, displayNameFa, formatCompact, formatNumber } from '@/lib/utils';

export default function ForecastsPage() {
  const { level, entityId, horizon, setEntityId } = useForecastFilters();
  const filters = { level, id: entityId, horizon };

  const timeseriesQuery = useQuery({
    queryKey: ['timeseries', level, entityId, horizon],
    queryFn: () => api.timeseries(filters),
  });
  const overviewQuery = useQuery({
    queryKey: ['overview', level, horizon],
    queryFn: () => api.overview({ level, horizon }),
  });
  const peaksQuery = useQuery({ queryKey: ['peaks'], queryFn: () => api.peaks(8) });
  const heatmapQuery = useQuery({
    queryKey: ['heatmap', level, horizon],
    queryFn: () => api.heatmap({ level, horizon, top_n: 14 }),
  });
  const summaryQuery = useQuery({
    queryKey: ['dashboard-summary', level, horizon],
    queryFn: () => api.dashboardSummary({ level, horizon }),
  });

  const timeseries = timeseriesQuery.data?.data;

  if (timeseriesQuery.isSuccess && !timeseriesQuery.data?.available) {
    return (
      <Card>
        <NoModelState detail={timeseriesQuery.data?.detailFa} />
      </Card>
    );
  }

  return (
    <div className="page-stack">
      <PageHeader eyebrow="چشم‌انداز بازار" title="پیش‌بینی تقاضا" description="تقاضای آینده را در سطح اقامتگاه، مقصد، دسته‌بندی یا کل بازار ببینید و عدم‌قطعیت را در تصمیم لحاظ کنید." />

      {timeseries ? (
        <FilterBar
          levels={timeseries.levels}
          members={timeseries.members}
          horizons={summaryQuery.data?.data?.available_horizons ?? [7, 14, 30]}
        />
      ) : (
        <Skeleton className="h-24 w-full rounded-2xl" />
      )}

      <Card className="overflow-hidden border-brand-100/80 shadow-lift">
        <CardHeader
          icon={<LineChart className="h-4.5 w-4.5" />}
          title={timeseries?.label ? displayNameFa(timeseries.label) : LEVEL_FA[level]}
          subtitle={`افق ${horizon} روزه · سطح ${LEVEL_FA[level]}`}
          action={
            timeseries ? (
              <div className="flex flex-wrap gap-2">
                <Badge tone="brand">مجموع: {formatNumber(timeseries.totals.forecast)}</Badge>
                <Badge tone="neutral">
                  {formatCompact(timeseries.totals.lower)} – {formatCompact(timeseries.totals.upper)}
                </Badge>
              </div>
            ) : null
          }
        />
        <AsyncBoundary
          isLoading={timeseriesQuery.isLoading}
          error={timeseriesQuery.error}
          isEmpty={!timeseries?.series?.length}
          onRetry={() => timeseriesQuery.refetch()}
          skeleton={<ChartSkeleton height={400} />}
        >
          <CardBody>
            {timeseries ? (
              <ForecastChart
                series={timeseries.series}
                forecastStart={timeseries.forecast_start}
                height={430}
              />
            ) : null}
          </CardBody>
        </AsyncBoundary>
      </Card>

      <div className="grid gap-6 lg:grid-cols-[2fr_1fr]">
        <AsyncBoundary
          isLoading={overviewQuery.isLoading}
          error={overviewQuery.error}
          onRetry={() => overviewQuery.refetch()}
          skeleton={
            <Card>
              <CardSkeleton lines={8} />
            </Card>
          }
        >
          <OverviewTable
            rows={overviewQuery.data?.data ?? []}
            onSelect={setEntityId}
            title={`جزئیات — ${LEVEL_FA[level]}`}
          />
        </AsyncBoundary>

        <AsyncBoundary
          isLoading={peaksQuery.isLoading}
          error={peaksQuery.error}
          onRetry={() => peaksQuery.refetch()}
          skeleton={
            <Card>
              <CardSkeleton lines={6} />
            </Card>
          }
        >
          <PeaksCard
            peaks={peaksQuery.data?.data?.peaks ?? []}
            troughs={peaksQuery.data?.data?.troughs ?? []}
          />
        </AsyncBoundary>
      </div>

      <AsyncBoundary
        isLoading={heatmapQuery.isLoading}
        error={heatmapQuery.error}
        onRetry={() => heatmapQuery.refetch()}
        skeleton={
          <Card>
            <ChartSkeleton height={280} />
          </Card>
        }
      >
        {heatmapQuery.data?.data ? <DemandHeatmap data={heatmapQuery.data.data} /> : null}
      </AsyncBoundary>

      <ForecastHierarchyCard levels={timeseries?.levels ?? []} />
    </div>
  );
}
