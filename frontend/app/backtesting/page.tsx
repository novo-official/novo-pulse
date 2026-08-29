'use client';

import { useQuery } from '@tanstack/react-query';
import { Activity, GitCompare, Target } from 'lucide-react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ComposedChart,
  Legend,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

import { CHART, axisProps } from '@/components/charts/palette';
import { ChartTooltip } from '@/components/charts/tooltip';
import { FilterBar } from '@/components/filters/forecast-filters';
import { Badge } from '@/components/ui/badge';
import { Card, CardBody, CardHeader } from '@/components/ui/card';
import { PageHeader } from '@/components/ui/page-header';
import { AsyncBoundary, CardSkeleton, ChartSkeleton, NoModelState } from '@/components/ui/states';
import { Td, TableWrap, Th } from '@/components/ui/table';
import { InfoHint } from '@/components/ui/tooltip';
import { useForecastFilters } from '@/hooks/useForecastFilters';
import { api } from '@/lib/api/endpoints';
import type { SegmentScore } from '@/lib/types/api';
import { formatCompact, formatDate, formatMetric, formatNumber } from '@/lib/utils';

const WEEKDAY_FA = ['دوشنبه', 'سه‌شنبه', 'چهارشنبه', 'پنجشنبه', 'جمعه', 'شنبه', 'یکشنبه'];
const MONTH_FA = [
  'ژانویه', 'فوریه', 'مارس', 'آوریل', 'مه', 'ژوئن',
  'ژوئیه', 'اوت', 'سپتامبر', 'اکتبر', 'نوامبر', 'دسامبر',
];

function SegmentChart({
  rows,
  metric,
  labelOf,
}: {
  rows: SegmentScore[];
  metric: string;
  labelOf: (key: string | number) => string;
}) {
  const data = rows
    .map((row) => ({ key: labelOf(row.key), value: Number(row[metric] ?? 0), n: row.n }))
    .filter((row) => Number.isFinite(row.value));
  const worst = Math.max(...data.map((row) => row.value), 0);

  return (
    <div className="chart-ltr h-[240px] w-full">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 8, right: 8, bottom: 4, left: 4 }}>
          <CartesianGrid stroke={CHART.grid} strokeDasharray="3 3" vertical={false} />
          <XAxis dataKey="key" {...axisProps} interval={0} angle={-30} textAnchor="end" height={54} />
          <YAxis {...axisProps} width={44} tickFormatter={(v: number) => formatMetric(v, metric)} />
          <Tooltip
            content={<ChartTooltip valueDigits={3} formatLabel={(value) => value} />}
            cursor={{ fill: 'rgba(148,163,184,0.12)' }}
          />
          <Bar dataKey="value" name={metric.toUpperCase()} radius={[4, 4, 0, 0]}>
            {data.map((row) => (
              <Cell
                key={row.key}
                fill={row.value >= worst * 0.9 ? CHART.negative : CHART.forecast}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export default function BacktestingPage() {
  const { level, entityId, horizon } = useForecastFilters();

  const metricsQuery = useQuery({ queryKey: ['backtest-metrics'], queryFn: api.backtestMetrics });
  const seriesQuery = useQuery({
    queryKey: ['backtests', level, entityId],
    queryFn: () => api.backtests({ level, id: entityId }),
  });
  const timeseriesQuery = useQuery({
    queryKey: ['timeseries', level, entityId, horizon],
    queryFn: () => api.timeseries({ level, id: entityId, horizon }),
  });

  const metrics = metricsQuery.data?.data;
  const series = seriesQuery.data?.data?.series ?? [];
  const metricName = metrics?.primary_metric ?? 'wape';

  if (metricsQuery.isSuccess && !metricsQuery.data?.available) {
    return (
      <Card>
        <NoModelState detail={metricsQuery.data?.detailFa} />
      </Card>
    );
  }

  return (
    <div className="page-stack">
      <PageHeader eyebrow="اعتماد به مدل" title="اعتبارسنجی گذشته‌نگر" description="عملکرد واقعی مدل را روی بازه‌هایی که هرگز ندیده است، با اعتبارسنجی متحرک زمانی بررسی کنید." />

      {timeseriesQuery.data?.data ? (
        <FilterBar
          levels={timeseriesQuery.data.data.levels}
          members={timeseriesQuery.data.data.members}
          horizons={[7, 14, 30]}
        />
      ) : null}

      {/* -------------------------------------------------- folds ribbon */}
      <AsyncBoundary
        isLoading={metricsQuery.isLoading}
        error={metricsQuery.error}
        onRetry={() => metricsQuery.refetch()}
        skeleton={
          <Card>
            <CardSkeleton lines={4} />
          </Card>
        }
      >
        <Card className="overflow-hidden border-cyan-100 bg-gradient-to-l from-cyan-50/50 to-surface shadow-lift">
          <CardHeader
            icon={<GitCompare className="h-4.5 w-4.5" />}
            title="طرح اعتبارسنجی"
            subtitle={`مدل منتخب: ${metrics?.champion ?? '—'}`}
            action={
              <InfoHint>
                در هر تا (fold) مدل فقط با داده‌های قبل از نقطه برش آموزش می‌بیند و روی افق بعد از آن
                ارزیابی می‌شود. هیچ داده‌ای از آینده به مدل نشت نمی‌کند.
              </InfoHint>
            }
          />
          <CardBody>
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
              {(metrics?.folds ?? []).map((fold) => (
                <div key={fold.fold} className="relative overflow-hidden rounded-xl border border-line/70 bg-surface p-3.5 shadow-sm before:absolute before:inset-y-0 before:right-0 before:w-1 before:bg-gradient-to-b before:from-brand-500 before:to-cyan-400">
                  <p className="text-xs font-semibold text-muted">تا {fold.fold}</p>
                  <p className="nums mt-1.5 text-xs text-ink" dir="ltr">
                    train ≤ {fold.train_end}
                  </p>
                  <p className="nums text-xs text-brand-600" dir="ltr">
                    valid {fold.valid_start} → {fold.valid_end}
                  </p>
                </div>
              ))}
            </div>
          </CardBody>
        </Card>
      </AsyncBoundary>

      {/* ------------------------------------------- actual vs predicted */}
      <Card className="signal-card overflow-hidden">
        <CardHeader
          icon={<Activity className="h-4.5 w-4.5" />}
          title="مقدار واقعی در برابر پیش‌بینی"
          subtitle="روی پنجره‌های اعتبارسنجی که مدل هنگام آموزش ندیده بود"
        />
        <AsyncBoundary
          isLoading={seriesQuery.isLoading}
          error={seriesQuery.error}
          isEmpty={series.length === 0}
          onRetry={() => seriesQuery.refetch()}
          skeleton={<ChartSkeleton height={340} />}
        >
          <CardBody>
            <div className="chart-ltr h-[360px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={series} margin={{ top: 8, right: 12, bottom: 4, left: 4 }}>
                  <CartesianGrid stroke={CHART.grid} strokeDasharray="3 3" vertical={false} />
                  <XAxis dataKey="ds" {...axisProps} tickFormatter={formatDate} minTickGap={28} />
                  <YAxis {...axisProps} width={48} tickFormatter={(v: number) => formatCompact(v)} />
                  <Tooltip content={<ChartTooltip />} />
                  <Legend
                    verticalAlign="top"
                    height={30}
                    iconType="plainline"
                    wrapperStyle={{ fontSize: 12, direction: 'rtl' }}
                  />
                  <Line
                    type="monotone"
                    dataKey="actual"
                    name="واقعی"
                    stroke={CHART.actual}
                    strokeWidth={2}
                    dot={false}
                    isAnimationActive={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="prediction"
                    name="پیش‌بینی"
                    stroke={CHART.forecast}
                    strokeWidth={2}
                    dot={false}
                    isAnimationActive={false}
                  />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          </CardBody>
        </AsyncBoundary>
      </Card>

      {/* -------------------------------------- horizon + coverage tables */}
      <div className="grid gap-6 lg:grid-cols-2">
        <Card className="signal-card">
          <CardHeader
            icon={<Target className="h-4.5 w-4.5" />}
            title="خطا بر حسب افق"
            subtitle="افت دقت مدل هرچه به آینده دورتر می‌رویم"
          />
          <CardBody>
            <TableWrap minWidth={420}>
              <thead>
                <tr>
                  <Th align="right">افق (روز)</Th>
                  <Th>{metricName.toUpperCase()}</Th>
                  <Th>MAE</Th>
                  <Th>تعداد</Th>
                </tr>
              </thead>
              <tbody>
                {(metrics?.by_horizon ?? []).map((row) => (
                  <tr key={row.bucket}>
                    <Td align="right" className="nums font-medium">
                      {row.bucket}
                    </Td>
                    <Td className="nums font-semibold">
                      {formatMetric(Number(row[metricName]), metricName)}
                    </Td>
                    <Td className="nums text-muted">{formatNumber(Number(row.mae), 2)}</Td>
                    <Td className="nums text-muted">{formatNumber(row.n)}</Td>
                  </tr>
                ))}
              </tbody>
            </TableWrap>
          </CardBody>
        </Card>

        <Card className="signal-card">
          <CardHeader
            title="پوشش بازه اطمینان"
            subtitle={`روش: ${metrics?.uncertainty_method ?? '—'}`}
            action={
              <InfoHint>
                پوشش مشاهده‌شده = سهم مقادیر واقعی که داخل بازه پیش‌بینی قرار گرفته‌اند. هرچه به
                پوشش اسمی (۸۰٪) نزدیک‌تر باشد، بازه بهتر کالیبره شده است.
              </InfoHint>
            }
          />
          <CardBody>
            <TableWrap minWidth={460}>
              <thead>
                <tr>
                  <Th align="right">افق</Th>
                  <Th>پوشش اسمی</Th>
                  <Th>پوشش مشاهده‌شده</Th>
                  <Th>پهنای میانگین</Th>
                </tr>
              </thead>
              <tbody>
                {(metrics?.coverage_by_horizon ?? []).map((row) => {
                  const gap = Math.abs(row.observed_coverage - row.nominal_coverage);
                  return (
                    <tr key={row.bucket}>
                      <Td align="right" className="nums font-medium">
                        {row.bucket}
                      </Td>
                      <Td className="nums text-muted">
                        {(row.nominal_coverage * 100).toFixed(0)}٪
                      </Td>
                      <Td>
                        <Badge tone={gap <= 0.05 ? 'success' : gap <= 0.1 ? 'warning' : 'danger'}>
                          {(row.observed_coverage * 100).toFixed(1)}٪
                        </Badge>
                      </Td>
                      <Td className="nums text-muted">{formatNumber(row.mean_width, 2)}</Td>
                    </tr>
                  );
                })}
              </tbody>
            </TableWrap>
          </CardBody>
        </Card>
      </div>

      {/* ------------------------------------------------------- segments */}
      <div className="grid gap-6 lg:grid-cols-2">
        {metrics?.segments?.by_destination?.length ? (
          <Card className="signal-card">
            <CardHeader title="خطا بر حسب مقصد" subtitle="کدام مقاصد سخت‌تر پیش‌بینی می‌شوند؟" />
            <CardBody>
              <SegmentChart
                rows={metrics.segments.by_destination}
                metric={metricName}
                labelOf={(key) => String(key)}
              />
            </CardBody>
          </Card>
        ) : null}

        {metrics?.segments?.by_weekday?.length ? (
          <Card className="signal-card">
            <CardHeader title="خطا بر حسب روز هفته" />
            <CardBody>
              <SegmentChart
                rows={metrics.segments.by_weekday}
                metric={metricName}
                labelOf={(key) => WEEKDAY_FA[Number(key)] ?? String(key)}
              />
            </CardBody>
          </Card>
        ) : null}

        {metrics?.segments?.by_month?.length ? (
          <Card className="signal-card">
            <CardHeader title="خطا بر حسب ماه" />
            <CardBody>
              <SegmentChart
                rows={metrics.segments.by_month}
                metric={metricName}
                labelOf={(key) => MONTH_FA[Number(key) - 1] ?? String(key)}
              />
            </CardBody>
          </Card>
        ) : null}

        {metrics?.segments?.by_horizon?.length ? (
          <Card className="signal-card">
            <CardHeader title="خطا بر حسب گام افق" subtitle="روز به روز" />
            <CardBody>
              <SegmentChart
                rows={metrics.segments.by_horizon}
                metric={metricName}
                labelOf={(key) => String(key)}
              />
            </CardBody>
          </Card>
        ) : null}
      </div>
    </div>
  );
}
