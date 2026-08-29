'use client';

import { useQuery } from '@tanstack/react-query';
import { Activity, CalendarDays, ChevronLeft, GitCompare, Target } from 'lucide-react';
import type { CSSProperties } from 'react';
import { Bar, BarChart, CartesianGrid, Cell, ComposedChart, Legend, Line, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { CHART, axisProps } from '@/components/charts/palette';
import { GlassTooltip } from '@/components/charts/glass-tooltip';
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

const WEEKDAY_FA = ['دوشنبه', 'سه‌شنبه', 'چهارشنبه', 'پنجشنبه', 'جمعه', 'شنبه', 'یکشنبه'];
const MONTH_FA = ['ژانویه', 'فوریه', 'مارس', 'آوریل', 'مه', 'ژوئن', 'ژوئیه', 'اوت', 'سپتامبر', 'اکتبر', 'نوامبر', 'دسامبر'];
const fa = new Intl.NumberFormat('fa-IR', { maximumFractionDigits: 2 });
const faOne = new Intl.NumberFormat('fa-IR', { minimumFractionDigits: 1, maximumFractionDigits: 1 });
const faCompact = new Intl.NumberFormat('fa-IR', { notation: 'compact', maximumFractionDigits: 1 });
const bdi = (value: string | number, className = '') => <bdi dir="ltr" className={`backtest-number tabular-nums ${className}`}>{value}</bdi>;
const percent = (value: number, digits = 1) => `${new Intl.NumberFormat('fa-IR', { minimumFractionDigits: digits, maximumFractionDigits: digits }).format(value * 100)}٪`;
const chartDate = (iso: string) => { const date = new Date(iso); return Number.isNaN(date.getTime()) ? iso : `${fa.format(date.getDate())} ${MONTH_FA[date.getMonth()]}`; };
const metricFA = (value: number, metric: string) => {
  if (!Number.isFinite(value)) return '—';
  if (metric === 'wape' || metric === 'rmsle' || metric === 'mape' || metric === 'smape') return percent(value);
  if (metric === 'r2') return new Intl.NumberFormat('fa-IR', { minimumFractionDigits: 3, maximumFractionDigits: 3 }).format(value);
  return new Intl.NumberFormat('fa-IR', { minimumFractionDigits: 3, maximumFractionDigits: 3 }).format(value);
};

function BacktestTooltip({ active, payload, label }: { active?: boolean; payload?: Array<{ dataKey?: string; value?: number }>; label?: string }) {
  if (!active || !payload?.length) return null;
  const actual = payload.find((item) => item.dataKey === 'actual')?.value;
  const prediction = payload.find((item) => item.dataKey === 'prediction')?.value;
  return <GlassTooltip><p className="glass-chart-tooltip__date">{label ? new Intl.DateTimeFormat('fa-IR', { day: 'numeric', month: 'long' }).format(new Date(label)) : '—'}</p><ul className="glass-chart-tooltip__rows">{actual !== undefined ? <li><span className="glass-chart-tooltip__dot" style={{ '--series-color': CHART.actual } as CSSProperties} /><span>واقعی</span><strong>{bdi(fa.format(actual))}</strong></li> : null}{prediction !== undefined ? <li><span className="glass-chart-tooltip__dot" style={{ '--series-color': CHART.forecast } as CSSProperties} /><span>پیش‌بینی</span><strong>{bdi(fa.format(prediction))}</strong></li> : null}</ul></GlassTooltip>;
}

function SegmentTooltip({ active, payload, label }: { active?: boolean; payload?: Array<{ value?: number }>; label?: string }) {
  if (!active || payload?.[0]?.value === undefined) return null;
  return <GlassTooltip><p className="glass-chart-tooltip__date">{label}</p><strong className="backtest-tooltip-value">{bdi(`${faOne.format(payload[0].value * 100)}٪`)}</strong></GlassTooltip>;
}

function SegmentChart({ rows, metric, labelOf, kind }: { rows: SegmentScore[]; metric: string; labelOf: (key: string | number) => string; kind: 'destination' | 'weekday' | 'month' | 'horizon' }) {
  const data = rows.map((row) => ({ key: labelOf(row.key), value: Number(row[metric] ?? 0) })).filter((row) => Number.isFinite(row.value));
  const worst = Math.max(...data.map((row) => row.value), 0); const destination = kind === 'destination';
  const title = kind === 'destination' ? 'خطا بر حسب مقصد' : kind === 'weekday' ? 'خطا بر حسب روز هفته' : kind === 'month' ? 'خطا بر حسب ماه' : 'خطا بر حسب گام افق';
  return <div className={`backtest-bar-scroll ${destination ? 'backtest-bar-scroll--destination' : ''}`}><div className={`chart-ltr backtest-bar-chart backtest-bar-chart--${kind}`} role="img" aria-label={title}><ResponsiveContainer width="100%" height="100%"><BarChart data={data} margin={{ top: 8, right: 8, bottom: 2, left: 0 }} barCategoryGap={kind === 'horizon' ? '10%' : '18%'}><defs><linearGradient id={`backtest-bar-${kind}`} x1="0" x2="0" y1="0" y2="1"><stop stopColor="var(--backtest-bar-start)" /><stop offset="1" stopColor="var(--backtest-bar-end)" /></linearGradient><linearGradient id={`backtest-hot-${kind}`} x1="0" x2="0" y1="0" y2="1"><stop stopColor="var(--backtest-hot-start)" /><stop offset="1" stopColor="var(--backtest-hot-end)" /></linearGradient></defs><CartesianGrid stroke={CHART.grid} strokeDasharray="2 4" vertical={false} /><XAxis dataKey="key" {...axisProps} interval={0} angle={destination ? -35 : 0} textAnchor={destination ? 'end' : 'middle'} height={destination ? 62 : 38} tick={{ fontSize: destination ? 9.5 : 10 }} /><YAxis {...axisProps} width={42} tickFormatter={(v: number) => `${fa.format(v * 100)}٪`} tick={{ fontSize: 10 }} /><Tooltip offset={12} content={<SegmentTooltip />} cursor={{ fill: 'rgb(99 102 241 / .05)' }} /><Bar dataKey="value" radius={[6, 6, 0, 0]} animationDuration={500}>{data.map((row, index) => <Cell key={row.key} className="backtest-bar" fill={`url(#${row.value >= worst * .9 ? 'backtest-hot' : 'backtest-bar'}-${kind})`} style={{ animationDelay: `${index * 20}ms` }} />)}</Bar></BarChart></ResponsiveContainer></div></div>;
}

export default function BacktestingPage() {
  const { level, entityId, horizon } = useForecastFilters();
  const metricsQuery = useQuery({ queryKey: ['backtest-metrics'], queryFn: api.backtestMetrics });
  const seriesQuery = useQuery({ queryKey: ['backtests', level, entityId], queryFn: () => api.backtests({ level, id: entityId }) });
  const timeseriesQuery = useQuery({ queryKey: ['timeseries', level, entityId, horizon], queryFn: () => api.timeseries({ level, id: entityId, horizon }) });
  const metrics = metricsQuery.data?.data; const series = seriesQuery.data?.data?.series ?? []; const metricName = metrics?.primary_metric ?? 'wape';
  if (metricsQuery.isSuccess && !metricsQuery.data?.available) return <Card><NoModelState detail={metricsQuery.data?.detailFa} /></Card>;
  return <main className="page-stack backtest-page">
    <PageHeader className="backtest-header backtest-enter" eyebrow="اعتماد به مدل" title="اعتبارسنجی گذشته‌نگر" description="عملکرد واقعی مدل را روی داده‌هایی که هرگز ندیده است، با اعتبارسنجی متحرک زمانی بررسی کنید." />
    {timeseriesQuery.data?.data ? <div className="backtest-enter"><FilterBar label="گزینه‌های تحلیل" memberLabel="مقصد" levels={timeseriesQuery.data.data.levels} members={timeseriesQuery.data.data.members} horizons={[7, 14, 30]} /></div> : null}
    <AsyncBoundary isLoading={metricsQuery.isLoading} error={metricsQuery.error} onRetry={() => metricsQuery.refetch()} skeleton={<Card><CardSkeleton lines={4} /></Card>}><Card className="backtest-card backtest-scheme backtest-enter"><CardHeader icon={<GitCompare className="h-4.5 w-4.5" />} title="طرح اعتبارسنجی" subtitle={<>مدل منتخب: {bdi(metrics?.champion ?? 'lightgbm', 'backtest-mono')}</>} action={<InfoHint>در هر تا (fold) مدل فقط با داده‌های قبل از نقطه برش آموزش می‌بیند و روی افق بعد از آن ارزیابی می‌شود. هیچ داده‌ای از آینده به مدل نشت نمی‌کند.</InfoHint>} /><CardBody><div className="backtest-folds">{(metrics?.folds ?? []).map((fold, index, all) => <div className="contents" key={fold.fold}><article className="backtest-fold"><p><CalendarDays aria-hidden="true" />تا {bdi(fa.format(fold.fold))}</p><span className="backtest-mono" dir="ltr">valid {fold.valid_start} → {fold.valid_end} · train ≤ {fold.train_end}</span></article>{index < all.length - 1 ? <span className="backtest-fold-connector" aria-hidden="true"><ChevronLeft /></span> : null}</div>)}</div></CardBody></Card></AsyncBoundary>
    <Card className="backtest-card backtest-enter"><CardHeader icon={<Activity className="h-4.5 w-4.5" />} title="مقدار واقعی در برابر پیش‌بینی" subtitle="روی پنجره‌های اعتبارسنجی که مدل هنگام آموزش ندیده بود" /><AsyncBoundary isLoading={seriesQuery.isLoading} error={seriesQuery.error} isEmpty={series.length === 0} onRetry={() => seriesQuery.refetch()} skeleton={<ChartSkeleton height={340} />}><CardBody><div className="chart-ltr backtest-comparison" role="img" aria-label="مقدار واقعی در برابر پیش‌بینی"><ResponsiveContainer width="100%" height="100%"><ComposedChart data={series} margin={{ top: 8, right: 12, bottom: 4, left: 4 }}><CartesianGrid stroke={CHART.grid} strokeDasharray="2 4" vertical={false} /><XAxis dataKey="ds" {...axisProps} tickFormatter={chartDate} minTickGap={28} tick={{ fontSize: 11 }} /><YAxis {...axisProps} width={48} tickFormatter={(v: number) => faCompact.format(v)} tick={{ fontSize: 11 }} /><Tooltip offset={12} content={<BacktestTooltip />} /><Legend verticalAlign="top" height={30} iconType="plainline" wrapperStyle={{ fontSize: 12, direction: 'rtl' }} /><Line type="monotone" dataKey="actual" name="واقعی" stroke={CHART.actual} strokeWidth={2.25} dot={false} animationDuration={900} /><Line className="backtest-prediction-line" type="monotone" dataKey="prediction" name="پیش‌بینی" stroke={CHART.forecast} strokeWidth={2.25} dot={false} animationDuration={900} /></ComposedChart></ResponsiveContainer></div></CardBody></AsyncBoundary></Card>
    <div className="grid gap-5 lg:grid-cols-2"><Card className="backtest-card backtest-enter"><CardHeader icon={<Target className="h-4.5 w-4.5" />} title="خطا بر حسب افق" subtitle="افت دقت مدل هرچه به آینده دورتر می‌رویم" /><CardBody><TableWrap className="backtest-table" minWidth={420}><thead><tr><Th scope="col">افق (روز)</Th><Th scope="col">WAPE</Th><Th scope="col">MAE</Th><Th scope="col">تعداد</Th></tr></thead><tbody>{(metrics?.by_horizon ?? []).map((row, index) => <tr key={row.bucket} style={{ '--row-index': index } as CSSProperties}><Td className="font-extrabold">{bdi(row.bucket)}</Td><Td className="font-extrabold">{bdi(metricFA(Number(row[metricName]), metricName))}</Td><Td className="text-muted">{bdi(fa.format(Number(row.mae)))}</Td><Td className="text-muted">{bdi(fa.format(row.n))}</Td></tr>)}</tbody></TableWrap></CardBody></Card><Card className="backtest-card backtest-enter"><CardHeader title="پوشش بازه اطمینان" subtitle={<>روش: {bdi(metrics?.uncertainty_method ?? 'native model quantiles', 'backtest-mono')}</>} action={<InfoHint>پوشش مشاهده‌شده = سهم مقادیر واقعی که داخل بازه پیش‌بینی قرار گرفته‌اند. هرچه به پوشش اسمی (۸۰٪) نزدیک‌تر باشد، بازه بهتر کالیبره شده است.</InfoHint>} /><CardBody><TableWrap className="backtest-table" minWidth={460}><thead><tr><Th scope="col">افق</Th><Th scope="col">پوشش اسمی</Th><Th scope="col">پوشش مشاهده‌شده</Th><Th scope="col">پهنای میانگین</Th></tr></thead><tbody>{(metrics?.coverage_by_horizon ?? []).map((row, index) => { const gap = Math.abs(row.observed_coverage - row.nominal_coverage); const tone = gap <= .05 ? 'success' : gap <= .1 ? 'warning' : 'danger'; return <tr key={row.bucket} style={{ '--row-index': index } as CSSProperties}><Td className="font-extrabold">{bdi(row.bucket)}</Td><Td className="text-muted">{bdi(percent(row.nominal_coverage, 0))}</Td><Td><Badge tone={tone} className="backtest-coverage"><i />{bdi(percent(row.observed_coverage))}</Badge></Td><Td className="text-muted">{bdi(fa.format(row.mean_width))}</Td></tr>; })}</tbody></TableWrap></CardBody></Card></div>
    <div className="grid gap-5 lg:grid-cols-2">{metrics?.segments?.by_destination?.length ? <Card className="backtest-card backtest-enter"><CardHeader title="خطا بر حسب مقصد" subtitle="کدام مقاصد سخت‌تر پیش‌بینی می‌شوند؟" /><CardBody><SegmentChart rows={metrics.segments.by_destination} metric={metricName} labelOf={(key) => String(key)} kind="destination" /></CardBody></Card> : null}{metrics?.segments?.by_weekday?.length ? <Card className="backtest-card backtest-enter"><CardHeader title="خطا بر حسب روز هفته" /><CardBody><SegmentChart rows={metrics.segments.by_weekday} metric={metricName} labelOf={(key) => WEEKDAY_FA[Number(key)] ?? String(key)} kind="weekday" /></CardBody></Card> : null}{metrics?.segments?.by_horizon?.length ? <Card className="backtest-card backtest-enter"><CardHeader title="خطا بر حسب گام افق" subtitle="روز به روز" /><CardBody><SegmentChart rows={metrics.segments.by_horizon} metric={metricName} labelOf={(key) => fa.format(Number(key))} kind="horizon" /></CardBody></Card> : null}{metrics?.segments?.by_month?.length ? <Card className="backtest-card backtest-enter"><CardHeader title="خطا بر حسب ماه" /><CardBody><SegmentChart rows={metrics.segments.by_month} metric={metricName} labelOf={(key) => MONTH_FA[Number(key) - 1] ?? String(key)} kind="month" /></CardBody></Card> : null}</div>
  </main>;
}
