'use client';

import { useQuery } from '@tanstack/react-query';
import { useState } from 'react';
import {
  Area,
  Bar,
  BarChart,
  CartesianGrid,
  ComposedChart,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

import { ChartFrame } from '@/components/pol4/chart-frame';
import { Kpi, KpiRow } from '@/components/pol4/kpi';
import { ChartTooltip } from '@/components/pol4/tooltip';
import { SERIES, axisProps, compact, heatStep, jalali, num, percent, ratio } from '@/components/pol4/theme';
import { AsyncBoundary, CardSkeleton, ChartSkeleton, NoModelState } from '@/components/ui/states';
import { pol4 } from '@/lib/pol4/api';

const TOP_N_CHOICES = [10, 20, 50];

export default function OverviewPage() {
  const [topN, setTopN] = useState(20);
  const overviewQuery = useQuery({ queryKey: ['pol4-overview'], queryFn: pol4.overview });
  const heatmapQuery = useQuery({
    queryKey: ['pol4-heatmap', topN],
    queryFn: () => pol4.heatmap(topN),
  });

  const data = overviewQuery.data?.data;

  if (overviewQuery.isSuccess && !overviewQuery.data?.available) {
    return <NoModelState detail={overviewQuery.data?.detailFa} />;
  }

  return (
    <div className="pol4-page">
      <header className="pol4-page__head">
        <h1>تقاضای اقامت در آذر ۱۴۰۴</h1>
        <p>
          پیش‌بینی تقاضای جست‌وجو برای هر شهر و هر شب اقامت، با داده‌های موجود تا{' '}
          <strong dir="ltr">{data?.cutoff ?? '—'}</strong> (۳۰ آذر ۱۴۰۴).
        </p>
        <p className="pol4-page__method">
          تقاضای واقعی کاربر مستقیماً قابل مشاهده نیست. آنچه اندازه می‌گیریم جست‌وجوی
          کاربران است؛ یعنی یک <strong>شاخص جایگزین قابل مشاهده</strong> برای قصد سفر،
          نه خودِ قصد سفر.
        </p>
      </header>

      <AsyncBoundary
        isLoading={overviewQuery.isLoading}
        error={overviewQuery.error}
        onRetry={() => overviewQuery.refetch()}
        skeleton={<CardSkeleton lines={3} />}
      >
        {data ? (
          <KpiRow>
            <Kpi
              label="کل تقاضای پیش‌بینی‌شده"
              value={num(data.kpis.total_predicted_demand)}
              note={`${num(data.kpis.observed_so_far)} تا کنون ثبت شده`}
            />
            <Kpi
              label="پرتقاضاترین شب"
              value={jalali(data.kpis.peak_date)}
              note={`${num(data.kpis.peak_demand)} جست‌وجو`}
            />
            <Kpi
              label="پرتقاضاترین شهر"
              value={<span dir="ltr">{data.kpis.top_city}</span>}
              note={`${num(data.kpis.top_city_demand)} جست‌وجو`}
            />
            <Kpi
              label="سریع‌ترین رشد"
              value={<span dir="ltr">{data.kpis.fastest_pickup_city ?? '—'}</span>}
              note={
                data.kpis.fastest_pickup_ratio
                  ? `${ratio(data.kpis.fastest_pickup_ratio)}× سریع‌تر از روند معمول خودش`
                  : undefined
              }
              tone="good"
            />
            <Kpi
              label="خطای مدل (WAPE)"
              value={percent(data.kpis.backtest_wape)}
              note={`مبنا: ${percent(data.kpis.baseline_wape)}`}
            />
            <Kpi
              label="پایداری پیش‌بینی"
              value={data.kpis.stability_score === null ? '—' : ratio(data.kpis.stability_score)}
              note="۱ یعنی پیش‌بینی اصلاً تغییر نمی‌کند"
            />
          </KpiRow>
        ) : null}
      </AsyncBoundary>

      {/* ------------------------------------------- national demand forecast */}
      <AsyncBoundary
        isLoading={overviewQuery.isLoading}
        error={overviewQuery.error}
        skeleton={<ChartSkeleton height={340} />}
      >
        {data ? (
          <ChartFrame
            title="تقاضای کل کشور در ۳۰ شب آذر"
            unit="تعداد جست‌وجو"
            hint="ستون‌ها آنچه تا امروز ثبت شده و آنچه مدل انتظار دارد هنوز برسد را از هم جدا می‌کنند."
            source="results.csv"
            height={340}
            legend={[
              { label: 'ثبت‌شده تا کنون', color: SERIES.observed },
              { label: 'باقی‌مانده پیش‌بینی‌شده', color: SERIES.forecast },
              { label: 'مجموع پیش‌بینی‌شده', color: SERIES.ink },
            ]}
          >
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={data.series} margin={{ top: 8, right: 12, bottom: 4, left: 4 }}>
                <CartesianGrid stroke={SERIES.grid} strokeDasharray="2 4" vertical={false} />
                <XAxis dataKey="checkin" {...axisProps} tickFormatter={jalali} minTickGap={18} />
                <YAxis {...axisProps} width={52} tickFormatter={compact} />
                <Tooltip
                  content={
                    <ChartTooltip
                      labelFormatter={jalali}
                      order={['observed_so_far', 'predicted_remaining', 'predicted_demand']}
                    />
                  }
                  cursor={{ fill: 'rgb(99 102 241 / .06)' }}
                />
                <Bar
                  dataKey="observed_so_far"
                  name="ثبت‌شده تا کنون"
                  stackId="demand"
                  fill={SERIES.observed}
                  radius={[0, 0, 0, 0]}
                />
                <Bar
                  dataKey="predicted_remaining"
                  name="باقی‌مانده پیش‌بینی‌شده"
                  stackId="demand"
                  fill={SERIES.forecast}
                  radius={[4, 4, 0, 0]}
                />
                <Line
                  dataKey="predicted_demand"
                  name="مجموع پیش‌بینی‌شده"
                  stroke={SERIES.ink}
                  strokeWidth={2}
                  dot={false}
                  activeDot={{ r: 4 }}
                />
              </ComposedChart>
            </ResponsiveContainer>
          </ChartFrame>
        ) : null}
      </AsyncBoundary>

      {/* -------------------------------------------------- city x date heatmap */}
      <AsyncBoundary
        isLoading={heatmapQuery.isLoading}
        error={heatmapQuery.error}
        onRetry={() => heatmapQuery.refetch()}
        skeleton={<ChartSkeleton height={420} />}
      >
        {heatmapQuery.data?.data ? (
          <ChartFrame
            title="نقشه حرارتی شهر × شب"
            unit="تعداد جست‌وجو"
            hint="هر خانه یک شهر در یک شب است؛ رنگ پررنگ‌تر یعنی تقاضای بیشتر."
            source="results.csv"
            height={Math.min(560, 60 + heatmapQuery.data.data.rows.length * 22)}
            action={
              <div className="pol4-toggle" role="group" aria-label="تعداد شهرها">
                {TOP_N_CHOICES.map((choice) => (
                  <button
                    key={choice}
                    type="button"
                    aria-pressed={choice === topN}
                    className={choice === topN ? 'is-active' : undefined}
                    onClick={() => setTopN(choice)}
                  >
                    {num(choice)}
                  </button>
                ))}
              </div>
            }
          >
            <Heatmap data={heatmapQuery.data.data} />
          </ChartFrame>
        ) : null}
      </AsyncBoundary>

      <div className="pol4-grid-2">
        {/* ------------------------------------------------------- top cities */}
        <AsyncBoundary isLoading={overviewQuery.isLoading} skeleton={<CardSkeleton lines={8} />}>
          {data ? (
            <ChartFrame
              title="پرتقاضاترین شهرها"
              unit="تعداد جست‌وجو"
              hint="مجموع ۳۰ شب آذر برای هر شهر."
              source="results_named.csv"
              height={Math.max(320, data.top_cities.length * 26)}
              legend={[
                { label: 'ثبت‌شده تا کنون', color: SERIES.observed },
                { label: 'باقی‌مانده پیش‌بینی‌شده', color: SERIES.forecast },
              ]}
            >
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  layout="vertical"
                  data={data.top_cities}
                  margin={{ top: 4, right: 56, bottom: 4, left: 4 }}
                  barCategoryGap={4}
                >
                  <CartesianGrid stroke={SERIES.grid} strokeDasharray="2 4" horizontal={false} />
                  <XAxis type="number" {...axisProps} tickFormatter={compact} />
                  <YAxis
                    type="category"
                    dataKey="city"
                    {...axisProps}
                    width={110}
                    tick={{ fontSize: 11 }}
                  />
                  <Tooltip
                    content={<ChartTooltip order={['observed_so_far', 'predicted_remaining']} />}
                    cursor={{ fill: 'rgb(99 102 241 / .06)' }}
                  />
                  <Bar
                    dataKey="observed_so_far"
                    name="ثبت‌شده تا کنون"
                    stackId="city"
                    fill={SERIES.observed}
                  />
                  <Bar
                    dataKey="predicted_remaining"
                    name="باقی‌مانده پیش‌بینی‌شده"
                    stackId="city"
                    fill={SERIES.forecast}
                    radius={[0, 4, 4, 0]}
                  />
                </BarChart>
              </ResponsiveContainer>
            </ChartFrame>
          ) : null}
        </AsyncBoundary>

        {/* -------------------------------------------------- province split */}
        <AsyncBoundary isLoading={overviewQuery.isLoading} skeleton={<CardSkeleton lines={8} />}>
          {data ? (
            <ChartFrame
              title="توزیع تقاضا بر حسب استان"
              unit="تعداد جست‌وجو"
              hint="هفت استان موجود در دادهٔ مسابقه."
              source="province_summary.json"
              height={320}
            >
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  layout="vertical"
                  data={data.provinces}
                  margin={{ top: 4, right: 56, bottom: 4, left: 4 }}
                  barCategoryGap={8}
                >
                  <CartesianGrid stroke={SERIES.grid} strokeDasharray="2 4" horizontal={false} />
                  <XAxis type="number" {...axisProps} tickFormatter={compact} />
                  <YAxis
                    type="category"
                    dataKey="province"
                    {...axisProps}
                    width={110}
                    tick={{ fontSize: 11 }}
                  />
                  <Tooltip
                    content={<ChartTooltip />}
                    cursor={{ fill: 'rgb(99 102 241 / .06)' }}
                  />
                  <Bar
                    dataKey="predicted_demand"
                    name="تقاضای پیش‌بینی‌شده"
                    fill={SERIES.forecast}
                    radius={[0, 4, 4, 0]}
                  />
                </BarChart>
              </ResponsiveContainer>
            </ChartFrame>
          ) : null}
        </AsyncBoundary>
      </div>

      {/* ----------------------------------------------------- emerging demand */}
      <AsyncBoundary isLoading={overviewQuery.isLoading} skeleton={<CardSkeleton lines={6} />}>
        {data ? (
          <section className="pol4-card">
            <header className="pol4-card__head">
              <div className="pol4-card__titles">
                <h2 className="pol4-card__title">تقاضای در حال شکل‌گیری</h2>
                <p className="pol4-card__hint">
                  شهرهایی که جست‌وجو برایشان در هفتهٔ گذشته سریع‌تر از روند تاریخی
                  خودشان رسیده است. این نسبت اندازه‌گیری شده است و دربارهٔ علت آن
                  ادعایی نمی‌کند.
                </p>
              </div>
            </header>
            <div className="pol4-table-wrap">
              <table className="pol4-table">
                <thead>
                  <tr>
                    <th>شهر</th>
                    <th>استان</th>
                    <th>ثبت‌شده تا کنون</th>
                    <th>پیکاپ ۷ روز اخیر</th>
                    <th>انتظار تاریخی</th>
                    <th>نسبت</th>
                  </tr>
                </thead>
                <tbody>
                  {data.momentum.map((row) => (
                    <tr key={row.city_code}>
                      <td dir="ltr">{row.city}</td>
                      <td dir="ltr">{row.province}</td>
                      <td dir="ltr">{num(row.observed)}</td>
                      <td dir="ltr">{num(row.recent_pickup)}</td>
                      <td dir="ltr">{num(row.expected_pickup)}</td>
                      <td dir="ltr">
                        <strong>{ratio(row.pickup_ratio)}×</strong>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="pol4-card__source">منبع: city_momentum.parquet</p>
          </section>
        ) : null}
      </AsyncBoundary>
    </div>
  );
}

/** City × date grid. A table, so screen readers and keyboards get the data too. */
function Heatmap({ data }: { data: { dates: string[]; rows: { city_code: number; city: string; province: string; values: number[] }[]; max: number } }) {
  return (
    <div className="pol4-heatmap" dir="rtl">
      <table>
        <caption className="sr-only">تقاضای پیش‌بینی‌شده برای هر شهر در هر شب آذر</caption>
        <thead>
          <tr>
            <th scope="col">شهر</th>
            {data.dates.map((date) => (
              <th key={date} scope="col" title={jalali(date)}>
                <span aria-hidden="true">{jalali(date).split(' ')[0]}</span>
                <span className="sr-only">{jalali(date)}</span>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.rows.map((row) => (
            <tr key={row.city_code}>
              <th scope="row" dir="ltr">
                {row.city}
              </th>
              {row.values.map((value, index) => (
                <td
                  key={data.dates[index]}
                  style={{ background: heatStep(value, data.max) }}
                  title={`${row.city} — ${jalali(data.dates[index])}: ${num(value)}`}
                >
                  <span className="sr-only">{num(value)}</span>
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
