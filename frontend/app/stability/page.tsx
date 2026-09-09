'use client';

import { useQuery } from '@tanstack/react-query';
import { useEffect, useState } from 'react';
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

import { ChartFrame } from '@/components/pol4/chart-frame';
import { CityPicker } from '@/components/pol4/city-picker';
import { Kpi, KpiRow } from '@/components/pol4/kpi';
import { SERIES, axisProps, compact, jalaliFull, num, percent, ratio } from '@/components/pol4/theme';
import { ChartTooltip } from '@/components/pol4/tooltip';
import { AsyncBoundary, CardSkeleton, ChartSkeleton, NoModelState } from '@/components/ui/states';
import { pol4 } from '@/lib/pol4/api';

export default function StabilityPage() {
  const [cityCode, setCityCode] = useState<number | null>(null);
  const [checkin, setCheckin] = useState<string | undefined>();

  const citiesQuery = useQuery({ queryKey: ['pol4-cities'], queryFn: pol4.cities });
  const stabilityQuery = useQuery({
    queryKey: ['pol4-stability', cityCode, checkin],
    queryFn: () => pol4.stability(cityCode ?? undefined, checkin),
  });

  const cities = citiesQuery.data?.data?.cities ?? [];
  const data = stabilityQuery.data?.data;

  // The API picks the most consequential pair when nothing is selected, so the
  // page opens on a case worth reading rather than an arbitrary one.
  useEffect(() => {
    if (cityCode === null && data) setCityCode(data.city.city_code);
  }, [data, cityCode]);

  if (stabilityQuery.isSuccess && !stabilityQuery.data?.available) {
    return <NoModelState detail={stabilityQuery.data?.detailFa} />;
  }

  return (
    <div className="pol4-page pol4-page--split">
      <aside className="pol4-side">
        <AsyncBoundary isLoading={citiesQuery.isLoading} skeleton={<CardSkeleton lines={10} />}>
          <CityPicker
            cities={cities}
            value={cityCode}
            onChange={(code) => {
              setCityCode(code);
              setCheckin(undefined);
            }}
          />
        </AsyncBoundary>
      </aside>

      <div className="pol4-main">
        <header className="pol4-page__head">
          <h1>پایداری پیش‌بینی</h1>
          <p>
            یک پیش‌بینی که هر هفته زیر و رو می‌شود، حتی اگر در پایان درست دربیاید،
            قابل برنامه‌ریزی نیست. اینجا برای یک شهر و یک شب مشخص، پیش‌بینی را در
            شش نقطهٔ زمانی پیش از اقامت دنبال می‌کنیم.
          </p>
          {data?.window?.target_start ? (
            <p className="pol4-page__method">
              پنجرهٔ تاریخی: شب‌های اقامت از{' '}
              <span dir="ltr">{data.window.target_start}</span>؛ همهٔ عکس‌ها با مدلی
              گرفته شده‌اند که در <span dir="ltr">{data.window.anchor_cutoff}</span>{' '}
              آموزش دیده — یعنی هیچ عکسی چیزی بیش از آنچه مجاز بوده نمی‌بیند.
            </p>
          ) : null}
        </header>

        <AsyncBoundary
          isLoading={stabilityQuery.isLoading}
          error={stabilityQuery.error}
          onRetry={() => stabilityQuery.refetch()}
          skeleton={<CardSkeleton lines={3} />}
        >
          {data ? (
            <KpiRow>
              <Kpi
                label="امتیاز پایداری"
                value={data.aggregate.stability_score === null ? '—' : ratio(data.aggregate.stability_score)}
                note="۱ یعنی پیش‌بینی اصلاً تغییر نمی‌کند"
              />
              <Kpi
                label="میانگین بازنگری نسبی"
                value={
                  data.aggregate.mean_relative_revision === null
                    ? '—'
                    : percent(data.aggregate.mean_relative_revision)
                }
                note="بین دو عکس متوالی"
              />
              <Kpi
                label="نرخ هم‌گرایی"
                value={
                  data.aggregate.convergence_rate === null
                    ? '—'
                    : percent(data.aggregate.convergence_rate)
                }
                note="سهم بازنگری‌هایی که به واقعیت نزدیک‌تر می‌شوند"
                tone="good"
              />
              <Kpi
                label="مقدار واقعی این مورد"
                value={num(data.actual)}
                note={`${data.city.city} — ${jalaliFull(data.checkin)}`}
              />
            </KpiRow>
          ) : null}
        </AsyncBoundary>

        {/* ------------------------------------------- the snapshot ladder */}
        <AsyncBoundary isLoading={stabilityQuery.isLoading} skeleton={<ChartSkeleton height={340} />}>
          {data ? (
            <ChartFrame
              title="تحول پیش‌بینی تا شب اقامت"
              unit="تعداد جست‌وجو"
              hint="هر نقطه یک پیش‌بینی است که در آن فاصله از شب اقامت گرفته شده. خط افقی مقدار واقعی است."
              source="stability.parquet"
              height={340}
              legend={[
                { label: 'پیش‌بینی', color: SERIES.forecast },
                { label: 'ثبت‌شده تا آن لحظه', color: SERIES.observed },
                { label: 'مقدار واقعی', color: SERIES.ink },
              ]}
              action={
                <select
                  className="pol4-select"
                  value={data.checkin}
                  onChange={(event) => setCheckin(event.target.value)}
                  aria-label="انتخاب شب اقامت"
                >
                  {data.available_checkins.map((date) => (
                    <option key={date} value={date}>
                      {jalaliFull(date)}
                    </option>
                  ))}
                </select>
              }
            >
              <ResponsiveContainer width="100%" height="100%">
                <LineChart
                  data={[...data.snapshots].sort((a, b) => b.horizon - a.horizon)}
                  margin={{ top: 8, right: 16, bottom: 4, left: 4 }}
                >
                  <CartesianGrid stroke={SERIES.grid} strokeDasharray="2 4" vertical={false} />
                  {/* Sorted D-30 first, so the story reads left to right: the
                      forecast tightening as the stay approaches. */}
                  <XAxis
                    dataKey="horizon"
                    {...axisProps}
                    tickFormatter={(value: number) => `D-${value}`}
                  />
                  <YAxis {...axisProps} width={52} tickFormatter={compact} />
                  <Tooltip
                    content={
                      <ChartTooltip
                        labelFormatter={(value) => `${value} روز مانده تا اقامت`}
                        order={['prediction', 'observed']}
                      />
                    }
                  />
                  <ReferenceLine
                    y={data.actual}
                    stroke={SERIES.ink}
                    strokeDasharray="4 4"
                    label={{ value: 'واقعی', position: 'insideTopRight', fontSize: 10 }}
                  />
                  <Line
                    dataKey="prediction"
                    name="پیش‌بینی"
                    stroke={SERIES.forecast}
                    strokeWidth={2}
                    dot={{ r: 4 }}
                    activeDot={{ r: 6 }}
                  />
                  <Line
                    dataKey="observed"
                    name="ثبت‌شده تا آن لحظه"
                    stroke={SERIES.observed}
                    strokeWidth={2}
                    strokeDasharray="4 4"
                    dot={{ r: 3 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </ChartFrame>
          ) : null}
        </AsyncBoundary>

        <div className="pol4-grid-2">
          {/* --------------------------------------------- revision by step */}
          <AsyncBoundary isLoading={stabilityQuery.isLoading} skeleton={<CardSkeleton lines={7} />}>
            {data ? (
              <section className="pol4-card">
                <header className="pol4-card__head">
                  <div className="pol4-card__titles">
                    <h2 className="pol4-card__title">بازنگری در هر گام</h2>
                    <p className="pol4-card__hint">
                      روی همهٔ ۹٬۶۳۰ جفت شهر/شب پنجرهٔ تاریخی — نه فقط مورد انتخاب‌شده.
                    </p>
                  </div>
                </header>
                <div className="pol4-table-wrap">
                  <table className="pol4-table">
                    <thead>
                      <tr>
                        <th>گام</th>
                        <th>بازنگری نسبی</th>
                        <th>هم‌گرایی</th>
                      </tr>
                    </thead>
                    <tbody>
                      {data.aggregate.by_step.map((step) => (
                        <tr key={step.step}>
                          <td dir="ltr">D-{step.step.replace('->', ' → D-')}</td>
                          <td dir="ltr">{percent(step.mean_relative_revision)}</td>
                          <td dir="ltr">{percent(step.convergence_rate)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <p className="pol4-card__source">منبع: run_summary.json → stability</p>
              </section>
            ) : null}
          </AsyncBoundary>

          {/* ------------------------------------------ accuracy by horizon */}
          <AsyncBoundary isLoading={stabilityQuery.isLoading} skeleton={<ChartSkeleton height={280} />}>
            {data ? (
              <ChartFrame
                title="دقت در هر فاصله از اقامت"
                unit="WAPE"
                hint="هرچه به شب اقامت نزدیک‌تر می‌شویم، خطا کمتر می‌شود."
                source="run_summary.json → stability"
                height={280}
              >
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart
                    data={[...data.aggregate.by_horizon].sort((a, b) => b.horizon - a.horizon)}
                    margin={{ top: 8, right: 16, bottom: 4, left: 4 }}
                  >
                    <CartesianGrid stroke={SERIES.grid} strokeDasharray="2 4" vertical={false} />
                    <XAxis
                      dataKey="horizon"
                      {...axisProps}
                      tickFormatter={(value: number) => `D-${value}`}
                    />
                    <YAxis
                      {...axisProps}
                      width={52}
                      tickFormatter={(value: number) => percent(value, 0)}
                    />
                    <Tooltip
                      content={
                        <ChartTooltip
                          labelFormatter={(value) => `${value} روز مانده`}
                          valueFormatter={(value) => percent(value)}
                        />
                      }
                    />
                    <Line
                      dataKey="wape"
                      name="WAPE"
                      stroke={SERIES.forecast}
                      strokeWidth={2}
                      dot={{ r: 4 }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </ChartFrame>
            ) : null}
          </AsyncBoundary>
        </div>
      </div>
    </div>
  );
}
