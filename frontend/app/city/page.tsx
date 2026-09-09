'use client';

import { useQuery } from '@tanstack/react-query';
import { useEffect, useMemo, useState } from 'react';
import {
  Area,
  AreaChart,
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
import { CityPicker } from '@/components/pol4/city-picker';
import { Kpi, KpiRow } from '@/components/pol4/kpi';
import { SERIES, axisProps, compact, jalali, jalaliFull, num, ratio } from '@/components/pol4/theme';
import { ChartTooltip } from '@/components/pol4/tooltip';
import { AsyncBoundary, CardSkeleton, ChartSkeleton, NoModelState } from '@/components/ui/states';
import { pol4 } from '@/lib/pol4/api';

export default function CityAnalysisPage() {
  const [cityCode, setCityCode] = useState<number | null>(null);
  const [checkin, setCheckin] = useState<string | undefined>();

  const citiesQuery = useQuery({ queryKey: ['pol4-cities'], queryFn: pol4.cities });
  // Memoised so the fallback array does not become a new dependency each render.
  const cities = useMemo(() => citiesQuery.data?.data?.cities ?? [], [citiesQuery.data]);

  useEffect(() => {
    if (cityCode === null && cities.length) setCityCode(cities[0].city_code);
  }, [cities, cityCode]);

  const detailQuery = useQuery({
    queryKey: ['pol4-city', cityCode],
    queryFn: () => pol4.city(cityCode as number),
    enabled: cityCode !== null,
  });
  const pickupQuery = useQuery({
    queryKey: ['pol4-pickup', cityCode, checkin],
    queryFn: () => pol4.pickup(cityCode as number, checkin),
    enabled: cityCode !== null,
  });

  const detail = detailQuery.data?.data;
  const pickup = pickupQuery.data?.data;

  if (citiesQuery.isSuccess && !citiesQuery.data?.available) {
    return <NoModelState detail={citiesQuery.data?.detailFa} />;
  }

  return (
    <div className="pol4-page pol4-page--split">
      <aside className="pol4-side">
        <AsyncBoundary isLoading={citiesQuery.isLoading} skeleton={<CardSkeleton lines={10} />}>
          <CityPicker cities={cities} value={cityCode} onChange={setCityCode} />
        </AsyncBoundary>
      </aside>

      <div className="pol4-main">
        <AsyncBoundary
          isLoading={detailQuery.isLoading}
          error={detailQuery.error}
          onRetry={() => detailQuery.refetch()}
          skeleton={<CardSkeleton lines={3} />}
        >
          {detail ? (
            <>
              <header className="pol4-page__head">
                <h1 dir="ltr">{detail.city.city}</h1>
                <p>
                  استان <span dir="ltr">{detail.city.province}</span> · شناسهٔ مسابقه{' '}
                  <span dir="ltr">{detail.city.city_code}</span>
                </p>
              </header>

              <KpiRow>
                <Kpi
                  label="تقاضای پیش‌بینی‌شدهٔ آذر"
                  value={num(detail.totals.predicted_demand)}
                  note="مجموع ۳۰ شب"
                />
                <Kpi
                  label="ثبت‌شده تا کنون"
                  value={num(detail.totals.observed_so_far)}
                  note="از evaluation.csv"
                />
                <Kpi
                  label="باقی‌مانده پیش‌بینی‌شده"
                  value={num(detail.totals.predicted_remaining)}
                  note="آنچه مدل انتظار دارد هنوز برسد"
                />
                <Kpi
                  label="سرعت رشد"
                  value={detail.momentum ? `${ratio(detail.momentum.pickup_ratio)}×` : '—'}
                  note="۷ روز اخیر نسبت به روند تاریخی همین شهر"
                  tone={detail.momentum && detail.momentum.pickup_ratio > 1.1 ? 'good' : 'default'}
                />
              </KpiRow>
            </>
          ) : null}
        </AsyncBoundary>

        {/* ------------------------------------------- 30-day city forecast */}
        <AsyncBoundary isLoading={detailQuery.isLoading} skeleton={<ChartSkeleton height={300} />}>
          {detail ? (
            <ChartFrame
              title="پیش‌بینی ۳۰ شب آذر"
              unit="تعداد جست‌وجو"
              hint="ثبت‌شده و باقی‌مانده روی هم، برای هر شب اقامت."
              source="results.csv"
              height={300}
              legend={[
                { label: 'ثبت‌شده تا کنون', color: SERIES.observed },
                { label: 'باقی‌مانده پیش‌بینی‌شده', color: SERIES.forecast },
              ]}
            >
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={detail.series} margin={{ top: 8, right: 12, bottom: 4, left: 4 }}>
                  <CartesianGrid stroke={SERIES.grid} strokeDasharray="2 4" vertical={false} />
                  <XAxis dataKey="checkin" {...axisProps} tickFormatter={jalali} minTickGap={16} />
                  <YAxis {...axisProps} width={52} tickFormatter={compact} />
                  <Tooltip
                    content={
                      <ChartTooltip
                        labelFormatter={jalaliFull}
                        order={['observed_so_far', 'predicted_remaining']}
                      />
                    }
                    cursor={{ fill: 'rgb(99 102 241 / .06)' }}
                  />
                  <Bar
                    dataKey="observed_so_far"
                    name="ثبت‌شده تا کنون"
                    stackId="d"
                    fill={SERIES.observed}
                  />
                  <Bar
                    dataKey="predicted_remaining"
                    name="باقی‌مانده پیش‌بینی‌شده"
                    stackId="d"
                    fill={SERIES.forecast}
                    radius={[4, 4, 0, 0]}
                  />
                </BarChart>
              </ResponsiveContainer>
            </ChartFrame>
          ) : null}
        </AsyncBoundary>

        {/* ------------------------------------------------------ pickup curve */}
        <AsyncBoundary
          isLoading={pickupQuery.isLoading}
          error={pickupQuery.error}
          onRetry={() => pickupQuery.refetch()}
          skeleton={<ChartSkeleton height={320} />}
        >
          {pickup ? (
            <ChartFrame
              title={`منحنی پیکاپ — شب ${jalaliFull(pickup.checkin)}`}
              unit="تجمعی تعداد جست‌وجو"
              hint="محور افقی از ۵۹ روز مانده (چپ) تا شب اقامت (راست) پیش می‌رود. خط پُر آنچه واقعاً ثبت شده و خط‌چین مسیری است که تاریخِ همین شهر معمولاً طی می‌کند؛ پایین‌تر بودن یعنی این شب کندتر از روال خودش پُر می‌شود."
              source={`target_pickup.parquet · pickup_curves.parquet (سطح منحنی: ${pickup.curve_level})`}
              height={320}
              legend={[
                { label: 'ثبت‌شدهٔ تجمعی', color: SERIES.observed },
                { label: 'مسیر معمول تاریخی', color: SERIES.expected },
              ]}
              action={
                <select
                  className="pol4-select"
                  value={pickup.checkin}
                  onChange={(event) => setCheckin(event.target.value)}
                  aria-label="انتخاب شب اقامت"
                >
                  {pickup.available_checkins.map((date) => (
                    <option key={date} value={date}>
                      {jalaliFull(date)}
                    </option>
                  ))}
                </select>
              }
            >
              <PickupChart pickup={pickup} />
            </ChartFrame>
          ) : null}
        </AsyncBoundary>

        <div className="pol4-grid-2">
          {/* ------------------------------------------------------ peak dates */}
          <AsyncBoundary isLoading={detailQuery.isLoading} skeleton={<CardSkeleton lines={6} />}>
            {detail ? (
              <section className="pol4-card">
                <header className="pol4-card__head">
                  <div className="pol4-card__titles">
                    <h2 className="pol4-card__title">پرتقاضاترین شب‌ها</h2>
                    <p className="pol4-card__hint">پنج شب برتر این شهر در آذر.</p>
                  </div>
                </header>
                <div className="pol4-table-wrap">
                  <table className="pol4-table">
                    <thead>
                      <tr>
                        <th>شب اقامت</th>
                        <th>پیش‌بینی</th>
                        <th>ثبت‌شده</th>
                      </tr>
                    </thead>
                    <tbody>
                      {detail.peaks.map((peak) => (
                        <tr key={peak.checkin}>
                          <td>{jalaliFull(peak.checkin)}</td>
                          <td dir="ltr">
                            <strong>{num(peak.predicted_demand)}</strong>
                          </td>
                          <td dir="ltr">{num(peak.observed_so_far)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <p className="pol4-card__source">منبع: results.csv</p>
              </section>
            ) : null}
          </AsyncBoundary>

          {/* -------------------------------------------------- recent history */}
          <AsyncBoundary isLoading={detailQuery.isLoading} skeleton={<ChartSkeleton height={280} />}>
            {detail ? (
              <ChartFrame
                title="تقاضای تاریخی این شهر"
                unit="تعداد جست‌وجو در هر شب"
                hint="۹۰ شب کامل‌شده پیش از تاریخ برش، برای مقایسه با پیش‌بینی بالا."
                source="city_history.parquet"
                height={280}
              >
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={detail.history} margin={{ top: 8, right: 12, bottom: 4, left: 4 }}>
                    <defs>
                      <linearGradient id="pol4-history" x1="0" x2="0" y1="0" y2="1">
                        <stop offset="0%" stopColor={SERIES.expected} stopOpacity={0.35} />
                        <stop offset="100%" stopColor={SERIES.expected} stopOpacity={0.02} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid stroke={SERIES.grid} strokeDasharray="2 4" vertical={false} />
                    <XAxis dataKey="checkin" {...axisProps} tickFormatter={jalali} minTickGap={28} />
                    <YAxis {...axisProps} width={52} tickFormatter={compact} />
                    <Tooltip content={<ChartTooltip labelFormatter={jalaliFull} />} />
                    <Area
                      dataKey="demand"
                      name="تقاضای کامل‌شده"
                      stroke={SERIES.expected}
                      strokeWidth={2}
                      fill="url(#pol4-history)"
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </ChartFrame>
            ) : null}
          </AsyncBoundary>
        </div>
      </div>
    </div>
  );
}

/**
 * Observed accumulation against the historical expectation.
 *
 * Both series are joined on days-to-check-in and ordered far-out first, so the
 * axis runs toward the stay - the direction demand actually accumulates.
 */
function PickupChart({ pickup }: { pickup: import('@/lib/pol4/types').PickupCurve }) {
  const byLead = new Map<number, { days: number; observed?: number; expected?: number }>();
  pickup.expected.forEach((row) => {
    byLead.set(row.days_to_checkin, {
      days: row.days_to_checkin,
      expected: Math.round(row.expected_cumulative),
    });
  });
  pickup.observed.forEach((row) => {
    const entry = byLead.get(row.days_to_checkin) ?? { days: row.days_to_checkin };
    entry.observed = row.observed_cumulative;
    byLead.set(row.days_to_checkin, entry);
  });

  const rows = [...byLead.values()]
    .filter((row) => row.days >= pickup.horizon)
    .sort((a, b) => b.days - a.days);

  return (
    <ResponsiveContainer width="100%" height="100%">
      <ComposedChart data={rows} margin={{ top: 8, right: 12, bottom: 4, left: 4 }}>
        <CartesianGrid stroke={SERIES.grid} strokeDasharray="2 4" vertical={false} />
        {/* Rows are already ordered 59 -> 0, so the axis runs from far-out on the
            left to the stay on the right: the direction demand accumulates. */}
        <XAxis
          dataKey="days"
          {...axisProps}
          tickFormatter={(value: number) => `${value}`}
          label={{ value: 'روز مانده تا اقامت', position: 'insideBottom', offset: -2, fontSize: 10 }}
        />
        <YAxis {...axisProps} width={52} tickFormatter={compact} />
        <Tooltip
          content={
            <ChartTooltip
              labelFormatter={(value) => `${value} روز مانده`}
              order={['observed', 'expected']}
            />
          }
        />
        <Line
          dataKey="expected"
          name="مسیر معمول تاریخی"
          stroke={SERIES.expected}
          strokeWidth={2}
          strokeDasharray="4 4"
          dot={false}
        />
        <Line
          dataKey="observed"
          name="ثبت‌شدهٔ تجمعی"
          stroke={SERIES.observed}
          strokeWidth={2}
          dot={false}
          activeDot={{ r: 4 }}
        />
      </ComposedChart>
    </ResponsiveContainer>
  );
}
