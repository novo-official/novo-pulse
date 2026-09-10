'use client';

import { useQuery } from '@tanstack/react-query';
import { useMemo, useState } from 'react';
import {
  Area,
  Bar,
  BarChart,
  Brush,
  CartesianGrid,
  Cell,
  ComposedChart,
  Label,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from 'recharts';

import { ChartFrame } from '@/components/pol4/chart-frame';
import { ChartTooltip } from '@/components/pol4/tooltip';
import { SERIES, axisProps, compact, heatStep, jalali, num, percent } from '@/components/pol4/theme';
import { ChartSkeleton, ErrorState, NoModelState } from '@/components/ui/states';
import { pol4 } from '@/lib/pol4/api';
import type { DashboardCity, HeatmapRow, ModelVariant } from '@/lib/pol4/types';

const TOP_N = [10, 20, 30] as const;
const SHOCK_FOLD = '2025-05-21';
const COLORS = {
  model: '#2a78d6',
  modelShock: '#d97706',
  baseline: '#9aa6b6',
  selected: '#eb6834',
} as const;

function Heatmap({ rows, dates, max, selectedCity, onSelect }: {
  rows: HeatmapRow[];
  dates: string[];
  max: number;
  selectedCity: number | null;
  onSelect: (cityCode: number) => void;
}) {
  return (
    <div className="presentation-heatmap" dir="ltr">
      <div className="presentation-heatmap__grid" style={{ gridTemplateColumns: `148px repeat(${dates.length}, 28px)` }}>
        <span className="presentation-heatmap__corner">شهر / شب</span>
        {dates.map((date) => <span key={date} className="presentation-heatmap__date">{jalali(date)}</span>)}
        {rows.map((row) => (
          <div key={row.city_code} className="presentation-heatmap__row">
            <button
              type="button"
              className={`presentation-heatmap__city${selectedCity === row.city_code ? ' is-selected' : ''}`}
              onClick={() => onSelect(row.city_code)}
              dir="ltr"
            >
              {row.city}
            </button>
            {row.values.map((value, index) => (
              <button
                type="button"
                key={`${row.city_code}-${dates[index]}`}
                className={`presentation-heatmap__cell${selectedCity === row.city_code ? ' is-selected' : ''}`}
                style={{ background: heatStep(value, max) }}
                title={`${row.city} · ${jalali(dates[index])}: ${num(value)}`}
                aria-label={`${row.city}، ${jalali(dates[index])}، ${num(value)} جست‌وجو`}
                onClick={() => onSelect(row.city_code)}
              />
            ))}
          </div>
        ))}
      </div>
    </div>
  );
}

function OpportunityTooltip({ active, payload, lens }: {
  active?: boolean;
  payload?: Array<{ payload?: DashboardCity }>;
  lens: 'marketing' | 'supply';
}) {
  const row = payload?.[0]?.payload;
  if (!active || !row) return null;
  return (
    <div className="pol4-tooltip" dir="rtl">
      <p className="pol4-tooltip__label" dir="ltr">{row.city} · {row.province}</p>
      <ul>
        <li><span>تقاضای نهایی</span><strong>{num(row.predicted_demand)}</strong></li>
        <li><span>تقاضای باقی‌مانده</span><strong>{num(row.predicted_remaining)}</strong></li>
        <li>
          <span>{lens === 'marketing' ? 'Pickup / انتظار' : 'سهم باقی‌مانده'}</span>
          <strong>{lens === 'marketing' ? `${row.pickup_ratio?.toFixed(2) ?? '—'}×` : percent(row.remaining_share)}</strong>
        </li>
      </ul>
    </div>
  );
}

export default function PresentationDashboard() {
  const [model, setModel] = useState<ModelVariant>('calibrated');
  const [province, setProvince] = useState('all');
  const [topN, setTopN] = useState<(typeof TOP_N)[number]>(20);
  const [lens, setLens] = useState<'marketing' | 'supply'>('marketing');
  const [selectedCity, setSelectedCity] = useState<number | null>(null);
  const query = useQuery({
    queryKey: ['pol4-presentation-dashboard', model],
    queryFn: () => pol4.dashboard(model),
  });
  const data = query.data?.data;

  const filteredCities = useMemo(() => {
    if (!data) return [];
    return province === 'all' ? data.cities : data.cities.filter((row) => row.province === province);
  }, [data, province]);

  const rankedCities = useMemo(
    () => [...filteredCities].sort((a, b) => b.predicted_demand - a.predicted_demand).slice(0, topN),
    [filteredCities, topN],
  );

  const heatRows = useMemo(() => {
    if (!data) return [];
    const allowed = new Set(rankedCities.map((row) => row.city_code));
    return data.heatmap.rows.filter((row) => allowed.has(row.city_code));
  }, [data, rankedCities]);

  const opportunityRows = useMemo(
    () => filteredCities.filter((row) => row.predicted_demand > 0 && (lens === 'supply' || row.pickup_ratio !== null)),
    [filteredCities, lens],
  );

  if (query.isLoading) return <div className="presentation-loading"><ChartSkeleton height={600} /></div>;
  if (query.isError) return <div className="presentation-loading"><ErrorState message={(query.error as Error).message} onRetry={() => query.refetch()} /></div>;
  if (query.isSuccess && !query.data.available) return <div className="presentation-loading"><NoModelState detail={query.data.detailFa} /></div>;
  if (!data) return null;

  const selected = data.cities.find((row) => row.city_code === selectedCity) ?? null;
  const heatMax = Math.max(1, ...heatRows.flatMap((row) => row.values));
  const matrixY = lens === 'marketing' ? 'pickup_ratio' : 'remaining_share';
  const matrixThreshold = lens === 'marketing' ? data.thresholds.pickup_median : data.thresholds.remaining_share_median;
  const forecastSource = model === 'raw' ? 'results_raw_named.csv' : 'results_calibrated_named.csv';
  const folds = data.folds.map((row) => ({ ...row, shock: row.cutoff === SHOCK_FOLD }));

  return (
    <div className="presentation-page">
      <header className="presentation-header">
        <div>
          <span>NOVO PULSE · POL 4</span>
          <h1>رادار تقاضای ۳۰ شب آینده</h1>
        </div>
        <p>۳۲۱ شهر · ۷ استان · داده تا {jalali(data.cutoff)}</p>
      </header>

      <nav className="presentation-filters" aria-label="فیلتر نمودارها">
        <fieldset>
          <legend>مدل</legend>
          <div className="presentation-segment" role="radiogroup">
            {data.models.map((item) => (
              <button
                key={item.key}
                type="button"
                role="radio"
                aria-checked={model === item.key}
                className={model === item.key ? 'is-active' : undefined}
                onClick={() => setModel(item.key)}
              >
                {item.key === 'raw' ? 'خام' : 'کالیبره'}
                <small>WAPE {percent(item.wape)}</small>
              </button>
            ))}
          </div>
        </fieldset>

        <label>
          <span>استان</span>
          <select value={province} onChange={(event) => { setProvince(event.target.value); setSelectedCity(null); }}>
            <option value="all">همه استان‌ها</option>
            {data.provinces.map((row) => <option key={row.province} value={row.province}>{row.province}</option>)}
          </select>
        </label>

        <fieldset>
          <legend>شهرهای برتر</legend>
          <div className="presentation-segment">
            {TOP_N.map((value) => (
              <button key={value} type="button" className={topN === value ? 'is-active' : undefined} onClick={() => setTopN(value)}>{num(value)}</button>
            ))}
          </div>
        </fieldset>

        <fieldset>
          <legend>زاویه تحلیل</legend>
          <div className="presentation-segment">
            <button type="button" className={lens === 'marketing' ? 'is-active' : undefined} onClick={() => setLens('marketing')}>Growth</button>
            <button type="button" className={lens === 'supply' ? 'is-active' : undefined} onClick={() => setLens('supply')}>Supply</button>
          </div>
        </fieldset>

        {selected ? (
          <button className="presentation-selected" type="button" onClick={() => setSelectedCity(null)}>
            <span dir="ltr">{selected.city}</span><b aria-hidden="true">×</b>
          </button>
        ) : null}
      </nav>

      <main className="presentation-charts">
        <ChartFrame
          title="ریتم تقاضای کل پنل"
          unit="جست‌وجو در هر شب"
          hint="تقاضای ثبت‌شده و بخش باقی‌مانده‌ای که مدل انتظار دارد؛ خطوط نقطه‌چین شب اوج و کمینه را مشخص می‌کنند."
          source={forecastSource}
          height={420}
          legend={[
            { label: 'ثبت‌شده تا cutoff', color: SERIES.observed },
            { label: 'باقی‌مانده پیش‌بینی‌شده', color: SERIES.forecast },
            { label: 'تقاضای نهایی', color: SERIES.ink },
          ]}
        >
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={data.national_series} margin={{ top: 18, right: 18, left: 10, bottom: 4 }}>
              <defs>
                <linearGradient id="observedArea" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stopColor={SERIES.observed} stopOpacity={0.88} /><stop offset="1" stopColor={SERIES.observed} stopOpacity={0.42} /></linearGradient>
                <linearGradient id="forecastArea" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stopColor={SERIES.forecast} stopOpacity={0.78} /><stop offset="1" stopColor={SERIES.forecast} stopOpacity={0.25} /></linearGradient>
              </defs>
              <CartesianGrid stroke={SERIES.grid} strokeDasharray="3 6" vertical={false} />
              <XAxis dataKey="checkin" {...axisProps} tickFormatter={jalali} minTickGap={28} />
              <YAxis {...axisProps} width={64} tickFormatter={compact} />
              <Tooltip content={<ChartTooltip labelFormatter={jalali} order={['predicted_demand', 'observed_so_far', 'predicted_remaining']} />} />
              <ReferenceLine x={data.summary.peak_date} stroke="#e11d48" strokeDasharray="5 5"><Label value="اوج" position="insideTopLeft" fill="#e11d48" fontSize={11} /></ReferenceLine>
              <ReferenceLine x={data.summary.low_date} stroke="#0d9488" strokeDasharray="5 5"><Label value="کمینه" position="insideTopRight" fill="#0d9488" fontSize={11} /></ReferenceLine>
              <Area isAnimationActive={false} type="monotone" dataKey="observed_so_far" name="ثبت‌شده" stackId="demand" stroke={SERIES.observed} fill="url(#observedArea)" />
              <Area isAnimationActive={false} type="monotone" dataKey="predicted_remaining" name="باقی‌مانده" stackId="demand" stroke={SERIES.forecast} fill="url(#forecastArea)" />
              <Line isAnimationActive={false} type="monotone" dataKey="predicted_demand" name="تقاضای نهایی" stroke={SERIES.ink} strokeWidth={2.6} dot={false} activeDot={{ r: 5, strokeWidth: 2 }} />
              <Brush dataKey="checkin" height={26} travellerWidth={8} tickFormatter={jalali} stroke={SERIES.forecast} />
            </ComposedChart>
          </ResponsiveContainer>
        </ChartFrame>

        <div className="presentation-grid">
          <ChartFrame
            title="تمرکز تقاضا در شهرها"
            unit={`${num(topN)} شهر برتر`}
            hint="مرتب‌شده بر اساس تقاضای نهایی در استان انتخاب‌شده؛ برای برجسته‌کردن شهر در نمودارهای دیگر روی ستون کلیک کنید."
            source={`${forecastSource} → city aggregation`}
            height={Math.max(390, rankedCities.length * 28)}
          >
            <ResponsiveContainer width="100%" height="100%">
              <BarChart layout="vertical" data={rankedCities} margin={{ top: 8, right: 20, left: 10, bottom: 2 }}>
                <CartesianGrid stroke={SERIES.grid} strokeDasharray="3 6" horizontal={false} />
                <XAxis type="number" {...axisProps} tickFormatter={compact} />
                <YAxis type="category" dataKey="city" width={104} {...axisProps} tick={{ fontSize: 11 }} />
                <Tooltip content={<ChartTooltip order={['predicted_demand', 'observed_so_far', 'predicted_remaining']} />} />
                <Bar isAnimationActive={false} dataKey="predicted_demand" name="تقاضای نهایی" radius={[0, 7, 7, 0]} onClick={(row) => setSelectedCity(Number((row as unknown as DashboardCity).city_code))}>
                  {rankedCities.map((row) => <Cell key={row.city_code} fill={row.city_code === selectedCity ? COLORS.selected : SERIES.forecast} fillOpacity={row.city_code === selectedCity ? 1 : 0.78} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </ChartFrame>

          <ChartFrame
            title={lens === 'marketing' ? 'ماتریس فرصت Growth' : 'ماتریس بررسی Supply'}
            unit="هر نقطه یک شهر"
            hint={lens === 'marketing'
              ? 'بالا-راست: حجم بالاتر و pickup سریع‌تر از انتظار؛ سیگنال بررسی تقویم کمپین، نه تأیید هزینه.'
              : 'بالا-راست: حجم بالاتر و تقاضای باقی‌مانده بیشتر؛ سیگنال بررسی عرضه، نه اثبات کمبود.'}
            source={lens === 'marketing' ? `${forecastSource} + city_momentum.parquet` : forecastSource}
            height={Math.max(390, rankedCities.length * 28)}
          >
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart margin={{ top: 20, right: 28, bottom: 34, left: 6 }}>
                <CartesianGrid stroke={SERIES.grid} strokeDasharray="3 6" />
                <XAxis type="number" dataKey="predicted_demand" name="تقاضای نهایی" scale="log" domain={['auto', 'auto']} {...axisProps} tickFormatter={compact}>
                  <Label value="تقاضای نهایی · مقیاس لگاریتمی" position="insideBottom" offset={-22} fill={SERIES.axis} fontSize={10} />
                </XAxis>
                <YAxis type="number" dataKey={matrixY} name={lens === 'marketing' ? 'Pickup / انتظار' : 'سهم باقی‌مانده'} {...axisProps} width={58} tickFormatter={(value) => lens === 'marketing' ? `${Number(value).toFixed(1)}×` : percent(Number(value), 0)} />
                <ZAxis type="number" dataKey="predicted_remaining" range={[28, 500]} />
                <ReferenceLine x={data.thresholds.demand_median} stroke={SERIES.axis} strokeDasharray="5 5" />
                <ReferenceLine y={matrixThreshold} stroke={SERIES.axis} strokeDasharray="5 5" />
                <Tooltip cursor={{ strokeDasharray: '4 4' }} content={<OpportunityTooltip lens={lens} />} />
                <Scatter isAnimationActive={false} data={opportunityRows} onClick={(row) => setSelectedCity(Number((row as DashboardCity).city_code))}>
                  {opportunityRows.map((row) => <Cell key={row.city_code} fill={row.city_code === selectedCity ? COLORS.selected : SERIES.forecast} fillOpacity={row.city_code === selectedCity ? 1 : 0.54} stroke={row.city_code === selectedCity ? SERIES.ink : 'transparent'} strokeWidth={1.5} />)}
                </Scatter>
              </ScatterChart>
            </ResponsiveContainer>
          </ChartFrame>
        </div>

        <ChartFrame
          title="نقشه حرارتی شهر × شب"
          unit={`${num(heatRows.length)} شهر · ۳۰ شب`}
          hint="شدت رنگ، تقاضای نهایی همان شهر و شب است. انتخاب شهر در تمام نمودارهای شهری هم‌زمان برجسته می‌شود."
          source={`${forecastSource} → city/date pivot`}
          height={Math.min(720, 100 + heatRows.length * 31)}
        >
          <Heatmap rows={heatRows} dates={data.heatmap.dates} max={heatMax} selectedCity={selectedCity} onSelect={setSelectedCity} />
        </ChartFrame>

        <div className="presentation-grid">
          <ChartFrame
            title="اعتبار زمانی مدل"
            unit="WAPE · کمتر بهتر"
            hint="مدل و pickup baseline روی پنج پنجره rolling-origin یکسان؛ fold نارنجی با شوک جنگ هم‌پوشانی دارد و در امتیاز اصلی حفظ شده است."
            source="backtest_metrics_phase2.json → folds"
            height={360}
            legend={[{ label: 'مدل فعال', color: COLORS.model }, { label: 'Pickup baseline', color: COLORS.baseline }, { label: 'Shock-overlap fold', color: COLORS.modelShock }]}
          >
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={folds} margin={{ top: 14, right: 12, left: 8, bottom: 8 }} barGap={3}>
                <CartesianGrid stroke={SERIES.grid} strokeDasharray="3 6" vertical={false} />
                <XAxis dataKey="cutoff" {...axisProps} tickFormatter={jalali} minTickGap={14} />
                <YAxis {...axisProps} width={52} tickFormatter={(value) => percent(Number(value), 0)} />
                <Tooltip content={<ChartTooltip labelFormatter={jalali} order={['model_wape', 'baseline_wape', 'model_bias']} valueFormatter={percent} />} />
                <Bar isAnimationActive={false} dataKey="baseline_wape" name="Pickup baseline" fill={COLORS.baseline} fillOpacity={0.56} radius={[6, 6, 0, 0]} />
                <Bar isAnimationActive={false} dataKey="model_wape" name="مدل فعال" radius={[6, 6, 0, 0]}>
                  {folds.map((row) => <Cell key={row.cutoff} fill={row.shock ? COLORS.modelShock : COLORS.model} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </ChartFrame>

          <ChartFrame
            title="ریسک بر اساس Lead time"
            unit="WAPE و سهم مشاهده‌شده"
            hint="هرچه check-in دورتر است، بخش کمتری از تقاضا مشاهده شده و خطای تاریخی افزایش می‌یابد."
            source={`backtest_metrics_phase2.json + ${forecastSource}`}
            height={360}
            legend={[{ label: 'WAPE تاریخی', color: SERIES.forecast }, { label: 'سهم مشاهده‌شده', color: SERIES.observed }]}
          >
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={data.lead_time} margin={{ top: 14, right: 10, left: 8, bottom: 8 }}>
                <CartesianGrid stroke={SERIES.grid} strokeDasharray="3 6" vertical={false} />
                <XAxis dataKey="key" {...axisProps} tickFormatter={(value) => `D-${value}`} />
                <YAxis yAxisId="error" {...axisProps} width={50} tickFormatter={(value) => percent(Number(value), 0)} />
                <YAxis yAxisId="seen" orientation="right" {...axisProps} width={50} tickFormatter={(value) => percent(Number(value), 0)} />
                <Tooltip content={<ChartTooltip order={['wape', 'observed_share', 'normalised_bias']} valueFormatter={percent} />} />
                <Bar isAnimationActive={false} yAxisId="error" dataKey="wape" name="WAPE" fill={SERIES.forecast} fillOpacity={0.82} radius={[7, 7, 0, 0]} />
                <Line isAnimationActive={false} yAxisId="seen" type="monotone" dataKey="observed_share" name="سهم مشاهده‌شده" stroke={SERIES.observed} strokeWidth={2.8} dot={{ r: 4, fill: SERIES.observed }} />
              </ComposedChart>
            </ResponsiveContainer>
          </ChartFrame>
        </div>
      </main>
    </div>
  );
}
