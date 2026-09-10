'use client';

import { useQuery } from '@tanstack/react-query';
import { Activity, Database, Sparkles } from 'lucide-react';
import { useMemo, useState } from 'react';
import {
  Bar, BarChart, Brush, CartesianGrid, Cell, ComposedChart, Line, LineChart,
  ReferenceLine, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis,
} from 'recharts';

import { ChartFrame } from '@/components/pol4/chart-frame';
import { ChartTooltip } from '@/components/pol4/tooltip';
import { SERIES, axisProps, compact, heatStep, jalali, num, percent } from '@/components/pol4/theme';
import { ThemeToggle } from '@/components/theme-toggle';
import { ChartSkeleton, ErrorState, NoModelState } from '@/components/ui/states';
import { pol4 } from '@/lib/pol4/api';
import type { DashboardCity, HeatmapRow, ModelVariant } from '@/lib/pol4/types';

const MODEL_COLORS = { raw: '#8b93a7', calibrated: '#2a78d6' } as const;
const TOP_N = [10, 20, 30];
const DEMAND_LABELS: Record<string, string> = {
  'q0.00-0.50': 'کم‌تقاضا · ۵۰٪ پایین',
  'q0.50-0.75': 'متوسط رو به پایین',
  'q0.75-0.90': 'متوسط رو به بالا',
  'q0.90-0.99': 'پرتقاضا · ۱۰٪ بالا',
  'q0.99-1.00': 'اوج · ۱٪ بالا',
};
const HIGH_LABELS: Record<string, string> = {
  top_1pct: '۱٪ پرترافیک', top_5pct: '۵٪ پرترافیک', top_10pct: '۱۰٪ پرترافیک',
};

function Metric({ label, value, note }: { label: string; value: string; note: string }) {
  return <div className="decision-metric"><span>{label}</span><strong dir="auto">{value}</strong><small>{note}</small></div>;
}

function Heatmap({ rows, dates, max, onSelect }: {
  rows: HeatmapRow[]; dates: string[]; max: number; onSelect: (cityCode: number) => void;
}) {
  return (
    <div className="decision-heatmap" dir="ltr">
      <div className="decision-heatmap__grid" style={{ gridTemplateColumns: `128px repeat(${dates.length}, 26px)` }}>
        <span className="decision-heatmap__corner">شهر / شب</span>
        {dates.map((date) => <span key={date} className="decision-heatmap__date">{jalali(date)}</span>)}
        {rows.map((row) => (
          <div key={row.city_code} className="decision-heatmap__row">
            <button type="button" onClick={() => onSelect(row.city_code)} className="decision-heatmap__city" dir="ltr">{row.city}</button>
            {row.values.map((value, index) => (
              <button type="button" key={`${row.city_code}-${dates[index]}`} className="decision-heatmap__cell"
                style={{ background: heatStep(value, max) }} title={`${row.city} · ${jalali(dates[index])}: ${num(value)}`}
                aria-label={`${row.city}، ${jalali(dates[index])}، ${num(value)} جست‌وجو`} onClick={() => onSelect(row.city_code)} />
            ))}
          </div>
        ))}
      </div>
    </div>
  );
}

function OpportunityTooltip({ active, payload, mode }: {
  active?: boolean; payload?: Array<{ payload?: DashboardCity }>; mode: 'marketing' | 'supply';
}) {
  const row = payload?.[0]?.payload;
  if (!active || !row) return null;
  return (
    <div className="pol4-tooltip" dir="rtl">
      <p className="pol4-tooltip__label" dir="ltr">{row.city} · {row.province}</p>
      <ul>
        <li><span>کل تقاضا</span><strong>{num(row.predicted_demand)}</strong></li>
        <li><span>تقاضای باقی‌مانده</span><strong>{num(row.predicted_remaining)}</strong></li>
        <li><span>{mode === 'marketing' ? 'شتاب جذب' : 'سهم تقاضای باقی‌مانده'}</span><strong>{mode === 'marketing' ? `${row.pickup_ratio?.toFixed(2) ?? '—'}×` : percent(row.remaining_share)}</strong></li>
      </ul>
    </div>
  );
}

export default function DecisionDashboard() {
  const [model, setModel] = useState<ModelVariant>('calibrated');
  const [province, setProvince] = useState('all');
  const [topN, setTopN] = useState(20);
  const [matrixMode, setMatrixMode] = useState<'marketing' | 'supply'>('marketing');
  const [selectedCity, setSelectedCity] = useState<number | null>(null);
  const query = useQuery({ queryKey: ['pol4-decision-dashboard', model], queryFn: () => pol4.dashboard(model) });
  const data = query.data?.data;

  const filteredCities = useMemo(() => {
    if (!data) return [];
    return province === 'all' ? data.cities : data.cities.filter((city) => city.province === province);
  }, [data, province]);
  const rankedCities = useMemo(() => {
    const rows = [...filteredCities].sort((a, b) => b.predicted_demand - a.predicted_demand).slice(0, topN);
    const base = filteredCities.reduce((sum, row) => sum + row.predicted_demand, 0) || 1;
    let running = 0;
    return rows.map((row) => { running += row.predicted_demand; return { ...row, local_cumulative_percent: (running / base) * 100 }; });
  }, [filteredCities, topN]);
  const heatRows = useMemo(() => {
    if (!data) return [];
    const allowed = new Set(rankedCities.map((city) => city.city_code));
    return data.heatmap.rows.filter((row) => allowed.has(row.city_code));
  }, [data, rankedCities]);
  const matrixRows = useMemo(() => filteredCities.filter((row) => matrixMode === 'supply' || row.pickup_ratio !== null), [filteredCities, matrixMode]);
  const selected = data?.cities.find((city) => city.city_code === selectedCity) ?? null;

  if (query.isLoading) return <div className="decision-loading"><ChartSkeleton height={560} /></div>;
  if (query.isError) return <div className="decision-loading"><ErrorState message={(query.error as Error).message} onRetry={() => query.refetch()} /></div>;
  if (query.isSuccess && !query.data.available) return <div className="decision-loading"><NoModelState detail={query.data.detailFa} /></div>;
  if (!data) return null;

  const matrixY = matrixMode === 'marketing' ? 'pickup_ratio' : 'remaining_share';
  const matrixThreshold = matrixMode === 'marketing' ? data.thresholds.pickup_median : data.thresholds.remaining_share_median;
  const heatMax = Math.max(1, ...heatRows.flatMap((row) => row.values));
  const forecastSource = model === 'raw' ? 'results_raw_named.csv' : 'results_calibrated_named.csv';

  return (
    <div className="decision-page">
      <header className="decision-header">
        <div className="decision-brand"><span className="decision-brand__mark"><Activity aria-hidden="true" /></span><div><p>NOVO PULSE · POL 4</p><h1>رادار تصمیم تقاضای جاباما</h1></div></div>
        <div className="decision-header__actions"><span className="decision-freshness"><span /> داده تا {jalali(data.cutoff)}</span><ThemeToggle compact /></div>
      </header>

      <section className="model-switcher" aria-label="انتخاب مدل">
        <div className="model-switcher__intro"><span><Sparkles aria-hidden="true" /> مدل فعال</span><p>تمام نمودارهای پیش‌بینی با انتخاب شما هم‌زمان محاسبه می‌شوند.</p></div>
        <div className="model-switcher__options" role="radiogroup">
          {data.models.map((item) => <button key={item.key} type="button" role="radio" aria-checked={model === item.key} className={model === item.key ? 'is-active' : undefined} onClick={() => setModel(item.key)}><span>{item.key === 'raw' ? 'مدل خام' : 'مدل کالیبره'}</span><strong dir="ltr">WAPE {percent(item.wape)}</strong><small dir="ltr">Bias {percent(item.normalised_bias)}</small></button>)}
        </div>
      </section>

      <section className="decision-summary" aria-label="خلاصه مدل فعال">
        <Metric label="تقاضای نهایی" value={num(data.summary.forecast_total)} note="۳۰ شب · ۳۲۱ شهر" />
        <Metric label="ثبت‌شده / باقی‌مانده" value={`${percent(data.summary.observed_so_far / data.summary.forecast_total)} / ${percent(data.summary.predicted_remaining / data.summary.forecast_total)}`} note="تا cutoff / خروجی مدل" />
        <Metric label="اوج تقاضا" value={jalali(data.summary.peak_date)} note={`${num(data.summary.peak_demand)} · ${percent(data.summary.peak_deviation)} بالاتر از میانگین`} />
        <Metric label="کمینه تقاضا" value={jalali(data.summary.low_date)} note={`${num(data.summary.low_demand)} · ${percent(Math.abs(data.summary.low_deviation))} پایین‌تر از میانگین`} />
        <Metric label="اعتبار backtest" value={percent(data.summary.wape)} note={`Bias ${percent(data.summary.normalised_bias)} · مبنا ${percent(data.summary.baseline_wape)}`} />
      </section>

      <div className="decision-toolbar">
        <label>استان<select value={province} onChange={(event) => { setProvince(event.target.value); setSelectedCity(null); }}><option value="all">همه استان‌ها</option>{data.provinces.map((row) => <option key={row.province} value={row.province}>{row.province}</option>)}</select></label>
        <div className="decision-toolbar__topn"><span>تعداد شهر</span><div className="pol4-toggle">{TOP_N.map((choice) => <button key={choice} type="button" className={topN === choice ? 'is-active' : undefined} onClick={() => setTopN(choice)}>{num(choice)}</button>)}</div></div>
        {selected ? <button className="selected-city" type="button" onClick={() => setSelectedCity(null)}><span dir="ltr">{selected.city}</span> ×</button> : null}
      </div>

      <section className="decision-section">
        <div className="decision-section__title"><span>۰۱</span><div><h2>چه زمانی؟</h2><p>ریتم بازار، شب‌های اوج و فاصله دو مدل</p></div></div>
        <ChartFrame title="تقاضای روزانه؛ ثبت‌شده در برابر باقی‌مانده" unit="تعداد جست‌وجو" hint="خطوط عمودی شب اوج و کمینه را مشخص می‌کنند؛ Brush پایین نمودار بازه را زوم می‌کند." source={forecastSource} height={390} legend={[{ label: 'ثبت‌شده تا cutoff', color: SERIES.observed }, { label: 'باقی‌مانده پیش‌بینی‌شده', color: SERIES.forecast }, { label: 'مجموع نهایی', color: SERIES.ink }]}>
          <ResponsiveContainer width="100%" height="100%"><ComposedChart data={data.national_series} margin={{ top: 12, right: 12, left: 8, bottom: 4 }}><CartesianGrid stroke={SERIES.grid} strokeDasharray="3 5" vertical={false} /><XAxis dataKey="checkin" {...axisProps} tickFormatter={jalali} minTickGap={22} /><YAxis {...axisProps} width={58} tickFormatter={compact} /><Tooltip content={<ChartTooltip labelFormatter={jalali} order={['predicted_demand', 'observed_so_far', 'predicted_remaining']} />} /><ReferenceLine x={data.summary.peak_date} stroke="#e11d48" strokeDasharray="4 4" label={{ value: 'اوج', fill: '#e11d48', fontSize: 10 }} /><ReferenceLine x={data.summary.low_date} stroke="#0d9488" strokeDasharray="4 4" label={{ value: 'کمینه', fill: '#0d9488', fontSize: 10 }} /><Bar isAnimationActive={false} dataKey="observed_so_far" name="ثبت‌شده" stackId="d" fill={SERIES.observed} /><Bar isAnimationActive={false} dataKey="predicted_remaining" name="باقی‌مانده" stackId="d" fill={SERIES.forecast} radius={[4, 4, 0, 0]} /><Line isAnimationActive={false} dataKey="predicted_demand" name="تقاضای نهایی" stroke={SERIES.ink} strokeWidth={2.3} dot={false} activeDot={{ r: 5 }} /><Brush dataKey="checkin" height={24} travellerWidth={8} tickFormatter={jalali} stroke={SERIES.forecast} /></ComposedChart></ResponsiveContainer>
        </ChartFrame>
        <ChartFrame title="فاصله خروجی مدل خام و کالیبره" unit="تعداد جست‌وجو" hint="کالیبراسیون فقط روی horizonهایی که guard ایمنی اجازه داده اثر می‌گذارد؛ روی نقاط بدون تغییر دو خط هم‌پوشان‌اند." source="results_raw_named.csv + results_calibrated_named.csv" height={310} legend={[{ label: 'Raw', color: MODEL_COLORS.raw }, { label: 'Calibrated', color: MODEL_COLORS.calibrated }]}>
          <ResponsiveContainer width="100%" height="100%"><LineChart data={data.model_comparison} margin={{ top: 8, right: 12, left: 8, bottom: 0 }}><CartesianGrid stroke={SERIES.grid} strokeDasharray="3 5" vertical={false} /><XAxis dataKey="checkin" {...axisProps} tickFormatter={jalali} minTickGap={22} /><YAxis {...axisProps} width={58} tickFormatter={compact} /><Tooltip content={<ChartTooltip labelFormatter={jalali} order={['predicted_demand_raw', 'predicted_demand_calibrated', 'delta']} />} /><Line isAnimationActive={false} dataKey="predicted_demand_raw" name="Raw" stroke={MODEL_COLORS.raw} strokeWidth={2} dot={false} /><Line isAnimationActive={false} dataKey="predicted_demand_calibrated" name="Calibrated" stroke={MODEL_COLORS.calibrated} strokeWidth={2.5} dot={false} activeDot={{ r: 5 }} /></LineChart></ResponsiveContainer>
        </ChartFrame>
      </section>

      <section className="decision-section">
        <div className="decision-section__title"><span>۰۲</span><div><h2>کجا؟</h2><p>تمرکز تقاضا و اولویت‌های Growth / Host Growth</p></div></div>
        <div className="decision-grid decision-grid--wide">
          <ChartFrame title="پارتوی شهرها؛ بودجه را کجا متمرکز کنیم؟" unit="تقاضا + سهم تجمعی" hint="ستون، حجم تقاضا و خط، سهم تجمعی شهرهای انتخاب‌شده از کل بازار فیلترشده است." source={`${forecastSource} → city aggregation`} height={Math.max(380, rankedCities.length * 29)}><ResponsiveContainer width="100%" height="100%"><ComposedChart layout="vertical" data={rankedCities} margin={{ top: 12, right: 16, left: 8, bottom: 4 }}><CartesianGrid stroke={SERIES.grid} strokeDasharray="3 5" horizontal={false} /><XAxis xAxisId="demand" type="number" {...axisProps} tickFormatter={compact} /><XAxis xAxisId="share" type="number" domain={[0, 100]} orientation="top" {...axisProps} tickFormatter={(value) => `${num(value)}٪`} /><YAxis type="category" dataKey="city" width={92} {...axisProps} tick={{ fontSize: 11 }} /><Tooltip content={<ChartTooltip order={['predicted_demand', 'local_cumulative_percent']} />} /><Bar isAnimationActive={false} xAxisId="demand" dataKey="predicted_demand" name="تقاضا" fill={SERIES.forecast} radius={[0, 5, 5, 0]} onClick={(row) => setSelectedCity(Number((row as unknown as DashboardCity).city_code))} /><Line isAnimationActive={false} xAxisId="share" dataKey="local_cumulative_percent" name="سهم تجمعی (%)" stroke={SERIES.observed} strokeWidth={2} dot={{ r: 2 }} /></ComposedChart></ResponsiveContainer></ChartFrame>
          <ChartFrame title="سهم استان‌ها" unit="تعداد جست‌وجو" hint="تفکیک تقاضای ثبت‌شده و تقاضایی که مدل هنوز انتظار دارد." source={`${forecastSource} → province aggregation`} height={360}><ResponsiveContainer width="100%" height="100%"><BarChart layout="vertical" data={data.provinces} margin={{ top: 8, right: 12, left: 8, bottom: 0 }}><CartesianGrid stroke={SERIES.grid} strokeDasharray="3 5" horizontal={false} /><XAxis type="number" {...axisProps} tickFormatter={compact} /><YAxis type="category" dataKey="province" width={92} {...axisProps} /><Tooltip content={<ChartTooltip order={['observed_so_far', 'predicted_remaining', 'predicted_demand']} />} /><Bar isAnimationActive={false} dataKey="observed_so_far" name="ثبت‌شده" stackId="p" fill={SERIES.observed} /><Bar isAnimationActive={false} dataKey="predicted_remaining" name="باقی‌مانده" stackId="p" fill={SERIES.forecast} radius={[0, 5, 5, 0]} /></BarChart></ResponsiveContainer></ChartFrame>
        </div>
        <ChartFrame title="نقشه حرارتی شهر × شب" unit="تعداد جست‌وجو" hint="روی نام شهر یا هر خانه کلیک کنید تا همان شهر در ماتریس فرصت برجسته شود؛ hover مقدار دقیق را نشان می‌دهد." source={`${forecastSource} → city/date pivot`} height={Math.min(680, 84 + heatRows.length * 29)}><Heatmap rows={heatRows} dates={data.heatmap.dates} max={heatMax} onSelect={setSelectedCity} /></ChartFrame>
        <ChartFrame title={matrixMode === 'marketing' ? 'ماتریس اولویت Marketing / Growth' : 'ماتریس اولویت Host Growth / Supply'} unit="هر حباب = یک شهر" hint={matrixMode === 'marketing' ? 'بالا-راست: شهرهای بزرگ با pickup سریع‌تر از الگوی تاریخی؛ نامزد افزایش بودجه یا کمپین سریع.' : 'بالا-راست: شهرهای بزرگ با سهم بیشتری از تقاضای هنوز محقق‌نشده؛ سیگنال بررسی جذب میزبان، نه اثبات کمبود عرضه.'} source={matrixMode === 'marketing' ? `${forecastSource} + city_momentum.parquet` : forecastSource} height={430} action={<div className="pol4-toggle"><button type="button" className={matrixMode === 'marketing' ? 'is-active' : undefined} onClick={() => setMatrixMode('marketing')}>Marketing</button><button type="button" className={matrixMode === 'supply' ? 'is-active' : undefined} onClick={() => setMatrixMode('supply')}>Host Growth</button></div>}><ResponsiveContainer width="100%" height="100%"><ScatterChart margin={{ top: 20, right: 26, bottom: 28, left: 8 }}><CartesianGrid stroke={SERIES.grid} strokeDasharray="3 5" /><XAxis type="number" dataKey="predicted_demand" name="تقاضای کل" scale="log" domain={['auto', 'auto']} {...axisProps} tickFormatter={compact} label={{ value: 'تقاضای کل (مقیاس لگاریتمی)', position: 'insideBottom', offset: -18, fontSize: 11 }} /><YAxis type="number" dataKey={matrixY} name={matrixMode === 'marketing' ? 'Pickup ratio' : 'سهم باقی‌مانده'} {...axisProps} tickFormatter={(value) => matrixMode === 'marketing' ? `${Number(value).toFixed(1)}×` : percent(Number(value), 0)} width={58} /><ZAxis type="number" dataKey="predicted_remaining" range={[30, 520]} /><ReferenceLine x={data.thresholds.demand_median} stroke={SERIES.axis} strokeDasharray="4 5" /><ReferenceLine y={matrixThreshold} stroke={SERIES.axis} strokeDasharray="4 5" /><Tooltip cursor={{ strokeDasharray: '3 3' }} content={<OpportunityTooltip mode={matrixMode} />} /><Scatter isAnimationActive={false} data={matrixRows} onClick={(row) => setSelectedCity(Number((row as DashboardCity).city_code))}>{matrixRows.map((row) => <Cell key={row.city_code} fill={row.city_code === selectedCity ? SERIES.observed : SERIES.forecast} fillOpacity={row.city_code === selectedCity ? 1 : 0.58} stroke={row.city_code === selectedCity ? SERIES.ink : 'transparent'} />)}</Scatter></ScatterChart></ResponsiveContainer></ChartFrame>
      </section>

      <section className="decision-section">
        <div className="decision-section__title"><span>۰۳</span><div><h2>چقدر قابل اتکاست؟</h2><p>خطا را جایی ببینید که تصمیم واقعاً گرفته می‌شود</p></div></div>
        <div className="decision-grid">
          <ChartFrame title="ریسک بر اساس Lead time" unit="درصد" hint="ستون WAPE تاریخی و خط سهم تقاضایی است که تا cutoff برای همین target grid دیده شده؛ دورتر شدن یعنی هم مشاهده کمتر و هم ریسک بیشتر." source={`backtest_metrics_phase2.json + ${forecastSource}`} height={330}><ResponsiveContainer width="100%" height="100%"><ComposedChart data={data.lead_time} margin={{ top: 8, right: 6, left: 6, bottom: 0 }}><CartesianGrid stroke={SERIES.grid} strokeDasharray="3 5" vertical={false} /><XAxis dataKey="key" {...axisProps} tickFormatter={(value) => `D${value}`} /><YAxis yAxisId="left" {...axisProps} tickFormatter={(value) => percent(Number(value), 0)} width={48} /><YAxis yAxisId="right" orientation="right" {...axisProps} tickFormatter={(value) => percent(Number(value), 0)} width={48} /><Tooltip content={<ChartTooltip order={['wape', 'normalised_bias', 'observed_share']} valueFormatter={percent} />} /><Bar isAnimationActive={false} yAxisId="left" dataKey="wape" name="WAPE" fill={SERIES.forecast} radius={[5, 5, 0, 0]} /><Line isAnimationActive={false} yAxisId="right" dataKey="observed_share" name="سهم مشاهده‌شده" stroke={SERIES.observed} strokeWidth={2.5} dot={{ r: 4 }} /></ComposedChart></ResponsiveContainer></ChartFrame>
          <ChartFrame title="خطا در Low-demand تا Peak-demand" unit="درصد" hint="WAPE در ردیف‌های کوچک به‌خاطر مخرج کم بزرگ‌تر دیده می‌شود؛ خط Bias جهت کم‌برآورد/بیش‌برآورد را نشان می‌دهد." source="backtest_metrics_phase2.json → by_demand_bucket" height={330}><ResponsiveContainer width="100%" height="100%"><ComposedChart data={data.demand_buckets.map((row) => ({ ...row, label: DEMAND_LABELS[String(row.key)] ?? row.key }))} margin={{ top: 8, right: 6, left: 6, bottom: 28 }}><CartesianGrid stroke={SERIES.grid} strokeDasharray="3 5" vertical={false} /><XAxis dataKey="label" {...axisProps} interval={0} angle={-16} textAnchor="end" height={52} /><YAxis {...axisProps} tickFormatter={(value) => percent(Number(value), 0)} width={48} /><ReferenceLine y={0} stroke={SERIES.axis} /><Tooltip content={<ChartTooltip order={['wape', 'normalised_bias']} valueFormatter={percent} />} /><Bar isAnimationActive={false} dataKey="wape" name="WAPE" fill={SERIES.forecast} radius={[5, 5, 0, 0]} /><Line isAnimationActive={false} dataKey="normalised_bias" name="Bias" stroke={SERIES.observed} strokeWidth={2.5} dot={{ r: 4 }} /></ComposedChart></ResponsiveContainer></ChartFrame>
          <ChartFrame title="دقت روی روزهای پرترافیک" unit="درصد" hint="عملکرد جداگانه روی ۱٪، ۵٪ و ۱۰٪ بالای تقاضای واقعی؛ این نمودار ریسک peak-demand را از میانگین کل جدا می‌کند." source="backtest_metrics_phase2.json → high_demand" height={310}><ResponsiveContainer width="100%" height="100%"><BarChart data={data.high_demand.map((row) => ({ ...row, label: HIGH_LABELS[String(row.key)] ?? row.key, absolute_bias: Math.abs(row.normalised_bias) }))} margin={{ top: 8, right: 8, left: 8, bottom: 0 }}><CartesianGrid stroke={SERIES.grid} strokeDasharray="3 5" vertical={false} /><XAxis dataKey="label" {...axisProps} /><YAxis {...axisProps} tickFormatter={(value) => percent(Number(value), 0)} width={48} /><Tooltip content={<ChartTooltip order={['wape', 'absolute_bias']} valueFormatter={percent} />} /><Bar isAnimationActive={false} dataKey="wape" name="WAPE" fill={SERIES.forecast} radius={[5, 5, 0, 0]} /><Bar isAnimationActive={false} dataKey="absolute_bias" name="|Bias|" fill={SERIES.observed} radius={[5, 5, 0, 0]} /></BarChart></ResponsiveContainer></ChartFrame>
          <ChartFrame title="اعتبار زمانی؛ مدل در برابر Baseline" unit="WAPE" hint="پنج rolling-origin fold با cutoffهای مستقل. اختلاف foldها همان نوسان واقعی تعمیم مدل در زمان است، نه train score." source="backtest_metrics_phase2.json → folds" height={310}><ResponsiveContainer width="100%" height="100%"><LineChart data={data.folds} margin={{ top: 8, right: 8, left: 8, bottom: 0 }}><CartesianGrid stroke={SERIES.grid} strokeDasharray="3 5" vertical={false} /><XAxis dataKey="cutoff" {...axisProps} tickFormatter={jalali} minTickGap={16} /><YAxis {...axisProps} tickFormatter={(value) => percent(Number(value), 0)} width={48} /><Tooltip content={<ChartTooltip labelFormatter={jalali} order={['model_wape', 'baseline_wape', 'model_bias']} valueFormatter={percent} />} /><Line isAnimationActive={false} dataKey="baseline_wape" name="Pickup baseline" stroke={MODEL_COLORS.raw} strokeDasharray="5 4" strokeWidth={2} dot={{ r: 3 }} /><Line isAnimationActive={false} dataKey="model_wape" name="مدل فعال" stroke={SERIES.forecast} strokeWidth={2.5} dot={{ r: 4 }} /></LineChart></ResponsiveContainer></ChartFrame>
        </div>
      </section>

      <footer className="decision-footer"><Database aria-hidden="true" /><p>تمام اعداد از artifactهای همان training run خوانده می‌شوند؛ هیچ داده نمونه یا محاسبه ساختگی در فرانت وجود ندارد.</p><span dir="ltr">{data.summary.cities} cities × {data.summary.dates} nights</span></footer>
    </div>
  );
}
