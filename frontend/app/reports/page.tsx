'use client';

import { useQuery } from '@tanstack/react-query';
import { useState } from 'react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

import { ChartFrame } from '@/components/pol4/chart-frame';
import { Kpi, KpiRow } from '@/components/pol4/kpi';
import { SERIES, axisProps, jalaliFull, num, percent, ratio } from '@/components/pol4/theme';
import { ChartTooltip } from '@/components/pol4/tooltip';
import { AsyncBoundary, CardSkeleton, ChartSkeleton, NoModelState } from '@/components/ui/states';
import { pol4 } from '@/lib/pol4/api';

const REPORT_LABELS: Record<string, string> = {
  demand: 'تقاضا (شهر × شب)',
  cities: 'شهرها',
  provinces: 'استان‌ها',
  peaks: 'اوج تقاضا',
  pickup: 'تحلیل پیکاپ',
  stability: 'پایداری پیش‌بینی',
  model: 'عملکرد مدل',
};

export default function ReportsPage() {
  const [kind, setKind] = useState('demand');
  const [city, setCity] = useState('');
  const [province, setProvince] = useState('');
  const [start, setStart] = useState('');
  const [end, setEnd] = useState('');

  const filters = { city: city || undefined, province: province || undefined, start: start || undefined, end: end || undefined };

  const reportsQuery = useQuery({ queryKey: ['pol4-reports'], queryFn: pol4.reports });
  const previewQuery = useQuery({
    queryKey: ['pol4-report-preview', kind, city, province, start, end],
    queryFn: () => pol4.reportPreview(kind, filters),
  });
  const performanceQuery = useQuery({
    queryKey: ['pol4-model-performance'],
    queryFn: pol4.modelPerformance,
  });

  const preview = previewQuery.data?.data;
  const performance = performanceQuery.data?.data;

  if (performanceQuery.isSuccess && !performanceQuery.data?.available) {
    return <NoModelState detail={performanceQuery.data?.detailFa} />;
  }

  return (
    <div className="pol4-page">
      <header className="pol4-page__head">
        <h1>گزارش‌ها و عملکرد مدل</h1>
        <p>
          هر گزارش از همان فایل‌هایی ساخته می‌شود که نمودارهای این داشبورد را
          می‌سازند؛ عدد دانلودی و عدد روی نمودار هرگز اختلاف ندارند.
        </p>
      </header>

      {/* ------------------------------------------------- model performance */}
      <AsyncBoundary
        isLoading={performanceQuery.isLoading}
        error={performanceQuery.error}
        onRetry={() => performanceQuery.refetch()}
        skeleton={<CardSkeleton lines={3} />}
      >
        {performance ? (
          <>
            <KpiRow>
              <Kpi
                label="مدل منتخب"
                value={<span dir="ltr">LightGBM</span>}
                note={`${num(performance.champion.n_features)} ویژگی · ${performance.champion.horizon_bands.length} باند افق`}
              />
              <Kpi label="WAPE مدل" value={percent(performance.champion.wape)} note="روی ۵ فولد walk-forward" />
              <Kpi label="WAPE مبنا" value={percent(performance.baseline.wape)} note="مدل پیکاپ فاز ۱" />
              <Kpi
                label="بهبود"
                value={percent(performance.improvement)}
                note="نسبت به مبنا"
                tone="good"
              />
              <Kpi
                label="اریبی نرمال‌شده"
                value={ratio(performance.champion.normalised_bias)}
                note="منفی یعنی کم‌برآوردی"
                tone="warn"
              />
              <Kpi
                label="سخت‌ترین فولد"
                value={<span dir="ltr">{performance.hardest_fold}</span>}
                note="پنهان نشده — پایین توضیح داده شده"
                tone="warn"
              />
            </KpiRow>

            <div className="pol4-grid-2">
              <ChartFrame
                title="خطا در هر فولد"
                unit="WAPE"
                hint="هر فولد یک مسابقهٔ شبیه‌سازی‌شده در گذشته است."
                source="backtest_metrics_phase2.json"
                height={300}
                legend={[
                  { label: 'مدل منتخب', color: SERIES.forecast },
                  { label: 'مبنای پیکاپ', color: SERIES.observed },
                ]}
              >
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={performance.folds} margin={{ top: 8, right: 12, bottom: 4, left: 4 }}>
                    <CartesianGrid stroke={SERIES.grid} strokeDasharray="2 4" vertical={false} />
                    <XAxis dataKey="cutoff" {...axisProps} tick={{ fontSize: 10 }} />
                    <YAxis {...axisProps} width={48} tickFormatter={(v: number) => percent(v, 0)} />
                    <Tooltip
                      content={<ChartTooltip valueFormatter={(v) => percent(v)} />}
                      cursor={{ fill: 'rgb(99 102 241 / .06)' }}
                    />
                    <Bar dataKey="baseline_wape" name="مبنای پیکاپ" fill={SERIES.observed} radius={[4, 4, 0, 0]} />
                    <Bar dataKey="champion_wape" name="مدل منتخب" radius={[4, 4, 0, 0]}>
                      {performance.folds.map((fold) => (
                        <Cell
                          key={fold.cutoff}
                          fill={SERIES.forecast}
                          fillOpacity={fold.cutoff === performance.hardest_fold ? 1 : 0.75}
                        />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </ChartFrame>

              <ChartFrame
                title="خطا بر حسب فاصله تا اقامت"
                unit="WAPE"
                hint="شب‌های نزدیک تقریباً حل‌شده‌اند؛ خطا در افق‌های دورتر متمرکز است."
                source="backtest_metrics_phase2.json"
                height={300}
                legend={[
                  { label: 'مدل منتخب', color: SERIES.forecast },
                  { label: 'مبنای پیکاپ', color: SERIES.observed },
                ]}
              >
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    data={performance.by_horizon_bucket.map((bucket) => ({
                      key: bucket.key,
                      champion: bucket.wape,
                      baseline:
                        performance.baseline_by_horizon_bucket.find((b) => b.key === bucket.key)?.wape ?? 0,
                    }))}
                    margin={{ top: 8, right: 12, bottom: 4, left: 4 }}
                  >
                    <CartesianGrid stroke={SERIES.grid} strokeDasharray="2 4" vertical={false} />
                    <XAxis dataKey="key" {...axisProps} tickFormatter={(v: string) => `${v} روز`} />
                    <YAxis {...axisProps} width={48} tickFormatter={(v: number) => percent(v, 0)} />
                    <Tooltip
                      content={<ChartTooltip valueFormatter={(v) => percent(v)} />}
                      cursor={{ fill: 'rgb(99 102 241 / .06)' }}
                    />
                    <Bar dataKey="baseline" name="مبنای پیکاپ" fill={SERIES.observed} radius={[4, 4, 0, 0]} />
                    <Bar dataKey="champion" name="مدل منتخب" fill={SERIES.forecast} radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </ChartFrame>
            </div>

            <div className="pol4-note">
              <h3>چرا فولد {performance.hardest_fold} سخت‌ترین است؟</h3>
              <p>
                در آن پنجره تقاضا نسبت به ۳۰ روز پیش از خودش ۱٫۵ برابر شده بود و تنها
                ۷۹٪ از پیکاپی که منحنی تاریخی انتظار داشت تا تاریخ برش رسیده بود. وقتی
                تقاضای یک دوره شتاب می‌گیرد، سهم بزرگ‌تری از جست‌وجوها دیر می‌رسد،
                منحنی میزان کامل‌شدن را بیش‌برآورد می‌کند و پیش‌بینی کم‌برآورد می‌شود.
                این فولد را پنهان نکرده‌ایم چون همان حالتی است که ویژگی‌های بازار و
                استان برای تشخیصش ساخته شده‌اند.
              </p>
            </div>

            <ChartFrame
              title="مهم‌ترین سیگنال‌های مدل"
              unit="سهم از gain"
              hint="این نمودار می‌گوید مدل بیشتر روی چه چیزی تقسیم می‌کند — نه اینکه چه چیزی علت تقاضاست."
              source="feature_importance.json"
              height={360}
            >
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  layout="vertical"
                  data={performance.feature_importance.slice(0, 12)}
                  margin={{ top: 4, right: 56, bottom: 4, left: 4 }}
                  barCategoryGap={5}
                >
                  <CartesianGrid stroke={SERIES.grid} strokeDasharray="2 4" horizontal={false} />
                  <XAxis type="number" {...axisProps} tickFormatter={(v: number) => percent(v, 0)} />
                  <YAxis type="category" dataKey="feature" {...axisProps} width={190} tick={{ fontSize: 10 }} />
                  <Tooltip
                    content={<ChartTooltip valueFormatter={(v) => percent(v)} />}
                    cursor={{ fill: 'rgb(99 102 241 / .06)' }}
                  />
                  <Bar dataKey="share" name="سهم" fill={SERIES.forecast} radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </ChartFrame>

            <div className="pol4-note">
              <h3>این ویژگی‌ها یعنی چه؟</h3>
              <ul>
                <li>
                  <code dir="ltr">city_hist_mean</code> و <code dir="ltr">city_hist_p75</code> —
                  سطح معمول تقاضای همان شهر. مدل اول اندازهٔ شهر را می‌فهمد.
                </li>
                <li>
                  <code dir="ltr">city_weekday_mean</code> — همان سطح، به تفکیک روز هفته.
                </li>
                <li>
                  <code dir="ltr">pickup_baseline_remaining</code> — تخمین منحنی پیکاپ از
                  آنچه هنوز باید برسد. این ویژگی در آزمون حذفی بیشترین ارزش نهایی را داشت.
                </li>
              </ul>
              <p className="pol4-note__caveat">
                اهمیت ویژگی یک ابزار تشخیص است، نه ادعای علیت: می‌گوید مدل روی چه چیزی
                تکیه کرده، نه اینکه چه چیزی باعث تقاضا شده است.
              </p>
            </div>
          </>
        ) : null}
      </AsyncBoundary>

      {/* --------------------------------------------------------- exports */}
      <section className="pol4-card">
        <header className="pol4-card__head">
          <div className="pol4-card__titles">
            <h2 className="pol4-card__title">خروجی CSV</h2>
            <p className="pol4-card__hint">فیلترها فقط سطرها را محدود می‌کنند؛ نحوهٔ محاسبهٔ اعداد تغییر نمی‌کند.</p>
          </div>
        </header>

        <div className="pol4-filters">
          <label>
            گزارش
            <select className="pol4-select" value={kind} onChange={(e) => setKind(e.target.value)}>
              {(reportsQuery.data?.data?.reports ?? []).map((report) => (
                <option key={report.kind} value={report.kind}>
                  {REPORT_LABELS[report.kind] ?? report.kind}
                </option>
              ))}
            </select>
          </label>
          <label>
            شهر
            <input className="pol4-input" dir="ltr" value={city} onChange={(e) => setCity(e.target.value)} placeholder="tehran" />
          </label>
          <label>
            استان
            <input className="pol4-input" dir="ltr" value={province} onChange={(e) => setProvince(e.target.value)} placeholder="gilan" />
          </label>
          <label>
            از تاریخ
            <input className="pol4-input" dir="ltr" type="date" value={start} onChange={(e) => setStart(e.target.value)} />
          </label>
          <label>
            تا تاریخ
            <input className="pol4-input" dir="ltr" type="date" value={end} onChange={(e) => setEnd(e.target.value)} />
          </label>
          <a className="pol4-button" href={pol4.reportUrl(kind, filters)} download>
            دانلود CSV
          </a>
        </div>

        <AsyncBoundary isLoading={previewQuery.isLoading} skeleton={<CardSkeleton lines={6} />}>
          {preview ? (
            <>
              <p className="pol4-card__hint">
                {num(preview.rows)} سطر · {num(preview.columns.length)} ستون · ۲۵ سطر نخست:
              </p>
              <div className="pol4-table-wrap">
                <table className="pol4-table">
                  <thead>
                    <tr>
                      {preview.columns.map((column) => (
                        <th key={column} dir="ltr">
                          {column}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {preview.preview.map((row, index) => (
                      <tr key={index}>
                        {preview.columns.map((column) => (
                          <td key={column} dir="ltr">
                            {String(row[column] ?? '')}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          ) : null}
        </AsyncBoundary>
      </section>
    </div>
  );
}
