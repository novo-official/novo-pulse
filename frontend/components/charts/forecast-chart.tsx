'use client';

/**
 * The main forecast chart: history, in-sample backtest, future forecast and
 * the P10-P90 prediction band, with a marker at the forecast origin.
 *
 * The band is drawn as a stacked area (invisible base + visible span) because
 * Recharts has no native interval mark.
 */
import { useMemo, useState, type CSSProperties } from 'react';
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

import { CHART, axisProps } from '@/components/charts/palette';
import { GlassTooltip } from '@/components/charts/glass-tooltip';
import { LegendChips } from '@/components/charts/legend-chips';
import type { SeriesPoint } from '@/lib/types/api';

interface Row extends SeriesPoint {
  bandBase?: number | null;
  bandSpan?: number | null;
}

const faCompact = new Intl.NumberFormat('fa-IR', { notation: 'compact', maximumFractionDigits: 1 });
const faDate = new Intl.DateTimeFormat('fa-IR', { day: 'numeric', month: 'short' });

export function ForecastChart({
  series,
  forecastStart,
  height = 380,
  showBacktest = true,
}: {
  series: SeriesPoint[];
  forecastStart?: string | null;
  height?: number;
  showBacktest?: boolean;
}) {
  const calendar = useCalendar();
  const rows = useMemo<Row[]>(
    () =>
      series.map((point) => {
        const hasBand =
          point.lower !== undefined &&
          point.upper !== undefined &&
          Number.isFinite(point.lower) &&
          Number.isFinite(point.upper);
        return {
          ...point,
          bandBase: hasBand ? point.lower : null,
          bandSpan: hasBand ? (point.upper as number) - (point.lower as number) : null,
        };
      }),
    [series],
  );
  const [visible, setVisible] = useState<Record<string, boolean>>({ actual: true, backtest: true, forecast: true, bandSpan: true });
  const toggle = (key: string) => setVisible((current) => ({ ...current, [key]: current[key] === false }));
  const latestActualIndex = useMemo(() => rows.reduce((latest, row, index) => row.actual !== undefined && row.actual !== null ? index : latest, -1), [rows]);

  return (
    <div
      className="forecast-glass-chart chart-ltr chart-responsive w-full"
      style={{ '--chart-height': `${height}px` } as CSSProperties}
      tabIndex={0}
      role="img"
      aria-label="نمودار روند تقاضا"
    >
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={rows} margin={{ top: 10, right: 12, bottom: 4, left: 4 }}>
          <defs>
            <linearGradient id="bandFill" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#6366f1" stopOpacity={0.16} />
              <stop offset="100%" stopColor="#8b5cf6" stopOpacity={0.03} />
            </linearGradient>
            <linearGradient id="forecastStroke" x1="0" y1="0" x2="1" y2="0"><stop offset="0%" stopColor="#6366f1" /><stop offset="100%" stopColor="#8b5cf6" /></linearGradient>
            <filter id="forecastGlow" x="-20%" y="-20%" width="140%" height="140%"><feDropShadow dx="0" dy="0" stdDeviation="3" floodColor="#6366f1" floodOpacity="0.55" /></filter>
          </defs>

          <CartesianGrid stroke="rgb(15 23 42 / 0.08)" strokeDasharray="2 4" vertical={false} />
          <XAxis dataKey="ds" {...axisProps} tickFormatter={(value) => faDate.format(new Date(value))} minTickGap={36} />
          <YAxis {...axisProps} tickFormatter={(value: number) => faCompact.format(value)} width={52} />
          <Tooltip content={<GlassTooltip />} cursor={{ stroke: 'rgb(15 23 42 / 0.15)', strokeWidth: 1 }} />
          <Legend
            verticalAlign="top"
            height={42}
            content={(props) => <LegendChips {...props} visible={visible} onToggle={toggle} />}
          />

          {/* Invisible base + visible span = the P10-P90 ribbon. */}
          <Area
            dataKey="bandBase"
            stackId="band"
            stroke="none"
            fill="transparent"
            legendType="none"
            isAnimationActive={false}
            connectNulls
          />
          <Area
            dataKey="bandSpan"
            stackId="band"
            stroke="none"
            fill="url(#bandFill)"
            name="بازه اطمینان ۸۰٪"
            hide={visible.bandSpan === false}
            isAnimationActive
            animationDuration={400}
            connectNulls
          />

          <Line
            type="monotone"
            dataKey="actual"
            name="مقدار واقعی"
            stroke={CHART.actual}
            strokeWidth={2.25}
            hide={visible.actual === false}
            dot={(props) => props.index === latestActualIndex ? <g key={`latest-actual-${props.index}`}><circle cx={props.cx} cy={props.cy} r="7" className="forecast-latest-halo" /><circle cx={props.cx} cy={props.cy} r="4" fill={CHART.actual} stroke="white" strokeWidth="2" /></g> : <circle key={`actual-${props.index}`} cx={props.cx} cy={props.cy} r="0" />}
            activeDot={{ r: 5, fill: CHART.actual, stroke: 'white', strokeWidth: 2 }}
            connectNulls
            isAnimationActive
            animationDuration={900}
          />
          {showBacktest ? (
            <Line
              type="monotone"
              dataKey="backtest"
              name="پیش‌بینی گذشته‌نگر"
              stroke={CHART.backtest}
              strokeWidth={2}
              strokeDasharray="6 4"
              strokeLinecap="round"
              hide={visible.backtest === false}
              dot={false}
              activeDot={{ r: 5, fill: CHART.backtest, stroke: 'white', strokeWidth: 2 }}
              connectNulls
              isAnimationActive
              animationDuration={900}
            />
          ) : null}
          <Line
            type="monotone"
            dataKey="forecast"
            name="پیش‌بینی آینده"
            stroke="url(#forecastStroke)"
            strokeWidth={2.75}
            filter="url(#forecastGlow)"
            hide={visible.forecast === false}
            dot={false}
            activeDot={{ r: 5, fill: CHART.forecast, stroke: 'white', strokeWidth: 2 }}
            connectNulls
            isAnimationActive
            animationDuration={900}
          />

          {forecastStart ? (
            <ReferenceLine
              x={forecastStart}
              stroke="rgb(100 116 139 / 0.75)"
              strokeWidth={1.5}
              strokeDasharray="5 5"
              label={{
                value: 'شروع پیش‌بینی',
                position: 'insideTopLeft',
                fontSize: 11,
                fill: '#64748b',
              }}
            />
          ) : null}
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
