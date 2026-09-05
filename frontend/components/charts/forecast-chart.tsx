'use client';

/**
 * The main forecast chart: history, in-sample backtest, future forecast and
 * the P10-P90 prediction band, with a marker at the forecast origin.
 *
 * The band is drawn as a stacked area (invisible base + visible span) because
 * Recharts has no native interval mark.
 */
import { useMemo } from 'react';
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
import { ChartTooltip } from '@/components/charts/tooltip';
import type { SeriesPoint } from '@/lib/types/api';
import { useCalendar } from '@/hooks/useCalendar';
import { formatCompact } from '@/lib/utils';

interface Row extends SeriesPoint {
  bandBase?: number | null;
  bandSpan?: number | null;
}

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

  return (
    <div className="chart-ltr w-full" style={{ height }}>
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={rows} margin={{ top: 8, right: 12, bottom: 4, left: 4 }}>
          <defs>
            <linearGradient id="bandFill" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={CHART.band} stopOpacity={0.22} />
              <stop offset="100%" stopColor={CHART.band} stopOpacity={0.06} />
            </linearGradient>
          </defs>

          <CartesianGrid stroke={CHART.grid} strokeDasharray="3 3" vertical={false} />
          <XAxis dataKey="ds" {...axisProps} tickFormatter={calendar.date} minTickGap={28} />
          <YAxis {...axisProps} tickFormatter={(value: number) => formatCompact(value)} width={48} />
          <Tooltip content={<ChartTooltip />} />
          <Legend
            verticalAlign="top"
            height={30}
            iconType="plainline"
            wrapperStyle={{ fontSize: 12, direction: 'rtl' }}
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
            isAnimationActive={false}
            connectNulls
          />

          <Line
            type="monotone"
            dataKey="actual"
            name="مقدار واقعی"
            stroke={CHART.actual}
            strokeWidth={2}
            dot={false}
            connectNulls
            isAnimationActive={false}
          />
          {showBacktest ? (
            <Line
              type="monotone"
              dataKey="backtest"
              name="پیش‌بینی گذشته‌نگر"
              stroke={CHART.backtest}
              strokeWidth={1.5}
              strokeDasharray="4 3"
              dot={false}
              connectNulls
              isAnimationActive={false}
            />
          ) : null}
          <Line
            type="monotone"
            dataKey="forecast"
            name="پیش‌بینی آینده"
            stroke={CHART.forecast}
            strokeWidth={2.5}
            dot={false}
            connectNulls
            isAnimationActive={false}
          />

          {forecastStart ? (
            <ReferenceLine
              x={forecastStart}
              stroke={CHART.axis}
              strokeDasharray="4 4"
              label={{
                value: 'شروع پیش‌بینی',
                position: 'insideTopLeft',
                fontSize: 11,
                fill: CHART.axis,
              }}
            />
          ) : null}
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
