'use client';

import {
  Area,
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import type { CSSProperties } from 'react';

import { CHART, axisProps } from '@/components/charts/palette';
import { ChartTooltip } from '@/components/charts/tooltip';
import { formatCompact, formatDate } from '@/lib/utils';

export function ScenarioChart({
  series,
  height = 340,
}: {
  series: { ds: string; baseline: number; scenario: number; delta: number }[];
  height?: number;
}) {
  return (
    <div
      className="chart-ltr chart-responsive w-full"
      style={{ '--chart-height': `${height}px` } as CSSProperties}
    >
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={series} margin={{ top: 8, right: 12, bottom: 4, left: 4 }}>
          <defs>
            <linearGradient id="deltaFill" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={CHART.scenario} stopOpacity={0.24} />
              <stop offset="100%" stopColor={CHART.scenario} stopOpacity={0.03} />
            </linearGradient>
          </defs>
          <CartesianGrid stroke={CHART.grid} strokeDasharray="3 3" vertical={false} />
          <XAxis dataKey="ds" {...axisProps} tickFormatter={formatDate} minTickGap={28} />
          <YAxis {...axisProps} tickFormatter={(value: number) => formatCompact(value)} width={48} />
          <Tooltip content={<ChartTooltip />} />
          <Legend
            verticalAlign="top"
            height={30}
            iconType="plainline"
            wrapperStyle={{ fontSize: 12, direction: 'rtl' }}
          />
          <Area
            type="monotone"
            dataKey="scenario"
            name="سناریو"
            stroke={CHART.scenario}
            strokeWidth={2.5}
            fill="url(#deltaFill)"
            isAnimationActive={false}
          />
          <Line
            type="monotone"
            dataKey="baseline"
            name="سناریوی پایه"
            stroke={CHART.actual}
            strokeWidth={2}
            strokeDasharray="5 4"
            dot={false}
            isAnimationActive={false}
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
