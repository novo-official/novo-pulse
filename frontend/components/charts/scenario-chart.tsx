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
import { GlassTooltip } from '@/components/charts/glass-tooltip';
import { cn } from '@/lib/utils';

const faNumber = new Intl.NumberFormat('fa-IR', { maximumFractionDigits: 1 });
const faDate = new Intl.DateTimeFormat('fa-IR-u-ca-gregory', { month: 'short', day: 'numeric' });

export function ScenarioChart({
  series,
  height = 340,
}: {
  series: { ds: string; baseline: number; scenario: number; delta: number }[];
  height?: number;
}) {
  const calendar = useCalendar();
  return (
    <div
      className="chart-ltr chart-responsive w-full"
      style={{ '--chart-height': `${height}px` } as CSSProperties}
    >
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={series} margin={{ top: 8, right: 12, bottom: 4, left: 4 }}>
          <defs>
            <linearGradient id="scenarioStroke" x1="0" y1="0" x2="1" y2="0">
              <stop offset="0%" stopColor="rgb(var(--indigo))" />
              <stop offset="100%" stopColor="rgb(var(--violet))" />
            </linearGradient>
            <linearGradient id="deltaFill" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={CHART.scenario} stopOpacity={0.24} />
              <stop offset="100%" stopColor={CHART.scenario} stopOpacity={0.03} />
            </linearGradient>
            <filter id="scenarioGlow" x="-20%" y="-30%" width="140%" height="160%">
              <feGaussianBlur stdDeviation="2" result="blur" />
              <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
            </filter>
          </defs>
          <CartesianGrid stroke={CHART.grid} strokeDasharray="3 3" vertical={false} />
          <XAxis dataKey="ds" {...axisProps} tickFormatter={(value: string) => faDate.format(new Date(`${value}T00:00:00`))} minTickGap={28} />
          <YAxis {...axisProps} tickFormatter={(value: number) => faNumber.format(value)} width={48} />
          <Tooltip content={<ScenarioTooltip />} cursor={{ stroke: 'rgb(var(--indigo) / .42)', strokeDasharray: '3 4' }} />
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
            stroke="url(#scenarioStroke)"
            strokeWidth={2.75}
            filter="url(#scenarioGlow)"
            fill="url(#deltaFill)"
            isAnimationActive
            animationDuration={900}
          />
          <Line
            type="monotone"
            dataKey="baseline"
            name="سناریوی پایه"
            stroke={CHART.actual}
            strokeWidth={2}
            strokeDasharray="5 4"
            dot={false}
            isAnimationActive
            animationDuration={900}
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}

function ScenarioTooltip({ active, payload, label }: { active?: boolean; payload?: { dataKey?: string | number; value?: number; color?: string }[]; label?: string }) {
  if (!active || !payload?.length) return null;
  const scenario = payload.find((item) => item.dataKey === 'scenario')?.value;
  const baseline = payload.find((item) => item.dataKey === 'baseline')?.value;
  const delta = typeof scenario === 'number' && typeof baseline === 'number' && baseline !== 0 ? ((scenario - baseline) / baseline) * 100 : null;
  return <GlassTooltip><p className="glass-chart-tooltip__date">{label ? new Intl.DateTimeFormat('fa-IR-u-ca-gregory', { dateStyle: 'medium' }).format(new Date(`${label}T00:00:00`)) : '—'}</p><ul className="glass-chart-tooltip__rows"><li><span className="glass-chart-tooltip__dot" style={{ '--series-color': 'rgb(var(--indigo))' } as CSSProperties} /><span>سناریو</span><strong>{typeof scenario === 'number' ? faNumber.format(scenario) : '—'}</strong></li><li><span className="glass-chart-tooltip__dot" style={{ '--series-color': 'rgb(var(--ink) / .6)' } as CSSProperties} /><span>سناریوی پایه</span><strong>{typeof baseline === 'number' ? faNumber.format(baseline) : '—'}</strong></li></ul>{delta !== null ? <bdi dir="ltr" className={cn('glass-chart-tooltip__delta', delta >= 0 ? 'glass-chart-tooltip__delta--up' : 'glass-chart-tooltip__delta--down')}>اختلاف {new Intl.NumberFormat('fa-IR', { maximumFractionDigits: 1, signDisplay: 'always' }).format(delta)}٪</bdi> : null}</GlassTooltip>;
}
