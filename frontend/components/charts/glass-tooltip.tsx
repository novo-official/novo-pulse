'use client';

import type { ReactNode } from 'react';

type Entry = { dataKey?: string | number; name?: string; value?: number | string; color?: string };

const LABELS: Record<string, string> = {
  actual: 'مقدار واقعی',
  backtest: 'پیش‌بینی گذشته‌نگر',
  forecast: 'پیش‌بینی آینده',
};

const number = new Intl.NumberFormat('fa-IR', { maximumFractionDigits: 1 });
const date = new Intl.DateTimeFormat('fa-IR', { weekday: 'long', day: 'numeric', month: 'long' });

export function GlassTooltip({ active, payload, label }: { active?: boolean; payload?: Entry[]; label?: string | number }) {
  if (!active || !payload?.length) return null;
  const rows = payload.filter((item) => ['actual', 'backtest', 'forecast'].includes(String(item.dataKey)) && item.value !== null && item.value !== undefined);
  if (!rows.length) return null;
  const actual = rows.find((item) => item.dataKey === 'actual')?.value;
  const forecast = rows.find((item) => item.dataKey === 'forecast')?.value;
  const delta = typeof actual === 'number' && typeof forecast === 'number' && actual !== 0
    ? ((forecast - actual) / actual) * 100
    : null;

  return (
    <div dir="rtl" role="status" className="glass-chart-tooltip">
      <p className="glass-chart-tooltip__date">{label ? date.format(new Date(String(label))) : '—'}</p>
      <ul className="glass-chart-tooltip__rows">
        {rows.map((item) => (
          <li key={String(item.dataKey)}>
            <span className="glass-chart-tooltip__dot" style={{ '--series-color': item.color } as React.CSSProperties} />
            <span>{LABELS[String(item.dataKey)] ?? item.name}</span>
            <strong>{typeof item.value === 'number' ? number.format(item.value) : item.value}</strong>
          </li>
        ))}
      </ul>
      {delta !== null ? <span className={delta >= 0 ? 'glass-chart-tooltip__delta glass-chart-tooltip__delta--up' : 'glass-chart-tooltip__delta glass-chart-tooltip__delta--down'}>{delta >= 0 ? '▲' : '▼'} {number.format(Math.abs(delta))}٪</span> : null}
    </div>
  );
}
