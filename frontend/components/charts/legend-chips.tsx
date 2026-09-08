'use client';

import type { LegendProps } from 'recharts';

const LABELS: Record<string, string> = {
  actual: 'مقدار واقعی',
  backtest: 'پیش‌بینی گذشته‌نگر',
  forecast: 'پیش‌بینی آینده',
  bandSpan: 'بازه اطمینان ۸۰٪',
};

export function LegendChips({ payload, visible, onToggle }: Pick<LegendProps, 'payload'> & { visible: Record<string, boolean>; onToggle: (key: string) => void }) {
  if (!payload) return null;
  return (
    <div dir="rtl" className="chart-legend-chips">
      {payload.filter((entry) => String(entry.dataKey) !== 'bandBase').map((entry, index) => {
        const key = String(entry.dataKey ?? entry.value);
        return (
          <button key={key} type="button" aria-pressed={visible[key] !== false} onClick={() => onToggle(key)} className="chart-legend-chip" style={{ animationDelay: `${index * 40}ms` }}>
            <span className="chart-legend-chip__dot" style={{ '--series-color': entry.color } as React.CSSProperties} />
            {LABELS[key] ?? entry.value}
          </button>
        );
      })}
    </div>
  );
}
