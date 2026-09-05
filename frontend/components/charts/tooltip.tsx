'use client';

import { useCalendar } from '@/hooks/useCalendar';
import { formatNumber } from '@/lib/utils';

interface Entry {
  name?: string;
  dataKey?: string | number;
  value?: number | string;
  color?: string;
}

const LABELS: Record<string, string> = {
  actual: 'مقدار واقعی',
  forecast: 'پیش‌بینی',
  backtest: 'پیش‌بینی گذشته‌نگر',
  prediction: 'پیش‌بینی گذشته‌نگر',
  lower: 'کران پایین (P10)',
  upper: 'کران بالا (P90)',
  band: 'بازه اطمینان ۸۰٪',
  baseline: 'سناریوی پایه',
  scenario: 'سناریو',
  delta: 'اختلاف',
};

/** Shared Recharts tooltip. Kept RTL while the chart body stays LTR. */
export function ChartTooltip({
  active,
  payload,
  label,
  valueDigits = 1,
  formatLabel,
}: {
  active?: boolean;
  payload?: Entry[];
  label?: string | number;
  valueDigits?: number;
  formatLabel?: (value: string) => string;
}) {
  const calendar = useCalendar();
  const label_ = formatLabel ?? calendar.date;
  if (!active || !payload?.length) return null;

  const rows = payload.filter(
    (entry) => entry.value !== undefined && entry.value !== null && entry.dataKey !== 'bandBase',
  );
  if (!rows.length) return null;

  return (
    <div dir="rtl" className="rounded-xl border border-line bg-surface/98 px-3 py-2 shadow-lift">
      <p className="mb-1.5 text-xs font-semibold text-ink">{label_(String(label ?? ''))}</p>
      <ul className="space-y-1">
        {rows.map((entry) => (
          <li key={String(entry.dataKey)} className="flex items-center gap-2 text-xs">
            <span
              className="h-2 w-2 shrink-0 rounded-full"
              style={{ backgroundColor: entry.color ?? '#94a3b8' }}
            />
            <span className="text-muted">{LABELS[String(entry.dataKey)] ?? entry.name}</span>
            <span className="nums mr-auto font-medium text-ink">
              {typeof entry.value === 'number'
                ? formatNumber(entry.value, valueDigits)
                : entry.value}
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}
