'use client';

/**
 * Destination x date demand heatmap.
 *
 * Colour encodes each row's deviation from its own median, so destinations of
 * very different sizes stay comparable in one grid.
 */
import { Grid3x3 } from 'lucide-react';

import { Card, CardBody, CardHeader } from '@/components/ui/card';
import { EmptyState } from '@/components/ui/states';
import { InfoHint } from '@/components/ui/tooltip';
import type { HeatmapResponse } from '@/lib/types/api';
import { formatDate, formatNumber } from '@/lib/utils';

function cellColor(intensity: number | null): string {
  if (intensity === null || !Number.isFinite(intensity)) return 'rgb(241 245 249)';
  const clamped = Math.max(-0.6, Math.min(0.6, intensity));
  const strength = Math.abs(clamped) / 0.6;
  const alpha = 0.1 + strength * 0.85;
  return clamped >= 0
    ? `rgba(53, 99, 233, ${alpha.toFixed(3)})`
    : `rgba(225, 29, 72, ${(alpha * 0.85).toFixed(3)})`;
}

export function DemandHeatmap({ data }: { data: HeatmapResponse }) {
  const { dates, rows } = data;
  const step = Math.max(1, Math.ceil(dates.length / 12));

  return (
    <Card>
      <CardHeader
        icon={<Grid3x3 className="h-4.5 w-4.5" />}
        title="نقشه حرارتی تقاضا"
        subtitle="مقصد × تاریخ — برای شناسایی سریع دوره‌های اوج"
        action={
          <InfoHint>
            رنگ هر خانه، انحراف پیش‌بینی آن روز از میانه همان مقصد را نشان می‌دهد. آبی = بالاتر از
            معمول، قرمز = پایین‌تر.
          </InfoHint>
        }
      />
      <CardBody>
        {rows.length === 0 ? (
          <EmptyState title="داده‌ای برای نقشه حرارتی نیست" />
        ) : (
          <div className="overflow-x-auto pb-1">
            <div className="min-w-[720px]">
              <div
                className="grid gap-[3px]"
                style={{ gridTemplateColumns: `110px repeat(${dates.length}, minmax(0, 1fr))` }}
              >
                <span />
                {dates.map((date, index) => (
                  <span
                    key={date}
                    className="nums h-4 text-center text-[9px] leading-4 text-muted"
                    title={date}
                  >
                    {index % step === 0 ? new Date(date).getDate() : ''}
                  </span>
                ))}

                {rows.map((row) => (
                  <div key={row.entity_id} className="contents">
                    <span className="truncate pl-2 text-xs leading-6 text-ink" title={row.label}>
                      {row.label}
                    </span>
                    {row.values.map((value, index) => (
                      <span
                        key={`${row.entity_id}-${dates[index]}`}
                        className="h-6 rounded-[3px] transition hover:ring-2 hover:ring-brand-400"
                        style={{ backgroundColor: cellColor(row.intensity[index]) }}
                        title={`${row.label} · ${formatDate(dates[index])} · ${formatNumber(value, 1)}`}
                      />
                    ))}
                  </div>
                ))}
              </div>

              <div className="mt-4 flex items-center justify-center gap-2 text-[11px] text-muted">
                <span>کمتر از معمول</span>
                <span className="flex">
                  {[-0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6].map((value) => (
                    <span
                      key={value}
                      className="h-3 w-6 first:rounded-r-sm last:rounded-l-sm"
                      style={{ backgroundColor: cellColor(value) }}
                    />
                  ))}
                </span>
                <span>بیشتر از معمول</span>
              </div>
            </div>
          </div>
        )}
      </CardBody>
    </Card>
  );
}
