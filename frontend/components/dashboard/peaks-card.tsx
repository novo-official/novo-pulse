'use client';

import { CalendarRange, TrendingDown, TrendingUp } from 'lucide-react';

import { Badge, toneForConfidence } from '@/components/ui/badge';
import { Card, CardBody, CardHeader } from '@/components/ui/card';
import { EmptyState } from '@/components/ui/states';
import type { PeakPeriod } from '@/lib/types/api';
import { useCalendar } from '@/hooks/useCalendar';
import { CONFIDENCE_FA, cn, formatPercent } from '@/lib/utils';

function PeakRow({ period }: { period: PeakPeriod }) {
  const calendar = useCalendar();
  const up = period.type === 'peak';
  return (
    <li className="flex items-center gap-3 rounded-xl border border-line/70 px-3 py-2.5">
      <span
        className={cn(
          'flex h-8 w-8 shrink-0 items-center justify-center rounded-lg',
          up ? 'bg-emerald-50 text-emerald-600' : 'bg-sky-50 text-sky-600',
        )}
      >
        {up ? <TrendingUp className="h-4 w-4" /> : <TrendingDown className="h-4 w-4" />}
      </span>
      <div className="min-w-0 flex-1">
        <p className="truncate text-sm font-medium text-ink">{period.label}</p>
        <p className="nums mt-0.5 text-xs text-muted">
          {calendar.date(period.start)} تا {calendar.date(period.end)} · {period.days} روز
        </p>
      </div>
      <div className="shrink-0 text-left">
        <p
          className={cn(
            'nums text-sm font-semibold',
            up ? 'text-emerald-600' : 'text-sky-600',
          )}
        >
          {formatPercent(period.expected_change_pct, 0)}
        </p>
        <Badge tone={toneForConfidence(period.confidence)} className="mt-1">
          {CONFIDENCE_FA[period.confidence] ?? period.confidence}
        </Badge>
      </div>
    </li>
  );
}

export function PeaksCard({
  peaks,
  troughs,
}: {
  peaks: PeakPeriod[];
  troughs: PeakPeriod[];
}) {
  const empty = peaks.length === 0 && troughs.length === 0;
  return (
    <Card className="h-full">
      <CardHeader
        icon={<CalendarRange className="h-4.5 w-4.5" />}
        title="دوره‌های اوج و افت پیش‌رو"
        subtitle="در مقایسه با سطح معمول همان روز هفته، نه میانگین ساده"
      />
      <CardBody className="max-h-[420px] space-y-4 overflow-y-auto">
        {empty ? (
          <EmptyState
            title="دوره غیرعادی‌ای پیش‌بینی نشده"
            description="تقاضا در افق انتخاب‌شده نزدیک به الگوی فصلی معمول باقی می‌ماند."
          />
        ) : null}
        {peaks.length > 0 ? (
          <div>
            <p className="mb-2 text-xs font-semibold text-muted">اوج تقاضا</p>
            <ul className="space-y-2.5">
              {peaks.map((period, index) => (
                <PeakRow key={`${period.entity_id}-${period.start}-${index}`} period={period} />
              ))}
            </ul>
          </div>
        ) : null}
        {troughs.length > 0 ? (
          <div>
            <p className="mb-2 text-xs font-semibold text-muted">افت تقاضا</p>
            <ul className="space-y-2.5">
              {troughs.map((period, index) => (
                <PeakRow key={`${period.entity_id}-${period.start}-${index}`} period={period} />
              ))}
            </ul>
          </div>
        ) : null}
      </CardBody>
    </Card>
  );
}
