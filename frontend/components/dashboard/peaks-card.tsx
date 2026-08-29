'use client';

import { CalendarRange, TrendingDown, TrendingUp } from 'lucide-react';

import { Card, CardBody, CardHeader } from '@/components/ui/card';
import { EmptyState } from '@/components/ui/states';
import { ScrollList } from '@/components/ui/scroll-list';
import { TypePill } from '@/components/ui/type-pill';
import type { PeakPeriod } from '@/lib/types/api';
import { CONFIDENCE_FA, cn } from '@/lib/utils';

const faDate = new Intl.DateTimeFormat('fa-IR-u-ca-gregory', { day: 'numeric', month: 'long', year: 'numeric' });
const faPercent = new Intl.NumberFormat('fa-IR', { style: 'percent', signDisplay: 'always', maximumFractionDigits: 0 });

function PeakRow({ period }: { period: PeakPeriod }) {
  const up = period.type === 'peak';
  return (
    <li className={cn('scroll-card-row peak-row', up ? 'peak-row--up' : 'peak-row--down')} tabIndex={0}>
      <span
        className={cn(
          'scroll-card-row__icon flex h-8 w-8 shrink-0 items-center justify-center rounded-[10px]',
          up ? 'bg-emerald-50 text-emerald-600' : 'bg-rose-50 text-rose-600',
        )}
      >
        {up ? <TrendingUp className="h-4 w-4" /> : <TrendingDown className="h-4 w-4" />}
      </span>
      <div className="min-w-0 flex-1">
        <p className="truncate text-[13.5px] font-bold text-ink">{period.label}</p>
        <p className="mt-0.5 text-xs text-muted">
          {faDate.format(new Date(period.start))} تا {faDate.format(new Date(period.end))} · {new Intl.NumberFormat('fa-IR').format(period.days)} روز
        </p>
      </div>
      <div className="shrink-0 text-left">
        <p
          className={cn(
            'scroll-card-row__delta text-sm font-extrabold',
            up ? 'text-emerald-600' : 'text-rose-600',
          )}
        >
          <bdi dir="ltr">{faPercent.format(period.expected_change_pct / 100)}</bdi>
        </p>
        <TypePill tone={period.confidence === 'high' ? 'ok' : period.confidence === 'medium' ? 'warn' : 'bad'} className="mt-1">
          {CONFIDENCE_FA[period.confidence] ?? period.confidence}
        </TypePill>
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
    <Card className="scroll-card h-full">
      <CardHeader
        icon={<CalendarRange className="h-4.5 w-4.5" />}
        title="دوره‌های اوج و افت پیش‌رو"
        subtitle="در مقایسه با سطح معمول همان روز هفته، نه میانگین ساده"
      />
      <CardBody className="scroll-card__body">
        {empty ? (
          <EmptyState
            title="دوره غیرعادی‌ای پیش‌بینی نشده"
            description="تقاضا در افق انتخاب‌شده نزدیک به الگوی فصلی معمول باقی می‌ماند."
          />
        ) : null}
        {!empty ? <><ScrollList label="دوره‌های اوج و افت پیش‌رو"><ul className="scroll-card-list">{[...peaks, ...troughs].map((period, index) => <PeakRow key={`${period.entity_id}-${period.start}-${index}`} period={period} />)}</ul></ScrollList><div className="scroll-card-legend"><span><i className="scroll-card-legend__dot scroll-card-legend__dot--up" />اوج تقاضا</span><span><i className="scroll-card-legend__dot scroll-card-legend__dot--down" />افت تقاضا</span></div></> : null}
      </CardBody>
    </Card>
  );
}
