'use client';

import { ArrowDown, ArrowUp, Lightbulb } from 'lucide-react';

import { Card, CardBody, CardHeader } from '@/components/ui/card';
import { EmptyState } from '@/components/ui/states';
import { InfoHint } from '@/components/ui/tooltip';
import type { DriverGroup, DriversResponse } from '@/lib/types/api';
import { cn } from '@/lib/utils';

const faPercent = new Intl.NumberFormat('fa-IR', { style: 'percent', maximumFractionDigits: 1 });

function DriverRow({ driver, max, index }: { driver: DriverGroup; max: number; index: number }) {
  const positive = driver.direction === 'positive';
  const width = max > 0 ? Math.max((driver.contribution_share / max) * 100, 3) : 0;
  return (
    <li className="factor-row" tabIndex={0}>
      <span
        className={cn(
          'factor-row__direction flex h-7 w-7 shrink-0 items-center justify-center rounded-[9px]',
          positive ? 'bg-emerald-50 text-emerald-600' : 'bg-rose-50 text-rose-600',
        )}
      >
        {positive ? <ArrowUp className="h-3.5 w-3.5" /> : <ArrowDown className="h-3.5 w-3.5" />}
      </span>
      <span className="factor-row__content">
        <span className="flex items-baseline justify-between gap-2">
          <span className="truncate text-[12.5px] font-semibold text-ink">{driver.label_fa}</span>
          <span className={cn('factor-row__percent shrink-0', positive ? 'text-emerald-600' : 'text-rose-600')}>
            {faPercent.format(driver.contribution_share)}
          </span>
        </span>
        <span className="factor-row__track mt-2 block h-1.5 w-full overflow-hidden rounded-full">
          <span
            className={cn('factor-row__fill block h-full rounded-full', positive ? 'factor-row__fill--positive' : 'factor-row__fill--negative')}
            style={{ '--factor-width': `${width}%`, animationDelay: `${index * 40}ms` } as React.CSSProperties}
          />
        </span>
      </span>
    </li>
  );
}

export function DriversCard({ drivers }: { drivers: DriversResponse }) {
  const groups = drivers.groups ?? [];
  const max = groups.reduce((acc, group) => Math.max(acc, group.contribution_share), 0);

  return (
    <Card className="insights-card factor-importance-card h-full">
      <CardHeader
        icon={<Lightbulb className="h-4.5 w-4.5" />}
        title="چه عواملی تقاضا را تغییر داده‌اند؟"
        subtitle="سهم هر گروه از مجموع اثر مدل، بر پایه مقادیر SHAP"
        action={
          <InfoHint label="روش محاسبه">
            سهم هر گروه = میانگین قدر مطلق اثر SHAP ویژگی‌های آن گروه، تقسیم بر مجموع کل. جهت
            (افزایشی/کاهشی) از میانگین علامت‌دار اثر می‌آید. روش: {drivers.method}
          </InfoHint>
        }
      />
      <CardBody className="insights-card__body">
        {groups.length === 0 ? (
          <EmptyState title="اثر عوامل محاسبه نشده است" description={drivers.note} />
        ) : (
          <ul className="factor-list">
            {groups.map((group, index) => (
              <DriverRow key={group.group} driver={group} max={max} index={index} />
            ))}
          </ul>
        )}
        {drivers.note ? <p className="mt-4 text-xs leading-5 text-muted">{drivers.note}</p> : null}
      </CardBody>
    </Card>
  );
}
