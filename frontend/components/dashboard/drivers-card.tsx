'use client';

import { ArrowDown, ArrowUp, Lightbulb } from 'lucide-react';

import { Card, CardBody, CardHeader } from '@/components/ui/card';
import { EmptyState } from '@/components/ui/states';
import { InfoHint } from '@/components/ui/tooltip';
import type { DriverGroup, DriversResponse } from '@/lib/types/api';
import { cn, formatRatioAsPercent } from '@/lib/utils';

function DriverRow({ driver, max }: { driver: DriverGroup; max: number }) {
  const positive = driver.direction === 'positive';
  const width = max > 0 ? Math.max((driver.contribution_share / max) * 100, 3) : 0;
  return (
    <li className="flex items-center gap-3">
      <span
        className={cn(
          'flex h-6 w-6 shrink-0 items-center justify-center rounded-md',
          positive ? 'bg-emerald-50 text-emerald-600' : 'bg-rose-50 text-rose-600',
        )}
      >
        {positive ? <ArrowUp className="h-3.5 w-3.5" /> : <ArrowDown className="h-3.5 w-3.5" />}
      </span>
      <span className="min-w-0 flex-1">
        <span className="flex items-baseline justify-between gap-2">
          <span className="truncate text-sm text-ink">{driver.label_fa}</span>
          <span className="nums shrink-0 text-xs font-semibold text-muted">
            {formatRatioAsPercent(driver.contribution_share)}
          </span>
        </span>
        <span className="mt-1.5 block h-1.5 w-full overflow-hidden rounded-full bg-slate-100">
          <span
            className={cn('block h-full rounded-full', positive ? 'bg-emerald-500' : 'bg-rose-500')}
            style={{ width: `${width}%` }}
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
    <Card className="h-full">
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
      <CardBody>
        {groups.length === 0 ? (
          <EmptyState title="اثر عوامل محاسبه نشده است" description={drivers.note} />
        ) : (
          <ul className="space-y-3.5">
            {groups.map((group) => (
              <DriverRow key={group.group} driver={group} max={max} />
            ))}
          </ul>
        )}
        {drivers.note ? <p className="mt-4 text-xs leading-5 text-muted">{drivers.note}</p> : null}
      </CardBody>
    </Card>
  );
}
