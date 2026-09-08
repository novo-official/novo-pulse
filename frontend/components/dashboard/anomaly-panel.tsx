'use client';

import { AlertTriangle, TrendingDown, TrendingUp } from 'lucide-react';

import { Badge } from '@/components/ui/badge';
import { Card, CardBody, CardHeader } from '@/components/ui/card';
import { EmptyState } from '@/components/ui/states';
import { ScrollList } from '@/components/ui/scroll-list';
import { TypePill } from '@/components/ui/type-pill';
import { useCalendar } from '@/hooks/useCalendar';
import type { Anomaly } from '@/lib/types/api';
import { SEVERITY_FA, cn } from '@/lib/utils';

const faNumber = new Intl.NumberFormat('fa-IR');
const faDate = new Intl.DateTimeFormat('fa-IR-u-ca-gregory', { day: 'numeric', month: 'long', year: 'numeric' });
const faPercent = new Intl.NumberFormat('fa-IR', { style: 'percent', signDisplay: 'always', maximumFractionDigits: 0 });

export function AnomalyPanel({ anomalies }: { anomalies: Anomaly[] }) {
  const calendar = useCalendar();
  return (
    <Card className="scroll-card anomaly-card h-full">
      <CardHeader
        icon={<AlertTriangle className="h-4.5 w-4.5" />}
        title="ناهنجاری‌های تقاضا"
        subtitle="انحراف معنادار از انتظار مدل، بر پایه امتیاز Z مقاوم (MAD)"
        action={<Badge className="anomaly-card__count" tone="neutral">{faNumber.format(anomalies.length)} مورد</Badge>}
      />
      <CardBody className="scroll-card__body">
        {anomalies.length === 0 ? (
          <EmptyState
            title="ناهنجاری مهمی شناسایی نشد"
            description="تقاضا در محدوده مورد انتظار مدل قرار دارد."
          />
        ) : (
          <ScrollList label="ناهنجاری‌های تقاضا"><ul className="scroll-card-list">
            {anomalies.map((anomaly, index) => {
              const spike = anomaly.type === 'spike';
              return (
                <li
                  key={`${anomaly.entity_id}-${anomaly.ds}-${index}`}
                  className={cn('scroll-card-row anomaly-row', spike ? 'anomaly-row--up' : 'anomaly-row--down')}
                  tabIndex={0}
                >
                  <span
                    className={cn(
                      'scroll-card-row__icon flex h-8 w-8 shrink-0 items-center justify-center rounded-[10px]',
                      spike ? 'bg-emerald-50 text-emerald-600' : 'bg-rose-50 text-rose-600',
                    )}
                  >
                    {spike ? (
                      <TrendingUp className="h-4 w-4" />
                    ) : (
                      <TrendingDown className="h-4 w-4" />
                    )}
                  </span>
                  <div className="min-w-0 flex-1">
                    <p className="flex items-center gap-2 text-[13.5px] font-bold text-ink">
                      <span className="truncate">{anomaly.label}</span>
                      <span className="scroll-card-row__delta shrink-0 text-xs font-extrabold text-muted">
                        {anomaly.deviation !== null
                          ? <bdi dir="ltr">{faPercent.format(anomaly.deviation)}</bdi>
                          : <bdi dir="ltr">Z={anomaly.score.toFixed(1)}</bdi>}
                      </span>
                    </p>
                    <p className="mt-0.5 text-xs text-muted">
                      {anomaly.type_fa} · {faDate.format(new Date(anomaly.ds))}
                    </p>
                  </div>
                  <TypePill tone={anomaly.severity === 'low' ? 'neutral' : anomaly.severity === 'medium' ? 'warn' : 'bad'} className="shrink-0">
                    {SEVERITY_FA[anomaly.severity] ?? anomaly.severity}
                  </TypePill>
                </li>
              );
            })}
          </ul></ScrollList>
        )}
      </CardBody>
    </Card>
  );
}
