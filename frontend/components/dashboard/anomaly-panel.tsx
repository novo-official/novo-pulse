'use client';

import { AlertTriangle, TrendingDown, TrendingUp } from 'lucide-react';

import { Badge, toneForSeverity } from '@/components/ui/badge';
import { Card, CardBody, CardHeader } from '@/components/ui/card';
import { EmptyState } from '@/components/ui/states';
import type { Anomaly } from '@/lib/types/api';
import { useCalendar } from '@/hooks/useCalendar';
import { SEVERITY_FA, cn, formatNumber, formatPercent } from '@/lib/utils';

export function AnomalyPanel({ anomalies }: { anomalies: Anomaly[] }) {
  const calendar = useCalendar();
  return (
    <Card className="h-full">
      <CardHeader
        icon={<AlertTriangle className="h-4.5 w-4.5" />}
        title="ناهنجاری‌های تقاضا"
        subtitle="انحراف معنادار از انتظار مدل، بر پایه امتیاز Z مقاوم (MAD)"
        action={<Badge tone="neutral">{formatNumber(anomalies.length)} مورد</Badge>}
      />
      <CardBody className="max-h-[420px] overflow-y-auto">
        {anomalies.length === 0 ? (
          <EmptyState
            title="ناهنجاری مهمی شناسایی نشد"
            description="تقاضا در محدوده مورد انتظار مدل قرار دارد."
          />
        ) : (
          <ul className="space-y-2.5">
            {anomalies.map((anomaly, index) => {
              const spike = anomaly.type === 'spike';
              return (
                <li
                  key={`${anomaly.entity_id}-${anomaly.ds}-${index}`}
                  className="flex items-center gap-3 rounded-xl border border-line/70 px-3 py-2.5"
                >
                  <span
                    className={cn(
                      'flex h-8 w-8 shrink-0 items-center justify-center rounded-lg',
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
                    <p className="flex items-center gap-2 text-sm font-medium text-ink">
                      <span className="truncate">{anomaly.label}</span>
                      <span className="nums shrink-0 text-xs font-semibold text-muted">
                        {anomaly.deviation !== null
                          ? formatPercent(anomaly.deviation * 100, 0)
                          : `Z=${anomaly.score.toFixed(1)}`}
                      </span>
                    </p>
                    <p className="mt-0.5 text-xs text-muted">
                      {anomaly.type_fa} · {calendar.fullDate(anomaly.ds)}
                    </p>
                  </div>
                  <Badge tone={toneForSeverity(anomaly.severity)} className="shrink-0">
                    {SEVERITY_FA[anomaly.severity] ?? anomaly.severity}
                  </Badge>
                </li>
              );
            })}
          </ul>
        )}
      </CardBody>
    </Card>
  );
}
