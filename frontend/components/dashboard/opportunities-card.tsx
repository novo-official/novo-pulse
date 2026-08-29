'use client';

import { AlertCircle, Lightbulb, TrendingDown, TrendingUp, Zap } from 'lucide-react';

import { Badge, toneForChange } from '@/components/ui/badge';
import { Card, CardBody, CardHeader } from '@/components/ui/card';
import { EmptyState } from '@/components/ui/states';
import type { Opportunity } from '@/lib/types/api';
import { cn, formatPercent } from '@/lib/utils';

const ICONS = {
  growth: TrendingUp,
  decline: TrendingDown,
  peak: Zap,
  uncertainty: AlertCircle,
} as const;

const TONES = {
  growth: 'bg-emerald-50 text-emerald-600',
  decline: 'bg-rose-50 text-rose-600',
  peak: 'bg-amber-50 text-amber-600',
  uncertainty: 'bg-slate-100 text-slate-500',
} as const;

export function OpportunitiesCard({ opportunities }: { opportunities: Opportunity[] }) {
  return (
    <Card className="signal-card h-full bg-gradient-to-b from-amber-50/30 to-surface">
      <CardHeader
        icon={<Lightbulb className="h-4.5 w-4.5" />}
        title="فرصت‌های قابل اقدام"
        subtitle="هر کارت تنها چیزی را می‌گوید که داده پشتیبانی می‌کند"
      />
      <CardBody>
        {opportunities.length === 0 ? (
          <EmptyState
            title="فرصت قابل‌توجهی شناسایی نشد"
            description="تغییرات پیش‌بینی‌شده در محدوده معمول بازار است."
          />
        ) : (
          <ul className="space-y-3">
            {opportunities.map((item, index) => {
              const Icon = ICONS[item.kind] ?? Lightbulb;
              return (
                <li
                  key={`${item.kind}-${item.entity_id}-${index}`}
                  className="insight-row overflow-hidden p-3.5 before:absolute before:inset-y-3 before:right-0 before:w-0.5 before:rounded-full before:bg-amber-400"
                >
                  <div className="flex items-start gap-3">
                    <span
                      className={cn(
                        'flex h-8 w-8 shrink-0 items-center justify-center rounded-lg',
                        TONES[item.kind] ?? TONES.uncertainty,
                      )}
                    >
                      <Icon className="h-4 w-4" />
                    </span>
                    <div className="min-w-0 flex-1">
                      <p className="flex flex-wrap items-center gap-2 text-sm font-semibold text-ink">
                        {item.headline_fa}
                        {item.change_pct !== null && item.change_pct !== undefined ? (
                          <Badge tone={toneForChange(item.change_pct)}>
                            {formatPercent(item.change_pct)}
                          </Badge>
                        ) : null}
                      </p>
                      <p className="mt-1.5 text-xs leading-6 text-muted">{item.evidence_fa}</p>
                    </div>
                  </div>
                </li>
              );
            })}
          </ul>
        )}
      </CardBody>
    </Card>
  );
}
