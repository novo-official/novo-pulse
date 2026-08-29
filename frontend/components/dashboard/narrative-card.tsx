'use client';

import { Bot, ShieldCheck } from 'lucide-react';

import { Badge } from '@/components/ui/badge';
import { Card, CardBody, CardHeader } from '@/components/ui/card';
import { Skeleton } from '@/components/ui/states';
import type { NarrativeResponse } from '@/lib/types/api';

export function NarrativeCard({
  narrative,
  isLoading,
}: {
  narrative: NarrativeResponse | null;
  isLoading: boolean;
}) {
  return (
    <Card className="signal-card h-full bg-gradient-to-br from-brand-50/80 via-violet-50/30 to-surface">
      <CardHeader
        icon={<Bot className="h-4.5 w-4.5" />}
        title="تحلیل هوشمند"
        subtitle="متن بر پایه اعداد محاسبه‌شده تولید می‌شود، نه حدس مدل زبانی"
        action={
          narrative ? (
            <Badge tone="success" title={`منبع: ${narrative.source}`}>
              <ShieldCheck className="h-3 w-3" />
              مبتنی بر داده
            </Badge>
          ) : null
        }
      />
      <CardBody>
        {isLoading ? (
          <div className="space-y-2.5">
            <Skeleton className="h-3.5 w-full" />
            <Skeleton className="h-3.5 w-11/12" />
            <Skeleton className="h-3.5 w-9/12" />
          </div>
        ) : (
          <blockquote className="relative border-r-2 border-brand-300 pr-4 text-sm leading-7 text-ink">
            {narrative?.text ?? 'داده کافی برای تولید تحلیل در دسترس نیست.'}
          </blockquote>
        )}
        {narrative ? (
          <p className="debug-only mt-4 text-[11px] text-muted" dir="ltr">
            source: {narrative.source}
          </p>
        ) : null}
      </CardBody>
    </Card>
  );
}
