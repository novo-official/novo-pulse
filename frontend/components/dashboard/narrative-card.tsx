'use client';

import { Bot, ShieldCheck } from 'lucide-react';

import { Badge } from '@/components/ui/badge';
import { Card, CardBody, CardHeader } from '@/components/ui/card';
import { Skeleton } from '@/components/ui/states';
import type { NarrativeResponse } from '@/lib/types/api';
import { toPersianDigits } from '@/lib/utils';

const presentNumbers = (text: string) => toPersianDigits(text).replace(/,/g, '٬').replace(/%/g, '٪');

export function NarrativeCard({
  narrative,
  isLoading,
}: {
  narrative: NarrativeResponse | null;
  isLoading: boolean;
}) {
  return (
    <Card className="insights-card smart-analysis-card h-full">
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
      <CardBody className="insights-card__body">
        {isLoading ? (
          <div className="space-y-2.5">
            <Skeleton className="h-3.5 w-full" />
            <Skeleton className="h-3.5 w-11/12" />
            <Skeleton className="h-3.5 w-9/12" />
          </div>
        ) : (
          <blockquote className="smart-analysis-card__copy relative pr-4 text-[13.5px] leading-7 text-ink">
            {narrative ? presentNumbers(narrative.text) : 'داده کافی برای تولید تحلیل در دسترس نیست.'}
          </blockquote>
        )}
        {narrative ? (
          <p className="smart-analysis-card__source debug-only mt-4" dir="ltr">
            <bdi>source: {narrative.source}</bdi>
          </p>
        ) : null}
      </CardBody>
    </Card>
  );
}
