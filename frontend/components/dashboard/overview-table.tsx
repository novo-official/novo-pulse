'use client';

import { Map } from 'lucide-react';

import { Badge, toneForChange, toneForConfidence } from '@/components/ui/badge';
import { Card, CardBody, CardHeader } from '@/components/ui/card';
import { EmptyState } from '@/components/ui/states';
import { Td, TableWrap, Th } from '@/components/ui/table';
import type { OverviewRow } from '@/lib/types/api';
import { CONFIDENCE_FA, formatNumber, formatPercent } from '@/lib/utils';

const STATUS_TONE: Record<string, 'success' | 'danger' | 'warning' | 'neutral'> = {
  growing: 'success',
  declining: 'danger',
  spike_risk: 'warning',
  stable: 'neutral',
  unknown: 'neutral',
};

export function OverviewTable({
  rows,
  onSelect,
  title = 'نمای کلی مقاصد',
}: {
  rows: OverviewRow[];
  onSelect?: (id: string) => void;
  title?: string;
}) {
  return (
    <Card>
      <CardHeader
        icon={<Map className="h-4.5 w-4.5" />}
        title={title}
        subtitle="تقاضای فعلی، پیش‌بینی، تغییر و وضعیت هر مقصد"
      />
      <CardBody>
        {rows.length === 0 ? (
          <EmptyState title="مقصدی برای نمایش نیست" />
        ) : (
          <TableWrap>
            <thead>
              <tr>
                <Th align="right">مقصد</Th>
                <Th>تقاضای فعلی</Th>
                <Th>پیش‌بینی</Th>
                <Th>تغییر</Th>
                <Th>بازه اطمینان</Th>
                <Th>اطمینان</Th>
                <Th align="center">وضعیت</Th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr
                  key={row.entity_id}
                  onClick={() => onSelect?.(row.entity_id)}
                  className={onSelect ? 'cursor-pointer transition hover:bg-slate-50' : undefined}
                >
                  <Td align="right" className="font-medium">
                    {row.label}
                  </Td>
                  <Td className="nums text-muted">{formatNumber(row.current_demand)}</Td>
                  <Td className="nums font-semibold">{formatNumber(row.forecast_total)}</Td>
                  <Td>
                    <Badge tone={toneForChange(row.change_pct)}>{formatPercent(row.change_pct)}</Badge>
                  </Td>
                  <Td className="nums text-xs text-muted">
                    {formatNumber(row.lower_total)} – {formatNumber(row.upper_total)}
                  </Td>
                  <Td>
                    <Badge tone={toneForConfidence(row.confidence)}>
                      {CONFIDENCE_FA[row.confidence] ?? row.confidence}
                    </Badge>
                  </Td>
                  <Td align="center">
                    <Badge tone={STATUS_TONE[row.status] ?? 'neutral'}>{row.status_fa}</Badge>
                  </Td>
                </tr>
              ))}
            </tbody>
          </TableWrap>
        )}
      </CardBody>
    </Card>
  );
}
