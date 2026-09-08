import { LineChart } from 'lucide-react';
import type { ReactNode } from 'react';

import { Card, CardBody, CardHeader } from '@/components/ui/card';

export function TrendCard({ title, children }: { title: string; children: ReactNode }) {
  return (
    <Card className="trend-card overflow-hidden">
      <CardHeader
        className="trend-card__header"
        icon={<LineChart className="h-5 w-5" />}
        title={title}
        subtitle="تاریخچه واقعی، پیش‌بینی گذشته‌نگر و پیش‌بینی آینده به همراه بازه اطمینان ۸۰٪"
      />
      <CardBody className="trend-card__body">{children}</CardBody>
    </Card>
  );
}
