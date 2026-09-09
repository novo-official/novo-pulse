import { Activity } from 'lucide-react';
import type { ReactNode } from 'react';

import { cn } from '@/lib/utils';

export function PageHeader({
  title,
  description,
  eyebrow = 'نوو پالس',
  action,
  className,
}: {
  title: string;
  description: ReactNode;
  eyebrow?: string;
  action?: ReactNode;
  className?: string;
}) {
  return (
    <header className={cn('relative flex flex-col gap-3 overflow-hidden rounded-2xl border border-line/70 bg-surface/70 px-4 py-4 shadow-sm backdrop-blur sm:flex-row sm:items-center sm:justify-between sm:px-5', className)}>
      <div className="pointer-events-none absolute -left-12 -top-20 h-40 w-40 rounded-full bg-brand-100/60 blur-3xl" />
      <div className="relative flex max-w-4xl items-start gap-3.5">
        <span className="mt-0.5 hidden h-10 w-10 shrink-0 items-center justify-center rounded-xl border border-brand-100 bg-gradient-to-br from-brand-50 to-cyan-50 text-brand-600 sm:flex">
          <Activity className="h-4.5 w-4.5" />
        </span>
        <div>
          <p className="eyebrow">{eyebrow}</p>
          <h2 className="mt-0.5 text-xl font-bold tracking-tight text-ink sm:text-2xl">{title}</h2>
          <p className="mt-1 text-sm leading-6 text-muted">{description}</p>
        </div>
      </div>
      {action ? <div className="relative shrink-0">{action}</div> : null}
    </header>
  );
}
