import { cn } from '@/lib/utils';
import type { ReactNode } from 'react';

export function Card({
  className,
  children,
  ...props
}: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn(
        'rounded-2xl border border-line/90 bg-surface shadow-card ring-1 ring-white/70 transition duration-200',
        className,
      )}
      {...props}
    >
      {children}
    </div>
  );
}

export function CardHeader({
  title,
  subtitle,
  action,
  icon,
  className,
}: {
  title: ReactNode;
  subtitle?: ReactNode;
  action?: ReactNode;
  icon?: ReactNode;
  className?: string;
}) {
  return (
    <div className={cn('flex flex-col items-start justify-between gap-3 px-4 pt-4 sm:flex-row sm:px-5 sm:pt-5', className)}>
      <div className="flex items-start gap-3">
        {icon ? (
          <span className="mt-0.5 flex h-9 w-9 shrink-0 items-center justify-center rounded-xl border border-brand-100 bg-brand-50 text-brand-600">
            {icon}
          </span>
        ) : null}
        <div>
          <h3 className="text-base font-semibold leading-6 text-ink">{title}</h3>
          {subtitle ? <p className="mt-1 text-sm leading-6 text-muted">{subtitle}</p> : null}
        </div>
      </div>
      {action ? <div className="max-w-full shrink-0">{action}</div> : null}
    </div>
  );
}

export function CardBody({
  className,
  children,
}: {
  className?: string;
  children: ReactNode;
}) {
  return <div className={cn('px-4 pb-4 pt-4 sm:px-5 sm:pb-5', className)}>{children}</div>;
}
