import { cn } from '@/lib/utils';

export function TypePill({ children, tone, className }: { children: string; tone: 'ok' | 'warn' | 'bad' | 'neutral'; className?: string }) {
  return <span className={cn('type-pill', `type-pill--${tone}`, className)}><span aria-hidden="true" />{children}</span>;
}
