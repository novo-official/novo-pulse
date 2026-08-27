import { ChevronDown } from 'lucide-react';

import { cn } from '@/lib/utils';
import type { SelectHTMLAttributes } from 'react';

export function Select({
  className,
  children,
  label,
  ...props
}: SelectHTMLAttributes<HTMLSelectElement> & { label?: string }) {
  return (
    <label className="flex flex-col gap-1.5">
      {label ? <span className="text-xs font-medium text-muted">{label}</span> : null}
      <div className="relative">
        <select
          className={cn(
            'h-10 w-full appearance-none rounded-xl border border-line bg-surface px-3 pl-9 text-sm text-ink',
            'transition focus:border-brand-400 focus:outline-none focus:ring-2 focus:ring-brand-100',
            'disabled:cursor-not-allowed disabled:bg-slate-50 disabled:text-slate-400',
            className,
          )}
          {...props}
        >
          {children}
        </select>
        <ChevronDown className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400" />
      </div>
    </label>
  );
}

export function TextInput({
  className,
  label,
  ...props
}: React.InputHTMLAttributes<HTMLInputElement> & { label?: string }) {
  return (
    <label className="flex flex-col gap-1.5">
      {label ? <span className="text-xs font-medium text-muted">{label}</span> : null}
      <input
        className={cn(
          'h-10 w-full rounded-xl border border-line bg-surface px-3 text-sm text-ink',
          'transition placeholder:text-slate-400 focus:border-brand-400 focus:outline-none focus:ring-2 focus:ring-brand-100',
          className,
        )}
        {...props}
      />
    </label>
  );
}
