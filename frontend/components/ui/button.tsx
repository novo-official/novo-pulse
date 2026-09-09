import { cn } from '@/lib/utils';
import type { ButtonHTMLAttributes } from 'react';

const VARIANTS = {
  primary: 'bg-brand-600 text-white shadow-sm shadow-brand-900/15 hover:-translate-y-px hover:bg-brand-700 hover:shadow-md disabled:bg-brand-300 disabled:shadow-none',
  secondary: 'border border-line bg-surface text-ink shadow-sm hover:border-brand-200 hover:bg-brand-50/40 disabled:text-slate-400',
  outline: 'border border-brand-200 bg-transparent text-brand-700 hover:bg-brand-50',
  ghost: 'text-muted hover:bg-black/5 hover:text-ink',
  danger: 'bg-rose-600 text-white hover:bg-rose-700',
  link: 'h-auto rounded-none px-0 text-brand-700 underline-offset-4 hover:underline',
} as const;

const SIZES = {
  sm: 'h-9 px-3 text-xs',
  md: 'h-10 px-4 text-sm',
  lg: 'h-11 px-5 text-sm',
  icon: 'h-10 w-10 p-0',
} as const;

export function Button({
  variant = 'primary',
  size = 'md',
  className,
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: keyof typeof VARIANTS;
  size?: keyof typeof SIZES;
}) {
  return (
    <button
      className={cn(
        'inline-flex cursor-pointer items-center justify-center gap-2 rounded-xl font-medium transition duration-200',
        'focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-brand-500',
        'disabled:cursor-not-allowed disabled:opacity-70',
        VARIANTS[variant],
        SIZES[size],
        className,
      )}
      {...props}
    />
  );
}
