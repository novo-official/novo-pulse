'use client';

import { Check, ChevronDown } from 'lucide-react';
import {
  Children,
  isValidElement,
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
  type ChangeEvent,
  type InputHTMLAttributes,
  type ReactNode,
} from 'react';

import { cn } from '@/lib/utils';

type SelectOption = { label: string; value: string; disabled: boolean };

function readOptions(children: ReactNode): SelectOption[] {
  return Children.toArray(children).flatMap((child) => {
    if (!isValidElement<{ value?: string; children?: ReactNode; disabled?: boolean }>(child)) return [];
    // React can wrap intrinsic option elements during development. Checking the
    // props instead of the element type keeps the menu populated in both modes.
    if (child.props.value === undefined) return readOptions(child.props.children);
    return [{
      value: child.props.value ?? '',
      label: typeof child.props.children === 'string' ? child.props.children : String(child.props.value ?? ''),
      disabled: Boolean(child.props.disabled),
    }];
  });
}

export function Select({
  className,
  containerClassName,
  children,
  label,
  icon,
  value = '',
  onChange,
  disabled = false,
  name,
  id,
}: {
  className?: string;
  containerClassName?: string;
  children: ReactNode;
  label?: string;
  icon?: ReactNode;
  value?: string | number | readonly string[];
  onChange?: (event: ChangeEvent<HTMLSelectElement>) => void;
  disabled?: boolean;
  name?: string;
  id?: string;
}) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);
  const generatedId = useId();
  const selectId = id ?? generatedId;
  const options = useMemo(() => readOptions(children), [children]);
  const selectedValue = String(value);
  const selected = options.find((option) => option.value === selectedValue) ?? options[0];

  useEffect(() => {
    const closeOnOutsideClick = (event: MouseEvent) => {
      if (!rootRef.current?.contains(event.target as Node)) setOpen(false);
    };
    document.addEventListener('mousedown', closeOnOutsideClick);
    return () => document.removeEventListener('mousedown', closeOnOutsideClick);
  }, []);

  const choose = (nextValue: string) => {
    setOpen(false);
    if (nextValue === selectedValue) return;
    onChange?.({ target: { value: nextValue } } as ChangeEvent<HTMLSelectElement>);
  };

  return (
    <label className={cn('flex min-w-0 flex-col gap-1.5', containerClassName)}>
      {label ? (
        <span className="flex items-center gap-1.5 px-0.5 text-xs font-semibold text-muted">
          {icon ? <span className="text-brand-500/80">{icon}</span> : null}
          {label}
        </span>
      ) : null}
      <div ref={rootRef} className={cn('relative', open && 'z-[9998]')}>
        {name ? <input type="hidden" name={name} value={selectedValue} /> : null}
        <button
          id={selectId}
          type="button"
          disabled={disabled}
          aria-haspopup="listbox"
          aria-expanded={open}
          onClick={() => setOpen((current) => !current)}
          onKeyDown={(event) => {
            if (event.key === 'Escape') setOpen(false);
            if (event.key === 'ArrowDown') {
              event.preventDefault();
              setOpen(true);
            }
          }}
          className={cn(
            'flex h-11 w-full min-w-0 items-center justify-between gap-3 rounded-xl border border-line bg-gradient-to-l from-surface to-slate-50/80 px-3 text-right text-sm font-medium text-ink shadow-sm',
            'transition duration-200 hover:border-brand-200 hover:shadow-card focus:border-brand-400 focus:outline-none focus:ring-4 focus:ring-brand-100/70',
            'disabled:cursor-not-allowed disabled:bg-slate-50 disabled:text-slate-400 disabled:hover:border-line disabled:hover:shadow-sm',
            open && 'border-brand-400 bg-surface ring-4 ring-brand-100/70',
            className,
          )}
        >
          <span className="min-w-0 flex-1 truncate text-ink">{selected?.label ?? 'انتخاب کنید'}</span>
          <ChevronDown className={cn('h-4 w-4 shrink-0 text-brand-500 transition-transform duration-200', open && 'rotate-180')} />
        </button>
        {open && !disabled ? (
          <div className="absolute z-50 mt-2 max-h-64 w-full overflow-y-auto rounded-xl border border-brand-100 bg-surface p-1.5 shadow-lift" role="listbox" aria-labelledby={selectId}>
            {options.map((option) => (
              <button
                key={option.value}
                type="button"
                role="option"
                aria-selected={option.value === selectedValue}
                disabled={option.disabled}
                onClick={() => choose(option.value)}
                className={cn(
                  'flex w-full items-center justify-between gap-3 rounded-lg px-3 py-2.5 text-right text-sm font-medium text-ink transition hover:bg-brand-50',
                  option.value === selectedValue && 'bg-brand-50 text-brand-800',
                  option.disabled && 'cursor-not-allowed opacity-50',
                )}
              >
                <span className="min-w-0 truncate">{option.label}</span>
                {option.value === selectedValue ? <Check className="h-4 w-4 shrink-0 text-brand-600" /> : null}
              </button>
            ))}
          </div>
        ) : null}
      </div>
    </label>
  );
}

export function TextInput({
  className,
  label,
  ...props
}: InputHTMLAttributes<HTMLInputElement> & { label?: string }) {
  return (
    <label className="flex flex-col gap-1.5">
      {label ? <span className="text-xs font-medium text-muted">{label}</span> : null}
      <input
        className={cn(
          'h-10 w-full rounded-xl border border-line bg-surface px-3 text-sm text-ink shadow-sm',
          'transition placeholder:text-slate-400 focus:border-brand-400 focus:outline-none focus:ring-4 focus:ring-brand-100/70',
          className,
        )}
        {...props}
      />
    </label>
  );
}
