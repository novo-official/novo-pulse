import { cn } from '@/lib/utils';
import type { ReactNode } from 'react';

/** Tables always scroll inside their own container, never the page body. */
export function TableWrap({
  children,
  className,
  minWidth = 640,
}: {
  children: ReactNode;
  className?: string;
  /** Wide tables need more room before columns start colliding. */
  minWidth?: number;
}) {
  return (
    <div className={cn('-mx-1 overflow-x-auto rounded-xl px-1', className)}>
      <table
        className="w-full border-separate border-spacing-0 text-sm [&_tbody_tr]:transition-colors [&_tbody_tr:hover]:bg-brand-50/35"
        style={{ minWidth: `${minWidth}px` }}
      >
        {children}
      </table>
    </div>
  );
}

export function Th({
  children,
  align = 'right',
  className,
  title,
  scope = 'col',
}: {
  children: ReactNode;
  align?: 'right' | 'left' | 'center';
  className?: string;
  title?: string;
  scope?: 'col' | 'row' | 'colgroup' | 'rowgroup';
}) {
  return (
    <th
      scope={scope}
      title={title}
      className={cn(
        'whitespace-nowrap border-b border-line bg-slate-50/80 px-3 py-3 text-[11px] font-bold text-muted',
        align === 'right' && 'text-right',
        align === 'left' && 'text-left',
        align === 'center' && 'text-center',
        className,
      )}
    >
      {children}
    </th>
  );
}

export function Td({
  children,
  align = 'right',
  className,
}: {
  children: ReactNode;
  align?: 'right' | 'left' | 'center';
  className?: string;
}) {
  return (
    <td
      className={cn(
        'whitespace-nowrap border-b border-line/60 px-3 py-3 text-sm text-ink',
        align === 'right' && 'text-right',
        align === 'left' && 'text-left',
        align === 'center' && 'text-center',
        className,
      )}
    >
      {children}
    </td>
  );
}
