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
    <div className={cn('-mx-1 overflow-x-auto px-1', className)}>
      <table
        className="w-full border-collapse text-sm"
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
}: {
  children: ReactNode;
  align?: 'right' | 'left' | 'center';
  className?: string;
  title?: string;
}) {
  return (
    <th
      title={title}
      className={cn(
        'whitespace-nowrap border-b border-line px-3 pb-2.5 pt-1 text-xs font-semibold text-muted',
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
        'whitespace-nowrap border-b border-line/70 px-3 py-2.5 text-sm text-ink',
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
