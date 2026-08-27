'use client';

import { Info } from 'lucide-react';
import type { ReactNode } from 'react';

/** Small info affordance. Technical English terms live here, per the UI spec. */
export function InfoHint({ children, label }: { children: ReactNode; label?: string }) {
  return (
    <span className="group relative inline-flex items-center">
      <Info className="h-3.5 w-3.5 cursor-help text-slate-400" aria-label={label ?? 'توضیح'} />
      <span
        role="tooltip"
        className="pointer-events-none absolute bottom-full right-1/2 z-30 mb-2 w-56 translate-x-1/2 rounded-lg bg-slate-900 px-3 py-2 text-xs leading-5 text-slate-100 opacity-0 shadow-lift transition group-hover:opacity-100"
      >
        {children}
      </span>
    </span>
  );
}
