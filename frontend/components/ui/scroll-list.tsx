'use client';

import { useRef, useState, type ReactNode, type UIEvent } from 'react';

import { cn } from '@/lib/utils';

export function ScrollList({ label, children, className }: { label: string; children: ReactNode; className?: string }) {
  const frame = useRef<number | null>(null);
  const [state, setState] = useState({ progress: 0, atStart: true, atEnd: true });
  const onScroll = (event: UIEvent<HTMLDivElement>) => {
    const element = event.currentTarget;
    if (frame.current !== null) cancelAnimationFrame(frame.current);
    frame.current = requestAnimationFrame(() => {
      const range = Math.max(element.scrollHeight - element.clientHeight, 0);
      const progress = range ? element.scrollTop / range : 0;
      setState({ progress, atStart: element.scrollTop <= 1, atEnd: element.scrollTop >= range - 1 });
    });
  };

  return (
    <div className={cn('scroll-list-shell', !state.atStart && 'scroll-list-shell--scrolled', !state.atEnd && 'scroll-list-shell--has-more', className)}>
      <span className="scroll-list-progress" style={{ transform: `scaleX(${state.progress})` }} aria-hidden="true" />
      <div className="scroll-list" role="region" aria-label={label} tabIndex={0} onScroll={onScroll}>
        {children}
      </div>
      <span className="scroll-list-fade scroll-list-fade--top" aria-hidden="true" />
      <span className="scroll-list-fade scroll-list-fade--bottom" aria-hidden="true" />
    </div>
  );
}
