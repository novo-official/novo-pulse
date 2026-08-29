'use client';

import { Info } from 'lucide-react';
import { useEffect, useId, useLayoutEffect, useRef, useState, type CSSProperties, type ReactNode } from 'react';
import { createPortal } from 'react-dom';

type Placement = 'top' | 'bottom' | 'left' | 'right';
type TooltipPosition = { left: number; top: number; placement: Placement; arrowX: number; arrowY: number };

const VIEWPORT_PADDING = 12;
const OFFSET = 10;
const ARROW_SAFE_AREA = 14;
const clamp = (value: number, min: number, max: number) => Math.min(Math.max(value, min), max);

/** A viewport-aware tooltip, portalled to body to escape clipping/stacking parents. */
export function InfoHint({ children, label }: { children: ReactNode; label?: string }) {
  const triggerRef = useRef<HTMLButtonElement>(null);
  const tooltipRef = useRef<HTMLSpanElement>(null);
  const closeTimer = useRef<number | null>(null);
  const tooltipId = useId();
  const [mounted, setMounted] = useState(false);
  const [open, setOpen] = useState(false);
  const [position, setPosition] = useState<TooltipPosition | null>(null);

  const clearCloseTimer = () => {
    if (closeTimer.current !== null) {
      window.clearTimeout(closeTimer.current);
      closeTimer.current = null;
    }
  };
  const show = () => { clearCloseTimer(); setOpen(true); };
  const hide = () => {
    clearCloseTimer();
    closeTimer.current = window.setTimeout(() => setOpen(false), 80);
  };

  useEffect(() => {
    setMounted(true);
    return () => clearCloseTimer();
  }, []);

  useLayoutEffect(() => {
    if (!open || !triggerRef.current || !tooltipRef.current) return;
    const updatePosition = () => {
      const anchor = triggerRef.current?.getBoundingClientRect();
      const tooltip = tooltipRef.current?.getBoundingClientRect();
      if (!anchor || !tooltip) return;

      const viewportWidth = window.innerWidth;
      const viewportHeight = window.innerHeight;
      const spaces: Record<Placement, number> = {
        top: anchor.top - VIEWPORT_PADDING - OFFSET,
        bottom: viewportHeight - anchor.bottom - VIEWPORT_PADDING - OFFSET,
        // Browser coordinates are physical; RTL only changes the side preference.
        right: viewportWidth - anchor.right - VIEWPORT_PADDING - OFFSET,
        left: anchor.left - VIEWPORT_PADDING - OFFSET,
      };
      const required: Record<Placement, number> = { top: tooltip.height, bottom: tooltip.height, right: tooltip.width, left: tooltip.width };
      const preference: Placement[] = document.documentElement.dir === 'rtl'
        ? ['top', 'bottom', 'left', 'right']
        : ['top', 'bottom', 'right', 'left'];
      const placement = preference.find((candidate) => spaces[candidate] >= required[candidate])
        ?? [...preference].sort((a, b) => (spaces[b] - required[b]) - (spaces[a] - required[a]))[0];

      let left = anchor.left + anchor.width / 2 - tooltip.width / 2;
      let top = anchor.top + anchor.height / 2 - tooltip.height / 2;
      if (placement === 'top') top = anchor.top - tooltip.height - OFFSET;
      if (placement === 'bottom') top = anchor.bottom + OFFSET;
      if (placement === 'right') left = anchor.right + OFFSET;
      if (placement === 'left') left = anchor.left - tooltip.width - OFFSET;
      left = clamp(left, VIEWPORT_PADDING, Math.max(VIEWPORT_PADDING, viewportWidth - tooltip.width - VIEWPORT_PADDING));
      top = clamp(top, VIEWPORT_PADDING, Math.max(VIEWPORT_PADDING, viewportHeight - tooltip.height - VIEWPORT_PADDING));

      setPosition({
        left, top, placement,
        arrowX: clamp(anchor.left + anchor.width / 2 - left, ARROW_SAFE_AREA, tooltip.width - ARROW_SAFE_AREA),
        arrowY: clamp(anchor.top + anchor.height / 2 - top, ARROW_SAFE_AREA, tooltip.height - ARROW_SAFE_AREA),
      });
    };
    updatePosition();
    const resizeObserver = new ResizeObserver(updatePosition);
    resizeObserver.observe(triggerRef.current);
    resizeObserver.observe(tooltipRef.current);
    window.addEventListener('resize', updatePosition);
    window.addEventListener('scroll', updatePosition, true);
    return () => {
      resizeObserver.disconnect();
      window.removeEventListener('resize', updatePosition);
      window.removeEventListener('scroll', updatePosition, true);
    };
  }, [open, children]);

  const tooltip = open && mounted ? (
    <span ref={tooltipRef} id={tooltipId} role="tooltip" data-placement={position?.placement ?? 'top'} className="app-tooltip" style={position ? ({ left: position.left, top: position.top, '--tooltip-arrow-x': `${position.arrowX}px`, '--tooltip-arrow-y': `${position.arrowY}px` } as CSSProperties) : { visibility: 'hidden' }}>
      {children}<span className="app-tooltip__arrow" aria-hidden="true" />
    </span>
  ) : null;

  return <>
    <button ref={triggerRef} type="button" className="info-hint" aria-label={label ?? 'توضیح تکمیلی'} aria-describedby={open ? tooltipId : undefined} onPointerEnter={show} onPointerLeave={hide} onFocus={show} onBlur={hide}>
      <Info className="h-3.5 w-3.5" aria-hidden="true" />
    </button>
    {mounted ? createPortal(tooltip, document.body) : null}
  </>;
}
