'use client';

/** Shared dashboard filter state (level, entity, horizon, presentation mode). */
import { create } from 'zustand';

import type { CalendarName } from '@/lib/calendar';
import type { Level } from '@/lib/types/api';

const CALENDAR_KEY = 'novo-pulse:calendar';

/** The stored preference, if the browser has one and will let us read it. */
function storedCalendar(): CalendarName {
  try {
    const saved = window.localStorage.getItem(CALENDAR_KEY);
    if (saved === 'jalali' || saved === 'gregorian') return saved;
  } catch {
    // Private windows and blocked site data throw on access; the default is fine.
  }
  return 'jalali';
}

interface FilterState {
  level: Level;
  entityId: string | null;
  horizon: number;
  presentation: boolean;
  calendar: CalendarName;
  setLevel: (level: Level) => void;
  setEntityId: (id: string | null) => void;
  setHorizon: (horizon: number) => void;
  setPresentation: (on: boolean) => void;
  setCalendar: (calendar: CalendarName) => void;
}

export const useForecastFilters = create<FilterState>((set) => ({
  level: 'destination',
  entityId: null,
  // Long horizons are the hard part of this problem and what the product is
  // built to answer, so the dashboard opens on one. HorizonSelector snaps this
  // down if the published run was trained shorter.
  horizon: 90,
  presentation: false,
  // Iranian tourism plans in Shamsi months, so that is the default. The store
  // starts on the default and the app shell hydrates the saved preference,
  // because reading localStorage during render would break SSR.
  calendar: 'jalali',
  // Changing the level invalidates the selected entity.
  setLevel: (level) => set({ level, entityId: null }),
  setEntityId: (entityId) => set({ entityId }),
  setHorizon: (horizon) => set({ horizon }),
  setPresentation: (presentation) => set({ presentation }),
  setCalendar: (calendar) => {
    set({ calendar });
    try {
      window.localStorage.setItem(CALENDAR_KEY, calendar);
    } catch {
      // Not being able to remember the choice is not a reason to refuse it.
    }
  },
}));

export { storedCalendar };
