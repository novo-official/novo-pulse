'use client';

/** Shared dashboard filter state (level, entity, horizon, presentation mode). */
import { create } from 'zustand';

import type { Level } from '@/lib/types/api';

interface FilterState {
  level: Level;
  entityId: string | null;
  horizon: number;
  presentation: boolean;
  setLevel: (level: Level) => void;
  setEntityId: (id: string | null) => void;
  setHorizon: (horizon: number) => void;
  setPresentation: (on: boolean) => void;
}

export const useForecastFilters = create<FilterState>((set) => ({
  level: 'destination',
  entityId: null,
  // Long horizons are the hard part of this problem and what the product is
  // built to answer, so the dashboard opens on one. HorizonSelector snaps this
  // down if the published run was trained shorter.
  horizon: 90,
  presentation: false,
  // Changing the level invalidates the selected entity.
  setLevel: (level) => set({ level, entityId: null }),
  setEntityId: (entityId) => set({ entityId }),
  setHorizon: (horizon) => set({ horizon }),
  setPresentation: (presentation) => set({ presentation }),
}));
