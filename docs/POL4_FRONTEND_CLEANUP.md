# POL 4 — Frontend Forensic Audit & Cleanup Plan

**Audit date:** 2026-09-09 · **Commit:** `3b39c12`
**Frontend size:** 6,449 LOC across 57 `.ts` / `.tsx` files (excl. `node_modules`, `next-env.d.ts`).
**Nothing was deleted.** This document lists exactly what to delete, and in what order.

---

## 0. Headline finding — stated honestly

The brief anticipated an ERP-like SaaS panel full of CRUD, settings and admin screens.
**That is not what this frontend is.** There is:

- no authentication, no login, no profile, no account page
- no settings, no notifications centre, no roles, no users, no teams
- no billing, no plans, no projects, no tasks
- **no CRUD screens of any kind**
- **no mock data.** A repo-wide `grep` for `Math.random`, `mockData`, `MOCK`, `dummy`, `faker`
  across `frontend/app`, `frontend/components`, `frontend/lib`, `frontend/hooks` returns
  exactly **one** hit: the string `placeholder=` on a search input (`data-lab/page.tsx:451`).
- every data surface routes through `AsyncBoundary` with real loading / empty / error states
  (`components/ui/states.tsx`)
- dependencies are lean: `recharts`, `@tanstack/react-query`, `zustand`, `clsx`,
  `tailwind-merge`, `lucide-react`. No animation library, no chart bloat, no UI kit.

So the cleanup is **not** "delete the admin panel". It is:

1. delete the two surfaces that are **dead on the Pol 4 dataset** (`/scenarios`, `/data-lab`),
2. collapse 6 nav routes into 4,
3. remove the cards that render permanently-empty state on this dataset,
4. and **build the two charts that actually win this competition** — pickup curve and
   forecast stability — which do not exist at all.

---

## 1. Current routes

| # | Route | File | LOC | In nav? |
|---|---|---|---:|---|
| 1 | `/` | `app/page.tsx` | 5 | no (redirects to `/dashboard`) |
| 2 | `/dashboard` | `app/dashboard/page.tsx` | 239 | ✅ داشبورد |
| 3 | `/forecasts` | `app/forecasts/page.tsx` | 158 | ✅ پیش‌بینی‌ها |
| 4 | `/scenarios` | `app/scenarios/page.tsx` | 101 | ✅ شبیه‌سازی سناریو |
| 5 | `/backtesting` | `app/backtesting/page.tsx` | 71 | ✅ اعتبارسنجی |
| 6 | `/models` | `app/models/page.tsx` | 321 | ✅ مدل‌ها |
| 7 | `/data-lab` | `app/data-lab/page.tsx` | **1,024** | ✅ آزمایشگاه داده |

Plus framework routes: `error.tsx`, `loading.tsx`, `not-found.tsx`, `layout.tsx`, `providers.tsx`.

## 2. Existing navigation

`components/layout/app-shell.tsx:26-33` — a collapsible right-hand RTL sidebar, 6 items, each
carrying a Persian "story" subtitle:

```
داشبورد            → چه اتفاقی می‌افتد؟
پیش‌بینی‌ها          → کجا اتفاق می‌افتد؟
شبیه‌سازی سناریو    → اگر شرایط تغییر کند؟
اعتبارسنجی         → مدل چقدر دقیق بوده؟
مدل‌ها              → کدام مدل برنده است؟
آزمایشگاه داده      → دیتاست جدید
```

The storytelling instinct is good. The story is just the wrong one for Pol 4 — it is the
generic-platform story, not the two-clocks story.

Also present: a `?presentation=true` judging mode, a theme toggle, a Jalali/Gregorian calendar
toggle, and a live `StatusStrip` (`components/layout/status-strip.tsx`).

---

## 3. Route classification matrix

| Route / Component | Current purpose | Real data? | Required for Pol 4? | **Action** | Reason | Replacement |
|---|---|---|---|---|---|---|
| `app/page.tsx` | redirect to `/dashboard` | n/a | yes | **KEEP** | 5 lines, retarget to `/overview` | → `/overview` |
| `app/dashboard/page.tsx` | KPIs + trend + narrative + drivers + opportunities + peaks + anomalies + table + heatmap | ✅ real | yes | **REBUILD** | Right idea, wrong content. Nine sections is too many; it must lead with national Azar demand and the pickup story. | Screen 1 — Demand Overview |
| `app/forecasts/page.tsx` | per-entity forecast + overview table + hierarchy + peaks + heatmap | ✅ real | partly | **MERGE** | ~70% duplicates `/dashboard` (same `timeseries`, `overview`, `peaks`, `heatmap` queries). | Screen 2 — City Analysis |
| `app/scenarios/page.tsx` | what-if on future covariates | ✅ real code, **no data** | **no** | **REMOVE** | `ScenarioEngine.adjustable` iterates `tensor.future`; Pol 4 has zero future covariates → the page renders controls over an empty list. Verified in `backend/apps/scenarios/services.py:80-104`. | none — move to Future Vision slide |
| `app/backtesting/page.tsx` | WAPE by fold / segment / horizon, actual-vs-predicted | ✅ real | yes | **MERGE** | Excellent content, wrong prominence. Judges need it, but as evidence inside Reports, not as its own nav item. | Screen 4 — Reports → *Model Performance* |
| `app/models/page.tsx` | leaderboard, registry, champion banner, censoring, tuning | ✅ real | partly | **MERGE** | Leaderboard + baseline comparison belong in Reports. Registry/tuning/censoring do not belong in the demo at all. | Screen 4 — Reports → *Model Performance* |
| `app/data-lab/page.tsx` | upload → profile → map schema → validate → train | ✅ real | **no** | **REMOVE from nav** | 1,024 LOC — the biggest file in the frontend — built to ingest an *unknown* dataset. Pol 4's dataset is known, fixed, and three files. Keep the route reachable by URL for the team; take it out of the judge-facing product. | none |
| **NEW** `/stability` | forecast revision D-30 → D-1 | — | **yes** | **BUILD** | The mentor's explicitly named differentiator. Does not exist. | Screen 3 — Forecast Stability |
| **NEW** `/reports` | filtered CSV export + model performance | — | **yes** | **BUILD** | No export of any kind exists today. | Screen 4 — Reports |

**Counts**

| | Routes |
|---|---:|
| Current (incl. redirect) | **7** (6 in nav) |
| KEEP as-is | 1 (`/` redirect) |
| REBUILD | 1 (`/dashboard` → `/overview`) |
| MERGE | 3 (`/forecasts`, `/backtesting`, `/models`) |
| REMOVE | 2 (`/scenarios`, `/data-lab` — the latter de-navigated, not deleted) |
| BUILD NEW | 2 (`/stability`, `/reports`) |
| **Recommended final** | **5** (4 in nav) |

---

## 4. Component classification matrix

| Component | Purpose | Data source | Real? | **Action** | Reason |
|---|---|---|---|---|---|
| `charts/forecast-chart.tsx` | history + backtest + forecast + P10–P90 band | `/forecasts/timeseries/` | ✅ | **KEEP** | Core chart. Add an *observed-so-far* series. |
| `charts/heatmap.tsx` | entity × date matrix | `/forecasts/heatmap/` | ✅ | **KEEP** | This is Chart 2 of Screen 1 already. |
| `charts/scenario-chart.tsx` | baseline vs scenario | `/scenarios/simulate/` | ✅ code, no data | **REMOVE** | Dies with `/scenarios`. |
| `charts/palette.ts`, `glass-tooltip.tsx`, `legend-chips.tsx`, `tooltip.tsx` | chart primitives | — | ✅ | **KEEP** | Shared, small, reused 2–7× each. |
| `dashboard/kpi-cards.tsx` (201 LOC) | 6 KPI tiles | `/dashboard/summary/` | ✅ | **REBUILD** | Tiles are right; the six metrics must become the Pol 4 six (see §8). |
| `dashboard/overview-table.tsx` | per-entity ranking table | `/forecasts/overview/` | ✅ | **KEEP** | Becomes Top Destinations. |
| `dashboard/peaks-card.tsx` | upcoming peaks/troughs | `/forecasts/peaks/` | ✅ | **KEEP** | Real phase-aware peak detection. |
| `dashboard/drivers-card.tsx` | SHAP driver groups | `/forecasts/drivers/` | ✅ | **KEEP** | Verified producing real SHAP on the real dataset. |
| `dashboard/anomaly-panel.tsx` | residual + forecast anomalies | `/anomalies/` | ✅ | **REBUILD** | Retarget from "anomaly" to **Emerging Demand**: pickup materially above the historical curve. Same slot, honest label. |
| `dashboard/opportunities-card.tsx` | evidence-bearing insight cards | `/dashboard/summary/` → `insights.decision_opportunities` | ✅ | **REBUILD** | Keep the discipline (every card carries its evidence). Retarget the card kinds to pickup/peak language. |
| `dashboard/narrative-card.tsx` | grounded prose summary | `/forecasts/narrative/` | ✅ | **KEEP (demote)** | Genuinely grounded and anti-hallucination-tested. One card, not a section. |
| `dashboard/censoring-card.tsx` (128 LOC) | supply-censoring report | `/models/` | ✅ code, **no data** | **REMOVE** | Needs a capacity column. Verified in my real-data run: `{"enabled": false, "reason": "censoring is disabled in the contract"}` → permanently dead card. |
| `dashboard/data-quality.tsx` (124 LOC) | validation findings | `/insights/data-quality/` | ✅ | **MERGE** | Used only by `/data-lab`. Fold one health line into Reports; drop the rest. |
| `dashboard/forecast-hierarchy-card.tsx` | listing/destination/category/market levels | `/forecasts/timeseries/` | ✅ | **REMOVE** | Pol 4 has only city + province (7). Two of four levels are always empty. |
| `dashboard/trend-card.tsx` | chart wrapper w/ title | — | ✅ | **KEEP** | 18 LOC layout primitive. |
| `filters/forecast-filters.tsx` | level / entity / horizon selector | — | ✅ | **REBUILD** | Levels collapse to city + national. Horizon options `[7,14,30,60,90]` must become the Azar window. |
| `layout/app-shell.tsx` (257 LOC) | sidebar, presentation mode, calendar | — | ✅ | **REBUILD** | 6 items → 4. Prefer a compact top nav; the collapsible sidebar costs 100+ LOC for six links. |
| `layout/status-strip.tsx` | demo-mode / champion / connectivity badge | `/health/` | ✅ | **KEEP** | It is the honesty badge. Make the synthetic-data warning louder (§6). |
| `ui/*` (badge, button, card, page-header, scroll-list, select, states, table, tooltip, type-pill) | primitives | — | ✅ | **KEEP** | All used ≥2×. `states.tsx` is the reason there is no fake data anywhere. |
| `theme-provider.tsx`, `theme-toggle.tsx` | dark mode | — | ✅ | **KEEP** | Cheap, and the dark theme is the newest commit. |
| `hooks/useCalendar.ts`, `useForecastFilters.ts` | Jalali toggle, filter store | — | ✅ | **KEEP** | Jalali display matters for Azar 1404. ⚠️ backend `jdatetime` is missing — see audit §8. |
| `e2e/ui_drive.mjs`, `ui_upload.mjs`, `ui_competition.mjs` (475 LOC) | browser walkthroughs | — | ✅ | **REMOVE `ui_upload` + `ui_competition`** | Both drive `/data-lab` upload flows that are going away. Keep `ui_drive.mjs`, retarget it. |

---

## 5. Components using fake or static data

**None.** This is a real and reportable strength.

The only place a judge could see a number that is not from the Pol 4 dataset is the
`DEMO_MODE` path, which serves `data/demo_artifacts/` — a **synthetic** run
(140 fabricated accommodations, `booking_count` target, fabricated prices, Persian city labels
تهران/مشهد/کیش/رامسر). It is real output of a real pipeline; it is just output for a fictional
dataset. `DEMO_MODE=true` is the default (`backend/config/settings.py:150`).

## 6. Components disconnected from Pol 4 forecasting

| Component | Why it is disconnected |
|---|---|
| `/scenarios` + `scenario-chart` | requires future covariates; Pol 4 has none |
| `censoring-card` | requires a capacity column; Pol 4 has none |
| `forecast-hierarchy-card` | requires listing/category levels; Pol 4 has city + province only |
| `/data-lab` + `data-quality` | requires an unknown dataset; Pol 4's is known and fixed |
| `models` registry / tuning cards | platform introspection, not demand intelligence |

## 7. Deletion plan — exact paths

**Tier 1 — delete outright (dead on this dataset)**

```
frontend/app/scenarios/page.tsx
frontend/components/charts/scenario-chart.tsx
frontend/components/dashboard/censoring-card.tsx
frontend/components/dashboard/forecast-hierarchy-card.tsx
frontend/e2e/ui_upload.mjs
frontend/e2e/ui_competition.mjs
frontend/e2e/browser_upload.csv
```
Backend counterparts (remove the nav/API surface, keep the code out of the demo):
```
backend/apps/scenarios/            (views, services, urls entries)
backend/config/api_urls.py:44-45   scenarios/options/, scenarios/simulate/
```

**Tier 2 — remove from navigation, keep reachable by URL**

```
frontend/app/data-lab/page.tsx            (1,024 LOC — drop the nav entry only)
frontend/components/dashboard/data-quality.tsx
frontend/components/layout/app-shell.tsx:32   ← the آزمایشگاه داده nav item
```

**Tier 3 — merge, then delete the shell**

```
frontend/app/forecasts/page.tsx      → merged into /city
frontend/app/backtesting/page.tsx    → merged into /reports
frontend/app/models/page.tsx         → leaderboard section merged into /reports
```

**Do not delete** (frequently mistaken for dead code): every `components/ui/*`,
`components/charts/palette.ts`, `glass-tooltip.tsx`, `legend-chips.tsx`, `trend-card.tsx`,
`type-pill.tsx`, `scroll-list.tsx` — all have ≥2 call sites, verified.

**Net effect:** ~2,000 LOC removed, ~1,000 LOC of new analytics added, 6 nav routes → 4.

---

## 8. Final desired route structure

```
/                     → redirect → /overview
/overview             Screen 1 — Demand Overview        (default, demo homepage)
/city                 Screen 2 — City Analysis
/stability            Screen 3 — Forecast Stability
/reports              Screen 4 — Reports & Model Performance
/data-lab             (unlisted, team-only)
```

Navigation: a compact **top** bar with four items. No sidebar.

```
نمای کلی   ·   تحلیل شهر   ·   پایداری پیش‌بینی   ·   گزارش‌ها
```

## 9. Information architecture

### Screen 1 — Demand Overview (`/overview`)

KPI row — **six tiles, all from the real pipeline, none hard-coded:**

| Tile | Source |
|---|---|
| Total predicted demand, Azar 1404 | `Σ predicted_demand` over the 9,630-row grid |
| Peak check-in date | `argmax` of the national daily total |
| Highest-demand city | `argmax` of the per-city 30-day total |
| Fastest-pickup city | max `pickup_ratio` = recent pickup ÷ historically expected pickup at the same horizon |
| Mean forecast stability | mean absolute revision D-30 → D-1 across the backtest |
| Backtested WAPE | pooled walk-forward WAPE |

Charts (in this order, charts dominate the viewport):

1. **National demand forecast** — X: 30 Azar dates · Y: total predicted demand. Overlays:
   observed-so-far (from `evaluation.csv`) and the historical Azar-1403 baseline.
2. **City × date heatmap** — rows: top ~25 cities · columns: 30 dates · intensity: predicted demand.
3. **Top destinations** — horizontal ranked bars, predicted total + pickup-growth badge.
4. **Emerging demand** — (city, date) pairs where `pickup_ratio` is materially above the
   historical curve. Computed, never asserted.

### Screen 2 — City Analysis (`/city`)

City selector + horizon filter. No forms.

- **A. 30-day forecast** — predicted final vs observed-so-far per check-in date.
- **B. Pickup curve** — X: days before check-in (59 → 0) · Y: cumulative searches.
  Two lines: this city/date's actual accumulation vs the historically expected curve.
- **C. Forecast vs observed** — stacked bar: `observed_so_far` + `predicted_remaining` = `predicted_final`.
- **D. Historical pattern** — the same check-in date in prior years, compact.
- **E. Peak dates** — ranked upcoming high-demand dates for this city.

### Screen 3 — Forecast Stability (`/stability`)

- X: forecast issue point (D-30, D-21, D-14, D-7, D-3, D-1) · Y: predicted final demand.
- One line per selected (city, check-in); a horizontal rule for the realised final demand
  on backtested examples.
- Panels: forecast revision (absolute + relative), convergence, and a stability score.

### Screen 4 — Reports (`/reports`)

- **Model performance:** WAPE by model (incl. every baseline), WAPE by horizon, WAPE by
  demand bucket, WAPE by province, forecast bias, baseline-vs-champion. All from
  `metrics.json` — never hard-coded.
- **Exports (CSV first, PDF only if free):** City Demand · Top Destinations · Peak Demand ·
  Pickup Analysis · Forecast Stability · Model Performance · **`results.csv` itself**.
- Filters: city, province, check-in date range, horizon, demand bucket.

---

## 10. Chart inventory & API requirements

| # | Chart | Screen | Endpoint | Exists today? |
|---|---|---|---|---|
| 1 | National demand forecast | Overview | `GET /forecasts/timeseries/?level=market` | ✅ reusable |
| 2 | City × date heatmap | Overview | `GET /forecasts/heatmap/` | ✅ reusable |
| 3 | Top destinations | Overview | `GET /forecasts/overview/` | ✅ reusable |
| 4 | Emerging demand | Overview | `GET /forecast/trends/` | ❌ **new** |
| 5 | City 30-day forecast | City | `GET /forecasts/timeseries/?level=listing&id=<city>` | ✅ reusable |
| 6 | Pickup curve | City | `GET /forecast/cities/{city}/pickup/?checkin=` | ❌ **new** |
| 7 | Forecast vs observed | City | extend `timeseries` with `observed_so_far` | 🟡 extend |
| 8 | Historical pattern | City | `GET /forecasts/timeseries/` (history slice) | ✅ reusable |
| 9 | Peak dates | City / Overview | `GET /forecasts/peaks/` | ✅ reusable |
| 10 | Stability fan | Stability | `GET /forecast/stability/` | ❌ **new** |
| 11 | Forecast revision | Stability | `GET /forecast/stability/` | ❌ **new** |
| 12 | WAPE by model / horizon / bucket | Reports | `GET /backtests/metrics/` | ✅ reusable (add demand-bucket + peak-date segments) |
| 13 | Forecast drivers (SHAP) | Overview / Reports | `GET /forecasts/drivers/` | ✅ reusable |
| 14 | KPI row | Overview | `GET /dashboard/summary/` | 🟡 extend |
| 15 | CSV export | Reports | `GET /reports/{kind}.csv` | ❌ **new** |

**Three new endpoints, two extensions.** Everything else is already built, already
aggregated server-side, and already returns real data. The API is read/analyse/report shaped,
not CRUD-first — that part of the architecture is right and should be kept.

## 11. Reporting requirements

- CSV export is mandatory and cheap. PDF only if it costs under an hour.
- Do **not** build a report builder. Fixed report kinds + filters, one `GET` each.
- Every export must be generated from the same artefacts the charts read, so a judge can
  reconcile a downloaded CSV against a chart on screen.

## 12. Data-integrity trace (must hold after cleanup)

| UI metric | Endpoint | Backend | Source |
|---|---|---|---|
| Total predicted demand | `/dashboard/summary/` | `apps/forecasting/services.py` | `runs/<id>/forecast.parquet` |
| Peak date | `/forecasts/peaks/` | `ml/anomaly/detector.py` | `runs/<id>/peaks.json` |
| Backtested WAPE | `/backtests/metrics/` | `apps/forecasting/services.py` | `runs/<id>/metrics.json` |
| Drivers | `/forecasts/drivers/` | `ml/explainability/shap_explainer.py` | `runs/<id>/feature_importance.json` |
| Observed-so-far | *new* | *new* | `evaluation.csv` |
| Pickup ratio | *new* | *new* | pickup curves from `search_data.csv` |
| Stability | *new* | *new* | backtest snapshots at D-30…D-1 |

**Rule to keep:** if a number cannot name its artefact, it does not ship.

## 13. Cleanup implementation order

1. Delete Tier 1. Confirm `npm run typecheck` + `next build` still pass.
2. Drop the `/data-lab` nav item (Tier 2). Keep the route.
3. Replace the sidebar with a 4-item top nav; rename `/dashboard` → `/overview`.
4. Merge `/forecasts` into `/city`; merge `/backtesting` + `/models` into `/reports`.
5. **Stop.** Do not touch the frontend again until the pickup formulation lands in the backend.
6. Then, in order: overview chart → heatmap → rankings → city explorer → pickup curve →
   stability → model performance → CSV export.

> UI polish must never precede forecast correctness. The frontend is currently the healthiest
> part of this repository; the forecasting core is the part that will lose the competition.
