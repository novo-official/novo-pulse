# POL 4 — Product & Scientific Credibility Review

**Audit date:** 2026-09-09 · **Commit:** `3b39c12`

---

## 1. Claim inventory

Every user-facing capability in the repository, classified against what the Pol 4 data can
actually support.

Legend: **REAL** = computed from competition data · **HEURISTIC** = rule-based but honest ·
**MOCKED** = fabricated · **UNSUPPORTED** = requires data that does not exist.

| # | Capability | Where | Classification | Verdict |
|---|---|---|---|---|
| 1 | Demand forecast by city × date | `/dashboard`, `/forecasts` | **REAL** (wrong formulation) | Keep, rebuild onto pickup |
| 2 | Prediction intervals (P10–P90) with **measured** coverage | `/dashboard`, `/backtesting` | **REAL** | Keep — coverage is measured out-of-sample, not assumed |
| 3 | SHAP driver attribution | `/dashboard` | **REAL** — verified producing real values on the real dataset | Keep |
| 4 | Rolling-origin backtest vs baselines | `/backtesting`, `/models` | **REAL** | Keep, move into Reports |
| 5 | Peak / trough detection (phase-aware) | `/forecasts/peaks/` | **REAL** | Keep |
| 6 | Residual & forecast anomalies (robust MAD) | `/anomalies/` | **REAL** | Keep, relabel toward pickup |
| 7 | Business insight cards | `/insights/opportunities/` | **HEURISTIC**, evidence-bearing | Keep the discipline, retarget the language |
| 8 | Natural-language narrative | `/forecasts/narrative/` | **HEURISTIC**, template or local LLM, anti-hallucination gated | Keep, demote to one card |
| 9 | Hierarchy: listing / destination / category / market | `/forecasts` | **UNSUPPORTED** — Pol 4 has city + 7 provinces only | Reduce to city + province + national |
| 10 | What-if scenario simulation | `/scenarios` | **UNSUPPORTED** — needs future covariates; Pol 4 has none | **Remove.** Move to Future Vision |
| 11 | Demand censoring / unconstrained demand | `/models` → `CensoringCard` | **UNSUPPORTED** — needs a capacity column | **Remove.** Verified self-disabled: `{"enabled": false, "reason": "censoring is disabled in the contract"}` |
| 12 | Price SHAP dependence curve | `price_dependence.json` | **UNSUPPORTED** — no price column | Remove from the demo |
| 13 | Data Lab: upload / profile / map / validate | `/data-lab` | **REAL** but irrelevant | De-navigate; the dataset is fixed and known |
| 14 | Pickup curve | — | **MISSING** | **Build.** It is the product. |
| 15 | Forecast stability D-30 → D-1 | — | **MISSING** | **Build.** The named differentiator. |
| 16 | Observed vs remaining decomposition | — | **MISSING** | **Build** |
| 17 | CSV / report export | — | **MISSING** | **Build** |

**No MOCKED capability was found anywhere in this repository.** I looked specifically for it.
That is worth stating plainly, because it is unusual.

---

## 2. Forbidden-claims sweep

The brief lists capabilities the Pol 4 data cannot support. I searched the entire repository
for each. **Result: none of them appear.**

| Forbidden claim | Present? | Evidence |
|---|---|---|
| User-level behavioural segmentation | ❌ absent | no user column exists anywhere in the schema |
| MBTI-like traveller segmentation | ❌ absent | no such string in the codebase |
| Luxury / economy traveller classification | ❌ absent | — |
| Religious / leisure traveller classification | ❌ absent | — |
| Property-level forecasting | 🟡 the *platform* supports a `listing` level | Pol 4 has no property data → the level is simply unavailable and self-disables |
| Price optimisation / host price recommendation | ❌ absent | scenarios expose price only as an input **when a price column exists**; they never recommend one |
| Capacity-shortage prediction | 🟡 censoring module exists | needs a capacity column; self-disables on Pol 4 — verified |
| Booking-conversion prediction | ❌ absent | — |
| User-level next-best-action | ❌ absent | — |
| Individual destination recommendation | ❌ absent | — |
| "Acquire exactly N properties" style prescriptions | ❌ absent | `build_decision_opportunities` (`ml/insights/engine.py:172-250`) emits only evidence-bearing statements |
| "Increase prices by N%" style prescriptions | ❌ absent | — |

**This is the repository's single greatest strength and it must be protected.** The temptation
during the final 72 hours will be to add impressive-sounding "AI insights". Do not.

---

## 3. Credibility defects to fix

| # | Defect | Where | Severity | Fix |
|---|---|---|---|---|
| C1 | **The proxy-vs-intent distinction is entirely absent.** Nothing says "search is a proxy for latent intent, not demand." | everywhere | **HIGH** | Add it to slide 2, report §1, and one line under the Overview KPI row |
| C2 | The framing that *is* present — "the champion forecasts **bookable** demand", the whole censoring module — belongs to a **booking** dataset. Pol 4 measures searches. | `README.md` "Known limitations", `ml/data/censoring.py` | **HIGH** | Replace the censoring narrative with the proxy narrative |
| C3 | Every headline number in the README (WAPE 25.5%, the horizon table, the 80% coverage table) comes from the **synthetic** dataset. Labelled once, then reused as product performance. | `README.md` | **HIGH** | Quarantine under an explicit "synthetic demo" heading, or replace with real-data numbers |
| C4 | README says WAPE **25.5%**; the committed artefact says **0.250388** (25.0%). | `README.md` vs `data/demo_artifacts/metrics.json` | LOW | Correct |
| C5 | README says "**218** tests" in one place and "**128** tests" in another. Actual: **252 collected, 1 failing, 13 skipped.** | `README.md` | MEDIUM | Correct, and fix the failure |
| C6 | `jdatetime` is imported but listed in neither requirements file; the `ImportError` is swallowed silently. | `ml/data/dates.py:252-258` | MEDIUM | Add the dependency; let the failure surface |
| C7 | Chronos/NHITS/Optuna/E2E claims are **NOT VERIFIED** in this environment. | `README.md` | MEDIUM | Either reproduce and cite, or mark as unverified |
| C8 | `DEMO_MODE=true` is the default and serves synthetic output with plausible Persian city names. | `backend/config/settings.py:150` | MEDIUM | Make the warning unmissable, or default to the real run |
| C9 | Bias is reported as `mean(p − t)` while the brief defines `Σ(p−a)/Σa`. | `ml/evaluation/metrics.py:78-79` | LOW | Report both, named distinctly |

---

## 4. Business-insight policy

Keep the existing discipline — every card carries its evidence — and apply it to the new
pickup vocabulary.

**Allowed (computed):**
- "Demand for city X is forecast to peak on 1404-09-12."
- "Pickup for (city X, 1404-09-15) is running 28% above the historical curve at this lead time."
- "82 cities have no observed Azar searches yet; the model still predicts non-zero demand for
  them based on their historical pickup shape."
- "This destination's forecast has moved less than 4% since D-30 — a stable signal."
- "High predicted demand here may warrant a supply-side investigation."

**Forbidden (not computable from this data):**
- "Acquire 52 new properties in city X."
- "Increase prices by 23%."
- "Conversion will rise 12%."
- "Luxury travellers are driving this."
- Any statement about a specific user, property, price or booking.

**The line:** if the number cannot name the artefact it came from, it does not ship.

---

## 5. Data limitations — to be stated verbatim in the report and on slide 13

The Pol 4 dataset contains **only**: `log_date`, `city_code`, `checkin`, `search_count`,
`province_code`, `lat`, `long`.

It contains **no**: user identity · property ID · accommodation category · villa/apartment/hotel
type · capacity · available room nights · actual bookings · booking conversion · price · host
pricing history · marketing channel · campaign attribution · travel purpose · user demographics ·
tour, ticket or restaurant information.

Therefore:

1. **Search is a proxy for intent, not demand.** A search may never convert; a booking may occur
   with no logged search. We optimise the competition's operational target and say so.
2. **We forecast search volume**, not room nights, revenue, or occupancy.
3. **We cannot separate supply-constrained from demand-constrained cities.** A flat forecast may
   mean low interest, or it may mean nothing was available to search for.
4. **Repeat searching is invisible.** One determined user and ten casual users are the same row.
5. **The 60-day booking window is a property of this dataset**, not necessarily of the market.
   Every pickup curve is conditional on it.
6. **7 provinces, not 31.** `province_code` in this file is a coarse regional grouping
   (18–78 cities each). Do not present it as Iran's provincial map.

---

## 6. What to demo, and in what order

**Three minutes, four beats:**

1. **The two clocks** (15 s) — one diagram. Search time vs travel time. This is the insight the
   whole solution rests on.
2. **The pickup curve** (45 s) — the measured 8.7% → 89.1% chart, then a live city where the
   curve is running hot. This is the "we understood the problem" moment.
3. **Overview + heatmap** (60 s) — where and when Azar demand concentrates, national and per city.
4. **Stability + honesty** (60 s) — the D-30 → D-1 fan chart, then the baseline table
   *including the baseline that beat the first model*, then the limitations slide.

**Do not demo:** scenarios, the Data Lab, the model registry, censoring, or the hierarchy
selector. Every one of them is either empty or irrelevant on this dataset, and each costs
credibility the moment a mentor asks what data backs it.
