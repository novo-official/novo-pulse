# POL 4 — Status Matrix & Implementation Backlog

**Audit date:** 2026-09-09 · **Commit:** `3b39c12`
Status legend: ✅ READY · 🟡 PARTIAL · ❌ MISSING · 🔴 BROKEN · ⚠️ RISK
Priority: **P0** forecasting-correctness blocker · **P1** required before submission ·
**P2** product/demo · **P3** future vision

---

## 0. Phase 1 status (implemented 2026-09-09)

The two-clock pipeline is built and the P0 block is closed. Everything below
this section is the original audit record and is kept unedited; this table is
the delta.

| ID | Item | Was | Now | Where |
|---|---|---|---|---|
| F-01 | Two time axes | ❌ | ✅ | `backend/ml/pol4/loader.py` - `log_date`, `checkin`, `days_to_checkin` |
| F-02 | `final = observed + remaining` | ❌ | ✅ | `backend/ml/pol4/baseline.py` |
| F-03 | `evaluation.csv` ingested | ❌ | ✅ | `Pol4Data.observed_at`, `submission.build_grid` |
| F-04 | Pickup curves | ❌ | ✅ | `backend/ml/pol4/pickup.py`, city → province → global |
| F-05 | Cutoff-aware simulation | ❌ | ✅ | `backend/ml/pol4/backtest.py`, 5 walk-forward cutoffs |
| F-05b | Curves fitted only on pre-cutoff check-ins | ⚠️ | ✅ | `PickupCurves.fit` + `test_pol4_leakage.py` |
| F-06 | Cutoff 2025-11-21 enforced | ❌ | ✅ | asserted in `test_pol4_competition.py` |
| F-07 | `predicted >= observed` | ❌ | ✅ | `PickupBaseline.predict` |
| B-02 | Pickup baseline | ❌ | ✅ | pooled WAPE **0.2200** across 5 folds |
| M-02 | Beats meaningful baselines | 🔴 | ✅ | 0.2200 vs 0.3919 last-year, 0.7355 observed-only |
| S-01 | `results.csv` | ❌ | ✅ | `artifacts/pol4/results.csv`, 9,630 rows, validated |
| D-02 | Ingestible without manual work | 🔴 | ✅ | `PYTHONPATH=backend python -m ml.pol4.pipeline` |
| D-03 | Target window never grid-filled to zero | ⚠️ | ✅ | conditional zero-prior; `test_zero_observation_pairs_are_not_predicted_as_zero` |
| E-02 | Normalised bias per the brief | 🟡 | ✅ | `backtest.normalised_bias` |
| E-04 | WAPE by demand bucket | ❌ | ✅ | `backtest_metrics.json` → `by_demand_bucket` |
| R-02 | `jdatetime` in requirements | 🔴 | ✅ | `requirements.txt`; the swallowed `ImportError` is gone |
| R-03 | Suite green | 🔴 | ✅ | **307 passed, 0 failed, 9 skipped** (skips are optional-model deps) |
| R-04 | Output contract covered by tests | ❌ | ✅ | `test_pol4.py`, `test_pol4_competition.py` |

**Still open, and now the top of the queue** (unchanged priority, see §2):

| ID | Item | Why it matters now |
|---|---|---|
| M-03 / E5 | Normalised bias is **-0.150**, worsening to -0.249 at 22-30 days | The single largest remaining WAPE loss |
| E-05 | Folds are not indexed by forecast issue point (D-30…D-1) | Blocks the stability story |
| ST-01 | Forecast stability machinery | Mentor's named differentiator |
| P1-1 | GBDT on remaining demand | Must beat 0.2200 or be rejected in writing |
| DOC-01/02 | Technical report + PPTX | Required deliverables |
| UI-* | Frontend cleanup and rebuild | Phase 2 |

---

## 1. Status matrix

| ID | Area | Requirement | Status | Evidence | File / Function | Problem | Impact | Recommended action | Pri |
|---|---|---|---|---|---|---|---|---|---|
| F-01 | Formulation | Two time axes (`log_date`, `checkin`) | ❌ | `grep -ri log_date` → 0 hits repo-wide | `ml/contract.py:20-25` | Only `TS = "ds"` exists | Competition problem inexpressible | Add `log_date` + `lead_time` to the panel contract | **P0** |
| F-02 | Formulation | `final = observed + remaining` | ❌ | no such target anywhere | — | Demand modelled as a plain daily series | Discards the dominant signal | Build the remaining-demand target | **P0** |
| F-03 | Data | `evaluation.csv` ingested | ❌ | `data/raw/` holds only `.gitkeep` | `ml/data/adapter.py` | 2,686,508 observed searches unused | Measured: WAPE 0.262 → 0.185 achievable | Load and join partial observations at the cutoff | **P0** |
| F-04 | Features | Pickup / completion-fraction curves | ❌ | `grep -ri pickup` → 0 hits | — | The single strongest feature family is absent | Dominant | Build per-city / per-province / global curves, cutoff-aware | **P0** |
| F-05 | Simulation | Cutoff-aware historical simulation (`log_date ≤ T − h`) | ❌ | folds split on `checkin` only | `ml/evaluation/splitters.py:36-101` | Training examples see log dates after the simulated cutoff | Any pickup feature will leak | Filter the log axis when assembling each training example | **P0** |
| F-05b | Leakage | Completion fractions fitted on the full history | ⚠️ | not yet written | — | Trap for the next commit | Silent leak | Estimate curves from check-ins ≤ simulated cutoff only | **P0** |
| S-01 | Submission | `results.csv` (9,630 rows) | ❌ | `grep -r results.csv` → 0 hits | — | No writer, no Azar grid, no `cluster_code` | **Nothing is submittable** | Build the grid + writer + validator | **P0** |
| F-06 | Inference | Cutoff `2025-11-21` enforced | ❌ | no `log_date` concept | — | Cannot be enforced | Correctness | Enforce at load and assert in a test | **P0** |
| F-07 | Constraint | `predicted_final ≥ observed_so_far` | ❌ | — | — | Not expressible today | Free accuracy + credibility | Clamp after inference | **P1** |
| L-01 | Leakage | No random split | ✅ | `grep "train_test_split\|shuffle=True"` → 0 hits | — | — | — | Keep | — |
| L-02 | Leakage | Rolling-origin CV | ✅ | folds strictly sequential in my run | `ml/evaluation/splitters.py` | — | — | Keep; extend to the log axis | — |
| L-03 | Leakage | Causal rolling stats, mechanically asserted | ✅ | 5 tests pass | `tests/test_leakage.py` | — | — | Keep; add two-clock cases | — |
| L-04 | Leakage | Bidirectional covariate fill | ⚠️ | `.ffill().bfill()`, `interpolate(limit_direction="both")` | `ml/data/adapter.py:517-523` | Backward fill leaks future→past | MEDIUM, dormant (Pol 4 has no covariates) | Make forward-only before adding any covariate | P1 |
| M-01 | Model | GBDT trained + CV-selected | ✅ | ran on real data, 47.6 s | `ml/pipelines/training.py` | — | — | Keep the harness | — |
| M-02 | Model | Beats meaningful baselines | 🔴 | 0.2620 vs pickup 0.1854 on identical folds | — | Loses to a 40-line baseline by 29% | **Competition-losing** | Re-target onto the pickup formulation | **P0** |
| M-03 | Model | log1p inverse-transform bias | 🔴 | bias −362 / −361 / −159 by horizon bucket | `ml/pipelines/training.py` `_decide_log1p` | `expm1(mean)` is a median, not a mean | Systematic under-prediction under WAPE | Add smearing/Duan correction, or train Tweedie/Poisson on the raw scale | P1 |
| M-04 | Model | Non-negativity | ✅ | `target_options.non_negative` honoured | `training.py` forecast block | — | — | Keep | — |
| E-01 | Eval | WAPE correct | ✅ | `Σ|t−p| / Σ|t|` | `ml/evaluation/metrics.py:47-49` | — | — | Keep | — |
| E-02 | Eval | Bias per the brief | 🟡 | `mean(p − t)` | `metrics.py:78-79` | Brief asks `Σ(p−a)/Σa` | Mislabelled diagnostic | Add the normalised variant | P1 |
| E-03 | Eval | WAPE by horizon / province / weekday / month | ✅ | `metrics["segment_scores"]`, `["horizon_scores"]` | `training.py` `_segment_scores` | — | — | Keep | — |
| E-04 | Eval | WAPE by demand bucket & on peak dates | ❌ | absent | — | — | Blind spot on the tail that dominates WAPE | Add two segments | P1 |
| E-05 | Eval | Evaluation at D-30/21/14/7/3/1 issue points | ❌ | folds step by horizon | `splitters.py` | — | Cannot show the mentor's stability story | Add lead-time-indexed folds | P1 |
| B-01 | Baselines | Statistical baselines scored | ✅ | 4 baselines on the leaderboard | `ml/models/baselines.py` | — | — | Keep | — |
| B-02 | Baselines | Pickup-projection baseline | ❌ | absent | — | The strongest simple model is missing | It is currently the best model available | Implement first; ship it as the fallback submission | **P0** |
| ST-01 | Stability | Snapshot machinery D-30 → D-1 | ❌ | absent | — | — | Mentor's named differentiator | Build snapshots + revision metrics | P2 |
| T-01 | Trends | `pickup_ratio` = recent ÷ expected | ❌ | absent | — | — | Emerging-demand chart impossible | Compute from the curves | P2 |
| T-02 | Trends | Anomaly detection is genuine | ✅ | robust MAD, phase-aware peaks | `ml/anomaly/detector.py` | — | — | Keep; relabel toward pickup | — |
| C-01 | Clustering | City sharing via global model + city categorical | ✅ | `ent_entity`, `ent_destination` categoricals | `ml/features/engineering.py` `_encode_static` | — | Correct choice — no aggregation penalty | Keep; **do not cluster** unless it measurably wins | — |
| D-01 | Data | 321 cities / 3,298,564 rows / no dupes / no impossible dates | ✅ | verified against the archive | — | — | — | — | — |
| D-02 | Data | Pol 4 ingestible without manual work | 🔴 | I had to pre-aggregate by hand | `ml/data/adapter.py` | Adapter cannot collapse a log-level file into a check-in target | Violates "runs without modification" | Add a Pol 4 loader | P1 |
| D-03 | Data | Missing row = zero **so far**, not final zero | ⚠️ | `_regular_grid` fills gaps with 0 | `adapter.py:487-529` | Correct for history, wrong if applied to Azar | Would zero out 4,282 target pairs | Never grid-fill the target window | **P0** |
| R-01 | Repro | Seeds, Makefile, Docker, one-command demo | ✅ | `set_global_seed(42)` | `training.py:57-70` | — | — | Keep | — |
| R-02 | Repro | `requirements.txt` complete | 🔴 | `jdatetime` imported, listed nowhere | `ml/data/dates.py:252` | Silent `except` hides the failure | 1 test fails on a clean install; Jalali display degrades | Add `jdatetime`; stop swallowing `ImportError` | P1 |
| R-03 | Tests | Suite green | 🔴 | 252 collected · **1 failed** · 13 skipped | `tests/test_competition_data.py::test_jalali_display_round_trips` | see R-02 | Broken suite in a submission | Fix with R-02 | P1 |
| R-04 | Tests | Competition output contract covered | ❌ | absent | — | No test asserts 9,630 rows / no NaN / no negatives / correct dates | A malformed `results.csv` scores zero | Add a submission-validator test | **P0** |
| DOC-01 | Report | Technical report PDF | ❌ | `reports/` = `.gitkeep` | — | — | Required deliverable | Write it | P1 |
| DOC-02 | Deck | PPTX | ❌ | absent | — | — | Required deliverable | Build the 14-slide story | P1 |
| SC-01 | Credibility | Search-as-proxy vs latent intent | ❌ | absent everywhere | README | Framing is "bookable demand" + censoring — a booking-dataset framing | Mentor will ask; there is no answer | State it in report, deck and UI | P1 |
| SC-02 | Credibility | README numbers are synthetic-only | ⚠️ | `metadata.json` → `"dataset": "synthetic_pol_e_chaharom"` | `README.md` | Presented as product performance | Misleads judges | Relabel or replace with real-data numbers | P1 |
| SC-03 | Credibility | Test counts wrong | ⚠️ | README says 218 and 128; actual 252 | `README.md` | — | Small but checkable | Fix | P2 |
| SC-04 | Credibility | No unsupported claims (segmentation, pricing, conversion) | ✅ | searched; none found | — | — | — | **Keep this discipline** | — |
| UI-01 | Frontend | No mock data anywhere | ✅ | 1 grep hit, and it is `placeholder=` | — | — | — | Keep | — |
| UI-02 | Frontend | `/scenarios` dead on this dataset | 🔴 | `adjustable` iterates an empty `tensor.future` | `apps/scenarios/services.py:80-104` | Renders controls over nothing | Broken surface in the demo | Remove | P1 |
| UI-03 | Frontend | `CensoringCard` permanently empty | 🔴 | `{"enabled": false}` in my run's `metrics.json` | `components/dashboard/censoring-card.tsx` | Needs a capacity column | Dead card | Remove | P1 |
| UI-04 | Frontend | 6 nav routes | 🟡 | `app-shell.tsx:26-33` | — | Two are dead, three overlap | Diluted story | Collapse to 4 | P1 |
| UI-05 | Frontend | Pickup / stability / export charts | ❌ | absent | — | — | The three charts that would win the demo | Build after the backend lands | P2 |
| UI-06 | Frontend | `DEMO_MODE=true` default serves synthetic data | ⚠️ | `settings.py:150` | — | Badged, but subtly | A judge could read fake Iranian cities as results | Make the banner unmissable, or default to the real run | P1 |

**Counts:** ✅ 15 · 🟡 5 · ❌ 17 · 🔴 8 · ⚠️ 7 → **P0: 10 · P1: 17 · P2: 6 · P3: 0**

---

## 2. Implementation backlog

### P0 — Forecast correctness (nothing else matters until these are done)

| # | Task | Why | Files | Depends on | Acceptance criteria | Size |
|---|---|---|---|---|---|---|
| P0-1 | **Pol 4 data loader.** Read all three CSVs; build `(city_code, checkin, log_date, search_count)`; derive `lead_time`; expose `observed_at(cutoff)` and `final()`. | Nothing can start without the two-clock panel. | new `backend/ml/data/pol4.py` | — | 3,298,564 rows load; 321 cities; `lead_time ∈ [0,59]`; zero dupes; a unit test pins all four numbers. | **M** |
| P0-2 | **Cutoff-aware aggregation.** `demand_observed(city, checkin, cutoff) = Σ search_count where log_date ≤ cutoff`; `demand_final(city, checkin) = Σ all`. | The definition of the target and of every feature. | `ml/data/pol4.py` | P0-1 | For any cutoff ≥ checkin, observed == final. Property test. | **S** |
| P0-3 | **Pickup curves, cutoff-aware.** Completion fraction by `days_to_checkin`, per city → province → global, fitted **only** on check-ins ≤ the simulated cutoff. | The dominant feature and the fallback model. | new `backend/ml/features/pickup.py` | P0-2 | Global curve reproduces D-30 = 0.087, D-7 = 0.461, D-1 = 0.891. A leakage test asserts the curve is unchanged when post-cutoff rows are replaced with garbage. | **M** |
| P0-4 | **Pickup baseline + `results.csv` writer.** `pred = observed / frac(h)`, clamped at `observed`. Emit 9,630 rows `cluster_code, checkin, predicted_demand`. | **Ships a valid submission today**, and sets the bar every later model must clear. | new `backend/ml/models/pickup_baseline.py`, `scripts/make_submission.py` | P0-3 | `results.csv` has exactly 9,630 rows, 321 × 30 unique pairs, no NaN/inf/negative, Gregorian `2025-11-22 … 2025-12-21`. Backtest WAPE ≤ 0.20 on the two reference folds. | **M** |
| P0-5 | **Submission validator + test.** | A malformed file scores zero regardless of model quality. | new `backend/tests/test_submission.py` | P0-4 | Test fails on: missing pair, duplicate, NaN, negative, wrong date range, wrong column names, Jalali dates. | **S** |
| P0-6 | **Historical simulation harness.** For check-in `T` and horizon `h`: cutoff `T − h`, features from `log_date ≤ T − h`, target `final(T)` (or `remaining`). Emit samples across many `(T, h)`. | Makes every later model leakage-safe by construction. | new `backend/ml/data/simulation.py` | P0-2, P0-3 | A test overwrites every `log_date > cutoff` row with garbage, rebuilds, and asserts features are bit-identical — mirroring `tests/test_leakage.py:53-88`. | **L** |
| P0-7 | **Never grid-fill the Azar window.** | Grid-filling would zero 4,282 pairs that can still receive demand. | `ml/data/adapter.py:487-529` | — | Test asserts a target-window pair with no observation is predicted > 0. | **XS** |

### P1 — Model accuracy

| # | Task | Why | Files | Depends on | Acceptance criteria | Size |
|---|---|---|---|---|---|---|
| P1-1 | **GBDT on remaining demand.** Target `remaining = final − observed`; features: `observed`, `h`, pickup projection, 1/3/7-day pickup, slope, acceleration, days-since-first-search, active-search-days, check-in weekday/weekend, Jalali month, `city_code`, `province_code`, lat/long, city historical mean/median/volatility, same-weekday history. | Where the ML earns its place — the D-7…D-21 band, where the baseline sits at 0.21–0.35. | `ml/models/gbdt.py` (reuse), new feature module | P0-6 | Beats P0-4 by ≥ 10% relative WAPE on ≥ 3 simulated cutoffs, or is rejected in writing. | **L** |
| P1-2 | **Fix the log1p bias.** Smearing/Duan correction, or Tweedie/Poisson on the raw scale. Compare all three. | Measured under-prediction at every horizon bucket. | `ml/pipelines/training.py` | — | Normalised bias within ±2%; WAPE not worse. | **M** |
| P1-3 | **Monotone clamp** `predicted_final ≥ observed_so_far`. | Physically impossible to have less demand than already observed. | inference stage | P0-4 | Zero violations in the output; WAPE non-worse. | **XS** |
| P1-4 | **Lead-time-indexed folds** at D-30/21/14/7/3/1 + WAPE by demand bucket and on peak dates. | The brief asks for exactly these breakdowns. | `ml/evaluation/splitters.py`, `_segment_scores` | P0-6 | `metrics.json` carries all six issue points and both new segments. | **M** |
| P1-5 | **Normalised bias metric** `Σ(p−a)/Σa`. | Match the brief's definition. | `ml/evaluation/metrics.py:78` | — | Both variants reported, distinctly named. | **XS** |
| P1-6 | **Forward-only covariate fill.** | Removes the dormant backward-fill leak. | `ml/data/adapter.py:517-523` | — | No `bfill` / `limit_direction="both"` on non-static columns. | **XS** |

### P1 — Submission requirements

| # | Task | Why | Files | Depends on | Acceptance criteria | Size |
|---|---|---|---|---|---|---|
| P1-7 | Add `jdatetime`; stop swallowing the `ImportError`. | 1 test fails on a clean install; a required deliverable is "requirements.txt listing all external packages". | `requirements.txt`, `ml/data/dates.py:252-258` | — | `pytest` green (252/252). | **XS** |
| P1-8 | One-command pipeline: `python -m src.pipeline` (or `make submission`) → load → simulate → train → validate → `results.csv`. | "Code must run without modification." | `Makefile`, new entrypoint | P0-4 | A fresh clone + `pip install -r requirements.txt` + one command produces a valid `results.csv`. | **M** |
| P1-9 | Technical report (PDF), 13 sections. | Required deliverable. | `reports/` | P1-1 | Covers: proxy vs intent · two clocks · pickup · leakage · walk-forward · baselines · model + why · WAPE · stability · business reading · limitations · future work. | **L** |
| P1-10 | Presentation (PPTX), 14 slides. | Required deliverable. | `reports/` | P1-9 | Follows the story arc in `POL4_EXPERIMENT_PLAN.md` §6. | **M** |
| P1-11 | Correct the README: real-data numbers, real test count, synthetic results clearly quarantined. | Three checkable inaccuracies. | `README.md` | — | Every number traceable to a committed artefact. | **S** |

### P1 — Frontend cleanup

| # | Task | Why | Files | Depends on | Acceptance criteria | Size |
|---|---|---|---|---|---|---|
| P1-12 | Delete Tier 1 (scenarios, censoring card, hierarchy card, 2 e2e scripts). | Dead on this dataset. | see `POL4_FRONTEND_CLEANUP.md` §7 | — | `typecheck` + `build` pass; no dead imports. | **S** |
| P1-13 | De-navigate `/data-lab`; sidebar → 4-item top nav; `/dashboard` → `/overview`. | 6 routes → 4. | `app-shell.tsx` | P1-12 | Four nav items; every one loads real data. | **M** |
| P1-14 | Make the synthetic-data banner unmissable, or default `DEMO_MODE=false` once a real run exists. | A judge must never mistake synthetic output for results. | `settings.py:150`, `status-strip.tsx` | — | Synthetic mode is obvious within one second of page load. | **XS** |

### P2 — Analytics dashboard & reporting

| # | Task | Files | Acceptance criteria | Size |
|---|---|---|---|---|
| P2-1 | Overview: national forecast + heatmap + rankings, all from `results.csv` | `app/overview` | Six KPIs, four charts, zero hard-coded numbers | **M** |
| P2-2 | City Analysis: 30-day forecast + pickup curve + observed/remaining decomposition | `app/city`, new `/forecast/cities/{id}/pickup` | Pickup curve renders actual vs expected for any city/date | **L** |
| P2-3 | Forecast Stability screen + `/forecast/stability` | `app/stability` | Six issue points, revision + convergence + stability score, real backtest data | **L** |
| P2-4 | Reports: model performance + CSV export (incl. `results.csv`) | `app/reports`, `/reports/*.csv` | Six report kinds; every figure reconciles with a chart | **M** |
| P2-5 | Emerging demand from `pickup_ratio` | `/forecast/trends` | Computed, never asserted; no invented business claims | **M** |

### P3 — Future vision (say it, do not build it)

| # | Item | Why it is P3 |
|---|---|---|
| P3-1 | Scenario / what-if simulation | needs covariates Pol 4 does not provide |
| P3-2 | Supply & capacity intelligence | no capacity, no inventory data |
| P3-3 | Price elasticity | no price data |
| P3-4 | Traveller segmentation, conversion, next-best-action | no user, booking or property data — **must never appear as a product claim** |

---

## 3. Three-day execution plan

Adapted to what actually exists. The forecasting core is the constraint; the frontend is not.

### Day 1 — Correctness

- **AM:** P0-1, P0-2, P0-3 — two-clock loader, cutoff-aware aggregation, pickup curves.
- **PM:** P0-4, P0-5, P0-7 — pickup baseline, `results.csv`, validator.
  **End of Day 1 you have a submittable file at WAPE ≈ 0.185.** Everything after this is upside.
- **Parallel (frontend, low risk):** P1-12, P1-13 — delete dead routes, 4-item nav.
- **Parallel (cheap):** P1-7 (`jdatetime`), P1-5 (bias metric), P1-6 (forward-only fill).

### Day 2 — Accuracy

- **AM:** P0-6 historical simulation + its leakage test. Do not skip the test.
- **PM:** P1-1 GBDT on remaining demand; P1-2 bias correction; P1-3 clamp.
  Compare every candidate against the P0-4 baseline on ≥ 3 cutoffs. **If it does not beat the
  baseline, ship the baseline** and say so in the report — that is a stronger result than a
  model you cannot defend.
- **PM:** P1-4 lead-time folds + segment breakdowns → this feeds both the report and Screen 3.
- **Parallel:** P2-1 overview screen against the real `results.csv`.

### Day 3 — Evidence & story

- **AM:** regenerate the final `results.csv` at cutoff `2025-11-21`; run the validator;
  P1-8 one-command pipeline.
- **AM:** P2-2 pickup curve, P2-3 stability screen — the two charts that differentiate you.
- **PM:** P1-9 technical report, P1-10 deck, P2-4 CSV export.
- **PM:** P1-11 README correction, P1-14 demo-mode banner, rehearse.

**Hard rule:** no UI work may start on a day until that day's P0/P1 forecasting items are done.
