# POL 4 — Forensic Audit of `novo-pulse`

**Audit date:** 2026-09-09
**Branch audited:** `claude/pol4-hackathon-audit-vbht5e` @ `3b39c12`
**Auditor scope:** forecasting correctness, temporal leakage, submission readiness, analytics product, frontend focus, scientific credibility.
**Datasets used for verification:** `Pol 4 Hackathon Datasets.rar` (extracted to a scratch directory — **not** committed).
**Nothing functional was changed.** Only documentation was added.

---

## 0. Executive summary

`novo-pulse` is a genuinely well-built, honest, production-shaped **generic** time-series
forecasting platform. The engineering quality is above what a hackathon usually produces:
rolling-origin CV everywhere, mechanically asserted leakage guarantees, real SHAP, real
conformal calibration, no mocked numbers anywhere in the frontend, explicit empty states,
252 tests.

It is also **solving a different problem from the one Pol 4 asks.**

The competition problem has **two time axes** — `log_date` (when the search happened) and
`checkin` (when the stay is wanted). The entire repository has exactly **one**. The strings
`log_date`, `days_to_checkin`, `pickup`, `city_code`, `cluster_code` and `results.csv` do
not appear **anywhere** in the codebase (verified by `grep -ri` across `*.py`, `*.ts`,
`*.tsx`, `*.yaml`, `*.mjs`). `evaluation.csv` — which carries the single most predictive
signal in the whole competition — has no consumer.

### The one number that matters

I built a supervised panel from the real `search_data.csv` (`demand(city, checkin) = Σ search_count`),
pointed the repository's own `TrainingPipeline` at it, and ran it. It works. Then I evaluated a
~40-line pickup-curve baseline on the **identical** backtest folds the pipeline chose.

| Approach | WAPE (folds `2025-09-23…10-22` + `2025-10-23…11-21`) | Evidence |
|---|---:|---|
| **Pickup-curve baseline** — `observed_so_far / expected_completion_fraction(h)`, city→province→global fallback | **0.1854** | reproduced, `scratchpad/headtohead.py` |
| Last-year same check-in date | 0.2638 | reproduced |
| **This repository's LightGBM champion** (check-in axis only) | **0.2620** | reproduced, run in-repo |
| This repository's CatBoost | 0.3014 | reproduced |
| City × weekday mean | 0.4045 | reproduced |
| Observed-so-far, no pickup correction | 0.6942 | reproduced |

**The repository's full ML stack is beaten by 29% by a baseline it cannot express, and it
beats a trivial last-year lookup by 0.7%.** That is not a tuning problem. It is a
formulation problem, and it is the P0 blocker.

### Why the gap exists (verified from the data)

The booking window in this dataset is capped: `max(checkin − log_date) = 59` days
(0…59, median 17). So for every Azar check-in, a **large and precisely knowable** share of
final demand is already visible in `evaluation.csv` at the cutoff:

| Standing at | Share of final demand already observed (global curve, measured) |
|---|---:|
| D-1  | 89.1% |
| D-3  | 69.5% |
| D-7  | 46.1% |
| D-14 | 26.8% |
| D-21 | 16.6% |
| D-30 |  8.7% |

`evaluation.csv` holds 2,686,508 observed searches across 5,348 of the 9,630 target pairs.
The current system throws all of it away.

### Readiness

| Score | /100 |
|---|---:|
| A — Forecasting core | **39** |
| B — Submission readiness | **28** |
| C — Analytics product | **45** |
| D — Frontend focus | **70** |
| E — Scientific credibility | **60** |
| **Overall (weighted 35/25/15/10/15)** | **43** |

### The shortest path to a strong submission

1. Add a second axis. Build the `(city, checkin, observed_at_cutoff, horizon) → final_demand`
   supervised frame. Everything else in the repo is reusable around it.
2. Ship the pickup baseline **first** (WAPE ≈ 0.185 is already a defensible submission).
3. Train a GBDT on *residual/remaining* demand with the pickup projection as a feature.
4. Write `results.csv` (9,630 rows, `cluster_code, checkin, predicted_demand`), no clustering.
5. Cut the frontend from 6 nav routes to 4 and rebuild them around pickup + stability.
6. Technical report PDF + PPTX.

---

## 1. Current architecture

```
data/raw/  (EMPTY — .gitkeep only; the competition data has never been in the repo)
config/data_contract.example.yaml   → points at data/synthetic/daily_demand.csv
backend/ml/
  contract.py        DataContract: ONE timestamp (`ds`), one entity, one target
  data/adapter.py    load → join → rename → collapse dupes → regular grid (missing → 0)
  features/tensor.py (entity × time) matrices, origin index, past/future split
  features/engineering.py  82 features: lags, rollings, calendar, static
  evaluation/        rolling-origin splitter, metric registry, conformal calibration
  models/            naive, seasonal naive 7/30, MA, historical mean, LightGBM, CatBoost,
                     optional Chronos/NHITS, CV-weighted ensemble
  pipelines/training.py  1,175-line end-to-end run → runs/<id>/*.parquet + *.json
backend/apps/        Django REST: dashboard, forecasts, anomalies, models, backtests,
                     insights, scenarios, datasets (Data Lab), experiments
frontend/            Next.js 15, 6 nav routes, Recharts, RTL Persian, zero mock data
data/demo_artifacts/ 4.6 MB of committed artefacts from a SYNTHETIC run
```

**Backend:** 15,975 LOC Python. **Frontend:** 6,449 LOC TS/TSX.

The canonical column names are fixed in `backend/ml/contract.py:20-25`:

```python
TS = "ds"; ENTITY = "entity_id"; TARGET = "y"
DESTINATION = "destination_id"; CATEGORY = "category_id"; MARKET = "market_id"
```

There is no slot for a second date. This is the architectural root of every P0 below.

---

## 2. Current forecasting approach

One training sample is `(entity e, forecast origin o, horizon h) → y[e, o+h]`
(`backend/ml/features/engineering.py:1-10`). Features come from index `≤ o` (history) or
index `= o+h` (known-future covariates). Direct multi-horizon, so a 30-day forecast is one
prediction, not 30 chained ones. Champion picked by rolling-origin CV on the contract's
primary metric. This is a correct and well-implemented design **for a single-clock problem.**

Mapped onto Pol 4 the best it can do is: *"treat `demand(city, checkin)` as a daily series
per city and extrapolate 30 days."* That is the 0.2620 result above. It structurally cannot
consume a partial observation of a **future** target date, because in its world a future
date has no observations by definition.

---

## 3. Temporal modelling assessment

| Concept | Present? | Evidence |
|---|---|---|
| Two time axes (`log_date`, `checkin`) | ❌ MISSING | `grep -ri log_date` → 0 hits repo-wide |
| `lead_time` / `days_to_checkin` | ❌ MISSING | 0 hits (only `avg_lead_time` inside the *synthetic generator*, `backend/ml/data/synthetic.py:311`) |
| Pickup / booking curve | ❌ MISSING | `grep -ri pickup` → 0 hits |
| `final = observed + remaining` formulation | ❌ MISSING | no such target anywhere |
| Partial-observation ingestion (`evaluation.csv`) | ❌ MISSING | no consumer; `data/raw/` is empty |
| Forecast origin / cutoff concept | ✅ PRESENT (one axis) | `PanelTensor.origin_index`, `backend/ml/features/tensor.py:50-53` |
| Direct multi-horizon | ✅ PRESENT | `engineering.py` `assemble()` takes explicit `horizons` |

**Verdict: P0 BLOCKER.** The competition's core concept is absent, not merely weak.

---

## 4. Leakage assessment

This is the repository's strongest area — for the axis it models.

| Check | Result |
|---|---|
| Random split / `shuffle=True` anywhere | **None.** `grep -rn "train_test_split\|shuffle=True"` → 0 hits in `backend/` |
| Rolling-origin walk-forward CV | ✅ `backend/ml/evaluation/splitters.py:36-101`, folds strictly increasing, train ≤ `train_end` |
| Causal rolling statistics | ✅ asserted in `backend/tests/test_leakage.py:14-27` (column *i* uses only ≤ *i*) |
| Future mutation cannot change past features | ✅ `test_leakage.py:53-88` — overwrite everything after the origin with `1e6`, rebuild, assert bit-identical |
| Past covariates never read at `t+h` | ✅ `test_leakage.py:91-118` |
| Training targets never cross the cutoff | ✅ `test_leakage.py:121-131` |
| Scaler/encoder fitted on validation | N/A — tree models, no scaler; static encodings are entity-constant |
| Target encoding using future rows | Not used |

**No leakage was found in the existing code.** The tests are real, mechanical, and they pass.

### But: the leakage risks that *matter for Pol 4* are unguarded because they are unmodelled

| # | Risk | Type | Severity | Why it is not yet caught |
|---|---|---|---|---|
| L1 | Aggregating `search_count` by `checkin` without filtering `log_date ≤ simulated_cutoff` when building a historical training example | Look-ahead on the second clock | **BLOCKER** | The adapter has no `log_date` concept, so every historical target is built from *all* log dates, including ones after the simulated cutoff. Any future pickup-feature work will inherit this unless a cutoff filter is added at the adapter. |
| L2 | Completion-fraction / pickup curves estimated on the **full** history including check-ins after the simulated origin | Statistic fitted on the future | **BLOCKER** | No pickup code exists yet; must be built cutoff-aware from day one. |
| L3 | `evaluation.csv` rows treated as final demand instead of partial | Target contamination | **BLOCKER** | No consumer yet; the trap is in the next commit, not this one. |
| L4 | `_regular_grid` back-fills covariates with `.ffill().bfill()` and `.interpolate(limit_direction="both")` | Backward fill = future→past | **MEDIUM** | `backend/ml/data/adapter.py:517-523`. Harmless today (Pol 4 has no covariates) but it *is* a bidirectional fill on non-target columns. Flag before any covariate is added. |
| L5 | Anomaly/peak detection reads the full backtest frame across folds | Reporting only | **LOW** | `backend/ml/anomaly/detector.py`; affects display, not training. |

**Answer to "Temporal Leakage: YES / NO / POSSIBLE" → NO in current code, POSSIBLE the moment
pickup features are added without a cutoff-aware adapter.**

---

## 5. Model assessment

Verified by an actual run on the real Pol 4 data (`demand-by-checkin` panel, 163,436 rows,
321 cities, `demo` profile, horizon 30, 2 folds, 47.6 s):

| Model | Target | Features | Loss | WAPE | Baseline? |
|---|---|---|---|---:|---|
| LightGBM (champion) | `log1p(demand)` | 82 (lags, rollings, calendar, city id, province, lat/long) | L2 on log1p | **0.2620** | no |
| Ensemble (CV weights) | same | same | — | 0.2710 | no |
| CatBoost | same | same | — | 0.3014 | no |
| Seasonal naive (7) | — | — | — | 0.3347 | yes |
| Historical mean | — | — | — | 0.4208 | yes |
| Moving average | — | — | — | 0.5984 | yes |
| Seasonal naive (30) | — | — | — | 0.6828 | yes |

**Why these models:** the repository does justify its choices (tabular panel, high-cardinality
categorical city/province, non-linear interactions, CPU-only, fast) and it verifies them
empirically rather than asserting them. That part is sound.

**Weaknesses found:**

1. **Systematic under-prediction.** `horizon_scores` shows bias of −362 (1–7d), −361 (8–14d),
   −159 (15–30d) units for CatBoost. `log1p` is applied automatically
   (warning emitted: *"target is skewed (zero ratio 0.21) — training on log1p(target)"*), and
   the inverse transform is `expm1` of the conditional mean in log space, which is the
   conditional **median**, not the mean. With WAPE this under-shoots on a heavy right tail
   (target max = 236,334; mean = 1,738; std = 7,486). No Duan/smearing correction, no
   bias-correction term. Fixable and worth real WAPE.
2. **The horizon feature is nearly useless here.** Errors are *flat-to-worse* at short horizons
   (0.323 at 1–7d vs 0.271 at 15–30d) — the exact inverse of the true structure, where D+1 is
   almost fully observed and therefore *easy*. That inversion is a direct symptom of the
   missing second clock.
3. **No pickup, no lead time, no observed-so-far.** The single most predictive family of
   features is absent.
4. Prediction floor `predicted_final ≥ observed_so_far` — the natural monotonicity constraint
   for this problem — cannot be expressed, because `observed_so_far` does not exist.

---

## 6. Backtest assessment

| Requirement | Status | Evidence |
|---|---|---|
| Walk-forward / rolling origin | ✅ READY | `splitters.py:36-101`; folds printed above are strictly sequential |
| Random 80/20 | ✅ correctly absent | — |
| WAPE implemented correctly | ✅ READY | `backend/ml/evaluation/metrics.py:47-49` — `Σ|t−p| / Σ|t|`, matches the brief exactly |
| MAE, bias | ✅ READY | `metrics.py:40-41`, `:78-79` (bias is **mean signed error**, not the brief's normalised `Σ(p−a)/Σa` — see below) |
| WAPE by horizon | ✅ READY | `metrics["horizon_scores"]`, bucketed |
| WAPE by city / province / weekday / month | ✅ READY | `metrics["segment_scores"]` → `by_destination`, `by_weekday`, `by_month`, `by_horizon` |
| WAPE by demand bucket (high/low destinations) | ❌ MISSING | no bucketing by target magnitude |
| WAPE on peak dates | ❌ MISSING | peaks are detected but never used as an evaluation segment |
| Evaluation at D-30/21/14/7/3/1 issue points | ❌ MISSING | folds step by horizon, not by forecast-issue lead time |
| Forecast stability across issue dates | ❌ MISSING | no snapshot machinery at all |

⚠️ **Bias definition mismatch.** The brief asks for `Σ(predicted − actual) / Σ(actual)`.
`metrics.py:78` implements `mean(p − t)` (absolute units). Both are legitimate; the reported
number is not the one the brief describes. Low severity, one-line fix, but it must not be
labelled as the brief's bias.

---

## 7. Dataset assessment

Verified directly against the extracted archive.

| Check | Result |
|---|---|
| `search_data.csv` loads | ✅ 3,298,564 rows — matches the brief exactly |
| `cities.csv` | ✅ 321 cities, 321 unique, **7** provinces (not 31 — province_code ∈ {149, 289, 424, 490, 754, 929, 987}) |
| `evaluation.csv` | ✅ 65,411 rows, 239 cities, log_date 2025-09-24 → 2025-11-21, checkin 2025-11-22 → 2025-12-21 |
| All 321 cities present in both directions | ✅ zero set difference between `search_data` and `cities` |
| Duplicates on `(log_date, city_code, checkin)` | ✅ zero, in both files |
| Impossible dates (`checkin < log_date`) | ✅ zero |
| `lead_time` range | 0 … **59** days (median 17) — **the booking window is capped at 60 days** |
| `search_data` log_date range | 2023-09-03 → 2025-11-21 (the cutoff) |
| `search_data` checkin range | 2023-11-01 → 2025-11-21 — **no check-in after the cutoff**, so every historical target is complete. No right-censoring in history. |
| Overlap between `search_data` and `evaluation` keys | ✅ zero rows — the two files are a clean split |
| Historical `(city, checkin)` pairs with any demand | 163,436 |
| Demand distribution | mean 1,738 · median 65 · p75 458 · max 236,334 — extremely heavy-tailed |
| `evaluation.csv` coverage | 5,348 / 9,630 target pairs (55.5%); **82 cities have zero observed Azar searches**; 4,282 pairs at zero-so-far |
| Zero semantics | Missing row = zero searches **so far** ≠ final demand zero. The repo's `_regular_grid` (`adapter.py:487-529`) fills gaps with 0, which is correct for the historical target and **wrong** if ever applied to the Azar window. |
| Cutoff enforcement | ❌ no code path enforces `log_date ≤ 2025-11-21` — there is no `log_date` |
| Target grid generation (321 × 30 = 9,630) | ❌ MISSING |
| Memory | ✅ the pre-aggregated panel is 163k rows; full log is 3.3M rows and loads in ~20 s |
| Determinism | ✅ `set_global_seed(42)` seeds `random`, `numpy`, `torch` (`training.py:57-70`) |

**Not in the data — so the product must not claim any of it:** user identity, property ID,
accommodation category, capacity, room nights, bookings, conversion, price, marketing channel,
travel purpose, demographics. See §11.

---

## 8. Submission readiness

| Artefact | Status | Note |
|---|---|---|
| `results.csv` (9,630 rows: `cluster_code, checkin, predicted_demand`) | ❌ MISSING | no writer, no grid builder, no Azar window anywhere |
| `clusters.csv` | N/A | no clustering used; correctly not required |
| Python source | 🟡 PARTIAL | runs and is reproducible, **but** the Pol 4 dataset cannot be ingested without a manual pre-aggregation step performed outside the repo (I had to write it myself). "Code must run without modification" is not met for this dataset. |
| Technical report (PDF) | ❌ MISSING | `reports/` contains only `.gitkeep` |
| `requirements.txt` | 🟡 PARTIAL | present and clean, but **`jdatetime` is imported (`backend/ml/data/dates.py:252`) and listed in neither requirements file** → the Jalali display silently degrades and one test fails on a clean install |
| Presentation (PPTX) | ❌ MISSING | — |
| Tests | 🟡 PARTIAL | 252 collected, **1 failing**, 13 skipped. Zero tests cover the competition output contract. |

**Test-suite reality check** (`python3 -m pytest -q`, this container):
`252 collected · 1 failed · 13 skipped`. The failure is
`backend/tests/test_competition_data.py::test_jalali_display_round_trips` —
`to_jalali_string(2024-08-02)` returns `'2024-08-02 00:00:00'` instead of `'1403-05-12'`
because `jdatetime` is not installed and the helper swallows the `ImportError`
(`dates.py:257-258`).

README claims "**218** unit/integration tests" in one place and "**128** tests" in another.
Neither matches 252. → **README metrics are NOT VERIFIED; two are demonstrably wrong.**

---

## 9. Frontend assessment

Full detail in **`docs/POL4_FRONTEND_CLEANUP.md`**. Summary:

**The premise "overbuilt SaaS/ERP panel" is only partly true, and I will not pretend otherwise.**
There is **no** authentication, no settings page, no profile, no users, no roles, no billing, no
notification centre, no generic CRUD. `grep` for `Math.random`, `mock`, `dummy`, `faker` across
`frontend/` returns **one** hit, and it is the word `placeholder=` on a search input. Every page
is API-driven with real loading / empty / error states. That is a genuine strength and it should
be kept.

The actual problems are different:

1. **6 nav routes where 4 would tell a better story.**
2. **`/scenarios` is a dead page on this dataset.** `ScenarioEngine.adjustable`
   (`backend/apps/scenarios/services.py:80-104`) iterates `tensor.future` — the known-future
   covariates. Pol 4 has **none** (no price, no capacity, no promotion, no holiday file), so the
   list is empty and the page renders controls over nothing. 101 LOC page + 97 LOC chart +
   400 LOC service, all unusable here.
3. **`/data-lab` is 1,024 LOC** — the largest file in the frontend — and exists to upload, profile
   and map an *unknown* dataset. The Pol 4 dataset is known, fixed and three files. This is
   infrastructure for a problem that no longer exists.
4. **`CensoringCard`** on `/models` visualises supply censoring from a capacity column Pol 4
   does not have. It correctly self-disables (`enabled: false, reason: "censoring is disabled in
   the contract"` — verified in my run's `metrics.json`), so it renders as a permanently dead card.
5. **`ForecastHierarchyCard`** offers `listing | destination | category | market`. Pol 4 has only
   city and province (7 of them). Two of four levels are always empty.
6. **Default `DEMO_MODE=true`** (`backend/config/settings.py:150`) serves
   `data/demo_artifacts/` — a **synthetic** run: 140 fake accommodations, fake `booking_count`,
   fake prices, and Persian city labels (تهران، مشهد، کیش، رامسر) that a judge could easily read
   as competition output. It *is* badged (`status-strip.tsx:33-34`, "داده‌های نمایشی مصنوعی فعال است"),
   which is honest — but it is one small badge next to a full dashboard of plausible numbers.
7. **Missing entirely:** pickup curve, forecast-vs-observed decomposition, forecast stability,
   any CSV export, any report surface.

---

## 10. Product assessment

The five questions the README organises around ("what will happen / how confident / why /
what if / how do we know") are the right questions for a general forecasting product. For
Pol 4 the questions are narrower and sharper:

| Pol 4 question | Supported today? |
|---|---|
| What demand, which city, which day, how strong? | 🟡 yes but on the wrong formulation |
| Where and when does demand concentrate? | ✅ heatmap + rankings + peaks exist and work |
| How fast is it picking up right now? | ❌ nothing |
| How stable is the forecast as the check-in approaches? | ❌ nothing |
| Why does the model think so? | ✅ real SHAP, grouped, signed — verified on the real-data run |
| How good is it, measurably? | ✅ leaderboard + segment scores + horizon buckets |
| Can I export it? | ❌ nothing |

"What if conditions change?" — the fifth README question — has **no data to stand on** in Pol 4
and should be demoted to future vision rather than demoed.

---

## 11. Scientific credibility

**What the repo gets right, and it is a lot:**

- No user-level segmentation, no MBTI-style traveller typing, no luxury/economy classification,
  no property-level forecasting, no price optimisation, no booking-conversion prediction, no
  next-best-action. **None of the forbidden claims appear anywhere.** I looked.
- `build_decision_opportunities` (`backend/ml/insights/engine.py:172-250`) emits only
  evidence-bearing cards ("forecast X vs Y in the comparable prior window"). No prescriptive
  "raise prices 23%" style output exists.
- The LLM narrator is explicitly narration-only and gated by an anti-hallucination test
  (`backend/tests/test_llm_narrator.py`).
- Censoring is measured and reported but the target is deliberately **not** silently uncensored —
  the README says so and the code does so.
- "Known limitations" section exists and is accurate.
- SHAP dependence curves are labelled association, not causation.

**What is wrong or missing:**

| # | Issue | Severity |
|---|---|---|
| S1 | **The proxy/intent distinction is absent.** Nowhere does the project say "search is a proxy for latent intent, not demand." The framing it *does* carry — "the champion forecasts **bookable** demand" and the whole censoring module — belongs to a booking dataset and is simply not what Pol 4 measures. | HIGH |
| S2 | The README's headline results (WAPE 25.5%, the horizon table, the 80% coverage table) are from the **synthetic** dataset. They are labelled "synthetic dataset, demo profile" once, then reused as if they were product performance. | HIGH |
| S3 | README WAPE says **25.5%**; the committed artefact says **0.250388** (25.0%). | LOW |
| S4 | Test counts stated as 218 and 128; actual collected is 252 with 1 failing. | MEDIUM |
| S5 | "Chronos-2 adapter verified against the real library", "Optuna took WAPE 0.163 → 0.159", "46 real interactions" — none reproducible in this environment. → **NOT VERIFIED** (not disproven; simply unverifiable here). | MEDIUM |
| S6 | Every modelling decision is justified *for a generic platform*. None is justified *for this dataset* — because none was made for it. | HIGH |

---

## 12. Top blockers

| # | Blocker | Priority | Impact |
|---|---|---|---|
| **B1** | No second time axis. `log_date` does not exist in the codebase. | **P0** | The competition problem cannot be expressed. |
| **B2** | `evaluation.csv` has no consumer. 2.69M observed searches over 5,348 target pairs are discarded. | **P0** | Directly costs ~29% relative WAPE (0.262 → 0.185 measured). |
| **B3** | No `final = observed + remaining` formulation and no pickup curve. | **P0** | The dominant signal is unmodelled. |
| **B4** | No cutoff-aware historical simulation (`log_date ≤ T − h`). | **P0** | Any pickup feature built without it leaks. |
| **B5** | No `results.csv` writer, no 9,630-row Azar grid, no `cluster_code` column. | **P0** | **Nothing is submittable today.** |
| **B6** | Pol 4 data cannot be ingested without hand-written pre-aggregation. | **P1** | Violates "code must run without modification". |
| **B7** | No technical report, no PPTX. | **P1** | Two of six required deliverables. |
| **B8** | `jdatetime` imported but not in requirements; 1 test fails on a clean install. | **P1** | Violates "requirements.txt listing all external packages". |
| **B9** | log1p inverse-transform bias — measured under-prediction at every horizon. | **P1** | Free WAPE left on the table. |
| **B10** | No forecast-stability machinery (D-30 → D-1 snapshots). | **P2** | The mentor's explicitly named differentiator is absent. |
| **B11** | `/scenarios` is a dead page; `CensoringCard` and 2 of 4 hierarchy levels are permanently empty. | **P2** | Demo shows broken surfaces to judges. |
| **B12** | Proxy-vs-intent framing missing from every artefact. | **P2** | Scientific credibility. |

---

## Appendix A — Verification log

Everything below was executed in this container. Scratch scripts live in the session
scratchpad and are intentionally **not** committed.

| # | What | Result |
|---|---|---|
| 1 | Extracted the archive with `bsdtar` | 3 CSVs, row counts match the brief exactly |
| 2 | `pip install -r requirements.txt` | clean |
| 3 | `python3 -m pytest -q` | 252 collected, **1 failed**, 13 skipped |
| 4 | Isolated the failure | `jdatetime` missing from both requirements files |
| 5 | Built `demand_by_checkin.csv` (163,436 rows) from `search_data.csv` | needed because the repo cannot do this itself |
| 6 | Ran `TrainingPipeline` on it, `demo` profile, horizon 30 | 47.6 s, champion LightGBM, **WAPE 0.2620** |
| 7 | Confirmed real SHAP on real data | `y_roll_mean_7` 17.6%, `y_roll_min_7` 16.8%, `y_last` 11.1% — all `demand_history` group |
| 8 | Confirmed censoring self-disables | `{"enabled": false, "reason": "censoring is disabled in the contract"}` |
| 9 | Measured the global pickup curve | D-30 8.7% → D-1 89.1% |
| 10 | Evaluated 8 baselines at 4 simulated cutoffs | pickup baseline 0.158–0.300 |
| 11 | Head-to-head on the pipeline's **own** two folds | pickup **0.1854** vs pipeline **0.2620** |
| 12 | `grep -ri` for `log_date`, `pickup`, `days_to_checkin`, `city_code`, `cluster_code`, `results.csv` | **0 hits each, repo-wide** |

## Appendix B — What "NOT VERIFIED" means here

Claims I could neither confirm nor refute in this environment, and which must not be repeated
as fact in the report or the deck:

- Chronos-2 / NHITS / NBEATSx training and coverage figures (optional deps not installed).
- The Optuna 0.163 → 0.159 improvement.
- The 63-call API audit, the 45-check semantic audit, the 46-interaction browser E2E.
- Every WAPE and coverage number in the README, all of which are synthetic-data results.

The only WAPE figures in this audit that are safe to quote are the ones in §0 and §5, which I
produced here on the real dataset.
