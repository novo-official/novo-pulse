# POL 4 — Experiment Plan

**Audit date:** 2026-09-09
Every experiment below is defined so that it can be **rejected**. Nothing enters the final
model because it sounds advanced; it enters because it beat the current champion on
walk-forward folds, or it does not enter.

---

## 0. Reference numbers (measured 2026-09-09 on the real dataset)

These are the bars everything else must clear. Reproduced in this container.

### Global pickup curve — share of final demand visible at each horizon

| Standing at | D-59 | D-45 | D-30 | D-21 | D-14 | D-7 | D-3 | D-1 | D-0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| observed fraction | 0.001 | 0.024 | **0.087** | 0.166 | 0.268 | **0.461** | 0.695 | **0.891** | 1.000 |

The booking window is capped at 60 days (`lead_time ∈ [0, 59]`, median 17). This single fact
is why partial observations dominate this competition.

### Baseline leaderboard — pooled over folds `2025-09-23…10-22` and `2025-10-23…11-21`

| Rank | Method | WAPE | Uses `evaluation.csv`? |
|---:|---|---:|---|
| 1 | Pickup: `observed / frac(h)`, city → province → global | **0.1854** | ✅ |
| 2 | **Repo's current LightGBM champion** | 0.2620 | ❌ |
| 3 | Last-year same check-in date | 0.2638 | ❌ |
| 4 | Repo's CatBoost | 0.3014 | ❌ |
| 5 | Seasonal naive (7) | 0.3347 | ❌ |
| 6 | City × weekday mean | 0.4045 | ❌ |
| 7 | Historical mean | 0.4208 | ❌ |
| 8 | Observed-so-far, uncorrected | 0.6942 | ✅ |

### Pickup baseline, WAPE by horizon (cutoff `2024-11-21`, the closest seasonal analogue)

| Horizon | D+1 | D+3 | D+7 | D+14 | D+21 | D+30 |
|---|---:|---:|---:|---:|---:|---:|
| observed share | 87.5% | 70.9% | 36.9% | 17.2% | 11.4% | 10.6% |
| pickup WAPE | **0.030** | 0.046 | 0.212 | **0.352** | **0.350** | 0.259 |
| last-year WAPE | 0.208 | 0.197 | 0.216 | 0.498 | 0.448 | 0.229 |

**Read this table before choosing what to model.** D+1…D+3 is essentially solved by
arithmetic. **All the remaining error lives in the D+7 … D+21 band**, where only 11–37% of
demand is visible. That is where — and only where — machine learning has room to earn its place.

The pickup baseline is **biased low by −9.5%**. Fixing that calibration alone is worth WAPE.

---

## 0b. Outcomes (Phase 2, run 2026-09-09)

Every experiment below was run on the five walk-forward cutoffs in §1 and scored
on the full 321 x 30 grid. Artefacts: `artifacts/pol4/experiments.csv`,
`experiment_summary.json`, `backtest_metrics_phase2.json`.

| # | Experiment | Pooled WAPE | Verdict |
|---|---|---:|---|
| E0 | Pickup baseline (Phase 1 champion) | 0.2200 | reference |
| E1 | Global multiplicative calibration | 0.2289 | **rejected** |
| E1 | Horizon-bucket calibration | 0.2204 | **rejected** (neutral) |
| E1 | Shrunk horizon calibration | 0.2206 | **rejected** |
| E1x | Scaling the total rather than the remainder | 0.2426 | **rejected** |
| E2 | GBDT: observed + horizon only | 0.3022 | worse than E0 |
| E3 | + pickup windows | 0.2774 | — |
| E4 | + velocity / acceleration | 0.2739 | — |
| E5 | + activity | 0.2729 | — |
| E6 | + historical pickup curves | 0.2157 | first to beat E0 |
| E7 | + calendar (incl. Jalali) | 0.1889 | — |
| E8 | + city history | 0.1802 | — |
| E9 | + market signals | 0.1755 | — |
| E10 | + province signals | 0.1745 | full feature set |
| M2 | LightGBM, 600 trees, raw target | 0.1692 | — |
| M2b | LightGBM, 600 trees, log1p target | 0.1618 | — |
| M3 | CatBoost (MAE), comparable compute budget | 0.1835 | **rejected** |
| M4 | LightGBM, 4 horizon bands | 0.1563 | rejected on fold risk |
| M6 | LightGBM, 3 horizon bands | 0.1574 | rejected on fold risk |
| **M5** | **LightGBM, 2 horizon bands (1-14, 15-30)** | **0.1581** | **CHAMPION** |
| E9-ens | Model / baseline blend | 0.1623 | **rejected** |

**E2 is the finding worth keeping.** A gradient-boosted model handed only
`observed_total` and `days_to_checkin` scores 0.3022 - materially *worse* than
the arithmetic baseline it was meant to replace. The model does not beat the
baseline by being a model; it beats it only once the pickup curve is handed to
it as a feature (E6, -0.057), and then again on calendar structure (E7, -0.027).

**Calibration (E2 in the original plan) was built, measured and rejected.**
It does what it was designed to do - pooled normalised bias moves from -0.150 to
-0.120 - and does not improve WAPE. Two reasons, both visible in the data:
under a sum-of-absolute-errors metric on a heavy-tailed target the optimal point
forecast sits near the conditional *median*, so some negative bias is correct;
and the fitted factors disagree across folds (0.82 to 1.15), which is a regime
effect rather than a fixed offset. WAPE is the metric, so calibration is not in
the champion.

**E5 (target transform) reversed the Phase 1 expectation, with evidence.** The
audit found the generic platform's automatic `log1p` under-predicting, so it was
A/B'd here rather than inherited. On *remaining* demand with an **L1** objective
it improves everything that was the reason for the original concern:

| | raw target | log1p target |
|---|---:|---:|
| pooled WAPE | 0.1692 | **0.1618** |
| top-1% demand WAPE | 0.1681 | **0.1544** |
| top-1% normalised bias | -0.119 | **-0.100** |
| worst fold | 0.2813 | **0.2756** |

An L1 objective in log space targets the conditional median, which is what WAPE
rewards. The platform's L2-on-log1p targeted a conditional mean and then
under-shot on inverse transform. Same transform, opposite outcome, because the
objective and the target changed.

**E7 (clustering) was not run and is not needed.** The global model already
carries `city_code` as a categorical, which shares information across cities
with no aggregation penalty. Clustering would have to beat 0.1618 by enough to
pay the brief's penalty; nothing in these results suggests it would.

**Horizon splitting (the plan's "one model or several?" question) was decided on
the seasonal analogue, not on pooled WAPE alone.** Every split beats the single
global model pooled, and every split is progressively worse on `2024-11-21` -
the Azar window one year earlier:

| Bands | Pooled WAPE | 2024-11-21 fold | Regression |
|---|---:|---:|---:|
| x1 | 0.1618 | 0.1193 | — |
| **x2 (1-14, 15-30)** | **0.1581** | 0.1222 | +2.5% |
| x3 | 0.1574 | 0.1281 | +7.4% |
| x4 | 0.1563 | 0.1318 | +10.5% |

The regression is monotone in the number of bands, which is a mechanism (less
data per model generalises worse on a low-season window) rather than noise. Two
bands take 2.3 of the 3.4 percentage points of pooled gain for a quarter of the
risk, and the competition is a single shot on precisely a low-season window.
This is a judgement call and it is recorded as one: `ChampionSpec.bands` is one
line, and x4 is there for anyone who weighs pooled WAPE more heavily.

**CatBoost lost on both axes.** At a compute budget comparable to LightGBM's
(300 iterations, depth 6) it scored 0.1835 against 0.1581. At roughly ten times
that budget (700 iterations, depth 8, MAE loss) it had not completed five folds
in half an hour - CatBoost's MAE objective is far more expensive than
LightGBM's - so the tie-break on complexity never had to be made.

**E8 (stability)** is implemented in `ml/pol4/stability.py` and writes
`artifacts/pol4/stability.parquet`.

## 1. Experiment protocol

**Simulated cutoffs (walk-forward, no overlap with each other's training windows):**
`2024-11-21` (seasonal analogue) · `2025-05-21` · `2025-08-21` · `2025-09-22` · `2025-10-22`.

For each cutoff `C`:
- features may read **only** rows with `log_date ≤ C`;
- pickup curves and any aggregate statistic are fitted **only** on check-ins `≤ C`;
- targets are `final(city, T)` for `T ∈ [C+1, C+30]`;
- the full 321 × 30 grid is scored, zeros included.

**Primary metric:** WAPE, pooled across cities and dates (exactly the competition definition).
**Diagnostics:** MAE, normalised bias `Σ(p−a)/Σa`, WAPE by horizon, by province, by weekday,
by demand decile, on detected peak dates.

**Acceptance rule:** a change ships only if it improves pooled WAPE on **at least 4 of 5**
cutoffs and does not degrade any single cutoff by more than 2% relative. Anything else is
noise, and gets written up as a rejected experiment.

---

## 2. Experiment queue

### E1 — Pickup fallback hierarchy *(do first)*
**Question:** city curve vs province vs global vs a shrinkage blend?
**Why:** 82 of 321 cities have zero observed Azar demand and many more are thin; a city curve
fitted on 200 searches is noise.
**Variants:** (a) global only · (b) province only · (c) city with a volume threshold — the
audit used > 20,000 historical searches · (d) empirical-Bayes shrinkage of the city curve
toward its province.
**Baseline to beat:** 0.1854 (variant c). Global-only measured 0.3184 at the 2024-11-21 cutoff,
so the fallback hierarchy is already worth ~35% relative — tune the threshold, do not remove it.
**Expected:** 0.170 – 0.185. **Size:** S.

### E2 — Bias calibration of the pickup projection
**Question:** the projection is 9.5% low. Is that a fixed multiplicative offset, or
horizon-dependent, or seasonal?
**Why:** WAPE punishes systematic bias linearly; this is the cheapest available win.
**Variants:** global scalar · per-horizon scalar · per-(horizon, province) scalar, each fitted
on cutoffs strictly before the evaluated one.
**Expected:** −0.005 to −0.015 WAPE. **Size:** XS. **Risk:** overfitting the correction —
fit on ≥ 3 prior cutoffs only.

### E3 — Weekday / calendar shape on the *remaining* demand
**Question:** does the residual after the pickup projection carry check-in weekday structure?
**Why:** Thursday/Friday check-ins behave differently, and the pickup curve is estimated
across all weekdays.
**Variant:** per-weekday completion fractions.
**Expected:** small (−0.003 to −0.010). **Reject if** it fails on 2 of 5 cutoffs.

### E4 — GBDT on remaining demand *(the main event)*
**Target:** `remaining = final − observed_at_cutoff`, trained on `log1p`, predicted then added
back to `observed`.
**Why this target, not `final`:** it removes the part already known exactly, so the model spends
its capacity on the genuinely uncertain part — the D+7…D+21 band.
**Features:**
- *A — current observation:* `observed_cumulative`, last-1/3/7-day pickup, pickup slope, pickup
  acceleration, days since first search, count of active search days, `observed / frac(h)`
  (the E1 projection as a feature).
- *B — horizon:* `days_to_checkin`.
- *C — calendar:* check-in weekday, weekend flag, Jalali month/day, Iranian holiday flags
  **only if** a defensible calendar is available.
- *D — city:* `city_code` (categorical), `province_code`, lat, long.
- *E — history:* city demand mean/median, same-weekday mean, rolling volatility, the city's own
  historical completion-fraction shape.
- *F — cross-city:* province aggregate pickup at the same horizon; k-nearest-geographic-neighbour
  mean pickup.
**Models:** LightGBM and CatBoost. **Why these:** the panel is tabular with one high-cardinality
categorical (`city_code`, 321 levels) plus non-linear interactions between horizon and observed
volume; both handle that natively on CPU in seconds, and the repository already has fitted,
tested, SHAP-capable adapters for both (`ml/models/gbdt.py`). **CatBoost specifically** for its
ordered target statistics on `city_code`; **LightGBM** for speed and for the quantile heads the
uncertainty band needs. No deep model is proposed: 163k historical `(city, checkin)` pairs with a
60-day window is a tabular problem, not a sequence-learning one, and nothing here justifies the
training cost.
**Loss:** WAPE is a sum of absolute errors → **MAE / L1 is the aligned objective**. Test L1
against L2-on-log1p and Tweedie/Poisson on the raw scale. Do not assume log1p; the audit
measured that the current log1p path under-predicts at every horizon bucket.
**Expected:** 0.150 – 0.175 pooled. **Reject if** it does not beat E1+E2 on 4 of 5 cutoffs.
**Size:** L.

### E5 — Target transform & inverse-transform bias
**Question:** `log1p` + `expm1` returns a conditional *median*. Under WAPE with a max/mean ratio
of 136 (236,334 / 1,738), how much does that cost?
**Variants:** raw + L1 · log1p + `expm1` · log1p + Duan smearing · Tweedie · Poisson.
**Why it matters:** measured bias in the current pipeline is −362 / −361 / −159 units across the
1–7 / 8–14 / 15–30 day buckets. That is not a rounding effect.
**Expected:** −0.005 to −0.020. **Size:** M.

### E6 — Monotone constraint `predicted_final ≥ observed_so_far`
**Why:** it is physically impossible to end below what has already been counted, and violations
are pure loss under WAPE.
Also enforce `predicted ≥ 0` and integrality.
**Expected:** small but free; also a credibility point in the report. **Size:** XS.

### E7 — City sharing / clustering
**Question:** does explicit clustering beat a single global model with `city_code` as a
categorical feature?
**Why it must be tested, not assumed:** the brief penalises aggregation, so clustering must
*earn* its penalty. The repository already does the right thing by default — one global model
with the city as a categorical, which shares information across cities with **zero** aggregation
penalty.
**Variants:** (a) global + `city_code` categorical *(current, and the default recommendation)* ·
(b) per-city models · (c) K-means on pickup-curve shape + demand scale + geography, then
per-cluster models · (d) cluster id as an extra feature on the global model.
**Prediction:** (a) or (d) wins. (b) fails on the 82 thin cities. (c) only pays if it beats (a)
by enough to justify the aggregation penalty — which, given that (a) already shares information
freely, is unlikely.
**Decision rule:** **submit unclustered** (`cluster_code = city_code`) unless (c) beats (a) by
> 5% relative WAPE. **Size:** M.

### E8 — Forecast stability
**Question:** how much does the prediction for a fixed `(city, checkin)` move as the cutoff
advances D-30 → D-21 → D-14 → D-7 → D-3 → D-1?
**Why:** the mentor named it explicitly, and it is a genuine model-quality signal — a forecast
that swings 40 → 190 → 80 is untrustworthy even if its final WAPE looks acceptable.
**Metrics:** mean absolute revision · mean relative revision · monotone-convergence rate
(does |error| shrink as `h` shrinks?) · a stability score.
**Note:** the pickup formulation should be *naturally* stable, because each new day of
observation moves the projection by a bounded amount. **Measure it — do not claim it.**
**Size:** M. **Output:** Screen 3 and a report section.

### E9 — Ensemble
Only after E4. Blend the pickup projection with the GBDT using CV-derived weights — the
repository already does exactly this (`ml/models/ensemble.py`, `compute_weights`). Expect the
blend to help most at long horizons, where the GBDT has little signal and the projection is
noisy.
**Reject if** it does not beat the better single model on 4 of 5 cutoffs.

---

## 3. What will not be attempted, and why

| Not doing | Why |
|---|---|
| Deep learning (NHITS / NBEATSx / Chronos) | 163k tabular rows with a 60-day window. No evidence a sequence model helps; large time cost; the repo's own tree models already train in 47 s. Revisit only if E4 plateaus with days to spare. |
| Hyper-parameter search (Optuna) | Last thing, if at all. Formulation is worth ~29% relative WAPE; hyper-parameters are worth low single digits. |
| Clustering as a headline feature | Penalised by the brief, and the global-model-plus-categorical already shares information for free. |
| Any user / property / price / capacity modelling | The data does not contain it. See `POL4_PRODUCT_REVIEW.md`. |

---

## 4. Feature audit — target state

| Class | Feature | Implemented today? | Leakage-safe? | Why useful | Evidence |
|---|---|---|---|---|---|
| A | `observed_cumulative` at cutoff | ❌ | needs `log_date ≤ C` filter | The strongest single predictor | 89% of D-1 demand is already visible |
| A | last-1 / last-3 / last-7-day pickup | ❌ | same | Distinguishes accelerating from stalling | — |
| A | pickup slope, acceleration | ❌ | same | Separates slow / normal / fast / sudden | — |
| A | days since first search, active search days | ❌ | same | Onset timing carries demand scale | — |
| A | `observed / frac(h)` (projection) | ❌ | fit on check-ins ≤ C | It *is* the baseline; as a feature it gives the GBDT a strong prior | WAPE 0.1854 alone |
| B | `days_to_checkin` | ❌ | trivially safe | Governs how much is knowable | curve in §0 |
| C | check-in weekday / weekend | 🟡 partial (`calendar_frame`) | safe | Weekly demand shape | — |
| C | Jalali month / day | 🟡 display only | safe | Azar-specific seasonality, Nowruz, Yalda | ⚠️ `jdatetime` missing from requirements |
| C | Holiday flags | ❌ | safe **if** the calendar is authored offline | Real driver | Only if defensible — do not fabricate |
| D | `city_code` categorical | ✅ | safe | Shares information with no aggregation penalty | `_encode_static` |
| D | `province_code` (7 levels) | ✅ | safe | Fallback for thin cities | verified: 7 provinces, 18–78 cities each |
| D | lat / long | ✅ (static) | safe | Geographic smoothing | — |
| E | city demand mean / median / volatility | 🟡 rolling stats exist on the wrong axis | needs cutoff filter | Level and dispersion prior | — |
| E | same-weekday history | ✅ (`y_same_phase_mean`) | safe | Weekly phase | verified in the run |
| E | city completion-curve shape | ❌ | fit ≤ C | Cities differ in how early they are searched | — |
| F | province aggregate pickup at same horizon | ❌ | fit ≤ C | Pools the thin cities | — |
| F | k-NN geographic neighbour pickup | ❌ | fit ≤ C | Regional shocks propagate | — |

---

## 5. Final inference checklist (cutoff `2025-11-21`)

- [ ] `log_date ≤ 2025-11-21` enforced at load, and asserted in a test.
- [ ] `evaluation.csv` merged as **partial** observation — never as final demand.
- [ ] Grid = 321 cities × 30 dates = **9,630** rows. Cities with zero observed Azar demand
      (82 of them) still receive a positive prediction.
- [ ] Columns exactly `cluster_code, checkin, predicted_demand`; `cluster_code = city_code`.
- [ ] `checkin` Gregorian, `2025-11-22 … 2025-12-21`.
- [ ] No duplicates, no NaN, no ±inf, no negatives, `predicted ≥ observed_so_far`.
- [ ] Sanity: national total in the same order of magnitude as the pickup-projected
      **7,948,595** (Azar 1403 actual over the analogous window was 6,787,514).
- [ ] `clusters.csv` **not** emitted (no clustering used).

---

## 6. Report & presentation outline

### Technical report (PDF) — 13 sections
1. **Problem definition** — latent intent vs the observed search proxy. Say plainly: search is
   not demand; the competition defines an operational proxy, and we optimise that proxy while
   naming the gap.
2. **Objective** — `demand(city, checkin) = Σ search_count`, 321 × 30, Azar 1404.
3. **Two clocks** — `log_date` vs `checkin`; `lead_time ∈ [0, 59]`, median 17.
4. **Pickup formulation** — `final = observed + remaining`; the measured curve (8.7% at D-30 →
   89.1% at D-1).
5. **Leakage prevention** — cutoff-simulated history; the mechanical bit-identity test.
6. **Validation** — walk-forward at five cutoffs; why random splits are never used.
7. **Baselines** — the full table from §0, including the ones we lost to.
8. **Modelling and why** — MAE aligned to WAPE; CatBoost/LightGBM for a tabular panel with one
   321-level categorical; why no deep model.
9. **Results** — WAPE overall, by horizon, by province, by demand bucket, plus bias.
10. **Forecast stability** — D-30 → D-1 revision behaviour.
11. **Business interpretation** — conservative, computed statements only.
12. **Limitations** — the full list in `POL4_PRODUCT_REVIEW.md` §2.
13. **Future work** — everything the data cannot support today.

### Presentation (PPTX) — 14 slides
1. Room-nights expire — demand must be known before the night arrives.
2. Demand is hidden — we observe searches, not intent.
3. Two clocks — search time vs travel time.
4. The pickup curve — **the measured 8.7% → 89.1% chart. This is the slide that wins the room.**
5. Leakage-safe historical simulation.
6. Baselines — including the one that beat our first model.
7. The model, and why.
8. WAPE results, by horizon.
9. Forecast stability.
10. The Azar 1404 forecast.
11. The product dashboard.
12. Business value.
13. Limitations — what search data cannot tell us.
14. Future vision.

**Slide 6 is not optional.** Showing that a 40-line baseline beat the first ML attempt, and then
showing what you did about it, is the single most credible thing this team can put in front of a
technical mentor.
