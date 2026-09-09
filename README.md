# Novo Pulse — AI Demand Forecasting

**پل چهارم | هوش پیش‌بینی تقاضا**
پیش‌بینی، تحلیل و شبیه‌سازی تقاضای بازار اقامت و گردشگری

A demand-forecasting platform for the Pol 4 competition: two-clock pickup
modelling, leakage-safe walk-forward evaluation, and a validated submission -
built on the competition data and nothing else.

---

## The dashboard

Four screens, and each answers one question a judge actually asks:

| Screen | Question | What it shows |
|---|---|---|
| `/overview` | What will happen, where and when? | KPI row, national 30-night forecast, city × date heatmap, top cities, province split, emerging demand |
| `/city` | What about *this* city? | 30-night forecast, pickup curve against the city's own history, peak nights, recent history |
| `/stability` | Can I trust it? | the same (city, night) predicted at D-30 → D-1, against the realised actual |
| `/reports` | How do we know it works? | champion vs baseline per fold, error by horizon, feature importance, CSV exports |

Every figure on every screen traces to a generated artefact -
**[docs/POL4_TRACEABILITY.md](docs/POL4_TRACEABILITY.md)** names the file and the
computation behind each one. There is no mock data, no demo mode and no
placeholder metric anywhere in the product.

The dashboard never opens `search_data.csv`. Its 3.3 million rows are read once,
offline, by `python -m ml.pol4.pipeline`; every page load reads a pre-aggregated
artefact through `/api/v1/pol4/*`.

**What it does not claim.** Demand is latent and unobservable. The product says
*search-based demand as an observable proxy for travel intent* - never that it
predicts intent directly. Feature importance is labelled as what the model
splits on, not what causes demand. There is nothing about bookable demand,
capacity, price or properties, because this dataset contains none of it.

---

## Quick start

```bash
git clone <repo> && cd novo-pulse

make demo        # venv + deps + the Pol 4 pipeline (~12 min)
make dev         # backend :8000, frontend :3000
```

Open <http://localhost:3000>. Add `?presentation=true` for the judging view.

With Docker:

```bash
docker compose up --build
docker compose exec backend make pol4
```

Manual setup:

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python backend/manage.py migrate
PYTHONPATH=backend .venv/bin/python -m ml.pol4.pipeline
.venv/bin/python backend/manage.py runserver          # :8000

cd frontend && npm install && npm run dev             # :3000
```

---

## Architecture

```
                    ┌──────────────────────────────────┐
   any dataset ───► │  Data Adapter  (contract-driven) │
                    │  profile → map → validate        │
                    └───────────────┬──────────────────┘
                                    ▼
                    ┌──────────────────────────────────┐
                    │  Panel Tensor  (entity × time)   │
                    │  history │ known-future split    │
                    └───────────────┬──────────────────┘
                                    ▼
                    ┌──────────────────────────────────┐
                    │  Feature Engine (leakage-safe)   │
                    │  lags · rollings · calendar      │
                    └───────────────┬──────────────────┘
                                    ▼
     ┌──────────────┬───────────────┼───────────────┬──────────────┐
     ▼              ▼               ▼               ▼              ▼
 Baselines     LightGBM        CatBoost        Chronos-2        NHITS
 (the floor)   (global)        (global)        (optional)     (optional)
     └──────────────┴───────────────┼───────────────┴──────────────┘
                                    ▼
                    ┌──────────────────────────────────┐
                    │  Rolling-Origin Cross Validation │
                    │  → leaderboard → ensemble weights│
                    └───────────────┬──────────────────┘
                                    ▼
        ┌───────────┬───────────────┼───────────────┬───────────┐
        ▼           ▼               ▼               ▼           ▼
    Forecast   Uncertainty     Explainability   Anomalies   Scenarios
                                    │
                                    ▼
                     runs/<id>/*.parquet + *.json
                                    │
                          Django REST  →  Next.js
```

```
backend/
  config/            Django project + API routes
  apps/
    core/            health, system info
    datasets/        Data Lab: upload, profile, map, validate
    forecasting/     dashboard, forecasts, anomalies, backtests, models
    experiments/     training runs, leaderboard, model report
    insights/        structured business insights
    scenarios/       what-if simulation
  ml/
    contract.py      the data contract - the seam to any dataset
    data/            adapter, profiler, validator, frequency
    features/        tensor, engineering, rolling, calendar
    models/          base, registry, baselines, gbdt, chronos, neural, ensemble
    evaluation/      metrics, splitters, backtest, uncertainty
    explainability/  SHAP
    anomaly/         residual + forecast anomalies, peak detection
    insights/        insight engine, narrator
    pipelines/       the end-to-end training pipeline
  tests/             369 tests
frontend/            Next.js 15 · TypeScript · Tailwind · Recharts (RTL/Persian)
config/              data contract + training profiles
docs/                COMPETITION_DAY.md
scripts/             download_models.py
```

---

## Features

**Forecasting**
- Global gradient-boosted models (LightGBM, CatBoost) trained across all entities
- Statistical baselines: naive, seasonal naive 7/30, moving average, historical mean, seasonal mean
- Optional Chronos-2 (zero-shot) and NHITS / NBEATSx — each degrades to a warning if absent
- Data-driven ensemble; weights come from CV, never hard-coded
- Direct multi-horizon: one fitted model serves 1…H with no recursive error compounding
- Hierarchical bottom-up aggregation: listing → destination → category → market

**Uncertainty**
- Native LightGBM/CatBoost quantile heads, or split-conformal calibration per horizon bucket
- Observed coverage is measured on out-of-sample folds and reported, never assumed
- A documented confidence score (interval width, validation error, horizon, history depth)

**Explainability**
- Exact tree SHAP via each library's native contribution API
- Global drivers rolled into human-readable groups (holiday, price, season, search activity…)
- SHAP dependence curves for price, labelled as association rather than causation

**Detection**
- Residual anomalies (robust MAD z-score) and forecast anomalies
- Phase-aware peak detection: a busy Friday is not a "peak", a festival is
- Demand censoring: measures how much of the history sits at capacity, and
  estimates the demand that was there but could not be sold

**Adaptability**
- Automatic schema detection with editable suggestions
- Frequency detection (hourly / daily / weekly / monthly) with override
- Data-quality gate with a health score and potential-leakage warnings
- Any target, any metric, any hierarchy — all from configuration

---

## Tech stack

| Layer | Choice |
|---|---|
| Backend | Python 3.12 · Django 5 · Django REST Framework |
| Data | pandas · NumPy · PyArrow |
| ML | LightGBM · CatBoost · scikit-learn · SHAP |
| Optional ML | Chronos-2 · NeuralForecast (NHITS/NBEATSx) · PyTorch · Optuna — all verified running |
| Database | SQLite (dev) — Postgres via `DATABASE_URL` |
| Frontend | Next.js 15 (App Router) · TypeScript · Tailwind · shadcn-style UI · TanStack Query · Recharts |
| Deployment | Docker + docker compose |
| LLM | Optional local Qwen3-4B through Ollama — **narration only** |

**No paid API is used anywhere.** No OpenAI, Anthropic, Gemini, Groq or
Together. Everything runs locally and free.

---

## No synthetic data

There is none. The repository contains exactly four data files, all of them the
competition's:

```
data/raw/pol4/search_data.csv        3,298,564 search events
data/raw/pol4/evaluation.csv            65,411 partial Azar observations
data/raw/pol4/cities.csv                   321 cities with coordinates
data/raw/pol4/city_code_mapping.csv        321 city and province names
```

The synthetic dataset, its generator, the precomputed demo run, the `seed_demo`
command and `DEMO_MODE` have all been removed. Every number this product shows
is computed from the four files above; when no model has been trained, each
endpoint reports that plainly rather than serving a fabricated run.

Those files are gitignored, so a fresh clone has no data until you place them.
`city_code_mapping.csv` is optional - without it every report falls back to
numeric city codes.

## ML pipeline

```
Load → Validate → Normalise schema → Aggregate → Feature engineer
     → Rolling-origin CV → Train baselines → Train ML models
     → Optional deep models → Compare → Build ensemble
     → Calibrate uncertainty → Refit on full history → Forecast
     → Explain → Detect anomalies → Save artefacts → Write report
```

### Why the features cannot leak

One training sample is `(entity e, forecast origin o, horizon h)` predicting
`y[e, o+h]`. Features come from exactly two places:

- index `≤ o` — lags, rolling statistics, past covariates (**historical**)
- index `= o+h` — calendar, planned price, holiday flags (**known-future**)

Nothing else is reachable. `backend/tests/test_leakage.py` asserts this
mechanically: it overwrites every post-origin value with garbage, rebuilds the
features, and requires them to be bit-identical.

### Training profiles

| Profile | Models | Rows | Folds | Time (4-core CPU) |
|---|---|---|---|---|
| `demo` | baselines + LightGBM + CatBoost | 150k | 2 | ~2 min |
| `competition` | + Chronos, more trees | 1.2M | 3 | 10–40 min |
| `full` | everything + Optuna | 4M | 4 | 30–90 min |

Hyper-parameter search (`full` only) runs Optuna against a held-out time
window, hard-capped by `tuning.max_minutes`, and keeps the profile defaults
unless the search genuinely beats them. Every outcome is reported - including
the runs that changed nothing.

```bash
make train PROFILE=competition HORIZON=90 METRIC=wape
```

---

## Evaluation

Random train/test splitting is **not used anywhere**. Every score comes from
rolling-origin (walk-forward) cross-validation:

```
Fold 1:  TRAIN ═════════════════╡ VALID ────
Fold 2:  TRAIN ═══════════════════════╡ VALID ────
Fold 3:  TRAIN ═════════════════════════════╡ VALID ────
```

Metrics: `mae`, `rmse`, `wape`, `mape`, `smape`, `rmsle`, `r2`, `bias`,
`mase`, `poisson_deviance`. Selecting one re-orients the leaderboard, champion
selection, ensemble weighting and all reported scores.

The generic platform, measured on the Pol 4 panel it can express - demand
aggregated by check-in date, one time axis, `evaluation.csv` unused (`demo`
profile, 30-day horizon, two rolling-origin folds):

| Model | WAPE |
|---|---:|
| **LightGBM (champion)** | **26.2%** |
| Ensemble | 27.1% |
| CatBoost | 30.1% |
| Seasonal naive (7) *(best baseline)* | 33.5% |
| Historical mean | 42.1% |
| Moving average | 59.8% |
| Seasonal naive (30) | 68.3% |

These are the honest numbers for the single-axis approach, and they are the
reason `backend/ml/pol4/` exists: the two-clock champion scores **15.8%** on the
same data. See [POL 4 Competition Mode](#pol-4-competition-mode).

### Long horizons

Each horizon is predicted **directly** — one sample is
`(entity, forecast origin, horizon)` with the horizon as an explicit feature —
so a 90-day forecast is a single prediction rather than 90 chained ones and
there is no error to compound. Accuracy is reported per horizon bucket rather
than as one headline number, and interval coverage is measured on held-out
folds rather than assumed: the served interval is whichever of the model's own
quantiles or the conformal residual bounds lands closer to nominal, and the run
records which it picked and why.

`reports/model_report.md` is regenerated automatically after each run.

---

## POL 4 Competition Mode

**Everything in this section is measured on the real Pol 4 datasets.**

### The problem has two clocks

| Clock | Column | Meaning |
|---|---|---|
| 1 | `log_date` | when somebody searched |
| 2 | `checkin` | the night they want to stay |

`days_to_checkin = checkin - log_date`, and in this dataset it never exceeds
**59** (median 17). So demand for a `(city, checkin)` pair accumulates over the
~60 days before the stay, and at any cutoff the pair is only *partially*
observed:

```
final_demand = observed_demand(cutoff) + remaining_pickup(cutoff)
```

The generic platform in `backend/ml/` models one time axis and cannot express
this, so the competition lives in its own domain package, `backend/ml/pol4/`.

### The measured pickup curve

Share of a check-in's final demand that is already visible, by horizon:

| Standing at | D-30 | D-21 | D-14 | D-7 | D-3 | D-1 |
|---|---:|---:|---:|---:|---:|---:|
| observed fraction | 0.087 | 0.166 | 0.268 | 0.461 | 0.695 | 0.891 |

This is why `evaluation.csv` matters: at the competition cutoff it already
carries 2,686,508 searches across 5,348 of the 9,630 target pairs.

### Run it

```bash
# 1. put the three competition CSVs here (they are gitignored)
#    data/raw/pol4/{search_data.csv,evaluation.csv,cities.csv}

make pol4                 # champion backtest + stability + results.csv   (~12 min)
make pol4-baseline        # the pickup baseline alone - fast fallback      (~15s)
make pol4-ablation        # rerun the full feature ladder                  (~35 min)
make pol4-submission      # regenerate results.csv, nothing else           (~1 min)
make test-pol4            # the Pol 4 test suite
```

Or directly, with no source edits:

```bash
PYTHONPATH=backend python -m ml.pol4.pipeline
```

Artefacts land in `artifacts/pol4/`:

| File | What it is |
|---|---|
| `results.csv` | the submission: 9,630 rows of `cluster_code, checkin, predicted_demand` |
| `results_named.csv` | the same rows for humans: city and province names, observed-so-far, predicted remaining |
| `backtest_metrics_phase2.json` | champion vs baseline: every fold, WAPE by horizon / province / weekday / demand bucket / observation state / high-demand slice |
| `experiments.csv` | every experiment run, one row each, sorted by WAPE |
| `experiment_summary.json` | the same with per-fold detail and the rejections |
| `feature_importance.json` | the champion's gain importance, all 66 features |
| `stability.parquet` | D-30 → D-1 forecast snapshots for a historical window |
| `pickup_curves.parquet` | the fitted completion curves (global, province, city) |
| `run_summary.json` | data validation, model config, submission report |

### Results — walk-forward, five simulated competitions

Each fold picks a historical cutoff, fits everything on data available at that
moment, predicts the following 30 check-in dates from partial observations, and
is scored on the full 321 x 30 grid.

| Model | Pooled WAPE | Normalised bias |
|---|---:|---:|
| **Champion — LightGBM on remaining demand, 2 horizon bands** | **0.1581** | -0.099 |
| Same model, one global band | 0.1618 | -0.103 |
| CatBoost, comparable compute budget | 0.1835 | -0.114 |
| Pickup baseline (Phase 1 champion) | 0.2200 | -0.150 |
| Calibrated pickup baseline | 0.2206 | -0.120 |
| Last-year same check-in date | 0.3919 | — |
| City x weekday mean | 0.5588 | — |
| Observed-so-far, uncorrected | 0.7355 | — |

**28.1% better than the Phase 1 champion.** Per fold:

| Fold cutoff | Champion | Pickup baseline |
|---|---:|---:|
| 2024-11-21 | 0.1222 | 0.2065 |
| 2025-05-21 | 0.2711 | 0.2999 |
| 2025-08-21 | 0.1276 | 0.1976 |
| 2025-09-22 | 0.1087 | 0.1599 |
| 2025-10-22 | 0.1182 | 0.2167 |

By horizon — the gain is largest exactly where the baseline was weakest:

| Days ahead | 1-3 | 4-7 | 8-14 | 15-21 | 22-30 |
|---|---:|---:|---:|---:|---:|
| Champion | 0.029 | 0.064 | 0.114 | 0.139 | 0.279 |
| Pickup baseline | 0.040 | 0.100 | 0.166 | 0.219 | 0.357 |

And where the demand actually is — the top 1% of pairs carry a quarter of it:

| Slice | Champion WAPE | Baseline WAPE |
|---|---:|---:|
| Top 1% of pairs by demand | **0.146** | 0.202 |
| Top 5% | 0.150 | 0.208 |
| Pairs with something observed | 0.157 | 0.219 |
| Pairs with nothing observed yet | 0.637 | 0.769 |

The model predicts **remaining** demand, not total:

```
predicted_final = observed_so_far + max(0, predicted_remaining)
```

so the Phase 1 floor holds by construction - it cannot predict away demand that
has already been counted - and all of its capacity goes on the uncertain part.

### Why two horizon bands, not one and not four

One model per horizon band trades specialisation against data per model. All
four variants beat the single global model on pooled WAPE, and all four are
progressively worse on `2024-11-21` - which is the Azar window one year earlier,
and therefore the closest thing to a dress rehearsal:

| Bands | Pooled WAPE | 2024-11-21 fold |
|---|---:|---:|
| x1 (one global model) | 0.1618 | 0.1193 |
| **x2 (1-14, 15-30)** | **0.1581** | 0.1222 (+2.5%) |
| x3 (1-7, 8-21, 22-30) | 0.1574 | 0.1281 (+7.4%) |
| x4 (1-7, 8-14, 15-21, 22-30) | 0.1563 | 0.1318 (+10.5%) |

The damage to the seasonal analogue rises monotonically with the split, which
reads as thinner bands generalising worse on a low-season window - and Azar is a
low-season window. Two bands take 2.3 of the 3.4 percentage points of pooled
gain for a quarter of the risk. Four bands score best pooled and are one line
away (`ChampionSpec.bands`) for anyone who disagrees.

### What the ablation actually showed

Nine feature groups, added one at a time, LightGBM, same folds
(`artifacts/pol4/experiments.csv`):

| Stage | Added | WAPE | Delta |
|---|---|---:|---:|
| E2 | observed + horizon only | 0.3022 | — |
| E3 | + pickup windows | 0.2774 | -0.025 |
| E4 | + velocity / acceleration | 0.2739 | -0.004 |
| E5 | + activity | 0.2729 | -0.001 |
| E6 | + historical pickup curves | 0.2157 | **-0.057** |
| E7 | + calendar (incl. Jalali) | 0.1889 | **-0.027** |
| E8 | + city history | 0.1802 | -0.009 |
| E9 | + market signals | 0.1755 | -0.005 |
| E10 | + province signals | 0.1745 | -0.001 |

Two things this makes plain. A GBDT given only `observed` and `days_to_checkin`
scores **0.3022 — far worse than the arithmetic baseline it replaces**; the
model only earns its place once it is handed the pickup curve. And the largest
single jump is the curve itself, which is the Phase 1 result showing up again as
a feature.

### Rejected, and why

Recording what did not work matters as much as what did.

| Change | Result | Verdict |
|---|---|---|
| Global multiplicative calibration | WAPE 0.2200 → 0.2289 | **rejected** |
| Horizon-bucket calibration | 0.2200 → 0.2204 | **rejected** (neutral) |
| Shrunk horizon calibration | 0.2200 → 0.2206 | **rejected** |
| Scaling the *total* rather than the remainder | 0.2200 → 0.2426 | **rejected** |
| CatBoost (MAE) on the same features | 0.1835, and slower | **rejected** |
| Model / baseline ensemble | 0.1623 vs 0.1618 alone | **rejected** |
| Clustering | not run — no aggregation penalty to pay | **not needed** |

Calibration removes bias (-0.150 → -0.120) and does not improve WAPE. The
reason is that under a sum-of-absolute-errors metric on a heavy-tailed target
the optimal point forecast is nearer the conditional median than the mean, so
some negative bias is *correct*, not a defect. The fitted factors also disagree
across folds (0.82 to 1.15), which is the signature of a regime effect rather
than a fixed offset. WAPE is the metric, so the calibration was dropped.

The ensemble was searched leave-one-fold-out: the weight on the model came back
1.0 on four folds and 0.95 on the fifth. There is nothing for the baseline to
add once its curve is already a feature.

CatBoost was given a compute budget comparable to LightGBM's and lost on
accuracy anyway. At roughly ten times that budget (700 iterations, depth 8, MAE
loss) it had not finished five folds in half an hour, so it also loses on the
one criterion where a tie would have mattered.

`log1p` on the target, by contrast, was tested rather than assumed - the Phase 1
audit found the generic platform's automatic log1p under-predicting - and here
it **helped on every measure**: WAPE 0.1692 → 0.1618, top-1% WAPE 0.1681 →
0.1544, top-1% bias -0.119 → -0.100. The difference is the setup: an L1
objective in log space targets the conditional median, which is what WAPE wants,
whereas the platform's L2-on-log1p targeted a mean and then under-shot on
inverse transform.

### Forecast stability

The same (city, check-in) pair is predicted at D-30, D-21, D-14, D-7, D-3 and
D-1 - a forecast that swings is unusable even if it eventually lands. Measured
on a 30-day historical window (`artifacts/pol4/stability.parquet`):

| | |
|---|---:|
| Stability score (1 = never moves) | **0.844** |
| Mean relative revision between snapshots | 15.6% |
| Convergence rate (revision moves toward the truth) | 72.2% |

And it tightens as the check-in approaches, which is the behaviour you want:

| Step | D-30→21 | D-21→14 | D-14→7 | D-7→3 | D-3→1 |
|---|---:|---:|---:|---:|---:|
| Mean relative revision | 15.1% | 14.8% | 18.4% | 15.6% | 13.9% |
| Convergence rate | 66.3% | 68.0% | 72.0% | 74.7% | 79.9% |
| WAPE at that snapshot | 0.17 (D-30) | 0.14 (D-21) | 0.12 (D-14) | 0.08 (D-7) | 0.02 (D-1) |

Every snapshot is fitted at the *earliest* cutoff in the window, so a later one
is given less information than it would really have - conservative by design, so
no snapshot can see anything it should not.

### What the model actually leans on

Gain importance for the champion (`artifacts/pol4/feature_importance.json`):

| # | Feature | Share |
|---|---|---:|
| 1 | `city_hist_mean` | 25.9% |
| 2 | `city_weekday_mean` | 22.9% |
| 3 | `city_hist_p75` | 22.0% |
| 4 | `city_hist_p90` | 10.7% |
| 5 | `pickup_baseline_remaining` | 6.7% |
| 6 | `city_hist_median` | 2.3% |
| 7 | `city_code` | 2.2% |
| 8 | `city_volatility` | 1.5% |
| 9 | `pickup_baseline_prediction` | 0.8% |
| 10 | `jalali_month` | 0.6% |

**Read this against the ablation, not instead of it.** Gain importance says the
city-level statistics do most of the splitting - they set the *scale* of a
prediction, and scale is most of a tree's work on a target spanning six orders
of magnitude. But adding the city block only bought 0.009 WAPE, while the
pickup-curve block bought 0.057. The curve features are worth six times more at
the margin and rank fifth on gain, because nothing else in the feature set can
supply horizon shape. Importance is a diagnostic; the ablation is the measure.

### The hard fold

`2025-05-21` is the worst fold for every model (champion 0.2756). It is a demand
**regime shift**, not a modelling bug: target-window demand is 1.50x the
preceding 30 days, and only 79% of the demand the historical curve expected to
see by the cutoff had actually arrived. When a period accelerates, late pickup is
disproportionate, the curve overstates how complete the observation is, and the
projection under-shoots - concentrated at 22-30 days, where WAPE reaches 0.525.
This is the case the market and province features exist to catch.

### Leakage safety

A curve fitted at cutoff C reads only check-ins completed by C, and observed
demand reads only log dates up to C. `backend/tests/test_pol4_leakage.py`
asserts this mechanically: it overwrites every post-cutoff search with garbage,
re-runs the whole pipeline, and requires every prediction to be unchanged.

### What this deliberately does not do

No clustering (`cluster_code = city_code` - the brief penalises aggregation and
a global model with `city_code` as a categorical already shares information for
free), no deep learning, no frontend changes yet. Each would have to beat 0.1618
on these five folds to earn its place.

---

## Data contract

The only thing that changes when the real dataset arrives:

```yaml
dataset:
  path: data/raw/bookings.csv
  joins:                       # the other competition files
    - {path: data/raw/accommodations.csv, on: accommodation_id}
    - {path: data/raw/holidays.csv,       on: date}

schema:
  timestamp: date
  target: booking_count        # any KPI: demand, occupancy, revenue, ...
  entity_id: accommodation_id  # or null for a single series
  frequency: D                 # or null to auto-detect
  aggregation: sum             # "count" when demand is the number of rows
  calendar: auto               # auto | jalali | gregorian

hierarchy:
  destination: destination_id
  category: category

features:
  future:      [price, is_holiday, is_event]   # known for future dates
  historical:  [search_count, view_count]      # past only → used via lags
  static:      [capacity, rating, category]    # constant per entity

evaluation:
  primary_metric: wape
  horizons: [7, 14, 30, 60, 90]
```

### Data shapes it already handles

The dataset is expected to arrive the way an Iranian marketplace exports one.
Each of these fails silently rather than loudly if it is not handled, so each
is detected, converted, and *reported* — never applied behind your back.

| Shape | What happens |
|---|---|
| **Jalali dates** — `1403/05/12`, or `۱۴۰۳/۰۵/۱۲` with Persian digits | Detected from the year field, converted to Gregorian, raised as a validation finding. `schema.calendar` overrides the guess. |
| **A raw booking log** — one row per booking, no demand column | `aggregation: count` makes demand the row count. The profiler suggests it and disables the target selector, so a plausible-looking amount column cannot be summed by mistake. |
| **Several separate files** — demand, accommodation/destination, bookings | `dataset.joins`, or **فایل‌های جانبی** in the Data Lab. Match rates are reported per join; date keys are parsed on both sides, so a Jalali booking file joins a Gregorian holiday calendar correctly. |
| **Persian column headers** — `تاریخ_رزرو`, `کد_اقامتگاه`, `شهر` | Recognised alongside the English names, including `ی`/`ي`, `ک`/`ك` and zero-width non-joiner variants. |

These shapes are exercised by the data-adapter test suite.

See `config/data_contract.example.yaml` and **[docs/COMPETITION_DAY.md](docs/COMPETITION_DAY.md)**.

---

## API

The dashboard reads these, and nothing else. All read-only, all served from
generated artefacts - no endpoint fits a model or opens a source CSV.

```
GET  /api/v1/pol4/overview/                 KPIs, national series, top cities, provinces, momentum
GET  /api/v1/pol4/forecast/                 the national daily series alone
GET  /api/v1/pol4/heatmap/?top_n=           city × date matrix
GET  /api/v1/pol4/cities/                   every city with totals and names
GET  /api/v1/pol4/cities/{code}/            one city: forecast, peaks, history, momentum
GET  /api/v1/pol4/cities/{code}/pickup/     observed accumulation vs the historical curve
GET  /api/v1/pol4/stability/                D-30 → D-1 snapshots plus the aggregate
GET  /api/v1/pol4/model-performance/        champion vs baseline, folds, breakdowns, importance
GET  /api/v1/pol4/reports/                  the seven report kinds
GET  /api/v1/pol4/reports/{kind}/           a preview of one
GET  /api/v1/pol4/reports/{kind}.csv        the download
```

`{code}` accepts either the numeric `city_code` or the city name, so the UI
never has to know which it is holding. The competition identifier is never
replaced - names are a display layer.

The generic-platform endpoints (`/dashboard/`, `/forecasts/`, `/models/`,
`/datasets/`, `/training/`, `/insights/`) still exist and are still tested, but
the Pol 4 product no longer calls them.

---

## Offline mode

The venue's internet will be bad. After one online setup the system runs fully
offline:

```bash
pip install -r requirements-optional.txt
python scripts/download_models.py          # pre-fetch weights
export HF_HOME=models/hf_cache HF_HUB_OFFLINE=1
```

If nothing is downloaded, the system runs on LightGBM, CatBoost and the
baselines — which is the configuration the benchmark table above was produced
with.

**Verified status of the optional models** (run on a 4-core CPU, no GPU):

| Model | Status | Note |
|---|---|---|
| NHITS | trains and forecasts | 79% empirical coverage on an 80% interval |
| NBEATSx | trains and forecasts | joins the ensemble with a CV-derived weight |
| Chronos-2 | adapter verified against the real library | pretrained weights need HuggingFace access |
| Optuna | searches and improves | 40 trials in 26s took WAPE 0.163 → 0.159 |

---

## Failure handling

Every optional component fails soft. None of them can take the demo down.

| If this fails | What happens |
|---|---|
| Chronos cannot load | Tree models continue; a warning is recorded |
| No GPU | CPU, automatically detected |
| Ollama not running | Template narrator (deterministic Persian prose) |
| NHITS fails to train | Ensemble rebuilds without it |
| Redis not running | `SYNC_TASKS=true` runs training inline |
| A column is missing | Pipeline continues with the features that exist |
| SHAP unavailable | Falls back to split-gain importance |

---

## Testing

```bash
make test           # backend pytest + frontend typecheck/lint/build
make test-backend
make audit          # live API audit against a running backend
make e2e            # browser walkthrough with real clicks
```

**369 unit/integration tests** covering: leakage guarantees, the data adapter,
schema detection, validation, metrics, time-series splitting, conformal
calibration, baselines, GBDT models, the optional Chronos/NHITS adapters, the
registry, the full pipeline, reproducibility, hierarchy coherence, anomaly and
peak detection, JSON safety, path-traversal defences, the narrator's
anti-hallucination gate, and every API endpoint in both the empty and populated
states.

Three further layers run against a live system rather than fixtures:

| Layer | What it does | Command |
|---|---|---|
| API audit | 63 calls across every endpoint, including malformed input and edge cases | `make audit` |
| Semantic audit | 45 checks that the *numbers* are right - hierarchy coherence, KPI/series agreement, scenario direction, coverage vs. nominal | `make audit` |
| Browser E2E | real interactions: filters, entity selection, horizon switching | `make e2e` |

All three were run against a bare clone (no DB, no data, no runs) as well as a
seeded one.

The optional-model tests build a tiny Chronos model locally rather than
downloading weights, so they exercise the adapter against the real library in
any environment. Accuracy is not asserted there - the integration is.

---

## Configuration

See `.env.example`. The most important settings:

```env
DEFAULT_HORIZON=30
PRIMARY_METRIC=wape
TRAINING_PROFILE=demo
SYNC_TASKS=true         # no Redis/Celery required
ENABLE_CHRONOS=false
ENABLE_NEURALFORECAST=false
ENABLE_LOCAL_LLM=false
```

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| Dashboard shows "no model trained" | `make pol4` |
| `ModuleNotFoundError: ml` | Run through `backend/manage.py`, or set `PYTHONPATH=backend` |
| Frontend cannot reach the API | Set `NEXT_PUBLIC_API_BASE_URL` and restart `next dev` |
| CORS error | Add the origin to `CORS_ALLOWED_ORIGINS` |
| Training is slow | `PROFILE=demo`, or lower `max_train_rows` in `config/profiles.yaml` |
| `No trainable samples` | History is shorter than horizon + context; reduce `--horizon` |
| LightGBM import error on macOS | `brew install libomp` |
| Everything broken before the demo | `make pol4-baseline` - 15 seconds, no model fitting |

---

## What this project deliberately does not do

No Kubernetes, no microservices, no Kafka, no multi-tenancy, no enterprise
RBAC. It is a hackathon MVP. The time went into forecast accuracy, data
adaptability, explainability, and a dashboard that survives a live demo.

## Known limitations

- Aggregated prediction intervals are bottom-up sums of listing-level bounds,
  which assumes correlated errors and therefore **overstates** their width.
- Driver contributions describe what the model learned from observational
  data. They are associations, not causal effects, and the UI says so.
- Known-future covariates beyond the observed range are projected seasonally
  unless real planned values are supplied via `--future-covariates`.
- The champion forecasts **bookable** demand, not unconstrained market demand.
  Censoring is measured and reported (with an estimated uplift where supply
  binds), but the target itself is not silently uncensored - that would be an
  assumption dressed up as data.
