# Novo Pulse — AI Demand Forecasting

**پل چهارم | هوش پیش‌بینی تقاضا**
پیش‌بینی، تحلیل و شبیه‌سازی تقاضای بازار اقامت و گردشگری

A production-shaped demand-forecasting platform for the hospitality market:
multi-model, multi-horizon, probabilistic, explainable, and built so the real
competition dataset can replace the synthetic one through configuration alone.

---

## The five questions

Everything in this repository exists to answer five questions, in order:

| Question | Answer | Where |
|---|---|---|
| What will happen? | Multi-horizon forecast (7 / 14 / 30 / 60 / 90 days) | `/dashboard` |
| How confident are we? | P10–P90 prediction interval with **measured** coverage | `/dashboard`, `/backtesting` |
| Why will it happen? | SHAP driver attribution, grouped and signed | `/dashboard` |
| What if conditions change? | What-if simulation on the fitted model | `/scenarios` |
| How do we know it is good? | Rolling-origin backtest vs. statistical baselines | `/backtesting`, `/models` |

**Predict → Explain → Detect → Simulate → Decide.**

---

## Quick start

```bash
git clone <repo> && cd novo-pulse

make demo        # venv + deps + synthetic data + trained demo model (~4 min)
make dev         # backend :8000, frontend :3000
```

Open <http://localhost:3000>. Add `?presentation=true` for the judging view.

With Docker:

```bash
docker compose up --build
docker compose exec backend python backend/manage.py seed_demo --publish
```

Manual setup:

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python backend/manage.py migrate
.venv/bin/python backend/manage.py seed_demo --publish
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
    core/            health, system info, seed_demo
    datasets/        Data Lab: upload, profile, map, validate
    forecasting/     dashboard, forecasts, anomalies, backtests, models
    experiments/     training runs, leaderboard, model report
    insights/        structured business insights
    scenarios/       what-if simulation
  ml/
    contract.py      the data contract - the seam to any dataset
    data/            adapter, profiler, validator, frequency, synthetic
    features/        tensor, engineering, rolling, calendar
    models/          base, registry, baselines, gbdt, chronos, neural, ensemble
    evaluation/      metrics, splitters, backtest, uncertainty
    explainability/  SHAP
    anomaly/         residual + forecast anomalies, peak detection
    insights/        insight engine, narrator
    pipelines/       the end-to-end training pipeline
  tests/             128 tests
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

## Demo mode

`DEMO_MODE=true` (the default) serves the synthetic dataset and the
precomputed artefacts in `data/demo_artifacts/`, so the dashboard works
immediately — even if training was never run on this machine.

`DEMO_MODE=false` makes a hard promise: **no fabricated numbers**. With no
trained model, every endpoint returns `available: false` and the UI shows an
explicit empty state instead of a plausible-looking figure.

---

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

Example run (synthetic dataset, `demo` profile, 30-day horizon, listing level):

| Model | WAPE | vs. baseline |
|---|---:|---:|
| **LightGBM (champion)** | **25.4%** | **+19.7%** |
| Ensemble | 25.4% | +19.6% |
| CatBoost | 26.1% | +17.7% |
| Moving average *(best baseline)* | 31.6% | — |
| Seasonal naive (7) | 38.0% | −20.2% |

Observed 80% interval coverage: 77–79% across every horizon bucket.

`reports/model_report.md` is regenerated automatically after each run.

---

## Data contract

The only thing that changes when the real dataset arrives:

```yaml
schema:
  timestamp: date
  target: booking_count        # any KPI: demand, occupancy, revenue, ...
  entity_id: accommodation_id  # or null for a single series
  frequency: D                 # or null to auto-detect

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

See `config/data_contract.example.yaml` and **[docs/COMPETITION_DAY.md](docs/COMPETITION_DAY.md)**.

---

## API

```
GET  /api/v1/health/                     liveness + model availability
GET  /api/v1/system/                     hardware, model registry, profiles

GET  /api/v1/dashboard/summary/          KPI block
GET  /api/v1/forecasts/                  raw forecast rows
GET  /api/v1/forecasts/timeseries/       history + backtest + forecast + band
GET  /api/v1/forecasts/drivers/          SHAP driver groups
GET  /api/v1/forecasts/peaks/            upcoming peaks and troughs
GET  /api/v1/forecasts/overview/         per-entity table
GET  /api/v1/forecasts/heatmap/          entity × date matrix
GET  /api/v1/forecasts/narrative/        grounded natural-language summary

GET  /api/v1/anomalies/
GET  /api/v1/models/                     registry + trained metadata
GET  /api/v1/models/leaderboard/
GET  /api/v1/backtests/
GET  /api/v1/backtests/metrics/
GET  /api/v1/insights/

POST /api/v1/scenarios/simulate/
GET  /api/v1/scenarios/options/

POST /api/v1/datasets/upload/
POST /api/v1/datasets/profile/
POST /api/v1/datasets/map/
POST /api/v1/datasets/validate/

POST /api/v1/training/run/
GET  /api/v1/training/{run_id}/
```

Filtering:

```
/api/v1/forecasts/timeseries/?level=destination&id=kish&horizon=30
```

`level` ∈ `listing | destination | category | market`.

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
| Training never ran | `DEMO_MODE` serves precomputed artefacts |

---

## Testing

```bash
make test           # backend pytest + frontend typecheck/lint/build
make test-backend
make audit          # live API audit against a running backend
make e2e            # browser walkthrough with real clicks
```

**216 unit/integration tests** covering: leakage guarantees, the data adapter,
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
| Browser E2E | 46 real interactions: filters, entity selection, horizon switching, running a scenario, uploading a CSV and training from the UI | `make e2e` |

The optional-model tests build a tiny Chronos model locally rather than
downloading weights, so they exercise the adapter against the real library in
any environment. Accuracy is not asserted there - the integration is.

---

## Configuration

See `.env.example`. The most important settings:

```env
DEMO_MODE=true          # false ⇒ never show a number we did not compute
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
| Dashboard shows "no model trained" | `make seed` |
| `ModuleNotFoundError: ml` | Run through `backend/manage.py`, or set `PYTHONPATH=backend` |
| Frontend cannot reach the API | Set `NEXT_PUBLIC_API_BASE_URL` and restart `next dev` |
| CORS error | Add the origin to `CORS_ALLOWED_ORIGINS` |
| Training is slow | `PROFILE=demo`, or lower `max_train_rows` in `config/profiles.yaml` |
| `No trainable samples` | History is shorter than horizon + context; reduce `--horizon` |
| LightGBM import error on macOS | `brew install libomp` |
| Everything broken before the demo | `DEMO_MODE=true` serves `data/demo_artifacts/` |

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
