# POL 4 — Where every number on the dashboard comes from

**Rule:** no figure appears in the product unless this table can name the file it
was computed from. There is no mock data, no demo mode, no placeholder metric
and no `Math.random` anywhere in the frontend.

The chain is always the same shape:

```
competition CSV  ->  offline pipeline  ->  artefact  ->  /api/v1/pol4/*  ->  component
   (data/raw/pol4)   python -m ml.pol4.pipeline   (artifacts/pol4)
```

The dashboard never opens `search_data.csv`. Its 3.3 million rows are read once,
offline, by the pipeline; every page load reads a pre-aggregated artefact.

---

## 1. Source data

| File | Rows | Role |
|---|---:|---|
| `data/raw/pol4/search_data.csv` | 3,298,564 | historical search events (both clocks) |
| `data/raw/pol4/evaluation.csv` | 65,411 | partial Azar observations at the cutoff |
| `data/raw/pol4/cities.csv` | 321 | city coordinates and province code |
| `data/raw/pol4/city_code_mapping.csv` | 321 | city and province display names |

## 2. Generated artefacts

| Artefact | Written by | Contains |
|---|---|---|
| `results.csv` | `pipeline.run` | the submission: `cluster_code, checkin, predicted_demand` |
| `results_named.csv` | `submission.build_named_submission` | same rows plus names, observed, remaining |
| `backtest_metrics_phase2.json` | `pipeline._write_phase2_backtest` | champion vs baseline, all folds and breakdowns |
| `experiment_summary.json` | `pipeline._write_experiments` | every experiment, including the rejected ones |
| `feature_importance.json` | `champion.importance` | gain importance, pooled across horizon bands |
| `stability.parquet` | `stability.analyse` | D-30 → D-1 snapshots for a historical window |
| `pickup_curves.parquet` | `pickup.PickupCurves.to_frame` | completion curves (city / province / global) |
| `target_pickup.parquet` | `analytics.target_pickup` | observed cumulative by lead time, Azar pairs |
| `city_momentum.parquet` | `analytics.city_momentum` | recent pickup vs each city's own expectation |
| `city_history.parquet` | `analytics.city_history` | 180 days of completed demand per city |
| `province_summary.json` | `analytics.province_summary` | forecast rolled up to the seven provinces |
| `run_summary.json` | `pipeline.run` | data validation, champion config, stability summary |

---

## 3. Screen 1 — Overview (`/overview`)

| What the judge sees | Component | Endpoint | Artefact | How it is computed |
|---|---|---|---|---|
| Total predicted demand | `Kpi` | `/pol4/overview/` | `results_named.csv` | `sum(predicted_demand)` over 9,630 rows |
| Peak check-in date | `Kpi` | `/pol4/overview/` | `results_named.csv` | `argmax` of the daily national total |
| Highest-demand city | `Kpi` | `/pol4/overview/` | `results_named.csv` | `argmax` of the per-city 30-day total |
| Fastest-pickup city | `Kpi` | `/pol4/overview/` | `city_momentum.parquet` | `recent_pickup / expected_pickup`, top-quartile cities only |
| Backtest WAPE | `Kpi` | `/pol4/overview/` | `backtest_metrics_phase2.json` | `champion.pooled.wape` |
| Stability score | `Kpi` | `/pol4/overview/` | `run_summary.json` | `stability.stability_score` |
| National 30-day forecast | `ComposedChart` | `/pol4/overview/` | `results_named.csv` | grouped by check-in; observed and remaining stacked |
| City × date heatmap | `Heatmap` | `/pol4/heatmap/` | `results_named.csv` | pivot of top-N cities by predicted demand |
| Top cities | `BarChart` | `/pol4/overview/` | `results_named.csv` | per-city totals, observed + remaining stacked |
| Province distribution | `BarChart` | `/pol4/overview/` | `province_summary.json` | rounded predictions grouped by province |
| Emerging demand table | table | `/pol4/overview/` | `city_momentum.parquet` | 7-day pickup vs the city's own completion curve |

## 4. Screen 2 — City Analysis (`/city`)

| What the judge sees | Endpoint | Artefact | How it is computed |
|---|---|---|---|
| City selector (names) | `/pol4/cities/` | `results_named.csv` + `city_momentum.parquet` | per-city totals, searchable by name or province |
| 30-day city forecast | `/pol4/cities/{code}/` | `results_named.csv` | the city's 30 rows, observed and remaining stacked |
| Observed / remaining / final | `/pol4/cities/{code}/` | `results_named.csv` | the three columns the submission already carries |
| Pickup curve — observed | `/pol4/cities/{code}/pickup/` | `target_pickup.parquet` | reverse-cumulative `search_count` by lead time |
| Pickup curve — expected | `/pol4/cities/{code}/pickup/` | `pickup_curves.parquet` | this city's completion fraction × predicted final |
| Peak nights | `/pol4/cities/{code}/` | `results_named.csv` | top 5 by predicted demand |
| Historical demand | `/pol4/cities/{code}/` | `city_history.parquet` | completed daily demand, 90 nights before the cutoff |

## 5. Screen 3 — Forecast Stability (`/stability`)

| What the judge sees | Endpoint | Artefact | How it is computed |
|---|---|---|---|
| Prediction ladder D-30 → D-1 | `/pol4/stability/` | `stability.parquet` | the champion re-applied at six lead times |
| Realised actual | `/pol4/stability/` | `stability.parquet` | complete historical demand for that pair |
| Stability score | `/pol4/stability/` | `run_summary.json` | `1 − mean relative revision`, clipped to [0,1] |
| Convergence rate | `/pol4/stability/` | `run_summary.json` | share of revisions that move toward the truth |
| Revision by step | `/pol4/stability/` | `run_summary.json` | `stability.by_step` |
| Accuracy by lead time | `/pol4/stability/` | `run_summary.json` | `stability.by_horizon` |

Every snapshot is produced by a model fitted at the *earliest* cutoff in the
window, so a later snapshot is given less information than it would really have.
Conservative by construction.

## 6. Screen 4 — Reports & model performance (`/reports`)

| What the judge sees | Endpoint | Artefact |
|---|---|---|
| Champion / baseline / improvement | `/pol4/model-performance/` | `backtest_metrics_phase2.json`, `run_summary.json` |
| WAPE per fold (hardest highlighted) | `/pol4/model-performance/` | `backtest_metrics_phase2.json` → `folds` |
| WAPE by horizon bucket | `/pol4/model-performance/` | `backtest_metrics_phase2.json` → `by_horizon_bucket` |
| Feature importance | `/pol4/model-performance/` | `feature_importance.json` |
| CSV exports (7 kinds) | `/pol4/reports/{kind}.csv` | the same artefacts the charts read |

Filters narrow rows; they never change how a number is computed, so a downloaded
report and the chart above it cannot disagree.

---

## 7. What is deliberately **not** claimed

| Not shown | Why |
|---|---|
| "customer intent" | Intent is latent. The product says *search-based demand as an observable proxy for travel intent*, which is what the competition measures. |
| Causal driver statements | Feature importance is labelled as what the model splits on, not what causes demand. |
| Bookable demand, capacity, censoring | No capacity or booking data exists in this dataset. |
| Price effects | No price column exists. |
| Property-level forecasting | No property data exists. |
| Scenario simulation | Requires future covariates the dataset does not contain. |

## 8. Identifier discipline

`city_code` is the competition identifier and is never replaced. It is the
primary key in every artefact, every API path and every React key. Names come
from `city_code_mapping.csv` and are a display layer only.

`results.csv` — the scored file — carries the numeric `cluster_code` exactly as
the brief specifies. `results_named.csv` is the human-readable companion.
