# Competition Day Runbook

The real dataset lands and the clock starts. This file exists so that nobody
has to think about architecture at that moment — only about the data.

**The core promise:** moving from synthetic to real data is a *configuration*
change, not a code change. Nothing in `backend/ml/` hard-codes a synthetic
column name, a target, a metric, or a hierarchy level.

---

## Timeline at a glance

| # | Step | Command | ~Time |
|---|------|---------|-------|
| 1 | Put the dataset in place | `cp <file> data/raw/` | 1 min |
| 2 | Profile it | `make profile FILE=data/raw/<file>` | 1 min |
| 3 | Inspect columns | open `/data-lab` | 5 min |
| 4 | Map the schema | Data Lab form, or edit the contract | 5 min |
| 5 | Pick the target | Data Lab → *متغیر هدف* | 1 min |
| 6 | Set the frequency | auto-detected; override if wrong | 1 min |
| 7 | Set the hierarchy | Data Lab → destination / category | 1 min |
| 8 | Set the official metric | Data Lab → *معیار رسمی مسابقه* | 1 min |
| 8b | Awkward shape? | Jalali dates / booking log / several files — see below | 5 min |
| 9 | Check for leakage | `make validate` → read the report | 3 min |
| 10 | Run the baselines | `make train PROFILE=demo` | ~2 min |
| 11 | Run the real profile | `make train PROFILE=competition` | 10–40 min |
| 12 | Compare CV results | `/models` page + `reports/model_report.md` | 5 min |
| 13 | Tune (optional) | `PROFILE=full` | 20–60 min |
| 14 | Build the ensemble | automatic — weights come from CV | — |
| 15 | Final forecast | automatic at the end of training | — |
| 16 | Verify the dashboard | open `/dashboard` | 5 min |

---

## 1. Put the dataset in place

```bash
cp ~/Downloads/competition_data.csv data/raw/
```

CSV, TSV, Parquet, JSON and Excel are all accepted. Large files are fine —
Parquet is preferred if the organisers offer it.

## 2. Profile it

```bash
make profile FILE=data/raw/competition_data.csv
```

This prints, for every column: dtype, null %, unique count, min/max, and the
system's guess at its role. It also reports the detected frequency and whether
timestamps are regularly spaced.

## 3–8. Map the schema

**The fast path** is the UI: open <http://localhost:3000/data-lab>, upload or
select the file, and the form arrives pre-filled with the profiler's
suggestions. Correct anything that's wrong, then press **ذخیره نگاشت**. That
writes `config/data_contract.active.yaml`, which every subsequent command uses.

**The file path** — if you would rather edit YAML directly:

```bash
cp config/data_contract.example.yaml config/data_contract.active.yaml
$EDITOR config/data_contract.active.yaml
```

The fields that matter:

```yaml
dataset:
  path: data/raw/competition_data.csv
  joins:                              # other files, merged on a shared key
    - path: data/raw/accommodations.csv
      on: accommodation_id
    - path: data/raw/holidays.csv
      on: date

schema:
  timestamp: <the date column>
  target: <THE OFFICIAL TARGET>       # booking_count | demand | occupancy | ...
  entity_id: <listing/hotel id, or null for a single series>
  frequency: D                        # null => auto-detect
  aggregation: sum                    # sum | mean | max | min | first | count
  calendar: auto                      # auto | jalali | gregorian

hierarchy:
  destination: <city/region column>   # optional
  category: <property type column>    # optional

features:
  future:      [price, is_holiday]    # KNOWN for future dates
  historical:  [search_count]         # only known for the past
  static:      [capacity, category]   # constant per entity
  ignored:     [internal_notes]

evaluation:
  primary_metric: wape                # THE OFFICIAL METRIC
  horizons: [7, 14, 30, 60, 90]
```

### The one decision that matters most

`features.future` versus `features.historical`.

- **future** — you will genuinely know this value for the days you are
  forecasting. Calendar flags, published event dates, planned prices,
  contracted capacity.
- **historical** — you only know it for the past. Search volume, views,
  clicks, competitor prices. The pipeline uses these strictly through lags and
  rolling windows computed *at the forecast origin*.

Putting a historical column in `future` is the single easiest way to build a
model that scores brilliantly in validation and fails on the leaderboard. When
unsure, put it in `historical` — you lose a little accuracy and keep your
integrity.

---

## When the data arrives in an awkward shape

Four shapes are common in Iranian marketplace exports and each one silently
destroys a run rather than failing loudly. All four are handled, and the
profiler detects all four — but check that it got them right.

### Jalali (Shamsi) dates

`1403/05/12`, sometimes written with Persian digits (`۱۴۰۳/۰۵/۱۲`).
`pandas.to_datetime` returns NaT for every one of those, which would drop the
entire dataset at the first step.

The profiler reads the year field: 1300–1500 is Jalali, because no Gregorian
dataset contains year 1403. The Data Lab shows a **تقویم شمسی** badge and a
note naming the sample values it converted, and the validation report raises a
`calendar_converted` finding so the conversion cannot be scrolled past.

If the guess is wrong, set it explicitly — `schema.calendar: gregorian`, or the
**تقویم ستون تاریخ** selector in the Data Lab. Everything downstream works in
Gregorian; the calendar is an input format, never a modelling concern.

### A raw booking log with no demand column

If the file has one row per booking — a booking id, a date, an accommodation
code, an amount — there is nothing to sum. Demand *is* the number of rows.

Set `schema.aggregation: count` and leave `schema.target` empty. The Data Lab
suggests this automatically when it sees many rows repeating an
`(entity, date)` pair, shows a **جدول خام رزرو** badge, and disables the target
selector so a plausible-looking numeric column (`مبلغ`, `تعداد_شب`) cannot be
summed by mistake — that would be measuring rials or nights, not demand.

Verify afterwards that the panel total equals the source row count; the adapter
reports both in its notes.

### Several files instead of one

The brief splits the data three ways: demand, accommodation/destination
information, and bookings. Nothing has to be pre-joined outside the product.

In the Data Lab, **فایل‌های جانبی → افزودن فایل جانبی** uploads each extra file
and picks the key column; only columns present in both files are offered. In
YAML it is `dataset.joins`, as above.

The adapter reports the match rate for every join, and says so loudly when a
join matched nothing — the failure mode is a key that exists on both sides with
values that never line up. Date keys are parsed on both sides before matching,
so a Jalali booking file joins correctly to a Gregorian holiday calendar. A
side table with duplicate keys is collapsed rather than allowed to multiply the
main frame's rows.

### Persian column headers

`تاریخ_رزرو`, `کد_اقامتگاه`, `شهر`, `تعداد_رزرو` are recognised alongside their
English equivalents, including the `ی`/`ي` and `ک`/`ك` variants and zero-width
non-joiners. Confirm the suggestions in the Data Lab as you would for any
dataset — they are suggestions, not decisions.

---

## 9. Check for leakage

```bash
make validate
```

Read the **Potential Leakage Warnings** section. A column flagged there
correlates ≥0.90 with the target. That is a *suspicion*, not a verdict:

- Flagged **and** declared as `future` → investigate immediately. This is the
  dangerous case.
- Flagged but declared as `historical` → usually benign; it is only read
  through lags.

Also check the **Data Health Score**. Below 60, fix the data before training.

## 10. Run the baselines first

```bash
make train PROFILE=demo HORIZON=90
```

Two minutes. This tells you:
- whether the pipeline can read the data at all,
- what a seasonal naive scores — **your floor**,
- roughly what a real model will add.

If LightGBM does not clearly beat the seasonal naive here, something is wrong
with the mapping. Stop and re-check before spending an hour on a bigger run.

## 11. Run the real profile

```bash
make train PROFILE=competition HORIZON=90 METRIC=wape
```

Or with everything explicit:

```bash
python backend/manage.py train_forecast \
    --dataset data/raw/competition_data.csv \
    --target booking_count \
    --timestamp date \
    --entity accommodation_id \
    --horizon 90 \
    --profile competition \
    --metric wape \
    --future-covariates data/raw/known_future.csv \
    --save-contract
```

`--future-covariates` is how you supply genuinely-known future values (a
holiday calendar, published event dates, planned prices). The file needs a
date column, optionally an `entity_id` column, plus the covariate columns.

## 12. Compare

```bash
make report          # reports/model_report.md
```

Open `/models` for the leaderboard and `/backtesting` for error by horizon,
destination, weekday and month. Confirm:

- the champion beats the best baseline (the report states the improvement),
- error does not explode at long horizons,
- observed interval coverage is close to the nominal 80%.

## 13. Tune (only if time allows)

```bash
make train PROFILE=full HORIZON=90
```

Enables more trees, more folds, and an Optuna search over LightGBM/CatBoost
hyper-parameters. The search:

- validates on a held-out **time** window, like everything else,
- is hard-capped by `tuning.max_minutes` in `config/profiles.yaml`,
- keeps the profile defaults unless it actually beats them,
- reports every outcome in `reports/model_report.md` and on `/models`.

Do **not** start this unless you have a working `competition` run already
saved — a completed good run beats an unfinished perfect one.

### If the target can hit a ceiling

If the dataset has a capacity/availability column, declare it:

```yaml
target_options:
  censoring_column: available_capacity
  censoring_enabled: true
```

The pipeline then measures how much of the history sits at capacity and
reports it. This matters for interpretation: a destination that looks flat may
be flat only because it is full, and the forecast describes bookable demand
rather than unconstrained demand.

## 14–15. Ensemble and final forecast

Both are automatic. Ensemble weights come from the rolling-CV scores on the
official metric; the ensemble only becomes champion if it actually wins. The
final forecast is produced by the champion refitted on the full history.

## 16. Verify the dashboard

```bash
make dev
```

Open <http://localhost:3000/dashboard> and walk the demo story:

1. KPI row shows the forecast total, the change, and the model's accuracy.
2. The main chart shows history, backtest and future with the P10–P90 band.
3. Drivers card explains *why*.
4. Peaks and anomalies show *when* and *where*.
5. `/scenarios` shows *what if*.
6. `/backtesting` shows *how do we know*.

Add `?presentation=true` for the judging session: bigger numbers, less chrome.

---

## Verifying a fresh machine before the demo

```bash
make test            # 216 tests
make dev             # start both services
make audit           # 63 API calls + 45 semantic checks against the live API
make e2e             # 46 browser interactions, including upload -> train
```

`make audit` is the fastest way to tell whether a machine is demo-ready: it
checks that levels add up, that KPIs agree with the underlying series, that a
price cut moves demand the right way, and that interval coverage is near
nominal. If those pass, the dashboard will behave.

## Emergencies

| Symptom | Fix |
|---|---|
| `Contract references column(s) [...] not in the dataset` | The mapping names a column that does not exist. Re-run the profiler and fix the contract. |
| `No trainable samples` | History is shorter than `horizon + context`. Lower `--horizon`. |
| Every model scores identically | The target is probably constant, or the entity column is unique per row. Check the validator output. |
| Training is too slow | `PROFILE=demo`, lower `max_train_rows` in `config/profiles.yaml`, or reduce `--horizon`. |
| Chronos / NHITS fails to load | Expected without the optional deps. The run continues and logs a warning. |
| Ollama is not running | Expected. The template narrator takes over automatically. |
| Redis is not running | Expected. `SYNC_TASKS=true` runs training inline. |
| Dashboard says "no data" | No successful run yet. Check `/api/v1/training/` for the failure reason. |
| Everything is broken 10 minutes before the demo | `DEMO_MODE=true` serves the committed artefacts in `data/demo_artifacts/`. Every page works with zero training, including the scenario simulator - the run ships its own model and panel. |

## Metric cheat-sheet

| Official metric says | Set `primary_metric` |
|---|---|
| WAPE, weighted MAPE, total error / total actual | `wape` |
| MAE, mean absolute error | `mae` |
| RMSE | `rmse` |
| MAPE | `mape` |
| SMAPE | `smape` |
| RMSLE / log error | `rmsle` |
| R², explained variance | `r2` |
| Poisson deviance | `poisson_deviance` |

Changing it re-orients the leaderboard, champion selection, ensemble weights
and every reported score. Nothing else needs to change.
