# Novo Pulse / Pol 4 — Independent Jury Audit

**Audit date:** 2026-09-10 · **Mode:** read-only · **Repo HEAD:** `0acd007` · working tree clean
**Nothing was fixed, patched, retrained, regenerated or overwritten.** The only file created is this one.

Every number below was recomputed independently from code, raw data or artifacts. No claim from
`README.md` or `docs/*.md` was accepted on trust. Commands are in the Evidence Appendix (§18).

---

## خلاصه اجرایی (Executive verdict — Persian)

پروژه از نظر **صحت روش‌شناسی** قوی است: قرارداد داده کاملاً سالم است، provenance با hash تأیید شد،
تعریف WAPE درست است، و leakage نسبت به cutoff مسابقه به‌صورت ساختاری بسته شده و با تست اثبات می‌شود.
تصمیم «بدون clustering» با شواهد کمّی گرفته شده، نه با ادعا — و این نقطه قوت اصلی پروژه است.

اما سه مشکل جدی وجود دارد که باید قبل از ارائه دیده شود:

1. **`scipy` در `requirements.txt` نیست** ولی `clustering.py` به آن وابسته است → روی clone تمیز،
   `make pol4-cluster` اجرا نمی‌شود. (Critical)
2. **fold `2025-05-21` با جنگ ۱۲روزه (۱۳ تا ۲۴ ژوئن ۲۰۲۵) هم‌پوشانی دارد** و به‌تنهایی **۴۶٪ کل خطای
   backtest** را تولید می‌کند. عدد سرتیتر ۰٫۱۴۷۹ بدون این fold به ۰٫۱۰۶۲ می‌رسد. این یک شوک
   غیرقابل‌پیش‌بینی است، نه ضعف مدل — ولی الان در هیچ مستندی توضیح داده نشده. (High)
3. **آرم انتخاب‌شده (`results_selected.csv`) از panel ای با ۱۴۹ گروه ساخته شده که هیچ‌جا ذخیره نشده.**
   trainset و model bundle موجود روی دیسک متعلق به آرم دیگری (۱۰۹ گروه) هستند. (High)

تصمیم نهایی: **READY WITH CONDITIONS**.

---

## §1 — Repository & Reproducibility · **PASS WITH LIMITATIONS**

| Check | Result | Evidence |
|---|---|---|
| Working tree clean, no uncommitted state | ✅ | `git status --short` → empty |
| Makefile targets exist and match docs | ✅ | `pol4`, `pol4-baseline`, `pol4-ablation`, `pol4-submission`, `pol4-cluster`, `pol4-cluster-quick` |
| Test suite green | ✅ | 423 passed, 9 skipped (`pytest backend/tests`) |
| Leakage suite green | ✅ | 17 passed (`test_pol4_leakage.py`, `test_leakage.py`, `test_pol4_validation.py`) |
| Dependencies pinned | ❌ | **0 lines contain `==`** in `requirements.txt` (34 lines, all `>=`) |
| All imports declared | ❌ | **`scipy` undeclared** |
| Artifacts self-describing | ⚠️ | absolute paths from another machine embedded |

### F-1.1 — `scipy` is an undeclared dependency · **CRITICAL**

**Fact.** `backend/ml/pol4/clustering.py:265` and `:329` import `scipy.cluster.hierarchy`
(`linkage`, `fcluster`). `grep -in scipy requirements.txt requirements-optional.txt` returns nothing.

**Risk.** On a clean clone with `pip install -r requirements.txt`, `make pol4-cluster` raises
`ModuleNotFoundError` at the first dendrogram fit. Model B is **not reproducible by a judge** as shipped.
The existing test suite does not catch it because the audit environment already has scipy (pulled in
transitively by `scikit-learn`, which is itself only in the optional file).

**Recommendation.** Add `scipy>=1.11` to `requirements.txt`. Do not rely on transitive installation.

### F-1.2 — Zero version pinning · **HIGH**

**Fact.** `lightgbm>=4.3`, `pandas>=2.2`, `numpy>=1.26,<3` — no exact pins anywhere.

**Risk.** LightGBM minor releases change histogram binning and tie-breaking. The project's strongest
reproducibility claim (this audit confirmed a teammate's Fedora run reproduced macOS WAPE to 5 decimals
at all 21 level×fold points) holds *for one pair of environments*, not in general. A judge on
lightgbm 4.9 may get different trees.

**Recommendation.** Ship a `requirements.lock` (`pip freeze`) alongside the range file.

### F-1.3 — Absolute foreign paths in committed artifacts · **LOW**

**Fact.** `artifacts/pol4/input_manifest.json → raw_dir` = `/home/parsa/Desktop/workspace/platforms/novo-pulse/data/raw/pol4`.
`artifacts/pol4/run_summary.json → champion.trainset.path` points at the same machine.

**Inference.** The committed champion artifacts were produced on a teammate's machine, not this clone.
Harmless for scoring; it does mean paths in artifacts cannot be followed by a judge.

### F-1.4 — Claims vs code: capabilities NOT used by the Pol 4 pipeline · **MEDIUM**

**Fact.** The entire `backend/ml/pol4/` package imports exactly three things from the generic platform:
`..paths`, `..evaluation.metrics`, `..data.dates`.

```
grep -rn "shap_explainer|anomaly|neural_model|chronos_model|models.tuning|models.registry|models.ensemble" backend/ml/pol4/*.py
→ NONE
```

**So the following exist in the repo but are NOT part of the Pol 4 forecast:** SHAP explainability,
uncertainty/conformal intervals, the anomaly detector, the neural model, Chronos, Optuna tuning, the
model registry, the ensemble module. `backend/ml/data/holidays.py` exists and is **not imported** by
Pol 4 either.

**Risk.** Presenting these as part of the competition solution would be an unsupported claim. See §14.10.

### Runtime / footprint (measured)

| | |
|---|---|
| `make pol4-cluster` end-to-end | 956 s (teammate, Fedora) / 1424 s (this machine) |
| Champion single-fold trainset build | 2.6 s; one 600-tree band fit on 1.62M rows: 35 s |
| Peak working set | training frame ≈ 2.65M × 66 float64 ≈ **1.4 GB**; ≥ 8 GB RAM advised |
| Tracked artifact bytes in git | **80.6 MB** |
| On-disk artifacts | `pol4` 51 MB · `pol4_cluster` 97 MB · `pol4_nolimit` 285 MB |

---

## §2 — Raw Data · **PASS**

All four files verified independently. Every contract invariant the brief implies holds.

| Check | search_data.csv | evaluation.csv | cities.csv | city_code_mapping.csv |
|---|---|---|---|---|
| physical lines | 3,298,565 | 65,412 | 322 | 322 |
| data rows (header excluded) | **3,298,564** | **65,411** | **321** | **321** |
| trailing newline | yes | yes | yes | yes |
| nulls | 0 | 0 | 0 | 0 |
| duplicate `(log_date, city, checkin)` | 0 | 0 | n/a | n/a |
| `search_count` min / max | 1 / 133,758 | 1 / 3,827 | — | — |
| negative or zero counts | 0 | 0 | — | — |
| lead time min / max | 0 / 59 | 1 / 59 | — | — |
| `log_date > checkin` | 0 | 0 | — | — |
| distinct cities | 321 | **239** | 321 | 321 |
| Σ `search_count` | **284,130,228** | **2,686,508** | — | — |

* **Coverage.** 0 city codes in the logs are missing from `cities.csv`; 0 cities lack search history.
  `city_code_mapping.csv` covers all 321 uniquely, with **0 province disagreements** against `cities.csv`.
* **Split integrity.** `search.checkin.max() = 2025-11-21`, `evaluation.checkin.min() = 2025-11-22` —
  no overlap. **0 rows share a `(log_date, city, checkin)` key across the two files.**
* **Cutoff.** `evaluation.log_date.max() = 2025-11-21`; **0 rows exceed the cutoff.**
* **Geography.** lat ∈ [25.64, 38.39], long ∈ [45.84, 61.15]; 0 null, 0 zero, 0 outside Iran's bounds.
* **Provenance.** Independently computed SHA-256 of all four files **matches `input_manifest.json` byte for byte**
  (e.g. `search_data.csv` = `101d310d…b44b1a`). A silent input change would alter `input_digest`
  `fc71f2f1…41636e` and the `--reuse-trainset` guard would refuse the run. **PASS.**

### F-2.1 — The dataset is 7 provinces, not national · **MEDIUM**

**Fact.** `province_code` has exactly **7** distinct values:
mazandaran (62 cities), khorasan_razavi (41), isfahan (78), hormozgan (25), gilan (53), alborz (18), tehran (44).
Iran has 31 provinces.

**Risk.** `docs/POL4_CLUSTERING.md:70`, `backend/ml/pol4/config.py:119`, `backend/ml/pol4/aggregate.py:28`
and `README.md` all use the word **"national"**. This is factually wrong: `market_*` features are a
**7-province total**, not a national one. Saying "national demand" to a jury is an overclaim.

**Recommendation.** Replace "national" with "in-panel" or "7-province" throughout. Do not restate
`market_*` as a country-level signal.

### Distribution (heavy tail confirmed)

City totals: min 2 · p25 429 · median 16,254 · p75 167,505 · max 30,084,842.
Top 1 city = 10.6 % of all demand; top 10 = 62.4 %; top 50 = 94.4 %.
Observed `(city, checkin)` pairs: 163,436 of 241,392 possible → **77,956 structural zeros (32.3 %)**.

### Why 3,298,564 ≠ 2,655,312 ≠ 901,648 (explicitly requested)

These three numbers count **three different objects**. They are not expected to match.

| Number | What it counts | Formula |
|---:|---|---|
| **3,298,564** | raw **event** rows — one per `(log_date, city, checkin)` where ≥1 search occurred. Sparse; absent rows mean "no search that day". | given by the file |
| **2,655,312** | **supervised city rows** — a *dense Cartesian* grid, one per `(city, check-in, horizon)`. Includes pairs with zero demand, because a zero is a real observation. | 321 × 752 × 11 |
| **901,648** | **supervised cluster rows** — same grid after 321 cities collapse to 109 virtual cities. | 109 × 752 × 11 |

Three transformations separate them, in order:

1. **Sparse → dense.** 3.30M event rows collapse to 163,436 distinct `(city, checkin)` pairs, then the
   grid is *completed* to 321 × 752 = 241,392 pairs (adding 77,956 structural zeros).
2. **× horizons.** Each pair becomes 11 training rows, one per sampled horizon
   (`config.train_horizons = (1,2,3,5,7,10,14,18,21,25,30)`) → 241,392 × 11 = **2,655,312**.
3. **× aggregation.** Clustering replaces 321 entities with 109 → 109 × 752 × 11 = **901,648**.

Demand is conserved throughout: the log axis is summed away, never dropped.

---

## §3 — Trainset Contract · **PASS WITH LIMITATIONS**

### Cluster trainset — `artifacts/pol4_cluster/trainset/` · verified **PASS**

| Check | Result |
|---|---|
| rows / cols | 901,648 / **69** |
| entities × dates × horizons | 109 × 752 × 11 = 901,648 → **formula matches exactly** |
| horizons present | 1, 2, 3, 5, 7, 10, 14, 18, 21, 25, 30 (= `config.train_horizons`) |
| check-in range | 2023-11-01 → 2025-11-21 (752 days) |
| duplicate `(entity, checkin, horizon)` | **0** |
| X / y / meta row alignment | ✅ equal lengths |
| `target == max(final − observed, 0)` | ✅ **exact** (`np.array_equal`) |
| target negatives / non-finite | 0 / 0 |
| zero-target share | 173,730 (**19.3 %**) |
| target p50 / p99 / max | 272 / 29,889 / 216,699 |
| infinite values in X | **0** |
| null columns | only the 3 intentional (`days_since_first_search`, `days_since_last_search`, `search_span`) — 184,546 rows (20.5 %) |
| manifest checksums | ✅ **all three parquet files verify** |
| sampling | `sampled=False`, `max_train_rows=None` — no subsampling |
| `train_window_days` / `city_history_days` / `seed` | 752 / 752 / 42 |
| Σ final / Σ observed | 3,125,432,508 / 1,306,523,998 |

### F-3.1 — Model A's trainset is absent; the on-disk city trainset is a different experiment · **HIGH**

**Fact.** `artifacts/pol4/trainset/` **does not exist** on this clone (gitignored, local-only).
`artifacts/pol4/run_summary.json → champion.trainset` records `rows: 2655312`, three SHA-256 file
hashes, and `"reused": true`.

The only city-level trainset present is `artifacts/pol4_nolimit/trainset/`, and it is **not** Model A's:

| | expected Model A | `pol4_nolimit` on disk |
|---|---:|---:|
| rows | 2,655,312 | **7,241,760** |
| horizons | 11 (sampled) | **30 (all)** |
| columns | 66 | 66 |
| manifest.json | recorded in run_summary | **absent** |

7,241,760 = 321 × 752 × 30. Its target contract still verifies (`max(final−observed,0)` exact, 0 negatives,
0 infinities, 35.0 % zero targets), but **without a manifest there are no checksums and no contract
metadata** — `train_window_days`, `seed`, `input_digest` and sampling status are NOT VERIFIABLE for it.

**Risk.** A judge cannot audit the supervised dataset that actually produced the submitted model.
`"reused": true` also means the champion run did not rebuild its trainset — it loaded a pre-existing one.
That path *is* guarded (`input_digest` + a 7-field contract check in `pipeline.py`), so this is a
traceability gap, not a correctness gap.

**Recommendation.** Regenerate `artifacts/pol4/trainset/` with `make pol4-submission` on the presenting
machine, or commit its `manifest.json` alone (7 KB) so the checksums travel with the repo.

### F-3.2 — Aggregation conservation · verified **PASS**

Independently checked on the real data:

* Σ `search_count` identical before/after aggregation (284,130,228) and for `evaluation` (2,686,508).
* Per-cluster totals equal the sum of member-city totals (`np.array_equal`).
* **Per-province totals bit-identical** at every cut height → no cluster crosses a province.
* All 321 cities appear in `clusters.csv` exactly once (109 cluster codes, 244 cities pooled, 77 singletons).
* Σ actual is invariant across all aggregation levels → the WAPE denominator does not move (§7).

---

## §4 — Feature Audit · **PASS WITH LIMITATIONS**

69 features verified against `feature_names(CLUSTER_GROUP_ORDER)` and the shipped manifest schema —
**they match exactly**. All are computed from the two-clock tensor; a feature at horizon `h` can only
read matrix columns `≥ h`, which makes cutoff safety structural rather than a convention.

| Family | n | Features | Availability at inference | Leakage risk | Notes |
|---|---:|---|---|---|---|
| base | 2 | `observed_total`, `days_to_checkin` | ✅ from `evaluation.csv` | none | the two anchors |
| pickup | 4 | `pickup_1d/3d/7d/14d` | ✅ | none | sum-of-sums under aggregation |
| velocity | 9 | `pickup_velocity_3d/7d/14d`, `pickup_acceleration_3d/7d`, `pickup_1d/3d/7d_share`, `pickup_3d_over_7d` | ✅ | none | ratios use `EPS=1e-6` guard |
| activity | 7 | `active_search_days`, `days_since_first/last_search`, `max_daily_search`, `mean/std_daily_search`, `search_span` | ✅ | none | 3 are **intentionally null** when never searched (20.5 % of cluster rows) |
| curve | 5 | `expected_completion_fraction`, `expected_remaining_fraction`, `pickup_baseline_prediction`, `pickup_baseline_remaining`, `pickup_surprise` | ✅ | **curves fitted only on check-ins ≤ cutoff** | strongest family; `pickup_baseline_remaining` = 0.182 gain |
| calendar | 5 | `checkin_weekday`, `is_weekend`, `checkin_day_of_month`, `jalali_month`, `jalali_day` | ✅ known-future | none | **no explicit event/holiday flag** — see §9 |
| city | 12 | `city_code`, `province_code`, `lat`, `long`, `city_hist_mean/median/p75/p90/std`, `city_volatility`, `city_weekend_ratio`, `city_weekday_mean` | ✅ | **see F-4.1** | dominates importance |
| market | 11 | `market_observed_total`, `market_pickup_*`, `market_pickup_velocity_*`, `market_pickup_acceleration_*`, `city_share_of_market` | ✅ | none | **7-province total, not national** (F-2.1) |
| province | 11 | `province_observed_total`, `province_pickup_*`, `province_pickup_velocity_*`, `province_pickup_acceleration_*`, `city_share_of_province` | ✅ | none | bit-identical across aggregation levels |
| **cluster** | 3 | `n_cities_in_cluster`, `cluster_min_member_volume`, `cluster_max_member_share` | ✅ | none | **two are effectively dead — F-4.2** |

`city_code` and `province_code` are declared **categorical** (`features.CATEGORICAL`) and cast to
LightGBM `category` dtype in `models.RemainingDemandModel._prepare` — they are *not* treated as ordinal
numerics. Verified in `model_spec.json → models[].categorical`.

### F-4.1 — City-history features are frozen at the fold cutoff, not rebuilt per training origin · **HIGH**

**Fact (empirically demonstrated, not inferred).** `FeatureBuilder.build` calls
`CityHistory.fit(data, cutoff, config)` **once per fold**, and `at_horizon` reuses it for every row.
Measured on the shipped trainset:

```
city_hist_mean for ONE entity across all 752 check-in dates → 1 distinct value
city_hist_std                                               → 1 distinct value
max distinct city_hist_mean per entity, all 109 entities    → 1
city_weekday_mean: exactly 1 value per (entity, weekday)
```

So a training row for check-in **2023-11-01** carries `city_hist_mean = 1296.86` computed over a window
ending **2025-11-21** — i.e. containing 750 days that occur *after* that row's own check-in, **including
that row's own target**.

**What this does and does not break:**

* ✅ **Cutoff safety is intact.** Nothing after the *fold* cutoff is read. The leakage tests pass
  (17/17) because they attack the fold boundary, and this respects it. Backtest WAPE is not inflated
  by post-cutoff information.
* ✅ **Train/serve consistency is preserved.** Inference features are computed at the same cutoff.
* ❌ **Within-training target leakage is real.** Each row's own outcome contributes ~1/752 of its own
  `city_hist_mean`. Systematically, `city_hist_mean` ≈ the city's mean target.
* ❌ **It likely inflates the measured value of the city-history block.** `city_code` (0.293) +
  `city_hist_mean` (0.244) + `city_weekday_mean` (0.087) = **62 % of all gain** in the clustered model.
  The model may be under-using dynamic pickup signal because a near-oracle level feature is cheaper.

**Important mitigation for the A/B question:** this affects Model A and Model B **identically**, so the
clustering comparison is not biased by it.

**Recommendation (do not implement now).** Add an ablation that recomputes `CityHistory` per training
origin (expanding window ending at `checkin − horizon`) and re-scores. If pooled WAPE degrades
materially, the current numbers are optimistic and must be restated. This is the single most important
outstanding experiment.

### F-4.2 — Two of the three cluster features are never used · **MEDIUM**

**Fact.** `artifacts/pol4_cluster/model_bundle_clustered/feature_importance.json`:

| feature | gain share | rank of 69 |
|---|---:|---:|
| `cluster_min_member_volume` | 0.00305 | 18 |
| `cluster_max_member_share` | 0.00017 | 61 |
| `n_cities_in_cluster` | **0.00000** | **69 — never split on** |

**Inference.** `city_code` as a categorical already identifies the pooled row and `city_hist_mean` gives
its scale; the composition columns are redundant. This is internally consistent with the sweep's verdict
(§7): the model was already sharing information across cities, so making pooling explicit adds nothing.

### F-4.3 — Feature importance is **not** stable between the two models · **LOW (informational)**

| feature | Model A (champion) | Model B (clustered) |
|---|---:|---:|
| `city_weekday_mean` | 0.468 | 0.087 |
| `city_code` | 0.028 | **0.293** |
| `city_hist_mean` | 0.158 | 0.244 |
| `pickup_baseline_remaining` | 0.066 | 0.182 |

**Inference.** With 109 categories instead of 321, each carries ~3× the data, so row *identity* becomes
dominant. **Caveat:** these two importances are not strictly comparable (different panels, different
calibration, different fold sets). Directional only. Per-fold importance stability is **NOT VERIFIABLE** —
the sweep does not persist per-fold importances.

---

## §5 — Feature-Set Comparison · **PASS — the audit premise is incorrect**

**The premise "the cluster model's features were reduced" is false, and I verified it rather than
assuming it.**

```
feature_names(CLUSTER_GROUP_ORDER)[:66] == feature_names(GROUP_ORDER)   → True
feature_names(CLUSTER_GROUP_ORDER)[66:] == ['n_cities_in_cluster',
                                            'cluster_min_member_volume',
                                            'cluster_max_member_share']
```

The 66 shared features are **identical in name and order**; 3 are appended. Nothing was removed.
`GROUP_CLUSTER` deliberately sits **outside** `GROUP_ORDER` so no previously-saved champion bundle
changes schema.

Further, singleton rows carry `n_cities_in_cluster = 1`, `cluster_min_member_volume =` own volume,
`cluster_max_member_share = 1.0` — so **the city arm also trains on 69 columns**. The two arms share one
schema and one code path; the only differing variable is the partition. **This is exactly the design the
audit asks for, and it is met.**

One caveat that *does* violate the single-variable principle: **calibration differs** (Model A calibrated,
Model B not). See F-6.1.

### Proposed experiment matrix (specification only — not run)

| # | Arm | Panel | Features | Calibration | Question it answers |
|---|---|---|---|---|---|
| A | City-66 | 321 | 66 | both | current champion reference |
| B | Cluster-(66+3) | swept 74–283 | 69 | both | net effect of aggregation |
| C | City-Compact | 321 | ~20 | both | how much of A is carried by a small core |
| D | Cluster-(Compact+3) | swept | ~23 | both | does aggregation help *more* when the model is weaker |
| E | Compact + Tweedie/Poisson GAM | 321 | ~20 | n/a | does a count-native likelihood beat L1-on-log1p |
| F | A and B with `CityHistory` rebuilt per origin | both | 66 / 69 | both | **quantifies F-4.1** |

Run every arm on the **same 5 cutoffs**, the same city-date rows, the same WAPE implementation, and one
calibration policy per comparison. Model/feature/calibration choices must be nested inside each fold.

### Compact challenger feature set (~20; specification only)

Chosen for inference availability, leakage safety, stability and low mutual correlation:

`observed_total`, `days_to_checkin`, `pickup_1d`, `pickup_3d`, `pickup_7d`, `pickup_14d`,
`pickup_velocity_7d`, `pickup_acceleration_7d`, `pickup_3d_over_7d`, `active_search_days`,
`days_since_last_search`, `expected_completion_fraction`, `pickup_baseline_remaining`, `pickup_surprise`,
`checkin_weekday`, `is_weekend`, `jalali_month`, `jalali_day`, `city_code` (categorical),
`province_code` (categorical), `city_share_of_market`, `province_observed_total`.

Deliberately excluded: all `city_hist_*` (F-4.1 contamination), `lat`/`long` (already implied by
`city_code`), `market_pickup_*` (near-collinear with `province_pickup_*`), `mean/std_daily_search`
(duplicated by `city_hist_*`).

---

## §6 — Evaluation Validity · **PASS WITH LIMITATIONS**

| Check | Result | Evidence |
|---|---|---|
| No random split / shuffle | ✅ | `grep "train_test_split|shuffle=True"` → 0 hits in `ml/pol4/` |
| Walk-forward cutoffs | ✅ | 5 folds, strictly increasing |
| Target windows non-overlapping | ✅ | verified below |
| Calibration fitted only on earlier closed folds | ✅ | `cross_fit_calibration` filters `outcome_dates ≤ cutoff` |
| Honest prequential evaluation reported | ✅ | `validation_audit.json → prequential_selection` (WAPE 0.148618) |
| Fold-cluster bootstrap CI | ✅ | 95 % CI **[0.0976, 0.2219]** |
| Model/feature spec nested in the run | ❌ | `"model_spec_status": "preselected; not nested inside this run"` — self-declared |

**Fold windows (verified, no overlap):**

```
2024-11-21 → 2024-11-22 .. 2024-12-21
2025-05-21 → 2025-05-22 .. 2025-06-20      ← overlaps the Twelve-Day War (§9)
2025-08-21 → 2025-08-22 .. 2025-09-20
2025-09-22 → 2025-09-23 .. 2025-10-22
2025-10-22 → 2025-10-23 .. 2025-11-21
competition: 2025-11-21 → 2025-11-22 .. 2025-12-21
```

### F-6.1 — The 5-fold vs 3-fold comparison is invalid, as the jury states · **HIGH**

**Fact.** Model A is scored on 5 folds **with** `guarded_bias_horizon` calibration → 0.14793.
Model B's sweep uses 3 folds (`2024-11-21`, `2025-08-21`, `2025-10-22`) **without** calibration → 0.12017
for its own unclustered arm.

Two variables differ simultaneously: **fold set** and **calibration**. The 3-fold set also *omits*
`2025-05-21`, the hardest window. **0.12017 and 0.14793 must never be quoted against each other.**

**Mitigation already in place (verified):** `docs/POL4_CLUSTERING.md:~95` carries an explicit warning
block saying exactly this, added at commit `c42243b`. `cluster_spec().calibration == "none"` is recorded
in `run_summary.json → spec`. **The project does not itself make this error.** The risk is a reader making it.

**What IS a valid comparison, and is the one the project actually relies on:** within the sweep, all
levels share cutoffs, rows, target, calibration policy and metric. That internal comparison is sound.

### F-6.2 — Fold count and uncertainty · **MEDIUM**

**Fact.** 5 folds; fold WAPE std = 0.0753; best 0.0926, worst 0.2722; 95 % CI [0.0976, 0.2219].

**Risk.** The CI is wider than the entire clustering effect being debated (≈0.0025). No aggregation or
blend conclusion is statistically separable at n=5 folds — the project's negative verdict is therefore
robust, but a *positive* verdict would not have been.

### Recommended valid comparison protocol

Score **every** arm — A-raw, A-calibrated, B-clustered, B-city, B-blended — on:
the same **5** cutoffs · the same 321×30 city-date rows (disaggregating clustered predictions by the
frozen train-only volume share) · the same `actual` · one calibration policy applied identically ·
`_wape` from `ml/evaluation/metrics.py` · no hyper-parameter chosen on the final holdout.
Report pooled WAPE **and** per-fold WAPE with the fold-cluster bootstrap CI.

### Breakdowns available vs missing

Available in `backtest_metrics_phase2.json`: `by_horizon`, `by_horizon_bucket`, `by_province`,
`by_weekday`, `by_demand_bucket`, `by_observation_state`, `high_demand` (top 1/5/10 %), `worst_cities`,
`city_wape_quantiles`. Stability D-30→D-1 in `stability.parquet`.
**NOT VERIFIABLE:** per-horizon breakdown *within* individual folds — `folds[*]` stores only
`n, actual_total, predicted_total, wape, mae, normalised_bias`, so the war-window decomposition inside
fold `2025-05-21` cannot be computed from artifacts (§9).

---

## §7 — Metric Audit · **PASS**

| Metric | Implementation | Verdict |
|---|---|---|
| WAPE | `metrics.py:47` — `np.sum(np.abs(t-p)) / np.sum(np.abs(t))`, NaN if denominator ≤ EPS | ✅ standard, correct |
| normalised bias | `backtest.py` — `np.sum(pred-actual) / np.sum(actual)` | ✅ matches the brief; distinct from `metrics.bias` (mean signed error), and the code says so |
| MAE | `np.mean(np.abs(t-p))` | ✅ |
| pooled vs fold-mean | pooled = demand-weighted over concatenated rows | ✅ correct choice; fold-mean would over-weight small folds |
| CI | exact fold-cluster bootstrap, 95 % | ✅ appropriate for clustered folds |
| train/validation gap | `experiments.csv → validation_minus_train_wape` | ✅ reported |

### Aggregation gain decomposition (the requested table) — verified from `run_summary.json`

Denominator invariance confirmed independently: Σ actual is identical at every level, so WAPE across
levels is like-for-like.

| rows | cut height | merged demand | clustered WAPE | control¹ | mechanical | modelling | modelling share | sparse-slice WAPE |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **321** | 0.000 | 0.0 % | **0.120168** | — | — | — | — | — |
| 283 | 0.147 | 0.34 % | 0.120787 | 0.120008 | +0.00016 | −0.00078 | — | 0.2141 |
| 233 | 0.336 | 1.91 % | 0.121923 | 0.119392 | +0.00078 | −0.00253 | — | 0.2241 |
| 199 | 0.511 | 2.36 % | 0.121290 | 0.118993 | +0.00118 | −0.00230 | — | 0.2355 |
| 153 | 0.865 | 2.73 % | 0.119969 | 0.118475 | +0.00169 | −0.00149 | — | 0.2390 |
| **110** | 1.970 | 2.94 % | **0.117699** | 0.117952 | +0.00222 | **+0.00025** | **10.3 %** | 0.2451 |
| 74 | 20.854 | 3.00 % | 0.118373 | 0.117540 | +0.00263 | −0.00083 | — | 0.2456 |

¹ control = the **city model's own predictions**, summed onto that level's rows.

**Answer to "does WAPE improve artificially after aggregation?" — Yes, and the project measures it.**
At the best level the apparent gain is 2.05 % relative; **89.7 % of it is mechanical** (offsetting errors
cancelling inside a merged row before the absolute value is taken). The earned modelling gain is
0.00025 WAPE = **0.21 % relative**, and it is *negative* at all five other levels.

Aggregation ratio at the shipped level: 109/321 = 0.340. Largest cluster: **26 cities**. Singletons: **77**.
Merged demand share: **3.04 %**.

**Disaggregation/reconciliation error: NOT VERIFIABLE.** No artifact stores clustered predictions pushed
back to city grain against city actuals. The blend does allocate cluster *remaining* by train-only volume
share (§8) and that allocation sums back exactly to the cluster forecast (test-verified), but the
resulting per-city reconciliation error is never scored.

---

## §8 — Calibration & Blend · **PASS WITH LIMITATIONS**

### F-8.1 — raw and calibrated are ONE booster plus post-processing · **verified, correct**

**Fact.** SHA-256 of the boosters in `model_variants/raw/` and `model_variants/calibrated/`:

```
lightgbm_h1_14.txt   raw = c0120f475b79691e … calibrated = c0120f475b79691e   IDENTICAL
lightgbm_h15_30.txt  raw = 9a3366adda1285a4 … calibrated = 9a3366adda1285a4   IDENTICAL
```

They are the **same trees**; the variants differ only in the attached `calibrator_state.joblib`.
This is the right design and it is honestly labelled.

### Calibration mechanics (verified from `calibration.py:47-52`)

```python
def apply_alpha(observed, predicted, alpha):
    remaining = np.maximum(predicted - observed, 0.0)
    return observed + alpha * remaining
```

**Calibration scales only the projected remainder, never the observed part** — so the Phase-1 floor
(`predicted ≥ observed`) survives calibration by construction. Correct.

Factors are fitted on **strictly out-of-fold, fully-closed earlier folds** (`cross_fit_calibration`
filters `fold_cutoff + 30d ≤ current cutoff`). Leakage-safe. ✅

### Guardrail (verified from `pipeline.py`)

A calibration candidate is accepted only if **all three** hold:
pooled WAPE ≤ raw WAPE · |bias| < |raw bias| · no fold worsens by more than +0.01 WAPE.
Among candidates passing, the one with **smallest |bias|** wins.

Measured: raw 0.15508 / bias −0.09855 → selected `guarded_bias_horizon` 0.147927 / bias −0.068189.
**WAPE improved and bias improved simultaneously** — the "bias fixed but WAPE damaged" failure mode did
*not* occur. The rejected candidate `bias_horizon_shrunk` (WAPE 0.172189, bias −0.013367) is exactly that
failure mode, and the guardrail correctly refused it. ✅ **This is a genuine strength.**

### Shrinkage blend formula (extracted from `cluster_experiment.blend_panels`)

```
w(c)    = 1                              if c is a singleton in its cluster
        = volume(c) / (volume(c) + k)    otherwise
pred(c) = observed(c) + w(c)·city_remaining(c) + (1−w(c))·share(c)·cluster_remaining(g)
```

* Applied to **remaining** demand → the observed floor holds structurally, not by clamping (test-verified).
* `share(c)` = the city's **train-only** volume share of its cluster; members of a zero-history cluster
  split evenly. Allocation sums back exactly to the cluster forecast (test-verified).
* **High-volume city:** `w → 1`, keeps its own forecast. **Low-volume city:** `w → 0`, leans on the pooled
  series. **Singleton:** `w = 1` always — forced, because blending a singleton would be a two-model
  ensemble, not shrinkage.

### F-8.2 — `select_blend` has no minimum-gain guard, and 36 combinations were screened on the same 3 folds · **MEDIUM**

**Fact.** `select_blend` accepts any blend with `wape < unclustered_wape` — **no margin**.
`blend_sweep.csv` contains **36 rows** (6 cut heights × 6 shrinkage constants), all scored on the same
3 folds. The winner: cut height 0.864659, k = 10⁴, pooled WAPE **0.120096** vs unclustered **0.120168**.

**Is 0.120168 → 0.120096 meaningful?**

* **Statistically: no.** Δ = 0.000072 absolute, **0.06 % relative**. The fold-cluster bootstrap CI on the
  champion spans 0.098–0.222 — four orders of magnitude wider than this delta. With 36 candidates
  screened on 3 folds, the expected best-of-36 gain under a pure-noise null is comfortably larger than
  0.06 %. This is **selection noise, not signal**.
* **Business: no.** On the 7,996,589-search forecast total, 0.06 % of WAPE is ≈ 2,300 searches across
  9,630 city-nights — far below any decision threshold.
* **Mildly reassuring:** it is better on **3 of 3 folds** (0.125852 / 0.118966 / 0.118323 vs
  0.125988 / 0.119047 / 0.118344), so it is at least not harmful.

**Recommendation.** Either impose a minimum relative gain on `select_blend` (as `select_level` has), or
present the blend as "measured, neutral" rather than as the selected arm. **Do not claim the blend
improves accuracy.**

### F-8.3 — The shipped selected arm's panel is not persisted · **HIGH**

**Fact (the discrepancy the jury flagged, confirmed exactly).**

| artifact | cut height | groups | rows if materialised |
|---|---:|---:|---:|
| `trainset/` + `model_bundle_clustered/` | **1.970305** | **109** | 901,648 |
| `blend_selection` → source panel | **0.864659** | **149** | **1,232,528** |

Independently recomputed: `plan.assign(0.864659).n_groups = 149` → 149 × 752 × 11 = **1,232,528**,
matching the jury's arithmetic exactly.

**Mechanism (read from `cluster_pipeline.py`):** the code is *functionally correct* — when the blend's
cut height differs from the clustered arm's, it calls `fit_panel(...)` again at 0.864659. So
`results_blended.csv` is built from a genuine 149-group fit. **But that fit is neither persisted nor
checksummed**, and its panel has no trainset on disk.

**Is this only artifact ambiguity, or does it affect reproducibility?** — **Both, and the second is the
serious one:**

1. `run_summary.json → selected_arm.arm = "blended"`, so `results_selected.csv` is the blended file.
2. The only clustered model bundle on disk is at cut height 1.970305 — a **different arm**.
3. A judge tracing `results_selected.csv` → a model will land on the **wrong model**.
4. The 149-group panel is deterministic (`seed=42`) and regenerable, so nothing is *lost* — but it cannot
   be audited without a 20-minute re-run, and that re-run depends on unpinned LightGBM (F-1.2).

**Recommendation.** Either persist the blend-source panel and bundle, or force the blend to reuse the
clustered arm's cut height, or (simplest and best-supported by the evidence) stop selecting the blend at
all — see F-8.2.

---

## §9 — Events, Holidays & Structural Shocks · **FAIL**

### Calendar feature inventory (verified)

`feature_names(('base','calendar'))` → `checkin_weekday`, `is_weekend`, `checkin_day_of_month`,
`jalali_month`, `jalali_day`. **That is all.**

* **No explicit event flag, holiday flag, bridge-day flag, or shock indicator exists anywhere in the
  Pol 4 feature set.**
* `backend/ml/data/holidays.py` exists in the repo but is **never imported by `ml/pol4/`** (verified).
* `features.py:246` states the design intent: Jalali month/day is meant to carry fixed-date events
  implicitly ("Nowruz is 1 Farvardin, and Yalda … is 30 Azar").

**Assessment.** For **solar fixed-date** events (Nowruz 1 Farvardin, Yalda 30 Azar) the Jalali pair is a
legitimate implicit encoding, and a tree can learn it. For everything else it is **insufficient**:

| Category | Covered by `jalali_month`/`jalali_day`? | Why |
|---|---|---|
| Fixed solar holidays (Nowruz, Yalda) | Partially ✅ | fixed Jalali date; tree can memorise |
| Bridge / connected holidays | ❌ | depend on weekday alignment, which varies by year |
| **Hijri-lunar events** (Ramadan, Eid al-Fitr, Muharram/Ashura, Arbaeen, martyrdom days) | ❌ **structurally impossible** | the lunar calendar drifts ~11 days/year against the solar one, so the same event has a *different* Jalali date every year. A fixed Jalali encoding cannot represent it. |
| School holidays | ❌ | not in the data |
| Destination-specific religious events (Mashhad, Qom) | ❌ | needs event × destination affinity |
| Weather / road closure | ❌ | **قابل محاسبه نیست** — no such data exists in the four files |
| War / crisis / earthquake | ❌ | not representable as a known-future feature at all |
| Internet disruption | ❌ | **قابل محاسبه نیست** — no such data |

**Recommendation for lunar events (do not implement from a fixed conversion).** Build a dated table from
an official Iranian calendar, one row per event **per year**, verified against an authoritative source;
record a **±1-day uncertainty flag** because Hijri month starts depend on crescent sighting and can differ
by a day. Add destination affinity (Mashhad/Qom weight high for religious dates) and geographic scope.
Do not derive lunar dates by arithmetic conversion from `jalali_month`/`jalali_day`.

### F-9.1 — The Twelve-Day War contaminates the headline metric and is documented nowhere · **HIGH**

**Externally verified fact (not inferred, not guessed).** The **Twelve-Day War** between **Israel and
Iran** ran **13 June 2025 → 24 June 2025**, ending in a US/Qatar-mediated ceasefire effective
24 June 2025. Jalali equivalents (computed with the repo's own `to_jalali_string`):
**1404-03-23 → 1404-04-03**.

**Fold overlap (computed):**

| fold cutoff | target window | overlaps war | training window contains war |
|---|---|---|---|
| 2024-11-21 | 2024-11-22 .. 2024-12-21 | no | no |
| **2025-05-21** | **2025-05-22 .. 2025-06-20** | **YES — 8 of 30 days (13–20 Jun)** | no |
| 2025-08-21 | 2025-08-22 .. 2025-09-20 | no | **yes** |
| 2025-09-22 | 2025-09-23 .. 2025-10-22 | no | **yes** |
| 2025-10-22 | 2025-10-23 .. 2025-11-21 | no | **yes** |

**The data shows an unambiguous structural break at exactly that date.** Daily search volume by
`log_date`, indexed to the 1 May–10 Jun mean (413,863/day):

```
2025-06-12   413,805   1.00x
2025-06-13   198,497   0.48x   ← war begins; collapse
2025-06-14   328,093   0.79x
2025-06-15   910,843   2.20x
2025-06-16   911,509   2.20x
2025-06-17 1,192,726   2.88x   ← peak
2025-06-18   734,277   1.77x
...
2025-06-24   255,602   0.62x   ← ceasefire; sustained slump
2025-06-26   205,122   0.50x
```

The **target itself** is anomalous, not just search behaviour. Final demand by check-in date inside that
fold's window, indexed to its own pre-war baseline (477,941/night):

```
h=22  2025-06-12   798,926   1.67x
h=23  2025-06-13   769,451   1.61x    ← war days begin
h=27  2025-06-17   818,233   1.71x
h=28  2025-06-18   936,100   1.96x
h=29  2025-06-19   983,653   2.06x
h=30  2025-06-20   725,858   1.52x
```

**Impact on the headline metric (computed):**

| | pooled WAPE |
|---|---:|
| all 5 folds (**the published headline**) | **0.14793** |
| excluding fold `2025-05-21` (**diagnostic only**) | **0.10615** |
| difference | **+0.04178 = 28.2 % of the headline** |

Fold `2025-05-21` holds 25.2 % of backtest demand but **46.3 % of total absolute error**.

**Interpretation, stated carefully.** The temporal coincidence is exact and the magnitude is extreme, so
the war is by far the most plausible explanation. But this audit did **not** run an event study, so
**causal attribution is not established** — only that the fold overlapping the war carries the anomaly.

**Risks in both directions — this cuts both ways and the presentation must say so:**

* Quoting 0.14793 as "our model's accuracy" **overstates error under normal conditions** by ~28 %.
* Quoting 0.10615 **without disclosure would be fold selection on outcome** — scientifically indefensible.
* 3 of 5 folds *train* on post-war data; the regime may have shifted, and the competition window
  (2025-11-22 .. 2025-12-21) is **after** the ceasefire — outside the shock but possibly in a new regime.
* The war must **never** be encoded as a known-future holiday feature. It is an intervention / regime
  shift, knowable only after the fact.

**Recommendation.** Report **both** numbers, always together, with the war disclosed. Add the fold's
per-horizon breakdown to the artifacts so the in-fold decomposition becomes verifiable
(currently NOT VERIFIABLE — `folds[*]` stores only 6 scalars).

### Proposed event-study method (specification only — do not run now)

Window: **pre** = 2025-05-14..06-12, **during** = 06-13..06-24, **post** = 06-25..07-24.
Match on weekday and on the same Jalali dates in 1403 (previous year) to absorb seasonality.
Treated = cities with a demand break at 06-13 (CUSUM on the daily series); control = cities without one.
Estimate uplift as a difference-in-differences on log demand, with a fold-cluster bootstrap CI.
**Limitation, to be stated explicitly:** a nationwide shock has no clean control group, so this is
descriptive attribution, **not causal identification**. Report as "associated with", never "caused by".
Then re-run the champion with and without the shock window as an **experiment** — not a retrain of the
submitted model.

---

## §10 — Business Review · **PASS WITH LIMITATIONS**

**The target variable is `search_count` — a search-intent proxy.** It is **not** bookings, revenue,
occupancy, ADR or conversion, and no causal marketing lift is measurable from it.

| Use case | Decision | KPI needed | Data present | Data missing | Confidence | Type |
|---|---|---|---|---|---|---|
| **Destination prioritisation** | where to point demand-gen | forecast demand share by city | ✅ `results.csv` | conversion rate, margin | Medium | Predictive |
| **Campaign timing** | which nights to push | forecast by check-in date + lead-time curve | ✅ `results.csv`, `pickup_curves.parquet` | marketing spend, elasticity | Medium | Predictive |
| **Regional campaigns** | budget by province | province forecast split | ✅ `province_summary.json` | regional cost/CAC | Medium | Predictive |
| **Pickup momentum** | detect accelerating destinations | `pickup_acceleration_7d`, `pickup_surprise` | ✅ `city_momentum.parquet` | — | Medium-High | Descriptive |
| **Emerging destinations** | long-tail discovery | YoY search growth by city | ✅ derivable | supply availability | Medium | Descriptive |
| **Budget under uncertainty** | allocate with risk | forecast + CI | ⚠️ CI only at fold level | per-city predictive intervals | **Low** | Predictive |
| **Host outreach targeting** | where to recruit hosts | demand-proxy growth by city | ✅ | **inventory, occupancy, unmet demand** | **Low** | Descriptive |
| **Seasonal supply planning** | pre-position supply | 30-day forecast by city | ✅ | capacity, lead time to onboard | Low-Medium | Predictive |
| **Peak monitoring / alerting** | ops staffing | forecast vs observed pickup | ✅ `stability.parquet` | ops capacity | Medium | Predictive |
| **Scenario planning** | shock response | event-window analysis | ⚠️ war window visible | causal model | **Low** | Descriptive |

### Claims that are NOT permitted without booking/revenue/conversion/ADR/inventory/occupancy/cancellation/capacity/marketing-spend data

**قابل محاسبه نیست** for all of the following — say none of them:

* "This city will generate X bookings / X revenue."
* "Demand exceeds supply here, so recruit hosts." (no inventory or occupancy data)
* "This campaign caused an N % lift." (no spend, no control group, no causal design)
* "Occupancy will reach X %." · "Optimal price is Y." · "Expected cancellation rate is Z."
* "Conversion is improving." · Any ROI, CAC, LTV or margin statement.
* Any *national* Iranian claim — the panel is **7 provinces** (F-2.1).

**Permitted framing:** "search interest", "demand proxy", "relative attention", "forecast search volume".

---

## §11 — Presentation Charts

Prioritised. **Every chart below is backed by an artifact that exists** — none require mock data or an
uncomputed metric. Charts requiring data the project does not have are listed as excluded.

| # | Chart | Judge question it answers | Type | x / y / colour / filter | Source artifact | Grain | Model toggle | Business takeaway | Technical caveat |
|---|---|---|---|---|---|---|---|---|---|
| **P1** | National* 30-day forecast | "What is your answer?" | line | checkin / predicted / variant / — | `artifacts/pol4/results.csv` | 30 dates | raw vs calibrated | the deliverable | *7-province, not national. **No actual — future window.** |
| **P2** | WAPE & bias by fold | "How do you know it works?" | bar + line | fold / WAPE, bias / metric / — | `backtest_metrics_phase2.json` | 5 folds | champion vs baseline | 0.1479 vs 0.2200 baseline | must annotate the war fold |
| **P3** | Aggregation frontier | "Did you over-aggregate?" | line | n groups / WAPE / arm / — | `cluster_sweep.csv` | 7 levels | clustered vs control | evidence-based "no clustering" | 3 folds, uncalibrated |
| **P4** | Mechanical vs modelling gain | "Is that gain real?" | stacked bar | n groups / ΔWAPE / component / — | `run_summary.json → gain_decomposition` | 6 levels | — | 89.7 % of the gain is free | the killer slide |
| **P5** | Event-impact timeline | "What about the war?" | line + shaded band | date / daily searches / — / — | `search_data.csv` | daily | — | honest disclosure of the 0.272 fold | association, not causation |
| **P6** | Error by demand bucket | "Where does error live?" | bar | quantile bucket / WAPE / — / — | `backtest_metrics_phase2.json → by_demand_bucket` | 5 buckets | — | top decile dominates | actual-based bucketing |
| **P7** | Demand by lead-time bucket | "Why is this hard?" | bar | horizon bucket / observed share, WAPE / — / — | `pickup_curves.parquet`, `by_horizon_bucket` | 5 buckets | — | D-7..D-21 is the battleground | — |
| **P8** | Selected-city pickup curve | "How does demand accumulate?" | line | days to check-in / cumulative share / city / city | `pickup_curves.parquet` | per city | — | the core mechanism | city/province/global fallback must be labelled |
| **P9** | Top opportunity cities | "Where should we act?" | bar | city / forecast, momentum / province / province | `results_named.csv`, `city_momentum.parquet` | 321 cities | variant | demand-gen shortlist | **search proxy — not bookings** |
| **P10** | Forecast stability D-30→D-1 | "Is the forecast trustworthy?" | line | days to check-in / prediction / city / city | `stability.parquet` | per city-date | — | convergence, not swinging | historical window only |
| **P11** | City × date heatmap | "Where and when?" | heatmap | checkin / city / predicted / province | `results.csv` | 321×30 | variant | the whole surface at once | 321 rows needs sorting/top-N |
| **P12** | Cluster composition | "What did clustering do?" | treemap/bar | cluster / member count, volume / province / — | `clusters.csv` | 109 clusters | — | hubs protected, tail pooled | only if P3/P4 are shown |
| **P13** | Feature-family contribution | "What drives the model?" | bar | family / gain share / — / — | `feature_importance.json` | 9 families | A vs B | curve + city history dominate | **disclose F-4.1** |
| **P14** | Geographic demand map | "Where on a map?" | scatter map | long / lat / forecast / province | `results.csv` + `cities.csv` | 321 cities | variant | regional concentration | 7 provinces only |
| **P15** | Calibration/reliability plot | "Is it biased?" | line | horizon bucket / bias / variant / — | `backtest_metrics_phase2.json` | 5 buckets | raw vs calibrated | bias −0.099 → −0.068 | backtest only |
| **P16** | Model comparison, identical fold | "A vs B fairly?" | grouped bar | fold / WAPE / arm / — | — | — | — | — | **DO NOT BUILD YET** — no artifact scores A and B on identical folds+calibration (F-6.1). Building it from current artifacts would be the exact error the audit forbids. |

**Excluded — would require mock data:** booking funnel, revenue forecast, occupancy, conversion,
campaign ROI, competitor comparison, weather overlay.

**Rule enforced throughout:** actual-vs-prediction appears **only** on backtest charts (P2, P6, P7, P15).
P1, P9, P11, P14 are the *future* window and must show **prediction only** — no actual exists.

---

## §12 — KPI Dictionary

### A. Executive / business

| KPI | Formula | Source | Grain | Refresh | Interpretation | Direction | Limitation |
|---|---|---|---|---|---|---|---|
| Forecast search volume | `Σ predicted_demand` | `results.csv` | panel × 30d | per run | expected search interest | — | proxy, not bookings |
| Top-N demand share | `Σ top-N / Σ all` | `results.csv` | city | per run | concentration | context | 7 provinces only |
| Province split | `Σ by province / total` | `province_summary.json` | 7 provinces | per run | regional mix | context | — |
| Pickup momentum | `pickup_acceleration_7d` | `city_momentum.parquet` | city | per run | accelerating interest | ↑ good | descriptive only |
| Emerging destination score | YoY growth of city demand | derivable from `search_data.csv` | city | per run | long-tail discovery | ↑ good | no supply data |

### B. Model trust

| KPI | Formula | Source | Grain | Refresh | Interpretation | Direction | Limitation |
|---|---|---|---|---|---|---|---|
| Pooled WAPE | `Σ|a−p| / Σ|a|` | `backtest_metrics_phase2.json` | pooled 5 folds | per backtest | headline accuracy | ↓ good | **28.2 % driven by the war fold** |
| Pooled WAPE ex-shock | same, 4 folds | computed | 4 folds | per backtest | normal-conditions accuracy | ↓ good | **diagnostic only — never the headline** |
| Normalised bias | `Σ(p−a) / Σa` | same | pooled | per backtest | systematic over/under | → 0 | −0.068 after calibration |
| Baseline improvement | `1 − WAPE_model / WAPE_baseline` | `run_summary.json` | pooled | per backtest | value over a 40-line baseline | ↑ good | 0.3276 |
| Fold-cluster 95 % CI | exact bootstrap | `validation_audit.json` | pooled | per backtest | uncertainty | narrow good | **[0.098, 0.222] — very wide** |
| Prequential WAPE | honest sequential selection | `validation_audit.json` | pooled | per backtest | no hindsight | ↓ good | 0.148618 |
| Train/validation gap | `WAPE_val − WAPE_train` | `experiments.csv` | per experiment | per ablation | overfit detector | small good | — |
| Forecast stability | mean abs revision D-30→D-1 | `stability.parquet` | city-date | per run | trustworthiness | ↓ good | historical only |

### C. Clustering / aggregation

| KPI | Formula | Source | Grain | Refresh | Interpretation | Direction | Limitation |
|---|---|---|---|---|---|---|---|
| Aggregation ratio | `n_groups / 321` | `cluster_sweep.csv` | level | per sweep | how much was conceded | ↑ good (less aggregation) | 0.340 at shipped level |
| Merged demand share | `Σ merged-row demand / Σ all` | `run_summary.json` | level | per sweep | true aggregation cost | ↓ good | 3.04 % |
| Mechanical gain | `WAPE_unclustered − WAPE_control` | `gain_decomposition` | level | per sweep | free, unearned | context | +0.00222 |
| **Modelling gain** | `WAPE_control − WAPE_clustered` | `gain_decomposition` | level | per sweep | **the only gain that counts** | ↑ good | **+0.00025 (0.21 % rel.)** |
| Modelling share of gain | modelling / total | `gain_decomposition` | level | per sweep | how much was earned | ↑ good | **10.3 %** |
| Sparse-slice WAPE | WAPE on mergeable cities only | `cluster_sweep.csv` | slice | per sweep | tail accuracy | ↓ good | 0.245 on 2.83 % of demand |
| Largest cluster / singletons | max size / count of size 1 | `clusters.csv` | level | per sweep | aggregation shape | — | 26 / 77 |

---

## §13 — Frontend & Traceability · **PASS WITH LIMITATIONS**

* **Model selector genuinely loads different artifacts** ✅ — `backend/apps/pol4/services.py:33,39` maps
  variants to `results_raw_named.csv` and `results_calibrated_named.csv`; both files exist and differ.
  This is not a cosmetic toggle.
* **Cluster arm is NOT wired to the API or the frontend** — `grep` for
  `pol4_cluster|results_clustered|results_blended|clusters.csv` across `frontend/` and `backend/apps/`
  returns **nothing**.
  * ✅ **Consequence for the jury's stated risk:** the "109 cluster rows vs 321 city rows displayed
    wrongly" failure **cannot occur**, because clustering never reaches the UI.
  * ❌ **Consequence for presentation:** the strongest scientific story in the project (P3/P4) has **no
    product surface**. It must be told from static charts.
* **NOT VERIFIABLE in this audit:** interactivity, tooltips, RTL correctness, Persian number formatting,
  accessibility, stale-artifact fallbacks. These require running the Next.js app and inspecting the DOM,
  which exceeds a read-only file audit. Recommend a separate UI pass before presenting.

---

## §14 — Jury Verdict

### 14.1 Executive verdict

Novo Pulse / Pol 4 is a **methodologically strong, honestly-reported forecasting project with three
fixable integrity gaps and one undisclosed metric contaminant.**

What is genuinely excellent: the data contract is spotless (0 duplicates, 0 impossible dates, 0 coverage
gaps, provenance hashes verified byte-for-byte); leakage against the competition cutoff is closed
*structurally* — a feature at horizon `h` cannot read below column `h` — and proven by tests that corrupt
post-cutoff rows and demand identical output; the calibration guardrail correctly **rejected** the
bias-optimal candidate that damaged WAPE; and the clustering question was answered with a swept Pareto
curve plus a control arm that separates *free* metric gain from *earned* modelling gain. That control arm
is the most sophisticated thing in the project: it turned an apparent 2.05 % win into a measured 0.21 %,
and the team then declined to ship it. **Choosing not to use a technique, with evidence, is the strongest
result here.**

What must be fixed before presenting: (1) `scipy` is undeclared, so Model B does not run on a clean
clone; (2) the `2025-05-21` fold overlaps the verified 13–24 June 2025 Twelve-Day War and alone drives
46 % of backtest error — the headline 0.1479 becomes 0.1062 without it, and this appears in no document;
(3) the shipped `results_selected.csv` comes from a 149-group panel that exists nowhere on disk, while
the persisted trainset and model bundle belong to a different 109-group arm.

None of these invalidate the submission (`artifacts/pol4/results.csv` is valid: 9,630 rows, 321×30,
0 NaN, 0 negatives, 0 rows below the observed floor). They are disclosure and reproducibility failures,
not correctness failures.

### 14.2–14.4 Scores

| Dimension | Score | Reasoning |
|---|---:|---|
| **Technical** | **78 / 100** | Data contract, leakage discipline, control-arm design and calibration guardrail are top-tier (would be 90+). Deductions: undeclared dependency (−6), unpinned deps (−4), F-4.1 city-history contamination (−7), unpersisted selected arm (−3), missing champion trainset (−2). |
| **Business** | **62 / 100** | Honest about proxy limits; real momentum/prioritisation insight. Deductions: no booking/revenue/occupancy data caps every use case at descriptive-or-predictive (−20), "national" mislabel on a 7-province panel (−8), no per-city predictive intervals (−10). |
| **Presentation readiness** | **65 / 100** | Artifacts exist for 15 of 16 proposed charts. Deductions: war contaminant undisclosed (−15), clustering story has no UI surface (−10), P16 not buildable from current artifacts (−5), UI quality unverified (−5). |

### 14.5 Section PASS/FAIL table

| § | Section | Status |
|---|---|---|
| 1 | Repository & Reproducibility | **PASS WITH LIMITATIONS** |
| 2 | Raw Data | **PASS** |
| 3 | Trainset Contract | **PASS WITH LIMITATIONS** |
| 4 | Feature Audit | **PASS WITH LIMITATIONS** |
| 5 | Feature-Set Comparison | **PASS** (audit premise was incorrect; corrected with evidence) |
| 6 | Evaluation Validity | **PASS WITH LIMITATIONS** |
| 7 | Metric Audit | **PASS** |
| 8 | Calibration & Blend | **PASS WITH LIMITATIONS** |
| 9 | Events, Holidays & Shocks | **FAIL** |
| 10 | Business Review | **PASS WITH LIMITATIONS** |
| 11 | Presentation Charts | **PASS WITH LIMITATIONS** |
| 12 | KPI Dictionary | **PASS** |
| 13 | Frontend & Traceability | **PASS WITH LIMITATIONS** (partly NOT VERIFIABLE) |

### 14.6 Top ten findings by severity

| # | Finding | Severity | § |
|---|---|---|---|
| 1 | `scipy` undeclared → `make pol4-cluster` fails on a clean clone | **CRITICAL** | F-1.1 |
| 2 | Twelve-Day War contaminates fold `2025-05-21`; 46.3 % of backtest error; undisclosed | **HIGH** | F-9.1 |
| 3 | `city_hist_*` frozen at fold cutoff → within-training target leakage | **HIGH** | F-4.1 |
| 4 | Shipped selected arm's 149-group panel/model not persisted | **HIGH** | F-8.3 |
| 5 | Champion trainset absent; on-disk city trainset is a different config with no manifest | **HIGH** | F-3.1 |
| 6 | Zero pinned dependencies | **HIGH** | F-1.2 |
| 7 | 5-fold-calibrated vs 3-fold-uncalibrated comparison invalid (documented, but fragile) | **HIGH** | F-6.1 |
| 8 | Blend selected on 0.06 % gain, best-of-36 on the same folds, no minimum-gain guard | **MEDIUM** | F-8.2 |
| 9 | "National" used for a 7-province panel | **MEDIUM** | F-2.1 |
| 10 | Two of three cluster features never used; clustering absent from UI | **MEDIUM** | F-4.2, §13 |

### 14.7 Critical blockers before presenting

1. **Add `scipy` to `requirements.txt`** — otherwise a judge reproducing Model B hits an import error.
2. **Disclose the war fold** on any slide quoting 0.1479. Show 0.1479 and 0.1062 together with the reason.
3. **Fix the `results_selected.csv` → model traceability gap** (persist the 149-group arm, or stop
   selecting the blend).

### 14.8 Defensible strengths

* Data contract verified independently: 0 duplicates, 0 impossible dates, 0 orphan cities, 0 province
  disagreements, hashes matching the manifest byte-for-byte.
* Structural leakage safety: the horizon-indexed tensor makes cutoff violation *unrepresentable*, and 17
  leakage tests prove it by corrupting post-cutoff data.
* The control arm (`aggregate_panel`) separating mechanical from modelling gain — a genuinely
  sophisticated piece of experimental design.
* A calibration guardrail that rejected the bias-optimal candidate because it damaged WAPE.
* Clustering built as a **data** step, so both arms share one code path and one 69-column schema —
  making the A/B attributable to the partition alone (verified, not claimed).
* Cross-machine reproducibility: a Fedora run reproduced macOS WAPE to 5 decimals at all 21 sweep points.
* 423 tests passing, including a submission-contract validator.

### 14.9 Claims permitted in the presentation

* "Pooled walk-forward WAPE 0.1479 across 5 simulated competitions, 32.8 % better than a strong pickup
  baseline (0.2200) — **and one fold overlaps the June 2025 war; excluding it the figure is 0.1062**."
* "Leakage against the competition cutoff is structurally impossible and proven by tests."
* "We tested clustering, swept the aggregation level, and found 90 % of its apparent gain was a metric
  artefact — so we submit unclustered."
* "The submission is validated: 9,630 rows, 321 cities × 30 dates, no NaN, no negatives, never below
  observed demand."
* "Forecasts are of **search interest** across a **7-province panel**."

### 14.10 Claims that must NOT be made

* ❌ Anything about bookings, revenue, occupancy, ADR, conversion, cancellation or ROI.
* ❌ "National"/"Iran-wide" demand — the panel is 7 provinces.
* ❌ "Our model achieves 0.1479" without disclosing the war fold — **or** "0.1062" without disclosing
  that a fold was excluded.
* ❌ Comparing 0.1202 (cluster sweep) with 0.1479 (champion) — different folds and calibration.
* ❌ "The blend improves accuracy" — 0.06 %, best-of-36, statistically indistinguishable from noise.
* ❌ SHAP, uncertainty intervals, anomaly detection, ensembling, Chronos or neural models as part of the
  Pol 4 solution — the package imports none of them.
* ❌ "The model handles holidays/Nowruz/Ramadan" — there is no event feature; lunar events are
  structurally unrepresentable by `jalali_month`/`jalali_day`.
* ❌ Any causal marketing-lift statement.

### 14.11 Final Model A vs Model B comparison

**They are the same model.** Verified field-by-field from the two `model_spec.json` files:
`kind=lightgbm`, `params={'n_estimators': 600}`, `objective=regression_l1`, `log1p=True`,
`bands=[[1,14],[15,30]]`, `blend=1.0`, `seed=42`, `cutoff=2025-11-21` — all identical. The first 66
feature columns are identical in name and order.

| | Model A (city) | Model B (cluster) |
|---|---:|---:|
| panel rows | 321 | 109 |
| submission rows | 9,630 | 3,270 |
| features | 66 (+3 unused in A's bundle) | 69 |
| training rows | 2,655,312 | 901,648 |
| calibration | `guarded_bias_horizon` | none |
| folds | 5 | 3 |

**Verdict:** on the only valid (internal, like-for-like) comparison, Model B's earned modelling gain is
**+0.00025 WAPE = 0.21 % relative**, and it is negative at 5 of 6 levels. It costs 66 % of the submission
rows and 66 % of the training data. **Model B does not justify itself.**

### 14.12 Champion / challenger / insight-only

* **Champion (submit):** Model A, calibrated — `artifacts/pol4/results.csv`. Unchanged.
* **Challenger (report, do not submit):** the Compact-20 model (§5) — tests whether 62 % of gain sitting
  in contaminated city-history features is load-bearing.
* **Insight-only:** Model B / clustering — an evidence artifact for the report and slides P3/P4/P12.
  Never a submission candidate. The blend should be demoted from "selected" to "measured, neutral".

### 14.13 Required experiments (specification only — none run)

1. **F-4.1 ablation** — rebuild `CityHistory` per training origin; re-score. *Highest priority: it
   determines whether the headline numbers are optimistic.*
2. **Uniform 5-fold, one-calibration-policy comparison** of all five arms at city grain (§6).
3. **Shock sensitivity** — champion with and without the 2025-06-13..24 window, reported as a diagnostic.
4. **Event study** on the war window (§9 method).
5. **Compact-20 challenger** (§5).
6. **Lunar-event feature** from a dated, source-verified table with ±1-day uncertainty.
7. **Blend minimum-gain guard** — re-select with a 1 % threshold; expect the blend to be rejected.
8. **Disaggregation error** — push clustered predictions to city grain and score the reconciliation.

### 14.14 Storyboard for a 10–15 minute presentation

| min | Slide | Chart | Message |
|---|---|---|---|
| 0–1 | The problem | — | Two clocks: search date vs check-in. 60-day booking window; at D-30 only 8.7 % is visible. |
| 1–3 | The formulation | P8 | We predict *remaining* demand: `final = observed + remaining`. The floor is structural. |
| 3–5 | Does it work? | P2, P7 | 0.1479 pooled vs 0.2200 baseline (−32.8 %). All error lives in D-7..D-21. |
| 5–6 | **Honesty slide** | **P5** | One fold overlaps the June 2025 war and drives 46 % of our error. Without it: 0.1062. We did not drop it. |
| 6–8 | **The clustering question** | **P3, P4** | We tested aggregation properly. 90 % of the apparent gain was the metric, not the model. **So we did not cluster.** |
| 8–10 | Where the value is | P9, P6 | Top destinations and momentum — as **search interest**, not bookings. |
| 10–12 | Trust | P10, P15 | Forecast converges D-30→D-1; calibration removes bias without damaging WAPE. |
| 12–14 | Limits | — | 7 provinces, no booking data, no event features, wide CI. What we would build next. |
| 14–15 | Q&A buffer | — | — |

### 14.15 KPI dictionary → §12.

### 14.16 Data gaps for Jabama use cases

**Missing entirely — قابل محاسبه نیست:** bookings, revenue, ADR/price, conversion, inventory/listing
counts, occupancy, cancellations, capacity, marketing spend/impressions, user identity or session data,
supply-side host data, competitor pricing, weather, road status, school-holiday calendar,
official/lunar holiday calendar, and the 24 Iranian provinces absent from this panel.

**Consequence:** every business use case is capped at **descriptive** or **predictive**. **No causal
claim is supportable from these four files.**

### 14.17 Fifteen hard jury questions, with evidence-based answers

1. **"Your WAPE is 0.1479 — is that good?"** Against the strongest simple baseline on identical folds
   (pickup projection, 0.2200) it is 32.8 % better; against last-year-same-date (0.2638) 44 % better.
   But 46 % of that error is one fold overlapping the June 2025 war; excluding it, 0.1062.
2. **"Prove there is no leakage."** Features are read from a `(pair × days_to_checkin)` matrix where
   horizon `h` can only touch columns ≥ `h` — a violation is unrepresentable. 17 tests overwrite every
   post-cutoff row with 10⁶ and require bit-identical predictions.
3. **"Why didn't you use clustering when the submission format asks for `cluster_code`?"** We built it,
   swept 6 levels × 3 windows, and measured that 89.7 % of the apparent 2.05 % gain was offsetting errors
   cancelling inside merged rows. Earned gain: 0.21 %. We submit `cluster_code = city_code`.
4. **"Isn't your clustered WAPE 0.1177 better than 0.1479?"** Different fold sets and different
   calibration — not comparable. On the like-for-like internal comparison the gain is 0.21 %.
5. **"Why is one fold three times worse?"** Its target window (2025-05-22..06-20) overlaps the
   Twelve-Day War (13–24 June 2025). Search volume hit 2.88× baseline on 17 June; final demand hit 2.06×.
   Unforecastable at the 2025-05-21 cutoff.
6. **"Then just drop that fold."** No — selecting folds on outcome. We report both numbers together.
7. **"Do you handle Nowruz and Ramadan?"** Nowruz and Yalda are implicitly available through
   `jalali_month`/`jalali_day` (fixed solar dates). Ramadan and all lunar events are **not** — they drift
   ~11 days/year against the solar calendar, so a fixed Jalali encoding cannot represent them. This is a
   known gap, not a claim.
8. **"Which feature matters most?"** `city_code` (0.293) and `city_hist_mean` (0.244) in the clustered
   model. **Caveat we volunteer:** city-history features are frozen at the fold cutoff, so for training
   rows they partly contain their own outcome. That inflates their apparent value and is our top
   outstanding experiment.
9. **"Is calibration just fitting the test set?"** No — factors come from strictly earlier, fully-closed
   folds. It is accepted only if pooled WAPE does not worsen, |bias| improves, and no fold degrades by
   >0.01. One candidate was rejected on exactly that rule.
10. **"Raw and calibrated — two models?"** One model. The boosters are byte-identical
    (SHA-256 `c0120f47…` both). Calibration scales only the projected remainder.
11. **"Your blend improves WAPE by 0.06 % — is that real?"** No. Best-of-36 combinations on 3 folds,
    against a bootstrap CI of [0.098, 0.222]. We report it as neutral, not as an improvement.
12. **"Can a judge reproduce this?"** Model A yes. Model B **no** as shipped — `scipy` is missing from
    `requirements.txt`. Nothing is pinned either.
13. **"Is this national demand?"** No. Seven provinces: Tehran, Isfahan, Mazandaran, Gilan,
    Khorasan Razavi, Hormozgan, Alborz. Our documents say "national" and that wording is wrong.
14. **"Can you tell us which cities to invest in?"** We can rank **search interest** and momentum. We
    cannot rank bookings, revenue or unmet supply — no such data exists in the four files.
15. **"What would you do with one more week?"** The city-history ablation (F-4.1), a uniform 5-fold
    comparison of all arms, and a source-verified lunar-event calendar. In that order.

### 14.18 Evidence appendix

All commands read-only; none write to the repo.

```bash
git status --short && git log --oneline -3
shasum -a 256 data/raw/pol4/*.csv                      # → matches input_manifest.json
wc -l data/raw/pol4/*.csv                              # physical lines vs data rows
python -m pytest backend/tests -p no:cacheprovider     # 423 passed, 9 skipped
python -m pytest backend/tests/test_pol4_leakage.py backend/tests/test_leakage.py \
                backend/tests/test_pol4_validation.py  # 17 passed
grep -in scipy requirements.txt requirements-optional.txt         # → no match (F-1.1)
grep -c "==" requirements.txt                                     # → 0 (F-1.2)
grep -rn "shap_explainer|anomaly|neural_model|chronos_model" backend/ml/pol4/*.py   # → none
grep -rn "pol4_cluster|results_clustered" frontend/ backend/apps/ # → none (§13)
```

Verification scripts used (in the session scratchpad, not written to the repo):
`audit_raw.py` (§2), `audit_trainset.py` (§3), plus inline one-liners for §4, §7, §8, §9.

Key artifacts read: `artifacts/pol4/{run_summary,backtest_metrics_phase2,validation_audit,input_manifest,feature_importance}.json`,
`artifacts/pol4/model_bundle/model_spec.json`, `artifacts/pol4/model_variants/{raw,calibrated}/*.txt`,
`artifacts/pol4_cluster/{run_summary.json,cluster_sweep.csv,blend_sweep.csv,clusters.csv}`,
`artifacts/pol4_cluster/trainset/{features,target,meta}.parquet` + `manifest.json`,
`artifacts/pol4_nolimit/trainset/*.parquet`.

External source for §9 (Twelve-Day War, 13–24 June 2025, ceasefire 24 June 2025):
[Twelve-Day War — Wikipedia](https://en.wikipedia.org/wiki/Twelve-Day_War) ·
[Twelve-Day War ceasefire — Wikipedia](https://en.wikipedia.org/wiki/Iran%E2%80%93Israel_war_ceasefire) ·
[12-Day War — Britannica](https://www.britannica.com/event/12-Day-War)

### 14.19 NOT VERIFIABLE in this audit

1. Per-horizon / per-date breakdown **within** individual folds — `folds[*]` stores only 6 scalars, so the
   war-window decomposition inside fold `2025-05-21` cannot be computed from artifacts.
2. Per-fold feature-importance stability — the sweep persists no per-fold importances.
3. Integrity of the champion's own trainset — `artifacts/pol4/trainset/` is absent; only recorded hashes exist.
4. Contract metadata of `artifacts/pol4_nolimit/trainset/` — no `manifest.json`.
5. Frontend interactivity, tooltips, RTL, Persian number formatting, accessibility, stale-artifact
   fallbacks — requires running the Next.js app.
6. Disaggregation/reconciliation error of clustered predictions at city grain — never computed.
7. Whether the champion's model/feature selection was truly nested — `validation_audit.json` self-declares
   `"preselected; not nested inside this run"`.
8. Causal attribution of the June 2025 demand break — only temporal coincidence is established.
9. Behaviour under a different LightGBM version — nothing is pinned.
10. Any booking/revenue/occupancy relationship — the data does not exist.

### 14.20 Final decision

# READY WITH CONDITIONS

Conditions, in order:

1. Add `scipy` to `requirements.txt` (**Critical** — Model B otherwise does not run).
2. Disclose the Twelve-Day War contamination wherever 0.1479 appears; present 0.1479 and 0.1062 together.
3. Resolve `results_selected.csv` → model traceability (persist the 149-group arm, or demote the blend).
4. Replace "national" with "7-province" in `README.md`, `docs/POL4_CLUSTERING.md`, `config.py:119`,
   `aggregate.py:28`.
5. Remove any slide claiming SHAP / uncertainty / anomaly / ensembling as part of Pol 4.

The submission itself (`artifacts/pol4/results.csv`) is **valid and needs no change**: 9,630 rows,
321 cities × 30 dates, 0 NaN, 0 negatives, 0 rows below the observed floor.

---
*Read-only audit. No code, data, model, artifact, frontend or document was modified.*
