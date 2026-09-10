# POL 4 — Clustering as aggregation (E7, run and decided)

**Run:** 2026-09-10 · `make pol4-cluster` · artefacts in `artifacts/pol4_cluster/`
**Verdict:** submit **unclustered** (`cluster_code = city_code`). Clustering was
built, swept across three windows and measured; the accuracy it appears to buy
is almost entirely an artefact of the metric, not a gain from pooling.

---

## 1. The architectural call

Clustering is a **data-aggregation step**, not a second model. A group of sparse
cities collapses into one pseudo-city carrying a `cluster_code` where a
`city_code` used to be, and every existing module — `pickup.py`, `baseline.py`,
`features.py`, `dataset.py`, `champion.py`, `backtest.py` — runs on the result
unchanged, never knowing whether a row is one real city or twenty-six merged
ones.

```
clustering.py   city profiles (train-only)  ->  per-province Ward dendrograms
                                            ->  cut height  ->  city -> cluster
aggregate.py    that mapping  ->  a Pol4Data of virtual cities
<everything downstream, unmodified>
```

The alternative — a bespoke "cluster model" with its own feature logic — would
have confounded the only question worth asking. If the clustered arm had scored
better, there would be no way to tell whether clustering helped or whether the
second pipeline was simply built better. Here **both arms are the same code**:
the unclustered arm is `assign(cut_height=0)`, the identity partition, which
travels through `aggregate_data` like every other level and provably reproduces
the original panel (`test_identity_partition_reproduces_the_panel`).

### How the aggregation actually works

Not every feature aggregates the same way, so none of them is aggregated at all.
The **raw search log is summed** over cluster members, and everything is then
recomputed from the pooled series:

| Feature family | Treatment |
|---|---|
| `observed_total`, `pickup_*d`, velocity, acceleration | read off the pooled `(pair × days_to_checkin)` tensor — sum of sums, never a mean of member values |
| `city_hist_std/p75/p90`, `city_volatility`, `city_weekend_ratio`, `max_daily_search` | recomputed from the cluster's own daily total. A cluster of five bursty cities can pool *smoother* (bursts land on different days) or *burstier* (Nowruz hits all of them at once); neither is the average of the members |
| `market_*` | untouched — already national |
| `province_*` | untouched — clusters never cross a province, so province totals are bit-identical at every level (`test_market_and_province_blocks_are_untouched_by_clustering`) |
| `lat` / `long` | volume-weighted centroid of the members (the brief's population weighting is unavailable; search volume is the closest proxy the data contains) |
| `city_code` | becomes `cluster_code` = the smallest member's city code |

Three columns are added for cluster rows — `n_cities_in_cluster`,
`cluster_min_member_volume`, `cluster_max_member_share` — which separate "five
roughly-equal small cities pooled together" from "four tiny ones plus one that
is almost a hub". **A singleton row carries `n=1`, its own volume and a share of
1.0**, so both arms are trained on an identical 69-column schema and any WAPE
difference is attributable to the partition rather than to the feature set. The
champion's own 66 columns are unchanged and in the same order; `GROUP_CLUSTER`
sits outside `GROUP_ORDER` so no saved model bundle is disturbed.

### How membership is decided

A per-city profile from **pre-cutoff history only**: seven normalised weekday
shares (the weekly shape), weekend ratio, volatility, p75/mean, p90/mean, the
share of days with zero demand, `log1p(volume)`, and `lat`/`long`. Each block is
z-scored over all 321 cities and divided by √(block width), so the seven-column
weekly shape does not outvote the one-column volatility by being wider. Ward
linkage runs **per province**; a single global cut height then produces a nested
family of partitions.

Two guards:

* **Hubs are protected.** A city holding ≥ 0.1 % of national demand (66 of 321)
  is never merged. Pooling a hub cancels large errors and flatters WAPE without
  being justifiable as sparsity handling — precisely what "excessive aggregation
  is penalised" is aimed at.
* **Train-only.** `ClusterPlan.fit` runs at each fold's own cutoff, so "this
  city is sparse enough to cluster" is never decided with the validation window.
  `test_the_assignment_ignores_everything_after_the_cutoff` corrupts every
  post-cutoff row with garbage and requires an identical dendrogram and an
  identical mapping.

---

## 2. The sweep

The level is a swept parameter, scored on **three** windows — the cut height is a
hyper-parameter chosen on validation performance, and one 45-day holdout would
risk picking whatever suited that slice. Cutoffs `2024-11-21` (the seasonal
analogue of the competition window), `2025-08-21`, `2025-10-22`; LightGBM at the
champion's own spec (two horizon bands, log1p target, 600 trees, 69 features);
the panel is rebuilt and the model retrained at every level.

> **These numbers are not the champion's 0.1479, and must not be quoted against
> it.** This sweep runs three folds, not five — it omits `2025-05-21` (WAPE
> 0.272, the hardest window by far) and `2025-09-22` — and it applies **no
> calibration**, because calibration is fitted per arm and would have varied
> between the levels being compared. Both choices are deliberate: every level in
> the table is scored under an identical budget, so the *comparison between rows*
> is exact. The *absolute level* of the column is not comparable to anything
> outside this table.

Everything is scored as **one global WAPE at the submission grain** — the mix of
city rows and cluster rows that would actually be submitted — because that is
what gets graded. Aggregation moves demand between rows but never creates or
destroys it, so the WAPE **denominator is identical at every level**
(`test_the_wape_denominator_does_not_move_with_the_level`), which is what makes
the curve a like-for-like comparison.

| rows | cut height | demand pooled | **WAPE** | control¹ | mechanical² | modelling³ | 2024-11-21 | 2025-08-21 | 2025-10-22 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **321** | 0.00 | 0.0 % | **0.12017** | — | — | — | 0.12599 | 0.11905 | 0.11834 |
| 283 | 0.15 | 0.3 % | 0.12079 | 0.12001 | +0.00016 | −0.00078 | 0.12215 | 0.12076 | 0.11987 |
| 233 | 0.34 | 1.9 % | 0.12192 | 0.11939 | +0.00078 | −0.00253 | 0.11873 | 0.12351 | 0.12095 |
| 199 | 0.51 | 2.4 % | 0.12129 | 0.11899 | +0.00118 | −0.00230 | 0.12007 | 0.12192 | 0.12087 |
| 153 | 0.86 | 2.7 % | 0.11997 | 0.11848 | +0.00169 | −0.00149 | 0.11787 | 0.12011 | 0.12116 |
| 110 | 1.97 | 2.9 % | **0.11770** | 0.11795 | +0.00222 | **+0.00025** | 0.11760 | 0.11505 | 0.12312 |
| 74 | 20.85 | 3.0 % | 0.11837 | 0.11754 | +0.00263 | −0.00083 | 0.11879 | 0.11571 | 0.12345 |

¹ **control** — the *city-level model's own predictions*, summed onto that
level's rows. Same rows, same outcome, unchanged predictor.
² **mechanical gain** = unclustered − control. Free: merging rows lets
offsetting errors cancel before the absolute value is taken.
³ **modelling gain** = control − clustered. The part that had to be earned by
fitting on the pooled series.

### What the control column settles

Reading the WAPE column alone, clustering to 110 rows looks like a **2.05 %
relative** improvement. The control column says that **90 % of it is the metric
being kinder to coarser rows**: simply summing the unclustered predictions onto
the same 110 rows already gets 0.11795. Training on the pooled series adds
0.00025 — **0.2 % relative** — and at every other level it *subtracts*:
pooling costs the model 40–65 % of its training rows and the smaller model
loses more than the smoother target gains.

The folds also disagree, coherently. `2024-11-21` — the seasonal analogue of
Azar — likes clustering (0.12599 → 0.11760). `2025-10-22`, the window closest to
the competition regime, dislikes it at every level (0.11834 → 0.12312). Neither
dominates, which is exactly the situation a single holdout would have hidden.

### Selection rule

> **Submit the coarsest level whose *modelling* gain is at least 2 % relative;
> unclustered otherwise. Ties go to unclustered.**

The rule was first written against the *total* gain and revised — after the
control arm showed that every level's total gain was dominated by the mechanical
term — to test the part that is not free. The change is recorded here rather
than quietly made: under the naive total-gain rule the run would have selected
110 rows on a 2.05 % gain worth 0.2 % of real modelling. Both outcomes are
reported side by side in `run_summary.json → selection.naive_total_gain_rule`.

**No level passes.** The submission stays at 321 rows.

---

## 3. Shrinkage instead of a hard cutoff

A binary "stay alone or be fully merged" decision has to answer *"why this
threshold and not one city over?"*. Shrinkage does not. The blend is applied to
**remaining** demand, so the Phase 1 floor holds by construction rather than by
clamping (a cluster's observed demand is exactly the sum of its members'):

```
w(c)    = volume(c) / (volume(c) + k)
pred(c) = observed(c) + w(c)·city_remaining(c)
                      + (1 − w(c))·share(c)·cluster_remaining(g)
```

A city that is nobody's cluster-mate keeps `w = 1` whatever `k` is. That matters:
blending a singleton would not be shrinkage at all, it would silently ensemble
two fits of the same series — an earlier version did exactly that and reported a
0.24 % "shrinkage gain" that was entirely the hub cities being ensembled. With
the bug fixed the effect is what it should be: only ever the effect on the
cities the partition actually pools.

The blend is scored at **full city grain — 321 rows, no aggregation penalty at
all**, so it only has to be better, not better by a margin that pays for
conceded rows.

| | WAPE | 2024-11-21 | 2025-08-21 | 2025-10-22 |
|---|---:|---:|---:|---:|
| city model | 0.120168 | 0.125988 | 0.119047 | 0.118344 |
| **+ shrinkage** (`h=0.86`, `k=10⁴`) | **0.120096** | 0.125852 | 0.118966 | 0.118323 |

Better on **3 of 3 folds**, by 0.06 % relative. Consistent, free, and far too
small to call a result. It is written as `results_blended.csv` and selected as
the shipped arm on the stated rule, but the honest summary is *no meaningful
difference*.

---

## 4. Why the ceiling was always low

WAPE is demand-weighted, and this market is extreme: **197 of 321 cities hold
0.5 % of all demand between them**, and the largest set this partition is ever
willing to merge is **3.0 %**. The city model's WAPE on that mergeable slice is
0.245 — roughly twice its global rate, so there is real error there — but it
contributes only ≈ 0.007 of the 0.120 headline. Even predicting those cities
perfectly would buy under 6 % relative, and the mechanisms available (pooling,
shrinkage) recover a small fraction of that.

The repo's earlier position — *the global model already carries `city_code` as a
categorical, which shares information across cities with no aggregation
penalty* — turns out to be right. It is now right **with a Pareto curve behind
it** instead of an assertion, which is the part the brief actually asks for.

---

## 5. Artefacts

| file | what it is |
|---|---|
| `cluster_sweep.csv` / `.json` | the accuracy-versus-aggregation curve, per level and per fold |
| `blend_sweep.csv` | 6 cut heights × 6 shrinkage constants, at city grain |
| `clusters.csv` | membership at the best clustered level (109 rows, 244 cities pooled) |
| `city_profiles.parquet` | the standardised demand-shape profiles the dendrogram was fitted on |
| `trainset/` | **the new aggregated dataset**: 901,648 rows × 69 features over 109 virtual cities |
| `results_clustered.csv` | 3,270 rows — the clustered submission at that level |
| `results_city.csv` | 9,630 rows — the unclustered arm |
| `results_blended.csv` | 9,630 rows — city grain with cluster shrinkage |
| `results_selected.csv` | the arm the rule chose |
| `run_summary.json` | selection, decomposition, assignment, validation of every arm |

Reproduce with `make pol4-cluster` (≈ 24 min). Re-argue the level without
re-fitting anything with
`python -m ml.pol4.cluster_pipeline --from-sweep artifacts/pol4_cluster/cluster_sweep.json --min-gain 0.01`.

## 6. Tests

`backend/tests/test_pol4_clustering.py` — 23 tests pinning: identity round-trip,
demand conservation, province containment, hub protection, denominator
invariance, sum-of-sums for summable features, recomputation (not averaging) of
history statistics, cutoff safety of the assignment, blend endpoints, the
observed floor, singleton non-interference, the gain decomposition, and the
clustered submission contract.
