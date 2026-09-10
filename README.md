# Novo Pulse · Pol 4

**Forecast remaining search interest. Give destination teams an earlier, auditable review queue.**

Novo Pulse forecasts 30 check-in dates across **321 cities in seven provinces**.
The target is search count. Bookings, occupancy, revenue, unmet supply and
marketing lift are not measured by the supplied data.

The product has a Persian demand dashboard at `/` and an English evidence room
at `/jury`, with model comparisons, a destination review queue, CSV export,
source checksums and an interactive pilot economics worksheet.

## Start the demo

Python 3.13 on Linux is the validated lock target; Node dependencies have a
committed `frontend/package-lock.json`. Raw competition files belong in
`data/raw/pol4/`: `search_data.csv`, `evaluation.csv`, `cities.csv`, and
`city_code_mapping.csv`.

```bash
make install-locked
cd frontend && npm ci && cd ..
make migrate
make pol4-jury
make dev
```

Open `http://localhost:3000/jury`. The forecast is a **historical competition
snapshot**, with cutoff **2025-11-21**, not live demand for today. The application
runs locally after setup; no paid model API is required.

For another Python/platform combination, use `make install`, validate the full
suite, then regenerate the environment snapshot with
`.venv/bin/python scripts/lock_requirements.py`. The lock records the tested
core dependency closure; it does not claim universal binary reproducibility.

## Verified competition result

| Measure | Submitted model | Pickup baseline |
|---|---:|---:|
| Pooled WAPE, all five folds | **0.147927** | **0.219991** |
| Normalised bias | −0.068189 | See generated evidence |
| Relative WAPE improvement | **32.8%** | Reference |

**Always disclose the difficult fold alongside this headline:** the target
window after cutoff `2025-05-21` overlaps the **June 13–24, 2025 Iran–Israel war**.
It contributes approximately **46.3% of absolute error**. Excluding that entire
fold gives **0.10615 WAPE**, a diagnostic only. The headline retains all five
folds. The temporal association does not establish causal attribution.
[UN event account](https://dppa.un.org/en/node/101504).

The 95% fold bootstrap interval for headline WAPE is **[0.097554, 0.221850]**.
Calibration selection using only earlier completed folds scores **0.148618**.
Model and feature choices were preselected, not nested within this run.

Metrics are generated from
[`backtest_metrics_phase2.json`](artifacts/pol4/backtest_metrics_phase2.json)
and [`jury_evidence.json`](artifacts/pol4/jury_evidence.json). The original
[independent audit](artifacts/pol4/PROJECT_JURY_AUDIT.md) is retained unchanged;
[the remediation report](docs/POL4_JURY_REMEDIATION.md) records subsequent work.

## The model and its boundaries

```text
forecast final searches = observed searches + max(0, forecast remaining searches)
```

Two LightGBM regressors cover horizons 1–14 and 15–30. Each uses 600 trees and
an L1 objective on log1p remaining demand. Calibration scales only the remainder,
so it preserves the observed floor. Raw and calibrated variants share boosters.

Pickup features read only searches available at the forecast origin. The
submitted model's city-history and pickup-curve summaries are fitted at the
fold cutoff. In historical training rows, city-history summaries include later
completed outcomes within that fold. **Fold-cutoff safety is not the same as
per-training-origin safety.** The origin-history challenger directly measures
this issue without silently altering the submitted model. The direct 66-feature comparison
reproduces the original scores exactly; origin-history calibrated WAPE is
**0.151430**, versus **0.147927** with the original fold-history setup. The
original result retains its limitation; the new challenger is reported separately.

The submission remains [`artifacts/pol4/results.csv`](artifacts/pol4/results.csv):
9,630 rows, 321 cities × 30 dates, finite and nonnegative predictions, no duplicate
keys, and no forecast below the observed count.

## What was implemented after the audit

- Declared SciPy and recorded exact versions for core dependencies and their
  installed transitive dependencies in `requirements.lock`.
- Reconstructed the champion trainset. **All three original Parquet hashes
  match exactly**. Small manifests travel with the repository; full training
  frames remain local.
- Enforced a **greater than 1% relative WAPE gain** for blend eligibility.
  Rebuilt cluster experiment bundles and selected the city arm. Any future
  accepted blend persists its source model and assignment.
- Implemented origin-specific rolling city history and a compact 22-feature
  challenger, plus a Poisson-objective challenger.
- Added a uniform five-fold audit experiment runner. Six arms use the same city
  target grid; cluster remainders are allocated with training-only shares and
  conservation is asserted before scoring. Both raw and the same cross-fitted
  calibration policy are reported. Per-fold dates, horizons, importance and
  prediction hashes are persisted.
- Added experimental residual bands with forward-only coverage evaluation.
  These belong to the named challenger and are **not deployed on the submitted
  model**. Dependent rows and regime shifts prevent a coverage guarantee.
- Added `/jury`: shock disclosure, clustering controls, downloadable destination
  reviews, pilot plan, an assumptions-only economics worksheet, and stale-source
  rejection.

## Why clustering is not the submission

Within the original **three-fold, uncalibrated** sweep, best clustered WAPE was
0.117699 versus 0.120168 for the city model. But the city model's predictions,
summed onto those same groups, score 0.117952: about 90% of the apparent gain is
error cancellation after aggregation. Earned modelling gain is only 0.21%
relative. That is insufficient to justify pooling the submission.

**Do not compare these three-fold scores with the five-fold headline above.**
The new common-grid challenger report provides a separate comparison. Its
specifications and cluster cut height are exploratory, not nested selection.
See [clustering methodology](docs/POL4_CLUSTERING.md).

## Reproduce and inspect

```bash
make pol4                         # Full champion pipeline
make pol4-predict                 # Reload and verify existing calibrated bundle
make pol4-recover-trainset        # Rebuild training data; compare original hashes
make pol4-cluster                 # Original sweep plus guarded selection
make pol4-jury-experiments        # Six arms × five folds, 600 trees per band
make pol4-history-ablation        # Direct 66-feature origin-history comparison
make pol4-uncertainty             # Forward-only experimental interval coverage
make pol4-jury                    # Refresh presentation evidence and review CSV
make pol4-pitch                   # Offline 7-minute deck and Q&A appendix
make test                        # Backend tests, frontend typecheck/lint/build
make test-jury-ui                 # Browser checks; both servers must be running
```

Reproduction runs write to `artifacts/pol4_jury_reproduction` and
`artifacts/pol4_history66_reproduction` by default, preserving the recorded
benchmarks. Override `JURY_RUN_DIR` / `HISTORY_RUN_DIR` as needed.

The full challenger run takes several minutes per fold on a CPU and needs
roughly 8 GB RAM. It checkpoints predictions and hashes. A changed experiment
contract is rejected; use `--output artifacts/another_experiment` for new settings.
Run `PYTHONPATH=backend .venv/bin/python -m ml.pol4.jury_experiments --help` for
arm and cutoff options. Reduced runs are exploratory smoke tests.

| Artifact | Purpose |
|---|---|
| `artifacts/pol4/results.csv` | Competition submission |
| `artifacts/pol4/model_bundle/` | Checksum-protected selected model |
| `artifacts/pol4/trainset_manifest.json` | Recovered training contract and hashes |
| `artifacts/pol4/trainset_recovery.json` | Original-versus-rebuilt comparison |
| `artifacts/pol4/jury_evidence.json` | Derived presentation metrics and source hashes |
| `artifacts/pol4/decision_queue.csv` | Analyst review queue with reasons and decision gates |
| `artifacts/pol4_jury/comparison.json` | Common-grid audit challenger results |
| `artifacts/pol4_jury/uncertainty.json` | Empirical coverage on later folds |
| `artifacts/pol4_cluster/run_summary.json` | Original sweep and current guarded selection |

## Business case

Initial user: a destination growth analyst. Sponsor hypothesis: the destination
growth lead. Start with planning efficiency and review quality, then validate
commercial outcomes using joined bookings, inventory, contribution margin and
campaign spend. Search interest alone cannot establish an investment return.

The [pilot and pitch pack](docs/POL4_BUSINESS_PITCH.md) specifies the buyer
hypothesis, workflow, four-week validation, success criteria, commercial offer,
data contract and demo script. All pricing and outcome assumptions must be
validated with a buyer; no TAM, customer traction or revenue lift is claimed.

## API

The frontend uses read-only `/api/v1/pol4/` endpoints:
`dashboard/`, `overview/`, `forecast/`, `cities/`, `heatmap/`, `stability/`,
`model-performance/`, `reports/`, **`jury/`**, and **`decision-queue.csv`**.
Endpoints read generated artifacts and do not train models or scan raw logs.
The evidence endpoint refuses a packet whose source checksums no longer match.

## Remaining constraints

The four competition files contain no bookings, prices, inventory, occupancy,
conversion or campaign spend. A business pilot needs those joins and a real
operator. There is no verified Iranian lunar-event table in the project, and
unexpected shocks must not be encoded as known future events. No per-city
predictive coverage guarantee is claimed.

Generic platform modules for neural models, Chronos, SHAP and anomaly detection
are separate from the Pol 4 submission. Their existence is not evidence that
those features were used here.
