# POL4 Full Data, Trainset, Model, and Evaluation Audit

Generated: `2026-09-10T11:38:34.919688+00:00`  
Input digest: `fc71f2f11bb1c781e2dff9aae3987b6af0afa3a7d51c43b1ef2dc8a72841636e`  
Competition cutoff: `2025-11-21`  
Target window: `2025-11-22` to `2025-12-21`

## Executive verdict

| Area | Status | Evidence |
|---|---|---|
| Raw inputs | PASS | 4 checksum-recorded sources |
| Trainset contract | PASS | 2,655,312 rows; sampled=False |
| Real-grid features | PASS | 9630 x 66; exact schema; no infinities |
| Temporal validation | PASS WITH LIMITATIONS | 5 walk-forward folds |
| Submission contract | PASS | 9630 rows, 321 cities, 30 dates |
| Saved variants | PASS | raw and calibrated bundles independently checksum-verified |

The two saved variants share the same fitted two-band LightGBM boosters. The raw variant has no post-model calibration; the calibrated variant applies `guarded_bias_horizon` only to predicted remaining demand.

## 1. Raw data audit

| Source | Role | Rows | SHA-256 |
|---|---|---:|---|
| `search_data.csv` | training history | 3,298,564 | `101d310dd844ec0370a9f1d5ded7ffdce379893602ad6c904ede8c2497b44b1a` |
| `evaluation.csv` | target observations | 65,411 | `3afcd5ab84189af74c8e452a62063f91ec6d5aedf0f3d8fbc17e408b5ed29a85` |
| `cities.csv` | city features | 321 | `5aa36d5fba191efa1de78bb56636932f34c126a3a0143d4303cfa91fd227c0b5` |
| `city_code_mapping.csv` | display labels | 321 | `04cefbc0b8bf8ac6dfca04dde1202c88d44e14d6ddf9357484e0f14067e35ffe` |

- Search rows: **3,298,564**.
- Evaluation rows: **65,411**.
- Cities/provinces: **321 / 7**.
- Log-date range: `2023-09-03` to `2025-11-21`.
- Check-in range: `2023-11-01` to `2025-11-21`.
- Maximum observed lead time: **59 days**.
- Loader findings: 3,298,564 rows, no duplicates, no impossible dates, lead time 0-59 days; 65,411 rows, no duplicates, no impossible dates, lead time 0-59 days; 321 of 321 cities named.

## 2. Trainset audit

| Property | Value |
|---|---:|
| Rows | 2,655,312 |
| Candidate rows before sampling | 2,655,312 |
| Sampled | False |
| `max_train_rows` | None |
| Train window | 752 days |
| City history | 752 days |
| Check-ins | 2023-11-01 to 2025-11-21 |
| Sampled horizons | 1, 2, 3, 5, 7, 10, 14, 18, 21, 25, 30 |
| Feature count | 66 |

Target contract:

```text
remaining_demand = max(final_demand - observed_so_far, 0)
model_target = log1p(remaining_demand)
predicted_final = observed_so_far + max(expm1(model_output), 0)
```

Target summary:

```json
{
  "minimum": 0.0,
  "median": 7.0,
  "p90": 828.0,
  "p99": 15313.0,
  "maximum": 216699.0,
  "zero_rows": 972904
}
```

## 3. Real submission-grid feature audit

The production grid was rebuilt independently as **9630 rows x 66 features**.

- Exact ordered train/inference schema: **True**.
- Coverage: **321 cities x 30 dates**, D-1 through D-30.
- Infinite values: **0**.
- Unexpected null columns: **[]**.
- Market aggregates: **10** features.
- Province aggregates: **10** features.
- Core city-history statistics: **5** features.
- Intentional activity nulls: `{'days_since_first_search': 4282, 'days_since_last_search': 4282, 'search_span': 4282}`; these mean no search has occurred yet.

## 4. Saved model variants

| Variant | Calibration | Backtest WAPE | Bias | Forecast total | Bundle |
|---|---|---:|---:|---:|---|
| raw_two_band_lightgbm | `none` | 0.155080 | -0.098551 | 7,977,311 | `/home/parsa/Desktop/workspace/platforms/novo-pulse/artifacts/pol4/model_variants/raw` |
| guarded_calibrated_two_band_lightgbm | `guarded_bias_horizon` | 0.147927 | -0.068189 | 8,352,768 | `/home/parsa/Desktop/workspace/platforms/novo-pulse/artifacts/pol4/model_variants/calibrated` |

Calibration changes the rounded national forecast by **375,457 (+4.71%)** and changes **3,973** of 9,630 city-date rows.

## 5. Evaluation audit

- Selected calibrated WAPE/bias: **0.147927 / -0.068189**.
- Honest prequential WAPE/bias: **0.148618 / -0.085217**.
- Raw train/validation WAPE: **0.068722 / 0.155080**.
- Train-validation gap: **0.086358**.
- 95% fold-cluster interval: **0.097554 to 0.221850**.

### Walk-forward folds

| Cutoff | Calibrated WAPE |
|---|---:|
| 2024-11-21 | 0.121684 |
| 2025-05-21 | 0.272169 |
| 2025-08-21 | 0.114597 |
| 2025-09-22 | 0.092630 |
| 2025-10-22 | 0.095324 |

### Lead-time risk

| Horizon | WAPE | Bias |
|---|---:|---:|
| D-1-3 | 0.029215 | -0.010399 |
| D-4-7 | 0.060656 | -0.006652 |
| D-8-14 | 0.107822 | -0.012869 |
| D-15-21 | 0.124440 | -0.027873 |
| D-22-30 | 0.263073 | -0.180883 |

## 6. Leakage and integrity controls

- Every horizon-h pickup feature reads only days-to-check-in columns >= h.
- Market aggregates pool the same check-in date only.
- Province aggregates pool the same province and check-in date only.
- Validation targets are strictly after each simulated cutoff.
- Calibration for a scored fold uses only earlier, fully closed OOF folds.
- Every saved bundle includes input provenance, native-model checksums, baseline state, and calibrator state.
- Raw and calibrated bundle reloads are checked for prediction parity before publication.

## 7. Material limitations and warnings

1. The 0.0864-class train/validation gap is an overfit or regime-drift warning, although all five folds beat the pickup baseline.
2. Only five temporal folds exist; the confidence interval is consequently wide.
3. The model specification was preselected and is not fully nested inside every outer fold.
4. D-22 to D-30 remains the weakest band and materially underpredicts in historical backtests.
5. City-history statistics are cutoff-safe for validation and production, but training rows currently reuse fold-level city history rather than recomputing it at every row's historical origin.
6. Search demand is not bookings, revenue, occupancy, or causal marketing lift.

## 8. Reproduction commands

```bash
# Train once, evaluate, save both variants, and regenerate this audit
make pol4

# Raw LightGBM without post-model calibration
make pol4-predict-raw

# Selected LightGBM with guarded horizon calibration
make pol4-predict-calibrated

# Tests
make test-pol4
```

## 9. Gate before developing model 3

Model 3 should use a genuinely different forecasting logic and must beat both saved variants on prequential WAPE, fold robustness, long-horizon bias, and stability. It should not replace the current champion merely by improving in-sample fit.
