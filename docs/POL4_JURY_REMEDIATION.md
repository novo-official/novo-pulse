# Pol 4 jury remediation — implementation and evidence

This report follows the independent audit dated 2026-09-10. The original audit
is preserved as a historical record. The confirmed judging format is **seven
minutes presenting plus seven minutes Q&A**.

The submitted `artifacts/pol4/results.csv` is preserved. Improvements below
strengthen reproducibility, isolate methodological questions, expose the
scientific evidence in the product and make the business hypothesis testable.
No hackathon ranking, customer traction or commercial lift is claimed.

## Findings addressed

| Audit finding | Implemented response | Evidence / status |
|---|---|---|
| F-1.1: undeclared SciPy | Explicit core dependency | `requirements.txt`; environment check passes |
| F-1.2: unpinned dependencies | 46-package exact core/transitive snapshot; pinned Playwright and npm lock | `requirements.lock`; Linux/Python 3.13 validation |
| F-3.1: missing champion trainset | Rebuilt from original bundle parameters without refitting submission | 2,655,312 rows; all three original Parquet hashes match exactly |
| F-4.1: fold-frozen city history | Added rolling statistics at each row's own origin; retained legacy behavior for saved bundles | Independent-refit, future-poisoning and cold-start tests; direct 66-feature benchmark |
| F-4.2 / §5: feature redundancy | Compact 22-feature and compact-plus-cluster 25-feature experiments | Scored on the common city grid; not silently promoted |
| F-6.1: incompatible comparison | Six matched-fold arms, plus direct 66-feature history comparison | Every arm scores 321 × 30 rows per fold; same raw / cross-fitted calibration policies |
| F-6: within-fold detail missing | Persisted date/horizon breakdowns and prediction hashes in the new benchmark; added persistence to future champion runs | `comparison.json`; per-fold prediction Parquet and importance CSVs |
| F-7: disaggregation error missing | Allocate cluster remainder by training-only city volume share and score at city grain | Conservation asserted against cluster totals on every fold |
| F-8.2: tiny blend gain | Require greater than 1% relative WAPE improvement | Observed 0.06% candidate rejected; margin is not statistical significance |
| F-8.3: wrong selected-arm trace | Rebuilt city/cluster source bundles, selected city export, added each bundle's checksummed assignment | Reload parity and assignment-tampering tests; future eligible blends persist a source bundle |
| F-9.1: undisclosed shock | Show full and without-fold metrics together; record descriptive weekday and prior-year comparisons | `jury_evidence.json`, `shock_event_study.json`, dashboard, deck |
| F-2.1: national scope overclaim | Corrected primary docs and Pol 4 comments to seven-province/in-panel scope | README, clustering documentation, dashboard labels |
| §10: business value unclear | Destination review queue, named owner, decision gate, CSV export, four-week pilot and economics worksheet | `/jury`; `POL4_BUSINESS_PITCH.md` |
| §13: no clustering UI / unverified UX | Added evidence room and browser checks | Charts, filters, CSV integrity, responsive layout, RTL, model switch and stale-state checks |
| Per-city uncertainty missing | Experimental city-challenger residual bands and forward-only coverage evaluation | Separate target CSV; not attached to the submitted model |

The reconstruction also exposed missing serialized states in the old cluster
city bundle. These were rebuilt, made reloadable and included in the repository's
allowlist. Each arm now carries its own city-to-group assignment, instead of
pointing both models at a shared clustered mapping.

## Six-arm experiment results

All rows below use the same five cutoffs, 600 trees per horizon band, full city
scoring and no automatic promotion. Identity and clustered full models use the
same 69-feature schema. Compact models use 22 features (25 with cluster
composition). The Poisson arm changes the objective and disables log1p.

| Arm | Raw WAPE | Same cross-fitted calibration policy |
|---|---:|---:|
| City, 69 features | 0.155356 | 0.147177 |
| Origin-history city, 69 features | 0.156485 | 0.149160 |
| Compact city, 22 features | 0.164538 | 0.158301 |
| Cluster, reconciled to city grain | 0.156421 | 0.148663 |
| Compact cluster, reconciled to city grain | 0.161829 | 0.154776 |
| Poisson compact city | 0.177166 | 0.173429 |

The full-schema origin variant is slightly worse overall, with mixed fold
results. Neither compactness nor a count-native objective earns promotion here.
The best retrospective 69-feature score is only a small change from the existing
submission's 0.147927; it is not evidence of superiority under a fresh holdout.

These are **exploratory fixed specifications**, not nested model selection.
The cluster cut height came from prior experiments. Raw and calibrated columns
are separate policies; compare like with like. The 69-feature identity model
is distinct from the submitted 66-feature model. The direct 66-feature
comparison is complete and reported in `artifacts/pol4_history66/` and the evidence room:

| Direct 66-feature arm | Raw WAPE | Same cross-fitted calibration |
|---|---:|---:|
| Original fold-history configuration | **0.155080** | **0.147927** |
| City history rebuilt at each origin | **0.156905** | **0.151430** |

The original scores reproduce exactly at the reported precision. Origin history
increases raw WAPE by 1.18% relative and calibrated WAPE by 2.37% relative. This
quantifies the history choice; it does not establish that one model is superior
under a new regime. The submitted model and its stated limitation are retained.

## Uncertainty findings

Nominal 80% empirical residual bands use only earlier, fully closed out-of-fold
errors. No interval is reported on the first fold because no prior OOF residuals
are available.

| Fold | Empirical coverage |
|---|---:|
| 2024-11-21 | Unavailable: no earlier OOF history |
| 2025-05-21 | 70.24% |
| 2025-08-21 | 84.42% |
| 2025-09-22 | 84.76% |
| 2025-10-22 | 81.46% |

The shock-window undercoverage is material. These are experimental diagnostics,
not guaranteed intervals or a deployment-ready uncertainty claim. A separately
named CSV provides city-challenger target bounds and support/fallback labels.
It does not alter `results.csv`.

## What remains unresolved

- **Original model's within-training-origin limitation:** the new city-history
  implementation is isolated and measured. The original submitted model retains
  its documented fold-history behavior. Pickup-curve summaries also remain
  fold-fitted; a fully origin-fitted pipeline would require a separate experiment.
- **Lunar holidays:** no year-specific official Iranian calendar was successfully
  verified. The official calendar site presented a redirect page and the PDF
  fetch failed. No dates were invented. Known-at dates, geographic scope,
  authoritative sources and ±1-day lunar uncertainty are required before adding
  these features. Unexpected shocks are never known-future calendar events.
- **New independent outcome data:** retrospective experiments cannot manufacture
  a fresh holdout or establish nested historical model selection. A future closed
  window is needed for prospective promotion evidence.
- **Business outcomes:** bookings, inventory, conversion, margin, campaign costs,
  a sponsor and real pilot participation are not present. The pilot and commercial
  offer are hypotheses. Search data cannot establish revenue, ROI or supply gaps.
- **Causal event attribution:** weekday and prior-year comparisons are descriptive.
  A broad shock lacks a clean untreated group. No causal war-effect estimate is
  claimed.
- **Production deployment:** the demo is a local historical snapshot. Live ingestion,
  durable decision logging, booking integration and operational SLAs require the
  pilot's deployment and data contracts.

## Reproduce and present

`make pol4-recover-trainset`, `make pol4-event-study`, `make pol4-jury`, and
`make pol4-pitch` rebuild the small supporting deliverables. Raw data and the
checksummed source model are required for training-frame recovery.

`make pol4-jury-experiments` writes a separate reproduction run by default, to
`artifacts/pol4_jury_reproduction`. `make pol4-history-ablation` likewise writes
`artifacts/pol4_history66_reproduction`. This preserves the recorded results
while the current implementation is rerun. Change `JURY_RUN_DIR` or
`HISTORY_RUN_DIR` for another run. A changed checkpoint contract is refused.
The original per-row experiment Parquets are local; hashes, scores, assignments
and importance records travel with the repository.

`artifacts/pol4/pitch.html` is a self-contained seven-slide pitch plus Q&A
appendix. Arrow keys advance, N toggles notes, and P prints. The browser check
also exports `pitch.pdf`. The interactive demo and evidence appendix are at
`http://localhost:3000/jury`.

Validation: **428 backend tests passed, 9 skipped**; targeted tests after
later edits pass; frontend typecheck, lint and production build pass; the browser
workflow passes. The final preflight verifies the dependency lock, submission,
recovered trainset, all three model bundles, selected cluster experiment arm and
source freshness. See `artifacts/pol4/jury_preflight.json`.
