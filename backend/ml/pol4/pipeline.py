"""The Pol 4 pipeline: one command from the raw CSVs to a validated results.csv.

    PYTHONPATH=backend python -m ml.pol4.pipeline

Runs, in order:

    load + validate  -> pickup curves at the competition cutoff
                     -> walk-forward comparison: champion vs baseline vs
                        calibration (and the full feature ladder with --ablation)
                     -> forecast stability across the D-30 -> D-1 ladder
                     -> final Azar 1404 inference from evaluation.csv
                     -> write results.csv -> validate it, loudly

`--model baseline` runs the Phase 1 pickup projection alone: fifteen seconds,
no model fitting, and the fallback if anything about the champion misbehaves on
the day.

Nothing here reads a log date after the cutoff it is standing at, and no source
file has to be edited to run it.
"""
from __future__ import annotations

import argparse
import json
import logging
import platform
import random
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import analytics as analytics_module
from . import stability as stability_module
from .audit import audit_inference_features, build_full_audit_markdown, write_full_audit
from .backtest import run_backtest
from .baseline import PickupBaseline
from .calibration import Calibrator
from .champion import ChampionSpec, FittedChampion
from .config import Pol4Config
from .artifacts import build_input_manifest, write_json_atomic, write_parquet_atomic, sha256_file
from .dataset import (
    build_training_frame,
    load_materialized_training_frame,
    materialize_training_frame,
)
from .experiments import (
    ExperimentResult,
    ExperimentSpec,
    baseline_predictor,
    calibrated_predictor,
    cross_fit_calibration,
    fit_oof_calibrator,
    gbdt_predictor,
    run_suite,
    staged_groups,
)
from .loader import CHECKIN, CITY, load_pol4
from .submission import (
    build_grid,
    build_named_submission,
    build_submission,
    validate_submission,
    write_submission,
)
from .validation import build_validation_audit

log = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _json_ready(value: Any) -> Any:
    """Make numpy / pandas scalars JSON-serialisable without losing precision."""
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float):
        return None if not np.isfinite(value) else value
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if value is pd.NaT or (isinstance(value, float) and pd.isna(value)):
        return None
    return value


def run(
    config: Pol4Config | None = None,
    *,
    model: str = "champion",
    ablation: bool = False,
    skip_backtest: bool = False,
    skip_stability: bool = False,
    reuse_trainset: bool = False,
    spec: ChampionSpec | None = None,
) -> dict[str, Any]:
    """Load, evaluate, forecast Azar 1404, write and validate results.csv.

    `model="baseline"` runs the Phase 1 pickup projection alone - fast, and the
    fallback if anything about the champion looks wrong on the day.
    """
    config = config or Pol4Config()
    spec = spec or ChampionSpec()
    set_seed(config.seed)
    started = time.perf_counter()
    config.artifacts_dir.mkdir(parents=True, exist_ok=True)

    # -- 1. load ------------------------------------------------------------
    log.info("loading Pol 4 datasets from %s", config.raw_dir)
    data = load_pol4(config)
    input_manifest = build_input_manifest(config, data)
    input_manifest_path = write_json_atomic(
        input_manifest, config.artifacts_dir / "input_manifest.json"
    )
    summary: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "seed": config.seed,
        "model": model,
        "cutoff": config.cutoff.date().isoformat(),
        "target_window": [
            config.target_start.date().isoformat(),
            config.target_end.date().isoformat(),
        ],
        "data": data.summary(),
        "provenance": {
            "input_manifest": str(input_manifest_path),
            "input_digest": input_manifest["input_digest"],
            "allowed_sources": input_manifest["allowed_sources"],
        },
    }

    # -- 2. pickup curves at the competition cutoff -------------------------
    baseline = PickupBaseline.fit(data, config.cutoff, config)
    curves = baseline.curves.to_frame()
    city_names = data.city_name_of
    province_names = data.cities.drop_duplicates("province_code").set_index("province_code")
    province_names = (
        province_names["province"] if "province" in province_names.columns else None
    )
    curves["name"] = [
        city_names.get(key, str(key))
        if level == "city"
        else (province_names.get(key, str(key)) if level == "province" and province_names is not None else level)
        for level, key in zip(curves["level"], curves["key"])
    ]
    curves.to_parquet(config.artifacts_dir / "pickup_curves.parquet", index=False)
    summary["baseline"] = baseline.summary()
    champion_calibrator = Calibrator(method="none", alpha_global=1.0)
    raw_champion_predictions = pd.DataFrame()
    phase2_payload: dict[str, Any] = {}
    feature_audit: dict[str, Any] = {}
    model_comparison: dict[str, Any] = {}

    # -- 3. backtest --------------------------------------------------------
    if skip_backtest:
        summary["backtest"] = {"skipped": True}
    elif model == "baseline":
        log.info("running the pickup-baseline backtest")
        backtest = run_backtest(data, config)
        # Only the baseline path writes this; the champion path's equivalent is
        # backtest_metrics_phase2.json, which carries the baseline inside it.
        (config.artifacts_dir / "backtest_metrics.json").write_text(
            json.dumps(_json_ready(backtest), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        summary["backtest"] = {
            "pooled_wape": backtest["pooled"]["overall"]["wape"],
            "pooled_normalised_bias": backtest["pooled"]["overall"]["normalised_bias"],
            "folds": {f["cutoff"]: f["overall"]["wape"] for f in backtest["folds"]},
        }
    else:
        specs = _experiment_specs(config, spec, ablation)
        log.info(
            "scoring %d predictor(s) across %d walk-forward folds",
            len(specs),
            len(config.backtest_cutoffs),
        )
        results = run_suite(data, config, specs)
        raw_champion = next(r for r in results if r.stage == "champion_raw")
        calibration_trials = []
        for method in ("horizon_shrunk", spec.calibration, "bias_horizon_shrunk"):
            candidate, fitted_calibrator = cross_fit_calibration(
                raw_champion, config, method=method
            )
            fold_safe = all(
                candidate.folds[label]["wape"]
                <= raw_champion.folds[label]["wape"] + 0.01
                for label in raw_champion.folds
            )
            safe = (
                candidate.pooled["wape"] <= raw_champion.pooled["wape"]
                and abs(candidate.pooled["normalised_bias"])
                < abs(raw_champion.pooled["normalised_bias"])
                and fold_safe
            )
            candidate.stage = "calibration_candidate"
            candidate.meta["safe"] = safe
            calibration_trials.append((candidate, fitted_calibrator, safe))

        eligible = [trial for trial in calibration_trials if trial[2]]
        selected_trial = (
            min(eligible, key=lambda trial: abs(trial[0].pooled["normalised_bias"]))
            if eligible
            else None
        )
        calibration_accepted = selected_trial is not None
        results.extend(trial[0] for trial in calibration_trials)
        if calibration_accepted:
            selected_candidate, champion_calibrator, _ = selected_trial
            selected_candidate.stage = "champion"
        else:
            results.append(
                replace(
                    raw_champion,
                    name=f"CHAMPION selected raw {spec.kind}",
                    stage="champion",
                    meta={
                        **raw_champion.meta,
                        "calibration": "none",
                        "calibration_rejected": spec.calibration,
                    },
                )
            )
        best_bias_candidate = min(
            (trial[0] for trial in calibration_trials),
            key=lambda candidate: abs(candidate.pooled["normalised_bias"]),
        )
        selected_champion = next(r for r in results if r.stage == "champion")
        baseline_result = next(r for r in results if r.name.startswith("E0"))
        validation_audit = build_validation_audit(
            raw_champion,
            selected_champion,
            [trial[0] for trial in calibration_trials],
            baseline_result,
            config,
        )
        write_json_atomic(
            _json_ready(validation_audit), config.artifacts_dir / "validation_audit.json"
        )
        raw_champion_predictions = raw_champion.predictions
        experiments = _write_experiments(results, config)
        payload = _write_phase2_backtest(
            results, config, data, validation_audit=validation_audit
        )
        phase2_payload = payload

        champion_scores = payload["champion"]
        summary["backtest"] = {
            "pooled_wape": champion_scores["pooled"]["wape"],
            "pooled_normalised_bias": champion_scores["pooled"]["normalised_bias"],
            "folds": {k: v["wape"] for k, v in champion_scores["folds"].items()},
            "baseline_wape": payload["baseline"]["pooled"]["wape"],
            "relative_improvement": round(
                1 - champion_scores["pooled"]["wape"] / payload["baseline"]["pooled"]["wape"], 4
            ),
            "experiments": experiments["n_experiments"],
            "calibration_accepted": calibration_accepted,
            "calibration_method": champion_calibrator.method,
            "raw_wape": raw_champion.pooled["wape"],
            "raw_normalised_bias": raw_champion.pooled["normalised_bias"],
            "calibrated_candidate_wape": best_bias_candidate.pooled["wape"],
            "calibrated_candidate_normalised_bias": best_bias_candidate.pooled[
                "normalised_bias"
            ],
            "calibration_candidates": [
                {
                    "method": trial[0].meta["calibration"],
                    "wape": trial[0].pooled["wape"],
                    "normalised_bias": trial[0].pooled["normalised_bias"],
                    "safe": trial[2],
                }
                for trial in calibration_trials
            ],
            "validation_audit": validation_audit,
        }

    # -- 4. stability -------------------------------------------------------
    if skip_stability or model == "baseline":
        summary["stability"] = {"skipped": True}
    else:
        log.info("measuring forecast stability across the D-30 -> D-1 ladder")
        summary["stability"] = _write_stability(
            data,
            spec,
            config,
            raw_champion_predictions,
            champion_calibrator.method,
        )

    # -- 5. final inference at the competition cutoff -----------------------
    target_dates = config.target_dates()
    if model == "baseline":
        predictions = baseline.predict(build_grid(data, config.cutoff, target_dates))
        summary["champion"] = {"model": "pickup_baseline", **baseline.summary()}
    else:
        train_dir = config.artifacts_dir / "trainset"
        if reuse_trainset:
            manifest_path = train_dir / "manifest.json"
            if not manifest_path.is_file():
                raise ValueError(f"--reuse-trainset requested but {manifest_path} is missing")
            train_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if train_manifest.get("input_digest") != input_manifest["input_digest"]:
                raise ValueError("materialised train set was built from different input CSVs")
            expected_contract = {
                "seed": config.seed,
                "cutoff": pd.Timestamp(config.cutoff).isoformat(),
                "train_window_days": config.train_window_days,
                "city_history_days": config.city_history_days,
                "city_history_mode": config.city_history_mode,
                "max_train_rows": config.max_train_rows,
                "horizons": list(config.train_horizons),
                "feature_groups": list(spec.groups),
            }
            mismatched = [
                key for key, expected in expected_contract.items()
                if train_manifest.get(key) != expected
            ]
            if mismatched:
                raise ValueError(
                    "materialised train set has a different training contract: "
                    + ", ".join(mismatched)
                )
            train = load_materialized_training_frame(train_dir)
            log.info("reusing %s materialised training rows", f"{len(train):,}")
        else:
            train = build_training_frame(data, config.cutoff, spec.groups, baseline, config)
            train_manifest = materialize_training_frame(
                train,
                train_dir,
                spec.groups,
                config,
                input_digest=input_manifest["input_digest"],
            )
            log.info("materialised %s training rows at %s", f"{len(train):,}", train_dir)

        fitted = FittedChampion.fit(
            data,
            config.cutoff,
            spec,
            config,
            baseline=baseline,
            train=train,
            calibrator=champion_calibrator,
        )
        raw_fitted = replace(
            fitted,
            spec=replace(fitted.spec, calibration="none"),
            calibrator=Calibrator(method="none", alpha_global=1.0),
        )
        variants = {
            "raw": raw_fitted,
            "calibrated": fitted,
        }
        variant_predictions: dict[str, pd.DataFrame] = {}
        variant_manifests: dict[str, dict[str, Any]] = {}
        variant_directories: dict[str, Path] = {}
        for variant, fitted_variant in variants.items():
            before_serialisation = fitted_variant.predict(target_dates, data)
            variant_dir = config.artifacts_dir / "model_variants" / variant
            manifest = fitted_variant.save(
                variant_dir, input_digest=input_manifest["input_digest"]
            )
            restored = FittedChampion.load(variant_dir)
            after_serialisation = restored.predict(target_dates, data)
            pd.testing.assert_frame_equal(
                before_serialisation,
                after_serialisation,
                check_exact=False,
                rtol=1e-12,
                atol=1e-12,
            )
            variant_predictions[variant] = after_serialisation
            variant_manifests[variant] = manifest
            variant_directories[variant] = variant_dir

        # Keep the original path as a backwards-compatible alias for the
        # selected calibrated model. Explicit comparisons use model_variants/.
        model_dir = config.artifacts_dir / "model_bundle"
        model_manifest = fitted.save(model_dir, input_digest=input_manifest["input_digest"])
        legacy_restored = FittedChampion.load(model_dir)
        predictions = variant_predictions["calibrated"]
        pd.testing.assert_frame_equal(
            predictions,
            legacy_restored.predict(target_dates, data),
            check_exact=False,
            rtol=1e-12,
            atol=1e-12,
        )
        feature_audit = audit_inference_features(
            data, baseline, config, spec.groups
        )
        importance = fitted.importance()
        importance.to_json(
            config.artifacts_dir / "feature_importance.json", orient="records", indent=2
        )
        summary["champion"] = {
            **spec.describe(),
            "calibration": fitted.calibrator.as_dict(),
            "training_rows": fitted.training_rows,
            "trainset": {
                "path": str(train_dir),
                "rows": train_manifest["rows"],
                "input_digest": train_manifest["input_digest"],
                "files": train_manifest["files"],
                "reused": reuse_trainset,
            },
            "model_bundle": {
                "path": str(model_dir),
                "bundle_digest": model_manifest["bundle_digest"],
                "input_digest": model_manifest["input_digest"],
                "files": model_manifest["files"],
                "load_parity_verified": True,
            },
            "model_variants": {
                variant: {
                    "path": str(variant_directories[variant]),
                    "bundle_digest": variant_manifests[variant]["bundle_digest"],
                    "calibration": variants[variant].calibrator.method,
                    "load_parity_verified": True,
                }
                for variant in variants
            },
            "real_grid_feature_audit": feature_audit,
            "top_features": importance.head(15).to_dict(orient="records"),
        }

    # -- 6. results.csv, then validate it -----------------------------------
    submission = build_submission(predictions)
    results_path = write_submission(submission, config.artifacts_dir / "results.csv")
    report = validate_submission(submission, data.city_codes, config, raise_on_error=True)

    # The scored file keeps the numeric cluster_code the brief asks for; the
    # readable companion carries the same rows with city and province names.
    named_path = write_submission(
        build_named_submission(predictions, data), config.artifacts_dir / "results_named.csv"
    )

    if model != "baseline":
        variant_submissions = {
            variant: build_submission(frame)
            for variant, frame in variant_predictions.items()
        }
        variant_paths: dict[str, Path] = {}
        for variant, frame in variant_submissions.items():
            validate_submission(frame, data.city_codes, config, raise_on_error=True)
            variant_paths[variant] = write_submission(
                frame, config.artifacts_dir / f"results_{variant}.csv"
            )
            write_submission(
                build_named_submission(variant_predictions[variant], data),
                config.artifacts_dir / f"results_{variant}_named.csv",
            )

        comparison_frame = variant_submissions["raw"].rename(
            columns={"predicted_demand": "raw_predicted_demand"}
        )
        comparison_frame["calibrated_predicted_demand"] = variant_submissions[
            "calibrated"
        ]["predicted_demand"]
        comparison_frame["calibration_delta"] = (
            comparison_frame["calibrated_predicted_demand"]
            - comparison_frame["raw_predicted_demand"]
        )
        comparison_path = config.artifacts_dir / "model_comparison.csv"
        comparison_frame.to_csv(comparison_path, index=False)

        raw_total = int(variant_submissions["raw"]["predicted_demand"].sum())
        calibrated_total = int(
            variant_submissions["calibrated"]["predicted_demand"].sum()
        )
        model_comparison = {
            "raw": {
                "name": "raw_two_band_lightgbm",
                "calibration": "none",
                "backtest_wape": summary["backtest"].get("raw_wape"),
                "normalised_bias": summary["backtest"].get("raw_normalised_bias"),
                "forecast_total": raw_total,
                "results_path": str(variant_paths["raw"]),
                "bundle_path": str(variant_directories["raw"]),
                "bundle_digest": variant_manifests["raw"]["bundle_digest"],
            },
            "calibrated": {
                "name": "guarded_calibrated_two_band_lightgbm",
                "calibration": fitted.calibrator.method,
                "backtest_wape": summary["backtest"].get("pooled_wape"),
                "normalised_bias": summary["backtest"].get("pooled_normalised_bias"),
                "forecast_total": calibrated_total,
                "results_path": str(variant_paths["calibrated"]),
                "bundle_path": str(variant_directories["calibrated"]),
                "bundle_digest": variant_manifests["calibrated"]["bundle_digest"],
            },
            "delta": {
                "forecast_total": calibrated_total - raw_total,
                "percent": (
                    100.0 * (calibrated_total - raw_total) / raw_total
                    if raw_total
                    else None
                ),
                "changed_rows": int((comparison_frame["calibration_delta"] != 0).sum()),
            },
            "comparison_path": str(comparison_path),
            "horizon_backtest": phase2_payload.get("champion", {}).get(
                "by_horizon_bucket", []
            ),
        }
        write_json_atomic(
            _json_ready(model_comparison),
            config.artifacts_dir / "model_comparison_summary.json",
        )
        summary["model_variants"] = model_comparison

    summary["submission"] = report.as_dict()
    summary["submission"]["path"] = str(results_path)
    summary["submission"]["named_path"] = str(named_path)

    city_totals = submission.groupby("cluster_code")["predicted_demand"].sum()
    silent = city_totals.index[city_totals == 0]
    historical = data.search.groupby(CITY)["search_count"].sum()
    unrounded = predictions.groupby(CITY)["predicted_demand"].sum()
    summary["inference"] = {
        "observed_total": float(predictions["observed"].sum()),
        "predicted_total": float(predictions["predicted_demand"].sum()),
        "pairs_with_no_observation": int((predictions["observed"] <= 0).sum()),
        "cities_with_no_observation": int(
            predictions.groupby(CITY)["observed"].sum().le(0).sum()
        ),
        # Some cities round to zero across the whole window. That is a measured
        # result, not an assumption: each still receives a positive prediction,
        # but its historical demand is below half a search per night.
        "top_cities": [
            {"city": name, "city_code": int(code), "predicted_demand": int(total)}
            for code, name, total in zip(
                city_totals.sort_values(ascending=False).head(10).index,
                data.name(city_totals.sort_values(ascending=False).head(10).index),
                city_totals.sort_values(ascending=False).head(10).to_numpy(),
            )
        ],
        "cities_rounding_to_zero": {
            "count": int(len(silent)),
            "cities": sorted(data.name(silent).tolist())[:20],
            "median_historical_demand": float(historical.reindex(silent).fillna(0.0).median())
            if len(silent)
            else None,
            "median_unrounded_prediction": float(unrounded.reindex(silent).fillna(0.0).median())
            if len(silent)
            else None,
            "note": (
                "these cities receive a positive prediction; it rounds to zero because "
                "their measured demand is below half a search per night"
            ),
        },
    }
    # -- 7. dashboard analytics --------------------------------------------
    # Pre-aggregated so the product never reads search_data.csv at request time.
    bundle = analytics_module.build(data, predictions, config)
    summary["analytics"] = {
        "artefacts": bundle.write(config.artifacts_dir),
        "provinces": len(bundle.provinces),
        "cities_with_momentum": int(bundle.city_momentum["pickup_ratio"].notna().sum()),
        "history_days": int(bundle.city_history[CHECKIN].nunique()),
    }

    # A compact, judge-facing contract: what was trained, on which exact data,
    # how it was evaluated, and what it must not be interpreted as.
    model_card = {
        "name": "pol4_remaining_demand_champion",
        "task": "predict final search-based demand per city and check-in date",
        "data_cutoff": config.cutoff.date().isoformat(),
        "target_window": summary["target_window"],
        "input_digest": input_manifest["input_digest"],
        "raw_sources": input_manifest["allowed_sources"],
        "target_definition": "remaining_demand = max(final_demand - observed_so_far, 0)",
        "algorithm": summary["champion"],
        "validation": {
            "method": "five leakage-safe walk-forward simulated competition folds"
            if model != "baseline"
            else "walk-forward pickup-baseline backtest",
            **summary["backtest"],
        },
        "safety_invariants": [
            "features at horizon h only read searches available by checkin - h",
            "predicted_demand is finite and non-negative",
            "predicted_demand is never below observed_so_far",
            "all 321 cities and all 30 target dates are emitted exactly once",
        ],
        "limitations": [
            "demand means observed search volume, not bookings or causal travel intent",
            "the model is validated on historical temporal folds from the supplied dataset",
            "the model specification was preselected; validation_audit.json reports this "
            "selection limitation and a conservative prequential estimate",
            "city_code is the competition city identifier; no statistical city clustering is used",
        ],
        "commands": {
            "train_and_evaluate": "make pol4",
            "inference_without_training": "make pol4-predict",
            "tests": "make test-pol4",
        },
    }
    model_card_path = write_json_atomic(model_card, config.artifacts_dir / "model_card.json")
    summary["model_card"] = str(model_card_path)

    if model != "baseline":
        audit_markdown = build_full_audit_markdown(
            summary,
            input_manifest,
            train_manifest,
            feature_audit,
            model_comparison,
        )
        audit_path = write_full_audit(
            audit_markdown, config.artifacts_dir / "FULL_AUDIT.md"
        )
        summary["full_audit"] = str(audit_path)

    summary["elapsed_seconds"] = round(time.perf_counter() - started, 2)

    (config.artifacts_dir / "run_summary.json").write_text(
        json.dumps(_json_ready(summary), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info("results.csv written to %s (valid=%s)", results_path, report.valid)
    return summary


# ------------------------------------------------------------------ phase 2
def _experiment_specs(config: Pol4Config, spec: ChampionSpec, ablation: bool) -> list:
    """The comparison every accepted change had to survive.

    The baseline and both calibration variants always run, because the champion
    has to be shown beating them rather than asserted to. The full feature
    ladder is opt-in: it is the expensive part and its conclusions are already
    recorded in experiments.csv.
    """
    specs = [
        ExperimentSpec("E0 pickup baseline", "baseline", baseline_predictor),
        ExperimentSpec(
            "E1 calibrated baseline (global)", "calibration", calibrated_predictor("global")
        ),
        ExperimentSpec(
            "E1 calibrated baseline (horizon, shrunk)",
            "calibration",
            calibrated_predictor("horizon_shrunk"),
        ),
    ]
    if ablation:
        for index, (name, groups) in enumerate(staged_groups(), start=2):
            specs.append(
                ExperimentSpec(
                    f"E{index} +{name}",
                    "ablation",
                    gbdt_predictor("lightgbm", groups, params=config.ablation_params),
                    {"model": "lightgbm", "groups": groups},
                )
            )
    specs.append(
        ExperimentSpec(
            f"CHAMPION raw {spec.kind}",
            "champion_raw",
            gbdt_predictor(
                spec.kind,
                spec.groups,
                log1p=spec.log1p,
                params=spec.params,
                calibration="none",
                blend=None if spec.blend >= 1.0 else spec.blend,
                bands=spec.bands,
                measure_training=True,
            ),
            {"model": spec.kind, "groups": spec.groups},
        )
    )
    return specs


def _write_experiments(results: list[ExperimentResult], config: Pol4Config) -> dict[str, Any]:
    rows = pd.DataFrame([r.row() for r in results]).sort_values("wape", ignore_index=True)
    rows.to_csv(config.artifacts_dir / "experiments.csv", index=False)

    best = min(results, key=lambda r: r.pooled["wape"])
    baseline = next((r for r in results if r.name.startswith("E0")), None)
    summary = {
        "n_experiments": len(results),
        "best": best.name,
        "best_wape": best.pooled["wape"],
        "baseline_wape": baseline.pooled["wape"] if baseline else None,
        "relative_improvement": (
            round(1 - best.pooled["wape"] / baseline.pooled["wape"], 4) if baseline else None
        ),
        "experiments": [
            {
                "name": r.name,
                "stage": r.stage,
                "pooled": r.pooled,
                "folds": r.folds,
                "horizon_bucket": r.horizon,
                "runtime_seconds": round(r.runtime_seconds, 1),
                "training": r.training or None,
                "validation_minus_train_wape": (
                    round(r.pooled["wape"] - r.training["wape"], 6)
                    if r.training
                    else None
                ),
                "meta": {k: list(v) if isinstance(v, tuple) else v for k, v in r.meta.items()},
            }
            for r in results
        ],
    }
    (config.artifacts_dir / "experiment_summary.json").write_text(
        json.dumps(_json_ready(summary), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def _name_segments(detail: dict[str, Any], data) -> dict[str, Any]:
    """Swap province codes for province names in a breakdown block."""
    named = dict(detail)
    provinces = data.cities.drop_duplicates("province_code").set_index("province_code")
    if "province" in provinces.columns:
        lookup = provinces["province"]
        named["by_province"] = [
            {**row, "key": lookup.get(row["key"], row["key"])}
            for row in detail.get("by_province", [])
        ]
    return named


def _write_phase2_backtest(
    results: list[ExperimentResult],
    config: Pol4Config,
    data,
    *,
    validation_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    champion = next(r for r in results if r.stage == "champion")
    raw_champion = next((r for r in results if r.stage == "champion_raw"), None)
    baseline = next(r for r in results if r.name.startswith("E0"))
    payload = {
        "config": {
            "cutoffs": list(config.backtest_cutoffs),
            "target_days": config.target_days,
            "train_horizons": list(config.train_horizons),
            "train_window_days": config.train_window_days,
            "city_history_days": config.city_history_days,
            "max_train_rows": config.max_train_rows,
        },
        "champion": {
            "name": champion.name,
            "pooled": champion.pooled,
            "folds": champion.folds,
            **_name_segments(champion.detail, data),
        },
        "champion_raw": {
            "name": raw_champion.name,
            "pooled": raw_champion.pooled,
            "training": raw_champion.training or None,
            "training_folds": raw_champion.training_folds,
            "validation_minus_train_wape": (
                round(raw_champion.pooled["wape"] - raw_champion.training["wape"], 6)
                if raw_champion.training
                else None
            ),
            "folds": raw_champion.folds,
            **_name_segments(raw_champion.detail, data),
        }
        if raw_champion is not None
        else None,
        "baseline": {
            "name": baseline.name,
            "pooled": baseline.pooled,
            "folds": baseline.folds,
            **_name_segments(baseline.detail, data),
        },
        "validation_audit": validation_audit,
    }
    # Persist row evidence and within-fold breakdowns, so a later audit can
    # inspect an event window without refitting the model.
    from .backtest import score
    for label, result in (("champion", champion), ("champion_raw", raw_champion), ("baseline", baseline)):
        if result is None or result.predictions.empty:
            continue
        path = config.artifacts_dir / f"oof_{label}.parquet"
        write_parquet_atomic(result.predictions, path)
        payload[label]["oof"] = {"file": path.name, "sha256": sha256_file(path)}
        for key in ("horizon", CHECKIN):
            payload[label][f"by_fold_{key}"] = [
                {"fold_cutoff": str(fold), "key": str(value), **score(rows.actual, rows.prediction)}
                for (fold, value), rows in result.predictions.groupby(["fold_cutoff", key])
            ]
    (config.artifacts_dir / "backtest_metrics_phase2.json").write_text(
        json.dumps(_json_ready(payload), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return payload


def _write_stability(
    data,
    spec: ChampionSpec,
    config: Pol4Config,
    raw_oof_predictions: pd.DataFrame,
    calibration_method: str,
) -> dict[str, Any]:
    """Snapshot the champion across the D-30 -> D-1 ladder on a historical window."""
    target_start = config.cutoff - pd.Timedelta(days=config.stability_window_days - 1)
    anchor = target_start - pd.Timedelta(days=max(stability_module.SNAPSHOT_HORIZONS))
    model, baseline = stability_module.fit_snapshot_model(
        data, anchor, spec.groups, spec.kind, spec.params, spec.log1p, config
    )
    calibrator = (
        fit_oof_calibrator(
            raw_oof_predictions,
            config,
            method=calibration_method,
            outcomes_available_by=anchor,
        )
        if not raw_oof_predictions.empty
        else Calibrator(method="none", alpha_global=1.0)
    )
    report = stability_module.analyse(
        data,
        target_start,
        config.stability_window_days,
        stability_module.model_snapshot(model, spec.groups, baseline, calibrator),
        config,
    )
    report.snapshots.to_parquet(config.artifacts_dir / "stability.parquet", index=False)
    return report.summary

def build_parser() -> argparse.ArgumentParser:
    defaults = Pol4Config()
    parser = argparse.ArgumentParser(
        prog="python -m ml.pol4.pipeline",
        description="Pol 4 demand forecasting: backtest, champion model, results.csv",
    )
    parser.add_argument("--raw-dir", type=Path, default=defaults.raw_dir)
    parser.add_argument("--artifacts-dir", type=Path, default=defaults.artifacts_dir)
    parser.add_argument(
        "--model",
        choices=("champion", "baseline"),
        default="champion",
        help="champion = the Phase 2 remaining-demand model; baseline = the pickup projection alone",
    )
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="also run the full feature ladder and rewrite experiments.csv (slow)",
    )
    parser.add_argument(
        "--zero-policy",
        choices=("prior", "zero"),
        default=defaults.zero_observation_policy,
        help="what to predict when a pair has no observed searches at the cutoff",
    )
    parser.add_argument("--min-city-support", type=int, default=defaults.min_city_support)
    parser.add_argument(
        "--train-window-days",
        type=int,
        default=defaults.train_window_days,
        help="number of check-in days included in training",
    )
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=defaults.max_train_rows,
        help="optional training-row cap (default: unlimited)",
    )
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--skip-backtest", action="store_true", help="final inference only")
    parser.add_argument("--skip-stability", action="store_true")
    parser.add_argument(
        "--reuse-trainset",
        action="store_true",
        help="reuse the checksum-verified materialised train set when input hashes match",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    config = replace(
        Pol4Config(),
        raw_dir=args.raw_dir,
        artifacts_dir=args.artifacts_dir,
        zero_observation_policy=args.zero_policy,
        min_city_support=args.min_city_support,
        train_window_days=args.train_window_days,
        max_train_rows=args.max_train_rows,
        seed=args.seed,
    )
    summary = run(
        config,
        model=args.model,
        ablation=args.ablation,
        skip_backtest=args.skip_backtest,
        skip_stability=args.skip_stability,
        reuse_trainset=args.reuse_trainset,
    )

    backtest = summary.get("backtest", {})
    stability = summary.get("stability", {})
    submission = summary["submission"]
    print("\nPOL 4 PIPELINE")
    print(f"  model                 {summary['model']}")
    print(f"  cutoff                {summary['cutoff']}")
    print(f"  target window         {summary['target_window'][0]} .. {summary['target_window'][1]}")
    if not backtest.get("skipped"):
        print(f"  backtest WAPE         {backtest['pooled_wape']:.4f}")
        print(f"  normalised bias       {backtest['pooled_normalised_bias']:+.4f}")
        if "baseline_wape" in backtest:
            print(f"  pickup baseline WAPE  {backtest['baseline_wape']:.4f}"
                  f"   (improvement {backtest['relative_improvement']:.1%})")
        if "calibration_accepted" in backtest:
            print(
                f"  raw champion           WAPE {backtest['raw_wape']:.4f}"
                f" / bias {backtest['raw_normalised_bias']:+.4f}"
            )
            for candidate in backtest["calibration_candidates"]:
                print(
                    f"  calibration {candidate['method']:<22}"
                    f" WAPE {candidate['wape']:.4f}"
                    f" / bias {candidate['normalised_bias']:+.4f}"
                    f" / safe {'YES' if candidate['safe'] else 'NO'}"
                )
            print(
                "  calibration accepted  "
                + (
                    backtest["calibration_method"]
                    if backtest["calibration_accepted"]
                    else "NO (raw champion kept)"
                )
            )
            audit = backtest.get("validation_audit", {})
            prequential = audit.get("prequential_selection", {}).get("pooled", {})
            holdout = audit.get("last_fold_holdout", {})
            overfit = audit.get("overfit_diagnostics", {})
            if prequential:
                print(
                    f"  honest prequential     WAPE {prequential['wape']:.4f}"
                    f" / bias {prequential['normalised_bias']:+.4f}"
                )
            if holdout:
                print(
                    f"  last-fold holdout      WAPE {holdout['wape']:.4f}"
                    f" ({holdout['selected_from_prior_folds']})"
                )
            if overfit.get("validation_minus_train_wape") is not None:
                print(
                    f"  train/valid WAPE       "
                    f"{overfit['raw_train']['wape']:.4f} / "
                    f"{overfit['raw_validation']['wape']:.4f}"
                )
        for cutoff, wape in backtest["folds"].items():
            print(f"    fold {cutoff}     {wape:.4f}")
    if not stability.get("skipped"):
        print(f"  stability score       {stability['stability_score']:.4f}"
              f"   (convergence {stability['convergence_rate']:.1%})")
    print(f"  results.csv           {submission['path']}")
    print(f"  rows / cities / dates {submission['rows']} / {submission['cities']} / {submission['dates']}")
    print(f"  total demand          {submission['total_predicted_demand']:,.0f}")
    print(f"  valid                 {'YES' if submission['valid'] else 'NO'}")
    print(f"  elapsed               {summary['elapsed_seconds']}s\n")
    return 0 if submission["valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
