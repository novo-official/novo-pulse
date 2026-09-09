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
from .backtest import run_backtest
from .baseline import PickupBaseline
from .champion import ChampionSpec, FittedChampion
from .config import Pol4Config
from .experiments import (
    ExperimentResult,
    ExperimentSpec,
    baseline_predictor,
    calibrated_predictor,
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
    spec: ChampionSpec | None = None,
    training_data_path: Path | None = None,
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
        experiments = _write_experiments(results, config)
        payload = _write_phase2_backtest(results, config, data)

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
        }

    # -- 4. stability -------------------------------------------------------
    if skip_stability or model == "baseline":
        # Carry forward what a previous full run measured rather than dropping
        # it. Labelled, so a stale number can never pass for a fresh one.
        previous = config.artifacts_dir / "stability_summary.json"
        if previous.exists():
            summary["stability"] = {
                **json.loads(previous.read_text(encoding="utf-8")),
                "measured_in_this_run": False,
            }
        else:
            summary["stability"] = {"skipped": True}
    else:
        log.info("measuring forecast stability across the D-30 -> D-1 ladder")
        summary["stability"] = _write_stability(data, spec, config)

    # -- 5. final inference at the competition cutoff -----------------------
    target_dates = config.target_dates()
    if model == "baseline":
        predictions = baseline.predict(build_grid(data, config.cutoff, target_dates))
        summary["champion"] = {"model": "pickup_baseline", **baseline.summary()}
    else:
        fitted = FittedChampion.fit(data, config.cutoff, spec, config)
        predictions = fitted.predict(target_dates, data)

        # A horizon-stratified sample of the exact rows the model was fitted on,
        # so the training data is readable without regenerating it. The full
        # frame is ~77 MB; `--export-training-data` writes that on request.
        sample_path = fitted.export_training_data(
            config.artifacts_dir / "training_sample.csv", sample=config.training_sample_rows
        )
        if training_data_path:
            full_path = fitted.export_training_data(training_data_path)
            log.info("full training frame written to %s", full_path)

        importance = fitted.importance()
        importance.to_json(
            config.artifacts_dir / "feature_importance.json", orient="records", indent=2
        )
        summary["champion"] = {
            **spec.describe(),
            "training_rows": fitted.training_rows,
            "training_sample": str(sample_path.name),
            "training_data_export": str(training_data_path) if training_data_path else None,
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
            f"CHAMPION {spec.kind}",
            "champion",
            gbdt_predictor(
                spec.kind,
                spec.groups,
                log1p=spec.log1p,
                params=spec.params,
                calibration="none",
                blend=None if spec.blend >= 1.0 else spec.blend,
                bands=spec.bands,
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
    results: list[ExperimentResult], config: Pol4Config, data
) -> dict[str, Any]:
    champion = next(r for r in results if r.stage == "champion")
    baseline = next(r for r in results if r.name.startswith("E0"))
    payload = {
        "config": {
            "cutoffs": list(config.backtest_cutoffs),
            "target_days": config.target_days,
            "train_horizons": list(config.train_horizons),
            "train_window_days": config.train_window_days,
            "max_train_rows": config.max_train_rows,
        },
        "champion": {
            "name": champion.name,
            "pooled": champion.pooled,
            "folds": champion.folds,
            **_name_segments(champion.detail, data),
        },
        "baseline": {
            "name": baseline.name,
            "pooled": baseline.pooled,
            "folds": baseline.folds,
            **_name_segments(baseline.detail, data),
        },
    }
    (config.artifacts_dir / "backtest_metrics_phase2.json").write_text(
        json.dumps(_json_ready(payload), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return payload


def _write_stability(
    data, spec: ChampionSpec, config: Pol4Config
) -> dict[str, Any]:
    """Snapshot the champion across the D-30 -> D-1 ladder on a historical window.

    The summary is written to its own artefact, not only into run_summary.json:
    a partial run (`--skip-stability`) rewrites run_summary and would otherwise
    silently delete a full run's stability numbers, leaving the dashboard to
    report a score of "—" with nothing to say it had ever been measured.
    """
    target_start = config.cutoff - pd.Timedelta(days=config.stability_window_days - 1)
    anchor = target_start - pd.Timedelta(days=max(stability_module.SNAPSHOT_HORIZONS))
    model, baseline = stability_module.fit_snapshot_model(
        data, anchor, spec.groups, spec.kind, spec.params, spec.log1p, config
    )
    report = stability_module.analyse(
        data,
        target_start,
        config.stability_window_days,
        stability_module.model_snapshot(model, spec.groups, baseline),
        config,
    )
    report.snapshots.to_parquet(config.artifacts_dir / "stability.parquet", index=False)
    (config.artifacts_dir / "stability_summary.json").write_text(
        json.dumps(_json_ready(report.summary), indent=2, ensure_ascii=False), encoding="utf-8"
    )
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
    parser.add_argument("--max-train-rows", type=int, default=defaults.max_train_rows)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument(
        "--export-training-data",
        type=Path,
        default=None,
        help="write the full supervised frame the champion is fitted on (~77 MB parquet)",
    )
    parser.add_argument("--skip-backtest", action="store_true", help="final inference only")
    parser.add_argument("--skip-stability", action="store_true")
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
        max_train_rows=args.max_train_rows,
        seed=args.seed,
    )
    summary = run(
        config,
        model=args.model,
        ablation=args.ablation,
        skip_backtest=args.skip_backtest,
        skip_stability=args.skip_stability,
        training_data_path=args.export_training_data,
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
