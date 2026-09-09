"""The Pol 4 pipeline: one command from the raw CSVs to a validated results.csv.

    PYTHONPATH=backend python -m ml.pol4.pipeline

Runs, in order:

    load + validate -> fit pickup curves at the competition cutoff
                    -> pseudo-competition backtest across historical cutoffs
                    -> final Azar 1404 inference from evaluation.csv
                    -> write results.csv -> validate it, loudly

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

from .backtest import run_backtest
from .baseline import PickupBaseline
from .config import Pol4Config
from .loader import CITY, load_pol4
from .submission import (
    build_grid,
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


def run(config: Pol4Config | None = None, skip_backtest: bool = False) -> dict[str, Any]:
    config = config or Pol4Config()
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
        "cutoff": config.cutoff.date().isoformat(),
        "target_window": [
            config.target_start.date().isoformat(),
            config.target_end.date().isoformat(),
        ],
        "data": data.summary(),
    }

    # -- 2. pickup curves at the competition cutoff -------------------------
    model = PickupBaseline.fit(data, config.cutoff, config)
    curves_path = config.artifacts_dir / "pickup_curves.parquet"
    model.curves.to_frame().to_parquet(curves_path, index=False)
    summary["model"] = model.summary()
    log.info("pickup curves written to %s", curves_path)

    # -- 3. backtest --------------------------------------------------------
    if skip_backtest:
        summary["backtest"] = {"skipped": True}
    else:
        log.info("running the pseudo-competition backtest")
        backtest = run_backtest(data, config)
        metrics_path = config.artifacts_dir / "backtest_metrics.json"
        metrics_path.write_text(
            json.dumps(_json_ready(backtest), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        summary["backtest"] = {
            "pooled_wape": backtest["pooled"]["overall"]["wape"],
            "pooled_normalised_bias": backtest["pooled"]["overall"]["normalised_bias"],
            "folds": {
                fold["cutoff"]: fold["overall"]["wape"] for fold in backtest["folds"]
            },
            "artefact": str(metrics_path.relative_to(config.artifacts_dir.parents[1])),
        }
        log.info("backtest written to %s", metrics_path)

    # -- 4. final inference -------------------------------------------------
    target_dates = config.target_dates()
    grid = build_grid(data, config.cutoff, target_dates)
    predictions = model.predict(grid)

    # -- 5. results.csv -----------------------------------------------------
    submission = build_submission(predictions)
    results_path = write_submission(submission, config.artifacts_dir / "results.csv")

    # -- 6. validate --------------------------------------------------------
    report = validate_submission(
        submission, data.city_codes, config, raise_on_error=True
    )
    summary["submission"] = report.as_dict()
    summary["submission"]["path"] = str(results_path)
    # Some cities round to zero across the whole window. That is a measured
    # result, not an assumption: each one still receives a positive prior, but
    # its historical demand is so small that the expectation lands below half a
    # search per night. Reporting their support here keeps that auditable.
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
        "curve_source_counts": predictions["curve_source"].value_counts().to_dict(),
        "cities_rounding_to_zero": {
            "count": int(len(silent)),
            "median_historical_demand": float(
                historical.reindex(silent).fillna(0.0).median()
            )
            if len(silent)
            else None,
            "max_historical_demand": float(
                historical.reindex(silent).fillna(0.0).max()
            )
            if len(silent)
            else None,
            "median_unrounded_prediction": float(
                unrounded.reindex(silent).fillna(0.0).median()
            )
            if len(silent)
            else None,
            "note": (
                "these cities receive a positive prior; it rounds to zero because "
                "their measured demand is below half a search per night"
            ),
        },
    }
    summary["elapsed_seconds"] = round(time.perf_counter() - started, 2)

    summary_path = config.artifacts_dir / "run_summary.json"
    summary_path.write_text(
        json.dumps(_json_ready(summary), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info("results.csv written to %s (valid=%s)", results_path, report.valid)
    return summary


def build_parser() -> argparse.ArgumentParser:
    defaults = Pol4Config()
    parser = argparse.ArgumentParser(
        prog="python -m ml.pol4.pipeline",
        description="Pol 4 demand forecasting: pickup baseline, backtest, results.csv",
    )
    parser.add_argument("--raw-dir", type=Path, default=defaults.raw_dir)
    parser.add_argument("--artifacts-dir", type=Path, default=defaults.artifacts_dir)
    parser.add_argument(
        "--zero-policy",
        choices=("prior", "zero"),
        default=defaults.zero_observation_policy,
        help="what to predict when a pair has no observed searches at the cutoff",
    )
    parser.add_argument("--min-city-support", type=int, default=defaults.min_city_support)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument(
        "--skip-backtest", action="store_true", help="final inference only (faster)"
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
        seed=args.seed,
    )
    summary = run(config, skip_backtest=args.skip_backtest)

    backtest = summary.get("backtest", {})
    print("\nPOL 4 PIPELINE")
    print(f"  cutoff                {summary['cutoff']}")
    print(f"  target window         {summary['target_window'][0]} .. {summary['target_window'][1]}")
    if not backtest.get("skipped"):
        print(f"  backtest WAPE         {backtest['pooled_wape']:.4f}")
        print(f"  normalised bias       {backtest['pooled_normalised_bias']:+.4f}")
        for cutoff, wape in backtest["folds"].items():
            print(f"    fold {cutoff}     {wape:.4f}")
    submission = summary["submission"]
    print(f"  results.csv           {submission['path']}")
    print(f"  rows / cities / dates {submission['rows']} / {submission['cities']} / {submission['dates']}")
    print(f"  total demand          {submission['total_predicted_demand']:,.0f}")
    print(f"  valid                 {'YES' if submission['valid'] else 'NO'}")
    print(f"  elapsed               {summary['elapsed_seconds']}s\n")
    return 0 if submission["valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
