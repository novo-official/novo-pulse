"""Reproducible audit challengers. Never promotes or overwrites the submission.

All arms use the same city/date grid and five cutoffs. Cluster predictions are
reconciled by train-only shares before scoring. Choices are exploratory and
preselected, not nested evidence of a winning model.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import logging
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .artifacts import build_input_manifest, sha256_file, write_json_atomic
from .backtest import score
from .champion import ChampionSpec
from .cluster_experiment import fit_panel, blend_panels
from .clustering import ClusterPlan
from .config import Pol4Config
from .experiments import ExperimentResult, cross_fit_calibration, breakdowns
from .features import CLUSTER_GROUP_ORDER, GROUP_ORDER
from .loader import load_pol4, CITY, CHECKIN
from .validation import _fold_cluster_interval

ARMS = ("city", "origin", "compact", "cluster", "cluster_compact", "poisson")
EXTRA_ARMS = ("champion66", "origin66")


def summarise(frame):
    folds = {str(k): score(g.actual, g.prediction) for k, g in frame.groupby("fold_cutoff")}
    return {"pooled": score(frame.actual, frame.prediction), "folds": folds,
            "interval": _fold_cluster_interval(folds),
            "by_fold_horizon": [{"fold_cutoff": str(f), "horizon": int(h), **score(g.actual, g.prediction)}
                                for (f, h), g in frame.groupby(["fold_cutoff", "horizon"])],
            "by_fold_date": [{"fold_cutoff": str(f), "checkin": str(pd.Timestamp(d).date()),
                              **score(g.actual, g.prediction)}
                             for (f, d), g in frame.groupby(["fold_cutoff", CHECKIN])],
            "shock_diagnostic": {
                "excluded_fold": "2025-05-21", "diagnostic_only": True,
                "without_fold": score(frame.loc[frame.fold_cutoff != "2025-05-21", "actual"],
                                      frame.loc[frame.fold_cutoff != "2025-05-21", "prediction"]),
                "war_dates_only": score(frame.loc[frame.checkin.between("2025-06-13", "2025-06-24"), "actual"],
                                        frame.loc[frame.checkin.between("2025-06-13", "2025-06-24"), "prediction"]),
            }}


def run(config=None, *, trees=600, arms=ARMS, output=None):
    config = config or Pol4Config()
    output = Path(output or config.artifacts_dir.parent / "pol4_jury")
    output.mkdir(parents=True, exist_ok=True)
    data = load_pol4(config)
    inputs = build_input_manifest(config, data)
    contract = {"input_digest": inputs["input_digest"], "cutoffs": config.backtest_cutoffs,
                "trees": trees, "arms": list(arms), "seed": config.seed,
                "train_window_days": config.train_window_days,
                "city_history_days": config.city_history_days,
                "horizons": list(config.train_horizons), "max_train_rows": config.max_train_rows,
                "cluster_cut_height": 1.970305, "calibration": "guarded_bias_horizon",
                "code_sha256": {p.name: sha256_file(p) for p in Path(__file__).parent.glob("*.py")}}
    path = output / "contract.json"
    if path.exists() and json.loads(path.read_text()) != contract:
        raise ValueError("Experiment contract changed; use a new output directory")
    write_json_atomic(contract, path)
    write_json_atomic(inputs, output / "input_manifest.json")
    for cutoff in config.backtest_cutoffs:
        plan = ClusterPlan.fit(data, pd.Timestamp(cutoff), config)
        identity = plan.assign(0)
        city_result = None
        for arm in arms:
            checkpoint = output / f"{arm}_{cutoff}.parquet"
            if checkpoint.exists():
                continue
            started = time.perf_counter()
            cfg = replace(config, city_history_mode="origin" if arm in {"origin", "origin66"} else "fold")
            compact = arm in {"compact", "cluster_compact", "poisson"}
            clustered = arm in {"cluster", "cluster_compact"}
            groups = (("compact", "cluster") if clustered else ("compact",)) if compact else CLUSTER_GROUP_ORDER
            if arm in EXTRA_ARMS:
                groups = GROUP_ORDER
            # Identity and clustered arms share the same schema (66 + 3).
            spec = ChampionSpec(groups=groups, calibration="none", log1p=arm != "poisson",
                                params={"n_estimators": trees, "n_jobs": 4,
                                        **({"objective": "poisson"} if arm == "poisson" else {})})
            assignment = plan.assign(contract["cluster_cut_height"]) if clustered else identity
            result = fit_panel(data, pd.Timestamp(cutoff), assignment, spec, cfg, keep_models=True)
            frame = result.panel.rename(columns={"cluster_code": CITY})
            if clustered:
                if city_result is None:
                    # A matching city grid supplies only truth and observed totals.
                    city_result = fit_panel(data, pd.Timestamp(cutoff), identity,
                                            replace(spec, params={"n_estimators": 1, "n_jobs": 4}), cfg)
                allocated = blend_panels(city_result, result, shrinkage=1)
                frame = allocated.assign(predicted_demand=allocated.observed + allocated.allocated_cluster_remaining)
                conserved = frame.assign(cluster=frame[CITY].map(assignment.mapping())).groupby(
                    ["cluster", CHECKIN]).predicted_demand.sum()
                expected = result.panel.set_index(["cluster_code", CHECKIN]).predicted_demand
                np.testing.assert_allclose(conserved.sort_index(), expected.sort_index(), rtol=1e-10)
                assignment.frame.to_csv(output / f"{arm}_{cutoff}_clusters.csv", index=False)
            frame = frame.rename(columns={"predicted_demand": "prediction"}).assign(fold_cutoff=cutoff)
            frame["horizon_bucket"] = frame.horizon.map(lambda h: next(f"{a}-{b}" for a,b in cfg.horizon_buckets if a <= h <= b))
            frame["province_code"] = frame[CITY].map(data.province_of)
            frame["weekday"] = frame.checkin.dt.dayofweek
            frame["demand_bucket"] = "all"
            frame["observation_state"] = np.where(frame.observed > 0, "observed", "zero")
            if len(frame) != len(data.city_codes) * cfg.target_days or frame[[CITY, CHECKIN]].duplicated().any():
                raise ValueError("arm did not produce the common city/date grid")
            frame.to_parquet(checkpoint, index=False)
            result.bundle.importance().to_csv(output / f"{arm}_{cutoff}_importance.csv", index=False)
            write_json_atomic({"seconds": time.perf_counter()-started, "training_rows": result.training_rows,
                               "history_mode": cfg.city_history_mode, "spec": spec.describe(),
                               "predictions_sha256": sha256_file(checkpoint)},
                              output / f"{arm}_{cutoff}.json")
            print(f"{cutoff} {arm}: WAPE={score(frame.actual, frame.prediction)['wape']} ({time.perf_counter()-started:.0f}s)", flush=True)
            if arm == "city":
                city_result = result
            del result
    results = {}
    for arm in arms:
        frames = []
        for cutoff in config.backtest_cutoffs:
            checkpoint = output / f"{arm}_{cutoff}.parquet"
            metadata = json.loads((output / f"{arm}_{cutoff}.json").read_text())
            if sha256_file(checkpoint) != metadata["predictions_sha256"]:
                raise ValueError(f"Modified predictions: {checkpoint.name}")
            frames.append(pd.read_parquet(checkpoint))
        frame = pd.concat(frames, ignore_index=True)
        raw = summarise(frame)
        result = ExperimentResult(arm, arm, raw["pooled"], raw["folds"], [], breakdowns(frame, "prediction"), 0,
                                  predictions=frame)
        calibrated, _ = cross_fit_calibration(result, config, method="guarded_bias_horizon")
        results[arm] = {"raw": raw, "calibrated": summarise(calibrated.predictions)}
        calibrated.predictions.to_parquet(output / f"{arm}_calibrated.parquet", index=False)
    payload = {"contract": contract, "status": "completed", "selection": "no automatic promotion",
               "limitations": ["Exploratory fixed specifications; selection is not nested",
                               "Origin ablation changes city history only; pickup curves remain fold-fitted",
                               "Cluster cut height was chosen on historical experiments",
                               "Five folds do not establish significance of small gains"], "arms": results}
    # JSON requires null for undefined scores on absent diagnostic slices.
    from .pipeline import _json_ready
    write_json_atomic(_json_ready(payload), output / "comparison.json")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trees", type=int, default=600)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--arms", nargs="+", choices=ARMS + EXTRA_ARMS, default=list(ARMS))
    parser.add_argument("--cutoffs", nargs="+")
    args = parser.parse_args()
    config = Pol4Config()
    if args.cutoffs:
        config.backtest_cutoffs = args.cutoffs
    logging.basicConfig(level=logging.WARNING)
    run(config, trees=args.trees, output=args.output, arms=args.arms)


if __name__ == "__main__":
    main()
