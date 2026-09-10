"""The clustered arm, end to end: sweep the aggregation level, then ship one.

    PYTHONPATH=backend python -m ml.pol4.cluster_pipeline

Runs, in order:

    load + validate  -> per-city demand-shape profiles at each backtest cutoff
                     -> province-wise Ward dendrograms, fitted train-only
                     -> sweep several cut heights across three windows,
                        rebuilding the panel and retraining at every one
                     -> pick the level from the accuracy/aggregation curve
                     -> materialise the aggregated training set at that level
                     -> fit at the competition cutoff, write results and
                        clusters.csv, validate them

Nothing here is a second model. Every level - including the unclustered one -
runs `aggregate.aggregate_data` and then the ordinary pickup baseline, feature
builder, training frame and LightGBM bands, so the only thing that differs
between the arms is the partition.

The selection rule is stated before the numbers arrive, so it cannot be fitted
to them: **submit the coarsest level whose pooled WAPE is at least
`--min-gain` better, in relative terms, than the unclustered panel; otherwise
submit unclustered.** The brief penalises excessive aggregation, so a level that
merely ties has not earned its rows back.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .aggregate import aggregate_data
from .artifacts import build_input_manifest, write_json_atomic
from .baseline import PickupBaseline
from .champion import ChampionSpec
from .champion import FittedChampion
from .clustering import ClusterAssignment, ClusterPlan
from .cluster_experiment import (
    CLUSTER,
    SweepLevel,
    decompose_gain,
    blend_panels,
    cluster_spec,
    fit_panel,
    merged_row_diagnostics,
    run_sweep,
    sparse_rows,
)
from .config import Pol4Config
from .dataset import build_training_frame, materialize_training_frame
from .loader import CHECKIN, CITY, PROVINCE, load_pol4
from .paths_compat import cluster_artifacts_dir
from .submission import DATE_FORMAT, write_submission

log = logging.getLogger(__name__)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, SweepLevel):
        return _json_ready(asdict(value))
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if value is pd.NaT:
        return None
    return value


# ------------------------------------------------------------------ selection
def select_level(
    levels: list[SweepLevel],
    *,
    min_gain: float,
    criterion: str = "modelling",
) -> tuple[SweepLevel, dict[str, Any]]:
    """The coarsest level whose *modelling* gain clears `min_gain`.

    The obvious rule - "take the level with the best WAPE" - is wrong here, and
    the control arm is what shows why. Merging rows lowers WAPE on its own,
    because offsetting errors inside a group cancel before the absolute value is
    taken. That gain is free, unearned, and charged for: the brief penalises the
    rows either way. So the quantity that has to clear the bar is the part that
    is not free -

        modelling gain = WAPE(city predictions summed onto this level's rows)
                       - WAPE(a model actually fitted on the pooled series)

    - which asks whether pooling taught the model anything. `criterion="total"`
    restores the naive rule and is reported alongside, so the two are visible
    together rather than one of them being quietly chosen.

    Ties go to the unclustered panel: "no worse" is not evidence that pooling
    helped.
    """
    if criterion not in ("modelling", "total"):
        raise ValueError(f"criterion must be 'modelling' or 'total', got {criterion!r}")
    reference = max(levels, key=lambda level: level.mean_groups)
    base = reference.pooled["wape"]

    def relative_gain(level: SweepLevel, kind: str) -> float:
        if level is reference or not base:
            return 0.0
        if kind == "total":
            return (base - level.pooled["wape"]) / base
        control = level.control.get("wape")
        if control is None:
            return float("-inf")
        return (control - level.pooled["wape"]) / base

    def passing(kind: str) -> list[SweepLevel]:
        return [
            level
            for level in levels
            if level.mean_groups < reference.mean_groups
            and relative_gain(level, kind) >= min_gain
        ]

    candidates = passing(criterion)
    selected = min(candidates, key=lambda level: level.mean_groups) if candidates else reference
    naive = passing("total")
    naive_choice = min(naive, key=lambda level: level.mean_groups) if naive else reference

    return selected, {
        "rule": (
            "coarsest level whose MODELLING gain - the part of the WAPE "
            "improvement that survives comparing against the city model's own "
            f"predictions summed onto the same rows - is at least {min_gain:.1%} "
            "relative; unclustered otherwise"
        ),
        "criterion": criterion,
        "min_relative_gain": min_gain,
        "unclustered_wape": base,
        "selected_cut_height": round(float(selected.cut_height), 6),
        "selected_mean_groups": round(selected.mean_groups, 2),
        "selected_wape": selected.pooled["wape"],
        "selected_relative_modelling_gain": round(relative_gain(selected, "modelling"), 6),
        "selected_relative_total_gain": round(relative_gain(selected, "total"), 6),
        "clustered_candidates_considered": len(levels) - 1,
        "clustered_candidates_passing": len(candidates),
        # What the naive "best pooled WAPE" rule would have chosen, and why the
        # difference matters. Reported, never acted on.
        "naive_total_gain_rule": {
            "selected_mean_groups": round(naive_choice.mean_groups, 2),
            "selected_wape": naive_choice.pooled["wape"],
            "relative_total_gain": round(relative_gain(naive_choice, "total"), 6),
            "relative_modelling_gain": round(relative_gain(naive_choice, "modelling"), 6),
            "candidates_passing": len(naive),
        },
        "gain_decomposition": decompose_gain(levels),
    }


def select_blend(blends: list[dict[str, Any]], *, unclustered_wape: float) -> dict[str, Any] | None:
    """The best shrinkage blend, if any beats the unclustered panel outright.

    A blend is scored at full city grain, so it concedes no rows at all: it only
    has to be better, not better by a margin that pays for aggregation.
    """
    scored = [b for b in blends if b["shrinkage"] > 0]
    if not scored:
        return None
    best = min(scored, key=lambda b: b["pooled"]["wape"])
    if best["pooled"]["wape"] >= unclustered_wape:
        return None
    return best


def load_sweep(path: Path) -> dict[str, Any]:
    """Rebuild a finished sweep from `cluster_sweep.json`.

    The sweep is the expensive artefact - twenty-odd model fits - and choosing a
    level from it is arithmetic. Separating the two means a selection rule can
    be corrected, or a different `--min-gain` argued for, without spending
    another half hour re-fitting models that would return identical numbers.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    levels = [
        SweepLevel(
            cut_height=float(entry["cut_height"]),
            folds=entry["folds"],
            pooled=entry["pooled"],
            n_groups=[int(n) for n in entry["n_groups"]],
            assignments=entry["assignments"],
            diagnostics=entry["diagnostics"],
            control=entry.get("control") or {},
            sparse=entry.get("sparse") or {},
        )
        for entry in payload["level_detail"]
    ]
    return {**payload, "levels": levels}


# ------------------------------------------------------------------- outputs
def build_cluster_submission(panel: pd.DataFrame) -> pd.DataFrame:
    """`cluster_code, checkin, predicted_demand` at whatever grain the panel is."""
    out = pd.DataFrame(
        {
            "cluster_code": panel[CLUSTER].to_numpy(),
            "checkin": pd.to_datetime(panel[CHECKIN]).dt.strftime(DATE_FORMAT),
            "predicted_demand": np.maximum(
                panel["predicted_demand"].to_numpy(dtype=np.float64), 0.0
            ),
        }
    )
    out["predicted_demand"] = np.rint(out["predicted_demand"]).astype(np.int64)
    return out.sort_values(["cluster_code", "checkin"], ignore_index=True)


def validate_cluster_submission(
    frame: pd.DataFrame,
    assignment: ClusterAssignment,
    expected_cities: np.ndarray,
    config: Pol4Config,
) -> dict[str, Any]:
    """Every rule the competition imposes, restated for a clustered grain.

    The row count is no longer 321 x 30, but the two rules that actually matter
    survive aggregation: every city must be represented by exactly one row per
    date, and the dates must be the 30 Azar check-ins.
    """
    problems: list[str] = []
    expected_dates = config.target_dates()
    expected_codes = set(assignment.clusters["cluster_code"].astype("int64").tolist())

    if tuple(frame.columns) != ("cluster_code", "checkin", "predicted_demand"):
        problems.append(f"columns are {tuple(frame.columns)}")
    parsed = pd.to_datetime(frame["checkin"], format=DATE_FORMAT, errors="coerce")
    if parsed.isna().any():
        problems.append(f"{int(parsed.isna().sum())} unparseable checkin value(s)")
    if set(parsed.dropna()) != set(expected_dates):
        problems.append("checkin dates do not match the Azar 1404 window exactly")
    present = set(frame["cluster_code"].astype("int64").tolist())
    if present != expected_codes:
        problems.append(
            f"{len(expected_codes - present)} missing and {len(present - expected_codes)} "
            "unexpected cluster_code(s)"
        )
    if int(frame.duplicated(["cluster_code", "checkin"]).sum()):
        problems.append("duplicate (cluster_code, checkin) rows")
    if len(frame) != len(expected_codes) * len(expected_dates):
        problems.append(
            f"{len(frame)} rows, expected {len(expected_codes) * len(expected_dates)}"
        )
    values = pd.to_numeric(frame["predicted_demand"], errors="coerce")
    if values.isna().any():
        problems.append(f"{int(values.isna().sum())} non-numeric predicted_demand value(s)")
    if np.isinf(values.fillna(0.0).to_numpy()).any():
        problems.append("infinite predicted_demand value(s)")
    if (values.fillna(0.0) < 0).any():
        problems.append(f"{int((values.fillna(0.0) < 0).sum())} negative value(s)")

    # A city that is silently absent from `clusters.csv` would be scored as a
    # miss on the whole city, so the mapping is checked against every city.
    covered = set(assignment.frame[CITY].astype("int64").tolist())
    expected = set(np.asarray(expected_cities).astype("int64").tolist())
    if covered != expected:
        problems.append(
            f"clusters.csv covers {len(covered)} cities, expected {len(expected)}"
        )

    if problems:
        raise ValueError("clustered submission is invalid: " + "; ".join(problems))
    return {
        "valid": True,
        "rows": len(frame),
        "cluster_codes": len(present),
        "cities_covered": len(covered),
        "dates": int(parsed.nunique()),
        "total_predicted_demand": int(values.sum()),
        "rows_predicting_zero": int((values <= 0).sum()),
    }


def write_clusters_csv(assignment: ClusterAssignment, data, path: Path) -> Path:
    """The membership file a clustered submission has to ship with."""
    frame = assignment.frame[[CITY, "cluster_code", PROVINCE, "volume"]].copy()
    frame = frame.rename(columns={"volume": "train_window_searches"})
    if data.has_names:
        frame.insert(1, "city", data.city_name_of.reindex(frame[CITY]).to_numpy())
        frame["province"] = data.province_name_of.reindex(frame[CITY]).to_numpy()
    sizes = assignment.clusters.set_index("cluster_code")["n_cities_in_cluster"]
    frame["n_cities_in_cluster"] = sizes.reindex(frame["cluster_code"]).to_numpy().astype(int)
    frame = frame.sort_values(["cluster_code", CITY], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def save_arm(
    result: Any,
    data: Any,
    directory: Path,
    *,
    input_digest: str,
    target_dates: pd.DataFrame,
) -> dict[str, Any]:
    """Persist a fitted arm as native LightGBM boosters, and prove it reloads.

    The city-level pipeline ships `model_bundle/lightgbm_h1_14.txt` and friends;
    an arm that only leaves a CSV behind cannot be audited, re-run or even shown
    to be the model it claims to be. This writes the same bundle format -
    checksummed, with a load-parity check - so the clustered arm is inspectable
    on the same terms.

    A clustered bundle is only meaningful **together with `clusters.csv`**: its
    features are read from a panel of virtual cities, so reloading it requires
    rebuilding that panel with `aggregate_data(data, assignment)` first.
    """
    if result.bundle is None:
        raise ValueError("this arm was fitted without keep_models=True")
    manifest = result.bundle.save(directory, input_digest=input_digest)

    panel_data = aggregate_data(data, result.assignment)
    restored = FittedChampion.load(directory)
    before = result.bundle.predict(target_dates, panel_data)
    after = restored.predict(target_dates, panel_data)
    pd.testing.assert_frame_equal(before, after, check_exact=False, rtol=1e-12, atol=1e-12)

    importance = result.bundle.importance()
    importance.to_json(directory / "feature_importance.json", orient="records", indent=2)
    return {
        "path": str(directory),
        "model": result.bundle.spec.kind,
        "bundle_digest": manifest["bundle_digest"],
        "boosters": [name for name in manifest["files"] if name.endswith((".txt", ".cbm"))],
        "panel_rows": int(len(panel_data.cities)),
        "training_rows": result.training_rows,
        "load_parity_verified": True,
        "requires": "clusters.csv - the bundle reads a panel of virtual cities",
        "top_features": importance.head(10).to_dict(orient="records"),
    }


# ---------------------------------------------------------------------- run
def run(
    config: Pol4Config | None = None,
    *,
    spec: ChampionSpec | None = None,
    cutoffs: list[str] | None = None,
    n_levels: int | None = None,
    min_gain: float = 0.02,
    artifacts_dir: Path | None = None,
    skip_sweep: bool = False,
    cut_height: float | None = None,
    from_sweep: Path | None = None,
    criterion: str = "modelling",
) -> dict[str, Any]:
    """Sweep the aggregation level, select one, and ship it."""
    config = config or Pol4Config()
    spec = spec or cluster_spec()
    artifacts_dir = Path(artifacts_dir or cluster_artifacts_dir(config))
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    log.info("loading Pol 4 datasets from %s", config.raw_dir)
    data = load_pol4(config)
    input_manifest = build_input_manifest(config, data)
    write_json_atomic(input_manifest, artifacts_dir / "input_manifest.json")

    summary: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cutoff": config.cutoff.date().isoformat(),
        "target_window": [
            config.target_start.date().isoformat(),
            config.target_end.date().isoformat(),
        ],
        "input_digest": input_manifest["input_digest"],
        "spec": spec.describe(),
        "clustering": {
            "eligible_share": config.cluster_eligible_share,
            "geo_weight": config.cluster_geo_weight,
            "linkage": "ward",
            "within_province": True,
        },
    }

    # -- 1. the sweep -------------------------------------------------------
    if skip_sweep:
        summary["sweep"] = {"skipped": True}
        selection = {"rule": "cut height supplied on the command line"}
        chosen_height = clustered_height = float(cut_height or 0.0)
        blend_choice = None
        summary["best_clustered_candidate"] = None
    else:
        if from_sweep is not None:
            log.info("re-selecting from the saved sweep at %s", from_sweep)
            sweep = load_sweep(Path(from_sweep))
        else:
            sweep = run_sweep(data, config, spec, cutoffs=cutoffs, n_levels=n_levels)
        levels: list[SweepLevel] = sweep["levels"]
        table = pd.DataFrame([level.row() for level in levels])
        table.to_csv(artifacts_dir / "cluster_sweep.csv", index=False)
        blend_table = pd.DataFrame(
            [
                {
                    "cut_height": b["cut_height"],
                    "shrinkage": b["shrinkage"],
                    "n_groups": b["n_groups"],
                    "wape": b["pooled"]["wape"],
                    "normalised_bias": b["pooled"]["normalised_bias"],
                    "sparse_slice_wape": b["sparse"]["wape"],
                    **{f"wape_{k}": v["wape"] for k, v in b["folds"].items()},
                }
                for b in sweep["blends"]
            ]
        )
        blend_table.to_csv(artifacts_dir / "blend_sweep.csv", index=False)

        selected, selection = select_level(
            levels, min_gain=min_gain, criterion=criterion
        )
        blend_choice = select_blend(
            sweep["blends"], unclustered_wape=selection["unclustered_wape"]
        )
        chosen_height = float(selected.cut_height)
        # The best clustered candidate is always built, even when the rule
        # declines to submit it: it is the thing being argued about.
        best_clustered = max(
            (level for level in levels if level.mean_groups < 321),
            key=lambda level: (level.control.get("wape", float("inf")))
            - level.pooled["wape"],
            default=None,
        )
        clustered_height = (
            float(cut_height)
            if cut_height is not None
            else float(best_clustered.cut_height)
            if best_clustered is not None
            else 0.0
        )
        summary["best_clustered_candidate"] = (
            {
                "cut_height": round(float(best_clustered.cut_height), 6),
                "mean_groups": round(best_clustered.mean_groups, 2),
                "wape": best_clustered.pooled["wape"],
                "control_wape": best_clustered.control.get("wape"),
            }
            if best_clustered is not None
            else None
        )
        summary["sweep"] = {
            "cutoffs": sweep["cutoffs"],
            "ladder": sweep["ladder"],
            "gain_decomposition": decompose_gain(levels),
            "plans": sweep["plans"],
            "levels": [level.row() for level in levels],
            "level_detail": [_json_ready(asdict(level)) for level in levels],
            "blends": _json_ready(sweep["blends"]),
        }
        write_json_atomic(
            _json_ready(summary["sweep"]), artifacts_dir / "cluster_sweep.json"
        )
    summary["selection"] = _json_ready(selection)
    summary["blend_selection"] = _json_ready(blend_choice)
    selected_is_identity = chosen_height <= 0.0

    # -- 2. the assignment at the competition cutoff ------------------------
    log.info("fitting the competition-cutoff dendrogram at %s", config.cutoff.date())
    plan = ClusterPlan.fit(data, config.cutoff, config)
    plan.profiles.to_parquet(artifacts_dir / "city_profiles.parquet")

    # Three arms are always built, whatever the rule chose. The clustered panel
    # and its aggregated training set are the artefact this work exists to
    # produce; the rule decides which arm is *submitted*, not which are kept.
    identity = plan.assign(0.0)
    clustered = plan.assign(clustered_height)
    summary["assignment"] = clustered.summary()
    summary["clustered_cut_height"] = round(float(clustered_height), 6)
    write_clusters_csv(clustered, data, artifacts_dir / "clusters.csv")

    # -- 3. the new dataset: aggregated features over the virtual cities ----
    panel_data = aggregate_data(data, clustered)
    baseline = PickupBaseline.fit(panel_data, config.cutoff, config)
    train = build_training_frame(panel_data, config.cutoff, spec.groups, baseline, config)
    train_manifest = materialize_training_frame(
        train,
        artifacts_dir / "trainset",
        spec.groups,
        config,
        input_digest=input_manifest["input_digest"],
    )
    summary["trainset"] = {
        "path": str(artifacts_dir / "trainset"),
        "rows": train_manifest["rows"],
        "features": train_manifest["feature_count"],
        "panel_rows": int(len(panel_data.cities)),
        "cut_height": round(float(clustered_height), 6),
    }
    log.info(
        "materialised %s aggregated training rows over %d virtual cities",
        f"{len(train):,}",
        len(panel_data.cities),
    )
    del train, baseline, panel_data

    # -- 4. fit every arm at the competition cutoff -------------------------
    target_dates = config.target_dates()
    arms: dict[str, Path] = {}

    city_result = fit_panel(
        data,
        config.cutoff,
        identity,
        spec,
        config,
        target_dates=target_dates,
        keep_models=True,
    )
    city_submission = build_cluster_submission(city_result.panel)
    arms["city"] = write_submission(city_submission, artifacts_dir / "results_city.csv")
    summary["city_submission"] = validate_cluster_submission(
        city_submission, identity, data.city_codes, config
    )
    summary["city_model_bundle"] = save_arm(
        city_result,
        data,
        artifacts_dir / "model_bundle_city",
        input_digest=input_manifest["input_digest"],
        target_dates=target_dates,
    )

    if clustered.is_identity:
        # Nothing to cluster at this height; do not pretend otherwise by
        # writing a second copy of the city file under a clustered name.
        summary["clustered_submission"] = None
    else:
        cluster_result = fit_panel(
            data,
            config.cutoff,
            clustered,
            spec,
            config,
            target_dates=target_dates,
            keep_models=True,
        )
        cluster_submission = build_cluster_submission(cluster_result.panel)
        arms["clustered"] = write_submission(
            cluster_submission, artifacts_dir / "results_clustered.csv"
        )
        summary["clustered_submission"] = validate_cluster_submission(
            cluster_submission, clustered, data.city_codes, config
        )
        summary["clustered_model_bundle"] = save_arm(
            cluster_result,
            data,
            artifacts_dir / "model_bundle_clustered",
            input_digest=input_manifest["input_digest"],
            target_dates=target_dates,
        )

    if blend_choice is not None and not clustered.is_identity:
        blend_assignment = plan.assign(float(blend_choice["cut_height"]))
        blend_source = (
            cluster_result
            if float(blend_choice["cut_height"]) == clustered_height
            else fit_panel(
                data,
                config.cutoff,
                blend_assignment,
                spec,
                config,
                target_dates=target_dates,
            )
        )
        blended = blend_panels(
            city_result, blend_source, shrinkage=float(blend_choice["shrinkage"])
        )
        blended_submission = build_cluster_submission(
            blended.rename(columns={CITY: CLUSTER})
        )
        arms["blended"] = write_submission(
            blended_submission, artifacts_dir / "results_blended.csv"
        )
        summary["blended_submission"] = validate_cluster_submission(
            blended_submission, identity, data.city_codes, config
        )

    # -- 5. the arm the rule actually chose ---------------------------------
    chosen_arm = "clustered" if not selected_is_identity else (
        "blended" if blend_choice is not None and "blended" in arms else "city"
    )
    source = arms.get(chosen_arm, arms["city"])
    selected_path = artifacts_dir / "results_selected.csv"
    selected_path.write_text(Path(source).read_text(encoding="utf-8"), encoding="utf-8")
    summary["selected_arm"] = {
        "arm": chosen_arm,
        "path": str(selected_path),
        "source": str(source),
        "rows": int(len(pd.read_csv(selected_path))),
        "arms_written": {name: str(path) for name, path in arms.items()},
    }

    summary["runtime_seconds"] = round(time.perf_counter() - started, 1)
    write_json_atomic(_json_ready(summary), artifacts_dir / "run_summary.json")
    log.info("clustered run complete in %.0fs -> %s", summary["runtime_seconds"], artifacts_dir)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--levels", type=int, default=None, help="cut heights to sweep")
    parser.add_argument(
        "--cutoffs", nargs="*", default=None, help="backtest cutoffs for the sweep"
    )
    parser.add_argument(
        "--min-gain",
        type=float,
        default=0.02,
        help="relative WAPE gain a clustered level must show to be submitted",
    )
    parser.add_argument("--trees", type=int, default=None, help="LightGBM n_estimators")
    parser.add_argument("--skip-sweep", action="store_true")
    parser.add_argument(
        "--from-sweep",
        default=None,
        help="re-select from a finished cluster_sweep.json instead of re-fitting",
    )
    parser.add_argument(
        "--criterion",
        choices=("modelling", "total"),
        default="modelling",
        help="which part of the WAPE gain a clustered level must earn",
    )
    parser.add_argument("--cut-height", type=float, default=None)
    parser.add_argument("--artifacts-dir", default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    spec = cluster_spec(params={"n_estimators": args.trees}) if args.trees else cluster_spec()
    run(
        spec=spec,
        cutoffs=args.cutoffs,
        n_levels=args.levels,
        min_gain=args.min_gain,
        artifacts_dir=Path(args.artifacts_dir) if args.artifacts_dir else None,
        skip_sweep=args.skip_sweep,
        cut_height=args.cut_height,
        from_sweep=Path(args.from_sweep) if args.from_sweep else None,
        criterion=args.criterion,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
