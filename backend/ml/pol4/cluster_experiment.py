"""Choose the aggregation level by measuring it, not by asserting a threshold.

The brief penalises excessive aggregation without defining it, so the level is
run as a swept parameter: at several dendrogram cut heights the panel is
rebuilt, the model retrained and the whole thing scored - and the answer is a
curve of accuracy against the number of submitted rows, not a defended guess.

Three things in here are specific to a clustered submission and are easy to get
subtly wrong:

1. **The assignment is frozen on train-only data.** `ClusterPlan.fit` is called
   at the fold's own cutoff, so "this city is sparse enough to cluster" is
   decided without the validation window - the same discipline the pickup curves
   already obey.
2. **One score, at the submission grain.** A clustered submission is a mix of
   city rows and cluster rows, and the grader computes one global WAPE over that
   mix. Reporting "city WAPE" and "cluster WAPE" separately would be reporting
   two numbers that are not the one being graded. Every level here is scored on
   its own panel, and because aggregation only moves demand between rows and
   never creates or destroys it, the WAPE denominator is identical at every
   level - which is what makes the curve a like-for-like comparison.
3. **Several windows, not one.** The cut height is now a hyper-parameter chosen
   on validation performance. With a single 45-day holdout the choice would be
   worth very little; `config.cluster_backtest_cutoffs` runs it over three
   windows in different seasons.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .aggregate import aggregate_data, city_volume_shares
from .backtest import score
from .baseline import PickupBaseline
from .champion import ChampionSpec
from .clustering import ClusterAssignment, ClusterPlan
from .config import Pol4Config
from .dataset import build_inference_frame, build_training_frame
from .features import CLUSTER_GROUP_ORDER, feature_names
from .loader import CHECKIN, CITY, Pol4Data
from .models import RemainingDemandModel
from .submission import build_grid

log = logging.getLogger(__name__)

CLUSTER = "cluster_code"


def cluster_spec(**overrides: Any) -> ChampionSpec:
    """The champion, unchanged, reading the three extra virtual-city columns."""
    return ChampionSpec(groups=CLUSTER_GROUP_ORDER, **overrides)


@dataclass
class PanelResult:
    """One fitted model's predictions over one panel, for one cutoff."""

    cutoff: pd.Timestamp
    assignment: ClusterAssignment
    panel: pd.DataFrame          # cluster_code, checkin, horizon, observed, predicted[, actual]
    training_rows: int
    seconds: float
    models: dict[tuple[int, int], RemainingDemandModel] = field(
        default_factory=dict, repr=False
    )

    @property
    def n_groups(self) -> int:
        return int(self.panel[CLUSTER].nunique())

    def scores(self) -> dict[str, float]:
        if "actual" not in self.panel.columns:
            raise ValueError("this panel has no outcome to score against")
        return score(self.panel["actual"], self.panel["predicted_demand"])


def fit_panel(
    data: Pol4Data,
    cutoff: pd.Timestamp,
    assignment: ClusterAssignment,
    spec: ChampionSpec,
    config: Pol4Config,
    *,
    target_dates: pd.DatetimeIndex | None = None,
    keep_models: bool = False,
) -> PanelResult:
    """Aggregate, featurise, train and predict - the unmodified pipeline.

    Nothing below this line knows that a row might be several cities. That is
    the whole point: `aggregate_data` produces a `Pol4Data` of virtual cities and
    `PickupBaseline`, `build_training_frame`, `build_inference_frame` and
    `RemainingDemandModel` are the same objects the city-level champion uses.
    """
    started = time.perf_counter()
    cutoff = pd.Timestamp(cutoff)
    columns = feature_names(spec.groups)
    if target_dates is None:
        target_dates = pd.date_range(cutoff + pd.Timedelta(days=1), periods=config.target_days)

    panel_data = aggregate_data(data, assignment)
    baseline = PickupBaseline.fit(panel_data, cutoff, config)
    train = build_training_frame(panel_data, cutoff, spec.groups, baseline, config)
    infer = build_inference_frame(
        panel_data, cutoff, target_dates, spec.groups, baseline, config
    )

    train_horizon = train.meta["horizon"].to_numpy()
    infer_horizon = infer.meta["horizon"].to_numpy()
    observed = infer.meta["observed"].to_numpy(dtype=np.float64)
    predicted = np.full(len(infer.meta), np.nan)
    models: dict[tuple[int, int], RemainingDemandModel] = {}

    for low, high in spec.bands:
        fit_rows = (train_horizon >= low) & (train_horizon <= high)
        predict_rows = (infer_horizon >= low) & (infer_horizon <= high)
        if not fit_rows.any() or not predict_rows.any():
            continue
        model = RemainingDemandModel(
            kind=spec.kind, params=spec.params, log1p=spec.log1p, seed=config.seed
        ).fit(train.X.loc[fit_rows, columns], train.y[fit_rows])
        predicted[predict_rows] = model.predict_final(
            infer.X.loc[predict_rows, columns], observed[predict_rows]
        )
        if keep_models:
            models[(low, high)] = model
    if np.isnan(predicted).any():
        missing = sorted(set(infer_horizon[np.isnan(predicted)]))
        raise RuntimeError(f"no horizon band covers horizon(s) {missing}")

    # The Phase 1 floor: a row cannot finish below what has already been counted.
    predicted = np.maximum(predicted, observed)
    panel = infer.meta.rename(columns={CITY: CLUSTER}).assign(predicted_demand=predicted)

    truth = panel_data.final_demand(target_dates.min(), target_dates.max())
    if len(truth):
        truth = truth.rename(columns={CITY: CLUSTER})
        panel = panel.merge(truth, on=[CLUSTER, CHECKIN], how="left")
        panel["actual"] = panel.pop("final").fillna(0.0).astype(np.float64)

    training_rows = len(train)
    del train, infer, panel_data, baseline
    return PanelResult(
        cutoff=cutoff,
        assignment=assignment,
        panel=panel,
        training_rows=training_rows,
        seconds=time.perf_counter() - started,
        models=models,
    )


# ----------------------------------------------------------------- blending
def blend_panels(
    city: PanelResult,
    cluster: PanelResult,
    *,
    shrinkage: float,
) -> pd.DataFrame:
    """Shrink a city's own forecast towards its cluster's, by how much history it has.

    A hard cutoff - stay alone or be fully merged - has to answer "why this
    threshold and not one city over?". Shrinkage does not: a city with real
    volume keeps its own forecast, a nearly-empty one leans on the pooled series
    its cluster provides, and everything in between slides smoothly.

    The blend is applied to *remaining* demand, not to the total, for the same
    reason the model predicts remaining in the first place: what has already
    been counted is known, and only the pickup is uncertain. Because a cluster's
    observed demand is exactly the sum of its members', allocating the cluster's
    remaining by the member's share keeps the Phase 1 floor true by
    construction rather than by clamping:

        w(c)    = volume(c) / (volume(c) + shrinkage)
        pred(c) = observed(c) + w(c) * city_remaining(c)
                              + (1 - w(c)) * share(c) * cluster_remaining(g)

    A city that is nobody's cluster-mate keeps `w = 1` whatever the shrinkage.
    Blending a singleton would not be shrinkage at all - it would silently
    ensemble two fits of the same series - so the effect measured here is only
    ever the effect on the cities the partition actually pools.

    The result is scored at full city grain, so it concedes **no rows at all**:
    the clustering is used as a regulariser, not as a submission grain.
    """
    assignment = cluster.assignment
    shares = city_volume_shares(assignment)
    mapping = assignment.mapping()
    volumes = assignment.frame.set_index(CITY)["volume"]
    sizes = assignment.clusters.set_index("cluster_code")["n_cities_in_cluster"]

    frame = city.panel.rename(columns={CLUSTER: CITY}).copy()
    codes = frame[CITY].to_numpy()
    frame["_cluster"] = mapping.reindex(codes).to_numpy()

    pooled = cluster.panel.assign(
        _remaining=cluster.panel["predicted_demand"].to_numpy(dtype=np.float64)
        - cluster.panel["observed"].to_numpy(dtype=np.float64)
    )
    remaining = pooled.set_index([CLUSTER, CHECKIN])["_remaining"]
    key = pd.MultiIndex.from_arrays(
        [frame["_cluster"].to_numpy(), frame[CHECKIN].to_numpy()]
    )
    cluster_remaining = remaining.reindex(key).to_numpy()
    share = shares.reindex(codes).to_numpy()

    volume = volumes.reindex(codes).to_numpy(dtype=np.float64)
    weight = np.ones_like(volume) if shrinkage <= 0 else volume / (volume + shrinkage)
    singleton = sizes.reindex(frame["_cluster"].to_numpy()).to_numpy() <= 1
    weight = np.where(singleton, 1.0, weight)

    observed = frame["observed"].to_numpy(dtype=np.float64)
    city_remaining = frame["predicted_demand"].to_numpy(dtype=np.float64) - observed
    allocated = share * cluster_remaining
    blended = observed + weight * city_remaining + (1.0 - weight) * allocated

    frame["city_prediction"] = frame["predicted_demand"]
    frame["allocated_cluster_remaining"] = allocated
    frame["blend_weight_on_city"] = weight
    frame["predicted_demand"] = np.maximum(np.nan_to_num(blended, nan=0.0), observed)
    return frame


def aggregate_panel(panel: pd.DataFrame, assignment: ClusterAssignment) -> pd.DataFrame:
    """Sum a city-grain panel onto an assignment's rows.

    This is the control the sweep needs. Comparing a clustered submission with
    an unclustered one compares two different row counts, and the merged rows
    are mechanically easier - offsetting errors inside a group cancel before the
    absolute value is taken. Summing the *city model's own* predictions onto the
    same rows removes that effect, so the difference between this and a model
    fitted on the pooled series is the part attributable to aggregation as a
    modelling decision rather than as a metric artefact.
    """
    mapping = assignment.mapping()
    frame = panel.rename(columns={CLUSTER: CITY}).copy()
    frame[CLUSTER] = mapping.reindex(frame[CITY].to_numpy()).to_numpy()
    columns = [c for c in ("observed", "predicted_demand", "actual") if c in frame.columns]
    return frame.groupby([CLUSTER, CHECKIN], as_index=False, observed=True)[columns].sum()


def merged_cities(assignment: ClusterAssignment) -> set[int]:
    """The cities this partition is willing to pool with a neighbour."""
    sizes = assignment.clusters.set_index("cluster_code")["n_cities_in_cluster"]
    merged = set(sizes.index[sizes > 1].tolist())
    members = assignment.frame
    return set(members.loc[members["cluster_code"].isin(merged), CITY].tolist())


def sparse_rows(panel: pd.DataFrame, assignment: ClusterAssignment) -> pd.DataFrame:
    """The city-grain rows belonging to cities the level merges.

    A global WAPE is demand-weighted, and every city this partition is willing
    to merge holds a fraction of a percent of the market, so a large improvement
    on them can only ever move the headline a little. Scoring the slice on its
    own denominator keeps both facts visible instead of letting the small global
    effect hide a real one - or the reverse.
    """
    codes = panel[CLUSTER] if CLUSTER in panel.columns else panel[CITY]
    return panel.loc[codes.isin(merged_cities(assignment)).to_numpy()]


# -------------------------------------------------------------------- sweep
def merged_row_diagnostics(result: PanelResult) -> dict[str, Any]:
    """Where the level's error actually sits: merged rows versus singletons."""
    panel = result.panel
    sizes = result.assignment.clusters.set_index("cluster_code")["n_cities_in_cluster"]
    merged = sizes.reindex(panel[CLUSTER].to_numpy()).to_numpy() > 1
    total_actual = float(panel["actual"].sum())
    out: dict[str, Any] = {"total_actual": round(total_actual, 2)}
    for label, mask in (("merged_rows", merged), ("singleton_rows", ~merged)):
        if not mask.any():
            out[label] = None
            continue
        rows = panel.loc[mask]
        absolute = float(np.abs(rows["actual"] - rows["predicted_demand"]).sum())
        out[label] = {
            "rows": int(mask.sum()),
            "actual_share": round(float(rows["actual"].sum()) / max(total_actual, 1e-9), 6),
            "absolute_error": round(absolute, 2),
            "wape_contribution": round(absolute / max(total_actual, 1e-9), 6),
        }
    return out


@dataclass
class SweepLevel:
    """One aggregation level, pooled over every backtest window."""

    cut_height: float
    folds: dict[str, dict[str, float]]
    pooled: dict[str, float]
    n_groups: list[int]
    assignments: dict[str, dict[str, Any]]
    diagnostics: dict[str, Any]
    #: The same rows, predicted by the city-level model and summed. Empty at the
    #: identity level, where it would be the level itself.
    control: dict[str, float] = field(default_factory=dict)
    sparse: dict[str, Any] = field(default_factory=dict)

    @property
    def mean_groups(self) -> float:
        return float(np.mean(self.n_groups))

    def row(self) -> dict[str, Any]:
        merged_share = float(
            np.mean([entry["merged_demand_share"] for entry in self.assignments.values()])
        )
        return {
            "cut_height": round(self.cut_height, 6),
            "mean_groups": round(self.mean_groups, 2),
            "aggregation_ratio": round(self.mean_groups / 321.0, 4),
            "merged_demand_share": round(merged_share, 8),
            "wape": self.pooled["wape"],
            "normalised_bias": self.pooled["normalised_bias"],
            "mae": self.pooled["mae"],
            "wape_city_model_same_rows": self.control.get("wape"),
            "sparse_slice_wape": (self.sparse.get("city_model") or {}).get("wape"),
            "sparse_slice_demand_share": self.sparse.get("demand_share"),
            **{f"wape_{cutoff}": s["wape"] for cutoff, s in self.folds.items()},
            **{f"groups_{cutoff}": self.assignments[cutoff]["n_groups"] for cutoff in self.folds},
        }


def decompose_gain(levels: list["SweepLevel"]) -> list[dict[str, Any]]:
    """Split each level's WAPE gain into the part that is real and the part that is free.

    Aggregating rows lowers WAPE on its own: offsetting errors inside a group
    cancel before the absolute value is taken, so a coarser panel scores better
    even with an unchanged predictor. That part is a property of the metric, not
    of the model, and the brief charges rows for it either way.

        mechanical = WAPE(city panel) - WAPE(city predictions summed to the level)
        modelling  = WAPE(city predictions summed) - WAPE(model fitted on the level)

    A level whose gain is almost all mechanical has not learned anything from
    pooling; it has only been scored more kindly.
    """
    reference = max(levels, key=lambda level: level.mean_groups)
    base = reference.pooled["wape"]
    out: list[dict[str, Any]] = []
    for level in levels:
        if not level.control or level is reference:
            continue
        control = level.control["wape"]
        total = base - level.pooled["wape"]
        out.append(
            {
                "cut_height": round(float(level.cut_height), 6),
                "mean_groups": round(level.mean_groups, 2),
                "unclustered_wape": base,
                "control_wape": control,
                "clustered_wape": level.pooled["wape"],
                "total_gain": round(total, 6),
                "mechanical_gain": round(base - control, 6),
                "modelling_gain": round(control - level.pooled["wape"], 6),
                "modelling_share_of_gain": (
                    round((control - level.pooled["wape"]) / total, 4)
                    if abs(total) > 1e-12
                    else None
                ),
                "relative_total_gain": round(total / base, 6) if base else None,
                "relative_modelling_gain": (
                    round((control - level.pooled["wape"]) / base, 6) if base else None
                ),
            }
        )
    return out


def _pool(frames: list[pd.DataFrame]) -> dict[str, float]:
    pooled = pd.concat(frames, ignore_index=True)
    return score(pooled["actual"], pooled["predicted_demand"])


def run_sweep(
    data: Pol4Data,
    config: Pol4Config,
    spec: ChampionSpec | None = None,
    *,
    cutoffs: list[str] | None = None,
    n_levels: int | None = None,
    heights: list[float] | None = None,
    blend_grid: tuple[float, ...] | None = None,
) -> dict[str, Any]:
    """Sweep the aggregation level across every backtest window.

    Fold-outer, like `experiments.run_suite`, because a fold's training frame is
    well over a gigabyte and holding several at once is the difference between
    running on a laptop and not.
    """
    spec = spec or cluster_spec()
    cutoffs = list(cutoffs or config.cluster_backtest_cutoffs)
    blend_grid = tuple(blend_grid if blend_grid is not None else config.blend_shrinkage_grid)

    level_frames: dict[float, list[pd.DataFrame]] = {}
    level_folds: dict[float, dict[str, dict[str, float]]] = {}
    level_groups: dict[float, list[int]] = {}
    level_assignments: dict[float, dict[str, Any]] = {}
    level_diagnostics: dict[float, dict[str, Any]] = {}
    blend_frames: dict[tuple[float, float], list[pd.DataFrame]] = {}
    blend_folds: dict[tuple[float, float], dict[str, dict[str, float]]] = {}
    blend_sparse: dict[tuple[float, float], list[pd.DataFrame]] = {}
    control_frames: dict[float, list[pd.DataFrame]] = {}
    sparse_frames: dict[float, list[pd.DataFrame]] = {}
    ladder: list[float] = []
    plans: dict[str, dict[str, Any]] = {}

    for cutoff_value in cutoffs:
        cutoff = pd.Timestamp(cutoff_value)
        label = cutoff.date().isoformat()
        log.info("cluster sweep fold %s: fitting the dendrogram", label)
        plan = ClusterPlan.fit(data, cutoff, config)
        if heights is not None:
            ladder = list(heights)
        elif not ladder:
            ladder = plan.height_ladder(n_levels or config.cluster_sweep_levels)
        plans[label] = {
            "protected_cities": int(len(plan.protected)),
            "merge_heights": {
                f"p{int(q * 100)}": round(float(v), 4)
                for q, v in zip(
                    (0.1, 0.25, 0.5, 0.75, 0.9),
                    np.quantile(plan.merge_heights(), (0.1, 0.25, 0.5, 0.75, 0.9)),
                )
            }
            if len(plan.merge_heights())
            else {},
        }

        city_result: PanelResult | None = None
        for height in ladder:
            # `assign(0)` is the identity partition, so the city-level arm is
            # produced by the same call as every clustered one.
            assignment = plan.assign(height)
            result = fit_panel(data, cutoff, assignment, spec, config)
            fold_score = result.scores()
            log.info(
                "  fold %s height %-8.4f groups=%-4d wape=%.5f (%.0fs)",
                label,
                height,
                result.n_groups,
                fold_score["wape"],
                result.seconds,
            )
            level_frames.setdefault(height, []).append(result.panel)
            level_folds.setdefault(height, {})[label] = fold_score
            level_groups.setdefault(height, []).append(result.n_groups)
            level_assignments.setdefault(height, {})[label] = assignment.summary()
            level_diagnostics.setdefault(height, {})[label] = merged_row_diagnostics(result)
            if height <= 0:
                city_result = result
            elif city_result is not None:
                control_frames.setdefault(height, []).append(
                    aggregate_panel(city_result.panel, assignment)
                )
                sparse_frames.setdefault(height, []).append(
                    sparse_rows(city_result.panel, assignment)
                )
                for shrinkage in blend_grid:
                    blended = blend_panels(city_result, result, shrinkage=shrinkage)
                    key = (height, shrinkage)
                    blend_frames.setdefault(key, []).append(blended)
                    blend_folds.setdefault(key, {})[label] = score(
                        blended["actual"], blended["predicted_demand"]
                    )
                    blend_sparse.setdefault(key, []).append(
                        sparse_rows(blended, assignment)
                    )
            del result

    levels: list[SweepLevel] = []
    for height in ladder:
        if height not in level_frames:
            continue
        sparse: dict[str, Any] = {}
        if height in sparse_frames:
            # Every fold contributes the slice its OWN assignment defines, so a
            # city that is mergeable in one window and not in another is counted
            # only where it was actually mergeable.
            pooled_city = pd.concat(sparse_frames[height], ignore_index=True)
            pooled_all = pd.concat(level_frames[0.0], ignore_index=True)
            sparse = {
                "cities": int(pooled_city[CLUSTER].nunique()),
                "demand_share": round(
                    float(pooled_city["actual"].sum())
                    / max(float(pooled_all["actual"].sum()), 1e-9),
                    8,
                ),
                "city_model": score(
                    pooled_city["actual"], pooled_city["predicted_demand"]
                ),
            }
        levels.append(
            SweepLevel(
                cut_height=height,
                folds=level_folds[height],
                pooled=_pool(level_frames[height]),
                n_groups=level_groups[height],
                assignments=level_assignments[height],
                diagnostics=level_diagnostics[height],
                control=_pool(control_frames[height]) if height in control_frames else {},
                sparse=sparse,
            )
        )

    # Every blend is scored at full city grain, so its row count is 321 by
    # construction and its WAPE is directly comparable to the height=0 level.
    blends = [
        {
            "cut_height": round(height, 6),
            "shrinkage": shrinkage,
            "n_groups": 321,
            "pooled": _pool(frames),
            "folds": blend_folds[(height, shrinkage)],
            "sparse": _pool(blend_sparse[(height, shrinkage)]),
        }
        for (height, shrinkage), frames in sorted(blend_frames.items())
    ]

    return {
        "cutoffs": cutoffs,
        "ladder": [round(float(h), 6) for h in ladder],
        "gain_decomposition": decompose_gain(levels),
        "plans": plans,
        "levels": levels,
        "blends": blends,
        "spec": spec.describe(),
    }
