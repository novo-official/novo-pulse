"""Demand-shape clustering: which cities may be pooled into one virtual city.

This module decides *membership only*. It never touches features, training or
prediction - that separation is the point. A cluster is produced here as a
mapping `city_code -> cluster_code`; `aggregate.py` turns that mapping into a
`Pol4Data` whose rows are virtual cities, and every existing module downstream
runs on it unchanged.

Three rules make the decision defensible rather than asserted:

* **Train-only.** Every number in a city's profile comes from check-ins that
  had already completed at the cutoff the assignment is fitted at. Deciding
  "this city is sparse enough to cluster" with volume that includes the
  validation window is leakage - a quieter form of the same bug as fitting a
  pickup curve on the future.
* **Within province.** Clusters never cross a province boundary, so the
  `province_*` feature block sums to exactly the same totals whether the panel
  is clustered or not, and the comparison between the two stays attributable to
  the aggregation itself.
* **Hubs are protected.** A city holding more than `cluster_eligible_share` of
  national demand is never merged. Pooling a hub with its small neighbours
  would cancel large errors and flatter the metric - which is precisely the
  "excessive aggregation" the brief penalises - and it cannot be justified by
  sparsity, because a hub is not sparse.

The aggregation level itself is NOT decided here. `assign()` takes a dendrogram
cut height, and the sweep in `cluster_experiment.py` scores several of them, so
the level is chosen from a measured accuracy-versus-aggregation curve instead of
a threshold someone picked.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .config import Pol4Config
from .loader import CHECKIN, CITY, PROVINCE, SEARCHES, Pol4Data

EPS = 1e-9

#: Thursday, Friday - the same definition `features.py` verified against the data.
WEEKEND_DAYS = (3, 4)

PROFILE_BLOCKS: dict[str, tuple[str, ...]] = {
    # Weekly shape: seven weekday means normalised to shares of the week, so a
    # tiny city with the same Thursday spike as a large one sits next to it.
    "weekly_shape": tuple(f"weekday_share_{d}" for d in range(7)),
    "weekend": ("weekend_ratio",),
    # Burstiness, scale-free. A cluster's volatility is not the average of its
    # members' - which is why it is a clustering input and then recomputed from
    # the pooled series afterwards.
    "burstiness": ("volatility", "p75_over_mean", "p90_over_mean"),
    "sparsity": ("zero_day_share",),
    "scale": ("log_volume",),
    "geography": ("lat", "long"),
}


def daily_demand_matrix(
    data: Pol4Data, cutoff: pd.Timestamp, config: Pol4Config
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """`(city x check-in date)` final demand over the pre-cutoff window.

    Only check-ins at or before `cutoff` appear, because only those have
    received every search they will ever receive. The grid is completed with
    zeros: a check-in date with no row saw no demand, and dropping it would
    overstate every statistic computed from this matrix.
    """
    cutoff = pd.Timestamp(cutoff)
    history = data.history_before(cutoff)
    if history.empty:
        raise ValueError(f"no completed check-ins at or before {cutoff.date()}")

    window_start = cutoff - pd.Timedelta(days=config.city_history_days - 1)
    dates = pd.date_range(max(window_start, history[CHECKIN].min()), cutoff, freq="D")
    cities = np.sort(data.city_codes)

    inside = history[history[CHECKIN].isin(dates)]
    rows = pd.Index(cities).get_indexer(inside[CITY].to_numpy())
    columns = dates.get_indexer(pd.DatetimeIndex(inside[CHECKIN]))
    matrix = np.zeros((len(cities), len(dates)), dtype=np.float64)
    np.add.at(
        matrix,
        (rows, columns),
        inside[SEARCHES].to_numpy(dtype=np.float64),
    )
    return cities, matrix, dates


def city_profiles(
    data: Pol4Data, cutoff: pd.Timestamp, config: Pol4Config | None = None
) -> pd.DataFrame:
    """Demand-shape profile per city, from pre-cutoff history only.

    The columns are the ones `features.py` already computes per city - weekly
    shape, weekend ratio, volatility, p75/p90, geography - plus the two this
    problem needs and that module has no reason to carry: total volume (which
    decides eligibility) and the share of days with no demand at all, which for
    this dataset separates "small" from "almost always empty".
    """
    config = config or Pol4Config()
    cities, matrix, dates = daily_demand_matrix(data, cutoff, config)

    volume = matrix.sum(axis=1)
    mean = matrix.mean(axis=1)
    safe_mean = np.maximum(mean, EPS)
    weekday = dates.dayofweek.to_numpy()

    frame = pd.DataFrame(index=pd.Index(cities, name=CITY))
    frame["volume"] = volume
    frame["log_volume"] = np.log1p(volume)
    frame["mean"] = mean
    frame["std"] = matrix.std(axis=1)
    frame["volatility"] = frame["std"] / safe_mean
    frame["p75_over_mean"] = np.quantile(matrix, 0.75, axis=1) / safe_mean
    frame["p90_over_mean"] = np.quantile(matrix, 0.90, axis=1) / safe_mean
    frame["zero_day_share"] = (matrix <= 0).mean(axis=1)

    weekday_totals = np.zeros((len(cities), 7), dtype=np.float64)
    for day in range(7):
        columns = weekday == day
        if columns.any():
            weekday_totals[:, day] = matrix[:, columns].mean(axis=1)
    week_total = np.maximum(weekday_totals.sum(axis=1), EPS)
    for day in range(7):
        # A city with no demand at all has no weekly shape; a flat week is the
        # honest stand-in and puts it next to the other empty cities.
        share = np.where(volume > 0, weekday_totals[:, day] / week_total, 1.0 / 7.0)
        frame[f"weekday_share_{day}"] = share

    weekend = weekday_totals[:, list(WEEKEND_DAYS)].mean(axis=1)
    weekdays = weekday_totals[:, [d for d in range(7) if d not in WEEKEND_DAYS]].mean(axis=1)
    frame["weekend_ratio"] = np.where(volume > 0, weekend / np.maximum(weekdays, EPS), 1.0)

    geography = data.cities.set_index(CITY).reindex(cities)
    frame["lat"] = geography["lat"].to_numpy()
    frame["long"] = geography["long"].to_numpy()
    frame[PROVINCE] = geography[PROVINCE].to_numpy()

    frame["demand_share"] = volume / max(volume.sum(), EPS)
    frame["window_days"] = len(dates)
    return frame.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _standardise(profiles: pd.DataFrame, geo_weight: float) -> tuple[np.ndarray, list[str]]:
    """Z-score every profile column, then balance the blocks.

    Standardising over *all* cities rather than the mergeable subset is what
    makes one cut height mean the same thing in every province and at every
    backtest cutoff. Each block is then divided by the square root of its width,
    so the seven-column weekly shape does not outvote the one-column volatility
    simply by being wider.
    """
    columns: list[str] = []
    blocks: list[np.ndarray] = []
    for name, names in PROFILE_BLOCKS.items():
        values = profiles.loc[:, list(names)].to_numpy(dtype=np.float64)
        centred = values - values.mean(axis=0)
        spread = centred.std(axis=0)
        scaled = centred / np.where(spread > EPS, spread, 1.0)
        weight = (geo_weight if name == "geography" else 1.0) / np.sqrt(len(names))
        blocks.append(scaled * weight)
        columns.extend(names)
    return np.hstack(blocks), columns


@dataclass(frozen=True)
class ClusterAssignment:
    """One partition of the 321 cities into submission rows.

    `frame` is the mapping the submission needs (`city_code -> cluster_code`).
    `clusters` is the virtual-city table `aggregate.py` turns into a `cities`
    frame: one row per submitted group, carrying the centroid, the province and
    the three cluster-shape features.
    """

    cutoff: pd.Timestamp
    cut_height: float
    frame: pd.DataFrame          # city_code, cluster_code, province_code, volume
    clusters: pd.DataFrame       # cluster_code + virtual-city attributes
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_cities(self) -> int:
        return len(self.frame)

    @property
    def n_groups(self) -> int:
        return len(self.clusters)

    @property
    def n_merged_cities(self) -> int:
        """Cities that share their submission row with at least one other."""
        return int((self.clusters["n_cities_in_cluster"] > 1).mul(
            self.clusters["n_cities_in_cluster"]
        ).sum())

    @property
    def is_identity(self) -> bool:
        return self.n_groups == self.n_cities

    @property
    def merged_demand_share(self) -> float:
        """Share of historical demand that sits inside a merged row.

        The honest headline for "how much did we aggregate": a partition that
        halves the row count but only pools 0.4% of the market has conceded far
        less than the row count suggests.
        """
        merged = self.clusters["n_cities_in_cluster"] > 1
        total = float(self.clusters["cluster_volume"].sum())
        if total <= 0:
            return 0.0
        return float(self.clusters.loc[merged, "cluster_volume"].sum() / total)

    def mapping(self) -> pd.Series:
        return self.frame.set_index(CITY)["cluster_code"]

    def summary(self) -> dict[str, Any]:
        sizes = self.clusters["n_cities_in_cluster"]
        return {
            "cutoff": pd.Timestamp(self.cutoff).date().isoformat(),
            "cut_height": round(float(self.cut_height), 6),
            "n_cities": self.n_cities,
            "n_groups": self.n_groups,
            "aggregation_ratio": round(self.n_groups / max(self.n_cities, 1), 6),
            "merged_cities": self.n_merged_cities,
            "merged_demand_share": round(self.merged_demand_share, 8),
            "largest_cluster": int(sizes.max()),
            "clusters_of_size_1": int((sizes == 1).sum()),
            **self.meta,
        }


@dataclass(frozen=True)
class ClusterPlan:
    """Per-province Ward dendrograms, fitted once per cutoff.

    Holding the linkage rather than a single partition is what lets the level be
    swept: every cut height reads the same trees, so the resulting partitions
    are nested and the accuracy-versus-aggregation curve compares one decision
    at a time.
    """

    cutoff: pd.Timestamp
    profiles: pd.DataFrame
    linkages: dict[int, tuple[np.ndarray, np.ndarray]]
    protected: np.ndarray
    config: Pol4Config
    geo_weight: float

    @classmethod
    def fit(
        cls,
        data: Pol4Data,
        cutoff: pd.Timestamp,
        config: Pol4Config | None = None,
        *,
        eligible_share: float | None = None,
        geo_weight: float | None = None,
    ) -> "ClusterPlan":
        from scipy.cluster.hierarchy import linkage

        config = config or Pol4Config()
        cutoff = pd.Timestamp(cutoff)
        eligible_share = (
            config.cluster_eligible_share if eligible_share is None else eligible_share
        )
        geo_weight = config.cluster_geo_weight if geo_weight is None else geo_weight

        profiles = city_profiles(data, cutoff, config)
        matrix, _ = _standardise(profiles, geo_weight)
        protected = profiles.index.to_numpy()[
            profiles["demand_share"].to_numpy() >= eligible_share
        ]

        linkages: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        eligible_mask = ~profiles.index.isin(protected)
        for province, rows in profiles.loc[eligible_mask].groupby(PROVINCE, sort=True):
            codes = rows.index.to_numpy()
            if len(codes) < 2:
                continue
            positions = profiles.index.get_indexer(codes)
            linkages[int(province)] = (linkage(matrix[positions], method="ward"), codes)

        return cls(
            cutoff=cutoff,
            profiles=profiles,
            linkages=linkages,
            protected=np.sort(protected),
            config=config,
            geo_weight=float(geo_weight),
        )

    # ------------------------------------------------------------- heights
    def merge_heights(self) -> np.ndarray:
        """Every height at which some pair of cities would merge."""
        if not self.linkages:
            return np.zeros(0)
        return np.sort(
            np.concatenate([matrix[:, 2] for matrix, _ in self.linkages.values()])
        )

    def height_ladder(self, n: int) -> list[float]:
        """A sweep ladder: `n` cut heights spanning no merging to heavy merging.

        Quantiles of the merge heights, not a linear grid, so every step of the
        ladder actually changes the partition - a linear grid over a skewed
        dendrogram wastes most of its points on identical answers.
        """
        heights = self.merge_heights()
        if not len(heights):
            return [0.0]
        quantiles = np.linspace(0.0, 1.0, n + 1)[1:]
        ladder = np.unique(np.round(np.quantile(heights, quantiles), 6))
        return [0.0, *(float(value) for value in ladder)]

    # ------------------------------------------------------------- assign
    def assign(self, cut_height: float) -> ClusterAssignment:
        """Cut every province's dendrogram at one height.

        `cut_height = 0` yields the identity partition: 321 groups, no merging,
        which is the city-level arm of the comparison and travels through
        exactly the same aggregation code as every clustered arm.
        """
        from scipy.cluster.hierarchy import fcluster

        cut_height = float(cut_height)
        labels = pd.Series(
            self.profiles.index.to_numpy(), index=self.profiles.index, name="group"
        ).astype("int64")

        if cut_height > 0:
            for province, (matrix, codes) in self.linkages.items():
                assigned = fcluster(matrix, t=cut_height, criterion="distance")
                # The cluster code is the smallest member's city code: stable,
                # inside the code space the organisers already know, and
                # reproducible from `clusters.csv` alone.
                for label in np.unique(assigned):
                    members = codes[assigned == label]
                    labels.loc[members] = int(members.min())

        return self._materialise(labels, cut_height)

    def assign_n_groups(self, n_groups: int) -> ClusterAssignment:
        """The coarsest partition with at least `n_groups` rows (for reporting)."""
        for height in [0.0, *self.merge_heights().tolist()]:
            assignment = self.assign(height)
            if assignment.n_groups <= n_groups:
                return assignment
        return self.assign(0.0)

    def _materialise(self, labels: pd.Series, cut_height: float) -> ClusterAssignment:
        profiles = self.profiles
        frame = pd.DataFrame(
            {
                CITY: profiles.index.to_numpy(),
                "cluster_code": labels.to_numpy(),
                PROVINCE: profiles[PROVINCE].to_numpy().astype("int64"),
                "volume": profiles["volume"].to_numpy(),
                "lat": profiles["lat"].to_numpy(),
                "long": profiles["long"].to_numpy(),
            }
        )

        grouped = frame.groupby("cluster_code", sort=True)
        weight = frame["volume"].to_numpy()
        # Volume-weighted centroid. The design note asks for a population
        # weighting; population is not in the competition data, and search
        # volume is the closest proxy it does contain - it also weights the
        # centroid towards the member that dominates the pooled series, which is
        # the member the model is really predicting.
        frame["_weight"] = np.where(weight > 0, weight, 1.0)
        frame["_wlat"] = frame["lat"] * frame["_weight"]
        frame["_wlong"] = frame["long"] * frame["_weight"]
        totals = frame.groupby("cluster_code", sort=True)[["_weight", "_wlat", "_wlong"]].sum()

        clusters = pd.DataFrame(
            {
                "cluster_code": totals.index.to_numpy(),
                PROVINCE: grouped[PROVINCE].first().to_numpy(),
                "lat": (totals["_wlat"] / totals["_weight"]).to_numpy(),
                "long": (totals["_wlong"] / totals["_weight"]).to_numpy(),
                "n_cities_in_cluster": grouped.size().to_numpy().astype("float64"),
                "cluster_volume": grouped["volume"].sum().to_numpy(),
                "cluster_min_member_volume": grouped["volume"].min().to_numpy(),
                "cluster_max_member_volume": grouped["volume"].max().to_numpy(),
            }
        )
        # "Five roughly-equal small cities" versus "four tiny ones and one that
        # is almost a hub" - the two behave differently and the size alone does
        # not separate them.
        clusters["cluster_max_member_share"] = clusters["cluster_max_member_volume"] / np.maximum(
            clusters["cluster_volume"], EPS
        )
        clusters.loc[clusters["cluster_volume"] <= 0, "cluster_max_member_share"] = 1.0

        frame = frame.drop(columns=["_weight", "_wlat", "_wlong"])
        return ClusterAssignment(
            cutoff=self.cutoff,
            cut_height=cut_height,
            frame=frame,
            clusters=clusters,
            meta={
                "protected_cities": int(len(self.protected)),
                "geo_weight": self.geo_weight,
                "eligible_share": float(self.config.cluster_eligible_share),
            },
        )


def identity_assignment(
    data: Pol4Data, cutoff: pd.Timestamp, config: Pol4Config | None = None
) -> ClusterAssignment:
    """Every city its own row - the city-level arm, built by the same code.

    Routing the unclustered arm through `ClusterPlan` rather than around it is
    what makes the comparison clean: both arms are aggregated, featurised and
    trained by identical code, so a WAPE difference can only come from the
    partition.
    """
    config = config or Pol4Config()
    profiles = city_profiles(data, cutoff, config)
    plan = ClusterPlan(
        cutoff=pd.Timestamp(cutoff),
        profiles=profiles,
        linkages={},
        protected=profiles.index.to_numpy(),
        config=config,
        geo_weight=config.cluster_geo_weight,
    )
    return plan.assign(0.0)
