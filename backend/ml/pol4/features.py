"""Cutoff-safe features for the remaining-demand model.

Every feature in this module is read at a *horizon*, never at a date. A pair's
search history is held as one row of a `(pair x days_to_checkin)` matrix, and a
feature at horizon `h` may only touch columns `>= h` - the searches that had
already happened `h` days before the check-in. That makes cutoff safety
structural rather than a rule someone has to remember: there is no code path
that can read a column below `h`.

    daily[pair, d]  searches logged exactly d days before the check-in
    cum[pair, h]    searches logged at or before h days out  =  sum(daily[:, h:])
                    which is exactly "observed demand at cutoff T - h"

Market and province aggregates are built the same way, from their own matrices,
so a cross-city signal cannot smuggle in a later log date either.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from .config import MAX_LEAD_TIME, Pol4Config
from .loader import CHECKIN, CITY, DTC, PROVINCE, SEARCHES, Pol4Data

EPS = 1e-6

# Iranian weekend. Verified against the data rather than assumed - see
# `scripts`-free check in tests/test_pol4_features.py.
WEEKEND_DAYS = (3, 4)  # Thursday, Friday (Monday = 0)

# ---------------------------------------------------------------- feature groups
# The ablation in `experiments.py` adds these one at a time, in this order.
GROUP_BASE = "base"
GROUP_PICKUP = "pickup"
GROUP_VELOCITY = "velocity"
GROUP_ACTIVITY = "activity"
GROUP_CURVE = "curve"
GROUP_CALENDAR = "calendar"
GROUP_CITY = "city"
GROUP_MARKET = "market"
GROUP_PROVINCE = "province"
#: Virtual-city shape. Only meaningful once a panel row can hold more than
#: one city, so it lives outside GROUP_ORDER and the ablation ladder that
#: selected the champion - adding it there would silently change the model
#: every saved bundle was fitted with.
GROUP_CLUSTER = "cluster"

GROUP_ORDER = (
    GROUP_BASE,
    GROUP_PICKUP,
    GROUP_VELOCITY,
    GROUP_ACTIVITY,
    GROUP_CURVE,
    GROUP_CALENDAR,
    GROUP_CITY,
    GROUP_MARKET,
    GROUP_PROVINCE,
)

#: The clustered arm's feature set: the champion's 66, unchanged and in the same
#: order, plus the three columns that describe the row's own composition. A
#: singleton row carries n=1, its own volume and a share of 1.0, so both arms
#: are trained on an identical schema and any WAPE difference between them is
#: attributable to the partition rather than to the feature set.
CLUSTER_GROUP_ORDER = GROUP_ORDER + (GROUP_CLUSTER,)

CATEGORICAL = ("city_code", "province_code")

#: Columns `aggregate.py` writes onto the virtual-city table.
CLUSTER_FEATURES = (
    "n_cities_in_cluster",
    "cluster_min_member_volume",
    "cluster_max_member_share",
)


@dataclass
class PairTensor:
    """`(pair x days_to_checkin)` search counts, plus reverse-cumulative reads."""

    keys: pd.DataFrame           # city_code, checkin - row order of the matrices
    daily: np.ndarray            # (n_pairs, n_dtc) float32
    n_dtc: int

    @classmethod
    def build(
        cls, events: pd.DataFrame, keys: pd.DataFrame, n_dtc: int | None = None
    ) -> "PairTensor":
        n_dtc = int(n_dtc or MAX_LEAD_TIME + 1)
        keys = keys.reset_index(drop=True)
        lookup = pd.Series(np.arange(len(keys)), index=pd.MultiIndex.from_frame(keys))

        rows = lookup.reindex(pd.MultiIndex.from_frame(events[[CITY, CHECKIN]])).to_numpy()
        columns = events[DTC].to_numpy()
        keep = np.isfinite(rows) & (columns >= 0) & (columns < n_dtc)

        daily = np.zeros((len(keys), n_dtc), dtype=np.float32)
        np.add.at(
            daily,
            (rows[keep].astype(np.int64), columns[keep].astype(np.int64)),
            events[SEARCHES].to_numpy(dtype=np.float32)[keep],
        )
        return cls(keys=keys, daily=daily, n_dtc=n_dtc)

    @property
    def n_pairs(self) -> int:
        return len(self.keys)

    def observed(self, horizon: int) -> np.ndarray:
        """Demand visible `horizon` days before check-in, for every pair."""
        return self.daily[:, horizon:].sum(axis=1, dtype=np.float64)

    def final(self) -> np.ndarray:
        """Complete demand. Only meaningful for check-ins that have passed."""
        return self.daily.sum(axis=1, dtype=np.float64)

    def pickup(self, horizon: int, window: int) -> np.ndarray:
        """Searches that arrived in the `window` days before the cutoff."""
        return self.daily[:, horizon : horizon + window].sum(axis=1, dtype=np.float64)

    def group_by(self, level: pd.Series | None = None) -> tuple["PairTensor", np.ndarray]:
        """Aggregate to check-in level, or to (level, check-in) level.

        Returns the aggregated tensor and, for each original pair, the row of
        the aggregate it belongs to - so a per-pair feature can read its own
        market or province total without a join.
        """
        keys = self.keys.copy()
        if level is None:
            group_columns = [CHECKIN]
        else:
            keys = keys.assign(**{PROVINCE: level.to_numpy()})
            group_columns = [PROVINCE, CHECKIN]

        codes, uniques = pd.MultiIndex.from_frame(keys[group_columns]).factorize()
        aggregated = np.zeros((len(uniques), self.n_dtc), dtype=np.float32)
        np.add.at(aggregated, codes, self.daily)
        return (
            PairTensor(
                keys=uniques.to_frame(index=False), daily=aggregated, n_dtc=self.n_dtc
            ),
            codes,
        )


# --------------------------------------------------------------------- blocks
def _pickup_block(tensor: PairTensor, horizon: int, prefix: str = "") -> dict[str, np.ndarray]:
    observed = tensor.observed(horizon)
    out = {f"{prefix}observed_total": observed}
    for window in (1, 3, 7, 14):
        out[f"{prefix}pickup_{window}d"] = tensor.pickup(horizon, window)
    return out


def _velocity_block(
    tensor: PairTensor, horizon: int, block: dict[str, np.ndarray], prefix: str = ""
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for window in (3, 7, 14):
        out[f"{prefix}pickup_velocity_{window}d"] = block[f"{prefix}pickup_{window}d"] / window
    # Acceleration compares a window with the equal window before it: is the
    # pair speeding up, or just large?
    for window in (3, 7):
        recent = block[f"{prefix}pickup_{window}d"] / window
        previous = tensor.pickup(horizon + window, window) / window
        out[f"{prefix}pickup_acceleration_{window}d"] = recent - previous
    return out


def _ratio_block(block: dict[str, np.ndarray], prefix: str = "") -> dict[str, np.ndarray]:
    observed = block[f"{prefix}observed_total"]
    out: dict[str, np.ndarray] = {}
    for window in (1, 3, 7):
        out[f"{prefix}pickup_{window}d_share"] = block[f"{prefix}pickup_{window}d"] / (
            observed + EPS
        )
    out[f"{prefix}pickup_3d_over_7d"] = block[f"{prefix}pickup_3d"] / (
        block[f"{prefix}pickup_7d"] + EPS
    )
    return out


def _activity_block(tensor: PairTensor, horizon: int) -> dict[str, np.ndarray]:
    window = tensor.daily[:, horizon:]
    active = window > 0
    n_days = window.shape[1]
    counts = active.sum(axis=1)
    total = window.sum(axis=1, dtype=np.float64)

    # Offsets are measured from the cutoff, so they stay comparable across
    # horizons: 0 means "today", larger means "longer ago".
    reversed_active = active[:, ::-1]
    first_offset = np.where(
        counts > 0, n_days - 1 - np.argmax(reversed_active, axis=1), np.nan
    )
    last_offset = np.where(counts > 0, np.argmax(active, axis=1), np.nan)

    mean = np.divide(total, np.maximum(counts, 1))
    variance = np.where(
        counts > 0,
        (window.astype(np.float64) ** 2).sum(axis=1) / np.maximum(counts, 1) - mean**2,
        0.0,
    )
    return {
        "active_search_days": counts.astype(np.float64),
        "days_since_first_search": first_offset,
        "days_since_last_search": last_offset,
        "max_daily_search": window.max(axis=1).astype(np.float64) if n_days else np.zeros(len(window)),
        "mean_daily_search": mean,
        "std_daily_search": np.sqrt(np.maximum(variance, 0.0)),
        "search_span": np.where(counts > 0, first_offset - last_offset, np.nan),
    }


@lru_cache(maxsize=8)
def _jalali_parts(dates: tuple[pd.Timestamp, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Jalali month and day for a set of dates.

    Cached and computed over the *unique* dates: the grid repeats each check-in
    once per city and once per horizon, so converting per row would run the
    calendar conversion millions of times for a few hundred distinct answers.
    """
    from ..data.dates import to_jalali_string

    converted = [to_jalali_string(stamp) for stamp in dates]
    month = np.array([int(v[5:7]) if v else 0 for v in converted], dtype=np.float64)
    day = np.array([int(v[8:10]) if v else 0 for v in converted], dtype=np.float64)
    return month, day


def _calendar_block(checkins: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    weekday = checkins.dayofweek.to_numpy()
    unique = pd.DatetimeIndex(checkins.unique()).sort_values()
    month, day = _jalali_parts(tuple(unique))
    position = unique.get_indexer(checkins)
    jalali_month, jalali_day = month[position], day[position]
    return {
        "checkin_weekday": weekday.astype(np.float64),
        "is_weekend": np.isin(weekday, WEEKEND_DAYS).astype(np.float64),
        "checkin_day_of_month": checkins.day.to_numpy().astype(np.float64),
        # Jalali month and day carry the fixed-date Iranian calendar without
        # needing an external holiday file: Nowruz is 1 Farvardin, and Yalda -
        # which lands on the last day of the competition window - is 30 Azar.
        "jalali_month": jalali_month,
        "jalali_day": jalali_day,
    }


@dataclass
class CityHistory:
    """Per-city demand statistics, computed only from completed check-ins."""

    cutoff: pd.Timestamp
    table: pd.DataFrame            # indexed by city_code
    weekday_mean: pd.Series        # (city_code, weekday) -> mean demand

    @classmethod
    def fit(
        cls, data: Pol4Data, cutoff: pd.Timestamp, config: Pol4Config | None = None
    ) -> "CityHistory":
        config = config or Pol4Config()
        cutoff = pd.Timestamp(cutoff)
        history = data.history_before(cutoff)
        window_start = cutoff - pd.Timedelta(days=config.city_history_days - 1)
        dates = pd.date_range(max(window_start, history[CHECKIN].min()), cutoff, freq="D")

        demand = (
            history[history[CHECKIN].isin(dates)]
            .groupby([CITY, CHECKIN])[SEARCHES]
            .sum()
        )
        # Complete the grid: a check-in with no row had zero demand, and leaving
        # it out would overstate every statistic below.
        grid = pd.MultiIndex.from_product([data.city_codes, dates], names=[CITY, CHECKIN])
        demand = demand.reindex(grid, fill_value=0).reset_index()

        grouped = demand.groupby(CITY)[SEARCHES]
        table = pd.DataFrame(
            {
                "city_hist_mean": grouped.mean(),
                "city_hist_median": grouped.median(),
                "city_hist_p75": grouped.quantile(0.75),
                "city_hist_p90": grouped.quantile(0.90),
                "city_hist_std": grouped.std().fillna(0.0),
            }
        )
        table["city_volatility"] = table["city_hist_std"] / (table["city_hist_mean"] + EPS)

        demand["weekday"] = demand[CHECKIN].dt.dayofweek
        weekday_mean = demand.groupby([CITY, "weekday"])[SEARCHES].mean()
        weekend = demand[demand["weekday"].isin(WEEKEND_DAYS)].groupby(CITY)[SEARCHES].mean()
        weekday = demand[~demand["weekday"].isin(WEEKEND_DAYS)].groupby(CITY)[SEARCHES].mean()
        table["city_weekend_ratio"] = (weekend / (weekday + EPS)).reindex(table.index).fillna(1.0)

        return cls(cutoff=cutoff, table=table, weekday_mean=weekday_mean)

    def block(self, cities: np.ndarray, weekdays: np.ndarray) -> dict[str, np.ndarray]:
        frame = self.table.reindex(cities).fillna(0.0)
        pairs = pd.MultiIndex.from_arrays([cities, weekdays])
        out = {column: frame[column].to_numpy() for column in self.table.columns}
        out["city_weekday_mean"] = self.weekday_mean.reindex(pairs).fillna(0.0).to_numpy()
        return out


# ------------------------------------------------------------------- builder
@dataclass
class FeatureBuilder:
    """Builds the design matrix for a set of (pair, horizon) rows at one cutoff."""

    data: Pol4Data
    cutoff: pd.Timestamp
    config: Pol4Config
    tensor: PairTensor
    market: PairTensor
    market_rows: np.ndarray
    province: PairTensor
    province_rows: np.ndarray
    city_history: CityHistory
    province_of: pd.Series

    @classmethod
    def build(
        cls,
        data: Pol4Data,
        cutoff: pd.Timestamp,
        keys: pd.DataFrame,
        events: pd.DataFrame,
        config: Pol4Config | None = None,
    ) -> "FeatureBuilder":
        """`keys` are the (city, check-in) pairs; `events` the searches for them."""
        config = config or Pol4Config()
        cutoff = pd.Timestamp(cutoff)
        tensor = PairTensor.build(events, keys)
        province_of = data.province_of
        market, market_rows = tensor.group_by(None)
        province, province_rows = tensor.group_by(keys[CITY].map(province_of))
        return cls(
            data=data,
            cutoff=cutoff,
            config=config,
            tensor=tensor,
            market=market,
            market_rows=market_rows,
            province=province,
            province_rows=province_rows,
            city_history=CityHistory.fit(data, cutoff, config),
            province_of=province_of,
        )

    def at_horizon(
        self, horizon: int, groups: tuple[str, ...], baseline: Any | None = None
    ) -> pd.DataFrame:
        """Every requested feature group, for every pair, at one horizon."""
        keys = self.tensor.keys
        cities = keys[CITY].to_numpy()
        checkins = pd.DatetimeIndex(keys[CHECKIN])
        columns: dict[str, np.ndarray] = {}

        pair_block = _pickup_block(self.tensor, horizon)
        observed = pair_block["observed_total"]

        columns["observed_total"] = observed
        columns["days_to_checkin"] = np.full(len(keys), float(horizon))

        if GROUP_PICKUP in groups:
            columns.update({k: v for k, v in pair_block.items() if k != "observed_total"})
        if GROUP_VELOCITY in groups:
            columns.update(_velocity_block(self.tensor, horizon, pair_block))
            columns.update(_ratio_block(pair_block))
        if GROUP_ACTIVITY in groups:
            columns.update(_activity_block(self.tensor, horizon))
        if GROUP_CURVE in groups and baseline is not None:
            fraction = baseline.curves.fraction(cities, np.full(len(keys), horizon))
            projection = observed / fraction
            columns["expected_completion_fraction"] = fraction
            columns["expected_remaining_fraction"] = 1.0 - fraction
            columns["pickup_baseline_prediction"] = np.maximum(projection, observed)
            columns["pickup_baseline_remaining"] = np.maximum(projection - observed, 0.0)
            # How the pair is tracking against the city's own historical curve:
            # >1 means it is running hot for this far out.
            expected_observed = baseline.zero_prior.level(
                cities, np.full(len(keys), horizon), baseline.curves.province_of
            ) * fraction
            columns["pickup_surprise"] = observed / (expected_observed + EPS)
        if GROUP_CALENDAR in groups:
            columns.update(_calendar_block(checkins))
        if GROUP_CITY in groups:
            columns["city_code"] = cities.astype(np.float64)
            columns["province_code"] = keys[CITY].map(self.province_of).to_numpy(dtype=np.float64)
            merged = self.data.cities.set_index(CITY).reindex(cities)
            columns["lat"] = merged["lat"].to_numpy()
            columns["long"] = merged["long"].to_numpy()
            columns.update(self.city_history.block(cities, checkins.dayofweek.to_numpy()))
        if GROUP_MARKET in groups:
            block = _pickup_block(self.market, horizon, prefix="market_")
            block.update(_velocity_block(self.market, horizon, block, prefix="market_"))
            for name, values in block.items():
                columns[name] = values[self.market_rows]
            columns["city_share_of_market"] = observed / (
                columns["market_observed_total"] + EPS
            )
        if GROUP_PROVINCE in groups:
            block = _pickup_block(self.province, horizon, prefix="province_")
            block.update(_velocity_block(self.province, horizon, block, prefix="province_"))
            for name, values in block.items():
                columns[name] = values[self.province_rows]
            columns["city_share_of_province"] = observed / (
                columns["province_observed_total"] + EPS
            )
        if GROUP_CLUSTER in groups:
            table = self.data.cities.set_index(CITY)
            missing = [name for name in CLUSTER_FEATURES if name not in table.columns]
            if missing:
                raise ValueError(
                    "the cluster feature group needs a panel built by "
                    f"aggregate.aggregate_data; missing column(s): {missing}"
                )
            block = table.reindex(cities)
            for name in CLUSTER_FEATURES:
                columns[name] = block[name].to_numpy(dtype=np.float64)

        # Keys are deliberately NOT attached here: `city_code` is also a
        # feature, and the caller owns the mapping from row to pair.
        return pd.DataFrame(columns)


def feature_names(groups: tuple[str, ...]) -> list[str]:
    """The columns `at_horizon` will produce for a set of groups (for tests)."""
    names = ["observed_total", "days_to_checkin"]
    if GROUP_PICKUP in groups:
        names += [f"pickup_{w}d" for w in (1, 3, 7, 14)]
    if GROUP_VELOCITY in groups:
        names += [f"pickup_velocity_{w}d" for w in (3, 7, 14)]
        names += [f"pickup_acceleration_{w}d" for w in (3, 7)]
        names += [f"pickup_{w}d_share" for w in (1, 3, 7)] + ["pickup_3d_over_7d"]
    if GROUP_ACTIVITY in groups:
        names += [
            "active_search_days",
            "days_since_first_search",
            "days_since_last_search",
            "max_daily_search",
            "mean_daily_search",
            "std_daily_search",
            "search_span",
        ]
    if GROUP_CURVE in groups:
        names += [
            "expected_completion_fraction",
            "expected_remaining_fraction",
            "pickup_baseline_prediction",
            "pickup_baseline_remaining",
            "pickup_surprise",
        ]
    if GROUP_CALENDAR in groups:
        names += [
            "checkin_weekday",
            "is_weekend",
            "checkin_day_of_month",
            "jalali_month",
            "jalali_day",
        ]
    if GROUP_CITY in groups:
        names += [
            "city_code",
            "province_code",
            "lat",
            "long",
            "city_hist_mean",
            "city_hist_median",
            "city_hist_p75",
            "city_hist_p90",
            "city_hist_std",
            "city_volatility",
            "city_weekend_ratio",
            "city_weekday_mean",
        ]
    if GROUP_MARKET in groups:
        names += ["market_observed_total"] + [f"market_pickup_{w}d" for w in (1, 3, 7, 14)]
        names += [f"market_pickup_velocity_{w}d" for w in (3, 7, 14)]
        names += [f"market_pickup_acceleration_{w}d" for w in (3, 7)]
        names += ["city_share_of_market"]
    if GROUP_PROVINCE in groups:
        names += ["province_observed_total"] + [
            f"province_pickup_{w}d" for w in (1, 3, 7, 14)
        ]
        names += [f"province_pickup_velocity_{w}d" for w in (3, 7, 14)]
        names += [f"province_pickup_acceleration_{w}d" for w in (3, 7)]
        names += ["city_share_of_province"]
    if GROUP_CLUSTER in groups:
        names += list(CLUSTER_FEATURES)
    return names
