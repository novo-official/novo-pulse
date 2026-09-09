"""The pickup baseline: project a partially-observed check-in to its final total.

    predicted_final = observed_so_far / expected_completion_fraction(city, h)

This is the model the audit measured at WAPE 0.185 - better than the generic
platform's LightGBM champion (0.262) - and it is deliberately the first thing
built, because every later model has to beat it to justify its complexity.

Two robustness rules matter more than the arithmetic:

* `predicted_final >= observed_so_far`. A pair cannot end below what has
  already been counted; a projection that says otherwise is an estimation
  error, not a forecast.
* A pair with zero searches so far is NOT a pair with zero final demand. At the
  real cutoff, 82 of 321 cities and 4,282 of 9,630 target pairs have nothing
  observed yet, and most of them will still receive searches. Those fall back
  to the city's own historical demand level.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .config import Pol4Config
from .loader import CHECKIN, CITY, DTC, PROVINCE, SEARCHES, Pol4Data
from .pickup import PickupCurves

ZERO_POLICIES = ("prior", "zero")


@dataclass
class DemandPrior:
    """Historical demand level per city, used only where nothing is observed.

    Built from the same cutoff-safe history as the pickup curves: check-ins at
    or before the cutoff, whose demand is complete.
    """

    cutoff: pd.Timestamp
    by_city_weekday: dict[tuple[int, int], float]
    by_city: dict[int, float]
    by_province: dict[int, float]
    fallback: float

    @classmethod
    def fit(
        cls, data: Pol4Data, cutoff: pd.Timestamp, config: Pol4Config | None = None
    ) -> "DemandPrior":
        config = config or Pol4Config()
        cutoff = pd.Timestamp(cutoff)
        history = data.history_before(cutoff)
        window_start = cutoff - pd.Timedelta(days=config.prior_window_days - 1)
        window = pd.date_range(max(window_start, history[CHECKIN].min()), cutoff, freq="D")
        if len(window) == 0:
            window = pd.DatetimeIndex([cutoff])

        recent = history[history[CHECKIN].isin(window)]
        demand = recent.groupby([CITY, CHECKIN], as_index=False)[SEARCHES].sum()

        # A check-in date with no row had ZERO searches, so the denominator has
        # to be every date in the window - not just the dates that produced a
        # row. Averaging over present rows only is the same "missing means
        # zero" trap the competition sets, and for a sparse city it overstates
        # the level by an order of magnitude.
        n_days = len(window)
        by_city = (demand.groupby(CITY)[SEARCHES].sum() / n_days).to_dict()

        weekday_counts = pd.Series(window.dayofweek).value_counts().to_dict()
        demand = demand.assign(weekday=demand[CHECKIN].dt.dayofweek)
        weekday_totals = demand.groupby([CITY, "weekday"])[SEARCHES].sum()
        by_city_weekday = {
            (int(city), int(weekday)): float(total) / weekday_counts.get(int(weekday), 1)
            for (city, weekday), total in weekday_totals.items()
        }

        province_of = data.province_of
        per_city = pd.Series(by_city, name="mean").rename_axis(CITY).reset_index()
        per_city[PROVINCE] = per_city[CITY].map(province_of)
        # The median city of the province, not the province total: this stands in
        # for one city, so a province sum would over-predict by ~50x.
        by_province = per_city.groupby(PROVINCE)["mean"].median().to_dict()

        return cls(
            cutoff=cutoff,
            by_city_weekday=by_city_weekday,
            by_city={int(c): float(v) for c, v in by_city.items()},
            by_province={int(p): float(v) for p, v in by_province.items() if pd.notna(p)},
            fallback=float(per_city["mean"].median()) if len(per_city) else 0.0,
        )

    def level(self, city_codes, checkins, province_of: dict[int, int]) -> np.ndarray:
        """City x weekday mean, falling back to city, province, then global."""
        cities = np.asarray(city_codes, dtype=np.int64)
        weekdays = pd.DatetimeIndex(checkins).dayofweek.to_numpy()

        out = np.full(len(cities), np.nan, dtype=np.float64)
        for position, (city, weekday) in enumerate(zip(cities, weekdays)):
            value = self.by_city_weekday.get((int(city), int(weekday)))
            if value is None:
                value = self.by_city.get(int(city))
            if value is None:
                value = self.by_province.get(province_of.get(int(city), -1))
            out[position] = self.fallback if value is None else value
        return np.maximum(out, 0.0)


@dataclass
class ZeroObservationPrior:
    """What a pair is worth when nothing has been searched for it yet.

    "Zero so far" is not "zero in the end": at the real cutoff 4,282 of the
    9,630 target pairs have no observed searches, and historically those pairs
    still finish with real demand. But the answer is small and horizon-shaped,
    not a generic demand level - a city with nothing booked 30 days out is a
    very different proposition from one with nothing booked tomorrow.

    So this estimates the conditional expectation directly:

        E[ final_demand | nothing observed at horizon h ]

    over completed historical check-ins, per city, falling back to province and
    then global. A pair is unobserved at horizon h exactly when its first
    search landed later than h days before check-in, i.e. when
    `max(days_to_checkin) < h`, which makes the whole table one grouped
    cumulative sum rather than a scan over horizons.
    """

    cutoff: pd.Timestamp
    horizons: int
    by_city: dict[int, np.ndarray]
    by_province: dict[int, np.ndarray]
    global_curve: np.ndarray

    @classmethod
    def fit(
        cls,
        data: Pol4Data,
        cutoff: pd.Timestamp,
        config: Pol4Config | None = None,
        horizons: int | None = None,
    ) -> "ZeroObservationPrior":
        config = config or Pol4Config()
        cutoff = pd.Timestamp(cutoff)
        horizons = int(horizons or config.target_days)
        history = data.history_before(cutoff)

        pairs = history.groupby([CITY, CHECKIN]).agg(
            final=(SEARCHES, "sum"), max_dtc=(DTC, "max")
        )

        # Pairs absent from the log are the purest form of "nothing observed":
        # final demand zero, no search ever. Leaving them out would bias the
        # conditional expectation upward, which is the mistake this class exists
        # to avoid, so the grid is completed before averaging.
        dates = pd.date_range(history[CHECKIN].min(), cutoff, freq="D")
        grid = pd.MultiIndex.from_product([data.city_codes, dates], names=[CITY, CHECKIN])
        pairs = pairs.reindex(grid, fill_value=0)
        pairs["max_dtc"] = pairs["max_dtc"].where(pairs["final"] > 0, -1).astype(np.int64)
        pairs = pairs.reset_index()

        province_of = data.province_of
        pairs[PROVINCE] = pairs[CITY].map(province_of)
        # Bucket index 0 == "never searched"; index k == first search k-1 days out.
        pairs["bucket"] = np.clip(pairs["max_dtc"] + 1, 0, horizons)

        def table(key: str | None) -> tuple[dict[int, np.ndarray], np.ndarray]:
            group = [key, "bucket"] if key else ["bucket"]
            totals = pairs.groupby(group)["final"].agg(["sum", "count"])
            if key is None:
                totals = totals.reindex(range(horizons + 1), fill_value=0)
                cumulative = totals.cumsum()
                # prior(h) reads the pairs whose first search was later than h
                # days out, which is exactly buckets 0..h-1.
                curve = np.divide(
                    cumulative["sum"].to_numpy()[:horizons],
                    np.maximum(cumulative["count"].to_numpy()[:horizons], 1),
                )
                return {}, curve
            out: dict[int, np.ndarray] = {}
            for value, rows in totals.groupby(level=0):
                rows = rows.droplevel(0).reindex(range(horizons + 1), fill_value=0).cumsum()
                out[int(value)] = np.divide(
                    rows["sum"].to_numpy()[:horizons],
                    np.maximum(rows["count"].to_numpy()[:horizons], 1),
                )
            return out, np.zeros(horizons)

        by_city, _ = table(CITY)
        by_province, _ = table(PROVINCE)
        _, global_curve = table(None)

        return cls(
            cutoff=cutoff,
            horizons=horizons,
            by_city=by_city,
            by_province=by_province,
            global_curve=global_curve,
        )

    def level(self, city_codes, horizons, province_of: dict[int, int]) -> np.ndarray:
        """Expected final demand for pairs with nothing observed, per (city, h)."""
        cities = np.asarray(city_codes, dtype=np.int64)
        # prior(h) is indexed by "buckets 0..h-1", so horizon h reads slot h-1.
        slots = np.clip(np.asarray(horizons, dtype=np.int64) - 1, 0, self.horizons - 1)
        out = np.empty(len(cities), dtype=np.float64)
        for position, (city, slot) in enumerate(zip(cities, slots)):
            curve = self.by_city.get(int(city))
            if curve is None:
                curve = self.by_province.get(province_of.get(int(city), -1))
            if curve is None:
                curve = self.global_curve
            out[position] = curve[slot]
        return np.maximum(out, 0.0)


@dataclass
class PickupBaseline:
    """Fitted at a cutoff; predicts final demand for any (city, check-in) grid."""

    curves: PickupCurves
    prior: DemandPrior
    zero_prior: ZeroObservationPrior
    cutoff: pd.Timestamp
    config: Pol4Config

    @classmethod
    def fit(
        cls, data: Pol4Data, cutoff: pd.Timestamp, config: Pol4Config | None = None
    ) -> "PickupBaseline":
        config = config or Pol4Config()
        if config.zero_observation_policy not in ZERO_POLICIES:
            raise ValueError(
                f"zero_observation_policy must be one of {ZERO_POLICIES}, "
                f"got {config.zero_observation_policy!r}"
            )
        cutoff = pd.Timestamp(cutoff)
        return cls(
            curves=PickupCurves.fit(data, cutoff, config),
            prior=DemandPrior.fit(data, cutoff, config),
            zero_prior=ZeroObservationPrior.fit(data, cutoff, config),
            cutoff=cutoff,
            config=config,
        )

    # ------------------------------------------------------------- predict
    def predict(self, grid: pd.DataFrame) -> pd.DataFrame:
        """Project a grid of (city_code, checkin, observed) to final demand.

        Returns the grid with `horizon`, `completion_fraction`, `curve_source`,
        `projection` and `predicted_demand` added. Nothing is rounded here - the
        submission writer decides that.
        """
        required = {CITY, CHECKIN, "observed"}
        missing = required - set(grid.columns)
        if missing:
            raise ValueError(f"grid is missing column(s): {sorted(missing)}")

        out = grid.copy()
        out[CHECKIN] = pd.to_datetime(out[CHECKIN])
        out["horizon"] = (out[CHECKIN] - self.cutoff).dt.days.astype(np.int64)
        if (out["horizon"] < 1).any():
            raise ValueError(
                "every target check-in must be strictly after the cutoff "
                f"({self.cutoff.date()}); the grid contains earlier dates"
            )

        observed = out["observed"].to_numpy(dtype=np.float64)
        fraction = self.curves.fraction(out[CITY].to_numpy(), out["horizon"].to_numpy())
        out["completion_fraction"] = fraction
        out["curve_source"] = self.curves.source(out[CITY].to_numpy())

        projection = observed / fraction
        out["projection"] = projection

        # A pair cannot finish below what has already been counted.
        predicted = np.maximum(projection, observed)

        nothing_seen = observed <= 0
        if nothing_seen.any():
            if self.config.zero_observation_policy == "prior":
                predicted[nothing_seen] = self.zero_prior.level(
                    out.loc[nothing_seen, CITY].to_numpy(),
                    out.loc[nothing_seen, "horizon"].to_numpy(),
                    self.curves.province_of,
                )
            else:
                predicted[nothing_seen] = 0.0

        out["predicted_demand"] = np.maximum(np.nan_to_num(predicted, nan=0.0), 0.0)
        return out

    def summary(self) -> dict[str, Any]:
        return {
            "cutoff": self.cutoff.date().isoformat(),
            "zero_observation_policy": self.config.zero_observation_policy,
            "min_city_support": self.config.min_city_support,
            "min_province_support": self.config.min_province_support,
            "min_fraction": self.config.min_fraction,
            "curves": self.curves.summary(),
        }
