"""Pre-aggregated analytics for the dashboard.

The dashboard must never touch `search_data.csv` - 3.3 million rows is not a
page load. So the pipeline computes everything the product needs once, offline,
and writes it next to the forecast. Every figure the UI shows traces back to one
of these files, and each is small enough to serve from memory.

    target_pickup.parquet   observed cumulative demand by days-to-check-in for
                            every Azar (city, check-in) - the "so far" line of
                            the pickup chart
    city_momentum.parquet   per city: how the last week's pickup compares with
                            what its own history says to expect at this lead
                            time - the emerging-demand signal
    city_history.parquet    per city: recent daily completed demand, for context
                            behind the forecast
    province_summary.json   forecast rolled up to the seven provinces
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import Pol4Config
from .loader import CHECKIN, CITY, DTC, SEARCHES, Pol4Data
from .pickup import PickupCurves

log = logging.getLogger(__name__)

#: Days of completed history kept per city for the context panel.
HISTORY_DAYS = 180
#: Lead-time window used to measure "recent" pickup.
MOMENTUM_WINDOW = 7


def target_pickup(data: Pol4Data, config: Pol4Config) -> pd.DataFrame:
    """Observed cumulative demand by lead time, per Azar (city, check-in).

    Read straight off `evaluation.csv`, which is exactly the partial history the
    competition hands us at the cutoff. Cumulative is taken from the far end
    inwards, so the value at lead time `h` is what a forecaster standing `h`
    days before the check-in would have seen.
    """
    rows = data.evaluation
    daily = (
        rows.groupby([CITY, CHECKIN, DTC], as_index=False)[SEARCHES]
        .sum()
        .sort_values([CITY, CHECKIN, DTC], ascending=[True, True, False])
    )
    daily["observed_cumulative"] = daily.groupby([CITY, CHECKIN])[SEARCHES].cumsum()
    daily = daily.rename(columns={SEARCHES: "searches", DTC: "days_to_checkin"})
    return data.label(daily).reset_index(drop=True)


def city_momentum(data: Pol4Data, config: Pol4Config) -> pd.DataFrame:
    """How hard each city is picking up, relative to its own history.

    `pickup_ratio` compares the demand observed in the last `MOMENTUM_WINDOW`
    days against what this city's completion curve says should have arrived over
    the same stretch. Above 1 means the city is running hot for this far out.

    This is a measured ratio, not a claim about why: it says demand is arriving
    faster than usual, and nothing about the cause.
    """
    curves = PickupCurves.fit(data, config.cutoff, config)
    observed = data.observed_at(
        config.cutoff, config.target_start, config.target_end
    )
    grid = pd.MultiIndex.from_product(
        [data.city_codes, config.target_dates()], names=[CITY, CHECKIN]
    ).to_frame(index=False)
    frame = grid.merge(observed, on=[CITY, CHECKIN], how="left").fillna({"observed": 0.0})
    frame["horizon"] = (frame[CHECKIN] - config.cutoff).dt.days

    recent = data.evaluation[
        data.evaluation["log_date"] > config.cutoff - pd.Timedelta(days=MOMENTUM_WINDOW)
    ]
    recent = (
        recent.groupby([CITY, CHECKIN], as_index=False)[SEARCHES]
        .sum()
        .rename(columns={SEARCHES: "recent_pickup"})
    )
    frame = frame.merge(recent, on=[CITY, CHECKIN], how="left").fillna({"recent_pickup": 0.0})

    cities = frame[CITY].to_numpy()
    horizon = frame["horizon"].to_numpy()
    # Expected share of final demand arriving between h + window and h.
    at_h = curves.fraction(cities, horizon)
    at_h_plus = curves.fraction(cities, horizon + MOMENTUM_WINDOW)
    frame["expected_window_share"] = np.maximum(at_h - at_h_plus, 1e-9)
    frame["expected_pickup"] = (
        frame["observed"] / np.maximum(at_h, 1e-9)
    ) * frame["expected_window_share"]

    per_city = frame.groupby(CITY, as_index=False).agg(
        observed=("observed", "sum"),
        recent_pickup=("recent_pickup", "sum"),
        expected_pickup=("expected_pickup", "sum"),
    )
    per_city["pickup_ratio"] = per_city["recent_pickup"] / per_city["expected_pickup"].replace(
        0.0, np.nan
    )
    per_city["curve_source"] = curves.source(per_city[CITY].to_numpy())
    return data.label(per_city).reset_index(drop=True)


def city_history(data: Pol4Data, config: Pol4Config, days: int = HISTORY_DAYS) -> pd.DataFrame:
    """Recent completed daily demand per city, for context behind the forecast."""
    start = config.cutoff - pd.Timedelta(days=days - 1)
    demand = data.final_demand(start, config.cutoff).rename(columns={"final": "demand"})
    grid = pd.MultiIndex.from_product(
        [data.city_codes, pd.date_range(start, config.cutoff, freq="D")],
        names=[CITY, CHECKIN],
    ).to_frame(index=False)
    frame = grid.merge(demand, on=[CITY, CHECKIN], how="left").fillna({"demand": 0.0})
    return data.label(frame).reset_index(drop=True)


def province_summary(predictions: pd.DataFrame, data: Pol4Data) -> list[dict[str, Any]]:
    """The forecast rolled up to the seven provinces.

    Rounded to integers *before* aggregating, exactly as `build_submission`
    does, so the province chart and results.csv report the same in-panel total.
    Summing the unrounded predictions instead drifts by tens of searches - small,
    but enough that a judge adding up the provinces would not get the KPI.
    """
    rounded = predictions[[CITY, "observed", "predicted_demand"]].copy()
    rounded["predicted_demand"] = np.rint(rounded["predicted_demand"])
    rounded["observed"] = np.rint(rounded["observed"])
    labelled = data.label(rounded)
    grouped = labelled.groupby("province", as_index=False).agg(
        predicted_demand=("predicted_demand", "sum"),
        observed_so_far=("observed", "sum"),
        cities=(CITY, "nunique"),
    )
    grouped["predicted_remaining"] = (
        grouped["predicted_demand"] - grouped["observed_so_far"]
    )
    total = grouped["predicted_demand"].sum()
    grouped["share"] = grouped["predicted_demand"] / total if total else 0.0
    return (
        grouped.sort_values("predicted_demand", ascending=False)
        .round(2)
        .to_dict(orient="records")
    )


@dataclass
class AnalyticsBundle:
    target_pickup: pd.DataFrame
    city_momentum: pd.DataFrame
    city_history: pd.DataFrame
    provinces: list[dict[str, Any]]

    def write(self, directory: Path) -> dict[str, str]:
        directory.mkdir(parents=True, exist_ok=True)
        self.target_pickup.to_parquet(directory / "target_pickup.parquet", index=False)
        self.city_momentum.to_parquet(directory / "city_momentum.parquet", index=False)
        self.city_history.to_parquet(directory / "city_history.parquet", index=False)
        (directory / "province_summary.json").write_text(
            json.dumps(self.provinces, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return {
            "target_pickup": "target_pickup.parquet",
            "city_momentum": "city_momentum.parquet",
            "city_history": "city_history.parquet",
            "provinces": "province_summary.json",
        }


def build(
    data: Pol4Data, predictions: pd.DataFrame, config: Pol4Config
) -> AnalyticsBundle:
    """Everything the dashboard reads, computed once."""
    log.info("building dashboard analytics")
    return AnalyticsBundle(
        target_pickup=target_pickup(data, config),
        city_momentum=city_momentum(data, config),
        city_history=city_history(data, config),
        provinces=province_summary(predictions, data),
    )
