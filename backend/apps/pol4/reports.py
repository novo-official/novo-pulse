"""CSV reports, built from the same artefacts the charts read.

A downloaded report and the chart above it must never disagree, so both come
from `services`, not from separate queries. Filters narrow the rows; they never
change how a number is computed.
"""
from __future__ import annotations

from typing import Any, Callable

import pandas as pd

from . import services

REPORTS: dict[str, str] = {
    "demand": "Predicted demand for every city and Azar check-in date",
    "cities": "Per-city totals, ranked, with pickup momentum",
    "provinces": "Demand rolled up to the seven provinces",
    "peaks": "The strongest predicted (city, check-in) pairs",
    "pickup": "Observed pickup so far against each city's expectation",
    "stability": "Forecast snapshots from D-30 to D-1",
    "model": "Champion vs baseline, by fold and by horizon",
}


def _apply_filters(
    frame: pd.DataFrame,
    city: str | None,
    province: str | None,
    start: str | None,
    end: str | None,
) -> pd.DataFrame:
    out = frame
    if city and "city" in out.columns:
        needle = str(city).strip().lower()
        out = out[
            (out["city"].str.lower() == needle)
            | (out["city_code"].astype(str) == str(city).strip())
        ]
    if province and "province" in out.columns:
        out = out[out["province"].str.lower() == str(province).strip().lower()]
    if "checkin" in out.columns:
        stamps = pd.to_datetime(out["checkin"])
        if start:
            out = out[stamps >= pd.Timestamp(start)]
        if end:
            out = out[pd.to_datetime(out["checkin"]) <= pd.Timestamp(end)]
    return out


def _demand() -> pd.DataFrame:
    frame = services.forecast().copy()
    frame["checkin"] = frame["checkin"].dt.strftime("%Y-%m-%d")
    return frame


def _cities() -> pd.DataFrame:
    return services.city_index()


def _provinces() -> pd.DataFrame:
    return pd.DataFrame(services.provinces())


def _peaks() -> pd.DataFrame:
    frame = _demand()
    return frame.sort_values("predicted_demand", ascending=False, ignore_index=True)


def _pickup() -> pd.DataFrame:
    return services.momentum()


def _stability() -> pd.DataFrame:
    frame = services.stability().copy()
    frame["checkin"] = frame["checkin"].dt.strftime("%Y-%m-%d")
    return frame


def _model() -> pd.DataFrame:
    performance = services.model_performance()
    rows: list[dict[str, Any]] = []
    for fold in performance["folds"]:
        rows.append(
            {
                "section": "fold",
                "key": fold["cutoff"],
                "champion_wape": fold["champion_wape"],
                "baseline_wape": fold["baseline_wape"],
                "actual_total": fold["actual_total"],
            }
        )
    for bucket in performance["by_horizon_bucket"]:
        rows.append(
            {
                "section": "horizon",
                "key": bucket["key"],
                "champion_wape": bucket["wape"],
                "baseline_wape": next(
                    (
                        b["wape"]
                        for b in performance["baseline_by_horizon_bucket"]
                        if b["key"] == bucket["key"]
                    ),
                    None,
                ),
                "actual_total": bucket["actual_total"],
            }
        )
    for province in performance["by_province"]:
        rows.append(
            {
                "section": "province",
                "key": province["key"],
                "champion_wape": province["wape"],
                "baseline_wape": None,
                "actual_total": province["actual_total"],
            }
        )
    return pd.DataFrame(rows)


BUILDERS: dict[str, Callable[[], pd.DataFrame]] = {
    "demand": _demand,
    "cities": _cities,
    "provinces": _provinces,
    "peaks": _peaks,
    "pickup": _pickup,
    "stability": _stability,
    "model": _model,
}


def build(
    kind: str,
    city: str | None = None,
    province: str | None = None,
    start: str | None = None,
    end: str | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    if kind not in BUILDERS:
        raise KeyError(f"Unknown report '{kind}'. Available: {sorted(BUILDERS)}")
    frame = _apply_filters(BUILDERS[kind](), city, province, start, end)
    if limit:
        frame = frame.head(int(limit))
    return frame.reset_index(drop=True)
