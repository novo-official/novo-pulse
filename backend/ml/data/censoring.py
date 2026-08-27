"""Demand censoring / unconstrained demand.

Observed bookings are not market demand. When a listing sells out, the number
we record is the *capacity*, not the demand that existed:

    observed = min(latent_demand, available_capacity)

A model trained on `observed` therefore learns a systematically
under-stated picture of any period where supply binds. That matters commercially:
a destination that looks "flat" may actually be flat only because it is full.

What this module does:

* measures how much of the history is censored (the share of periods sitting at
  or above capacity),
* estimates unconstrained demand for those periods, and
* reports which forecast periods are expected to hit the capacity ceiling.

What it deliberately does not do: silently replace the target. Uncensoring is
an *assumption*, so the adjusted series is reported alongside the observed one
rather than substituted for it, and the whole feature turns off when the
dataset has no capacity column.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..contract import ENTITY, TARGET, TS, DataContract

# A period counts as censored when observed demand reaches this share of the
# capacity available that day.
CENSORING_RATIO = 0.98


@dataclass
class CensoringReport:
    enabled: bool
    reason: str
    capacity_column: str | None = None
    censored_share: float = 0.0
    censored_periods: int = 0
    affected_entities: int = 0
    mean_uplift: float = 0.0
    by_entity: list[dict[str, Any]] = None  # type: ignore[assignment]

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "reason": self.reason,
            "capacity_column": self.capacity_column,
            "censored_share": round(float(self.censored_share), 4),
            "censored_periods": int(self.censored_periods),
            "affected_entities": int(self.affected_entities),
            "mean_uplift": round(float(self.mean_uplift), 4),
            "by_entity": (self.by_entity or [])[:20],
            "method": (
                "observed >= {:.0%} of available capacity is treated as censored; "
                "unconstrained demand is estimated from the entity's uncensored "
                "periods at the same seasonal phase".format(CENSORING_RATIO)
            ),
        }


def analyse(frame: pd.DataFrame, contract: DataContract, season: int = 7) -> CensoringReport:
    """Measure censoring in the observed history."""
    options = contract.target_options
    column = options.censoring_column

    if not options.censoring_enabled:
        return CensoringReport(False, "censoring is disabled in the contract")
    if not column:
        return CensoringReport(False, "no censoring (capacity) column is configured")
    if column not in frame.columns:
        return CensoringReport(
            False, f"the configured capacity column '{column}' is not in the dataset", column
        )
    if not pd.api.types.is_numeric_dtype(frame[column]):
        return CensoringReport(False, f"capacity column '{column}' is not numeric", column)

    work = frame.loc[:, [ENTITY, TS, TARGET, column]].copy()
    capacity = work[column].to_numpy(dtype=float)
    observed = work[TARGET].to_numpy(dtype=float)
    valid = np.isfinite(capacity) & (capacity > 0) & np.isfinite(observed)
    if valid.sum() == 0:
        return CensoringReport(False, "capacity column has no usable values", column)

    censored = np.zeros(len(work), dtype=bool)
    censored[valid] = observed[valid] >= capacity[valid] * CENSORING_RATIO
    work["is_censored"] = censored

    share = float(censored[valid].mean())
    per_entity = (
        work.groupby(ENTITY)["is_censored"]
        .agg(["mean", "sum", "count"])
        .rename(columns={"mean": "share", "sum": "periods", "count": "observations"})
        .sort_values("share", ascending=False)
    )
    affected = int((per_entity["periods"] > 0).sum())

    uplift = estimate_unconstrained(work, season)
    mean_uplift = 0.0
    if censored.any():
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = uplift[censored] / np.maximum(observed[censored], 1e-9)
        finite = ratio[np.isfinite(ratio)]
        mean_uplift = float(np.mean(finite) - 1.0) if finite.size else 0.0

    return CensoringReport(
        enabled=True,
        reason="capacity column present; censoring measured",
        capacity_column=column,
        censored_share=share,
        censored_periods=int(censored.sum()),
        affected_entities=affected,
        mean_uplift=mean_uplift,
        by_entity=[
            {
                "entity_id": str(entity),
                "censored_share": round(float(row.share), 4),
                "censored_periods": int(row.periods),
            }
            for entity, row in per_entity.head(20).iterrows()
            if row.periods > 0
        ],
    )


def estimate_unconstrained(work: pd.DataFrame, season: int = 7) -> np.ndarray:
    """Estimate demand for censored periods.

    For each censored period we take the entity's median demand across its
    *uncensored* periods at the same seasonal phase (same weekday for daily
    data) and use the larger of that and the observed value - demand was at
    least what we saw. Where an entity has no uncensored period at that phase
    we fall back to its overall uncensored median, and finally to the observed
    value itself, which makes the estimate a no-op rather than a guess.
    """
    frame = work.copy()
    frame["__phase"] = _phase(frame[TS], season)
    observed = frame[TARGET].to_numpy(dtype=float)
    estimate = observed.copy()

    uncensored = frame.loc[~frame["is_censored"]]
    if uncensored.empty:
        return estimate

    phase_median = uncensored.groupby([ENTITY, "__phase"])[TARGET].median()
    entity_median = uncensored.groupby(ENTITY)[TARGET].median()

    censored_rows = np.where(frame["is_censored"].to_numpy())[0]
    entities = frame[ENTITY].to_numpy()
    phases = frame["__phase"].to_numpy()

    for index in censored_rows:
        key = (entities[index], phases[index])
        reference = phase_median.get(key, np.nan)
        if not np.isfinite(reference):
            reference = entity_median.get(entities[index], np.nan)
        if np.isfinite(reference):
            estimate[index] = max(observed[index], float(reference))
    return estimate


def _phase(stamps: pd.Series, season: int) -> np.ndarray:
    stamps = pd.to_datetime(stamps)
    if season == 7:
        return stamps.dt.dayofweek.to_numpy()
    if season == 12:
        return stamps.dt.month.to_numpy()
    if season == 24:
        return stamps.dt.hour.to_numpy()
    return stamps.dt.dayofyear.to_numpy() % max(season, 1)


def capacity_constrained_forecasts(
    forecast: pd.DataFrame, capacity: pd.DataFrame, ratio: float = CENSORING_RATIO
) -> pd.DataFrame:
    """Flag forecast periods expected to hit the capacity ceiling.

    These are the periods where the forecast understates demand for the same
    reason the history did - useful for a revenue team deciding where extra
    supply would actually sell.
    """
    if forecast.empty or capacity.empty:
        return forecast.assign(capacity_constrained=False)

    merged = forecast.merge(capacity, on=[ENTITY, "ds"], how="left")
    ceiling = merged["capacity"].to_numpy(dtype=float)
    predicted = merged["forecast"].to_numpy(dtype=float)
    with np.errstate(invalid="ignore"):
        constrained = np.isfinite(ceiling) & (ceiling > 0) & (predicted >= ceiling * ratio)
    merged["capacity_constrained"] = constrained
    return merged
