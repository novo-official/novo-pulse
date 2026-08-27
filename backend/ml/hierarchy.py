"""Hierarchical aggregation.

    listing -> destination -> category -> market

Bottom-up reconciliation: a forecast produced at the listing level is summed to
every higher level, which keeps the levels mutually consistent by construction.
When a dataset has no listing column the pipeline simply forecasts at whichever
level exists - no level is mandatory.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .contract import CATEGORY, DESTINATION, ENTITY, LEVELS, MARKET, MARKET_VALUE

# Columns summed on roll-up; everything else is averaged.
ADDITIVE = ("forecast", "lower", "upper", "actual", "prediction", "y")


def entity_map(panel_frame: pd.DataFrame) -> pd.DataFrame:
    """entity_id -> its destination / category / market keys."""
    columns = [ENTITY]
    for column in (DESTINATION, CATEGORY, MARKET):
        if column in panel_frame.columns:
            columns.append(column)
    mapping = panel_frame.loc[:, columns].drop_duplicates(subset=[ENTITY])
    if MARKET not in mapping.columns:
        mapping[MARKET] = MARKET_VALUE
    return mapping.reset_index(drop=True)


def available_levels(panel_frame: pd.DataFrame) -> list[str]:
    levels = ["listing"]
    if DESTINATION in panel_frame.columns:
        levels.append("destination")
    if CATEGORY in panel_frame.columns:
        levels.append("category")
    levels.append("market")
    return levels


def aggregate_bottom_up(
    frame: pd.DataFrame, mapping: pd.DataFrame, level: str, extra_keys: list[str] | None = None
) -> pd.DataFrame:
    """Roll a listing-level frame up to `level`.

    Interval bounds are summed too. That is deliberately conservative: summing
    P10s and P90s assumes the errors move together, which over-states the width
    of an aggregate interval. The dashboard labels aggregated intervals as
    bottom-up sums for exactly this reason.
    """
    if level == "listing":
        return frame.copy()
    column = {"destination": DESTINATION, "category": CATEGORY, "market": MARKET}.get(level)
    if column is None:
        raise ValueError(f"Unknown hierarchy level '{level}'")
    if column not in mapping.columns:
        raise ValueError(f"Level '{level}' is not available for this dataset")

    keys = ["ds", *(extra_keys or [])]
    merged = frame.merge(mapping[[ENTITY, column]], on=ENTITY, how="left")
    merged[column] = merged[column].fillna("unknown").astype(str)

    numeric = merged.select_dtypes(include=[np.number]).columns
    agg = {
        col: ("sum" if col in ADDITIVE else "mean")
        for col in numeric
        if col not in keys and col != column
    }
    if "horizon" in agg:
        agg["horizon"] = "first"
    grouped = merged.groupby([*keys, column], as_index=False).agg(agg)
    return grouped.rename(columns={column: ENTITY})


def aggregate_all_levels(
    frame: pd.DataFrame, mapping: pd.DataFrame, levels: list[str] | None = None
) -> pd.DataFrame:
    """Stack every requested level into one long frame with a `level` column."""
    pieces = []
    for level in levels or LEVELS:
        try:
            rolled = aggregate_bottom_up(frame, mapping, level)
        except ValueError:
            continue
        rolled["level"] = level
        pieces.append(rolled)
    if not pieces:
        return frame.assign(level="listing")
    return pd.concat(pieces, ignore_index=True)


def coherence_check(stacked: pd.DataFrame, value_column: str = "forecast") -> dict[str, float]:
    """Verify that the levels really do add up (a bottom-up sanity check)."""
    if "level" not in stacked.columns:
        return {}
    totals = stacked.groupby("level")[value_column].sum()
    if "listing" not in totals.index:
        return {}
    base = float(totals["listing"])
    out = {}
    for level, value in totals.items():
        if level == "listing" or base == 0:
            continue
        out[f"{level}_vs_listing_pct_diff"] = round(float((value - base) / base) * 100, 6)
    return out
