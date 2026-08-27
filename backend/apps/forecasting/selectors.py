"""Query helpers shared by the forecasting endpoints.

Kept out of the views so the API layer stays thin and the same filtering logic
can be reused by the scenario engine and the report generator.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ml.contract import ENTITY

from .store import ArtifactStore


def parse_filters(params) -> dict[str, Any]:
    """Read the standard forecast filters off a query string."""
    level = params.get("level") or params.get("entity_level") or "destination"
    entity = params.get("id") or params.get("entity_id")
    return {
        "level": level,
        "entity_id": entity if entity not in {"", "all", None} else None,
        "start_date": _date(params.get("start_date")),
        "end_date": _date(params.get("end_date")),
        "horizon": _int(params.get("horizon")),
        "model": params.get("model") or None,
        "limit": _int(params.get("limit")),
    }


def _date(value: str | None) -> pd.Timestamp | None:
    if not value:
        return None
    try:
        return pd.Timestamp(value)
    except (ValueError, TypeError):
        return None


def _int(value: str | None) -> int | None:
    try:
        return int(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def resolve_level(store: ArtifactStore, requested: str | None) -> str:
    """Fall back to the finest level the dataset actually has."""
    levels = store.levels()
    if requested in levels:
        return requested
    for candidate in ("destination", "listing", "category", "market"):
        if candidate in levels:
            return candidate
    return levels[0] if levels else "listing"


def filter_frame(
    frame: pd.DataFrame,
    entity_id: str | None = None,
    start_date: pd.Timestamp | None = None,
    end_date: pd.Timestamp | None = None,
    horizon: int | None = None,
    model: str | None = None,
) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame
    if entity_id:
        out = out[out[ENTITY].astype(str) == str(entity_id)]
    if start_date is not None and "ds" in out.columns:
        out = out[out["ds"] >= start_date]
    if end_date is not None and "ds" in out.columns:
        out = out[out["ds"] <= end_date]
    if horizon and "horizon" in out.columns:
        out = out[out["horizon"] <= horizon]
    if model and "model" in out.columns:
        out = out[out["model"] == model]
    return out


def collapse_to_series(frame: pd.DataFrame, value_columns: list[str]) -> pd.DataFrame:
    """Sum a multi-entity frame into a single daily series."""
    if frame.empty:
        return frame
    present = [c for c in value_columns if c in frame.columns]
    return frame.groupby("ds", as_index=False)[present].sum().sort_values("ds")


def records(frame: pd.DataFrame, columns: list[str] | None = None) -> list[dict[str, Any]]:
    """JSON-safe records with ISO dates and no NaNs."""
    if frame.empty:
        return []
    out = frame.copy()
    if columns:
        out = out.loc[:, [c for c in columns if c in out.columns]]
    for column in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[column]):
            out[column] = out[column].dt.strftime("%Y-%m-%d")
    out = out.replace({np.nan: None, np.inf: None, -np.inf: None})
    return out.to_dict(orient="records")


def history_window(
    store: ArtifactStore, level: str, entity_id: str | None, periods: int, until: pd.Timestamp | None
) -> pd.DataFrame:
    """The trailing slice of actuals shown behind the forecast."""
    history = store.level_frame("history", level)
    if history.empty:
        return history
    if entity_id:
        history = history[history[ENTITY].astype(str) == str(entity_id)]
    if until is not None:
        history = history[history["ds"] <= until]
    series = collapse_to_series(history, ["y"])
    return series.tail(periods) if periods else series
