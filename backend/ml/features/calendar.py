"""Calendar features.

Everything here is derived from the timestamp alone, so it is known for future
dates by construction and can never leak.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Iran's weekend: Thursday(3) / Friday(4) with pandas' Monday=0 numbering.
WEEKEND_DAYS = (3, 4)


def calendar_frame(index: pd.DatetimeIndex, freq: str = "D") -> pd.DataFrame:
    """Build the calendar feature block for a date index."""
    frame = pd.DataFrame(index=range(len(index)))
    frame["cal_year"] = index.year.to_numpy()
    frame["cal_quarter"] = index.quarter.to_numpy()
    frame["cal_month"] = index.month.to_numpy()
    frame["cal_week_of_year"] = index.isocalendar().week.to_numpy().astype(int)
    frame["cal_day_of_month"] = index.day.to_numpy()
    frame["cal_day_of_year"] = index.dayofyear.to_numpy()
    frame["cal_day_of_week"] = index.dayofweek.to_numpy()
    frame["cal_is_weekend"] = np.isin(index.dayofweek.to_numpy(), WEEKEND_DAYS).astype(np.int8)
    frame["cal_is_month_start"] = index.is_month_start.astype(np.int8)
    frame["cal_is_month_end"] = index.is_month_end.astype(np.int8)

    # Cyclic encodings so the model sees December and January as neighbours.
    frame["cal_sin_dow"] = np.sin(2 * np.pi * frame["cal_day_of_week"] / 7)
    frame["cal_cos_dow"] = np.cos(2 * np.pi * frame["cal_day_of_week"] / 7)
    frame["cal_sin_month"] = np.sin(2 * np.pi * frame["cal_month"] / 12)
    frame["cal_cos_month"] = np.cos(2 * np.pi * frame["cal_month"] / 12)
    frame["cal_sin_doy"] = np.sin(2 * np.pi * frame["cal_day_of_year"] / 365.25)
    frame["cal_cos_doy"] = np.cos(2 * np.pi * frame["cal_day_of_year"] / 365.25)

    if freq in {"W", "MS"}:
        # Sub-weekly features carry no information at these frequencies.
        frame = frame.drop(
            columns=[
                "cal_day_of_week",
                "cal_is_weekend",
                "cal_sin_dow",
                "cal_cos_dow",
                "cal_day_of_month",
            ],
            errors="ignore",
        )
    if freq == "h":
        frame["cal_hour"] = index.hour.to_numpy()
        frame["cal_sin_hour"] = np.sin(2 * np.pi * frame["cal_hour"] / 24)
        frame["cal_cos_hour"] = np.cos(2 * np.pi * frame["cal_hour"] / 24)
    return frame


def event_distance_features(flag: np.ndarray, cap: int = 60) -> dict[str, np.ndarray]:
    """Days until / since the next / previous flagged day.

    `flag` must be known for the whole timeline including future dates (it is
    built from calendar covariates such as `is_holiday`), so these features are
    known-future by construction.
    """
    binary = (np.nan_to_num(np.asarray(flag, dtype=float)) > 0).astype(np.int8)
    n = len(binary)
    forward = np.full(n, cap, dtype=np.float32)
    backward = np.full(n, cap, dtype=np.float32)

    running = cap
    for i in range(n - 1, -1, -1):
        running = 0 if binary[i] else min(running + 1, cap)
        forward[i] = running

    running = cap
    for i in range(n):
        running = 0 if binary[i] else min(running + 1, cap)
        backward[i] = running

    return {"days_to": forward, "days_from": backward}


def is_event_like(name: str) -> bool:
    """Does this covariate look like a holiday/event flag?"""
    lowered = name.lower()
    return any(token in lowered for token in ("holiday", "event", "vacation", "festival"))


def is_price_like(name: str) -> bool:
    lowered = name.lower()
    return any(token in lowered for token in ("price", "rate", "fare", "cost", "tariff"))
