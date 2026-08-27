"""Timestamp frequency detection."""
from __future__ import annotations

import numpy as np
import pandas as pd

# Median spacing (in days) -> pandas frequency alias.
_DAY_BUCKETS: list[tuple[float, float, str]] = [
    (0.0, 0.06, "h"),        # <= ~1.5h
    (0.06, 0.9, "h"),
    (0.9, 1.6, "D"),
    (1.6, 9.0, "W"),
    (9.0, 45.0, "MS"),
    (45.0, 130.0, "QS"),
    (130.0, 10_000.0, "YS"),
]


def detect_frequency(timestamps: pd.Series) -> dict:
    """Infer the sampling frequency of a timestamp column.

    Returns the inferred alias, a confidence score and whether the series is
    regularly spaced. The caller is always free to override the result.
    """
    stamps = pd.to_datetime(pd.Series(timestamps), errors="coerce").dropna()
    unique = np.sort(stamps.unique())
    if len(unique) < 3:
        return {
            "frequency": "D",
            "confidence": 0.0,
            "regular": False,
            "median_delta_days": None,
            "reason": "insufficient distinct timestamps",
        }

    deltas = np.diff(unique).astype("timedelta64[s]").astype(float) / 86400.0
    deltas = deltas[deltas > 0]
    if len(deltas) == 0:
        return {
            "frequency": "D",
            "confidence": 0.0,
            "regular": False,
            "median_delta_days": None,
            "reason": "all timestamps identical",
        }

    median = float(np.median(deltas))
    alias = next((a for lo, hi, a in _DAY_BUCKETS if lo < median <= hi), "D")
    share_at_median = float(np.mean(np.abs(deltas - median) <= max(median * 0.15, 1e-6)))

    pandas_guess = None
    try:
        pandas_guess = pd.infer_freq(pd.DatetimeIndex(unique))
    except (ValueError, TypeError):
        pandas_guess = None

    return {
        "frequency": alias,
        "pandas_inferred": pandas_guess,
        "confidence": round(share_at_median, 3),
        "regular": bool(share_at_median > 0.95),
        "median_delta_days": round(median, 4),
        "n_timestamps": int(len(unique)),
        "reason": (
            "regular spacing" if share_at_median > 0.95 else "irregular spacing - aggregation advised"
        ),
    }
