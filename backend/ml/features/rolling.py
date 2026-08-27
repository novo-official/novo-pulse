"""Causal rolling statistics over an (entity x time) matrix.

Every window ends *at and includes* the column it is written to, so reading
column `o` only ever mixes in data from `o - w + 1 .. o`. That is the property
the leakage tests assert.
"""
from __future__ import annotations

import numpy as np


def _cumulative(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.nan_to_num(arr, nan=0.0).astype(np.float64)
    valid = np.isfinite(arr).astype(np.float64)
    zeros = np.zeros((arr.shape[0], 1), dtype=np.float64)
    csum = np.concatenate([zeros, np.cumsum(values, axis=1)], axis=1)
    csum_sq = np.concatenate([zeros, np.cumsum(values**2, axis=1)], axis=1)
    ccount = np.concatenate([zeros, np.cumsum(valid, axis=1)], axis=1)
    return csum, csum_sq, ccount


def rolling_mean_std(arr: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """NaN-aware causal rolling mean and (population) std."""
    csum, csum_sq, ccount = _cumulative(arr)
    n = arr.shape[1]
    starts = np.maximum(np.arange(n) - window + 1, 0)
    ends = np.arange(n) + 1

    total = csum[:, ends] - csum[:, starts]
    total_sq = csum_sq[:, ends] - csum_sq[:, starts]
    count = ccount[:, ends] - ccount[:, starts]

    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(count > 0, total / np.maximum(count, 1), np.nan)
        variance = np.where(count > 1, total_sq / np.maximum(count, 1) - mean**2, np.nan)
    std = np.sqrt(np.maximum(variance, 0.0))
    return mean.astype(np.float32), std.astype(np.float32)


def rolling_min_max(arr: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """NaN-aware causal rolling min and max."""
    n_rows, n_cols = arr.shape
    padded = np.full((n_rows, n_cols + window - 1), np.nan, dtype=np.float32)
    padded[:, window - 1:] = arr
    windows = np.lib.stride_tricks.sliding_window_view(padded, window, axis=1)
    with np.errstate(invalid="ignore"):
        all_nan = np.all(~np.isfinite(windows), axis=-1)
        minimum = np.where(all_nan, np.nan, np.nanmin(np.where(np.isfinite(windows), windows, np.inf), axis=-1))
        maximum = np.where(all_nan, np.nan, np.nanmax(np.where(np.isfinite(windows), windows, -np.inf), axis=-1))
    return minimum.astype(np.float32), maximum.astype(np.float32)


def shifted(arr: np.ndarray, lag: int) -> np.ndarray:
    """`arr` shifted right by `lag` columns; the first `lag` columns become NaN."""
    out = np.full_like(arr, np.nan, dtype=np.float32)
    if lag <= 0:
        return arr.astype(np.float32)
    if lag < arr.shape[1]:
        out[:, lag:] = arr[:, :-lag]
    return out


def gather(matrix: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Safe fancy-index gather: out-of-range columns yield NaN."""
    valid = (cols >= 0) & (cols < matrix.shape[1])
    out = np.full(len(rows), np.nan, dtype=np.float32)
    if valid.any():
        out[valid] = matrix[rows[valid], cols[valid]]
    return out
