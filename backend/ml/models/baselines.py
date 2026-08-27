"""Statistical baselines.

Beating a well-implemented seasonal naive is the bar an "AI" forecast has to
clear before it is worth anything. Every run trains these, and the dashboard
reports the improvement over the best of them.
"""
from __future__ import annotations

import numpy as np

from ..features.rolling import gather, rolling_mean_std
from .base import FitContext, ForecastModel, PredictContext


class _SeriesBaseline(ForecastModel):
    family = "series"

    def fit(self, context: FitContext) -> "ForecastModel":
        self.fitted = True
        return self

    def _history(self, context: PredictContext) -> np.ndarray:
        """Target matrix with anything unobserved masked out."""
        tensor = context.engine.tensor
        history = tensor.y.copy()
        history[~tensor.observed] = np.nan
        # Nothing after the forecast origin may be read.
        history[:, context.origin_idx + 1:] = np.nan
        return history


class NaiveModel(_SeriesBaseline):
    name = "naive"
    label = "Naive (last value)"

    def predict(self, context: PredictContext) -> np.ndarray:
        meta = context.frame.meta
        history = self._history(context)
        rows = meta["entity_pos"].to_numpy()
        origins = meta["origin_idx"].to_numpy()
        return np.nan_to_num(gather(history, rows, origins), nan=0.0)


class SeasonalNaiveModel(_SeriesBaseline):
    """ŷ(t+h) = y(t + h - m*ceil(h/m)) - the same phase, one or more cycles back."""

    family = "series"

    def __init__(self, season: int = 7, **params):
        super().__init__(season=season, **params)
        self.season = int(season)
        self.name = f"seasonal_naive_{self.season}"
        self.label = f"Seasonal Naive ({self.season})"

    def predict(self, context: PredictContext) -> np.ndarray:
        meta = context.frame.meta
        history = self._history(context)
        rows = meta["entity_pos"].to_numpy()
        targets = meta["target_idx"].to_numpy()
        horizons = meta["horizon"].to_numpy()
        cycles = np.ceil(horizons / self.season).astype(np.int64)
        values = gather(history, rows, targets - self.season * cycles)

        # Fall back through older cycles, then to the entity mean.
        for extra in (1, 2, 3):
            missing = ~np.isfinite(values)
            if not missing.any():
                break
            values[missing] = gather(
                history,
                rows[missing],
                (targets - self.season * (cycles + extra))[missing],
            )
        missing = ~np.isfinite(values)
        if missing.any():
            with np.errstate(invalid="ignore"):
                entity_mean = np.nanmean(history, axis=1)
            values[missing] = np.nan_to_num(entity_mean[rows[missing]], nan=0.0)
        return np.nan_to_num(values, nan=0.0)


class MovingAverageModel(_SeriesBaseline):
    name = "moving_average"
    label = "Moving Average"

    def __init__(self, window: int = 28, **params):
        super().__init__(window=window, **params)
        self.window = int(window)

    def predict(self, context: PredictContext) -> np.ndarray:
        meta = context.frame.meta
        history = self._history(context)
        mean, _ = rolling_mean_std(history, self.window)
        values = gather(mean, meta["entity_pos"].to_numpy(), meta["origin_idx"].to_numpy())
        return np.nan_to_num(values, nan=0.0)


class HistoricalMeanModel(_SeriesBaseline):
    name = "historical_mean"
    label = "Historical Mean"

    def predict(self, context: PredictContext) -> np.ndarray:
        history = self._history(context)
        with np.errstate(invalid="ignore"):
            entity_mean = np.nanmean(history, axis=1)
        values = entity_mean[context.frame.meta["entity_pos"].to_numpy()]
        return np.nan_to_num(values, nan=0.0)


class SeasonalMeanModel(_SeriesBaseline):
    """Mean of the same seasonal phase (e.g. every past Friday)."""

    name = "seasonal_mean"
    label = "Seasonal Mean"

    def __init__(self, season: int = 7, n_cycles: int = 8, **params):
        super().__init__(season=season, n_cycles=n_cycles, **params)
        self.season = int(season)
        self.n_cycles = int(n_cycles)

    def predict(self, context: PredictContext) -> np.ndarray:
        meta = context.frame.meta
        history = self._history(context)
        rows = meta["entity_pos"].to_numpy()
        targets = meta["target_idx"].to_numpy()
        horizons = meta["horizon"].to_numpy()
        cycles = np.ceil(horizons / self.season).astype(np.int64)
        stacked = np.vstack(
            [
                gather(history, rows, targets - self.season * (cycles + offset))
                for offset in range(self.n_cycles)
            ]
        )
        with np.errstate(invalid="ignore"):
            values = np.nanmean(stacked, axis=0)
        return np.nan_to_num(values, nan=0.0)
