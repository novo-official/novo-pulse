"""NeuralForecast adapters (NHITS, NBEATSx). Optional by design.

NHITS is the default deep model: it handles long horizons well and trains in
minutes on CPU with the demo hyper-parameters. NBEATSx is only worth running
when there are genuine exogenous covariates to feed it.
"""
from __future__ import annotations

import logging
import os
import time

import numpy as np
import pandas as pd

from ..contract import ENTITY
from .base import FitContext, ForecastModel, ModelUnavailable, PredictContext, postprocess

log = logging.getLogger(__name__)


def neuralforecast_available() -> tuple[bool, str]:
    if os.getenv("ENABLE_NEURALFORECAST", "false").lower() not in {"1", "true", "yes"}:
        return False, "ENABLE_NEURALFORECAST is disabled"
    try:
        import neuralforecast  # noqa: F401
    except ImportError:
        return False, "neuralforecast is not installed"
    return True, "ready"


class _NeuralBase(ForecastModel):
    family = "series"
    is_optional = True
    supports_quantiles = True
    architecture = "NHITS"

    def __init__(self, max_steps: int = 200, input_size_multiplier: int = 3, **params):
        super().__init__(max_steps=max_steps, input_size_multiplier=input_size_multiplier, **params)
        self.max_steps = int(max_steps)
        self.input_size_multiplier = int(input_size_multiplier)
        self.nf = None
        self.horizon = 0
        self.quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)
        self._cache: dict[int, pd.DataFrame] = {}

    def _long_frame(self, context: FitContext | PredictContext, end_idx: int) -> pd.DataFrame:
        tensor = context.engine.tensor
        history = tensor.y[:, : end_idx + 1].copy()
        observed = tensor.observed[:, : end_idx + 1]
        rows, cols = np.where(observed)
        return pd.DataFrame(
            {
                "unique_id": tensor.entities[rows],
                "ds": tensor.dates[cols],
                "y": history[rows, cols].astype(float),
            }
        ).sort_values(["unique_id", "ds"])

    def fit(self, context: FitContext) -> "_NeuralBase":
        available, reason = neuralforecast_available()
        if not available:
            raise ModelUnavailable(f"NeuralForecast unavailable - {reason}")

        started = time.perf_counter()
        try:
            from neuralforecast import NeuralForecast
            from neuralforecast.losses.pytorch import MQLoss
            from neuralforecast.models import NBEATSx, NHITS
        except ImportError as exc:
            raise ModelUnavailable(f"NeuralForecast import failed: {exc}") from exc

        self.horizon = int(context.engine.config.max_horizon)
        self.quantiles = tuple(context.quantiles)
        frame = self._long_frame(context, context.train_end_idx)
        if frame["unique_id"].nunique() == 0 or len(frame) < 200:
            raise ModelUnavailable("Not enough history to train a neural model")

        architecture = {"NHITS": NHITS, "NBEATSx": NBEATSx}[self.architecture]
        try:
            model = architecture(
                h=self.horizon,
                input_size=self.horizon * self.input_size_multiplier,
                max_steps=self.max_steps,
                loss=MQLoss(level=[80]),
                scaler_type="robust",
                random_seed=context.seed,
                enable_progress_bar=False,
                logger=False,
            )
            self.nf = NeuralForecast(models=[model], freq=context.engine.tensor.freq)
            self.nf.fit(frame, verbose=False)
        except Exception as exc:  # noqa: BLE001 - training failure must not be fatal
            raise ModelUnavailable(f"{self.architecture} training failed: {exc}") from exc

        self.fit_seconds = time.perf_counter() - started
        self.fitted = True
        self.metadata = {"architecture": self.architecture, "max_steps": self.max_steps}
        return self

    def _forecast(self, context: PredictContext) -> pd.DataFrame:
        if context.origin_idx in self._cache:
            return self._cache[context.origin_idx]
        if self.nf is None:
            raise ModelUnavailable(f"{self.architecture} was never fitted")
        frame = self._long_frame(context, context.origin_idx)
        result = self.nf.predict(df=frame).reset_index()
        self._cache = {context.origin_idx: result}
        return result

    def _column(self, result: pd.DataFrame, quantile: float) -> str | None:
        prefix = self.architecture
        if abs(quantile - 0.5) < 1e-9:
            for candidate in (f"{prefix}-median", prefix, f"{prefix}-mean"):
                if candidate in result.columns:
                    return candidate
        level = int(round(abs(quantile - 0.5) * 200))
        suffix = "lo" if quantile < 0.5 else "hi"
        candidate = f"{prefix}-{suffix}-{level}"
        return candidate if candidate in result.columns else None

    def _gather(self, context: PredictContext, result: pd.DataFrame, column: str) -> np.ndarray:
        lookup = result.set_index(["unique_id", "ds"])[column]
        meta = context.frame.meta
        keys = pd.MultiIndex.from_arrays([meta[ENTITY].to_numpy(), meta["ds"].to_numpy()])
        values = lookup.reindex(keys).to_numpy(dtype=float)
        return postprocess(values)

    def predict(self, context: PredictContext) -> np.ndarray:
        result = self._forecast(context)
        column = self._column(result, 0.5)
        if column is None:
            raise ModelUnavailable(f"{self.architecture} produced no median column")
        return self._gather(context, result, column)

    def predict_quantiles(
        self, context: PredictContext, quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)
    ) -> dict[float, np.ndarray]:
        result = self._forecast(context)
        out: dict[float, np.ndarray] = {}
        for quantile in quantiles:
            column = self._column(result, float(quantile))
            if column is not None:
                out[float(quantile)] = self._gather(context, result, column)
        return out


class NHITSModel(_NeuralBase):
    name = "nhits"
    label = "NHITS"
    architecture = "NHITS"


class NBEATSxModel(_NeuralBase):
    name = "nbeatsx"
    label = "NBEATSx"
    architecture = "NBEATSx"
