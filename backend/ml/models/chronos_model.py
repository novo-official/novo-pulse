"""Chronos-2 adapter (zero-shot foundation forecasting).

Entirely optional. If `ENABLE_CHRONOS` is off, torch is missing, or the weights
were never downloaded, `fit` raises `ModelUnavailable` and the pipeline simply
carries on without it. No network call is ever made at predict time - weights
must be pre-fetched with `scripts/download_models.py` so the system works
offline on competition day.
"""
from __future__ import annotations

import logging
import os
import time

import numpy as np

from .base import FitContext, ForecastModel, ModelUnavailable, PredictContext, postprocess

log = logging.getLogger(__name__)


def detect_device() -> str:
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def chronos_available() -> tuple[bool, str]:
    """Can Chronos actually run here? Returns (available, reason)."""
    if os.getenv("ENABLE_CHRONOS", "false").lower() not in {"1", "true", "yes"}:
        return False, "ENABLE_CHRONOS is disabled"
    try:
        import torch  # noqa: F401
    except ImportError:
        return False, "PyTorch is not installed"
    try:
        import chronos  # noqa: F401
    except ImportError:
        return False, "chronos-forecasting is not installed"
    return True, "ready"


class ChronosModel(ForecastModel):
    name = "chronos"
    label = "Chronos-2 (zero-shot)"
    family = "series"
    supports_quantiles = True
    is_optional = True

    def __init__(self, model_id: str | None = None, context_length: int = 512, **params):
        super().__init__(model_id=model_id, context_length=context_length, **params)
        self.model_id = model_id or os.getenv("CHRONOS_MODEL_ID", "amazon/chronos-2")
        self.context_length = int(context_length)
        self.device = "cpu"
        self.pipeline = None
        self._cache: dict[tuple[int, int], dict[float, np.ndarray]] = {}

    # ------------------------------------------------------------------ fit
    def fit(self, context: FitContext) -> "ChronosModel":
        """Zero-shot: 'fitting' only loads the weights."""
        available, reason = chronos_available()
        if not available:
            raise ModelUnavailable(f"Chronos unavailable - {reason}")

        started = time.perf_counter()
        try:
            import torch
            from chronos import BaseChronosPipeline

            self.device = detect_device()
            dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
            self.pipeline = BaseChronosPipeline.from_pretrained(
                self.model_id,
                device_map=self.device,
                torch_dtype=dtype,
                local_files_only=os.getenv("HF_HUB_OFFLINE", "0") == "1",
            )
        except Exception as exc:  # noqa: BLE001 - any load failure is a fallback
            raise ModelUnavailable(f"Could not load Chronos weights: {exc}") from exc

        self.fit_seconds = time.perf_counter() - started
        self.fitted = True
        self.metadata = {"model_id": self.model_id, "device": self.device, "zero_shot": True}
        log.info("Chronos loaded (%s) on %s", self.model_id, self.device)
        return self

    # -------------------------------------------------------------- predict
    def _forecast(self, context: PredictContext, quantiles: tuple[float, ...]):
        key = (context.origin_idx, context.horizon)
        if key in self._cache:
            return self._cache[key]
        if self.pipeline is None:
            raise ModelUnavailable("Chronos pipeline was never loaded")

        import torch

        tensor = context.engine.tensor
        history = tensor.y.copy()
        history[~tensor.observed] = np.nan
        history = history[:, : context.origin_idx + 1]
        start = max(0, history.shape[1] - self.context_length)
        window = history[:, start:]

        series = [
            torch.tensor(np.nan_to_num(row[np.isfinite(row)], nan=0.0), dtype=torch.float32)
            for row in window
        ]
        levels = sorted({*quantiles, 0.5})
        batch = int(self.params.get("batch_size", 64))
        chunks: list[np.ndarray] = []
        for offset in range(0, len(series), batch):
            piece = series[offset : offset + batch]
            piece = [s if len(s) >= 8 else torch.zeros(8) for s in piece]
            values, _ = self.pipeline.predict_quantiles(
                context=piece,
                prediction_length=context.horizon,
                quantile_levels=levels,
            )
            chunks.append(np.asarray(values, dtype=np.float32))
        stacked = np.concatenate(chunks, axis=0)  # (E, horizon, n_quantiles)

        out = {
            float(level): stacked[:, :, i] for i, level in enumerate(levels)
        }
        self._cache = {key: out}
        return out

    def _gather(self, context: PredictContext, matrix: np.ndarray) -> np.ndarray:
        meta = context.frame.meta
        rows = meta["entity_pos"].to_numpy()
        steps = meta["horizon"].to_numpy() - 1
        valid = (rows < matrix.shape[0]) & (steps < matrix.shape[1])
        out = np.zeros(len(meta), dtype=float)
        out[valid] = matrix[rows[valid], steps[valid]]
        return postprocess(out)

    def predict(self, context: PredictContext) -> np.ndarray:
        forecasts = self._forecast(context, (0.5,))
        return self._gather(context, forecasts[0.5])

    def predict_quantiles(
        self, context: PredictContext, quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)
    ) -> dict[float, np.ndarray]:
        forecasts = self._forecast(context, tuple(quantiles))
        return {
            float(q): self._gather(context, forecasts[float(q)])
            for q in quantiles
            if float(q) in forecasts
        }
