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

from .base import (
    FitContext,
    ForecastModel,
    ModelUnavailable,
    PredictContext,
    enforce_monotone,
    postprocess,
)

log = logging.getLogger(__name__)


def _to_array(values: Any, horizon: int, n_quantiles: int) -> np.ndarray:
    """Normalise a pipeline's quantile output to `(batch, horizon, n_quantiles)`.

    Chronos-2 hands back a list of per-series tensors shaped
    `(n_variates, horizon, n_quantiles)`; Bolt and T5 hand back one stacked
    `(batch, horizon, n_quantiles)` tensor.
    """
    if isinstance(values, (list, tuple)):
        rows = []
        for item in values:
            array = np.asarray(
                item.detach().cpu().numpy() if hasattr(item, "detach") else item,
                dtype=np.float32,
            )
            if array.ndim == 3:
                # (n_variates, horizon, n_quantiles) - univariate series here.
                array = array[0]
            rows.append(array)
        stacked = np.stack(rows, axis=0)
    else:
        stacked = np.asarray(
            values.detach().cpu().numpy() if hasattr(values, "detach") else values,
            dtype=np.float32,
        )

    if stacked.ndim != 3:
        raise ValueError(f"unexpected Chronos output shape {stacked.shape}")
    if stacked.shape[1] != horizon and stacked.shape[2] == horizon:
        # Some builds return (batch, n_quantiles, horizon); transpose to match.
        stacked = np.transpose(stacked, (0, 2, 1))
    return stacked[:, :horizon, :n_quantiles]


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

    def __init__(
        self,
        model_id: str | None = None,
        context_length: int = 512,
        min_context: int = 16,
        **params,
    ):
        super().__init__(
            model_id=model_id, context_length=context_length, min_context=min_context, **params
        )
        self.model_id = model_id or os.getenv("CHRONOS_MODEL_ID", "amazon/chronos-2")
        self.context_length = int(context_length)
        self.min_context = int(min_context)
        self.device = "cpu"
        self.pipeline = None
        self._cache: dict[tuple, dict[float, np.ndarray]] = {}

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
        """Run the pipeline once per origin and cache the (E, H, Q) result.

        Two shapes have to be normalised here. Chronos-Bolt and Chronos-T5
        return a single stacked tensor `(batch, horizon, n_quantiles)`, while
        Chronos-2 returns a *list* of `(n_variates, horizon, n_quantiles)`
        tensors - one per input series. Assuming either one alone silently
        mis-indexes against the other.
        """
        # The cache key must include the quantile levels. Keying on the origin
        # alone lets a point-forecast call (which asks for the median only)
        # poison the cache, so a later interval request silently comes back
        # with nothing but P50 and the model looks like it has no quantiles.
        levels = tuple(sorted({*(float(q) for q in quantiles), 0.5}))
        key = (context.origin_idx, context.horizon, levels)
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

        # A series with almost no history cannot be forecast from its own past.
        # Padding it with zeros would feed the model invented data, so those
        # entities are excluded and filled with NaN afterwards instead.
        series: list = []
        usable: list[int] = []
        for row_index, row in enumerate(window):
            observed = row[np.isfinite(row)]
            if observed.size < self.min_context:
                continue
            series.append(torch.tensor(observed, dtype=torch.float32))
            usable.append(row_index)

        if not series:
            raise ModelUnavailable(
                f"no entity has the {self.min_context} observations Chronos needs"
            )

        batch = int(self.params.get("batch_size", 32))
        collected: list[np.ndarray] = []
        for offset in range(0, len(series), batch):
            piece = series[offset : offset + batch]
            values, _ = self.pipeline.predict_quantiles(
                inputs=piece,
                prediction_length=context.horizon,
                quantile_levels=list(levels),
            )
            collected.append(_to_array(values, context.horizon, len(levels)))
        stacked = np.concatenate(collected, axis=0)

        # Re-expand to the full entity set; skipped entities stay NaN.
        full = np.full((tensor.n_entities, context.horizon, len(levels)), np.nan, dtype=np.float32)
        full[np.asarray(usable, dtype=int)] = stacked

        out = {float(level): full[:, :, i] for i, level in enumerate(levels)}
        self._cache[key] = out
        self.metadata["entities_forecast"] = len(usable)
        self.metadata["entities_skipped"] = int(tensor.n_entities - len(usable))
        return out

    def _gather(self, context: PredictContext, matrix: np.ndarray) -> np.ndarray:
        """Map the (entity, horizon) grid back onto the supervised frame rows."""
        meta = context.frame.meta
        rows = meta["entity_pos"].to_numpy()
        steps = meta["horizon"].to_numpy() - 1
        valid = (rows < matrix.shape[0]) & (steps >= 0) & (steps < matrix.shape[1])
        out = np.full(len(meta), np.nan, dtype=float)
        out[valid] = matrix[rows[valid], steps[valid]]

        # Entities Chronos could not forecast fall back to their last observed
        # value rather than to a fabricated zero.
        missing = ~np.isfinite(out)
        if missing.any():
            tensor = context.engine.tensor
            history = tensor.y.copy()
            history[~tensor.observed] = np.nan
            history[:, context.origin_idx + 1 :] = np.nan
            with np.errstate(invalid="ignore"):
                entity_mean = np.nanmean(history, axis=1)
            fallback = entity_mean[rows[missing]]
            out[missing] = np.nan_to_num(fallback, nan=0.0)
        return postprocess(out)

    def predict(self, context: PredictContext) -> np.ndarray:
        forecasts = self._forecast(context, (0.5,))
        return self._gather(context, forecasts[0.5])

    def predict_quantiles(
        self, context: PredictContext, quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)
    ) -> dict[float, np.ndarray]:
        forecasts = self._forecast(context, tuple(quantiles))
        return enforce_monotone(
            {
                float(q): self._gather(context, forecasts[float(q)])
                for q in quantiles
                if float(q) in forecasts
            }
        )
