"""The single interface every forecaster in this project implements.

Two families share it:

* **tabular** models consume the supervised design matrix (LightGBM, CatBoost),
* **series** models read the panel tensor directly (baselines, Chronos, NHITS).

Both receive the same contexts and return predictions row-aligned with
`context.frame.meta`, so the evaluator, the ensemble and the API never need to
know which family they are talking to.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..features.engineering import FeatureEngine, SupervisedFrame


@dataclass
class FitContext:
    engine: FeatureEngine
    train_end_idx: int
    frame: SupervisedFrame                  # shared supervised training frame
    seed: int = 42
    params: dict[str, Any] = field(default_factory=dict)
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)
    non_negative: bool = True
    log1p: bool = False


@dataclass
class PredictContext:
    engine: FeatureEngine
    origin_idx: int
    horizon: int
    frame: SupervisedFrame                  # shared inference frame


class ForecastModel(abc.ABC):
    """Base class. Subclasses set `name`, `family` and `label`."""

    name: str = "model"
    label: str = "Model"
    family: str = "series"                  # "tabular" | "series"
    supports_quantiles: bool = False
    is_optional: bool = False

    def __init__(self, **params: Any):
        self.params = params
        self.fitted = False
        self.fit_seconds: float = 0.0
        self.metadata: dict[str, Any] = {}

    # ------------------------------------------------------------------ api
    @abc.abstractmethod
    def fit(self, context: FitContext) -> "ForecastModel":
        ...

    @abc.abstractmethod
    def predict(self, context: PredictContext) -> np.ndarray:
        ...

    def predict_quantiles(
        self, context: PredictContext, quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)
    ) -> dict[float, np.ndarray]:
        """Native quantiles when the model has them, otherwise nothing.

        Returning an empty dict is the signal that the conformal calibrator
        should supply the interval instead.
        """
        return {}

    def get_feature_importance(self) -> dict[str, float]:
        return {}

    # ------------------------------------------------------------ artefacts
    def save(self, directory: str | Path) -> Path:
        import joblib

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.name}.joblib"
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "ForecastModel":
        import joblib

        return joblib.load(Path(path))

    # ------------------------------------------------------------- helpers
    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "label": self.label,
            "family": self.family,
            "supports_quantiles": self.supports_quantiles,
            "optional": self.is_optional,
            "params": {k: v for k, v in self.params.items() if _jsonable(v)},
            "fit_seconds": round(self.fit_seconds, 3),
            **self.metadata,
        }


def _jsonable(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool, type(None), list, dict))


def postprocess(
    predictions: np.ndarray, non_negative: bool = True, integer: bool = False
) -> np.ndarray:
    """Apply target-domain constraints declared in the contract."""
    out = np.asarray(predictions, dtype=float)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    if non_negative:
        out = np.maximum(out, 0.0)
    if integer:
        out = np.round(out)
    return out


def enforce_monotone(quantiles: dict[float, np.ndarray]) -> dict[float, np.ndarray]:
    """Sort quantile predictions so a lower level never exceeds a higher one.

    Independently fitted quantile heads can cross, which would produce an
    interval whose lower bound sits above its upper bound. Sorting each row is
    the standard, distribution-free repair.
    """
    if len(quantiles) < 2:
        return quantiles
    keys = sorted(quantiles)
    stacked = np.sort(np.vstack([np.asarray(quantiles[k], dtype=float) for k in keys]), axis=0)
    return {key: stacked[index] for index, key in enumerate(keys)}


class ModelUnavailable(RuntimeError):
    """Raised when an optional dependency or model weight is missing.

    The pipeline catches this and continues without the model - an optional
    component must never take the demo down.
    """
