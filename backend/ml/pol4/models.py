"""Gradient-boosted models for remaining demand.

The target is what is still to come, not the total:

    predicted_final = observed_so_far + max(0, predicted_remaining)

which guarantees the Phase 1 floor by construction - a model cannot predict
away demand that has already been counted - and lets the model spend all its
capacity on the genuinely uncertain part.

WAPE is a sum of absolute errors, so the aligned objective is L1. Both
libraries are configured that way by default and the alternative is measured
rather than assumed, as is the log1p transform: the audit found the generic
platform's automatic log1p under-predicting at every horizon, so here it is off
unless an experiment earns it.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .features import CATEGORICAL

log = logging.getLogger(__name__)

LIGHTGBM_DEFAULTS: dict[str, Any] = {
    "objective": "regression_l1",
    "n_estimators": 400,
    "learning_rate": 0.06,
    "num_leaves": 127,
    "min_child_samples": 40,
    "subsample": 0.9,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "verbose": -1,
}

CATBOOST_DEFAULTS: dict[str, Any] = {
    "loss_function": "MAE",
    "iterations": 500,
    "learning_rate": 0.08,
    "depth": 8,
    "verbose": False,
    "allow_writing_files": False,
}


class ModelUnavailable(RuntimeError):
    """The library for this model is not installed."""


@dataclass
class RemainingDemandModel:
    kind: str                       # "lightgbm" | "catboost"
    params: dict[str, Any] = field(default_factory=dict)
    log1p: bool = False
    seed: int = 42
    model: Any = None
    feature_names: list[str] = field(default_factory=list)
    categorical: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------ fit
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "RemainingDemandModel":
        self.feature_names = list(X.columns)
        self.categorical = [c for c in CATEGORICAL if c in X.columns]
        target = np.log1p(np.maximum(y, 0.0)) if self.log1p else np.asarray(y, dtype=np.float64)
        frame = self._prepare(X)

        if self.kind == "lightgbm":
            try:
                from lightgbm import LGBMRegressor
            except ImportError as exc:  # pragma: no cover
                raise ModelUnavailable("lightgbm is not installed") from exc
            params = {**LIGHTGBM_DEFAULTS, **self.params, "random_state": self.seed}
            self.model = LGBMRegressor(**params)
            self.model.fit(
                frame, target, categorical_feature=self.categorical or "auto"
            )
        elif self.kind == "catboost":
            try:
                from catboost import CatBoostRegressor
            except ImportError as exc:  # pragma: no cover
                raise ModelUnavailable("catboost is not installed") from exc
            params = {**CATBOOST_DEFAULTS, **self.params, "random_seed": self.seed}
            self.model = CatBoostRegressor(**params)
            self.model.fit(frame, target, cat_features=self.categorical or None)
        else:
            raise ValueError(f"unknown model kind {self.kind!r}")
        return self

    def _prepare(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = X.copy()
        for column in self.categorical:
            # Both libraries want an integral code, not a float that happens to
            # hold a city id.
            frame[column] = frame[column].astype(np.int64)
            if self.kind == "lightgbm":
                frame[column] = frame[column].astype("category")
        return frame

    # -------------------------------------------------------------- predict
    def predict_remaining(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("model has not been fitted")
        raw = np.asarray(self.model.predict(self._prepare(X[self.feature_names])), dtype=np.float64)
        if self.log1p:
            raw = np.expm1(raw)
        return np.maximum(raw, 0.0)

    def predict_final(self, X: pd.DataFrame, observed: np.ndarray) -> np.ndarray:
        """Observed demand plus predicted pickup - never less than observed."""
        observed = np.asarray(observed, dtype=np.float64)
        return observed + self.predict_remaining(X)

    # ----------------------------------------------------------- importance
    def importance(self) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("model has not been fitted")
        if self.kind == "lightgbm":
            values = self.model.booster_.feature_importance(importance_type="gain")
        else:
            values = self.model.get_feature_importance()
        frame = pd.DataFrame({"feature": self.feature_names, "importance": values})
        total = frame["importance"].sum()
        frame["share"] = frame["importance"] / total if total > 0 else 0.0
        return frame.sort_values("importance", ascending=False, ignore_index=True)
