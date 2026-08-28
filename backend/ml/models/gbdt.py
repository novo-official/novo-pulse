"""Global gradient-boosted forecasters (LightGBM, CatBoost).

"Global" means one model is fitted across every entity at once, with the
entity identity and its static profile as features. That is what makes
cold-start listings workable: a listing with two weeks of history still
inherits the behaviour of its destination, category and price band.

The horizon is an explicit feature, so a single fitted model serves 1..H
without recursive feedback - no compounding of its own errors.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd

from .base import (
    FitContext,
    ForecastModel,
    ModelUnavailable,
    PredictContext,
    enforce_monotone,
    postprocess,
)


class _GBDTBase(ForecastModel):
    family = "tabular"

    def __init__(self, **params: Any):
        super().__init__(**params)
        self.model = None
        self.quantile_models: dict[float, Any] = {}
        self.feature_names: list[str] = []
        self.categorical: list[str] = []
        self.log1p = False
        self.non_negative = True

    # -------------------------------------------------------------- helpers
    def _prepare_target(self, y: np.ndarray) -> np.ndarray:
        return np.log1p(np.maximum(y, 0.0)) if self.log1p else y

    def _restore_target(self, pred: np.ndarray) -> np.ndarray:
        return np.expm1(pred) if self.log1p else pred

    def _finalise(self, raw: np.ndarray) -> np.ndarray:
        return postprocess(self._restore_target(raw), non_negative=self.non_negative)

    def _align(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.reindex(columns=self.feature_names)


class LightGBMModel(_GBDTBase):
    """LightGBM with native quantile heads for the prediction interval."""

    name = "lightgbm"
    label = "LightGBM"
    supports_quantiles = True

    def fit(self, context: FitContext) -> "LightGBMModel":
        try:
            import lightgbm as lgb
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ModelUnavailable(f"LightGBM is not installed: {exc}") from exc

        started = time.perf_counter()
        frame = context.frame
        self.feature_names = list(frame.X.columns)
        self.categorical = [c for c in frame.categorical if c in frame.X.columns]
        self.log1p = context.log1p
        self.non_negative = context.non_negative

        params = {
            "objective": self.params.get("objective", "tweedie" if context.non_negative else "l2"),
            "n_estimators": self.params.get("n_estimators", 600),
            "learning_rate": self.params.get("learning_rate", 0.05),
            "num_leaves": self.params.get("num_leaves", 127),
            "min_child_samples": self.params.get("min_child_samples", 20),
            "subsample": self.params.get("subsample", 0.85),
            "subsample_freq": 1,
            "colsample_bytree": self.params.get("colsample_bytree", 0.8),
            "reg_lambda": self.params.get("reg_lambda", 1.0),
            "random_state": context.seed,
            "n_jobs": self.params.get("n_jobs", -1),
            "verbose": -1,
        }
        if params["objective"] == "tweedie":
            params["tweedie_variance_power"] = self.params.get("tweedie_variance_power", 1.2)
        if self.log1p and params["objective"] == "tweedie":
            # log1p already handles the skew; Tweedie on top of it double-counts.
            params["objective"] = "l2"
            params.pop("tweedie_variance_power", None)

        target = self._prepare_target(frame.y)
        self.model = lgb.LGBMRegressor(**params)
        self.model.fit(frame.X, target, categorical_feature=self.categorical or "auto")

        # Quantile heads are cheaper than the point model - fewer trees is enough
        # for a usable interval and keeps the demo profile fast.
        quantile_params = dict(params)
        quantile_params["objective"] = "quantile"
        quantile_params["n_estimators"] = max(
            80, int(params["n_estimators"] * self.params.get("quantile_tree_ratio", 0.5))
        )
        quantile_params.pop("tweedie_variance_power", None)
        for quantile in context.quantiles:
            if abs(quantile - 0.5) < 1e-9:
                continue
            model = lgb.LGBMRegressor(**{**quantile_params, "alpha": float(quantile)})
            model.fit(frame.X, target, categorical_feature=self.categorical or "auto")
            self.quantile_models[float(quantile)] = model

        self.fit_seconds = time.perf_counter() - started
        self.fitted = True
        self.metadata = {
            "n_features": len(self.feature_names),
            "n_train_rows": int(len(frame.X)),
            "objective": params["objective"],
            "log1p": self.log1p,
        }
        return self

    def predict(self, context: PredictContext) -> np.ndarray:
        if self.model is None:
            raise ModelUnavailable("LightGBM model has not been fitted")
        return self._finalise(self.model.predict(self._align(context.frame.X)))

    def predict_quantiles(
        self, context: PredictContext, quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)
    ) -> dict[float, np.ndarray]:
        out: dict[float, np.ndarray] = {}
        X = self._align(context.frame.X)
        for quantile in quantiles:
            model = self.quantile_models.get(float(quantile))
            if model is not None:
                out[float(quantile)] = self._finalise(model.predict(X))
        if out and 0.5 in [float(q) for q in quantiles] and 0.5 not in out:
            out[0.5] = self.predict(context)
        return enforce_monotone(out)

    def get_feature_importance(self) -> dict[str, float]:
        if self.model is None:
            return {}
        gains = np.asarray(self.model.booster_.feature_importance(importance_type="gain"), dtype=float)
        total = gains.sum()
        if total <= 0:
            return {}
        return {
            name: round(float(value / total), 6)
            for name, value in sorted(
                zip(self.feature_names, gains), key=lambda item: -item[1]
            )
        }


class CatBoostModel(_GBDTBase):
    """CatBoost - strong on categorical entity/destination identities."""

    name = "catboost"
    label = "CatBoost"
    supports_quantiles = True

    def fit(self, context: FitContext) -> "CatBoostModel":
        try:
            from catboost import CatBoostRegressor, Pool
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ModelUnavailable(f"CatBoost is not installed: {exc}") from exc

        started = time.perf_counter()
        frame = context.frame
        self.feature_names = list(frame.X.columns)
        self.categorical = [c for c in frame.categorical if c in frame.X.columns]
        self.log1p = context.log1p
        self.non_negative = context.non_negative

        X = self._catboost_frame(frame.X)
        target = self._prepare_target(frame.y)
        pool = Pool(X, target, cat_features=self.categorical)

        params = {
            "iterations": self.params.get("iterations", 600),
            "depth": self.params.get("depth", 8),
            "learning_rate": self.params.get("learning_rate", 0.06),
            "loss_function": self.params.get("loss_function", "RMSE"),
            "l2_leaf_reg": self.params.get("l2_leaf_reg", 3.0),
            "random_seed": context.seed,
            "verbose": False,
            "allow_writing_files": False,
            "thread_count": self.params.get("thread_count", -1),
        }
        self.model = CatBoostRegressor(**params)
        self.model.fit(pool)

        quantile_iterations = max(
            80, int(params["iterations"] * self.params.get("quantile_tree_ratio", 0.5))
        )
        for quantile in context.quantiles:
            if abs(quantile - 0.5) < 1e-9:
                continue
            model = CatBoostRegressor(
                **{
                    **params,
                    "iterations": quantile_iterations,
                    "loss_function": f"Quantile:alpha={float(quantile)}",
                }
            )
            model.fit(pool)
            self.quantile_models[float(quantile)] = model

        self.fit_seconds = time.perf_counter() - started
        self.fitted = True
        self.metadata = {
            "n_features": len(self.feature_names),
            "n_train_rows": int(len(frame.X)),
            "loss_function": params["loss_function"],
            "log1p": self.log1p,
        }
        return self

    def _catboost_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        out = self._align(X).copy()
        for column in self.categorical:
            out[column] = out[column].astype(str).fillna("__na__")
        numeric = [c for c in out.columns if c not in self.categorical]
        out[numeric] = out[numeric].astype(np.float64).fillna(np.nan)
        return out

    def predict(self, context: PredictContext) -> np.ndarray:
        if self.model is None:
            raise ModelUnavailable("CatBoost model has not been fitted")
        return self._finalise(self.model.predict(self._catboost_frame(context.frame.X)))

    def predict_quantiles(
        self, context: PredictContext, quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)
    ) -> dict[float, np.ndarray]:
        out: dict[float, np.ndarray] = {}
        X = self._catboost_frame(context.frame.X)
        for quantile in quantiles:
            model = self.quantile_models.get(float(quantile))
            if model is not None:
                out[float(quantile)] = self._finalise(model.predict(X))
        return enforce_monotone(out)

    def get_feature_importance(self) -> dict[str, float]:
        if self.model is None:
            return {}
        values = np.asarray(self.model.get_feature_importance(), dtype=float)
        total = values.sum()
        if total <= 0:
            return {}
        return {
            name: round(float(value / total), 6)
            for name, value in sorted(zip(self.feature_names, values), key=lambda item: -item[1])
        }


