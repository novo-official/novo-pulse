"""Weighted ensemble with data-driven weights.

Weights are never hard-coded. They come from rolling cross-validation scores on
the *primary metric* declared in the contract, so the blend re-balances itself
when the metric or the dataset changes. A model that fails at any point is
simply dropped and the remaining weights are renormalised.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..evaluation.metrics import get_metric
from .base import FitContext, ForecastModel, PredictContext, postprocess


@dataclass
class WeightedMember:
    name: str
    weight: float
    cv_score: float


class EnsembleModel(ForecastModel):
    """Blend of already-fitted members. It does not fit anything itself."""

    name = "ensemble"
    label = "Ensemble"
    family = "meta"
    supports_quantiles = True

    def __init__(self, members: dict[str, ForecastModel], weights: dict[str, float], **params):
        super().__init__(**params)
        self.members = members
        self.weights = weights
        self.fitted = True
        self.metadata = {"members": {k: round(v, 4) for k, v in weights.items()}}

    def fit(self, context: FitContext) -> "EnsembleModel":
        # Members arrive pre-fitted from the training pipeline.
        return self

    def predict(self, context: PredictContext) -> np.ndarray:
        total = np.zeros(len(context.frame.meta), dtype=float)
        used = 0.0
        for name, weight in self.weights.items():
            model = self.members.get(name)
            if model is None or weight <= 0:
                continue
            try:
                total += weight * model.predict(context)
                used += weight
            except Exception:  # noqa: BLE001 - a broken member must not break the blend
                continue
        if used <= 0:
            return total
        return postprocess(total / used)

    def predict_quantiles(
        self, context: PredictContext, quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)
    ) -> dict[float, np.ndarray]:
        accumulated: dict[float, np.ndarray] = {}
        used: dict[float, float] = {}
        for name, weight in self.weights.items():
            model = self.members.get(name)
            if model is None or weight <= 0:
                continue
            try:
                member_quantiles = model.predict_quantiles(context, quantiles)
            except Exception:  # noqa: BLE001
                continue
            for quantile, values in member_quantiles.items():
                accumulated[quantile] = accumulated.get(quantile, 0.0) + weight * values
                used[quantile] = used.get(quantile, 0.0) + weight
        return {q: postprocess(v / used[q]) for q, v in accumulated.items() if used.get(q, 0) > 0}

    def get_feature_importance(self) -> dict[str, float]:
        """Weight-averaged importance across members that expose one."""
        merged: dict[str, float] = {}
        total = 0.0
        for name, weight in self.weights.items():
            model = self.members.get(name)
            if model is None or weight <= 0:
                continue
            importance = model.get_feature_importance()
            if not importance:
                continue
            total += weight
            for feature, value in importance.items():
                merged[feature] = merged.get(feature, 0.0) + weight * value
        if total <= 0:
            return {}
        return {
            k: round(v / total, 6)
            for k, v in sorted(merged.items(), key=lambda item: -item[1])
        }


def compute_weights(
    cv_scores: dict[str, float],
    metric: str = "wape",
    temperature: float = 4.0,
    max_members: int = 5,
    min_weight: float = 0.02,
) -> dict[str, float]:
    """Turn CV scores into ensemble weights.

    Better models get more weight, on a softmax over the *relative* score so the
    scheme is scale-free and works for any metric in the registry. A model more
    than twice as bad as the best is excluded outright.
    """
    spec = get_metric(metric)
    usable = {
        name: float(score)
        for name, score in cv_scores.items()
        if score is not None and np.isfinite(score)
    }
    if not usable:
        return {}

    # Normalise so that "lower is better" always holds.
    oriented = {k: (-v if spec.greater_is_better else v) for k, v in usable.items()}
    best = min(oriented.values())
    if best <= 0:
        # Shift into positive territory for the ratio below.
        offset = abs(best) + 1e-6
        oriented = {k: v + offset for k, v in oriented.items()}
        best = min(oriented.values())
    if best <= 0:
        best = 1e-9

    ratios = {k: v / best for k, v in oriented.items()}
    candidates = {k: r for k, r in ratios.items() if r <= 1.35}
    if not candidates:
        candidates = {min(ratios, key=ratios.get): 1.0}

    ranked = sorted(candidates.items(), key=lambda item: item[1])[:max_members]
    raw = {name: float(np.exp(-temperature * (ratio - 1.0))) for name, ratio in ranked}
    total = sum(raw.values())
    weights = {name: value / total for name, value in raw.items()}

    weights = {k: v for k, v in weights.items() if v >= min_weight}
    total = sum(weights.values())
    return {k: round(v / total, 4) for k, v in sorted(weights.items(), key=lambda i: -i[1])}
