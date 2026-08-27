"""Prediction intervals for models without native quantiles.

Split-conformal calibration: residuals are collected on out-of-sample backtest
folds, then the empirical quantiles of those residuals - *per horizon bucket*,
because uncertainty grows with distance - are added to the point forecast.

This is distribution-free and honest: the resulting interval is calibrated by
construction on data the model never saw, and the observed coverage is measured
and reported rather than assumed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class ConformalCalibrator:
    """Per-horizon-bucket multiplicative + additive residual quantiles."""

    quantiles: tuple[float, ...] = (0.1, 0.9)
    buckets: list[tuple[int, int]] = field(default_factory=lambda: [(1, 7), (8, 30), (31, 90)])
    offsets: dict[tuple[int, int], dict[float, float]] = field(default_factory=dict)
    scale: float = 1.0
    fitted: bool = False

    def _bucket_for(self, horizon: np.ndarray) -> list[tuple[int, int]]:
        out = []
        for h in horizon:
            match = next(
                (b for b in self.buckets if b[0] <= h <= b[1]), self.buckets[-1]
            )
            out.append(match)
        return out

    def fit(
        self, y_true: np.ndarray, y_pred: np.ndarray, horizon: np.ndarray
    ) -> "ConformalCalibrator":
        true = np.asarray(y_true, dtype=float)
        pred = np.asarray(y_pred, dtype=float)
        steps = np.asarray(horizon, dtype=int)
        mask = np.isfinite(true) & np.isfinite(pred)
        true, pred, steps = true[mask], pred[mask], steps[mask]
        if len(true) == 0:
            return self

        residuals = true - pred
        assignment = np.array([f"{b[0]}_{b[1]}" for b in self._bucket_for(steps)])
        for bucket in self.buckets:
            key = f"{bucket[0]}_{bucket[1]}"
            selected = residuals[assignment == key]
            if len(selected) < 30:
                selected = residuals  # too few points: fall back to the pooled set
            self.offsets[bucket] = {
                float(q): float(np.quantile(selected, q)) for q in self.quantiles
            }
        self.fitted = True
        return self

    def apply(
        self, y_pred: np.ndarray, horizon: np.ndarray, non_negative: bool = True
    ) -> dict[float, np.ndarray]:
        pred = np.asarray(y_pred, dtype=float)
        if not self.fitted:
            return {}
        out: dict[float, np.ndarray] = {}
        buckets = self._bucket_for(np.asarray(horizon, dtype=int))
        for quantile in self.quantiles:
            offset = np.array(
                [self.offsets.get(b, {}).get(float(quantile), 0.0) for b in buckets]
            )
            values = pred + offset * self.scale
            out[float(quantile)] = np.maximum(values, 0.0) if non_negative else values
        return out

    def describe(self) -> dict:
        return {
            "method": "split conformal (per-horizon-bucket residual quantiles)",
            "quantiles": list(self.quantiles),
            "buckets": [f"{a}-{b}" for a, b in self.buckets],
            "offsets": {
                f"{a}-{b}": {str(q): round(v, 4) for q, v in values.items()}
                for (a, b), values in self.offsets.items()
            },
            "fitted": self.fitted,
        }


def buckets_from_contract(horizon_buckets: list[list[int]]) -> list[tuple[int, int]]:
    return [(int(a), int(b)) for a, b in horizon_buckets] or [(1, 7), (8, 30), (31, 90)]


def confidence_score(
    interval_width: float,
    actual_level: float,
    validation_error: float,
    horizon: int,
    max_horizon: int,
    history_periods: int,
    min_history: int = 90,
) -> dict[str, float | str]:
    """A transparent, documented confidence score - not a probability.

    Four penalties, each in [0, 1], averaged with fixed weights:

      * `interval`  - relative width of the prediction interval
      * `accuracy`  - validation WAPE-style error of the champion model
      * `horizon`   - how far out this point is, relative to the max horizon
      * `history`   - how much history the entity actually has

    The result is bucketed into High / Medium / Low. Every input is reported so
    the number can be audited rather than trusted.
    """
    level = max(abs(float(actual_level)), 1e-6)
    interval_penalty = float(np.clip(interval_width / level, 0.0, 1.0))
    accuracy_penalty = float(np.clip(validation_error, 0.0, 1.0))
    horizon_penalty = float(np.clip(horizon / max(max_horizon, 1), 0.0, 1.0))
    history_penalty = float(np.clip(1.0 - history_periods / max(min_history, 1), 0.0, 1.0))

    score = 1.0 - (
        0.35 * interval_penalty
        + 0.35 * accuracy_penalty
        + 0.15 * horizon_penalty
        + 0.15 * history_penalty
    )
    score = float(np.clip(score, 0.0, 1.0))
    label = "high" if score >= 0.66 else ("medium" if score >= 0.4 else "low")
    return {
        "score": round(score, 4),
        "label": label,
        "interval_penalty": round(interval_penalty, 4),
        "accuracy_penalty": round(accuracy_penalty, 4),
        "horizon_penalty": round(horizon_penalty, 4),
        "history_penalty": round(history_penalty, 4),
        "formula": (
            "1 - (0.35*relative_interval_width + 0.35*validation_error "
            "+ 0.15*horizon_share + 0.15*history_shortfall)"
        ),
    }


def coverage_by_horizon(
    backtest: pd.DataFrame, buckets: list[tuple[int, int]], nominal: float = 0.8
) -> list[dict]:
    """Observed interval coverage per horizon bucket."""
    out = []
    for low, high in buckets:
        window = backtest[(backtest["horizon"] >= low) & (backtest["horizon"] <= high)]
        if window.empty or "lower" not in window.columns:
            continue
        inside = (window["actual"] >= window["lower"]) & (window["actual"] <= window["upper"])
        out.append(
            {
                "bucket": f"{low}-{high}",
                "horizon_min": low,
                "horizon_max": high,
                "n": int(len(window)),
                "nominal_coverage": nominal,
                "observed_coverage": round(float(inside.mean()), 4),
                "mean_width": round(float((window["upper"] - window["lower"]).mean()), 4),
            }
        )
    return out
