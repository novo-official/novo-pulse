"""Bias calibration for pickup and remaining-demand predictions.

Phase 1 measured a pooled normalised bias of -0.150: the projection
systematically under-predicts, and worsens with horizon (-0.02 at 1-3 days,
-0.25 at 22-30). That is the cheapest accuracy left on the table, because a
systematic offset costs WAPE linearly.

Two design decisions matter more than the arithmetic:

* **Scale the remaining demand, not the total.** `observed_so_far` is a fact -
  it has already been counted. Multiplying the whole prediction by 1.15 inflates
  a number we know exactly. Only the projected *remainder* is uncertain, so only
  it is calibrated:

      predicted_final = observed + alpha * predicted_remaining

* **Fit on earlier cutoffs only.** A calibration factor learned on the fold it
  scores is not a calibration, it is the answer. For a fold at cutoff C the
  factors come from simulated cutoffs whose entire target window closes at or
  before C, so every number used was knowable to a forecaster standing at C.
  The champion follows the same rule with out-of-fold model predictions and
  only accepts calibration when WAPE does not regress.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .config import Pol4Config

# Searched on a grid rather than solved: the WAPE-optimal scalar is not the
# bias-matching one (WAPE is a sum of absolute errors, not squared), and the
# grid makes that difference visible instead of assuming it away.
ALPHA_GRID = np.round(np.arange(0.80, 2.401, 0.01), 3)


def wape(actual: np.ndarray, predicted: np.ndarray) -> float:
    total = float(np.sum(np.abs(actual)))
    if total < 1e-9:
        return float("nan")
    return float(np.sum(np.abs(predicted - actual)) / total)


def apply_alpha(
    observed: np.ndarray, predicted: np.ndarray, alpha: float | np.ndarray
) -> np.ndarray:
    """Scale only the projected remainder, then re-impose the floor."""
    remaining = np.maximum(predicted - observed, 0.0)
    return observed + np.asarray(alpha, dtype=np.float64) * remaining


def best_alpha(
    actual: np.ndarray, observed: np.ndarray, predicted: np.ndarray
) -> tuple[float, float]:
    """The WAPE-minimising scale factor on a grid, and the bias-matching one."""
    if len(actual) == 0 or np.sum(actual) < 1e-9:
        return 1.0, 1.0
    errors = [wape(actual, apply_alpha(observed, predicted, a)) for a in ALPHA_GRID]
    optimal = float(ALPHA_GRID[int(np.nanargmin(errors))])

    remaining_actual = float(np.sum(actual - observed))
    remaining_pred = float(np.sum(np.maximum(predicted - observed, 0.0)))
    unbiased = remaining_actual / remaining_pred if remaining_pred > 1e-9 else 1.0
    return optimal, float(np.clip(unbiased, ALPHA_GRID[0], ALPHA_GRID[-1]))


@dataclass
class Calibrator:
    """A fitted calibration: one global factor, optionally one per horizon bucket.

    `shrinkage` pulls a bucket's factor toward the global one in proportion to
    how little demand supported it, so a thin bucket cannot swing the forecast
    on noise. It is expressed in units of demand (the same units WAPE weights
    by), not row counts.
    """

    method: str
    alpha_global: float
    alpha_by_bucket: dict[str, float] = field(default_factory=dict)
    bucket_edges: tuple[tuple[int, int], ...] = ()
    support: dict[str, float] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    METHODS = (
        "none",
        "global",
        "horizon",
        "horizon_shrunk",
        "bias_global",
        "bias_horizon_shrunk",
        "guarded_bias_horizon",
    )

    # ------------------------------------------------------------------ fit
    @classmethod
    def fit(
        cls,
        frame: pd.DataFrame,
        method: str = "horizon_shrunk",
        config: Pol4Config | None = None,
        shrinkage: float | None = None,
    ) -> "Calibrator":
        """Fit on a frame of `actual`, `observed`, `predicted_demand`, `horizon`."""
        config = config or Pol4Config()
        if method not in cls.METHODS:
            raise ValueError(f"method must be one of {cls.METHODS}, got {method!r}")
        if method == "none":
            return cls(method="none", alpha_global=1.0)

        actual = frame["actual"].to_numpy(dtype=np.float64)
        observed = frame["observed"].to_numpy(dtype=np.float64)
        predicted = frame["predicted_demand"].to_numpy(dtype=np.float64)
        alpha_global, unbiased_global = best_alpha(actual, observed, predicted)

        diagnostics: dict[str, Any] = {
            "rows": int(len(frame)),
            "alpha_global_wape_optimal": alpha_global,
            "alpha_global_bias_matching": round(unbiased_global, 4),
        }
        guarded = method == "guarded_bias_horizon"
        bias_matching = method.startswith("bias_") or guarded
        if bias_matching:
            alpha_global = unbiased_global
        if guarded:
            alpha_global = float(np.clip(alpha_global, 0.95, 1.10))

        if method in ("global", "bias_global"):
            return cls(
                method=method, alpha_global=alpha_global, diagnostics=diagnostics
            )

        buckets = tuple(tuple(b) for b in config.horizon_buckets)
        labels = _bucket_labels(frame["horizon"].to_numpy(), buckets)
        shrink = config.calibration_shrinkage if shrinkage is None else shrinkage

        alpha_by_bucket: dict[str, float] = {}
        support: dict[str, float] = {}
        for label in {f"{low}-{high}" for low, high in buckets}:
            mask = labels == label
            demand = float(actual[mask].sum())
            support[label] = demand
            if not mask.any():
                alpha_by_bucket[label] = alpha_global
                continue
            alpha_wape, alpha_unbiased = best_alpha(
                actual[mask], observed[mask], predicted[mask]
            )
            alpha = alpha_unbiased if bias_matching else alpha_wape
            if guarded:
                _low, high = (int(value) for value in label.split("-"))
                if high <= 7:
                    alpha_by_bucket[label] = 1.0
                    continue
                # The measured raw bias is already near zero at D-1..D-7.
                # Far-horizon full bias matching fixed totals but damaged WAPE,
                # so only a bounded fraction of that correction is allowed.
                upper = 1.12 if high <= 21 else 1.08
                weight = demand / (demand + shrink) if demand + shrink > 0 else 0.0
                alpha = 1.0 + weight * (alpha - 1.0)
                alpha_by_bucket[label] = float(np.clip(alpha, 0.95, upper))
                continue
            if method in ("horizon_shrunk", "bias_horizon_shrunk"):
                weight = demand / (demand + shrink) if demand + shrink > 0 else 0.0
                alpha = weight * alpha + (1.0 - weight) * alpha_global
            alpha_by_bucket[label] = float(alpha)

        diagnostics["shrinkage"] = shrink
        return cls(
            method=method,
            alpha_global=alpha_global,
            alpha_by_bucket=alpha_by_bucket,
            bucket_edges=buckets,
            support=support,
            diagnostics=diagnostics,
        )

    # ---------------------------------------------------------------- apply
    def alphas(self, horizons: np.ndarray) -> np.ndarray:
        if self.method in ("none", "global", "bias_global") or not self.alpha_by_bucket:
            return np.full(len(horizons), self.alpha_global, dtype=np.float64)
        labels = _bucket_labels(np.asarray(horizons), self.bucket_edges)
        return np.array(
            [self.alpha_by_bucket.get(label, self.alpha_global) for label in labels],
            dtype=np.float64,
        )

    def apply(self, frame: pd.DataFrame, column: str = "predicted_demand") -> np.ndarray:
        """Calibrated predictions, still floored at the observed demand."""
        if self.method == "none":
            return frame[column].to_numpy(dtype=np.float64)
        observed = frame["observed"].to_numpy(dtype=np.float64)
        predicted = frame[column].to_numpy(dtype=np.float64)
        calibrated = apply_alpha(observed, predicted, self.alphas(frame["horizon"].to_numpy()))
        return np.maximum(calibrated, observed)

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "alpha_global": round(self.alpha_global, 4),
            "alpha_by_bucket": {k: round(v, 4) for k, v in sorted(self.alpha_by_bucket.items())},
            "support": {k: round(v, 1) for k, v in sorted(self.support.items())},
            "diagnostics": self.diagnostics,
        }


def _bucket_labels(horizons: np.ndarray, buckets: Iterable[tuple[int, int]]) -> np.ndarray:
    labels = np.empty(len(horizons), dtype=object)
    labels[:] = "other"
    for low, high in buckets:
        mask = (horizons >= low) & (horizons <= high)
        labels[mask] = f"{low}-{high}"
    return labels


def calibration_cutoffs(
    cutoff: pd.Timestamp, config: Pol4Config
) -> list[pd.Timestamp]:
    """Simulated cutoffs whose target windows close at or before `cutoff`.

    Anything later would have the fold's own outcome in it. Anything earlier is
    fair game: a forecaster standing at `cutoff` has already watched those
    windows play out.
    """
    span = config.target_days
    return [
        cutoff - pd.Timedelta(days=span * step)
        for step in range(1, config.calibration_windows + 1)
    ]
