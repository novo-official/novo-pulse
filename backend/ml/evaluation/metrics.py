"""Metric registry.

The competition's official metric is unknown until the day. Every consumer -
leaderboard, champion selection, ensemble weighting, backtest reporting - reads
the metric by name from this registry, so switching is a one-line config change:

    evaluation:
      primary_metric: wape
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

EPS = 1e-9


@dataclass(frozen=True)
class MetricSpec:
    name: str
    label: str
    fn: Callable[[np.ndarray, np.ndarray], float]
    greater_is_better: bool
    unit: str = ""
    description: str = ""

    def __call__(self, y_true, y_pred) -> float:
        true = np.asarray(y_true, dtype=float).ravel()
        pred = np.asarray(y_pred, dtype=float).ravel()
        mask = np.isfinite(true) & np.isfinite(pred)
        if mask.sum() == 0:
            return float("nan")
        return float(self.fn(true[mask], pred[mask]))


# ------------------------------------------------------------------ metrics
def _mae(t, p):
    return np.mean(np.abs(t - p))


def _rmse(t, p):
    return np.sqrt(np.mean((t - p) ** 2))


def _wape(t, p):
    denominator = np.sum(np.abs(t))
    return np.sum(np.abs(t - p)) / denominator if denominator > EPS else np.nan


def _mape(t, p):
    mask = np.abs(t) > EPS
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((t[mask] - p[mask]) / t[mask])) * 100


def _smape(t, p):
    denominator = (np.abs(t) + np.abs(p)) / 2.0
    mask = denominator > EPS
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs(t[mask] - p[mask]) / denominator[mask]) * 100


def _rmsle(t, p):
    t = np.maximum(t, 0.0)
    p = np.maximum(p, 0.0)
    return np.sqrt(np.mean((np.log1p(t) - np.log1p(p)) ** 2))


def _r2(t, p):
    total = np.sum((t - np.mean(t)) ** 2)
    if total < EPS:
        return np.nan
    return 1.0 - np.sum((t - p) ** 2) / total


def _bias(t, p):
    return np.mean(p - t)


def _mase_naive(t, p):
    """MASE against a lag-1 naive on the evaluation window itself."""
    if len(t) < 2:
        return np.nan
    scale = np.mean(np.abs(np.diff(t)))
    return np.mean(np.abs(t - p)) / scale if scale > EPS else np.nan


def _poisson_deviance(t, p):
    p = np.maximum(p, EPS)
    t_safe = np.maximum(t, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(t_safe > 0, t_safe * np.log(t_safe / p), 0.0)
    return float(2.0 * np.mean(term - (t_safe - p)))


REGISTRY: dict[str, MetricSpec] = {
    spec.name: spec
    for spec in (
        MetricSpec("mae", "MAE", _mae, False, "units", "Mean absolute error."),
        MetricSpec("rmse", "RMSE", _rmse, False, "units", "Root mean squared error."),
        MetricSpec(
            "wape",
            "WAPE",
            _wape,
            False,
            "ratio",
            "Sum of absolute errors divided by the sum of actuals. Robust to zeros.",
        ),
        MetricSpec("mape", "MAPE", _mape, False, "%", "Mean absolute percentage error."),
        MetricSpec("smape", "sMAPE", _smape, False, "%", "Symmetric MAPE."),
        MetricSpec("rmsle", "RMSLE", _rmsle, False, "log", "RMSE in log1p space."),
        MetricSpec("r2", "R²", _r2, True, "ratio", "Coefficient of determination."),
        MetricSpec("bias", "Bias", _bias, False, "units", "Mean signed error."),
        MetricSpec("mase", "MASE", _mase_naive, False, "ratio", "Error relative to lag-1 naive."),
        MetricSpec(
            "poisson_deviance",
            "Poisson Dev.",
            _poisson_deviance,
            False,
            "dev",
            "Deviance for count targets.",
        ),
    )
}

# Metrics that only make sense for non-negative count-like targets.
COUNT_ONLY = {"rmsle", "poisson_deviance"}


def get_metric(name: str) -> MetricSpec:
    key = str(name).strip().lower()
    if key not in REGISTRY:
        raise KeyError(f"Unknown metric '{name}'. Available: {sorted(REGISTRY)}")
    return REGISTRY[key]


def available_metrics(non_negative: bool = True) -> list[str]:
    if non_negative:
        return sorted(REGISTRY)
    return sorted(set(REGISTRY) - COUNT_ONLY)


def evaluate(
    y_true, y_pred, metrics: list[str] | None = None, non_negative: bool = True
) -> dict[str, float]:
    """Evaluate several metrics at once, skipping ones that do not apply."""
    names = metrics or ["mae", "rmse", "wape", "smape", "mape", "r2", "bias"]
    out: dict[str, float] = {}
    for name in names:
        if name in COUNT_ONLY and not non_negative:
            continue
        try:
            value = get_metric(name)(y_true, y_pred)
        except (KeyError, ValueError, ZeroDivisionError):
            continue
        out[name] = None if value is None or not np.isfinite(value) else round(float(value), 6)
    return out


# ---------------------------------------------------- probabilistic metrics
def pinball_loss(y_true, y_pred, quantile: float) -> float:
    """Quantile (pinball) loss - the proper score for a single quantile."""
    true = np.asarray(y_true, dtype=float).ravel()
    pred = np.asarray(y_pred, dtype=float).ravel()
    delta = true - pred
    return float(np.mean(np.maximum(quantile * delta, (quantile - 1) * delta)))


def interval_coverage(y_true, lower, upper) -> float:
    """Share of actuals that fall inside the prediction interval."""
    true = np.asarray(y_true, dtype=float).ravel()
    low = np.asarray(lower, dtype=float).ravel()
    high = np.asarray(upper, dtype=float).ravel()
    mask = np.isfinite(true) & np.isfinite(low) & np.isfinite(high)
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean((true[mask] >= low[mask]) & (true[mask] <= high[mask])))


def interval_width(lower, upper, y_true=None) -> float:
    """Mean interval width, normalised by the mean actual when available."""
    low = np.asarray(lower, dtype=float).ravel()
    high = np.asarray(upper, dtype=float).ravel()
    width = float(np.mean(high - low))
    if y_true is not None:
        scale = float(np.mean(np.abs(np.asarray(y_true, dtype=float))))
        if scale > EPS:
            return width / scale
    return width


def evaluate_intervals(
    y_true, lower, upper, nominal: float = 0.8, y_median=None
) -> dict[str, float]:
    coverage = interval_coverage(y_true, lower, upper)
    result = {
        "nominal_coverage": round(float(nominal), 4),
        "observed_coverage": None if not np.isfinite(coverage) else round(coverage, 4),
        "coverage_gap": (
            None if not np.isfinite(coverage) else round(coverage - float(nominal), 4)
        ),
        "mean_interval_width": round(interval_width(lower, upper), 4),
        "relative_interval_width": round(interval_width(lower, upper, y_true), 4),
    }
    tail = (1.0 - nominal) / 2.0
    result["pinball_lower"] = round(pinball_loss(y_true, lower, tail), 6)
    result["pinball_upper"] = round(pinball_loss(y_true, upper, 1.0 - tail), 6)
    if y_median is not None:
        result["pinball_median"] = round(pinball_loss(y_true, y_median, 0.5), 6)
    return result
