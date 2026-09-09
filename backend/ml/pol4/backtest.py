"""Pseudo-competition backtest.

The generic platform's rolling-origin folds split on a single axis and cannot
express this problem: they ask "given the series up to here, what comes next?",
while the competition asks "given what has been *searched so far* for a night
that has not happened yet, what will the total be?".

So each fold here reproduces the real task exactly:

    pick a historical cutoff C
    fit pickup curves and priors on check-ins <= C          (complete, knowable)
    take the demand observed at C for check-ins in C+1..C+30 (partial)
    predict their final demand
    score against the complete history we happen to have

Nothing in a fold may read a log date after C. `test_pol4_leakage.py` proves it
by corrupting every post-cutoff row and requiring identical predictions.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..evaluation.metrics import get_metric
from .baseline import PickupBaseline
from .config import Pol4Config
from .loader import CHECKIN, CITY, PROVINCE, Pol4Data
from .submission import build_grid

log = logging.getLogger(__name__)

WAPE = get_metric("wape")
MAE = get_metric("mae")


def normalised_bias(actual: np.ndarray, predicted: np.ndarray) -> float:
    """SUM(pred - actual) / SUM(actual) - the brief's definition.

    Distinct from `metrics.bias`, which is the mean signed error in units. This
    one is scale-free and directly comparable to WAPE.
    """
    total = float(np.sum(actual))
    if abs(total) < 1e-9:
        return float("nan")
    return float(np.sum(predicted - actual) / total)


def score(actual, predicted) -> dict[str, float]:
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    return {
        "n": int(len(actual)),
        "actual_total": round(float(actual.sum()), 2),
        "predicted_total": round(float(predicted.sum()), 2),
        "wape": round(float(WAPE(actual, predicted)), 6),
        "mae": round(float(MAE(actual, predicted)), 4),
        "normalised_bias": round(normalised_bias(actual, predicted), 6),
    }


def _score_by(frame: pd.DataFrame, key: str, prediction: str = "predicted_demand") -> list[dict]:
    rows = []
    for value, group in frame.groupby(key, sort=True):
        entry = {"key": value.item() if hasattr(value, "item") else value}
        entry.update(score(group["actual"], group[prediction]))
        rows.append(entry)
    return rows


def demand_buckets(actual: pd.Series, quantiles: tuple[float, ...]) -> pd.Series:
    """Label each pair by where its ACTUAL demand sits in the fold.

    Bucketing on the actual (not the prediction) is what makes the breakdown
    diagnostic: it answers "where does our error live?", and under WAPE the
    answer is almost always the top decile.
    """
    edges = np.unique(np.quantile(actual, quantiles))
    if len(edges) < 2:
        return pd.Series(["all"] * len(actual), index=actual.index)
    labels = [f"q{quantiles[i]:.2f}-{quantiles[i + 1]:.2f}" for i in range(len(edges) - 1)]
    return pd.cut(actual, bins=edges, labels=labels, include_lowest=True, duplicates="drop")


@dataclass
class FoldResult:
    cutoff: str
    target_start: str
    target_end: str
    overall: dict[str, float]
    reference: dict[str, dict[str, float]]
    by_horizon: list[dict]
    by_horizon_bucket: list[dict]
    by_province: list[dict]
    by_weekday: list[dict]
    by_demand_bucket: list[dict]
    by_curve_source: list[dict]
    worst_cities: list[dict]
    city_wape_quantiles: dict[str, float]
    observed_share: float
    predictions: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)

    def as_dict(self) -> dict[str, Any]:
        payload = {k: v for k, v in self.__dict__.items() if k != "predictions"}
        return payload


def fold_frame(
    data: Pol4Data, cutoff: pd.Timestamp, config: Pol4Config | None = None
) -> pd.DataFrame:
    """The scoring grid for one simulated competition.

    Every (city, check-in) pair in the 30-day window, the demand observed at the
    cutoff, and - because this is history - the complete demand it went on to
    reach. The `actual` column is the only thing here that a real forecaster at
    the cutoff would not have.
    """
    config = config or Pol4Config()
    cutoff = pd.Timestamp(cutoff)
    target_dates = pd.date_range(cutoff + pd.Timedelta(days=1), periods=config.target_days)

    grid = build_grid(data, cutoff, target_dates)
    truth = data.final_demand(target_dates.min(), target_dates.max())
    frame = grid.merge(truth, on=[CITY, CHECKIN], how="left")
    frame["actual"] = frame["final"].fillna(0.0).astype(np.float64)
    return frame.drop(columns=["final"])


def annotate(frame: pd.DataFrame, province_of: dict[int, int], config: Pol4Config) -> pd.DataFrame:
    """Add the segment columns every breakdown in this module reports on."""
    out = frame.copy()
    out[PROVINCE] = out[CITY].map(province_of)
    out["weekday"] = out[CHECKIN].dt.dayofweek
    out["horizon_bucket"] = pd.cut(
        out["horizon"],
        bins=[0, *[high for _, high in config.horizon_buckets]],
        labels=[f"{low}-{high}" for low, high in config.horizon_buckets],
    )
    out["demand_bucket"] = demand_buckets(out["actual"], config.demand_bucket_quantiles)
    out["observation_state"] = np.where(out["observed"] > 0, "observed", "unobserved")
    return out


def run_fold(
    data: Pol4Data, cutoff: pd.Timestamp, config: Pol4Config | None = None
) -> FoldResult:
    """One simulated competition, end to end."""
    config = config or Pol4Config()
    cutoff = pd.Timestamp(cutoff)
    target_dates = pd.date_range(cutoff + pd.Timedelta(days=1), periods=config.target_days)

    frame = fold_frame(data, cutoff, config)
    model = PickupBaseline.fit(data, cutoff, config)
    frame = model.predict(frame)

    # -- reference baselines, scored on the identical grid ------------------
    reference: dict[str, dict[str, float]] = {
        "observed_so_far": score(frame["actual"], frame["observed"]),
    }
    last_year = data.final_demand(
        target_dates.min() - pd.Timedelta(days=364),
        target_dates.max() - pd.Timedelta(days=364),
    ).rename(columns={"final": "last_year"})
    last_year[CHECKIN] = last_year[CHECKIN] + pd.Timedelta(days=364)
    frame = frame.merge(last_year, on=[CITY, CHECKIN], how="left")
    frame["last_year"] = frame["last_year"].fillna(0.0)
    reference["last_year_same_date"] = score(frame["actual"], frame["last_year"])

    frame["city_weekday_mean"] = model.prior.level(
        frame[CITY].to_numpy(), frame[CHECKIN], model.curves.province_of
    )
    reference["city_weekday_mean"] = score(frame["actual"], frame["city_weekday_mean"])

    # -- breakdowns ---------------------------------------------------------
    frame = annotate(frame, model.curves.province_of, config)

    by_city = pd.DataFrame(_score_by(frame, CITY))
    scored_cities = by_city[np.isfinite(by_city["wape"])]

    return FoldResult(
        cutoff=cutoff.date().isoformat(),
        target_start=target_dates.min().date().isoformat(),
        target_end=target_dates.max().date().isoformat(),
        overall=score(frame["actual"], frame["predicted_demand"]),
        reference=reference,
        by_horizon=_score_by(frame, "horizon"),
        by_horizon_bucket=_score_by(frame, "horizon_bucket"),
        by_province=_score_by(frame, PROVINCE),
        by_weekday=_score_by(frame, "weekday"),
        by_demand_bucket=_score_by(frame, "demand_bucket"),
        by_curve_source=_score_by(frame, "curve_source"),
        worst_cities=(
            scored_cities.sort_values("wape", ascending=False)
            .head(20)
            .to_dict(orient="records")
        ),
        city_wape_quantiles={
            f"p{int(q * 100)}": round(float(scored_cities["wape"].quantile(q)), 4)
            for q in (0.1, 0.25, 0.5, 0.75, 0.9)
        }
        if len(scored_cities)
        else {},
        observed_share=round(
            float(frame["observed"].sum() / max(frame["actual"].sum(), 1e-9)), 4
        ),
        predictions=frame,
    )


def run_backtest(
    data: Pol4Data, config: Pol4Config | None = None
) -> dict[str, Any]:
    """Every fold, plus the pooled result across all of them."""
    config = config or Pol4Config()
    folds: list[FoldResult] = []
    for cutoff in config.backtest_cutoffs:
        log.info("pol4 backtest fold at cutoff %s", cutoff)
        folds.append(run_fold(data, pd.Timestamp(cutoff), config))

    pooled_frame = pd.concat([f.predictions for f in folds], ignore_index=True)
    pooled = {
        "overall": score(pooled_frame["actual"], pooled_frame["predicted_demand"]),
        "reference": {
            "observed_so_far": score(pooled_frame["actual"], pooled_frame["observed"]),
            "last_year_same_date": score(pooled_frame["actual"], pooled_frame["last_year"]),
            "city_weekday_mean": score(
                pooled_frame["actual"], pooled_frame["city_weekday_mean"]
            ),
        },
        "by_horizon": _score_by(pooled_frame, "horizon"),
        "by_horizon_bucket": _score_by(pooled_frame, "horizon_bucket"),
        "by_province": _score_by(pooled_frame, PROVINCE),
        "by_weekday": _score_by(pooled_frame, "weekday"),
        "by_demand_bucket": _score_by(pooled_frame, "demand_bucket"),
        "by_curve_source": _score_by(pooled_frame, "curve_source"),
    }
    return {
        "config": {
            "cutoffs": list(config.backtest_cutoffs),
            "target_days": config.target_days,
            "min_city_support": config.min_city_support,
            "min_province_support": config.min_province_support,
            "min_fraction": config.min_fraction,
            "zero_observation_policy": config.zero_observation_policy,
            "prior_window_days": config.prior_window_days,
        },
        "folds": [f.as_dict() for f in folds],
        "pooled": pooled,
    }
