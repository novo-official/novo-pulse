"""Metrics, splitters, uncertainty and ensemble-weighting tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.evaluation.metrics import (
    available_metrics,
    evaluate,
    evaluate_intervals,
    get_metric,
    interval_coverage,
    pinball_loss,
)
from ml.evaluation.splitters import RollingOriginSplitter, holdout_split
from ml.evaluation.uncertainty import ConformalCalibrator, confidence_score
from ml.models.ensemble import compute_weights


# ------------------------------------------------------------------ metrics
def test_perfect_prediction_scores_perfectly():
    truth = np.array([1.0, 2.0, 3.0, 4.0])
    scores = evaluate(truth, truth, ["mae", "rmse", "wape", "smape", "r2"])

    assert scores["mae"] == 0
    assert scores["rmse"] == 0
    assert scores["wape"] == 0
    assert scores["r2"] == 1


def test_wape_is_scale_relative():
    truth = np.array([10.0, 20.0, 30.0])
    prediction = truth + 3.0
    assert get_metric("wape")(truth, prediction) == pytest.approx(9 / 60)


def test_wape_survives_zero_heavy_targets():
    """WAPE is chosen precisely because MAPE explodes on zeros."""
    truth = np.array([0.0, 0.0, 5.0, 10.0])
    prediction = np.array([1.0, 0.0, 4.0, 11.0])

    assert np.isfinite(get_metric("wape")(truth, prediction))


def test_metric_orientation_is_declared():
    assert get_metric("wape").greater_is_better is False
    assert get_metric("r2").greater_is_better is True


def test_unknown_metric_raises():
    with pytest.raises(KeyError):
        get_metric("not_a_metric")


def test_count_only_metrics_are_hidden_for_signed_targets():
    assert "poisson_deviance" in available_metrics(non_negative=True)
    assert "poisson_deviance" not in available_metrics(non_negative=False)


def test_evaluate_ignores_nans():
    truth = np.array([1.0, np.nan, 3.0])
    prediction = np.array([1.0, 5.0, 3.0])
    assert evaluate(truth, prediction, ["mae"])["mae"] == 0


# ---------------------------------------------------------- probabilistic
def test_interval_coverage_counts_containment():
    truth = np.array([1.0, 2.0, 3.0, 4.0])
    assert interval_coverage(truth, truth - 1, truth + 1) == 1.0
    assert interval_coverage(truth, truth + 1, truth + 2) == 0.0


def test_pinball_loss_penalises_asymmetrically():
    truth = np.full(100, 10.0)
    # A P90 forecast that is too low is penalised more than one that is too high.
    too_low = pinball_loss(truth, np.full(100, 8.0), 0.9)
    too_high = pinball_loss(truth, np.full(100, 12.0), 0.9)
    assert too_low > too_high


def test_evaluate_intervals_reports_the_gap():
    rng = np.random.default_rng(0)
    truth = rng.normal(10, 2, 2000)
    result = evaluate_intervals(truth, np.full(2000, 10 - 2.56), np.full(2000, 10 + 2.56), 0.8)

    assert result["observed_coverage"] == pytest.approx(0.8, abs=0.05)
    assert abs(result["coverage_gap"]) < 0.06


# ------------------------------------------------------------------ splits
def test_rolling_origin_folds_never_overlap_train_and_valid():
    stamps = pd.Series(pd.date_range("2024-01-01", periods=400))
    folds = RollingOriginSplitter(horizon=30, n_folds=3, step=30).split(stamps)

    assert len(folds) == 3
    for fold in folds:
        assert fold.train_end < fold.valid_start
        assert fold.valid_start <= fold.valid_end


def test_rolling_origin_folds_move_forward():
    stamps = pd.Series(pd.date_range("2024-01-01", periods=400))
    folds = RollingOriginSplitter(horizon=30, n_folds=3, step=30).split(stamps)

    ends = [fold.train_end for fold in folds]
    assert ends == sorted(ends), "later folds must train on more data"


def test_splitter_degrades_gracefully_on_short_history():
    stamps = pd.Series(pd.date_range("2024-01-01", periods=60))
    folds = RollingOriginSplitter(horizon=30, n_folds=4).split(stamps)

    assert len(folds) >= 1
    assert folds[0].train_end < folds[0].valid_start


def test_holdout_reserves_the_final_window():
    stamps = pd.Series(pd.date_range("2024-01-01", periods=100))
    fold = holdout_split(stamps, 14)

    assert (fold.valid_end - fold.valid_start).days == 13
    assert fold.valid_end == stamps.iloc[-1]


# -------------------------------------------------------------- conformal
def test_conformal_calibration_reaches_nominal_coverage():
    rng = np.random.default_rng(3)
    n = 4000
    horizon = rng.integers(1, 31, n)
    prediction = rng.uniform(5, 50, n)
    truth = prediction + rng.normal(0, 4, n)

    calibrator = ConformalCalibrator(quantiles=(0.1, 0.9), buckets=[(1, 7), (8, 30)])
    calibrator.fit(truth, prediction, horizon)
    bounds = calibrator.apply(prediction, horizon)

    coverage = interval_coverage(truth, bounds[0.1], bounds[0.9])
    assert coverage == pytest.approx(0.8, abs=0.04)


def test_conformal_widens_the_interval_for_longer_horizons():
    rng = np.random.default_rng(5)
    n = 6000
    horizon = rng.integers(1, 31, n)
    prediction = np.full(n, 20.0)
    # Noise grows with the horizon, exactly as forecast error does.
    truth = prediction + rng.normal(0, 1, n) * horizon

    calibrator = ConformalCalibrator(quantiles=(0.1, 0.9), buckets=[(1, 7), (8, 30)])
    calibrator.fit(truth, prediction, horizon)

    short = calibrator.offsets[(1, 7)]
    long = calibrator.offsets[(8, 30)]
    assert (long[0.9] - long[0.1]) > (short[0.9] - short[0.1])


def test_unfitted_calibrator_returns_nothing():
    assert ConformalCalibrator().apply(np.array([1.0]), np.array([1])) == {}


def test_confidence_score_is_bounded_and_explained():
    result = confidence_score(
        interval_width=5,
        actual_level=10,
        validation_error=0.2,
        horizon=7,
        max_horizon=30,
        history_periods=365,
    )
    assert 0 <= result["score"] <= 1
    assert result["label"] in {"high", "medium", "low"}
    assert "0.35" in result["formula"]


def test_confidence_falls_with_a_wider_interval():
    tight = confidence_score(1, 10, 0.1, 7, 30, 365)["score"]
    wide = confidence_score(9, 10, 0.1, 7, 30, 365)["score"]
    assert tight > wide


# --------------------------------------------------------------- ensemble
def test_ensemble_weights_favour_the_better_model():
    weights = compute_weights({"good": 0.10, "ok": 0.12}, "wape")
    assert weights["good"] > weights["ok"]
    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-3)


def test_ensemble_excludes_a_much_worse_model():
    weights = compute_weights({"good": 0.10, "terrible": 0.90}, "wape")
    assert "terrible" not in weights


def test_ensemble_respects_metric_orientation():
    """For R², higher is better - the weighting must not invert it."""
    weights = compute_weights({"better": 0.95, "worse": 0.80}, "r2")
    assert weights["better"] > weights.get("worse", 0)


def test_ensemble_weights_are_not_hardcoded():
    first = compute_weights({"a": 0.10, "b": 0.11}, "wape")
    second = compute_weights({"a": 0.11, "b": 0.10}, "wape")
    assert first != second
