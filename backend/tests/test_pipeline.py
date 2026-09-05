"""End-to-end training pipeline test.

Runs the real pipeline on the small fixture panel: every stage, every artefact.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from ml.contract import ENTITY, DataContract
from ml.pipelines.training import (
    INTERVAL_SWITCH_MARGIN,
    TrainingConfig,
    TrainingPipeline,
    _weighted_blend,
)


@pytest.fixture(scope="module")
def trained(tmp_path_factory, contract, raw_frame):
    data_dir = tmp_path_factory.mktemp("data")
    csv = data_dir / "panel.csv"
    raw_frame.to_csv(csv, index=False)

    run_contract = DataContract.from_dict(contract.to_dict())
    run_contract.path = str(csv)
    run_contract.evaluation.horizons = [7, 14]
    run_contract.evaluation.horizon_buckets = [[1, 7], [8, 14]]

    profile = {
        "name": "test",
        "max_train_rows": 40_000,
        "samples_per_target": 2,
        "cv_folds": 2,
        "models": ["seasonal_naive_7", "moving_average", "lightgbm"],
        "lightgbm": {"n_estimators": 120, "num_leaves": 31, "learning_rate": 0.1},
    }
    config = TrainingConfig(
        contract=run_contract,
        profile_name="test",
        horizon=14,
        seed=42,
        runs_dir=tmp_path_factory.mktemp("runs"),
        profile=profile,
    )
    return TrainingPipeline(config).run()


def test_pipeline_produces_every_artefact(trained):
    expected = {
        "forecast", "backtest", "history", "anomalies", "entity_map",
        "metrics", "insights", "feature_importance", "peaks", "labels",
        "uncertainty", "config", "metadata",
    }
    assert expected.issubset(set(trained.artefacts))
    for name in ("forecast", "backtest", "history"):
        assert (trained.run_dir / f"{name}.parquet").exists()
    assert (trained.run_dir / "metadata.json").exists()


def test_champion_is_a_learned_model_that_beats_the_baseline(trained):
    champion = next(r for r in trained.leaderboard if r["model"] == trained.champion)
    baseline = next(r for r in trained.leaderboard if r["is_baseline"])

    assert not champion["is_baseline"], "a baseline should not win on learnable data"
    assert champion["primary_value"] < baseline["primary_value"]
    assert champion["improvement_vs_baseline"] > 0


def test_leaderboard_is_ranked_by_the_primary_metric(trained):
    values = [
        row["primary_value"] for row in trained.leaderboard if row["primary_value"] is not None
    ]
    assert values == sorted(values), "lower WAPE must rank higher"


def test_forecast_covers_every_hierarchy_level(trained):
    forecast = trained.forecast
    levels = set(forecast["level"].unique())
    assert {"listing", "destination", "category", "market"}.issubset(levels)

    # Bottom-up aggregation must be coherent across levels.
    totals = forecast.groupby("level")["forecast"].sum()
    for level in ("destination", "category", "market"):
        assert totals[level] == pytest.approx(totals["listing"], rel=1e-6)


def test_forecast_starts_after_the_last_observation(trained):
    listing = trained.forecast[trained.forecast["level"] == "listing"]
    history = pd.read_parquet(trained.run_dir / "history.parquet")
    history = history[history["level"] == "listing"]

    assert listing["ds"].min() > history["ds"].max()
    assert listing["horizon"].min() == 1


def test_prediction_interval_brackets_the_point_forecast(trained):
    forecast = trained.forecast
    assert (forecast["lower"] <= forecast["forecast"]).all()
    assert (forecast["forecast"] <= forecast["upper"]).all()
    assert (forecast["lower"] >= 0).all(), "a count target cannot go negative"


def test_backtest_does_not_double_count_overlapping_folds(trained):
    """Overlapping CV folds must be averaged, not summed, in the artefact."""
    backtest = trained.backtest
    listing = backtest[backtest["level"] == "listing"]
    assert not listing.duplicated(subset=["entity_id", "ds"]).any()

    history = pd.read_parquet(trained.run_dir / "history.parquet")
    history = history[history["level"] == "listing"]
    merged = listing.merge(history, on=["entity_id", "ds"], suffixes=("_bt", "_h"))
    assert len(merged) > 0
    # `actual` in the backtest must equal the recorded history for that day.
    assert (merged["actual"] - merged["y"]).abs().max() < 1e-6


def test_metrics_report_horizon_buckets_within_the_trained_horizon(trained):
    buckets = trained.metrics["horizon_scores"][trained.champion]
    assert buckets
    for bucket in buckets:
        assert bucket["horizon_max"] <= trained.metrics["horizon"]


def test_interval_coverage_is_measured_not_assumed(trained):
    coverage = trained.metrics["coverage_by_horizon"]
    assert coverage
    for row in coverage:
        assert 0.0 <= row["observed_coverage"] <= 1.0
        assert row["nominal_coverage"] == pytest.approx(0.8)


def test_metadata_records_everything_needed_to_reproduce(trained):
    metadata = json.loads((trained.run_dir / "metadata.json").read_text(encoding="utf-8"))
    for key in ("run_id", "seed", "champion", "target", "primary_metric", "horizon",
                "feature_names", "training_window", "environment"):
        assert key in metadata, key
    assert metadata["seed"] == 42
    assert len(metadata["feature_names"]) == metadata["n_features"]


def test_run_is_reproducible_under_a_fixed_seed(tmp_path_factory, contract, raw_frame):
    """Two runs with the same seed must produce the same forecast."""
    data_dir = tmp_path_factory.mktemp("repro")
    csv = data_dir / "panel.csv"
    raw_frame.to_csv(csv, index=False)

    def run_once():
        run_contract = DataContract.from_dict(contract.to_dict())
        run_contract.path = str(csv)
        run_contract.evaluation.horizons = [7]
        run_contract.evaluation.horizon_buckets = [[1, 7]]
        config = TrainingConfig(
            contract=run_contract,
            profile_name="test",
            horizon=7,
            seed=11,
            runs_dir=tmp_path_factory.mktemp("runs"),
            profile={
                "max_train_rows": 20_000,
                "samples_per_target": 1,
                "cv_folds": 1,
                "models": ["seasonal_naive_7", "lightgbm"],
                "lightgbm": {"n_estimators": 60, "num_leaves": 15},
            },
        )
        return TrainingPipeline(config).run()

    first, second = run_once(), run_once()
    assert first.champion == second.champion
    pd.testing.assert_series_equal(
        first.forecast["forecast"].reset_index(drop=True),
        second.forecast["forecast"].reset_index(drop=True),
        rtol=1e-9,
    )


def test_pipeline_records_skipped_optional_models(tmp_path_factory, contract, raw_frame):
    """A missing optional model is a warning, never a failure."""
    data_dir = tmp_path_factory.mktemp("optional")
    csv = data_dir / "panel.csv"
    raw_frame.to_csv(csv, index=False)

    run_contract = DataContract.from_dict(contract.to_dict())
    run_contract.path = str(csv)
    run_contract.evaluation.horizons = [7]
    run_contract.evaluation.horizon_buckets = [[1, 7]]

    result = TrainingPipeline(
        TrainingConfig(
            contract=run_contract,
            profile_name="test",
            horizon=7,
            seed=1,
            runs_dir=tmp_path_factory.mktemp("runs"),
            profile={
                "max_train_rows": 20_000,
                "samples_per_target": 1,
                "cv_folds": 1,
                # `chronos` and `nhits` are deliberately unavailable here.
                "models": ["seasonal_naive_7", "lightgbm", "chronos", "nhits"],
                "lightgbm": {"n_estimators": 60, "num_leaves": 15},
            },
        )
    ).run()

    assert result.champion, "the run must still complete"
    assert any("chronos" in warning for warning in result.warnings)
    assert any("nhits" in warning for warning in result.warnings)


def test_insights_are_grounded_in_the_forecast(trained):
    insights = trained.insights
    assert insights["market"]["total_forecast"] == pytest.approx(
        trained.forecast[trained.forecast["level"] == insights["level"]]["forecast"].sum(),
        rel=1e-3,
    )
    assert "decision_opportunities" in insights


# ---------------------------------------------------------- interval choice
def test_the_reported_interval_is_the_one_that_gets_served(trained):
    """Coverage must describe the bounds the product actually shows.

    The champion's own quantiles and conformal residual bounds are different
    intervals; reporting one while serving the other makes every coverage
    number in the run a fiction.
    """
    choice = trained.metrics["interval_selection"]
    assert choice["chosen"] in {"native", "conformal"}
    assert choice["reason"]

    served = trained.metrics["uncertainty_method"]
    if choice["chosen"] == "native":
        assert served == "native model quantiles"
    else:
        assert "conformal" in served


def test_the_chosen_interval_is_the_better_calibrated_one(trained):
    choice = trained.metrics["interval_selection"]
    native, conformal = choice["native_coverage"], choice["conformal_coverage"]
    if native is None or conformal is None:
        pytest.skip("only one interval method was measurable on this run")

    nominal = choice["nominal"]
    picked = native if choice["chosen"] == "native" else conformal
    other = conformal if choice["chosen"] == "native" else native
    # Native wins ties, so it is allowed to be marginally worse by the margin.
    assert abs(picked - nominal) <= abs(other - nominal) + INTERVAL_SWITCH_MARGIN


def test_interval_metrics_are_never_silently_zero(trained):
    """A missing interval must read as "not measured", not as 0% coverage."""
    intervals = trained.metrics["intervals"]
    if intervals["observed_coverage"] is None:
        assert intervals["measured_share"] == 0.0
        assert "note" in intervals
    else:
        assert intervals["measured_share"] > 0
        assert intervals["mean_interval_width"] > 0


def test_a_winning_ensemble_still_reports_an_interval():
    """Regression: the scored ensemble used to hard-code its bounds to NaN.

    Whenever the blend won - which it often does - every coverage number in the
    run collapsed to zero.
    """
    predictions = pd.DataFrame(
        {
            ENTITY: ["a"] * 8,
            "ds": pd.date_range("2024-01-01", periods=4).tolist() * 2,
            "horizon": [1, 2, 3, 4] * 2,
            "fold": [0] * 8,
            "model": ["lightgbm"] * 4 + ["catboost"] * 4,
            "actual": [10.0, 11.0, 12.0, 13.0] * 2,
            "prediction": [9.0, 10.0, 11.0, 12.0, 11.0, 12.0, 13.0, 14.0],
            "lower": [7.0, 8.0, 9.0, 10.0, 9.0, 10.0, 11.0, 12.0],
            "upper": [11.0, 12.0, 13.0, 14.0, 13.0, 14.0, 15.0, 16.0],
        }
    )

    pivot = predictions.pivot_table(
        index=[ENTITY, "ds", "horizon", "fold"], columns="model", values="lower"
    )
    weights = {"lightgbm": 0.5, "catboost": 0.5}
    blended = _weighted_blend(pivot, weights, ["lightgbm", "catboost"])

    assert blended.notna().all(), "the blended bound must exist for every row"
    assert blended.to_numpy().tolist() == [8.0, 9.0, 10.0, 11.0]


def test_a_member_missing_from_a_fold_does_not_drag_the_blend_to_zero():
    """A NaN from a failed member must renormalise, not count as a zero."""
    pivot = pd.DataFrame({"lightgbm": [10.0, 10.0], "catboost": [20.0, np.nan]})
    blended = _weighted_blend(pivot, {"lightgbm": 0.5, "catboost": 0.5}, ["lightgbm", "catboost"])

    assert blended.iloc[0] == pytest.approx(15.0)
    assert blended.iloc[1] == pytest.approx(10.0), "not 5.0 - the absent member is not a zero"
