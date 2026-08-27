"""Demand censoring, integer targets and hyper-parameter search."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.contract import ENTITY, TARGET, TS, DataContract
from ml.data.censoring import CENSORING_RATIO, analyse, estimate_unconstrained
from ml.models.tuning import SEARCH_SPACES, optuna_available, tune_model


def _contract(enabled: bool = True, column: str | None = "capacity") -> DataContract:
    return DataContract.from_dict(
        {
            "schema": {"timestamp": "date", "target": "bookings"},
            "target_options": {"censoring_column": column, "censoring_enabled": enabled},
        }
    )


def _panel(capacity: float = 10.0, sellout_days: int = 20, n: int = 70) -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=n)
    observed = np.full(n, 4.0)
    observed[:sellout_days] = capacity          # sold out
    return pd.DataFrame({ENTITY: "a", TS: dates, TARGET: observed, "capacity": capacity})


# ------------------------------------------------------------------ analysis
def test_censoring_is_off_without_a_capacity_column():
    report = analyse(_panel(), _contract(column=None))
    assert report.enabled is False
    assert "no censoring" in report.reason


def test_censoring_is_off_when_disabled_in_the_contract():
    report = analyse(_panel(), _contract(enabled=False))
    assert report.enabled is False


def test_censoring_is_off_when_the_column_is_absent_from_the_data():
    frame = _panel().drop(columns=["capacity"])
    report = analyse(frame, _contract())
    assert report.enabled is False
    assert "not in the dataset" in report.reason


def test_censoring_measures_the_sold_out_share():
    report = analyse(_panel(sellout_days=21, n=70), _contract())
    assert report.enabled is True
    assert report.censored_share == pytest.approx(0.3, abs=0.01)
    assert report.censored_periods == 21
    assert report.affected_entities == 1


def test_a_never_full_series_reports_no_censoring():
    frame = _panel(capacity=100.0, sellout_days=0)
    report = analyse(frame, _contract())
    assert report.censored_share == 0.0


def test_censoring_threshold_is_a_ratio_not_equality():
    """Selling 99 of 100 rooms is effectively sold out."""
    frame = _panel(capacity=100.0, sellout_days=0)
    frame.loc[frame.index[:10], TARGET] = 99.0
    report = analyse(frame, _contract())

    assert CENSORING_RATIO < 1.0
    assert report.censored_periods == 10


# ------------------------------------------------------- unconstrained demand
def test_unconstrained_estimate_never_falls_below_the_observed_value():
    """Demand was at least what we saw - the estimate must not shrink it."""
    frame = _panel(sellout_days=20)
    frame["is_censored"] = frame[TARGET] >= frame["capacity"] * CENSORING_RATIO

    estimate = estimate_unconstrained(frame, season=7)
    assert (estimate >= frame[TARGET].to_numpy()).all()


def test_unconstrained_estimate_lifts_sold_out_days_toward_comparable_days():
    """A sold-out Friday is estimated from other Fridays, not from Mondays."""
    dates = pd.date_range("2026-01-05", periods=84)  # starts on a Monday
    weekday = dates.dayofweek
    observed = np.where(weekday == 4, 30.0, 8.0)     # Fridays are busy
    capacity = np.full(84, 60.0)
    # Cap three Fridays at a low capacity so they read as sold out.
    fridays = np.where(weekday == 4)[0][:3]
    capacity[fridays] = 12.0
    observed[fridays] = 12.0

    frame = pd.DataFrame({ENTITY: "a", TS: dates, TARGET: observed, "capacity": capacity})
    frame["is_censored"] = frame[TARGET] >= frame["capacity"] * CENSORING_RATIO

    estimate = estimate_unconstrained(frame, season=7)
    # The censored Fridays are lifted toward the uncensored Friday level (30).
    assert estimate[fridays].min() > 12.0
    assert estimate[fridays].max() == pytest.approx(30.0)
    # Untouched days stay exactly as observed.
    untouched = ~frame["is_censored"].to_numpy()
    np.testing.assert_allclose(estimate[untouched], observed[untouched])


def test_unconstrained_estimate_is_a_noop_without_comparable_periods():
    """With nothing uncensored to learn from, we must not invent an uplift."""
    frame = _panel(sellout_days=70, n=70)  # every period sold out
    frame["is_censored"] = True

    estimate = estimate_unconstrained(frame, season=7)
    np.testing.assert_allclose(estimate, frame[TARGET].to_numpy())


def test_validator_reports_censoring_as_a_finding():
    from ml.data.validator import DataValidator

    report = DataValidator(_contract()).validate(_panel(sellout_days=30, n=70))
    finding = next(f for f in report["findings"] if f["code"] == "demand_censoring")
    assert "capacity" in finding["title"].lower()
    assert finding["context"]["capacity_column"] == "capacity"


# --------------------------------------------------------------- integer target
def test_integer_targets_round_outwards(tmp_path_factory, contract, raw_frame):
    """Rounding a count forecast must not narrow the prediction interval."""
    from ml.pipelines.training import TrainingConfig, TrainingPipeline

    data_dir = tmp_path_factory.mktemp("int-target")
    csv = data_dir / "panel.csv"
    raw_frame.to_csv(csv, index=False)

    run_contract = DataContract.from_dict(contract.to_dict())
    run_contract.path = str(csv)
    run_contract.target_options.integer = True
    run_contract.evaluation.horizons = [7]
    run_contract.evaluation.horizon_buckets = [[1, 7]]

    result = TrainingPipeline(
        TrainingConfig(
            contract=run_contract,
            profile_name="test",
            horizon=7,
            seed=5,
            runs_dir=tmp_path_factory.mktemp("runs"),
            profile={
                "max_train_rows": 20_000,
                "samples_per_target": 1,
                "cv_folds": 1,
                "models": ["seasonal_naive_7", "lightgbm"],
                "lightgbm": {"n_estimators": 60, "num_leaves": 15},
            },
        )
    ).run()

    forecast = result.forecast[result.forecast["level"] == "listing"]
    values = forecast[["forecast", "lower", "upper"]].to_numpy()
    assert np.allclose(values, np.round(values)), "a count forecast must be whole numbers"
    assert (forecast["lower"] <= forecast["forecast"]).all()
    assert (forecast["forecast"] <= forecast["upper"]).all()


def test_censoring_appears_in_the_run_metrics(tmp_path_factory, contract, raw_frame):
    from ml.pipelines.training import TrainingConfig, TrainingPipeline

    data_dir = tmp_path_factory.mktemp("censor-run")
    csv = data_dir / "panel.csv"
    raw_frame.to_csv(csv, index=False)

    run_contract = DataContract.from_dict(contract.to_dict())
    run_contract.path = str(csv)
    run_contract.target_options.censoring_column = "capacity"
    run_contract.target_options.censoring_enabled = True
    run_contract.evaluation.horizons = [7]
    run_contract.evaluation.horizon_buckets = [[1, 7]]

    result = TrainingPipeline(
        TrainingConfig(
            contract=run_contract,
            profile_name="test",
            horizon=7,
            seed=5,
            runs_dir=tmp_path_factory.mktemp("runs"),
            profile={
                "max_train_rows": 20_000,
                "samples_per_target": 1,
                "cv_folds": 1,
                "models": ["seasonal_naive_7", "lightgbm"],
                "lightgbm": {"n_estimators": 60, "num_leaves": 15},
            },
        )
    ).run()

    censoring = result.metrics["censoring"]
    assert censoring["enabled"] is True
    assert censoring["capacity_column"] == "capacity"
    assert "method" in censoring


# ------------------------------------------------------------------- tuning
def test_tuning_only_covers_models_with_a_search_space():
    """A model with no search space returns None - there is nothing to tune."""
    assert set(SEARCH_SPACES) == {"lightgbm", "catboost", "nhits"}
    assert tune_model("seasonal_naive_7", lambda **_: None, {}, None, None) is None  # type: ignore[arg-type]


def test_tuning_reports_an_accurate_skip_reason(engine):
    """Skipping must say why. A wrong reason is worse than no reason."""
    from ml.models.base import FitContext, PredictContext
    from ml.models.gbdt import LightGBMModel

    horizon = engine.config.max_horizon
    origin = engine.tensor.origin_index - horizon
    inference = engine.build_inference(origin, horizon)
    # Only a couple of rows, far below the minimum for a meaningful search.
    tiny = type(inference)(
        X=inference.X.head(3),
        y=inference.y[:3],
        meta=inference.meta.head(3),
        categorical=inference.categorical,
    )
    outcome = tune_model(
        "lightgbm",
        LightGBMModel,
        {},
        FitContext(engine=engine, train_end_idx=origin, frame=tiny),
        PredictContext(engine=engine, origin_idx=origin, horizon=horizon, frame=tiny),
    )

    assert outcome is not None, "a skip must be reported, not swallowed"
    assert outcome.n_trials == 0
    assert outcome.improved is False
    available, _ = optuna_available()
    expected = "validation points" if available else "optuna"
    assert expected in outcome.note.lower(), outcome.note


@pytest.mark.skipif(not optuna_available()[0], reason="Optuna is not installed")
def test_tuning_searches_and_never_returns_worse_parameters(engine):
    """The search must keep the defaults unless it genuinely beat them."""
    from ml.models.base import FitContext, PredictContext
    from ml.models.gbdt import LightGBMModel

    horizon = engine.config.max_horizon
    origin = engine.tensor.origin_index - horizon
    train = engine.build_training(train_end_idx=origin, samples_per_target=1, seed=2)
    inference = engine.build_inference(origin, horizon)

    outcome = tune_model(
        "lightgbm",
        LightGBMModel,
        {"n_estimators": 40, "num_leaves": 15},
        FitContext(engine=engine, train_end_idx=origin, frame=train, seed=2),
        PredictContext(engine=engine, origin_idx=origin, horizon=horizon, frame=inference),
        metric="wape",
        max_minutes=0.5,
        max_trials=4,
        seed=2,
    )

    assert outcome is not None
    assert outcome.n_trials > 0, "at least one trial should complete in 30s"
    if outcome.improved:
        assert outcome.best_score <= outcome.baseline_score
    else:
        # Not improved => the defaults must come back untouched.
        assert outcome.best_params == {"n_estimators": 40, "num_leaves": 15}


def test_tuning_is_off_by_default_in_the_demo_profile():
    from ml.contract import load_profile

    assert load_profile("demo")["tuning"]["enabled"] is False
    assert load_profile("competition")["tuning"]["enabled"] is False
    assert load_profile("full")["tuning"]["enabled"] is True
