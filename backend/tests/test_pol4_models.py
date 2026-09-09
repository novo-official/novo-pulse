"""Calibration, remaining-demand models, ensembling and stability."""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from ml.pol4 import load_pol4

from ml.pol4.baseline import PickupBaseline
from ml.pol4.calibration import Calibrator, apply_alpha, best_alpha, calibration_cutoffs
from ml.pol4.dataset import (
    build_inference_frame,
    build_training_frame,
    load_materialized_training_frame,
    materialize_training_frame,
)
from ml.pol4.experiments import build_context, run_experiment, baseline_predictor
from ml.pol4.features import GROUP_ORDER, feature_names
from ml.pol4.loader import CHECKIN, SEARCHES
from ml.pol4.models import RemainingDemandModel
from ml.pol4 import stability as stability_module
from tests.pol4_fixtures import write_dataset


@pytest.fixture(scope="module")
def pol4(tmp_path_factory):
    config = write_dataset(tmp_path_factory.mktemp("pol4_models"))
    return load_pol4(config), config


@pytest.fixture(scope="module")
def fitted(pol4):
    data, config = pol4
    baseline = PickupBaseline.fit(data, config.cutoff, config)
    train = build_training_frame(data, config.cutoff, GROUP_ORDER, baseline, config)
    model = RemainingDemandModel(
        "lightgbm", params={"n_estimators": 60, "num_leaves": 15}, seed=config.seed
    ).fit(train.X, train.y)
    return data, config, baseline, model


# ------------------------------------------------------------- calibration
def test_calibration_scales_only_the_remaining_demand():
    observed = np.array([100.0, 0.0, 50.0])
    predicted = np.array([200.0, 10.0, 50.0])
    calibrated = apply_alpha(observed, predicted, 2.0)
    # The known part is untouched; only the projected remainder doubles.
    np.testing.assert_allclose(calibrated, [300.0, 20.0, 50.0])


def test_calibration_never_drops_below_observed():
    frame = pd.DataFrame(
        {
            "observed": [100.0, 40.0],
            "predicted_demand": [200.0, 60.0],
            "horizon": [5, 20],
        }
    )
    calibrator = Calibrator(method="global", alpha_global=0.1)
    assert (calibrator.apply(frame) >= frame["observed"]).all()


def test_best_alpha_recovers_a_known_scale():
    rng = np.random.default_rng(0)
    observed = rng.uniform(10, 100, 4000)
    remaining = rng.uniform(5, 200, 4000)
    actual = observed + 1.4 * remaining
    optimal, unbiased = best_alpha(actual, observed, observed + remaining)
    assert optimal == pytest.approx(1.4, abs=0.02)
    assert unbiased == pytest.approx(1.4, abs=0.02)


def test_calibration_windows_all_close_before_the_fold(pol4):
    _, config = pol4
    cutoff = pd.Timestamp("2024-10-01")
    for earlier in calibration_cutoffs(cutoff, config):
        window_end = earlier + pd.Timedelta(days=config.target_days)
        assert window_end <= cutoff, "a calibration window must close before the fold"


def test_calibrators_cannot_see_the_fold_they_score(pol4):
    """The decisive calibration-leakage test.

    Corrupt the fold's own outcome - every check-in after its cutoff - and
    require the fitted calibration factors to be identical.
    """
    data, config = pol4
    cutoff = pd.Timestamp("2024-09-01")
    before = build_context(data, cutoff, config).calibrators

    poisoned = data.search.copy()
    poisoned.loc[poisoned[CHECKIN] > cutoff, SEARCHES] = 10**6
    after = build_context(
        dataclasses.replace(data, search=poisoned), cutoff, config
    ).calibrators

    for method in before:
        assert before[method].alpha_global == after[method].alpha_global, method
        assert before[method].alpha_by_bucket == after[method].alpha_by_bucket, method


def test_shrinkage_pulls_a_thin_bucket_toward_the_global_factor(pol4):
    _, config = pol4
    rng = np.random.default_rng(3)
    n = 3000
    thin_rows = np.arange(n) < 20
    # The thin bucket wants a very different factor, but is supported by almost
    # no demand - exactly the case shrinkage exists to damp.
    horizon = np.where(thin_rows, 2, 9)
    observed = np.where(thin_rows, 0.5, 400.0) * rng.uniform(0.8, 1.2, n)
    remaining = np.where(thin_rows, 0.5, 400.0) * rng.uniform(0.8, 1.2, n)
    scale = np.where(thin_rows, 2.3, 1.5)
    frame = pd.DataFrame(
        {
            "observed": observed,
            "predicted_demand": observed + remaining,
            "actual": observed + scale * remaining,
            "horizon": horizon,
        }
    )
    plain = Calibrator.fit(frame, "horizon", config)
    shrunk = Calibrator.fit(frame, "horizon_shrunk", config)
    thin = "1-3"
    assert abs(shrunk.alpha_by_bucket[thin] - shrunk.alpha_global) < abs(
        plain.alpha_by_bucket[thin] - plain.alpha_global
    )


# ------------------------------------------------------------------ models
def test_remaining_demand_target_keeps_the_floor(fitted):
    data, config, baseline, model = fitted
    infer = build_inference_frame(
        data, config.cutoff, config.target_dates(), GROUP_ORDER, baseline, config
    )
    observed = infer.meta["observed"].to_numpy()
    predicted = model.predict_final(infer.X, observed)
    assert (predicted >= observed - 1e-9).all()
    assert np.isfinite(predicted).all()


def test_predicted_remaining_is_never_negative(fitted):
    data, config, baseline, model = fitted
    infer = build_inference_frame(
        data, config.cutoff, config.target_dates(), GROUP_ORDER, baseline, config
    )
    assert (model.predict_remaining(infer.X) >= 0).all()


def test_model_is_deterministic(fitted):
    data, config, baseline, model = fitted
    train = build_training_frame(data, config.cutoff, GROUP_ORDER, baseline, config)
    params = {"n_estimators": 40, "num_leaves": 15}
    first = RemainingDemandModel("lightgbm", params=params, seed=7).fit(train.X, train.y)
    second = RemainingDemandModel("lightgbm", params=params, seed=7).fit(train.X, train.y)
    np.testing.assert_allclose(
        first.predict_remaining(train.X.head(500)),
        second.predict_remaining(train.X.head(500)),
    )


def test_log1p_is_off_unless_asked(fitted):
    _, _, _, model = fitted
    assert model.log1p is False


def test_importance_covers_every_feature(fitted):
    _, _, _, model = fitted
    importance = model.importance()
    assert set(importance["feature"]) == set(model.feature_names)
    assert importance["share"].sum() == pytest.approx(1.0, abs=1e-6)


def test_feature_subset_selection_matches_declared_names(fitted):
    data, config, baseline, _ = fitted
    train = build_training_frame(data, config.cutoff, GROUP_ORDER, baseline, config)
    for size in (1, 4, len(GROUP_ORDER)):
        columns = feature_names(GROUP_ORDER[:size])
        assert set(columns).issubset(train.X.columns)


def test_materialised_training_frame_round_trip(fitted, tmp_path):
    data, config, baseline, _ = fitted
    train = build_training_frame(data, config.cutoff, GROUP_ORDER, baseline, config)
    manifest = materialize_training_frame(
        train,
        tmp_path / "trainset",
        GROUP_ORDER,
        config,
        input_digest="fixture-input",
    )
    restored = load_materialized_training_frame(tmp_path / "trainset")

    assert manifest["input_digest"] == "fixture-input"
    assert manifest["rows"] == len(train)
    pd.testing.assert_frame_equal(train.X, restored.X)
    pd.testing.assert_frame_equal(train.meta, restored.meta)
    np.testing.assert_array_equal(train.y, restored.y)


def test_materialised_training_frame_rejects_modified_data(fitted, tmp_path):
    data, config, baseline, _ = fitted
    train = build_training_frame(data, config.cutoff, GROUP_ORDER, baseline, config)
    directory = tmp_path / "trainset"
    materialize_training_frame(
        train, directory, GROUP_ORDER, config, input_digest="fixture-input"
    )
    with (directory / "target.parquet").open("ab") as handle:
        handle.write(b"corrupt")

    with pytest.raises(ValueError, match="checksum mismatch"):
        load_materialized_training_frame(directory)


# --------------------------------------------------------------- ensemble
def test_ensemble_blend_is_deterministic_and_bounded():
    left = np.array([10.0, 20.0, 30.0])
    right = np.array([20.0, 20.0, 60.0])
    for alpha in (0.0, 0.35, 1.0):
        blended = alpha * left + (1 - alpha) * right
        np.testing.assert_allclose(blended, alpha * left + (1 - alpha) * right)
        assert (blended >= np.minimum(left, right) - 1e-9).all()
        assert (blended <= np.maximum(left, right) + 1e-9).all()


def test_run_experiment_imposes_the_observed_floor(pol4):
    data, config = pol4
    contexts = [build_context(data, pd.Timestamp("2024-09-01"), config)]

    def too_low(context):
        return np.zeros(len(context.frame))

    result = run_experiment("floored", "test", contexts, too_low)
    np.testing.assert_allclose(
        result.predictions["prediction"].to_numpy(),
        result.predictions["observed"].to_numpy(),
    )


def test_run_experiment_reproduces_the_baseline_score(pol4):
    data, config = pol4
    contexts = [build_context(data, pd.Timestamp("2024-09-01"), config)]
    result = run_experiment("baseline", "test", contexts, baseline_predictor)
    from ml.pol4.backtest import run_fold

    assert result.pooled["wape"] == pytest.approx(
        run_fold(data, pd.Timestamp("2024-09-01"), config).overall["wape"], abs=1e-9
    )


# -------------------------------------------------------------- stability
def test_stability_snapshots_cover_every_horizon(pol4):
    data, config = pol4
    baseline = PickupBaseline.fit(data, pd.Timestamp("2024-08-15"), config)
    report = stability_module.analyse(
        data,
        pd.Timestamp("2024-09-15"),
        6,
        stability_module.baseline_snapshot(baseline),
        config,
    )
    assert set(report.snapshots["horizon"]) == set(stability_module.SNAPSHOT_HORIZONS)
    assert len(report.snapshots) == len(data.city_codes) * 6 * len(
        stability_module.SNAPSHOT_HORIZONS
    )


def test_stability_snapshot_observations_grow_as_checkin_approaches(pol4):
    data, config = pol4
    baseline = PickupBaseline.fit(data, pd.Timestamp("2024-08-15"), config)
    report = stability_module.analyse(
        data,
        pd.Timestamp("2024-09-15"),
        6,
        stability_module.baseline_snapshot(baseline),
        config,
    )
    totals = report.snapshots.groupby("horizon")["observed"].sum()
    # Fewer days to go means more of the demand has already been searched.
    assert totals.loc[1] >= totals.loc[7] >= totals.loc[30]


def test_stability_revisions_are_between_consecutive_snapshots(pol4):
    data, config = pol4
    baseline = PickupBaseline.fit(data, pd.Timestamp("2024-08-15"), config)
    report = stability_module.analyse(
        data,
        pd.Timestamp("2024-09-15"),
        6,
        stability_module.baseline_snapshot(baseline),
        config,
    )
    ladder = list(stability_module.SNAPSHOT_HORIZONS)
    expected = {f"{a}->{b}" for a, b in zip(ladder, ladder[1:])}
    assert set(report.revisions["step"]) == expected
    assert 0.0 <= report.summary["stability_score"] <= 1.0
    assert 0.0 <= report.summary["convergence_rate"] <= 1.0


def test_stability_never_reads_past_its_anchor(pol4):
    """A snapshot at horizon h must not move when later searches are corrupted."""
    data, config = pol4
    target_start = pd.Timestamp("2024-09-15")
    baseline = PickupBaseline.fit(data, pd.Timestamp("2024-08-15"), config)
    predict = stability_module.baseline_snapshot(baseline)
    before = stability_module.analyse(data, target_start, 6, predict, config)

    poisoned = data.search.copy()
    # Everything logged in the last 7 days before each check-in.
    poisoned.loc[poisoned["days_to_checkin"] < 7, SEARCHES] = 10**6
    after = stability_module.analyse(
        dataclasses.replace(data, search=poisoned), target_start, 6, predict, config
    )
    for horizon in (30, 21, 14, 7):
        left = before.snapshots.query("horizon == @horizon")["prediction"].to_numpy()
        right = after.snapshots.query("horizon == @horizon")["prediction"].to_numpy()
        np.testing.assert_allclose(left, right, err_msg=f"horizon {horizon} leaked")
