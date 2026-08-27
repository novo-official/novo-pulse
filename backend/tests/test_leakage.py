"""Leakage tests.

These are the most important tests in the project. A forecasting model that
sees the future scores brilliantly and is worthless, so the guarantees below
are asserted mechanically rather than trusted.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.features.rolling import rolling_mean_std, rolling_min_max, shifted
from ml.features.tensor import build_tensor


def test_rolling_mean_is_causal():
    """A rolling statistic at column i may only use columns <= i."""
    values = np.arange(20, dtype=np.float32).reshape(1, 20)
    mean, _ = rolling_mean_std(values, 5)

    for i in range(20):
        window = values[0, max(0, i - 4) : i + 1]
        assert mean[0, i] == pytest.approx(window.mean(), rel=1e-5), f"column {i}"


def test_rolling_mean_ignores_future_changes():
    """Mutating the future must not change any past rolling value."""
    original = np.arange(40, dtype=np.float32).reshape(1, 40)
    mean_before, std_before = rolling_mean_std(original, 7)
    min_before, max_before = rolling_min_max(original, 7)

    mutated = original.copy()
    mutated[0, 25:] = 9_999.0
    mean_after, std_after = rolling_mean_std(mutated, 7)
    min_after, max_after = rolling_min_max(mutated, 7)

    # Columns strictly before the mutation must be untouched.
    np.testing.assert_allclose(mean_before[0, :25], mean_after[0, :25])
    np.testing.assert_allclose(std_before[0, :25], std_after[0, :25])
    np.testing.assert_allclose(min_before[0, :25], min_after[0, :25])
    np.testing.assert_allclose(max_before[0, :25], max_after[0, :25])


def test_shift_never_reads_forward():
    values = np.arange(10, dtype=np.float32).reshape(1, 10)
    lagged = shifted(values, 3)
    assert np.isnan(lagged[0, :3]).all()
    np.testing.assert_allclose(lagged[0, 3:], values[0, :7])


def test_supervised_features_ignore_post_origin_target(engine):
    """The decisive test.

    Build features for a fixed (entity, origin, horizon), then overwrite every
    target value strictly after the origin with garbage and rebuild. Identical
    features prove the model can only be reading history.
    """
    tensor = engine.tensor
    origin = tensor.n_observed - 30
    entities = np.array([0, 1, 2])
    horizons = np.array([1, 7, 14])
    targets = origin + horizons

    before = engine.assemble(entities, targets, horizons)

    polluted = build_tensor(
        _panel_from(tensor),
        freq=tensor.freq,
        horizon=engine.config.max_horizon,
        future_features=list(tensor.future),
        past_features=list(tensor.past),
        static_features=[c for c in tensor.static.columns],
    )
    polluted.y[:, origin + 1 :] = 1e6
    from ml.features.engineering import FeatureEngine

    after = FeatureEngine(polluted, engine.config).prepare().assemble(entities, targets, horizons)

    shared = [c for c in before.X.columns if c in after.X.columns]
    numeric = [c for c in shared if not isinstance(before.X[c].dtype, pd.CategoricalDtype)]
    pd.testing.assert_frame_equal(
        before.X[numeric].reset_index(drop=True),
        after.X[numeric].reset_index(drop=True),
        check_dtype=False,
        rtol=1e-6,
    )


def test_past_covariates_are_not_read_at_the_target_date(engine):
    """Historical-only covariates must be read at the origin, never at t+h."""
    tensor = engine.tensor
    origin = tensor.n_observed - 40
    entities = np.array([0, 1])
    horizons = np.array([10, 14])
    targets = origin + horizons

    before = engine.assemble(entities, targets, horizons)

    import copy

    from ml.features.engineering import FeatureEngine

    polluted = copy.copy(tensor)
    polluted.past = {k: v.copy() for k, v in tensor.past.items()}
    for matrix in polluted.past.values():
        matrix[:, origin + 1 :] = 1e6

    after = FeatureEngine(polluted, engine.config).prepare().assemble(entities, targets, horizons)
    past_columns = [c for c in before.X.columns if c.startswith("past_")]
    assert past_columns, "the fixture should produce past-covariate features"
    pd.testing.assert_frame_equal(
        before.X[past_columns].reset_index(drop=True),
        after.X[past_columns].reset_index(drop=True),
        check_dtype=False,
        rtol=1e-6,
    )


def test_training_samples_never_target_beyond_the_cutoff(engine):
    """No training row may have a target date after the training cutoff."""
    tensor = engine.tensor
    cutoff = tensor.n_observed - 50
    frame = engine.build_training(train_end_idx=cutoff, samples_per_target=2, seed=1)

    assert frame.meta["target_idx"].max() <= cutoff
    assert (frame.meta["origin_idx"] < frame.meta["target_idx"]).all()
    assert frame.meta["ds"].max() <= tensor.dates[cutoff]


def test_inference_targets_are_strictly_after_the_origin(engine):
    tensor = engine.tensor
    origin = tensor.origin_index
    frame = engine.build_inference(origin, engine.config.max_horizon)

    assert (frame.meta["target_idx"] > origin).all()
    assert (frame.meta["horizon"] >= 1).all()
    assert frame.meta["ds"].min() > tensor.dates[origin]


def _panel_from(tensor) -> pd.DataFrame:
    """Rebuild a long panel frame from a tensor (test helper)."""
    from ml.contract import ENTITY, TARGET, TS

    rows, cols = np.where(tensor.observed[:, : tensor.n_observed])
    frame = pd.DataFrame(
        {
            TS: tensor.dates[cols],
            ENTITY: tensor.entities[rows],
            TARGET: tensor.y[rows, cols],
        }
    )
    for name, matrix in {**tensor.past, **tensor.future}.items():
        frame[name] = matrix[rows, cols]
    static = tensor.static.reset_index()
    return frame.merge(static, on=ENTITY, how="left")
