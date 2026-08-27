"""Model interface, baselines, GBDT and registry behaviour."""
from __future__ import annotations

import numpy as np
import pytest

from ml.models.base import FitContext, PredictContext, postprocess
from ml.models.baselines import (
    HistoricalMeanModel,
    MovingAverageModel,
    NaiveModel,
    SeasonalNaiveModel,
)
from ml.models.gbdt import LightGBMModel
from ml.models.registry import build_models, describe_registry, is_baseline


@pytest.fixture(scope="module")
def contexts(engine):
    origin = engine.tensor.origin_index - engine.config.max_horizon
    train = engine.build_training(train_end_idx=origin, samples_per_target=2, seed=3)
    fit = FitContext(
        engine=engine,
        train_end_idx=origin,
        frame=train,
        seed=3,
        quantiles=(0.1, 0.5, 0.9),
        non_negative=True,
    )
    inference = engine.build_inference(origin, engine.config.max_horizon)
    predict = PredictContext(
        engine=engine, origin_idx=origin, horizon=engine.config.max_horizon, frame=inference
    )
    return fit, predict


@pytest.mark.parametrize(
    "model",
    [NaiveModel(), SeasonalNaiveModel(7), MovingAverageModel(14), HistoricalMeanModel()],
)
def test_baselines_produce_aligned_finite_predictions(model, contexts):
    fit, predict = contexts
    model.fit(fit)
    prediction = model.predict(predict)

    assert len(prediction) == len(predict.frame.meta)
    assert np.isfinite(prediction).all()
    assert (prediction >= 0).all()


def test_seasonal_naive_repeats_the_previous_cycle(engine):
    """ŷ(t+h) must equal y at the same phase one cycle back."""
    tensor = engine.tensor
    origin = tensor.n_observed - 20
    entities = np.array([0, 0, 0])
    horizons = np.array([1, 3, 7])
    frame = engine.assemble(entities, origin + horizons, horizons)

    model = SeasonalNaiveModel(7)
    prediction = model.predict(
        PredictContext(engine=engine, origin_idx=origin, horizon=7, frame=frame)
    )

    for i, horizon in enumerate(horizons):
        target = origin + horizon
        expected = tensor.y[0, target - 7 * int(np.ceil(horizon / 7))]
        assert prediction[i] == pytest.approx(expected, rel=1e-4)


def test_baselines_never_read_past_the_origin(engine, contexts):
    """Corrupting post-origin actuals must not change a baseline forecast."""
    fit, predict = contexts
    model = SeasonalNaiveModel(7).fit(fit)
    before = model.predict(predict)

    original = engine.tensor.y.copy()
    try:
        engine.tensor.y[:, predict.origin_idx + 1 :] = 1e6
        after = model.predict(predict)
        np.testing.assert_allclose(before, after)
    finally:
        engine.tensor.y = original


def test_lightgbm_beats_the_seasonal_baseline(contexts):
    """The learned model must actually earn its place."""
    from ml.evaluation.metrics import get_metric

    fit, predict = contexts
    actual = predict.frame.y
    mask = np.isfinite(actual)

    baseline = SeasonalNaiveModel(7).fit(fit).predict(predict)
    model = LightGBMModel(n_estimators=150, learning_rate=0.1, num_leaves=31).fit(fit)
    learned = model.predict(predict)

    wape = get_metric("wape")
    assert wape(actual[mask], learned[mask]) < wape(actual[mask], baseline[mask])


def test_lightgbm_quantiles_are_ordered_and_bracket_the_point(contexts):
    fit, predict = contexts
    model = LightGBMModel(n_estimators=120, num_leaves=31).fit(fit)

    quantiles = model.predict_quantiles(predict, (0.1, 0.5, 0.9))
    assert set(quantiles) >= {0.1, 0.9}
    assert (quantiles[0.1] <= quantiles[0.9]).all(), "quantile crossing"
    assert (quantiles[0.1] >= 0).all()


def test_lightgbm_reports_normalised_feature_importance(contexts):
    fit, _ = contexts
    model = LightGBMModel(n_estimators=100, num_leaves=31).fit(fit)

    importance = model.get_feature_importance()
    assert importance
    assert sum(importance.values()) == pytest.approx(1.0, abs=1e-3)
    assert list(importance.values()) == sorted(importance.values(), reverse=True)


def test_model_round_trips_through_disk(contexts, tmp_path):
    fit, predict = contexts
    model = LightGBMModel(n_estimators=80, num_leaves=31).fit(fit)
    before = model.predict(predict)

    path = model.save(tmp_path)
    restored = LightGBMModel.load(path)
    np.testing.assert_allclose(before, restored.predict(predict), rtol=1e-6)


def test_postprocess_enforces_target_constraints():
    values = np.array([-3.0, 1.4, np.nan, np.inf])
    result = postprocess(values, non_negative=True, integer=True)

    assert (result >= 0).all()
    assert np.isfinite(result).all()
    assert np.allclose(result, np.round(result))


# ------------------------------------------------------------------ registry
def test_registry_skips_unavailable_models_without_raising():
    models, skipped = build_models(["seasonal_naive_7", "lightgbm", "chronos", "nope"], {}, 7)

    assert "lightgbm" in models
    assert {entry["model"] for entry in skipped} >= {"nope"}
    for entry in skipped:
        assert entry["reason"]


def test_registry_reports_availability():
    described = {entry["name"]: entry for entry in describe_registry()}
    assert described["lightgbm"]["available"] is True
    assert described["chronos"]["optional"] is True


def test_baseline_classification():
    assert is_baseline("seasonal_naive_7")
    assert is_baseline("historical_mean")
    assert not is_baseline("lightgbm")
    assert not is_baseline("ensemble")


def test_optional_model_reports_unavailability_rather_than_crashing():
    from ml.models.chronos_model import ChronosModel, chronos_available
    from ml.models.base import ModelUnavailable

    available, reason = chronos_available()
    if available:
        pytest.skip("Chronos is installed in this environment")
    assert reason
    with pytest.raises(ModelUnavailable):
        ChronosModel().fit(None)  # type: ignore[arg-type]
