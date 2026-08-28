"""Optional model adapters: Chronos-2 and NeuralForecast.

These tests only run when the optional dependencies are installed. When they
are, they exercise the adapters against the *real* libraries rather than mocks -
which is how the `context=` vs `inputs=` signature mismatch and the quantile
cache-poisoning bug were found in the first place.

The Chronos test builds a tiny randomly-initialised model locally instead of
downloading pretrained weights. Accuracy is meaningless with random weights;
what is being verified is the integration: signature, output shape, row
alignment, quantile ordering and graceful degradation.
"""
from __future__ import annotations

import numpy as np
import pytest

from ml.models.base import FitContext, ModelUnavailable, PredictContext, enforce_monotone
from ml.models.chronos_model import chronos_available, detect_device
from ml.models.neural_model import neuralforecast_available

CHRONOS_DEPS = pytest.importorskip  # readability alias


def _has(module: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(module) is not None


torch_missing = pytest.mark.skipif(not _has("torch"), reason="PyTorch is not installed")
chronos_missing = pytest.mark.skipif(
    not (_has("torch") and _has("chronos")), reason="chronos-forecasting is not installed"
)
neural_missing = pytest.mark.skipif(
    not (_has("torch") and _has("neuralforecast")), reason="neuralforecast is not installed"
)


@pytest.fixture(scope="module")
def contexts(engine):
    horizon = engine.config.max_horizon
    origin = engine.tensor.origin_index - horizon
    train = engine.build_training(train_end_idx=origin, samples_per_target=1, seed=4)
    fit = FitContext(
        engine=engine, train_end_idx=origin, frame=train, seed=4, quantiles=(0.1, 0.5, 0.9)
    )
    predict = PredictContext(
        engine=engine,
        origin_idx=origin,
        horizon=horizon,
        frame=engine.build_inference(origin, horizon),
    )
    return fit, predict


@pytest.fixture(scope="module")
def tiny_chronos(tmp_path_factory):
    """A randomly-initialised Chronos-Bolt saved to disk.

    Pretrained weights need network access; the adapter's contract with the
    library does not.
    """
    pytest.importorskip("torch")
    pytest.importorskip("chronos")
    from chronos.chronos_bolt import ChronosBoltConfig, ChronosBoltModelForForecasting
    from transformers import T5Config

    config = T5Config(
        d_model=64, d_kv=16, d_ff=128, num_layers=1, num_decoder_layers=1,
        num_heads=2, vocab_size=32, is_encoder_decoder=False, use_cache=False,
        decoder_start_token_id=0, pad_token_id=0, eos_token_id=1,
    )
    config.chronos_config = ChronosBoltConfig(
        context_length=128, prediction_length=16,
        input_patch_size=8, input_patch_stride=8,
        quantiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        use_reg_token=True,
    ).__dict__
    config.chronos_pipeline_class = "ChronosBoltPipeline"

    path = tmp_path_factory.mktemp("chronos") / "tiny"
    ChronosBoltModelForForecasting(config).save_pretrained(path)
    return str(path)


# ------------------------------------------------------------------ shared
def test_monotone_repair_sorts_crossed_quantiles():
    crossed = {0.1: np.array([5.0, 9.0]), 0.5: np.array([4.0, 8.0]), 0.9: np.array([3.0, 7.0])}
    fixed = enforce_monotone(crossed)

    assert (fixed[0.1] <= fixed[0.5]).all()
    assert (fixed[0.5] <= fixed[0.9]).all()


def test_monotone_repair_leaves_ordered_quantiles_alone():
    ordered = {0.1: np.array([1.0]), 0.5: np.array([2.0]), 0.9: np.array([3.0])}
    fixed = enforce_monotone(ordered)
    for level, values in ordered.items():
        np.testing.assert_allclose(fixed[level], values)


@torch_missing
def test_device_detection_reports_something_usable():
    assert detect_device() in {"cpu", "cuda", "mps"}


# ----------------------------------------------------------------- chronos
@chronos_missing
def test_chronos_adapter_integrates_with_the_real_library(monkeypatch, contexts, tiny_chronos):
    """Signature, shape, alignment and ordering - the integration contract."""
    monkeypatch.setenv("ENABLE_CHRONOS", "true")
    from ml.models.chronos_model import ChronosModel

    fit, predict = contexts
    model = ChronosModel(model_id=tiny_chronos, context_length=128).fit(fit)

    point = model.predict(predict)
    assert len(point) == len(predict.frame.meta), "predictions must align with the frame"
    assert np.isfinite(point).all()
    assert (point >= 0).all(), "a non-negative target must not receive negative forecasts"


@chronos_missing
def test_chronos_point_call_does_not_poison_the_quantile_cache(
    monkeypatch, contexts, tiny_chronos
):
    """Regression: the cache key must include the requested quantile levels.

    Without it a point forecast (median only) fills the cache, and the later
    interval request silently comes back with nothing but P50 - so the model
    appears to work while never supplying a native interval.
    """
    monkeypatch.setenv("ENABLE_CHRONOS", "true")
    from ml.models.chronos_model import ChronosModel

    fit, predict = contexts
    model = ChronosModel(model_id=tiny_chronos, context_length=128).fit(fit)

    model.predict(predict)  # median only - would poison a level-blind cache
    quantiles = model.predict_quantiles(predict, (0.1, 0.5, 0.9))

    assert set(quantiles) == {0.1, 0.5, 0.9}, f"got {sorted(quantiles)}"
    assert (quantiles[0.1] <= quantiles[0.5]).all()
    assert (quantiles[0.5] <= quantiles[0.9]).all()


@chronos_missing
def test_chronos_normalises_both_output_shapes():
    """Bolt returns one stacked tensor; Chronos-2 returns a list of tensors."""
    from ml.models.chronos_model import _to_array

    horizon, n_quantiles, batch = 5, 3, 4
    stacked = np.random.rand(batch, horizon, n_quantiles).astype(np.float32)
    listed = [np.random.rand(1, horizon, n_quantiles).astype(np.float32) for _ in range(batch)]

    assert _to_array(stacked, horizon, n_quantiles).shape == (batch, horizon, n_quantiles)
    assert _to_array(listed, horizon, n_quantiles).shape == (batch, horizon, n_quantiles)


@chronos_missing
def test_chronos_transposes_a_quantile_major_output():
    from ml.models.chronos_model import _to_array

    horizon, n_quantiles, batch = 6, 3, 2
    quantile_major = np.random.rand(batch, n_quantiles, horizon).astype(np.float32)
    assert _to_array(quantile_major, horizon, n_quantiles).shape == (batch, horizon, n_quantiles)


def test_chronos_is_unavailable_without_the_feature_flag(monkeypatch):
    monkeypatch.delenv("ENABLE_CHRONOS", raising=False)
    available, reason = chronos_available()
    assert available is False
    assert "ENABLE_CHRONOS" in reason


def test_chronos_raises_model_unavailable_rather_than_crashing(monkeypatch):
    """A missing model must degrade, never take the run down."""
    monkeypatch.setenv("ENABLE_CHRONOS", "true")
    from ml.models.chronos_model import ChronosModel

    model = ChronosModel(model_id="/nonexistent/path/to/nowhere")
    with pytest.raises(ModelUnavailable):
        model.fit(None)  # type: ignore[arg-type]


# ----------------------------------------------------------------- neural
@neural_missing
def test_nhits_trains_predicts_and_produces_ordered_quantiles(monkeypatch, contexts):
    monkeypatch.setenv("ENABLE_NEURALFORECAST", "true")
    from ml.models.neural_model import NHITSModel

    fit, predict = contexts
    model = NHITSModel(max_steps=30).fit(fit)

    point = model.predict(predict)
    assert len(point) == len(predict.frame.meta)
    assert np.isfinite(point).all()
    assert (point >= 0).all()

    quantiles = model.predict_quantiles(predict, (0.1, 0.5, 0.9))
    assert {0.1, 0.9}.issubset(quantiles)
    assert (quantiles[0.1] <= quantiles[0.9]).all(), "quantile crossing"


@neural_missing
def test_nhits_beats_a_naive_forecast(monkeypatch, contexts):
    """A deep model that cannot beat 'repeat the last value' is not working."""
    monkeypatch.setenv("ENABLE_NEURALFORECAST", "true")
    from ml.evaluation.metrics import get_metric
    from ml.models.baselines import NaiveModel
    from ml.models.neural_model import NHITSModel

    fit, predict = contexts
    actual = predict.frame.y
    mask = np.isfinite(actual)

    naive = NaiveModel().fit(fit).predict(predict)
    nhits = NHITSModel(max_steps=60).fit(fit).predict(predict)

    wape = get_metric("wape")
    assert wape(actual[mask], nhits[mask]) < wape(actual[mask], naive[mask])


def test_neuralforecast_is_unavailable_without_the_feature_flag(monkeypatch):
    monkeypatch.delenv("ENABLE_NEURALFORECAST", raising=False)
    available, reason = neuralforecast_available()
    assert available is False
    assert "ENABLE_NEURALFORECAST" in reason


@neural_missing
def test_neural_model_reports_unavailable_on_impossible_input(monkeypatch, engine):
    """Too little history must raise ModelUnavailable, not an opaque error."""
    monkeypatch.setenv("ENABLE_NEURALFORECAST", "true")
    from ml.features.engineering import SupervisedFrame
    from ml.models.neural_model import NHITSModel

    empty = SupervisedFrame(
        X=engine.build_inference(engine.tensor.origin_index, 1).X.head(0),
        y=np.array([]),
        meta=engine.build_inference(engine.tensor.origin_index, 1).meta.head(0),
        categorical=[],
    )
    context = FitContext(engine=engine, train_end_idx=2, frame=empty, seed=1)
    with pytest.raises(ModelUnavailable):
        NHITSModel(max_steps=5).fit(context)
