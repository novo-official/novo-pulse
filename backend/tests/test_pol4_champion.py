"""The champion, end to end: it must keep every Phase 1 guarantee."""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from ml.pol4 import load_pol4
from ml.pol4.champion import ChampionSpec, FittedChampion
from ml.pol4.loader import CHECKIN, CITY, SEARCHES
from ml.pol4.submission import build_submission, validate_submission
from tests.pol4_fixtures import write_dataset

SMALL = ChampionSpec(params={"n_estimators": 60, "num_leaves": 15})


@pytest.fixture(scope="module")
def pol4(tmp_path_factory):
    config = write_dataset(tmp_path_factory.mktemp("pol4_champ"))
    return load_pol4(config), config


@pytest.fixture(scope="module")
def fitted(pol4):
    data, config = pol4
    return FittedChampion.fit(data, config.cutoff, SMALL, config), data, config


def test_champion_predicts_the_whole_grid(fitted):
    champion, data, config = fitted
    predictions = champion.predict(config.target_dates(), data)
    assert len(predictions) == len(data.city_codes) * config.target_days
    assert not predictions.duplicated([CITY, CHECKIN]).any()


def test_champion_never_predicts_below_observed(fitted):
    champion, data, config = fitted
    predictions = champion.predict(config.target_dates(), data)
    assert (predictions["predicted_demand"] >= predictions["observed"] - 1e-9).all()
    assert (predictions["predicted_remaining"] >= -1e-9).all()


def test_champion_output_is_finite_and_non_negative(fitted):
    champion, data, config = fitted
    values = champion.predict(config.target_dates(), data)["predicted_demand"].to_numpy()
    assert np.isfinite(values).all()
    assert (values >= 0).all()


def test_champion_is_deterministic(pol4):
    data, config = pol4
    first = FittedChampion.fit(data, config.cutoff, SMALL, config).predict(
        config.target_dates(), data
    )
    second = FittedChampion.fit(data, config.cutoff, SMALL, config).predict(
        config.target_dates(), data
    )
    pd.testing.assert_frame_equal(first, second)


def test_champion_submission_passes_the_validator(fitted):
    champion, data, config = fitted
    submission = build_submission(champion.predict(config.target_dates(), data))
    report = validate_submission(submission, data.city_codes, config)
    assert report.valid, report.problems


def test_champion_native_bundle_round_trip_is_prediction_identical(fitted, tmp_path):
    champion, data, config = fitted
    before = champion.predict(config.target_dates(), data)
    manifest = champion.save(tmp_path / "bundle", input_digest="fixture-input")
    restored = FittedChampion.load(tmp_path / "bundle")
    after = restored.predict(config.target_dates(), data)

    assert manifest["input_digest"] == "fixture-input"
    assert {path.name for path in (tmp_path / "bundle").glob("lightgbm_*.txt")} == {
        "lightgbm_h1_14.txt",
        "lightgbm_h15_30.txt",
    }
    pd.testing.assert_frame_equal(before, after, check_exact=False, rtol=1e-12, atol=1e-12)


def test_champion_bundle_rejects_a_modified_model(fitted, tmp_path):
    champion, _, _ = fitted
    directory = tmp_path / "bundle"
    champion.save(directory, input_digest="fixture-input")
    model_path = next(directory.glob("lightgbm_*.txt"))
    model_path.write_text(model_path.read_text() + "\n# corrupted\n")

    with pytest.raises(ValueError, match="checksum mismatch"):
        FittedChampion.load(directory)


def test_champion_cannot_see_searches_logged_after_the_cutoff(pol4):
    """End-to-end leakage test for the model, not just the baseline.

    Corrupt every post-cutoff search - including the ones for the target nights
    themselves - refit, and require identical predictions.
    """
    data, config = pol4
    before = FittedChampion.fit(data, config.cutoff, SMALL, config).predict(
        config.target_dates(), data
    )

    def poison(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        out.loc[out["log_date"] > config.cutoff, SEARCHES] = 10**6
        return out

    poisoned = dataclasses.replace(
        data, search=poison(data.search), evaluation=poison(data.evaluation)
    )
    after = FittedChampion.fit(poisoned, config.cutoff, SMALL, config).predict(
        config.target_dates(), poisoned
    )
    pd.testing.assert_frame_equal(before, after)


def test_champion_zero_observation_pairs_still_get_demand(fitted):
    champion, data, config = fitted
    predictions = champion.predict(config.target_dates(), data)
    unobserved = predictions[predictions["observed"] <= 0]
    assert len(unobserved), "the fixture should leave some pairs unobserved"
    assert unobserved["predicted_demand"].sum() > 0


def test_champion_spec_records_what_was_selected():
    spec = ChampionSpec()
    described = spec.describe()
    assert described["model"] == "lightgbm"
    # log1p and the blend weight are experiment outcomes, not preferences.
    assert described["log1p_target"] is True
    assert described["blend_weight_on_model"] == 1.0
    assert described["n_features"] == 66


def test_importance_is_reported_for_every_feature(fitted):
    champion, _, _ = fitted
    importance = champion.importance()
    assert len(importance) == len(champion.model.feature_names)
    assert (importance["importance"] >= 0).all()


# ------------------------------------------------------------ horizon bands
def test_horizon_bands_cover_every_target_horizon(pol4):
    data, config = pol4
    spec = dataclasses.replace(
        SMALL, bands=((1, 5), (6, config.target_days))
    )
    champion = FittedChampion.fit(data, config.cutoff, spec, config)
    predictions = champion.predict(config.target_dates(), data)
    assert len(champion.models) == 2
    assert predictions["predicted_demand"].notna().all()


def test_a_gap_in_the_bands_is_an_error_not_a_silent_nan(pol4):
    data, config = pol4
    spec = dataclasses.replace(SMALL, bands=((1, 3),))
    champion = FittedChampion.fit(data, config.cutoff, spec, config)
    with pytest.raises(RuntimeError, match="no horizon band covers"):
        champion.predict(config.target_dates(), data)


def test_banded_importance_is_pooled_across_bands(pol4):
    data, config = pol4
    spec = dataclasses.replace(SMALL, bands=((1, 5), (6, config.target_days)))
    champion = FittedChampion.fit(data, config.cutoff, spec, config)
    importance = champion.importance()
    assert len(importance) == len(champion.model.feature_names)
    assert importance["share"].sum() == pytest.approx(1.0, abs=1e-6)


def test_banded_champion_keeps_the_observed_floor(pol4):
    data, config = pol4
    spec = dataclasses.replace(SMALL, bands=((1, 5), (6, config.target_days)))
    predictions = FittedChampion.fit(data, config.cutoff, spec, config).predict(
        config.target_dates(), data
    )
    assert (predictions["predicted_demand"] >= predictions["observed"] - 1e-9).all()
