"""Pol 4 competition semantics: two clocks, pickup, and the output contract."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from ml.pol4 import load_pol4
from ml.pol4.backtest import normalised_bias, run_fold
from ml.pol4.baseline import PickupBaseline, ZeroObservationPrior
from ml.pol4.config import (
    N_SUBMISSION_ROWS,
    SUBMISSION_COLUMNS,
    TARGET_END,
    TARGET_START,
)
from ml.pol4.loader import CHECKIN, CITY, DTC, Pol4DataError
from ml.pol4.pickup import PickupCurves
from ml.pol4.submission import (
    SubmissionError,
    build_grid,
    build_named_submission,
    build_submission,
    validate_submission,
)
from tests.pol4_fixtures import CUTOFF, TARGET_DAYS, full_truth, write_dataset


@pytest.fixture(scope="module")
def pol4(tmp_path_factory):
    config = write_dataset(tmp_path_factory.mktemp("pol4"))
    return load_pol4(config), config


# --------------------------------------------------------------- two clocks
def test_loader_keeps_both_time_axes(pol4):
    data, _ = pol4
    assert {"log_date", CHECKIN, DTC}.issubset(data.search.columns)
    # The log axis is preserved as events, not collapsed at load time.
    assert data.search.duplicated([CITY, CHECKIN]).any()


def test_days_to_checkin_is_the_gap_between_the_clocks(pol4):
    data, _ = pol4
    expected = (data.search[CHECKIN] - data.search["log_date"]).dt.days
    assert (data.search[DTC] == expected).all()
    assert data.search[DTC].min() >= 0


def test_loader_rejects_a_checkin_before_its_log_date(tmp_path):
    config = write_dataset(tmp_path)
    broken = pd.read_csv(config.search_path)
    broken.loc[0, "checkin"] = "2023-01-01"
    broken.to_csv(config.search_path, index=False)
    with pytest.raises(Pol4DataError, match="checkin < log_date"):
        load_pol4(config)


def test_loader_rejects_duplicate_events(tmp_path):
    config = write_dataset(tmp_path)
    rows = pd.read_csv(config.search_path)
    pd.concat([rows, rows.head(1)]).to_csv(config.search_path, index=False)
    with pytest.raises(Pol4DataError, match="duplicate"):
        load_pol4(config)


def test_loader_rejects_a_city_missing_from_cities_csv(tmp_path):
    config = write_dataset(tmp_path)
    cities = pd.read_csv(config.cities_path)
    cities.head(len(cities) - 1).to_csv(config.cities_path, index=False)
    with pytest.raises(Pol4DataError, match="not in cities.csv"):
        load_pol4(config)


# ---------------------------------------------------- historical final demand
def test_final_demand_sums_every_log_date_for_the_pair(pol4):
    data, config = pol4
    final = data.final_demand(CUTOFF - pd.Timedelta(days=20), CUTOFF)
    expected = full_truth().rename(columns={"final": "expected"})
    merged = final.merge(expected, on=[CITY, CHECKIN], how="left")
    assert (merged["final"] == merged["expected"]).all()


def test_final_demand_is_deterministic(pol4):
    data, _ = pol4
    first = data.final_demand(CUTOFF - pd.Timedelta(days=30), CUTOFF)
    second = data.final_demand(CUTOFF - pd.Timedelta(days=30), CUTOFF)
    pd.testing.assert_frame_equal(first, second)


# --------------------------------------------------------- cutoff enforcement
def test_observed_at_ignores_log_dates_after_the_cutoff(pol4):
    data, config = pol4
    dates = config.target_dates()
    observed = data.observed_at(config.cutoff, dates.min(), dates.max())
    truth = full_truth()
    merged = observed.merge(truth, on=[CITY, CHECKIN], how="left")
    # Partial by construction: some demand still arrives after the cutoff.
    assert (merged["observed"] <= merged["final"]).all()
    assert merged["observed"].sum() < merged["final"].sum()


def test_history_before_only_returns_completed_checkins(pol4):
    data, config = pol4
    history = data.history_before(config.cutoff)
    assert history[CHECKIN].max() <= config.cutoff
    assert history["log_date"].max() <= config.cutoff


def test_predict_refuses_a_target_at_or_before_the_cutoff(pol4):
    data, config = pol4
    model = PickupBaseline.fit(data, config.cutoff, config)
    grid = pd.DataFrame(
        {CITY: [1], CHECKIN: [config.cutoff], "observed": [10.0]}
    )
    with pytest.raises(ValueError, match="strictly after the cutoff"):
        model.predict(grid)


# ------------------------------------------------------------- pickup curves
def test_completion_fraction_is_one_at_checkin_and_decreasing(pol4):
    data, config = pol4
    curves = PickupCurves.fit(data, config.cutoff, config)
    assert curves.global_curve[0] == pytest.approx(1.0)
    assert np.all(np.diff(curves.global_curve) <= 1e-12)


def test_completion_fraction_matches_a_hand_computed_share(pol4):
    data, config = pol4
    curves = PickupCurves.fit(data, config.cutoff, config)
    history = data.history_before(config.cutoff)
    for horizon in (1, 3, 7):
        expected = (
            history.loc[history[DTC] >= horizon, "search_count"].sum()
            / history["search_count"].sum()
        )
        assert curves.global_curve[horizon] == pytest.approx(expected, rel=1e-9)


def test_hierarchical_fallback_city_then_province_then_global(pol4):
    data, config = pol4
    curves = PickupCurves.fit(data, config.cutoff, config)
    assert set(curves.source([1, 4])) == {"city"}
    assert set(curves.source([2, 3, 5, 6])) == {"province"}
    # An unknown city has neither a curve nor a province and falls all the way
    # through rather than failing.
    assert curves.source([9999])[0] == "global"
    assert curves.fraction([9999], [7])[0] == pytest.approx(curves.global_curve[7])


def test_a_thin_province_falls_through_to_the_global_curve(pol4):
    data, config = pol4
    strict = replace(config, min_city_support=10**12, min_province_support=10**12)
    curves = PickupCurves.fit(data, config.cutoff, strict)
    assert set(curves.source([1, 2, 3, 4, 5, 6])) == {"global"}


def test_fraction_never_falls_below_the_floor(pol4):
    data, config = pol4
    curves = PickupCurves.fit(data, config.cutoff, config)
    fractions = curves.fraction([1] * 40, list(range(40)))
    assert fractions.min() >= config.min_fraction
    assert fractions.max() <= 1.0


# ----------------------------------------------------------------- baseline
def test_prediction_never_falls_below_what_is_already_observed(pol4):
    data, config = pol4
    model = PickupBaseline.fit(data, config.cutoff, config)
    grid = build_grid(data, config.cutoff, config.target_dates())
    predictions = model.predict(grid)
    assert (predictions["predicted_demand"] >= predictions["observed"] - 1e-9).all()


def test_predictions_are_finite_and_non_negative(pol4):
    data, config = pol4
    model = PickupBaseline.fit(data, config.cutoff, config)
    predictions = model.predict(build_grid(data, config.cutoff, config.target_dates()))
    values = predictions["predicted_demand"].to_numpy()
    assert np.isfinite(values).all()
    assert (values >= 0).all()


def test_zero_observation_pairs_are_not_predicted_as_zero(pol4):
    data, config = pol4
    model = PickupBaseline.fit(data, config.cutoff, config)
    predictions = model.predict(build_grid(data, config.cutoff, config.target_dates()))
    unobserved = predictions[predictions["observed"] <= 0]
    assert len(unobserved), "the fixture should contain unobserved pairs"
    # "Zero so far" is not "zero in the end" - the prior has to carry them.
    assert unobserved["predicted_demand"].sum() > 0


def test_zero_policy_zero_predicts_nothing_for_unobserved_pairs(pol4):
    data, config = pol4
    model = PickupBaseline.fit(data, config.cutoff, replace(config, zero_observation_policy="zero"))
    predictions = model.predict(build_grid(data, config.cutoff, config.target_dates()))
    unobserved = predictions[predictions["observed"] <= 0]
    assert (unobserved["predicted_demand"] == 0).all()


def test_unknown_zero_policy_is_rejected(pol4):
    data, config = pol4
    with pytest.raises(ValueError, match="zero_observation_policy"):
        PickupBaseline.fit(data, config.cutoff, replace(config, zero_observation_policy="guess"))


def test_zero_prior_grows_with_the_horizon(pol4):
    data, config = pol4
    prior = ZeroObservationPrior.fit(data, config.cutoff, config, horizons=TARGET_DAYS)
    near = prior.level([1], [1], {1: 10})[0]
    far = prior.level([1], [TARGET_DAYS], {1: 10})[0]
    # Nothing observed the day before check-in is far worse news than nothing
    # observed ten days out, when little was expected yet anyway.
    assert near <= far


def test_prediction_is_deterministic(pol4):
    data, config = pol4
    grid = build_grid(data, config.cutoff, config.target_dates())
    first = PickupBaseline.fit(data, config.cutoff, config).predict(grid)
    second = PickupBaseline.fit(data, config.cutoff, config).predict(grid)
    pd.testing.assert_frame_equal(first, second)


# --------------------------------------------------------------------- grid
def test_grid_covers_every_city_and_every_target_date(pol4):
    data, config = pol4
    grid = build_grid(data, config.cutoff, config.target_dates())
    assert len(grid) == len(data.city_codes) * config.target_days
    assert grid[CITY].nunique() == len(data.city_codes)
    assert grid[CHECKIN].nunique() == config.target_days
    assert not grid.duplicated([CITY, CHECKIN]).any()
    assert grid["observed"].notna().all()


def test_grid_observed_matches_the_evaluation_file(pol4):
    data, config = pol4
    grid = build_grid(data, config.cutoff, config.target_dates())
    assert grid["observed"].sum() == pytest.approx(data.evaluation["search_count"].sum())


# --------------------------------------------------------------- submission
def _valid_submission(cities, dates):
    index = pd.MultiIndex.from_product([cities, dates]).to_frame(index=False)
    return pd.DataFrame(
        {
            "cluster_code": index[0].to_numpy(),
            "checkin": pd.DatetimeIndex(index[1]).strftime("%Y-%m-%d"),
            "predicted_demand": np.arange(len(index), dtype=np.int64),
        }
    )


def test_submission_has_the_exact_competition_columns(pol4):
    data, config = pol4
    model = PickupBaseline.fit(data, config.cutoff, config)
    predictions = model.predict(build_grid(data, config.cutoff, config.target_dates()))
    submission = build_submission(predictions)
    assert tuple(submission.columns) == SUBMISSION_COLUMNS
    assert len(submission) == len(data.city_codes) * config.target_days
    report = validate_submission(submission, data.city_codes, config)
    assert report.valid


def test_cluster_code_is_the_city_code_when_nothing_is_clustered(pol4):
    data, config = pol4
    model = PickupBaseline.fit(data, config.cutoff, config)
    predictions = model.predict(build_grid(data, config.cutoff, config.target_dates()))
    submission = build_submission(predictions)
    assert sorted(submission["cluster_code"].unique()) == sorted(data.city_codes.tolist())


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda f: f.iloc[:-1], "rows, expected"),
        (lambda f: pd.concat([f, f.iloc[[0]]]), "duplicate"),
        (lambda f: f.assign(predicted_demand=f["predicted_demand"].astype(float) * -1), "negative"),
        (lambda f: f.assign(predicted_demand=np.nan), "NaN"),
        (lambda f: f.assign(predicted_demand=np.inf), "infinite"),
        (lambda f: f.rename(columns={"predicted_demand": "demand"}), "columns are"),
        (lambda f: f.assign(checkin="1404/09/01"), "not Gregorian"),
        (lambda f: f.assign(checkin="2030-01-01"), "outside"),
        (lambda f: f.assign(cluster_code=999), "missing from the output"),
    ],
)
def test_validator_rejects_a_broken_submission(pol4, mutate, message):
    data, config = pol4
    frame = _valid_submission(data.city_codes, config.target_dates())
    with pytest.raises(SubmissionError, match=message):
        validate_submission(mutate(frame), data.city_codes, config)


def test_validator_accepts_a_correct_submission(pol4):
    data, config = pol4
    frame = _valid_submission(data.city_codes, config.target_dates())
    report = validate_submission(frame, data.city_codes, config)
    assert report.valid and report.problems == []


# ---------------------------------------------------------------- backtest
def test_normalised_bias_matches_the_brief_definition():
    actual = np.array([10.0, 20.0, 30.0])
    predicted = np.array([12.0, 20.0, 34.0])
    assert normalised_bias(actual, predicted) == pytest.approx(6.0 / 60.0)


def test_backtest_fold_scores_the_whole_grid(pol4):
    data, config = pol4
    fold = run_fold(data, pd.Timestamp("2024-09-01"), config)
    assert fold.overall["n"] == len(data.city_codes) * config.target_days
    assert 0 <= fold.overall["wape"] < 2
    # The pickup projection must beat the raw observation it is built from.
    assert fold.overall["wape"] < fold.reference["observed_so_far"]["wape"]


def test_competition_constants_match_the_brief():
    assert TARGET_START == pd.Timestamp("2025-11-22")
    assert TARGET_END == pd.Timestamp("2025-12-21")
    assert N_SUBMISSION_ROWS == 321 * 30 == 9630
    assert SUBMISSION_COLUMNS == ("cluster_code", "checkin", "predicted_demand")


# ------------------------------------------------------------- city names
def test_names_are_optional(tmp_path):
    """A clone with only the three official files must still run."""
    config = write_dataset(tmp_path)
    config.city_names_path.unlink(missing_ok=True)
    data = load_pol4(config)
    assert data.has_names is False
    # Falls back to the code as a string rather than failing or inventing one.
    assert data.name([1])[0] == "1"


def test_names_are_attached_when_the_mapping_is_present(tmp_path):
    config = write_dataset(tmp_path)
    pd.DataFrame(
        {
            "city": [f"city_{c}" for c in [1, 2, 3, 4, 5, 6]],
            "city_code": [1, 2, 3, 4, 5, 6],
            "province": ["north"] * 3 + ["south"] * 3,
            "province_code": [10, 10, 10, 20, 20, 20],
        }
    ).to_csv(config.city_names_path, index=False)

    data = load_pol4(config)
    assert data.has_names is True
    assert list(data.name([1, 4])) == ["city_1", "city_4"]
    assert data.province_name_of[1] == "north"


def test_a_mismatched_province_in_the_mapping_is_rejected(tmp_path):
    config = write_dataset(tmp_path)
    pd.DataFrame(
        {
            "city": [f"city_{c}" for c in [1, 2, 3, 4, 5, 6]],
            "city_code": [1, 2, 3, 4, 5, 6],
            "province": ["north"] * 6,
            "province_code": [10] * 6,  # cities 4-6 are really province 20
        }
    ).to_csv(config.city_names_path, index=False)
    with pytest.raises(Pol4DataError, match="different province_code"):
        load_pol4(config)


def test_an_incomplete_mapping_is_rejected(tmp_path):
    config = write_dataset(tmp_path)
    pd.DataFrame(
        {
            "city": ["city_1"],
            "city_code": [1],
            "province": ["north"],
            "province_code": [10],
        }
    ).to_csv(config.city_names_path, index=False)
    with pytest.raises(Pol4DataError, match="no name"):
        load_pol4(config)


def test_label_inserts_names_next_to_the_code(tmp_path):
    config = write_dataset(tmp_path)
    pd.DataFrame(
        {
            "city": [f"city_{c}" for c in [1, 2, 3, 4, 5, 6]],
            "city_code": [1, 2, 3, 4, 5, 6],
            "province": ["north"] * 3 + ["south"] * 3,
            "province_code": [10, 10, 10, 20, 20, 20],
        }
    ).to_csv(config.city_names_path, index=False)
    data = load_pol4(config)
    labelled = data.label(pd.DataFrame({CITY: [1, 6], "x": [0, 0]}))
    assert list(labelled.columns) == [CITY, "city", "province", "x"]
    assert list(labelled["city"]) == ["city_1", "city_6"]


def test_results_csv_still_uses_the_numeric_cluster_code(pol4):
    """Names are for humans; the scored file keeps the codes the brief asks for."""
    data, config = pol4
    model = PickupBaseline.fit(data, config.cutoff, config)
    predictions = model.predict(build_grid(data, config.cutoff, config.target_dates()))
    submission = build_submission(predictions)
    assert tuple(submission.columns) == SUBMISSION_COLUMNS
    assert pd.api.types.is_integer_dtype(submission["cluster_code"])


def test_named_submission_matches_the_scored_one_row_for_row(pol4):
    data, config = pol4
    model = PickupBaseline.fit(data, config.cutoff, config)
    predictions = model.predict(build_grid(data, config.cutoff, config.target_dates()))
    scored = build_submission(predictions)
    named = build_named_submission(predictions, data)

    assert len(named) == len(scored)
    assert named["predicted_demand"].sum() == scored["predicted_demand"].sum()
    assert (named["predicted_remaining"] >= 0).all()
    assert (named["predicted_demand"] >= named["observed_so_far"]).all()
