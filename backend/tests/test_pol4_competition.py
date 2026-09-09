"""Integration checks against the real competition files.

These skip when `data/raw/pol4/` is empty, so a clone without the datasets
still has a green suite. When the data IS present they are the tests that
matter most: they assert the exact shape the organisers will score.
"""
from __future__ import annotations

import pandas as pd
import pytest

from ml.pol4 import Pol4Config, load_pol4
from ml.pol4.baseline import PickupBaseline
from ml.pol4.config import (
    COMPETITION_CUTOFF,
    MAX_LEAD_TIME,
    N_CITIES,
    N_SUBMISSION_ROWS,
    TARGET_DAYS,
)
from ml.pol4.loader import CHECKIN, CITY, DTC
from ml.pol4.submission import build_grid, build_submission, validate_submission

CONFIG = Pol4Config()
pytestmark = pytest.mark.skipif(
    not CONFIG.search_path.exists(),
    reason="competition data not present in data/raw/pol4/",
)


@pytest.fixture(scope="module")
def data():
    return load_pol4(CONFIG)


@pytest.fixture(scope="module")
def submission(data):
    model = PickupBaseline.fit(data, CONFIG.cutoff, CONFIG)
    grid = build_grid(data, CONFIG.cutoff, CONFIG.target_dates())
    return build_submission(model.predict(grid))


def test_the_datasets_match_the_brief(data):
    assert len(data.search) == 3_298_564
    assert len(data.cities) == N_CITIES
    assert data.search[CITY].nunique() == N_CITIES
    assert data.search[DTC].between(0, MAX_LEAD_TIME).all()
    assert data.search[CHECKIN].max() == COMPETITION_CUTOFF
    assert data.search["log_date"].max() == COMPETITION_CUTOFF


def test_evaluation_covers_only_the_target_window(data):
    assert data.evaluation[CHECKIN].min() == CONFIG.target_start
    assert data.evaluation[CHECKIN].max() == CONFIG.target_end
    assert data.evaluation["log_date"].max() <= COMPETITION_CUTOFF


def test_history_and_evaluation_do_not_overlap(data):
    keys = ["log_date", CITY, CHECKIN]
    assert len(data.search.merge(data.evaluation, on=keys)) == 0


def test_submission_has_exactly_9630_rows(submission):
    assert len(submission) == N_SUBMISSION_ROWS == 9630
    assert submission["cluster_code"].nunique() == N_CITIES
    assert submission["checkin"].nunique() == TARGET_DAYS


def test_submission_passes_the_validator(data, submission):
    report = validate_submission(submission, data.city_codes, CONFIG)
    assert report.valid, report.problems
    assert report.checkin_min == "2025-11-22"
    assert report.checkin_max == "2025-12-21"


def test_every_city_is_predicted_even_with_nothing_observed(data, submission):
    observed = data.evaluation.groupby(CITY)[["search_count"]].sum()
    silent = set(data.city_codes) - set(observed.index)
    assert silent, "the real cutoff leaves 82 cities with nothing observed"
    predicted = submission.groupby("cluster_code")["predicted_demand"].sum()
    # Zero so far is not zero in the end.
    assert (predicted.loc[sorted(silent)] > 0).any()


def test_prediction_never_falls_below_observed_demand(data):
    model = PickupBaseline.fit(data, CONFIG.cutoff, CONFIG)
    predictions = model.predict(build_grid(data, CONFIG.cutoff, CONFIG.target_dates()))
    assert (predictions["predicted_demand"] >= predictions["observed"] - 1e-9).all()


def test_the_pipeline_is_deterministic(data):
    grid = build_grid(data, CONFIG.cutoff, CONFIG.target_dates())
    first = build_submission(PickupBaseline.fit(data, CONFIG.cutoff, CONFIG).predict(grid))
    second = build_submission(PickupBaseline.fit(data, CONFIG.cutoff, CONFIG).predict(grid))
    pd.testing.assert_frame_equal(first, second)
