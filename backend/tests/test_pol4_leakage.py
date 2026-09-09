"""Pol 4 leakage guarantees.

The generic platform proves its single-axis guarantee by overwriting everything
after the forecast origin with garbage and requiring identical features
(`tests/test_leakage.py`). These tests do the same thing on the *second* clock,
which is where this competition's leakage actually lives:

* a pickup curve fitted at cutoff C may only read check-ins completed by C;
* the demand "observed at C" may only read log dates up to C;
* corrupting anything after C must leave every prediction bit-identical.

If any of these fail, a backtest score is meaningless - the model would be
reading the answer.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from ml.pol4 import load_pol4
from ml.pol4.backtest import run_fold
from ml.pol4.baseline import PickupBaseline, ZeroObservationPrior
from ml.pol4.loader import CHECKIN, CITY, SEARCHES, Pol4Data
from ml.pol4.pickup import PickupCurves
from ml.pol4.submission import build_grid
from tests.pol4_fixtures import write_dataset

GARBAGE = 10**6


@pytest.fixture(scope="module")
def pol4(tmp_path_factory):
    config = write_dataset(tmp_path_factory.mktemp("pol4_leak"))
    return load_pol4(config), config


def _corrupt_after(data: Pol4Data, cutoff: pd.Timestamp, axis: str) -> Pol4Data:
    """Replace every row after `cutoff` on the given axis with garbage.

    `axis="checkin"` attacks the completion curves (which need finished
    check-ins); `axis="log_date"` attacks the observed-so-far aggregation.
    """
    def poison(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        mask = out[axis] > cutoff
        out.loc[mask, SEARCHES] = GARBAGE
        return out

    return dataclasses.replace(
        data, search=poison(data.search), evaluation=poison(data.evaluation)
    )


# ------------------------------------------------------------- pickup curves
def test_pickup_curves_ignore_checkins_after_the_cutoff(pol4):
    data, config = pol4
    cutoff = pd.Timestamp("2024-09-01")
    before = PickupCurves.fit(data, cutoff, config)
    after = PickupCurves.fit(_corrupt_after(data, cutoff, CHECKIN), cutoff, config)

    np.testing.assert_array_equal(before.global_curve, after.global_curve)
    assert before.city_curves.keys() == after.city_curves.keys()
    for city, curve in before.city_curves.items():
        np.testing.assert_array_equal(curve, after.city_curves[city])
    for province, curve in before.province_curves.items():
        np.testing.assert_array_equal(curve, after.province_curves[province])


def test_a_curve_is_the_same_whether_or_not_later_data_exists(pol4):
    """Fitting at C on the full file must equal fitting at C on a file that ends at C."""
    data, config = pol4
    cutoff = pd.Timestamp("2024-08-15")
    truncated = dataclasses.replace(
        data,
        search=data.search[data.search[CHECKIN] <= cutoff],
        evaluation=data.evaluation.iloc[0:0],
    )
    np.testing.assert_array_equal(
        PickupCurves.fit(data, cutoff, config).global_curve,
        PickupCurves.fit(truncated, cutoff, config).global_curve,
    )


def test_zero_prior_ignores_checkins_after_the_cutoff(pol4):
    data, config = pol4
    cutoff = pd.Timestamp("2024-09-01")
    before = ZeroObservationPrior.fit(data, cutoff, config)
    after = ZeroObservationPrior.fit(_corrupt_after(data, cutoff, CHECKIN), cutoff, config)
    np.testing.assert_array_equal(before.global_curve, after.global_curve)
    for city, curve in before.by_city.items():
        np.testing.assert_array_equal(curve, after.by_city[city])


# --------------------------------------------------------------- observation
def test_observed_demand_ignores_log_dates_after_the_cutoff(pol4):
    data, config = pol4
    cutoff = config.cutoff
    dates = config.target_dates()
    before = data.observed_at(cutoff, dates.min(), dates.max())
    after = _corrupt_after(data, cutoff, "log_date").observed_at(
        cutoff, dates.min(), dates.max()
    )
    pd.testing.assert_frame_equal(before, after)


def test_the_grid_is_unchanged_by_post_cutoff_searches(pol4):
    data, config = pol4
    before = build_grid(data, config.cutoff, config.target_dates())
    after = build_grid(
        _corrupt_after(data, config.cutoff, "log_date"), config.cutoff, config.target_dates()
    )
    pd.testing.assert_frame_equal(before, after)


# ---------------------------------------------------------------- end to end
def test_predictions_are_identical_after_poisoning_everything_post_cutoff(pol4):
    """The decisive test.

    Corrupt every search that happens after the cutoff - the exact information a
    forecaster standing at the cutoff cannot have - and require the prediction
    for every (city, check-in) pair to be unchanged.
    """
    data, config = pol4
    cutoff = config.cutoff
    grid = build_grid(data, cutoff, config.target_dates())

    before = PickupBaseline.fit(data, cutoff, config).predict(grid)
    poisoned = _corrupt_after(data, cutoff, "log_date")
    after = PickupBaseline.fit(poisoned, cutoff, config).predict(
        build_grid(poisoned, cutoff, config.target_dates())
    )

    pd.testing.assert_frame_equal(
        before.drop(columns=["curve_source"]),
        after.drop(columns=["curve_source"]),
        check_dtype=False,
    )


def test_backtest_fold_predictions_survive_post_cutoff_poisoning(pol4):
    data, config = pol4
    cutoff = pd.Timestamp("2024-09-01")
    before = run_fold(data, cutoff, config)
    after = run_fold(_corrupt_after(data, cutoff, "log_date"), cutoff, config)
    np.testing.assert_allclose(
        before.predictions["predicted_demand"].to_numpy(),
        after.predictions["predicted_demand"].to_numpy(),
    )


def test_a_fold_never_reads_a_log_date_after_its_cutoff(pol4):
    """Structural check: the rows a fold consumes are all at or before the cutoff."""
    data, config = pol4
    cutoff = pd.Timestamp("2024-09-01")
    history = data.history_before(cutoff)
    assert history["log_date"].max() <= cutoff
    assert history[CHECKIN].max() <= cutoff

    dates = pd.date_range(cutoff + pd.Timedelta(days=1), periods=config.target_days)
    grid = build_grid(data, cutoff, dates)
    truth_free = data.observed_at(cutoff, dates.min(), dates.max())
    # Observed demand must be strictly less than final demand somewhere, or the
    # target window is not actually in the future.
    final = data.final_demand(dates.min(), dates.max())
    merged = grid.merge(final, on=[CITY, CHECKIN], how="left").fillna({"final": 0.0})
    assert (merged["observed"] <= merged["final"] + 1e-9).all()
    assert merged["observed"].sum() < merged["final"].sum()
    assert len(truth_free) <= len(merged)
