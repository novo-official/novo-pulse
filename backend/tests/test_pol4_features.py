"""Feature-engine semantics and cutoff safety."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.pol4 import load_pol4
from ml.pol4.baseline import PickupBaseline
from ml.pol4.dataset import build_inference_frame, build_training_frame
from ml.pol4.features import (
    GROUP_ORDER,
    CityHistory,
    FeatureBuilder,
    PairTensor,
    feature_names,
)
from ml.pol4.loader import CHECKIN, CITY, SEARCHES
from tests.pol4_fixtures import write_dataset


@pytest.fixture(scope="module")
def pol4(tmp_path_factory):
    config = write_dataset(tmp_path_factory.mktemp("pol4_feat"))
    data = load_pol4(config)
    return data, config


@pytest.fixture(scope="module")
def builder(pol4):
    data, config = pol4
    history = data.history_before(config.cutoff)
    dates = pd.date_range(config.cutoff - pd.Timedelta(days=60), config.cutoff, freq="D")
    keys = pd.MultiIndex.from_product(
        [data.city_codes, dates], names=[CITY, CHECKIN]
    ).to_frame(index=False)
    events = history[history[CHECKIN].isin(dates)]
    return FeatureBuilder.build(data, config.cutoff, keys, events, config)


# ------------------------------------------------------------------ tensor
def test_tensor_reproduces_the_raw_counts(builder, pol4):
    data, config = pol4
    tensor = builder.tensor
    total = tensor.daily.sum()
    history = data.history_before(config.cutoff)
    expected = history[history[CHECKIN].isin(pd.DatetimeIndex(tensor.keys[CHECKIN]))][
        SEARCHES
    ].sum()
    assert total == pytest.approx(expected)


def test_observed_at_horizon_is_the_suffix_sum(builder):
    tensor = builder.tensor
    for horizon in (0, 1, 5, 12):
        np.testing.assert_allclose(
            tensor.observed(horizon), tensor.daily[:, horizon:].sum(axis=1), rtol=1e-5
        )


def test_observed_is_monotone_in_the_horizon(builder):
    tensor = builder.tensor
    previous = tensor.observed(0)
    for horizon in range(1, 15):
        current = tensor.observed(horizon)
        assert (current <= previous + 1e-6).all()
        previous = current


def test_final_equals_observed_at_horizon_zero(builder):
    np.testing.assert_allclose(builder.tensor.final(), builder.tensor.observed(0))


def test_pickup_windows_sum_to_the_difference_in_observed(builder):
    tensor = builder.tensor
    for horizon in (1, 4, 9):
        for window in (1, 3, 7):
            np.testing.assert_allclose(
                tensor.pickup(horizon, window),
                tensor.observed(horizon) - tensor.observed(horizon + window),
                rtol=1e-5,
                atol=1e-4,
            )


# ----------------------------------------------------------- cutoff safety
def test_a_feature_at_horizon_h_never_reads_below_h(builder, pol4):
    """The decisive feature test.

    Overwrite every search that arrived later than h days before check-in -
    exactly what a forecaster standing at that cutoff cannot know - and require
    every feature to be unchanged.
    """
    data, config = pol4
    baseline = PickupBaseline.fit(data, config.cutoff, config)
    horizon = 7

    before = builder.at_horizon(horizon, GROUP_ORDER, baseline)

    poisoned = FeatureBuilder(
        data=builder.data,
        cutoff=builder.cutoff,
        config=builder.config,
        tensor=PairTensor(
            keys=builder.tensor.keys,
            daily=builder.tensor.daily.copy(),
            n_dtc=builder.tensor.n_dtc,
        ),
        market=builder.market,
        market_rows=builder.market_rows,
        province=builder.province,
        province_rows=builder.province_rows,
        city_history=builder.city_history,
        province_of=builder.province_of,
    )
    poisoned.tensor.daily[:, :horizon] = 1e6
    poisoned.market, poisoned.market_rows = poisoned.tensor.group_by(None)
    poisoned.province, poisoned.province_rows = poisoned.tensor.group_by(
        poisoned.tensor.keys[CITY].map(builder.province_of)
    )

    after = poisoned.at_horizon(horizon, GROUP_ORDER, baseline)
    pd.testing.assert_frame_equal(before, after)


def test_market_aggregates_only_pool_the_same_checkin(builder):
    """A market feature is the sum over cities for that night, not all nights."""
    tensor, market, rows = builder.tensor, builder.market, builder.market_rows
    horizon = 5
    observed = tensor.observed(horizon)
    market_observed = market.observed(horizon)[rows]

    by_checkin = (
        pd.DataFrame({"checkin": tensor.keys[CHECKIN], "observed": observed})
        .groupby("checkin")["observed"]
        .transform("sum")
        .to_numpy()
    )
    np.testing.assert_allclose(market_observed, by_checkin, rtol=1e-4)
    assert (market_observed >= observed - 1e-6).all()


def test_province_aggregates_only_pool_the_same_province_and_checkin(builder):
    tensor, province, rows = builder.tensor, builder.province, builder.province_rows
    horizon = 5
    frame = pd.DataFrame(
        {
            "province": tensor.keys[CITY].map(builder.province_of).to_numpy(),
            "checkin": tensor.keys[CHECKIN],
            "observed": tensor.observed(horizon),
        }
    )
    expected = frame.groupby(["province", "checkin"])["observed"].transform("sum").to_numpy()
    np.testing.assert_allclose(province.observed(horizon)[rows], expected, rtol=1e-4)


def test_city_history_only_uses_completed_checkins(pol4):
    data, config = pol4
    cutoff = pd.Timestamp("2024-08-01")
    before = CityHistory.fit(data, cutoff, config)

    poisoned = data.search.copy()
    poisoned.loc[poisoned[CHECKIN] > cutoff, SEARCHES] = 10**6
    import dataclasses

    after = CityHistory.fit(dataclasses.replace(data, search=poisoned), cutoff, config)
    pd.testing.assert_frame_equal(before.table, after.table)


def test_city_history_counts_days_with_no_searches(pol4):
    """A check-in with no row had zero demand and must be in the denominator.

    Averaging over only the days that produced a row is the "missing means
    zero" trap, and for a sparse city it inflates the level several-fold.
    """
    data, config = pol4
    history = CityHistory.fit(data, config.cutoff, config)
    quiet_city = 6  # the near-silent fixture city

    demand = (
        data.history_before(config.cutoff)
        .query("city_code == @quiet_city")
        .groupby(CHECKIN)[SEARCHES]
        .sum()
    )
    window = pd.date_range(
        config.cutoff - pd.Timedelta(days=config.city_history_days - 1), config.cutoff, freq="D"
    )
    window = window[window >= data.history_before(config.cutoff)[CHECKIN].min()]
    complete = demand.reindex(window, fill_value=0)

    assert history.table.loc[quiet_city, "city_hist_mean"] == pytest.approx(
        complete.mean(), rel=1e-6
    )
    # The trap: the same statistic computed over present rows only.
    assert complete.mean() < demand.mean()


# --------------------------------------------------------------- datasets
def test_training_target_is_final_minus_observed(pol4):
    data, config = pol4
    baseline = PickupBaseline.fit(data, config.cutoff, config)
    frame = build_training_frame(data, config.cutoff, GROUP_ORDER, baseline, config)
    np.testing.assert_allclose(
        frame.y, (frame.meta["final"] - frame.meta["observed"]).clip(lower=0).to_numpy()
    )
    assert (frame.y >= 0).all()


def test_training_rows_never_target_a_checkin_after_the_cutoff(pol4):
    data, config = pol4
    baseline = PickupBaseline.fit(data, config.cutoff, config)
    frame = build_training_frame(data, config.cutoff, GROUP_ORDER, baseline, config)
    assert frame.meta[CHECKIN].max() <= config.cutoff


def test_training_frame_covers_every_requested_horizon(pol4):
    data, config = pol4
    baseline = PickupBaseline.fit(data, config.cutoff, config)
    frame = build_training_frame(data, config.cutoff, GROUP_ORDER, baseline, config)
    assert set(frame.meta["horizon"]) == set(config.train_horizons)


def test_inference_frame_covers_the_whole_grid_at_the_right_horizons(pol4):
    data, config = pol4
    baseline = PickupBaseline.fit(data, config.cutoff, config)
    frame = build_inference_frame(
        data, config.cutoff, config.target_dates(), GROUP_ORDER, baseline, config
    )
    assert len(frame) == len(data.city_codes) * config.target_days
    expected = (pd.DatetimeIndex(frame.meta[CHECKIN]) - config.cutoff).days
    np.testing.assert_array_equal(frame.meta["horizon"].to_numpy(), expected.to_numpy())


def test_inference_observed_matches_the_evaluation_file(pol4):
    data, config = pol4
    baseline = PickupBaseline.fit(data, config.cutoff, config)
    frame = build_inference_frame(
        data, config.cutoff, config.target_dates(), GROUP_ORDER, baseline, config
    )
    assert frame.meta["observed"].sum() == pytest.approx(
        data.evaluation[SEARCHES].sum(), rel=1e-6
    )


def test_inference_ignores_searches_logged_after_the_cutoff(pol4):
    """Post-cutoff searches for a target night must not reach the features."""
    import dataclasses

    data, config = pol4
    baseline = PickupBaseline.fit(data, config.cutoff, config)
    before = build_inference_frame(
        data, config.cutoff, config.target_dates(), GROUP_ORDER, baseline, config
    )

    future = data.search.copy()
    mask = future["log_date"] > config.cutoff
    future.loc[mask, SEARCHES] = 10**6
    after = build_inference_frame(
        dataclasses.replace(data, search=future),
        config.cutoff,
        config.target_dates(),
        GROUP_ORDER,
        baseline,
        config,
    )
    pd.testing.assert_frame_equal(before.X, after.X)


def test_declared_feature_names_match_what_is_built(pol4, builder):
    data, config = pol4
    baseline = PickupBaseline.fit(data, config.cutoff, config)
    for size in range(1, len(GROUP_ORDER) + 1):
        groups = GROUP_ORDER[:size]
        frame = builder.at_horizon(3, groups, baseline)
        assert list(frame.columns) == feature_names(groups), groups


def test_weekend_flag_matches_the_iranian_week(builder, pol4):
    data, config = pol4
    baseline = PickupBaseline.fit(data, config.cutoff, config)
    frame = builder.at_horizon(3, GROUP_ORDER, baseline)
    weekday = frame["checkin_weekday"].to_numpy()
    weekend = frame["is_weekend"].to_numpy()
    # Thursday (3) and Friday (4) are the Iranian weekend.
    np.testing.assert_array_equal(weekend, np.isin(weekday, (3, 4)).astype(float))


# ------------------------------------------------------------------ export
def test_export_writes_features_target_and_keys(pol4, tmp_path):
    from ml.pol4.dataset import export_training_frame

    data, config = pol4
    baseline = PickupBaseline.fit(data, config.cutoff, config)
    frame = build_training_frame(data, config.cutoff, GROUP_ORDER, baseline, config)

    path = export_training_frame(frame, tmp_path / "train.csv")
    written = pd.read_csv(path)

    assert len(written) == len(frame)
    # Keys first, so the file is readable without the schema beside it.
    assert list(written.columns[:6]) == [
        "city_code",
        "checkin",
        "horizon",
        "observed",
        "final",
        "remaining_demand",
    ]
    assert set(feature_names(GROUP_ORDER)).issubset(written.columns)


def test_exported_target_reconstructs_the_forecast(pol4, tmp_path):
    """observed + remaining must equal final, in the file as well as in memory."""
    from ml.pol4.dataset import export_training_frame

    data, config = pol4
    baseline = PickupBaseline.fit(data, config.cutoff, config)
    frame = build_training_frame(data, config.cutoff, GROUP_ORDER, baseline, config)
    written = pd.read_csv(export_training_frame(frame, tmp_path / "train.csv"))

    np.testing.assert_allclose(
        written["observed"] + written["remaining_demand"], written["final"], rtol=1e-6
    )


def test_sampled_export_covers_every_horizon(pol4, tmp_path):
    from ml.pol4.dataset import export_training_frame

    data, config = pol4
    baseline = PickupBaseline.fit(data, config.cutoff, config)
    frame = build_training_frame(data, config.cutoff, GROUP_ORDER, baseline, config)

    path = export_training_frame(frame, tmp_path / "sample.csv", sample=220)
    written = pd.read_csv(path)
    # Stratified: a band the champion fits separately must not be under-covered.
    assert set(written["horizon"]) == set(config.train_horizons)
    assert len(written) < len(frame)


def test_export_is_deterministic(pol4, tmp_path):
    from ml.pol4.dataset import export_training_frame

    data, config = pol4
    baseline = PickupBaseline.fit(data, config.cutoff, config)
    frame = build_training_frame(data, config.cutoff, GROUP_ORDER, baseline, config)
    first = pd.read_csv(export_training_frame(frame, tmp_path / "a.csv", sample=220, seed=7))
    second = pd.read_csv(export_training_frame(frame, tmp_path / "b.csv", sample=220, seed=7))
    pd.testing.assert_frame_equal(first, second)
