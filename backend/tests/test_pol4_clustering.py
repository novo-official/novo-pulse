"""Clustering-as-aggregation: the guarantees that make the comparison honest.

Clustering only earns its place here if it is a *data* step. These tests pin the
four properties that claim depends on:

* the identity partition reproduces the original panel exactly, so the
  unclustered arm is not a different pipeline wearing the same name;
* aggregation conserves demand and never crosses a province, so the WAPE
  denominator and the `province_*` feature block are untouched by the level;
* every feature is recomputed from the pooled log rather than averaged from the
  members' own values - the mistake the design note warns about, and the one
  that would silently corrupt every cluster row;
* the assignment is decided without reading anything after the cutoff.
"""
from __future__ import annotations

import dataclasses
import json

import numpy as np
import pandas as pd
import pytest

from ml.pol4 import load_pol4
from ml.pol4.aggregate import CLUSTER_ATTRIBUTES, aggregate_data, city_volume_shares
from ml.pol4.cluster_experiment import (
    CLUSTER,
    aggregate_panel,
    blend_panels,
    cluster_spec,
    fit_panel,
)
from ml.pol4.cluster_pipeline import (
    build_cluster_submission,
    load_sweep,
    select_level,
    validate_cluster_submission,
)
from ml.pol4.clustering import ClusterPlan, identity_assignment
from ml.pol4.features import (
    CLUSTER_GROUP_ORDER,
    GROUP_ACTIVITY,
    GROUP_BASE,
    GROUP_CITY,
    GROUP_ORDER,
    GROUP_PICKUP,
    GROUP_PROVINCE,
    CityHistory,
    feature_names,
)
from ml.pol4.dataset import build_training_frame
from ml.pol4.loader import CHECKIN, CITY, PROVINCE, SEARCHES, Pol4Data
from tests.pol4_fixtures import CUTOFF, write_dataset

GARBAGE = 10**6
#: The fixture has six cities; at 5% of the market cities 1 and 4 are hubs and
#: the other four are mergeable, which is the mix the real data has at 0.1%.
ELIGIBLE_SHARE = 0.05
NO_CURVE_GROUPS = (GROUP_BASE, GROUP_PICKUP, GROUP_ACTIVITY, GROUP_CITY, GROUP_PROVINCE)


@pytest.fixture(scope="module")
def pol4(tmp_path_factory):
    config = dataclasses.replace(
        write_dataset(tmp_path_factory.mktemp("pol4_cluster")),
        cluster_eligible_share=ELIGIBLE_SHARE,
    )
    return load_pol4(config), config


@pytest.fixture(scope="module")
def plan(pol4):
    data, config = pol4
    return ClusterPlan.fit(data, config.cutoff, config)


@pytest.fixture(scope="module")
def merged(plan):
    """The coarsest partition the fixture supports - every eligible city pooled."""
    return plan.assign(float(plan.merge_heights().max()) + 1.0)


# ------------------------------------------------------- the identity arm
def test_identity_partition_reproduces_the_panel(pol4):
    data, config = pol4
    assignment = identity_assignment(data, config.cutoff, config)
    assert assignment.is_identity
    assert assignment.n_groups == len(data.cities)

    panel = aggregate_data(data, assignment)
    pd.testing.assert_frame_equal(panel.search, data.search)
    pd.testing.assert_frame_equal(panel.evaluation, data.evaluation)
    assert list(panel.cities[CITY]) == sorted(data.cities[CITY])
    assert (panel.cities["n_cities_in_cluster"] == 1).all()
    assert (panel.cities["cluster_max_member_share"] == 1.0).all()


def test_cut_height_zero_is_the_identity_partition(plan, pol4):
    data, config = pol4
    assert plan.assign(0.0).is_identity
    assert set(plan.assign(0.0).frame["cluster_code"]) == set(data.cities[CITY])


def test_champion_feature_schema_is_unchanged(pol4):
    """The clustered arm must not have moved the model already submitted."""
    assert len(feature_names(GROUP_ORDER)) == 66
    assert feature_names(CLUSTER_GROUP_ORDER)[:66] == feature_names(GROUP_ORDER)
    assert feature_names(CLUSTER_GROUP_ORDER)[66:] == list(CLUSTER_ATTRIBUTES)


# ---------------------------------------------------------- the aggregation
def test_aggregation_conserves_demand(pol4, merged):
    data, _ = pol4
    panel = aggregate_data(data, merged)
    assert panel.search[SEARCHES].sum() == data.search[SEARCHES].sum()
    assert panel.evaluation[SEARCHES].sum() == data.evaluation[SEARCHES].sum()

    members = merged.mapping()
    expected = (
        data.search.assign(g=data.search[CITY].map(members))
        .groupby("g")[SEARCHES]
        .sum()
        .sort_index()
    )
    actual = panel.search.groupby(CITY)[SEARCHES].sum().sort_index()
    assert np.array_equal(expected.to_numpy(), actual.to_numpy())


def test_clusters_never_cross_a_province(plan, pol4):
    data, _ = pol4
    province_of = data.province_of
    for height in plan.height_ladder(4):
        frame = plan.assign(height).frame
        provinces = frame.assign(p=frame[CITY].map(province_of)).groupby("cluster_code")["p"]
        assert (provinces.nunique() == 1).all(), f"a cluster crossed a province at {height}"


def test_hub_cities_are_never_merged(plan, merged):
    assert set(plan.protected.tolist()) == {1, 4}
    singletons = merged.clusters.loc[
        merged.clusters["n_cities_in_cluster"] == 1, "cluster_code"
    ]
    assert set(plan.protected.tolist()) <= set(singletons.tolist())


def test_the_wape_denominator_does_not_move_with_the_level(pol4, plan):
    """Aggregation moves demand between rows; it never creates or destroys it."""
    data, config = pol4
    start, end = config.target_start, config.target_end
    totals = set()
    for height in plan.height_ladder(4):
        panel = aggregate_data(data, plan.assign(height))
        totals.add(round(float(panel.final_demand(start, end)["final"].sum()), 6))
    assert len(totals) == 1


# ------------------------------------- features: recomputed, never averaged
def test_summable_features_are_the_sum_of_their_members(pol4, merged):
    data, config = pol4
    city = build_training_frame(
        data, config.cutoff, NO_CURVE_GROUPS, None, config
    )
    panel = build_training_frame(
        aggregate_data(data, merged), config.cutoff, NO_CURVE_GROUPS, None, config
    )
    members = merged.mapping()

    city_rows = city.meta.assign(
        observed_total=city.X["observed_total"].to_numpy(),
        pickup_7d=city.X["pickup_7d"].to_numpy(),
        g=city.meta[CITY].map(members),
    )
    summed = city_rows.groupby(["g", CHECKIN, "horizon"])[
        ["observed_total", "pickup_7d"]
    ].sum()
    cluster_rows = panel.meta.assign(
        observed_total=panel.X["observed_total"].to_numpy(),
        pickup_7d=panel.X["pickup_7d"].to_numpy(),
    ).set_index([CITY, CHECKIN, "horizon"])

    aligned = summed.reindex(cluster_rows.index)
    np.testing.assert_allclose(
        cluster_rows["observed_total"].to_numpy(), aligned["observed_total"].to_numpy()
    )
    np.testing.assert_allclose(
        cluster_rows["pickup_7d"].to_numpy(), aligned["pickup_7d"].to_numpy()
    )


def test_history_statistics_are_recomputed_from_the_pooled_series(pol4, merged):
    """A cluster's volatility is not the average of its members' volatilities."""
    data, config = pol4
    panel = aggregate_data(data, merged)
    pooled = CityHistory.fit(panel, config.cutoff, config).table
    per_city = CityHistory.fit(data, config.cutoff, config).table

    sizes = merged.clusters.set_index("cluster_code")["n_cities_in_cluster"]
    multi = sizes.index[sizes > 1]
    assert len(multi), "the fixture must produce at least one multi-city cluster"

    members = merged.frame.set_index(CITY)["cluster_code"]
    for code in multi:
        group = members.index[members == code]
        averaged = per_city.loc[group, "city_hist_std"].mean()
        recomputed = pooled.loc[code, "city_hist_std"]
        assert not np.isclose(recomputed, averaged), (
            "cluster history statistics were averaged from the members instead of "
            "being recomputed from the pooled daily series"
        )
        # Pooling cannot make the series smaller than its largest member.
        assert recomputed >= per_city.loc[group, "city_hist_std"].max() - 1e-9


def test_market_and_province_blocks_are_untouched_by_clustering(pol4, merged):
    data, config = pol4
    city = build_training_frame(data, config.cutoff, NO_CURVE_GROUPS, None, config)
    panel = build_training_frame(
        aggregate_data(data, merged), config.cutoff, NO_CURVE_GROUPS, None, config
    )
    province_of = data.province_of

    city_view = (
        city.meta.assign(
            total=city.X["province_observed_total"].to_numpy(),
            p=city.meta[CITY].map(province_of),
        )
        .drop_duplicates(["p", CHECKIN, "horizon"])
        .set_index(["p", CHECKIN, "horizon"])["total"]
    )
    cluster_province = merged.clusters.set_index("cluster_code")[PROVINCE]
    panel_view = (
        panel.meta.assign(
            total=panel.X["province_observed_total"].to_numpy(),
            p=panel.meta[CITY].map(cluster_province),
        )
        .drop_duplicates(["p", CHECKIN, "horizon"])
        .set_index(["p", CHECKIN, "horizon"])["total"]
    )
    aligned = city_view.reindex(panel_view.index)
    np.testing.assert_allclose(panel_view.to_numpy(), aligned.to_numpy())


def test_cluster_columns_describe_the_row(pol4, merged):
    data, _ = pol4
    panel = aggregate_data(data, merged)
    volumes = merged.frame.set_index(CITY)["volume"]
    members = merged.frame.groupby("cluster_code")[CITY].apply(list)
    table = panel.cities.set_index(CITY)
    for code, group in members.items():
        assert table.loc[code, "n_cities_in_cluster"] == len(group)
        assert table.loc[code, "cluster_min_member_volume"] == volumes[group].min()
        expected = volumes[group].max() / max(volumes[group].sum(), 1e-9)
        assert np.isclose(table.loc[code, "cluster_max_member_share"], expected)


def test_the_cluster_group_requires_an_aggregated_panel(pol4):
    """Asking for the cluster columns on a raw panel must fail loudly."""
    data, config = pol4
    with pytest.raises(ValueError, match="aggregate_data"):
        build_training_frame(data, config.cutoff, CLUSTER_GROUP_ORDER, None, config)


# -------------------------------------------------------------- leakage
def test_the_assignment_ignores_everything_after_the_cutoff(pol4):
    data, config = pol4
    cutoff = pd.Timestamp("2024-08-01")

    def poison(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        out.loc[out[CHECKIN] > cutoff, SEARCHES] = GARBAGE
        out.loc[out["log_date"] > cutoff, SEARCHES] = GARBAGE
        return out

    corrupted = dataclasses.replace(
        data, search=poison(data.search), evaluation=poison(data.evaluation)
    )
    clean_plan = ClusterPlan.fit(data, cutoff, config)
    dirty_plan = ClusterPlan.fit(corrupted, cutoff, config)

    pd.testing.assert_frame_equal(clean_plan.profiles, dirty_plan.profiles)
    for height in clean_plan.height_ladder(3):
        pd.testing.assert_frame_equal(
            clean_plan.assign(height).frame, dirty_plan.assign(height).frame
        )


# --------------------------------------------------------------- blending
def test_blend_endpoints_are_the_two_pure_predictions(pol4, plan, merged):
    data, config = pol4
    spec = cluster_spec(params={"n_estimators": 20}, bands=((1, 30),))
    cutoff = pd.Timestamp(config.backtest_cutoffs[-1])
    city = fit_panel(data, cutoff, plan.assign(0.0), spec, config)
    cluster = fit_panel(data, cutoff, merged, spec, config)

    pure_city = blend_panels(city, cluster, shrinkage=0.0)
    np.testing.assert_allclose(
        pure_city["predicted_demand"].to_numpy(),
        city.panel["predicted_demand"].to_numpy(),
    )

    pure_cluster = blend_panels(city, cluster, shrinkage=1e18)
    expected = (
        pure_cluster["observed"].to_numpy()
        + pure_cluster["allocated_cluster_remaining"].to_numpy()
    )
    merged_rows = pure_cluster["blend_weight_on_city"].to_numpy() < 1.0
    assert merged_rows.any()
    np.testing.assert_allclose(
        pure_cluster.loc[merged_rows, "predicted_demand"].to_numpy(),
        expected[merged_rows],
        rtol=1e-9,
    )


def test_blending_never_touches_a_city_that_is_nobody_s_cluster_mate(pol4, plan, merged):
    """Otherwise the 'blend' would silently be a two-model ensemble."""
    data, config = pol4
    spec = cluster_spec(params={"n_estimators": 20}, bands=((1, 30),))
    cutoff = pd.Timestamp(config.backtest_cutoffs[-1])
    city = fit_panel(data, cutoff, plan.assign(0.0), spec, config)
    cluster = fit_panel(data, cutoff, merged, spec, config)

    blended = blend_panels(city, cluster, shrinkage=1e18)
    singletons = set(plan.protected.tolist())
    untouched = blended[CITY].isin(singletons).to_numpy()
    assert untouched.any()
    np.testing.assert_allclose(blended.loc[untouched, "blend_weight_on_city"], 1.0)
    np.testing.assert_allclose(
        blended.loc[untouched, "predicted_demand"].to_numpy(),
        blended.loc[untouched, "city_prediction"].to_numpy(),
    )


def test_the_blend_can_never_fall_below_what_is_already_observed(pol4, plan, merged):
    data, config = pol4
    spec = cluster_spec(params={"n_estimators": 20}, bands=((1, 30),))
    cutoff = pd.Timestamp(config.backtest_cutoffs[-1])
    city = fit_panel(data, cutoff, plan.assign(0.0), spec, config)
    cluster = fit_panel(data, cutoff, merged, spec, config)
    for shrinkage in (0.0, 1e3, 1e6, 1e18):
        blended = blend_panels(city, cluster, shrinkage=shrinkage)
        assert (
            blended["predicted_demand"].to_numpy()
            >= blended["observed"].to_numpy() - 1e-9
        ).all()


def test_gain_decomposition_separates_the_metric_from_the_model():
    from ml.pol4.cluster_experiment import SweepLevel, decompose_gain

    def level(groups, wape, control=None):
        return SweepLevel(
            cut_height=0.0 if groups == 321 else 1.0,
            folds={},
            pooled={"wape": wape, "normalised_bias": 0.0, "mae": 0.0},
            n_groups=[groups],
            assignments={},
            diagnostics={},
            control={"wape": control} if control is not None else {},
        )

    # A level that scores better only because its rows are coarser.
    decomposed = decompose_gain([level(321, 0.120), level(110, 0.118, control=0.118)])
    assert decomposed[0]["mechanical_gain"] == pytest.approx(0.002)
    assert decomposed[0]["modelling_gain"] == pytest.approx(0.0)
    assert decomposed[0]["modelling_share_of_gain"] == pytest.approx(0.0)


def test_volume_shares_sum_to_one_per_cluster(merged):
    shares = city_volume_shares(merged)
    grouped = shares.groupby(merged.mapping().reindex(shares.index).to_numpy()).sum()
    np.testing.assert_allclose(grouped.to_numpy(), 1.0)


def test_summing_the_city_panel_lands_on_the_cluster_grain(pol4, plan, merged):
    """The control arm: same rows, same outcome, different predictor."""
    data, config = pol4
    # A historical cutoff, so the window has already closed and carries truth.
    cutoff = pd.Timestamp(config.backtest_cutoffs[-1])
    spec = cluster_spec(params={"n_estimators": 20}, bands=((1, 30),))
    city = fit_panel(data, cutoff, plan.assign(0.0), spec, config)
    assert "actual" in city.panel.columns
    control = aggregate_panel(city.panel, merged)
    assert len(control) == merged.n_groups * config.target_days
    assert np.isclose(control["actual"].sum(), city.panel["actual"].sum())
    assert np.isclose(
        control["predicted_demand"].sum(), city.panel["predicted_demand"].sum()
    )


# ------------------------------------------------------------- submission
def test_clustered_submission_contract(pol4, plan, merged):
    data, config = pol4
    spec = cluster_spec(params={"n_estimators": 20}, bands=((1, 30),))
    result = fit_panel(
        data, config.cutoff, merged, spec, config, target_dates=config.target_dates()
    )
    frame = build_cluster_submission(result.panel)
    report = validate_cluster_submission(frame, merged, data.city_codes, config)
    assert report["valid"]
    assert report["rows"] == merged.n_groups * config.target_days
    assert report["cities_covered"] == len(data.cities)

    dropped = frame[frame["cluster_code"] != frame["cluster_code"].iloc[0]]
    with pytest.raises(ValueError, match="missing"):
        validate_cluster_submission(dropped, merged, data.city_codes, config)


def _level(height, groups, wape, control=None):
    from ml.pol4.cluster_experiment import SweepLevel

    return SweepLevel(
        cut_height=height,
        folds={},
        pooled={"wape": wape, "normalised_bias": 0.0, "mae": 0.0},
        n_groups=[groups],
        assignments={"x": {"merged_demand_share": 0.0, "n_groups": groups}},
        diagnostics={},
        control={"wape": control} if control is not None else {},
    )


def test_selection_ignores_a_gain_that_is_only_the_metric_being_kinder():
    """A level whose whole gain survives the control is the only one that counts."""
    free_lunch = [
        _level(0.0, 321, 0.1200),
        # Better pooled WAPE, but the city model's own predictions score the
        # same on those rows: nothing was learned from pooling.
        _level(1.0, 110, 0.1170, control=0.1170),
    ]
    selected, report = select_level(free_lunch, min_gain=0.02)
    assert selected.mean_groups == 321
    assert report["clustered_candidates_passing"] == 0
    # ...whereas the naive rule would have taken it.
    assert report["naive_total_gain_rule"]["selected_mean_groups"] == 110

    earned = [
        _level(0.0, 321, 0.1200),
        _level(1.0, 110, 0.1140, control=0.1170),
    ]
    selected, report = select_level(earned, min_gain=0.02)
    assert selected.mean_groups == 110
    assert report["selected_relative_modelling_gain"] == pytest.approx(0.025, abs=1e-4)


def test_selection_prefers_the_unclustered_panel_on_a_tie():
    levels = [_level(0.0, 321, 0.150), _level(1.0, 200, 0.1499, control=0.1502)]
    selected, report = select_level(levels, min_gain=0.02)
    assert selected.mean_groups == 321
    assert report["clustered_candidates_passing"] == 0


def test_a_saved_sweep_round_trips_into_the_same_selection(tmp_path):
    """`--from-sweep` must reproduce the decision, not approximate it."""
    from dataclasses import asdict

    levels = [
        _level(0.0, 321, 0.1200),
        _level(1.0, 110, 0.1140, control=0.1170),
    ]
    payload = {"level_detail": [asdict(level) for level in levels]}
    path = tmp_path / "cluster_sweep.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    restored = load_sweep(path)["levels"]
    assert [level.pooled for level in restored] == [level.pooled for level in levels]
    assert select_level(restored, min_gain=0.02)[0].mean_groups == 110


def test_the_clustered_arm_ships_a_reloadable_lightgbm_bundle(pol4, plan, merged, tmp_path):
    """A model nobody can reload is a claim, not an artefact."""
    from ml.pol4.aggregate import aggregate_data
    from ml.pol4.champion import FittedChampion
    from ml.pol4.cluster_pipeline import save_arm

    data, config = pol4
    spec = cluster_spec(params={"n_estimators": 20}, bands=((1, 30),))
    result = fit_panel(
        data,
        config.cutoff,
        merged,
        spec,
        config,
        target_dates=config.target_dates(),
        keep_models=True,
    )
    assert result.bundle is not None
    assert result.bundle.spec.kind == "lightgbm"

    report = save_arm(
        result,
        data,
        tmp_path / "bundle",
        input_digest="test-digest",
        target_dates=config.target_dates(),
    )
    assert report["model"] == "lightgbm"
    assert report["boosters"] == ["lightgbm_h1_30.txt"]
    assert (tmp_path / "bundle" / "lightgbm_h1_30.txt").is_file()

    # And it really does reload into the same predictions on the same panel.
    restored = FittedChampion.load(tmp_path / "bundle")
    panel_data = aggregate_data(data, merged)
    pd.testing.assert_frame_equal(
        restored.predict(config.target_dates(), panel_data),
        result.bundle.predict(config.target_dates(), panel_data),
    )
    # An assignment is part of the model, not an unchecked neighboring CSV.
    assignment_path = tmp_path / "bundle" / "clusters.csv"
    saved_assignment = pd.read_csv(assignment_path)
    assert saved_assignment["cluster_code"].nunique() == merged.n_groups
    saved_assignment.loc[0, "cluster_code"] = -999
    saved_assignment.to_csv(assignment_path, index=False)
    with pytest.raises(ValueError, match="checksum mismatch"):
        FittedChampion.load(tmp_path / "bundle")


def test_the_sweep_does_not_retain_a_bundle_per_level(pol4, plan):
    """Twenty-odd fits with their boosters held in memory is how a sweep OOMs."""
    data, config = pol4
    spec = cluster_spec(params={"n_estimators": 20}, bands=((1, 30),))
    result = fit_panel(data, config.cutoff, plan.assign(0.0), spec, config)
    assert result.bundle is None
    assert result.models == {}


def test_the_clustered_arm_records_that_it_is_uncalibrated(pol4):
    """The sweep compares partitions, so no arm may carry its own calibrator."""
    assert cluster_spec().calibration == "none"
    assert cluster_spec().describe()["calibration_method"] == "none"
