"""The Pol 4 CLI, end to end, on the synthetic two-clock fixture."""
from __future__ import annotations

import json
from dataclasses import replace

import pandas as pd
import pytest

from ml.pol4.champion import ChampionSpec
from ml.pol4.config import SUBMISSION_COLUMNS
from ml.pol4.inference import run_from_bundle
from ml.pol4.pipeline import build_parser, run
from tests.pol4_fixtures import write_dataset

SMALL = ChampionSpec(params={"n_estimators": 40, "num_leaves": 15})


@pytest.fixture(scope="module")
def baseline_run(tmp_path_factory):
    config = write_dataset(tmp_path_factory.mktemp("pol4_pipe_base"))
    return run(config, model="baseline"), config


@pytest.fixture(scope="module")
def champion_run(tmp_path_factory):
    directory = tmp_path_factory.mktemp("pol4_pipe_champ")
    config = replace(write_dataset(directory), calibration_windows=1)
    return run(config, model="champion", spec=SMALL), config


# -------------------------------------------------------------- baseline
def test_baseline_run_writes_every_artefact(baseline_run):
    _, config = baseline_run
    for name in ("results.csv", "pickup_curves.parquet", "backtest_metrics.json", "run_summary.json"):
        assert (config.artifacts_dir / name).exists(), name


def test_baseline_results_are_valid(baseline_run):
    summary, config = baseline_run
    assert summary["submission"]["valid"]
    frame = pd.read_csv(config.artifacts_dir / "results.csv")
    assert tuple(frame.columns) == SUBMISSION_COLUMNS
    assert len(frame) == summary["data"]["cities"] * config.target_days


# -------------------------------------------------------------- champion
def test_champion_run_writes_every_phase2_artefact(champion_run):
    _, config = champion_run
    for name in (
        "results.csv",
        "input_manifest.json",
        "experiments.csv",
        "experiment_summary.json",
        "backtest_metrics_phase2.json",
        "feature_importance.json",
        "model_card.json",
        "stability.parquet",
        "pickup_curves.parquet",
        "run_summary.json",
    ):
        assert (config.artifacts_dir / name).exists(), name
    for name in ("features.parquet", "target.parquet", "meta.parquet", "manifest.json"):
        assert (config.artifacts_dir / "trainset" / name).exists(), name
    for name in ("manifest.json", "model_spec.json", "baseline_state.joblib"):
        assert (config.artifacts_dir / "model_bundle" / name).exists(), name


def test_champion_run_reports_a_valid_submission(champion_run):
    summary, _ = champion_run
    assert summary["submission"]["valid"]
    assert summary["submission"]["problems"] == []


def test_experiments_table_compares_champion_against_the_baseline(champion_run):
    _, config = champion_run
    frame = pd.read_csv(config.artifacts_dir / "experiments.csv")
    assert {"experiment", "stage", "wape", "normalised_bias"} <= set(frame.columns)
    assert "baseline" in set(frame["stage"])
    assert "champion" in set(frame["stage"])
    assert frame["wape"].notna().all()


def test_phase2_backtest_carries_the_required_breakdowns(champion_run):
    _, config = champion_run
    payload = json.loads((config.artifacts_dir / "backtest_metrics_phase2.json").read_text())
    for side in ("champion", "baseline"):
        for key in (
            "by_horizon",
            "by_horizon_bucket",
            "by_province",
            "by_weekday",
            "by_demand_bucket",
            "by_observation_state",
            "high_demand",
        ):
            assert key in payload[side], f"{side}.{key}"
        assert "normalised_bias" in payload[side]["pooled"]


def test_stability_snapshots_are_written(champion_run):
    summary, config = champion_run
    snapshots = pd.read_parquet(config.artifacts_dir / "stability.parquet")
    assert set(snapshots["horizon"]) == {30, 21, 14, 7, 3, 1}
    assert 0.0 <= summary["stability"]["stability_score"] <= 1.0


def test_feature_importance_is_written_for_the_champion(champion_run):
    _, config = champion_run
    importance = json.loads((config.artifacts_dir / "feature_importance.json").read_text())
    assert len(importance) > 0
    assert {"feature", "importance", "share"} <= set(importance[0])


def test_run_summary_records_how_the_champion_was_configured(champion_run):
    summary, _ = champion_run
    champion = summary["champion"]
    assert champion["model"] == "lightgbm"
    assert champion["log1p_target"] is True
    assert champion["training_rows"] > 0
    assert champion["model_bundle"]["load_parity_verified"] is True
    assert champion["trainset"]["rows"] == champion["training_rows"]
    assert len(champion["top_features"]) > 0


def test_run_records_exactly_the_four_allowed_input_sources(champion_run):
    summary, config = champion_run
    manifest = json.loads((config.artifacts_dir / "input_manifest.json").read_text())
    assert manifest["allowed_sources"] == [
        "search_data.csv",
        "evaluation.csv",
        "cities.csv",
        "city_code_mapping.csv",
    ]
    assert summary["provenance"]["input_digest"] == manifest["input_digest"]


def test_saved_bundle_can_produce_submission_without_training(champion_run, tmp_path):
    _, config = champion_run
    output = tmp_path / "from_bundle.csv"
    result = run_from_bundle(config, output_path=output)
    assert result["valid"] is True
    assert result["rows"] == len(pd.read_csv(config.artifacts_dir / "results.csv"))
    pd.testing.assert_frame_equal(
        pd.read_csv(config.artifacts_dir / "results.csv"),
        pd.read_csv(output),
    )


def test_champion_and_baseline_agree_on_the_grid_shape(baseline_run, champion_run):
    baseline_summary, _ = baseline_run
    champion_summary, _ = champion_run
    assert baseline_summary["submission"]["rows"] == champion_summary["submission"]["rows"]
    assert baseline_summary["submission"]["cities"] == champion_summary["submission"]["cities"]


# -------------------------------------------------------------------- CLI
def test_cli_defaults_to_the_champion():
    args = build_parser().parse_args([])
    assert args.model == "champion"
    assert args.ablation is False
    assert args.reuse_trainset is False


def test_cli_exposes_the_baseline_fallback():
    args = build_parser().parse_args(["--model", "baseline", "--skip-stability"])
    assert args.model == "baseline"
    assert args.skip_stability is True
