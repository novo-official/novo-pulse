"""Non-finite numbers must never reach JSON.

`json.dumps` writes bare `NaN` and `Infinity` by default. Neither is valid
JSON: SQLite's JSON_VALID constraint rejects them (so the run fails to save),
and JavaScript's `JSON.parse` throws (so one NaN blanks the whole dashboard).

This was a real failure - a champion without native quantiles left the interval
bounds empty, `mean_width` came out NaN, and the training run died at the
database write with an IntegrityError.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from ml.evaluation.uncertainty import coverage_by_horizon
from ml.pipelines.training import sanitise_json


def _strict(payload) -> str:
    """Serialise the way a browser would read it: NaN is a hard error."""
    return json.dumps(payload, allow_nan=False)


# ------------------------------------------------------------- the sanitiser
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_floats_become_null(value):
    assert sanitise_json({"x": value}) == {"x": None}


def test_numpy_non_finite_becomes_null():
    assert sanitise_json({"x": np.float64("nan"), "y": np.float32("inf")}) == {
        "x": None,
        "y": None,
    }


def test_finite_values_survive_unchanged():
    payload = {"a": 1.5, "b": 0.0, "c": -3, "d": "text", "e": True, "f": None}
    assert sanitise_json(payload) == payload


def test_sanitiser_recurses_through_nesting():
    payload = {
        "rows": [{"v": float("nan")}, {"v": 2.0}],
        "nested": {"deep": {"deeper": [float("inf"), 1]}},
    }
    cleaned = sanitise_json(payload)

    assert cleaned["rows"][0]["v"] is None
    assert cleaned["rows"][1]["v"] == 2.0
    assert cleaned["nested"]["deep"]["deeper"] == [None, 1]
    _strict(cleaned)  # must not raise


def test_booleans_are_not_coerced_to_numbers():
    """bool is a subclass of int; a careless numeric branch would flatten it."""
    cleaned = sanitise_json({"flag": True, "off": False})
    assert cleaned["flag"] is True and cleaned["off"] is False


def test_numpy_integers_become_plain_ints():
    cleaned = sanitise_json({"n": np.int64(7)})
    assert cleaned["n"] == 7
    _strict(cleaned)


# --------------------------------------------------- the original root cause
def test_coverage_is_null_not_nan_when_a_model_has_no_interval():
    """A model without native quantiles leaves the bounds empty."""
    backtest = pd.DataFrame(
        {
            "horizon": [1, 2, 3, 4],
            "actual": [10.0, 11.0, 12.0, 13.0],
            "prediction": [10.0, 11.0, 12.0, 13.0],
            "lower": [np.nan] * 4,
            "upper": [np.nan] * 4,
        }
    )
    rows = coverage_by_horizon(backtest, [(1, 7)], nominal=0.8)

    assert rows, "the bucket should still be reported"
    assert rows[0]["observed_coverage"] is None
    assert rows[0]["mean_width"] is None
    assert "note" in rows[0]
    _strict(rows)  # the exact serialisation that used to fail


def test_coverage_is_measured_when_bounds_exist():
    backtest = pd.DataFrame(
        {
            "horizon": [1, 2, 3, 4],
            "actual": [10.0, 11.0, 12.0, 13.0],
            "prediction": [10.0] * 4,
            "lower": [9.0] * 4,
            "upper": [14.0] * 4,
        }
    )
    rows = coverage_by_horizon(backtest, [(1, 7)], nominal=0.8)

    assert rows[0]["observed_coverage"] == 1.0
    assert rows[0]["mean_width"] == 5.0


def test_coverage_ignores_only_the_unbounded_rows():
    backtest = pd.DataFrame(
        {
            "horizon": [1, 2, 3, 4],
            "actual": [10.0, 11.0, 12.0, 13.0],
            "prediction": [10.0] * 4,
            "lower": [9.0, 9.0, np.nan, np.nan],
            "upper": [14.0, 14.0, np.nan, np.nan],
        }
    )
    rows = coverage_by_horizon(backtest, [(1, 7)], nominal=0.8)

    assert rows[0]["n"] == 2, "only the rows that actually have bounds are measured"
    _strict(rows)


# ------------------------------------------------------------ end to end
def test_a_run_whose_champion_has_no_interval_still_saves(
    tmp_path_factory, contract, raw_frame
):
    """The exact scenario that failed: baseline-only run, then read it back."""
    from ml.contract import DataContract
    from ml.pipelines.training import TrainingConfig, TrainingPipeline

    data_dir = tmp_path_factory.mktemp("nan-run")
    csv = data_dir / "panel.csv"
    raw_frame.to_csv(csv, index=False)

    run_contract = DataContract.from_dict(contract.to_dict())
    run_contract.path = str(csv)
    run_contract.evaluation.horizons = [7]
    run_contract.evaluation.horizon_buckets = [[1, 7]]

    result = TrainingPipeline(
        TrainingConfig(
            contract=run_contract,
            profile_name="baselines-only",
            horizon=7,
            seed=3,
            runs_dir=tmp_path_factory.mktemp("runs"),
            # Baselines have no native quantiles, so the bounds come back empty.
            profile={
                "max_train_rows": 20_000,
                "samples_per_target": 1,
                "cv_folds": 1,
                "models": ["seasonal_naive_7", "moving_average"],
            },
        )
    ).run()

    # Every artefact must be strictly parseable, exactly as the browser reads it.
    for name in ("metrics", "insights", "metadata", "feature_importance", "peaks"):
        path = result.run_dir / f"{name}.json"
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        json.loads(
            text, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(f"{name}: {c}"))
        )


def test_every_shipped_demo_artifact_is_strict_json():
    """The committed demo artefacts are what DEMO_MODE serves on a fresh clone."""
    from ml.paths import DEMO_ARTIFACTS_DIR

    files = list(DEMO_ARTIFACTS_DIR.glob("*.json"))
    if not files:
        pytest.skip("no precomputed demo artefacts in this checkout")

    for path in files:
        json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda c: (_ for _ in ()).throw(ValueError(f"{path.name}: {c}")),
        )
