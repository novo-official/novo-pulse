"""Adapter, profiler, validator and frequency-detection tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.contract import ENTITY, TARGET, TS, DataContract
from ml.data.adapter import DataAdapter
from ml.data.frequency import detect_frequency
from ml.data.profiler import profile_dataset, suggest_schema
from ml.data.validator import DataValidator


def test_adapter_renames_to_canonical_columns(panel):
    assert {TS, ENTITY, TARGET}.issubset(panel.frame.columns)
    assert panel.frame[TS].is_monotonic_increasing is False  # sorted by entity first
    assert panel.frame.groupby(ENTITY)[TS].is_monotonic_increasing.all()


def test_adapter_collapses_duplicate_rows(contract, raw_frame):
    """Two rows for the same (entity, day) must be summed, not duplicated."""
    doubled = pd.concat([raw_frame, raw_frame.head(50)], ignore_index=True)
    panel = DataAdapter(contract).build(doubled)

    assert not panel.frame.duplicated(subset=[ENTITY, TS]).any()
    assert any("duplicate" in note for note in panel.notes)


def test_adapter_fills_calendar_gaps(contract, raw_frame):
    """A missing day becomes a real zero, not a silently skipped period."""
    gapped = raw_frame[raw_frame["date"] != raw_frame["date"].unique()[100]]
    panel = DataAdapter(contract).build(gapped)

    per_entity = panel.frame.groupby(ENTITY)[TS].count()
    assert per_entity.nunique() == 1, "every entity should share one gap-free grid"
    assert any("missing periods" in note for note in panel.notes)


def test_adapter_survives_a_missing_declared_feature(contract, raw_frame):
    """A contract feature the dataset lacks is dropped with a note, not an error."""
    stripped = raw_frame.drop(columns=["searches"])
    panel = DataAdapter(contract).build(stripped)

    assert "searches" not in panel.historical_features
    assert any("absent from data" in note for note in panel.notes)


def test_adapter_handles_a_dataset_with_no_entity_column(raw_frame):
    contract = DataContract.from_dict(
        {
            "schema": {"timestamp": "date", "target": "bookings", "entity_id": None},
            "evaluation": {"horizons": [7]},
        }
    )
    panel = DataAdapter(contract).build(raw_frame)

    assert panel.frame[ENTITY].nunique() == 1
    assert any("single series" in note for note in panel.notes)


def test_aggregate_to_destination_sums_the_target(panel):
    rolled = panel.aggregate("destination")
    assert rolled.frame[ENTITY].nunique() == 3
    assert rolled.frame[TARGET].sum() == pytest.approx(panel.frame[TARGET].sum())


@pytest.mark.parametrize(
    ("freq", "expected"),
    [("D", "D"), ("W", "W"), ("MS", "MS"), ("h", "h")],
)
def test_frequency_detection(freq, expected):
    stamps = pd.Series(pd.date_range("2024-01-01", periods=60, freq=freq))
    assert detect_frequency(stamps)["frequency"] == expected


def test_frequency_detection_flags_irregular_spacing():
    stamps = pd.Series(pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-09", "2024-02-01"]))
    result = detect_frequency(stamps)
    assert result["regular"] is False


def test_profiler_suggests_a_usable_schema(tmp_path, raw_frame):
    path = tmp_path / "sample.csv"
    raw_frame.to_csv(path, index=False)

    profile = profile_dataset(path)
    suggested = profile["suggested_schema"]

    assert suggested["timestamp"] == "date"
    assert suggested["target"] == "bookings"
    assert suggested["entity_id"] == "listing_id"
    assert profile["frequency"]["frequency"] == "D"


def test_profiler_prefers_demand_over_funnel_metrics():
    columns = [
        {"name": "search_count", "kind": "numeric", "constant": False, "unique_pct": 40},
        {"name": "booking_count", "kind": "numeric", "constant": False, "unique_pct": 30},
    ]
    from ml.data.profiler import _candidates

    candidates = _candidates(columns)
    assert candidates["target"][0]["column"] == "booking_count"


def test_validator_scores_clean_data_highly(contract, panel):
    report = DataValidator(contract).validate(panel.frame)
    assert report["health_score"] >= 70
    assert report["by_severity"]["critical"] == 0


def test_validator_flags_a_constant_target(contract, panel):
    broken = panel.frame.copy()
    broken[TARGET] = 5.0

    report = DataValidator(contract).validate(broken)
    codes = {finding["code"] for finding in report["findings"]}
    assert "constant_target" in codes
    assert report["health_score"] < 80


def test_validator_flags_negative_targets(contract, panel):
    broken = panel.frame.copy()
    broken.loc[broken.index[:200], TARGET] = -3.0

    report = DataValidator(contract).validate(broken)
    assert "negative_target" in {finding["code"] for finding in report["findings"]}


def test_validator_flags_a_sparse_target(contract, panel):
    sparse = panel.frame.copy()
    sparse[TARGET] = np.where(np.arange(len(sparse)) % 3 == 0, sparse[TARGET], 0)

    report = DataValidator(contract).validate(sparse)
    assert "sparse_target" in {finding["code"] for finding in report["findings"]}
