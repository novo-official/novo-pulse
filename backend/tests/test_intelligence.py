"""Anomaly detection, peaks, hierarchy, insights and the narrator."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.anomaly.detector import (
    detect_forecast_anomalies,
    detect_peak_periods,
    detect_residual_anomalies,
    robust_z,
)
from ml.contract import ENTITY
from ml.hierarchy import aggregate_bottom_up, coherence_check, entity_map
from ml.insights.engine import build_decision_opportunities, build_insights, status_for
from ml.insights.narrator import LocalLLMNarrator, TemplateNarrator


def _history(days: int = 120, level: float = 10.0, entity: str = "a") -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=days)
    return pd.DataFrame({ENTITY: entity, "ds": dates, "y": level})


# ------------------------------------------------------------------ anomaly
def test_robust_z_is_not_dragged_by_one_outlier():
    values = np.array([10.0] * 50 + [1000.0])
    scores = robust_z(values)
    assert abs(scores[0]) < 1
    assert scores[-1] > 10


def test_residual_anomaly_detection_finds_the_injected_spike():
    dates = pd.date_range("2026-01-01", periods=90)
    frame = pd.DataFrame(
        {ENTITY: "a", "ds": dates, "actual": 10.0, "prediction": 10.0, "horizon": 1}
    )
    frame.loc[45, "actual"] = 60.0

    found = detect_residual_anomalies(frame)
    assert len(found) >= 1
    assert found.iloc[0]["type"] == "spike"
    assert found.iloc[0]["ds"] == dates[45]


def test_residual_detection_stays_quiet_on_clean_data():
    """A threshold detector will occasionally fire on noise; it must not shout.

    At 3 robust sigma a handful of hits per few hundred clean points is
    expected. What must never happen on clean data is a *high severity* alert.
    """
    dates = pd.date_range("2026-01-01", periods=300)
    rng = np.random.default_rng(1)
    frame = pd.DataFrame(
        {
            ENTITY: "a",
            "ds": dates,
            "actual": 10 + rng.normal(0, 0.4, 300),
            "prediction": 10.0,
            "horizon": 1,
        }
    )
    found = detect_residual_anomalies(frame)

    assert len(found) <= 5, "clean data should not produce a wall of anomalies"
    if len(found):
        assert (found["severity"] != "high").all()


def test_forecast_anomaly_detection_compares_against_history():
    history = _history()
    forecast = pd.DataFrame(
        {ENTITY: "a", "ds": pd.date_range("2026-05-01", periods=10), "forecast": 40.0}
    )
    found = detect_forecast_anomalies(forecast, history, season=7)
    assert len(found) > 0
    assert (found["type"] == "spike").all()


# -------------------------------------------------------------------- peaks
def test_peak_detection_ignores_the_regular_weekly_cycle():
    """A weekend that is always busy is not a peak - it is Tuesday's opposite."""
    dates = pd.date_range("2026-01-01", periods=112)
    weekly = np.where(dates.dayofweek.isin([3, 4]), 30.0, 10.0)
    history = pd.DataFrame({ENTITY: "a", "ds": dates, "y": weekly})

    future = pd.date_range(dates[-1] + pd.Timedelta(days=1), periods=28)
    forecast = pd.DataFrame(
        {
            ENTITY: "a",
            "ds": future,
            "forecast": np.where(future.dayofweek.isin([3, 4]), 30.0, 10.0),
        }
    )
    assert detect_peak_periods(forecast, history, season=7) == []


def test_peak_detection_finds_a_genuine_departure():
    dates = pd.date_range("2026-01-01", periods=112)
    weekly = np.where(dates.dayofweek.isin([3, 4]), 30.0, 10.0)
    history = pd.DataFrame({ENTITY: "a", "ds": dates, "y": weekly})

    future = pd.date_range(dates[-1] + pd.Timedelta(days=1), periods=28)
    values = np.where(future.dayofweek.isin([3, 4]), 30.0, 10.0)
    values[10:15] *= 2.0  # a five-day festival
    forecast = pd.DataFrame({ENTITY: "a", "ds": future, "forecast": values})

    peaks = detect_peak_periods(forecast, history, season=7)
    assert len(peaks) == 1
    assert peaks[0]["type"] == "peak"
    assert peaks[0]["days"] == 5
    assert peaks[0]["expected_change_pct"] == pytest.approx(100, abs=1)


def test_peak_detection_skips_near_zero_series():
    """1 -> 2 bookings is not a '+100% peak' worth showing anyone."""
    dates = pd.date_range("2026-01-01", periods=112)
    history = pd.concat(
        [
            pd.DataFrame({ENTITY: "big", "ds": dates, "y": 100.0}),
            pd.DataFrame({ENTITY: "tiny", "ds": dates, "y": 1.0}),
        ]
    )
    future = pd.date_range(dates[-1] + pd.Timedelta(days=1), periods=14)
    forecast = pd.DataFrame({ENTITY: "tiny", "ds": future, "forecast": 2.0})

    assert detect_peak_periods(forecast, history, season=7) == []


# ---------------------------------------------------------------- hierarchy
def test_bottom_up_aggregation_preserves_the_total(panel):
    mapping = entity_map(panel.frame)
    frame = panel.frame.rename(columns={"y": "forecast"})[[ENTITY, "ds", "forecast"]]

    rolled = aggregate_bottom_up(frame, mapping, "destination")
    assert rolled["forecast"].sum() == pytest.approx(frame["forecast"].sum())
    assert rolled[ENTITY].nunique() == 3


def test_coherence_check_reports_zero_drift(panel):
    from ml.hierarchy import aggregate_all_levels

    mapping = entity_map(panel.frame)
    frame = panel.frame.rename(columns={"y": "forecast"})[[ENTITY, "ds", "forecast"]]
    stacked = aggregate_all_levels(frame, mapping)

    for value in coherence_check(stacked).values():
        assert abs(value) < 1e-6


def test_aggregation_rejects_an_unavailable_level(panel):
    mapping = entity_map(panel.frame).drop(columns=["category_id"], errors="ignore")
    frame = panel.frame.rename(columns={"y": "forecast"})[[ENTITY, "ds", "forecast"]]

    with pytest.raises(ValueError):
        aggregate_bottom_up(frame, mapping, "category")


# ----------------------------------------------------------------- insights
def test_insights_compute_change_against_the_matching_window():
    history = _history(days=60, level=10.0)
    forecast = pd.DataFrame(
        {
            ENTITY: "a",
            "ds": pd.date_range("2026-03-02", periods=30),
            "forecast": 12.0,
            "lower": 10.0,
            "upper": 14.0,
        }
    )
    insights = build_insights(forecast, history, [], [], pd.DataFrame(), horizon=30)

    entity = insights["entities"][0]
    assert entity["forecast_total"] == pytest.approx(360)
    assert entity["recent_total"] == pytest.approx(300)
    assert entity["change_pct"] == pytest.approx(20)


def test_status_reflects_the_change_and_spike_risk():
    assert status_for(25, 0.3, False) == "growing"
    assert status_for(-25, 0.3, False) == "declining"
    assert status_for(1, 0.3, False) == "stable"
    assert status_for(1, 0.3, True) == "spike_risk"
    assert status_for(None, None, False) == "unknown"


def test_opportunities_only_appear_with_supporting_evidence():
    flat = build_insights(
        pd.DataFrame(
            {ENTITY: "a", "ds": pd.date_range("2026-03-02", periods=30),
             "forecast": 10.0, "lower": 9.0, "upper": 11.0}
        ),
        _history(days=60, level=10.0),
        [], [], pd.DataFrame(), horizon=30,
    )
    assert build_decision_opportunities(flat) == []


# ----------------------------------------------------------------- narrator
def test_template_narrator_uses_only_supplied_numbers():
    payload = {
        "label": "کیش",
        "change_pct": 18.2,
        "forecast_total": 14500,
        "horizon": 30,
        "drivers": [{"group": "holiday", "label_fa": "تعطیلات", "direction": "positive"}],
        "confidence": {"lower": 12000, "upper": 17800, "label": "medium"},
    }
    result = TemplateNarrator().narrate(payload)

    assert "کیش" in result["text"]
    assert "18.2" in result["text"]
    assert result["grounded"] is True


def test_template_narrator_says_so_when_there_is_nothing_to_say():
    result = TemplateNarrator().narrate({})
    assert result["text"]


def test_llm_validator_rejects_an_invented_number():
    narrator = LocalLLMNarrator()
    payload = {"forecast_total": 100, "change_pct": 5}

    assert narrator._validate("تقاضا ۵ درصد رشد می‌کند و به ۱۰۰ می‌رسد.", payload) is True
    assert narrator._validate("تقاضا ۴۲ درصد رشد می‌کند.", payload) is False


def test_llm_narrator_falls_back_when_unavailable(monkeypatch):
    narrator = LocalLLMNarrator()
    monkeypatch.setattr(type(narrator), "available", property(lambda self: False))

    result = narrator.narrate({"forecast_total": 10, "horizon": 7})
    assert "template" in result["source"]
    assert result["text"]
