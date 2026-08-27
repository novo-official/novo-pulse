"""Structured business insights.

The numbers are computed here, deterministically, from forecast and backtest
artefacts. An LLM - when one is available - only ever *narrates* this structure;
it never produces a figure. See `narrator.py`.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..contract import ENTITY


def _safe_pct(new: float, old: float) -> float | None:
    if old is None or not np.isfinite(old) or abs(old) < 1e-9:
        return None
    return round(float((new - old) / abs(old) * 100), 2)


def build_insights(
    forecast: pd.DataFrame,
    history: pd.DataFrame,
    leaderboard: list[dict[str, Any]],
    peaks: list[dict[str, Any]],
    anomalies: pd.DataFrame,
    primary_metric: str = "wape",
    horizon: int = 30,
    labels: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Every structured insight the dashboard and the narrator consume."""
    labels = labels or {}
    insights: dict[str, Any] = {"generated_for_horizon": horizon, "primary_metric": primary_metric}

    entity_view = _entity_view(forecast, history, horizon, labels)
    insights["entities"] = entity_view

    growing = [e for e in entity_view if e["change_pct"] is not None]
    growing.sort(key=lambda e: -(e["change_pct"] or 0))
    if growing:
        insights["fastest_growing"] = growing[0]
        insights["fastest_declining"] = growing[-1]

    by_uncertainty = [e for e in entity_view if e["relative_interval_width"] is not None]
    by_uncertainty.sort(key=lambda e: -(e["relative_interval_width"] or 0))
    if by_uncertainty:
        insights["highest_uncertainty"] = by_uncertainty[0]
        insights["lowest_uncertainty"] = by_uncertainty[-1]

    insights["upcoming_peaks"] = [p for p in peaks if p["type"] == "peak"][:5]
    insights["upcoming_troughs"] = [p for p in peaks if p["type"] == "trough"][:5]

    champion = next((row for row in leaderboard if not row.get("is_baseline")), None)
    best_baseline = next((row for row in leaderboard if row.get("is_baseline")), None)
    if champion:
        insights["model_confidence"] = {
            "champion": champion["model"],
            "metric": primary_metric,
            "value": champion.get("primary_value"),
            "baseline": best_baseline["model"] if best_baseline else None,
            "baseline_value": best_baseline.get("primary_value") if best_baseline else None,
            "improvement_pct": (
                round(champion.get("improvement_vs_baseline", 0) * 100, 2)
                if champion.get("improvement_vs_baseline") is not None
                else None
            ),
        }

    insights["anomaly_summary"] = {
        "total": int(len(anomalies)),
        "spikes": int((anomalies["type"] == "spike").sum()) if len(anomalies) else 0,
        "drops": int((anomalies["type"] == "drop").sum()) if len(anomalies) else 0,
        "high_severity": (
            int((anomalies["severity"] == "high").sum()) if len(anomalies) else 0
        ),
    }

    total_forecast = float(forecast["forecast"].sum()) if len(forecast) else 0.0
    baseline_total = _historical_reference_total(history, horizon)
    insights["market"] = {
        "total_forecast": round(total_forecast, 2),
        "previous_period_total": None if baseline_total is None else round(baseline_total, 2),
        "change_pct": None if baseline_total is None else _safe_pct(total_forecast, baseline_total),
        "horizon": horizon,
        "n_entities": int(forecast[ENTITY].nunique()) if len(forecast) else 0,
    }
    return insights


def _entity_view(
    forecast: pd.DataFrame, history: pd.DataFrame, horizon: int, labels: dict[str, str]
) -> list[dict[str, Any]]:
    """Per-entity forecast total vs. the equivalent trailing window."""
    if forecast.empty:
        return []

    forward = forecast.groupby(ENTITY).agg(
        forecast_total=("forecast", "sum"),
        forecast_mean=("forecast", "mean"),
        lower_total=("lower", "sum"),
        upper_total=("upper", "sum"),
        periods=("ds", "count"),
    )

    rows: list[dict[str, Any]] = []
    if not history.empty:
        cutoff = pd.Timestamp(history["ds"].max()) - pd.Timedelta(days=horizon - 1)
        trailing = (
            history[history["ds"] >= cutoff].groupby(ENTITY)["y"].sum().rename("recent_total")
        )
    else:
        trailing = pd.Series(dtype=float, name="recent_total")

    merged = forward.join(trailing, how="left").reset_index()
    for row in merged.itertuples(index=False):
        entity = str(getattr(row, ENTITY))
        recent = getattr(row, "recent_total", np.nan)
        recent = None if recent is None or not np.isfinite(recent) else float(recent)
        width = None
        if np.isfinite(row.lower_total) and np.isfinite(row.upper_total) and row.forecast_total > 0:
            width = round(float((row.upper_total - row.lower_total) / row.forecast_total), 4)
        rows.append(
            {
                "entity_id": entity,
                "label": labels.get(entity, entity),
                "forecast_total": round(float(row.forecast_total), 2),
                "forecast_mean": round(float(row.forecast_mean), 3),
                "recent_total": None if recent is None else round(recent, 2),
                "change_pct": None if recent is None else _safe_pct(float(row.forecast_total), recent),
                "lower_total": round(float(row.lower_total), 2),
                "upper_total": round(float(row.upper_total), 2),
                "relative_interval_width": width,
                "periods": int(row.periods),
            }
        )
    rows.sort(key=lambda item: -item["forecast_total"])
    return rows


def _historical_reference_total(history: pd.DataFrame, horizon: int) -> float | None:
    if history.empty:
        return None
    cutoff = pd.Timestamp(history["ds"].max()) - pd.Timedelta(days=horizon - 1)
    window = history[history["ds"] >= cutoff]
    return float(window["y"].sum()) if len(window) else None


def status_for(change_pct: float | None, interval_width: float | None, spike: bool) -> str:
    """Destination status chip used in the overview table."""
    if spike:
        return "spike_risk"
    if change_pct is None:
        return "unknown"
    if change_pct >= 10:
        return "growing"
    if change_pct <= -10:
        return "declining"
    return "stable"


STATUS_FA = {
    "growing": "در حال رشد",
    "stable": "پایدار",
    "declining": "در حال کاهش",
    "spike_risk": "احتمال جهش",
    "unknown": "نامشخص",
}


def build_decision_opportunities(
    insights: dict[str, Any], labels: dict[str, str] | None = None, limit: int = 4
) -> list[dict[str, Any]]:
    """Actionable-opportunity cards.

    Every card states what the data shows and how certain it is. No card claims
    a business action the data cannot support.
    """
    labels = labels or {}
    cards: list[dict[str, Any]] = []

    growing = insights.get("fastest_growing")
    if growing and (growing.get("change_pct") or 0) > 5:
        cards.append(
            {
                "kind": "growth",
                "entity_id": growing["entity_id"],
                "label": labels.get(growing["entity_id"], growing["label"]),
                "headline_fa": f"رشد تقاضا در {labels.get(growing['entity_id'], growing['label'])}",
                "change_pct": growing["change_pct"],
                "evidence_fa": (
                    f"تقاضای پیش‌بینی‌شده {growing['forecast_total']:,.0f} در برابر "
                    f"{(growing.get('recent_total') or 0):,.0f} در دوره مشابه قبل."
                ),
                "uncertainty": growing.get("relative_interval_width"),
            }
        )

    declining = insights.get("fastest_declining")
    if declining and (declining.get("change_pct") or 0) < -5:
        cards.append(
            {
                "kind": "decline",
                "entity_id": declining["entity_id"],
                "label": labels.get(declining["entity_id"], declining["label"]),
                "headline_fa": f"افت تقاضا در {labels.get(declining['entity_id'], declining['label'])}",
                "change_pct": declining["change_pct"],
                "evidence_fa": (
                    f"تقاضای پیش‌بینی‌شده {declining['forecast_total']:,.0f} در برابر "
                    f"{(declining.get('recent_total') or 0):,.0f} در دوره مشابه قبل."
                ),
                "uncertainty": declining.get("relative_interval_width"),
            }
        )

    for peak in (insights.get("upcoming_peaks") or [])[:2]:
        label = labels.get(peak["entity_id"], peak["entity_id"])
        cards.append(
            {
                "kind": "peak",
                "entity_id": peak["entity_id"],
                "label": label,
                "headline_fa": f"دوره اوج تقاضا در {label}",
                "change_pct": peak["expected_change_pct"],
                "evidence_fa": (
                    f"از {peak['start']} تا {peak['end']} "
                    f"({peak['days']} روز) تقاضا حدود "
                    f"{peak['expected_change_pct']:.0f}٪ بالاتر از سطح معمول است."
                ),
                "confidence": peak.get("confidence"),
            }
        )

    uncertain = insights.get("highest_uncertainty")
    if uncertain and (uncertain.get("relative_interval_width") or 0) > 0.8:
        label = labels.get(uncertain["entity_id"], uncertain["label"])
        cards.append(
            {
                "kind": "uncertainty",
                "entity_id": uncertain["entity_id"],
                "label": label,
                "headline_fa": f"عدم‌قطعیت بالا در {label}",
                "change_pct": uncertain.get("change_pct"),
                "evidence_fa": (
                    "بازه اطمینان این مقصد نسبت به سطح پیش‌بینی پهن است؛ "
                    "تصمیم‌گیری قطعی بر پایه این پیش‌بینی توصیه نمی‌شود."
                ),
                "uncertainty": uncertain.get("relative_interval_width"),
            }
        )
    return cards[:limit]
