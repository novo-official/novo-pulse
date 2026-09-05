"""Business logic for the dashboard and forecast endpoints."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ml.contract import ENTITY
from ml.evaluation.uncertainty import confidence_score
from ml.insights.engine import STATUS_FA, status_for

from .selectors import collapse_to_series, filter_frame, records
from .store import ArtifactStore

HISTORY_PERIODS = 120


def dashboard_summary(store: ArtifactStore, horizon: int, level: str) -> dict[str, Any]:
    """The KPI block at the top of the dashboard."""
    metrics = store.metrics
    insights = store.insights
    metadata = store.metadata

    forecast = filter_frame(store.level_frame("forecast", level), horizon=horizon)
    history = store.level_frame("history", level)

    total = float(forecast["forecast"].sum()) if len(forecast) else 0.0
    lower = float(forecast["lower"].sum()) if len(forecast) else 0.0
    upper = float(forecast["upper"].sum()) if len(forecast) else 0.0

    previous_total = None
    change_pct = None
    if len(history):
        cutoff = history["ds"].max() - pd.Timedelta(days=horizon - 1)
        previous_total = float(history[history["ds"] >= cutoff]["y"].sum())
        if previous_total > 0:
            change_pct = round((total - previous_total) / previous_total * 100, 2)

    entities = insights.get("entities") or []
    top_growth = next(
        (e for e in sorted(entities, key=lambda x: -(x.get("change_pct") or -1e9))
         if e.get("change_pct") is not None),
        None,
    )

    leaderboard = metrics.get("leaderboard") or []
    champion = next((row for row in leaderboard if row["model"] == metrics.get("champion")), None)
    baseline = next((row for row in leaderboard if row.get("is_baseline")), None)

    anomalies = store.frame("anomalies")
    important = 0
    if len(anomalies) and "severity" in anomalies.columns:
        important = int((anomalies["severity"].isin(["high", "medium"])).sum())

    validation_error = _as_error_ratio(champion, metrics.get("primary_metric", "wape"))
    relative_width = (upper - lower) / total if total > 0 else None
    confidence = confidence_score(
        interval_width=(upper - lower) / max(len(forecast), 1),
        actual_level=total / max(len(forecast), 1),
        validation_error=validation_error if validation_error is not None else 0.3,
        horizon=horizon,
        max_horizon=metadata.get("horizon", horizon),
        history_periods=int(history["ds"].nunique()) if len(history) else 0,
    )

    return {
        "horizon": horizon,
        "level": level,
        "forecast_total": round(total, 2),
        "forecast_lower": round(lower, 2),
        "forecast_upper": round(upper, 2),
        "relative_interval_width": None if relative_width is None else round(relative_width, 4),
        "previous_period_total": None if previous_total is None else round(previous_total, 2),
        "change_pct": change_pct,
        "top_growth": top_growth,
        "confidence": confidence,
        "model": {
            "champion": metrics.get("champion"),
            "primary_metric": metrics.get("primary_metric"),
            "primary_value": champion.get("primary_value") if champion else None,
            "baseline_model": baseline.get("model") if baseline else None,
            "baseline_value": baseline.get("primary_value") if baseline else None,
            "improvement_pct": (
                round(champion["improvement_vs_baseline"] * 100, 2)
                if champion and champion.get("improvement_vs_baseline") is not None
                else None
            ),
            "last_trained": metadata.get("created_at"),
            "training_window": metadata.get("training_window"),
            "uncertainty_method": metrics.get("uncertainty_method"),
            "n_features": metadata.get("n_features"),
        },
        "anomalies": {
            "total": int(len(anomalies)),
            "important": important,
            **(insights.get("anomaly_summary") or {}),
        },
        "data_quality": {
            "health_score": (metrics.get("data_quality") or {}).get("health_score"),
            "grade": (metrics.get("data_quality") or {}).get("grade"),
        },
        "run_id": store.run_id,
    }


def _as_error_ratio(champion: dict | None, metric: str) -> float | None:
    """Normalise the champion's score onto a 0-1 error scale for confidence."""
    if not champion:
        return None
    value = champion.get("primary_value")
    if value is None:
        return None
    if metric in {"wape", "rmsle"}:
        return float(value)
    if metric in {"mape", "smape"}:
        return float(value) / 100.0
    if metric == "r2":
        return float(max(0.0, 1.0 - value))
    metrics = champion.get("metrics") or {}
    if metrics.get("wape") is not None:
        return float(metrics["wape"])
    return None


def timeseries(
    store: ArtifactStore,
    level: str,
    entity_id: str | None,
    horizon: int,
    history_periods: int = HISTORY_PERIODS,
) -> dict[str, Any]:
    """History + in-sample backtest + future forecast, ready to chart."""
    forecast = filter_frame(
        store.level_frame("forecast", level), entity_id=entity_id, horizon=horizon
    )
    forecast_series = collapse_to_series(forecast, ["forecast", "lower", "upper"])

    history = store.level_frame("history", level)
    if entity_id:
        history = history[history[ENTITY].astype(str) == str(entity_id)]
    history_series = collapse_to_series(history, ["y"]).tail(history_periods)

    backtest = filter_frame(store.level_frame("backtest", level), entity_id=entity_id)
    backtest_series = collapse_to_series(backtest, ["actual", "prediction", "lower", "upper"])
    if len(backtest_series) and len(history_series):
        backtest_series = backtest_series[backtest_series["ds"] >= history_series["ds"].min()]

    points: dict[str, dict[str, Any]] = {}

    def merge(frame: pd.DataFrame, mapping: dict[str, str]) -> None:
        for row in frame.to_dict(orient="records"):
            key = pd.Timestamp(row["ds"]).strftime("%Y-%m-%d")
            entry = points.setdefault(key, {"ds": key})
            for source, target in mapping.items():
                value = row.get(source)
                if value is not None and np.isfinite(value):
                    entry[target] = round(float(value), 4)

    merge(history_series, {"y": "actual"})
    merge(backtest_series, {"prediction": "backtest", "lower": "backtest_lower", "upper": "backtest_upper"})
    merge(forecast_series, {"forecast": "forecast", "lower": "lower", "upper": "upper"})

    series = [points[key] for key in sorted(points)]
    split = forecast_series["ds"].min() if len(forecast_series) else None

    return {
        "level": level,
        "entity_id": entity_id,
        "label": store.label_for(entity_id) if entity_id else "کل بازار",
        "horizon": horizon,
        "forecast_start": None if split is None else split.strftime("%Y-%m-%d"),
        "series": series,
        "totals": {
            "forecast": round(float(forecast_series["forecast"].sum()), 2) if len(forecast_series) else 0.0,
            "lower": round(float(forecast_series["lower"].sum()), 2) if len(forecast_series) else 0.0,
            "upper": round(float(forecast_series["upper"].sum()), 2) if len(forecast_series) else 0.0,
            "history_mean": round(float(history_series["y"].mean()), 3) if len(history_series) else None,
        },
    }


def overview_table(store: ArtifactStore, level: str, horizon: int) -> list[dict[str, Any]]:
    """The destination overview table: current, forecast, change, status."""
    forecast = filter_frame(store.level_frame("forecast", level), horizon=horizon)
    if forecast.empty:
        return []
    history = store.level_frame("history", level)

    forward = forecast.groupby(ENTITY).agg(
        forecast_total=("forecast", "sum"),
        lower_total=("lower", "sum"),
        upper_total=("upper", "sum"),
    )
    recent = pd.Series(dtype=float)
    if len(history):
        cutoff = history["ds"].max() - pd.Timedelta(days=horizon - 1)
        recent = history[history["ds"] >= cutoff].groupby(ENTITY)["y"].sum()

    # Only a *forecast* spike is a forward-looking risk. A residual spike is
    # something that already happened and says nothing about the coming window.
    anomalies = store.frame("anomalies")
    spike_entities: set[str] = set()
    if len(anomalies) and "type" in anomalies.columns:
        upcoming_spikes = anomalies[anomalies["type"] == "spike"]
        if "source" in upcoming_spikes.columns:
            upcoming_spikes = upcoming_spikes[upcoming_spikes["source"] == "forecast"]
        spike_entities = {str(e) for e in upcoming_spikes[ENTITY].unique()}

    rows = []
    for entity, row in forward.iterrows():
        key = str(entity)
        current = float(recent.get(key, np.nan))
        current = None if not np.isfinite(current) else current
        change = (
            None
            if not current
            else round((float(row.forecast_total) - current) / current * 100, 2)
        )
        width = (
            round(float(row.upper_total - row.lower_total) / float(row.forecast_total), 4)
            if row.forecast_total > 0
            else None
        )
        status = status_for(change, width, key in spike_entities)
        rows.append(
            {
                "entity_id": key,
                "label": store.label_for(key),
                "current_demand": None if current is None else round(current, 2),
                "forecast_total": round(float(row.forecast_total), 2),
                "lower_total": round(float(row.lower_total), 2),
                "upper_total": round(float(row.upper_total), 2),
                "change_pct": change,
                "relative_interval_width": width,
                "confidence": _width_to_confidence(width),
                "status": status,
                "status_fa": STATUS_FA.get(status, status),
            }
        )
    rows.sort(key=lambda item: -item["forecast_total"])
    return rows


def _width_to_confidence(width: float | None) -> str:
    if width is None:
        return "unknown"
    if width <= 0.5:
        return "high"
    if width <= 1.0:
        return "medium"
    return "low"


def heatmap(store: ArtifactStore, level: str, horizon: int, top_n: int = 12) -> dict[str, Any]:
    """Entity x date forecast matrix for the demand heatmap."""
    forecast = filter_frame(store.level_frame("forecast", level), horizon=horizon)
    if forecast.empty:
        return {"dates": [], "rows": []}

    totals = forecast.groupby(ENTITY)["forecast"].sum().sort_values(ascending=False)
    keep = list(totals.head(top_n).index)
    selected = forecast[forecast[ENTITY].isin(keep)]

    pivot = selected.pivot_table(
        index=ENTITY, columns="ds", values="forecast", aggfunc="sum"
    ).reindex(keep)
    dates = [pd.Timestamp(c).strftime("%Y-%m-%d") for c in pivot.columns]

    rows = []
    for entity, values in pivot.iterrows():
        raw = values.to_numpy(dtype=float)
        baseline = float(np.nanmedian(raw)) if np.isfinite(raw).any() else 0.0
        intensity = (
            ((raw - baseline) / baseline).tolist() if baseline > 0 else [0.0] * len(raw)
        )
        rows.append(
            {
                "entity_id": str(entity),
                "label": store.label_for(entity),
                "values": [None if not np.isfinite(v) else round(float(v), 2) for v in raw],
                "intensity": [None if not np.isfinite(v) else round(float(v), 4) for v in intensity],
                "total": round(float(np.nansum(raw)), 2),
            }
        )
    return {"dates": dates, "rows": rows, "level": level}


def drivers(store: ArtifactStore, top_n: int = 8) -> dict[str, Any]:
    """Driver groups behind the forecast, split into positive and negative.

    The list is truncated for readability, but the remainder is folded into an
    explicit "other" group so the shares still sum to 100%. Silently dropping
    the tail would make the chart read as though 93% were the whole story.
    """
    importance = store.json("feature_importance", {})
    groups = importance.get("group_importance") or []
    shown = list(groups[:top_n])
    remainder = list(groups[top_n:])

    # The explainer already emits its own "other" group for features that match
    # no known family. Folding the truncated tail into a *second* group of the
    # same name would collide their ids and show two rows both labelled سایر,
    # so the two are merged into one honest remainder.
    if remainder or any(g.get("group") == "other" for g in shown):
        existing = next((g for g in shown if g.get("group") == "other"), None)
        if existing is not None:
            shown = [g for g in shown if g is not existing]
            remainder = [existing, *remainder]

    if remainder:
        residual_share = sum(g.get("contribution_share", 0.0) for g in remainder)
        residual_effect = sum(g.get("signed_effect", 0.0) for g in remainder)
        shown = [
            *shown,
            {
                "group": "other",
                "label_fa": f"سایر عوامل ({len(remainder)})",
                "contribution_share": round(residual_share, 6),
                "contribution_score": round(
                    sum(g.get("contribution_score", 0.0) for g in remainder), 6
                ),
                "signed_effect": round(residual_effect, 6),
                "direction": (
                    "positive" if residual_effect > 0
                    else "negative" if residual_effect < 0 else "neutral"
                ),
            },
        ]

    positive = [g for g in groups if g.get("direction") == "positive"][:top_n]
    negative = [g for g in groups if g.get("direction") == "negative"][:top_n]
    return {
        "method": importance.get("method"),
        "base_value": importance.get("base_value"),
        "groups": shown,
        "n_groups": len(groups),
        "positive": positive,
        "negative": negative,
        "features": (importance.get("global_importance") or [])[:20],
        "note": importance.get("note", ""),
        "price_dependence": store.json("price_dependence", []),
    }


def peaks(store: ArtifactStore, kind: str | None = None, limit: int = 10) -> list[dict[str, Any]]:
    items = store.json("peaks", []) or []
    if kind:
        items = [p for p in items if p.get("type") == kind]
    for item in items:
        item["label"] = store.label_for(item.get("entity_id", ""))
    return items[:limit]


def anomalies(
    store: ArtifactStore,
    entity_id: str | None = None,
    severity: str | None = None,
    source: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    frame = store.frame("anomalies")
    if frame.empty:
        return []
    if entity_id:
        frame = frame[frame[ENTITY].astype(str) == str(entity_id)]
    if severity:
        frame = frame[frame["severity"] == severity]
    if source and "source" in frame.columns:
        frame = frame[frame["source"] == source]
    frame = frame.reindex(frame["score"].abs().sort_values(ascending=False).index).head(limit)
    rows = records(frame)
    for row in rows:
        row["label"] = store.label_for(row.get(ENTITY, ""))
        row["type_fa"] = {"spike": "جهش تقاضا", "drop": "افت تقاضا"}.get(row.get("type"), "")
    return rows
