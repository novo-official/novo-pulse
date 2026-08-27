"""Forecast, dashboard, anomaly and backtest endpoints."""
from __future__ import annotations

from django.conf import settings
from rest_framework.decorators import api_view
from rest_framework.response import Response

from apps.core.responses import no_data, ok
from ml.contract import ENTITY

from . import services
from .selectors import filter_frame, parse_filters, records, resolve_level
from .store import ArtifactStore, active_store, store_for


def _store(request) -> ArtifactStore | None:
    return store_for(request.query_params.get("run_id"))


def _horizon(request, store: ArtifactStore) -> int:
    raw = request.query_params.get("horizon")
    try:
        requested = int(raw) if raw else settings.DEFAULT_HORIZON
    except (TypeError, ValueError):
        requested = settings.DEFAULT_HORIZON
    trained = int(store.metadata.get("horizon") or settings.DEFAULT_HORIZON)
    return max(1, min(requested, trained))


@api_view(["GET"])
def dashboard_summary(request):
    store = _store(request)
    if store is None:
        return no_data()
    level = resolve_level(store, request.query_params.get("level"))
    horizon = _horizon(request, store)
    payload = services.dashboard_summary(store, horizon, level)
    payload["levels"] = store.levels()
    payload["available_horizons"] = _available_horizons(store)
    payload["insights"] = {
        "decision_opportunities": (store.insights.get("decision_opportunities") or []),
        "fastest_growing": store.insights.get("fastest_growing"),
        "fastest_declining": store.insights.get("fastest_declining"),
        "highest_uncertainty": store.insights.get("highest_uncertainty"),
    }
    return ok(payload)


@api_view(["GET"])
def forecast_list(request):
    """Raw forecast rows with the standard filters."""
    store = _store(request)
    if store is None:
        return no_data()
    filters = parse_filters(request.query_params)
    level = resolve_level(store, filters["level"])
    frame = filter_frame(
        store.level_frame("forecast", level),
        entity_id=filters["entity_id"],
        start_date=filters["start_date"],
        end_date=filters["end_date"],
        horizon=filters["horizon"] or _horizon(request, store),
    )
    limit = filters["limit"] or 5000
    return ok(
        records(frame.head(limit), [ENTITY, "ds", "horizon", "forecast", "lower", "upper", "model"]),
        level=level,
        count=int(len(frame)),
        truncated=bool(len(frame) > limit),
    )


@api_view(["GET"])
def forecast_timeseries(request):
    store = _store(request)
    if store is None:
        return no_data()
    filters = parse_filters(request.query_params)
    level = resolve_level(store, filters["level"])
    horizon = _horizon(request, store)
    payload = services.timeseries(store, level, filters["entity_id"], horizon)
    payload["members"] = store.members(level)
    payload["levels"] = store.levels()
    return ok(payload)


@api_view(["GET"])
def forecast_drivers(request):
    store = _store(request)
    if store is None:
        return no_data()
    return ok(services.drivers(store))


@api_view(["GET"])
def forecast_peaks(request):
    store = _store(request)
    if store is None:
        return no_data()
    kind = request.query_params.get("type")
    limit = int(request.query_params.get("limit") or 10)
    return ok(
        {
            "peaks": services.peaks(store, "peak", limit),
            "troughs": services.peaks(store, "trough", limit),
            "all": services.peaks(store, kind, limit) if kind else None,
        }
    )


@api_view(["GET"])
def forecast_overview(request):
    store = _store(request)
    if store is None:
        return no_data()
    level = resolve_level(store, request.query_params.get("level"))
    horizon = _horizon(request, store)
    return ok(services.overview_table(store, level, horizon), level=level, horizon=horizon)


@api_view(["GET"])
def forecast_heatmap(request):
    store = _store(request)
    if store is None:
        return no_data()
    level = resolve_level(store, request.query_params.get("level"))
    horizon = _horizon(request, store)
    top_n = int(request.query_params.get("top_n") or 12)
    return ok(services.heatmap(store, level, horizon, top_n))


@api_view(["GET"])
def anomaly_list(request):
    store = _store(request)
    if store is None:
        return no_data()
    return ok(
        services.anomalies(
            store,
            entity_id=request.query_params.get("id"),
            severity=request.query_params.get("severity"),
            source=request.query_params.get("source"),
            limit=int(request.query_params.get("limit") or 50),
        )
    )


@api_view(["GET"])
def backtest_list(request):
    store = _store(request)
    if store is None:
        return no_data()
    filters = parse_filters(request.query_params)
    level = resolve_level(store, filters["level"])
    frame = filter_frame(
        store.level_frame("backtest", level),
        entity_id=filters["entity_id"],
        start_date=filters["start_date"],
        end_date=filters["end_date"],
    )
    limit = filters["limit"] or 5000
    series = (
        frame.groupby("ds", as_index=False)[["actual", "prediction", "lower", "upper"]]
        .sum()
        .sort_values("ds")
        if len(frame)
        else frame
    )
    return ok(
        {
            "series": records(series),
            "rows": records(frame.head(limit)),
            "count": int(len(frame)),
            "level": level,
        }
    )


@api_view(["GET"])
def backtest_metrics(request):
    store = _store(request)
    if store is None:
        return no_data()
    metrics = store.metrics
    champion = metrics.get("champion")
    return ok(
        {
            "champion": champion,
            "primary_metric": metrics.get("primary_metric"),
            "folds": metrics.get("folds"),
            "fold_scores": metrics.get("fold_scores"),
            "by_horizon": (metrics.get("horizon_scores") or {}).get(champion, []),
            "segments": metrics.get("segment_scores"),
            "intervals": metrics.get("intervals"),
            "coverage_by_horizon": metrics.get("coverage_by_horizon"),
            "uncertainty_method": metrics.get("uncertainty_method"),
        }
    )


@api_view(["GET"])
def model_list(request):
    from ml.models.registry import describe_registry

    store = _store(request)
    payload = {"registry": describe_registry(), "trained": None}
    if store is not None:
        metrics = store.metrics
        metadata = store.metadata
        payload["trained"] = {
            "run_id": store.run_id,
            "champion": metrics.get("champion"),
            "primary_metric": metrics.get("primary_metric"),
            "ensemble_weights": metrics.get("ensemble_weights"),
            "last_trained": metadata.get("created_at"),
            "profile": metadata.get("profile"),
            "horizon": metadata.get("horizon"),
            "frequency": metadata.get("frequency"),
            "n_features": metadata.get("n_features"),
            "training_window": metadata.get("training_window"),
            "dataset": metadata.get("dataset"),
            "target": metadata.get("target"),
            "uncertainty_method": metadata.get("uncertainty_method"),
            "environment": metadata.get("environment"),
            "warnings": metadata.get("warnings"),
            "tuning": metrics.get("tuning") or [],
            "censoring": metrics.get("censoring") or {},
        }
    return ok(payload)


@api_view(["GET"])
def model_leaderboard(request):
    store = _store(request)
    if store is None:
        return no_data()
    metrics = store.metrics
    leaderboard = metrics.get("leaderboard") or []
    weights = metrics.get("ensemble_weights") or {}
    for row in leaderboard:
        row["ensemble_weight"] = weights.get(row["model"])
        row["is_champion"] = row["model"] == metrics.get("champion")
    return ok(
        {
            "leaderboard": leaderboard,
            "champion": metrics.get("champion"),
            "primary_metric": metrics.get("primary_metric"),
            "horizon_buckets": [
                b.get("bucket")
                for b in ((metrics.get("horizon_scores") or {}).get(metrics.get("champion")) or [])
            ],
            "folds": metrics.get("folds"),
        }
    )


@api_view(["GET"])
def forecast_narrative(request):
    """Grounded natural-language summary of the current forecast."""
    from ml.insights.narrator import get_narrator

    store = _store(request)
    if store is None:
        return no_data()
    level = resolve_level(store, request.query_params.get("level"))
    entity_id = request.query_params.get("id")
    horizon = _horizon(request, store)

    insights = store.insights
    summary = services.dashboard_summary(store, horizon, level)
    driver_block = services.drivers(store, top_n=5)

    payload = {
        "horizon": horizon,
        "label": store.label_for(entity_id) if entity_id else "کل بازار",
        "forecast_total": summary["forecast_total"],
        "change_pct": summary["change_pct"],
        "drivers": driver_block["groups"],
        "confidence": {
            "lower": summary["forecast_lower"],
            "upper": summary["forecast_upper"],
            "label": summary["confidence"]["label"],
        },
        "model_confidence": insights.get("model_confidence"),
    }
    if entity_id:
        entity = next(
            (e for e in (insights.get("entities") or []) if e["entity_id"] == entity_id), None
        )
        if entity:
            payload.update(
                {
                    "forecast_total": entity["forecast_total"],
                    "change_pct": entity["change_pct"],
                    "confidence": {
                        "lower": entity["lower_total"],
                        "upper": entity["upper_total"],
                        "label": summary["confidence"]["label"],
                    },
                }
            )

    narrator = get_narrator()
    result = narrator.narrate(payload)
    return ok({**result, "facts": payload})


def _available_horizons(store: ArtifactStore) -> list[int]:
    trained = int(store.metadata.get("horizon") or settings.DEFAULT_HORIZON)
    return [h for h in (7, 14, 30, 60, 90) if h <= trained] or [trained]
