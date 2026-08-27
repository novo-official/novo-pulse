"""Business insight endpoints."""
from __future__ import annotations

from rest_framework.decorators import api_view

from apps.core.responses import no_data, ok
from apps.forecasting.store import store_for


@api_view(["GET"])
def insight_list(request):
    store = store_for(request.query_params.get("run_id"))
    if store is None:
        return no_data()
    insights = dict(store.insights)
    # The full per-entity table is large; the dashboard asks for it separately.
    entities = insights.pop("entities", [])
    return ok(
        {
            **insights,
            "top_entities": entities[:10],
            "n_entities": len(entities),
            "run_id": store.run_id,
        }
    )


@api_view(["GET"])
def opportunity_list(request):
    store = store_for(request.query_params.get("run_id"))
    if store is None:
        return no_data()
    return ok(store.insights.get("decision_opportunities") or [])


@api_view(["GET"])
def data_quality(request):
    store = store_for(request.query_params.get("run_id"))
    if store is None:
        return no_data()
    return ok((store.metrics.get("data_quality") or store.insights.get("data_quality") or {}))
