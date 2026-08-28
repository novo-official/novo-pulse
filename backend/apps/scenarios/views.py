from __future__ import annotations

import logging

from rest_framework.decorators import api_view

from apps.core.responses import error, no_data, ok
from apps.forecasting.store import active_store

from . import services

log = logging.getLogger(__name__)


@api_view(["GET"])
def scenario_options(request):
    """Which covariates can be simulated, and over what window."""
    store = active_store()
    if store is None:
        return no_data()
    try:
        scenario = services.load_engine(store.directory, store.run_id)
    except Exception as exc:  # noqa: BLE001
        log.exception("scenario engine failed to load")
        return error(
            f"Scenario engine unavailable: {exc}",
            503,
            detail_fa=f"موتور شبیه‌سازی در دسترس نیست: {exc}",
        )

    tensor = scenario.engine.tensor
    origin = tensor.origin_index
    return ok(
        {
            "run_id": store.run_id,
            "model": getattr(scenario.model, "name", "unknown"),
            "horizon": scenario.horizon,
            "adjustable": scenario.adjustable,
            "window": {
                "start": str(tensor.dates[origin + 1].date()),
                "end": str(tensor.dates[min(origin + scenario.horizon, tensor.n_periods - 1)].date()),
            },
            "levels": ["market", "destination", "category", "listing"],
        }
    )


@api_view(["POST"])
def scenario_simulate(request):
    store = active_store()
    if store is None:
        return no_data()

    adjustments = request.data.get("adjustments") or []
    if not isinstance(adjustments, list):
        return error("`adjustments` must be a list of {column, change_pct|value}.")

    try:
        scenario = services.load_engine(store.directory, store.run_id)
        result = services.simulate(
            scenario,
            adjustments=adjustments,
            level=request.data.get("level") or "destination",
            entity_id=request.data.get("entity_id"),
            start_date=request.data.get("start_date"),
            end_date=request.data.get("end_date"),
            horizon=request.data.get("horizon"),
        )
    except Exception as exc:  # noqa: BLE001
        log.exception("scenario simulation failed")
        return error(
            f"Simulation failed: {exc}",
            500,
            detail_fa=f"اجرای سناریو با خطا مواجه شد: {exc}",
        )
    return ok(result)
