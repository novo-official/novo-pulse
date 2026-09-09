"""Pol 4 dashboard API.

Read-only, artefact-backed, and deliberately narrow: every endpoint answers one
question the dashboard asks. There is no training trigger and no write path
here - the model is fitted offline by `python -m ml.pol4.pipeline`.
"""
from __future__ import annotations

import logging

from django.http import HttpResponse
from rest_framework.decorators import api_view

from apps.core.responses import error, no_data, ok

from . import reports as reports_module
from . import services

log = logging.getLogger(__name__)

NOT_GENERATED_FA = "خروجی‌های پل ۴ هنوز ساخته نشده‌اند. ابتدا make pol4 را اجرا کنید."


def _guard(builder):
    """Run a service call, turning a missing artefact into an honest empty state."""
    try:
        return ok(builder())
    except services.ArtifactMissing as exc:
        log.warning("pol4 artefact missing: %s", exc)
        return no_data(str(exc), detail_fa=NOT_GENERATED_FA)
    except LookupError as exc:
        return error(str(exc), 404, detail_fa="مورد درخواستی پیدا نشد.")


@api_view(["GET"])
def overview(request):
    """KPI block, national daily series, top cities, provinces, pickup leaders."""
    return _guard(services.overview)


@api_view(["GET"])
def forecast(request):
    """The national daily series on its own, for the headline chart."""
    return _guard(lambda: {"series": services.national_series()})


@api_view(["GET"])
def heatmap(request):
    try:
        top_n = max(1, min(int(request.GET.get("top_n", 20)), 100))
    except ValueError:
        return error("top_n must be an integer", 400)
    return _guard(lambda: services.heatmap(top_n))


@api_view(["GET"])
def city_list(request):
    """Every city with its totals - the selector's source, names included."""
    return _guard(lambda: {"cities": services.records(services.city_index().round(3))})


@api_view(["GET"])
def city_detail(request, city_code: str):
    return _guard(lambda: services.city_detail(city_code))


@api_view(["GET"])
def city_pickup(request, city_code: str):
    checkin = request.GET.get("checkin")
    return _guard(lambda: services.city_pickup(city_code, checkin))


@api_view(["GET"])
def stability(request):
    city = request.GET.get("city_code") or request.GET.get("city")
    checkin = request.GET.get("checkin")
    return _guard(lambda: services.stability_detail(city, checkin))


@api_view(["GET"])
def model_performance(request):
    return _guard(services.model_performance)


@api_view(["GET"])
def report_list(request):
    return ok({"reports": [{"kind": k, "description": v} for k, v in reports_module.REPORTS.items()]})


@api_view(["GET"])
def report_download(request, kind: str):
    """A report as CSV, built from the same artefacts the charts read."""
    try:
        frame = reports_module.build(
            kind,
            city=request.GET.get("city"),
            province=request.GET.get("province"),
            start=request.GET.get("start"),
            end=request.GET.get("end"),
            limit=request.GET.get("limit"),
        )
    except KeyError as exc:
        return error(str(exc), 404, detail_fa="این گزارش وجود ندارد.")
    except services.ArtifactMissing as exc:
        return no_data(str(exc))

    response = HttpResponse(frame.to_csv(index=False), content_type="text/csv; charset=utf-8")
    response["Content-Disposition"] = f'attachment; filename="pol4_{kind}.csv"'
    return response


@api_view(["GET"])
def report_preview(request, kind: str):
    """The first rows of a report, so the UI can show what will be downloaded."""
    try:
        frame = reports_module.build(
            kind,
            city=request.GET.get("city"),
            province=request.GET.get("province"),
            start=request.GET.get("start"),
            end=request.GET.get("end"),
        )
    except KeyError as exc:
        return error(str(exc), 404, detail_fa="این گزارش وجود ندارد.")
    except services.ArtifactMissing as exc:
        return no_data(str(exc))
    return ok(
        {
            "kind": kind,
            "rows": int(len(frame)),
            "columns": list(frame.columns),
            "preview": frame.head(25).round(3).to_dict(orient="records"),
        }
    )
