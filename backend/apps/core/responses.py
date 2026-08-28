"""Shared response helpers.

`DEMO_MODE=false` is a hard promise: if there is no trained run, the API says
so instead of inventing a number.
"""
from __future__ import annotations

from typing import Any

from django.conf import settings
from rest_framework import status
from rest_framework.response import Response

NO_DATA_MESSAGE_FA = "داده‌ای در دسترس نیست. ابتدا یک مدل آموزش دهید."
NO_DATA_MESSAGE_EN = "No data available. Train a model first."


def no_data(detail: str | None = None, http_status: int = status.HTTP_200_OK) -> Response:
    return Response(
        {
            "available": False,
            "demo_mode": settings.DEMO_MODE,
            "detail": detail or NO_DATA_MESSAGE_EN,
            "detail_fa": NO_DATA_MESSAGE_FA,
            "data": None,
        },
        status=http_status,
    )


def ok(data: Any, **extra: Any) -> Response:
    payload = {"available": True, "demo_mode": settings.DEMO_MODE, "data": data}
    payload.update(extra)
    return Response(payload)


def error(
    detail: str,
    http_status: int = status.HTTP_400_BAD_REQUEST,
    detail_fa: str | None = None,
) -> Response:
    """An error the UI can show verbatim.

    `detail_fa` is what a Persian-speaking user actually reads; `detail` stays
    English for logs and for anyone driving the API directly.
    """
    return Response(
        {
            "available": False,
            "detail": detail,
            "detail_fa": detail_fa or detail,
            "data": None,
        },
        status=http_status,
    )
