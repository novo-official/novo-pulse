from django.http import JsonResponse
from django.urls import include, path


def root(_request):
    return JsonResponse(
        {
            "service": "Novo Pulse - AI Demand Forecasting",
            "api": "/api/v1/",
            "health": "/api/v1/health/",
        }
    )


urlpatterns = [
    path("", root),
    path("api/v1/", include("config.api_urls")),
]
