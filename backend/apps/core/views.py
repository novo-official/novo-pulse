from __future__ import annotations

from django.conf import settings
from rest_framework.decorators import api_view
from rest_framework.response import Response

from ml.contract import list_profiles
from ml.models.registry import describe_registry
from ml.resources import detect_resources

from .responses import ok


@api_view(["GET"])
def health(_request):
    """Liveness plus everything the UI needs to decide what to render."""
    from apps.experiments.models import TrainingRun

    latest = TrainingRun.objects.filter(status=TrainingRun.Status.SUCCEEDED).order_by(
        "-finished_at"
    ).first()
    return Response(
        {
            "status": "ok",
            "demo_mode": settings.DEMO_MODE,
            "sync_tasks": settings.SYNC_TASKS,
            "default_horizon": settings.DEFAULT_HORIZON,
            "primary_metric": settings.PRIMARY_METRIC,
            "training_profile": settings.TRAINING_PROFILE,
            "has_trained_model": latest is not None,
            "latest_run": latest.run_id if latest else None,
        }
    )


@api_view(["GET"])
def system_info(_request):
    """Hardware, available models and training profiles."""
    return ok(
        {
            "resources": detect_resources(),
            "models": describe_registry(),
            "profiles": list_profiles(),
            "demo_mode": settings.DEMO_MODE,
        }
    )
