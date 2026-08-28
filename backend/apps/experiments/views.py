"""Training run endpoints."""
from __future__ import annotations

from django.conf import settings
from rest_framework.decorators import api_view
from rest_framework.response import Response

from apps.core.responses import error, no_data, ok
from apps.datasets.models import Dataset
from apps.datasets.serializers import ColumnMappingSerializer
from apps.datasets.services import contract_from_mapping
from ml.contract import DataContract
from ml.evaluation.metrics import available_metrics

from . import services
from .models import Experiment, TrainingRun
from .serializers import (
    ExperimentSerializer,
    TrainingRunListSerializer,
    TrainingRunSerializer,
)


@api_view(["GET"])
def training_list(_request):
    runs = TrainingRun.objects.all()[:50]
    return ok(
        {
            "runs": TrainingRunListSerializer(runs, many=True).data,
            "profiles": services.profile_options(),
            "metrics": available_metrics(),
            "default_profile": settings.TRAINING_PROFILE,
            "sync_tasks": settings.SYNC_TASKS,
        }
    )


@api_view(["POST"])
def training_run(request):
    """Kick off a training run. Returns immediately with the run id."""
    dataset_id = request.data.get("dataset_id")
    contract = None

    if dataset_id:
        dataset = Dataset.objects.filter(id=dataset_id).first()
        if dataset is None:
            return error(
                "Dataset not found", 404, detail_fa="دیتاست پیدا نشد."
            )

        # A mapping sent with the request wins over the stored one, so training
        # always uses the mapping the user is actually looking at. Requiring a
        # separate "save" first is a trap: the run silently uses stale roles, or
        # fails with an error the user has to translate back into a click.
        inline = request.data.get("mapping")
        if inline:
            serializer = ColumnMappingSerializer(data=inline)
            if not serializer.is_valid():
                return Response(
                    {"available": False, "errors": serializer.errors}, status=400
                )
            mapping = dict(serializer.validated_data)
            dataset.mapping = {k: v for k, v in mapping.items() if k != "dataset_id"}
            dataset.save(update_fields=["mapping", "updated_at"])
        elif dataset.mapping:
            mapping = dataset.mapping
        else:
            return error(
                "This dataset has no column mapping yet. Map the columns first, "
                "or send a `mapping` object with the training request.",
                detail_fa=(
                    "برای این دیتاست هنوز نگاشت ستون‌ها ثبت نشده است. ابتدا ستون‌ها را "
                    "نگاشت کنید."
                ),
            )
        contract = contract_from_mapping(mapping, dataset.path, dataset.name)
    else:
        contract = DataContract.load()

    horizon = request.data.get("horizon")
    metric = request.data.get("metric")
    if metric and metric not in available_metrics():
        return error(
            f"Unknown metric '{metric}'. Available: {available_metrics()}",
            detail_fa=f"معیار «{metric}» شناخته‌شده نیست.",
        )

    try:
        run = services.start_training(
            contract=contract,
            profile_name=request.data.get("profile"),
            horizon=int(horizon) if horizon else None,
            metric=metric,
            seed=request.data.get("seed"),
            future_covariates=request.data.get("future_covariates"),
            experiment_name=request.data.get("experiment"),
        )
    except Exception as exc:  # noqa: BLE001
        return error(
            f"Could not start training: {exc}",
            500,
            detail_fa=f"شروع آموزش ممکن نشد: {exc}",
        )
    return ok(TrainingRunSerializer(run).data)


@api_view(["GET"])
def training_detail(_request, run_id: str):
    run = TrainingRun.objects.filter(run_id=run_id).first()
    if run is None:
        return no_data(f"No training run '{run_id}'", 404)
    return ok(TrainingRunSerializer(run).data)


@api_view(["GET"])
def experiment_list(_request):
    return ok(ExperimentSerializer(Experiment.objects.all()[:50], many=True).data)
