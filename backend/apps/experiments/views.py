"""Training run endpoints."""
from __future__ import annotations

from django.conf import settings
from rest_framework.decorators import api_view

from apps.core.responses import error, no_data, ok
from apps.datasets.models import Dataset
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
            return error("Dataset not found", 404)
        if not dataset.mapping:
            return error("Map the dataset columns before training.")
        contract = contract_from_mapping(dataset.mapping, dataset.path, dataset.name)
    else:
        contract = DataContract.load()

    horizon = request.data.get("horizon")
    metric = request.data.get("metric")
    if metric and metric not in available_metrics():
        return error(f"Unknown metric '{metric}'. Available: {available_metrics()}")

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
        return error(f"Could not start training: {exc}", 500)
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
