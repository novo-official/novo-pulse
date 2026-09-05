"""Data Lab endpoints: upload -> profile -> map -> validate."""
from __future__ import annotations

import logging

from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import FormParser, JSONParser, MultiPartParser
from rest_framework.response import Response

from apps.core.responses import error, ok
from ml.contract import DataContract
from ml.evaluation.metrics import available_metrics
from ml.paths import ACTIVE_CONTRACT_FILE, REPO_ROOT

from . import services
from .models import Dataset
from .serializers import ColumnMappingSerializer, DatasetSerializer

log = logging.getLogger(__name__)


@api_view(["GET"])
def dataset_list(request):
    datasets = Dataset.objects.all()[:50]
    return ok(
        {
            "datasets": DatasetSerializer(datasets, many=True).data,
            "active_contract": _active_contract(),
            "metrics": available_metrics(),
        }
    )


@api_view(["POST"])
@parser_classes([MultiPartParser, FormParser, JSONParser])
def dataset_upload(request):
    """Accept a CSV/Parquet/Excel upload and profile it immediately."""
    uploaded = request.FILES.get("file")
    local_path = request.data.get("path")

    try:
        if uploaded is not None:
            path = services.store_upload(uploaded)
            source = Dataset.Source.UPLOAD
            original = uploaded.name
        elif local_path:
            path = services.register_local_file(local_path)
            source = Dataset.Source.LOCAL
            original = str(local_path)
        else:
            return error(
                "Provide either a `file` upload or a `path` to a local file.",
                detail_fa="یک فایل انتخاب کنید یا مسیر یک فایل محلی را وارد کنید.",
            )
    except (ValueError, FileNotFoundError) as exc:
        return error(str(exc))

    try:
        profile = services.profile(path)
    except Exception as exc:  # noqa: BLE001 - a bad file must produce a message
        services.cleanup(path) if uploaded is not None else None
        return error(
            f"Could not read the dataset: {exc}",
            detail_fa=f"خواندن دیتاست ممکن نشد: {exc}",
        )

    existing = set(Dataset.objects.values_list("slug", flat=True))
    name = request.data.get("name") or path.stem
    dataset = Dataset.objects.create(
        name=name,
        slug=services.unique_slug(name, existing),
        source=source,
        path=str(path.relative_to(REPO_ROOT)) if str(path).startswith(str(REPO_ROOT)) else str(path),
        original_filename=original,
        size_bytes=path.stat().st_size,
        n_rows=profile["rows"],
        n_columns=profile["n_columns"],
        profile=profile,
    )
    return ok({"dataset": DatasetSerializer(dataset).data, "profile": profile})


@api_view(["POST", "GET"])
def dataset_profile(request):
    """Re-profile an existing dataset (or any path) without re-uploading."""
    dataset_id = request.data.get("dataset_id") or request.query_params.get("dataset_id")
    path = request.data.get("path") or request.query_params.get("path")

    if dataset_id:
        dataset = Dataset.objects.filter(id=dataset_id).first()
        if dataset is None:
            return error("Dataset not found", 404)
        target = dataset.absolute_path
    elif path:
        try:
            target = services.register_local_file(path)
        except (ValueError, FileNotFoundError) as exc:
            return error(str(exc))
        dataset = None
    else:
        return error("Provide `dataset_id` or `path`.")

    try:
        profile = services.profile(target)
    except Exception as exc:  # noqa: BLE001
        return error(
            f"Could not read the dataset: {exc}",
            detail_fa=f"خواندن دیتاست ممکن نشد: {exc}",
        )

    if dataset is not None:
        dataset.profile = profile
        dataset.n_rows = profile["rows"]
        dataset.n_columns = profile["n_columns"]
        dataset.save(update_fields=["profile", "n_rows", "n_columns", "updated_at"])
    return ok({"profile": profile, "dataset_id": dataset.id if dataset else None})


@api_view(["POST"])
def dataset_map(request):
    """Persist a column mapping as a data contract."""
    serializer = ColumnMappingSerializer(data=request.data)
    if not serializer.is_valid():
        return Response({"available": False, "errors": serializer.errors}, status=400)
    mapping = serializer.validated_data

    dataset = Dataset.objects.filter(id=mapping.get("dataset_id")).first()
    if dataset is None:
        return error(
            "Dataset not found - upload it first.",
            404,
            detail_fa="دیتاست پیدا نشد؛ ابتدا آن را بارگذاری کنید.",
        )

    try:
        contract = services.contract_from_mapping(mapping, dataset.path, dataset.name)
    except ValueError as exc:
        return error(str(exc), 400, detail_fa=f"فایل جانبی پیدا نشد: {exc}")
    dataset.mapping = {k: v for k, v in mapping.items() if k != "dataset_id"}

    if mapping.get("save_as_active", True):
        path = services.activate(contract)
        dataset.contract_path = str(path.relative_to(REPO_ROOT))
    dataset.save(update_fields=["mapping", "contract_path", "updated_at"])

    return ok(
        {
            "contract": contract.to_dict(),
            "contract_path": dataset.contract_path,
            "dataset": DatasetSerializer(dataset).data,
        }
    )


@api_view(["POST"])
def dataset_validate(request):
    """Full data-quality report for a mapping, before anything is trained."""
    dataset_id = request.data.get("dataset_id")
    dataset = Dataset.objects.filter(id=dataset_id).first() if dataset_id else None
    if dataset_id and dataset is None:
        # Falling back to the active contract here would answer a question the
        # caller did not ask - a report about a completely different dataset.
        return error(
            f"Dataset {dataset_id} not found",
            404,
            detail_fa=f"دیتاست با شناسه {dataset_id} پیدا نشد.",
        )

    if request.data.get("mapping"):
        if dataset is None:
            return error("Dataset not found", 404)
        serializer = ColumnMappingSerializer(data=request.data["mapping"])
        if not serializer.is_valid():
            return Response({"available": False, "errors": serializer.errors}, status=400)
        try:
            contract = services.contract_from_mapping(
                serializer.validated_data, dataset.path, dataset.name
            )
        except ValueError as exc:
            return error(str(exc), 400, detail_fa=f"فایل جانبی پیدا نشد: {exc}")
    elif dataset is not None and dataset.mapping:
        contract = services.contract_from_mapping(dataset.mapping, dataset.path, dataset.name)
    else:
        contract = DataContract.load()

    try:
        report = services.validate_contract(contract)
    except Exception as exc:  # noqa: BLE001 - report the failure, do not 500
        return error(
            f"Validation failed: {exc}",
            detail_fa=f"اعتبارسنجی داده با خطا مواجه شد: {exc}",
        )

    if dataset is not None:
        dataset.validation = report
        dataset.save(update_fields=["validation", "updated_at"])
    return ok(report)


@api_view(["GET"])
def contract_detail(_request):
    return ok(_active_contract())


def _active_contract() -> dict:
    contract = DataContract.load()
    return {
        "contract": contract.to_dict(),
        "source": str(ACTIVE_CONTRACT_FILE.relative_to(REPO_ROOT))
        if ACTIVE_CONTRACT_FILE.exists()
        else "config/data_contract.example.yaml",
        "is_custom": ACTIVE_CONTRACT_FILE.exists(),
        "levels": contract.available_levels,
    }
