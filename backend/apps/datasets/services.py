"""Data Lab services: upload, profile, map, validate."""
from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

from django.utils.text import slugify

from ml.contract import DataContract
from ml.data.adapter import DataAdapter
from ml.data.profiler import profile_dataset
from ml.data.validator import DataValidator
from ml.paths import ACTIVE_CONTRACT_FILE, REPO_ROOT, UPLOAD_DIR

ALLOWED_SUFFIXES = {".csv", ".tsv", ".parquet", ".pq", ".xlsx", ".xls", ".json"}
MAX_UPLOAD_BYTES = 2 * 1024**3


def safe_filename(name: str) -> str:
    """Strip any path component and anything that is not filename-safe."""
    base = Path(str(name)).name
    stem = Path(base).stem
    suffix = Path(base).suffix.lower()
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-") or "dataset"
    return f"{cleaned}{suffix}"


def store_upload(uploaded_file) -> Path:
    """Persist an uploaded file under data/uploads/ and return its path."""
    filename = safe_filename(uploaded_file.name)
    suffix = Path(filename).suffix
    if suffix not in ALLOWED_SUFFIXES:
        raise ValueError(
            f"Unsupported file type '{suffix}'. Allowed: {', '.join(sorted(ALLOWED_SUFFIXES))}"
        )
    if uploaded_file.size and uploaded_file.size > MAX_UPLOAD_BYTES:
        raise ValueError("File exceeds the 2 GB upload limit")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    target = UPLOAD_DIR / filename
    counter = 1
    while target.exists():
        target = UPLOAD_DIR / f"{Path(filename).stem}_{counter}{suffix}"
        counter += 1

    with target.open("wb") as handle:
        for chunk in uploaded_file.chunks():
            handle.write(chunk)
    return target


def register_local_file(path: str | Path) -> Path:
    """Adopt a file already sitting in data/raw/ (the competition-day path)."""
    source = Path(path)
    if not source.is_absolute():
        source = REPO_ROOT / source
    source = source.resolve()
    # Never read outside the repository. A string prefix check is not enough:
    # "/srv/novo-pulse-backup/secrets.csv" starts with "/srv/novo-pulse", so
    # compare path components instead.
    if not source.is_relative_to(REPO_ROOT.resolve()):
        raise ValueError("Path must be inside the project directory")
    if not source.exists():
        raise FileNotFoundError(f"No such file: {path}")
    if source.suffix.lower() not in ALLOWED_SUFFIXES:
        raise ValueError(f"Unsupported file type '{source.suffix}'")
    return source


def profile(path: str | Path) -> dict[str, Any]:
    return profile_dataset(path)


def resolve_joins(joins) -> list[dict]:
    """Turn the Data Lab's join rows into the `{path, on}` a contract wants.

    A side table is normally another uploaded dataset, referenced by id; a
    literal path is accepted too, but only from inside the project. Resolution
    lives here rather than in a view so that no caller can build a contract
    that quietly drops its joins.
    """
    from .models import Dataset

    resolved = []
    for join in joins or []:
        keys = [k for k in (join.get("on") or []) if k]
        if not keys:
            continue
        path = join.get("path")
        if join.get("dataset_id"):
            side = Dataset.objects.filter(id=join["dataset_id"]).first()
            if side is None:
                raise ValueError(f"Join dataset {join['dataset_id']} not found")
            path = side.path
        elif path:
            path = str(register_local_file(path))
        if not path:
            continue
        resolved.append({"path": path, "on": keys})
    return resolved


def contract_from_mapping(
    mapping: dict[str, Any], dataset_path: str, name: str = "uploaded_dataset"
) -> DataContract:
    """Build a DataContract from the Data Lab form."""
    horizons = sorted({int(h) for h in (mapping.get("horizons") or [])}) or [7, 14, 30, 60, 90]
    buckets = _buckets_for(horizons)
    raw = {
        "dataset": {
            "name": name,
            "path": dataset_path,
            # The competition ships demand, accommodation/destination and
            # booking data as separate files. Each join is merged onto the main
            # frame before the panel is built.
            "joins": resolve_joins(mapping.get("joins")),
        },
        "schema": {
            "timestamp": mapping["timestamp"],
            "target": mapping.get("target") or None,
            "entity_id": mapping.get("entity_id") or None,
            "frequency": mapping.get("frequency") or None,
            "aggregation": mapping.get("aggregation") or "sum",
            "calendar": mapping.get("calendar") or "auto",
        },
        "hierarchy": {
            "destination": mapping.get("destination") or None,
            "category": mapping.get("category") or None,
        },
        "features": {
            "future": list(mapping.get("future_features") or []),
            "historical": list(mapping.get("historical_features") or []),
            "static": list(mapping.get("static_features") or []),
            "ignored": list(mapping.get("ignored") or []),
        },
        "target_options": {
            "non_negative": bool(mapping.get("non_negative", True)),
            "integer": bool(mapping.get("integer", False)),
            "log1p_transform": "auto",
            "censoring_column": mapping.get("censoring_column") or None,
            "censoring_enabled": bool(mapping.get("censoring_column")),
        },
        "evaluation": {
            "primary_metric": mapping.get("primary_metric") or "wape",
            "secondary_metrics": ["mae", "rmse", "smape", "mape", "r2"],
            "horizons": horizons,
            "horizon_buckets": buckets,
            "quantiles": [0.1, 0.5, 0.9],
            "cv": {"n_folds": 3, "step": None},
        },
    }
    return DataContract.from_dict(raw)


def _buckets_for(horizons: list[int]) -> list[list[int]]:
    buckets: list[list[int]] = []
    start = 1
    for horizon in horizons:
        if horizon >= start:
            buckets.append([start, horizon])
            start = horizon + 1
    return buckets or [[1, 7], [8, 30], [31, 90]]


def validate_contract(contract: DataContract) -> dict[str, Any]:
    """Run the adapter + validator so the user sees real problems before training."""
    panel = DataAdapter(contract).build()
    report = DataValidator(contract).validate(panel.frame)
    report["panel"] = panel.summary()
    report["adapter_notes"] = panel.notes

    # A silent calendar conversion is dangerous: if the guess is wrong, every
    # date is wrong. Raise it to a finding so it cannot be scrolled past.
    calendar_note = next((n for n in panel.notes if "Jalali" in n), None)
    if calendar_note:
        report["findings"].insert(
            0,
            {
                "code": "calendar_converted",
                "severity": "medium",
                "title": "Dates were read as Jalali and converted to Gregorian",
                "detail": calendar_note,
                "recommendation": (
                    "Check the converted range looks right. If the dataset is actually "
                    "Gregorian, set the calendar explicitly instead of leaving it on auto."
                ),
                "context": {"calendar": "jalali"},
            },
        )
        report["n_findings"] = len(report["findings"])
        report["by_severity"]["medium"] = report["by_severity"].get("medium", 0) + 1
    return report


def activate(contract: DataContract) -> Path:
    """Persist the contract as the active one used by the next training run."""
    return contract.save(ACTIVE_CONTRACT_FILE)


def unique_slug(name: str, existing: set[str]) -> str:
    base = slugify(name) or "dataset"
    slug, counter = base, 1
    while slug in existing:
        slug = f"{base}-{counter}"
        counter += 1
    return slug


def cleanup(path: Path) -> None:
    if path.exists() and path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path, ignore_errors=True)
