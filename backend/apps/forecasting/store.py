"""Artefact store - the read path for everything the dashboard shows.

Training writes parquet/JSON into `runs/<run_id>/`. The API reads it back
through this module, which keeps a small in-process cache keyed by
(run_id, artefact, mtime) so repeated dashboard requests do not re-read parquet.

Serving artefacts rather than a database of forecast rows is deliberate: it
keeps SQLite fast, and it means a precomputed demo run can be committed and
served without any training happening at request time.
"""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

import pandas as pd

from ml.contract import ENTITY
from ml.paths import REPO_ROOT

log = logging.getLogger(__name__)

_LOCK = threading.Lock()
_FRAME_CACHE: dict[tuple[str, float], pd.DataFrame] = {}
_JSON_CACHE: dict[tuple[str, float], Any] = {}
_MAX_CACHE_ENTRIES = 24


class ArtifactStore:
    """Read-only view over one training run's directory."""

    def __init__(self, run_dir: str | Path, run_id: str = ""):
        path = Path(run_dir)
        self.directory = path if path.is_absolute() else (REPO_ROOT / path)
        self.run_id = run_id or self.directory.name

    # ------------------------------------------------------------- exists
    def exists(self) -> bool:
        return self.directory.is_dir() and (self.directory / "metadata.json").exists()

    def has(self, name: str) -> bool:
        return (self.directory / f"{name}.parquet").exists() or (
            self.directory / f"{name}.json"
        ).exists()

    # --------------------------------------------------------------- read
    def frame(self, name: str) -> pd.DataFrame:
        path = self.directory / f"{name}.parquet"
        if not path.exists():
            return pd.DataFrame()
        return _cached_frame(path)

    def json(self, name: str, default: Any = None) -> Any:
        path = self.directory / f"{name}.json"
        if not path.exists():
            return {} if default is None else default
        return _cached_json(path)

    # ------------------------------------------------------------ helpers
    @property
    def metadata(self) -> dict[str, Any]:
        return self.json("metadata")

    @property
    def metrics(self) -> dict[str, Any]:
        return self.json("metrics")

    @property
    def insights(self) -> dict[str, Any]:
        return self.json("insights")

    @property
    def labels(self) -> dict[str, str]:
        return self.json("labels", {}) or {}

    def label_for(self, key: str) -> str:
        return self.labels.get(str(key), str(key))

    def levels(self) -> list[str]:
        forecast = self.frame("forecast")
        if forecast.empty or "level" not in forecast.columns:
            return ["listing"]
        order = ["listing", "destination", "category", "market"]
        present = set(forecast["level"].unique())
        return [level for level in order if level in present]

    def level_frame(self, name: str, level: str) -> pd.DataFrame:
        frame = self.frame(name)
        if frame.empty:
            return frame
        if "level" not in frame.columns:
            return frame if level == "listing" else frame.iloc[0:0]
        return frame[frame["level"] == level]

    def members(self, level: str) -> list[dict[str, str]]:
        """Selectable entities at a level, largest forecast first."""
        forecast = self.level_frame("forecast", level)
        if forecast.empty:
            return []
        totals = forecast.groupby(ENTITY)["forecast"].sum().sort_values(ascending=False)
        return [
            {"id": str(key), "label": self.label_for(key), "forecast_total": round(float(value), 2)}
            for key, value in totals.items()
        ]


def _trim(cache: dict) -> None:
    while len(cache) > _MAX_CACHE_ENTRIES:
        cache.pop(next(iter(cache)))


def _cached_frame(path: Path) -> pd.DataFrame:
    key = (str(path), path.stat().st_mtime)
    with _LOCK:
        cached = _FRAME_CACHE.get(key)
        if cached is not None:
            return cached
    frame = pd.read_parquet(path)
    if "ds" in frame.columns:
        frame["ds"] = pd.to_datetime(frame["ds"])
    with _LOCK:
        _FRAME_CACHE[key] = frame
        _trim(_FRAME_CACHE)
    return frame


def _cached_json(path: Path) -> Any:
    key = (str(path), path.stat().st_mtime)
    with _LOCK:
        cached = _JSON_CACHE.get(key)
        if cached is not None:
            return cached
    payload = json.loads(path.read_text(encoding="utf-8"))
    with _LOCK:
        _JSON_CACHE[key] = payload
        _trim(_JSON_CACHE)
    return payload


def clear_cache() -> None:
    with _LOCK:
        _FRAME_CACHE.clear()
        _JSON_CACHE.clear()


# --------------------------------------------------------------------------
def active_run():
    """The run the dashboard should show: the newest successful one, or nothing.

    There is no synthetic fallback. If no model has been trained on real data,
    every endpoint reports that plainly rather than serving a fabricated run.
    """
    from apps.experiments.models import TrainingRun

    run = (
        TrainingRun.objects.filter(status=TrainingRun.Status.SUCCEEDED)
        .order_by("-finished_at", "-created_at")
        .first()
    )
    if run and ArtifactStore(run.run_dir, run.run_id).exists():
        return run
    return None


def active_store() -> ArtifactStore | None:
    run = active_run()
    return ArtifactStore(run.run_dir, run.run_id) if run is not None else None


def store_for(run_id: str | None) -> ArtifactStore | None:
    if not run_id:
        return active_store()
    from apps.experiments.models import TrainingRun

    run = TrainingRun.objects.filter(run_id=run_id).first()
    if run is None:
        return None
    store = ArtifactStore(run.run_dir, run.run_id)
    return store if store.exists() else None
