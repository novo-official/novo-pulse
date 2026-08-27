"""Model registry.

Profiles name models as strings (`models: [lightgbm, chronos, ...]`); the
registry turns those names into instances, applying profile hyper-parameters.
Optional models that cannot be constructed are reported, never raised.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .base import ForecastModel
from .baselines import (
    HistoricalMeanModel,
    MovingAverageModel,
    NaiveModel,
    SeasonalMeanModel,
    SeasonalNaiveModel,
)
from .chronos_model import ChronosModel, chronos_available
from .gbdt import CatBoostModel, LightGBMModel
from .neural_model import NBEATSxModel, NHITSModel, neuralforecast_available


@dataclass(frozen=True)
class ModelEntry:
    name: str
    label: str
    factory: Callable[..., ForecastModel]
    kind: str                # "baseline" | "ml" | "deep" | "foundation"
    optional: bool = False
    availability: Callable[[], tuple[bool, str]] | None = None

    def build(self, **params: Any) -> ForecastModel:
        return self.factory(**params)

    def is_available(self) -> tuple[bool, str]:
        return self.availability() if self.availability else (True, "ready")


REGISTRY: dict[str, ModelEntry] = {}


def register(entry: ModelEntry) -> None:
    REGISTRY[entry.name] = entry


for season in (7, 14, 30, 52, 365):
    register(
        ModelEntry(
            name=f"seasonal_naive_{season}",
            label=f"Seasonal Naive ({season})",
            factory=(lambda s=season, **kw: SeasonalNaiveModel(season=s, **kw)),
            kind="baseline",
        )
    )

register(ModelEntry("naive", "Naive (last value)", NaiveModel, "baseline"))
register(ModelEntry("moving_average", "Moving Average", MovingAverageModel, "baseline"))
register(ModelEntry("historical_mean", "Historical Mean", HistoricalMeanModel, "baseline"))
register(ModelEntry("seasonal_mean", "Seasonal Mean", SeasonalMeanModel, "baseline"))
register(ModelEntry("lightgbm", "LightGBM", LightGBMModel, "ml"))
register(ModelEntry("catboost", "CatBoost", CatBoostModel, "ml"))
register(
    ModelEntry(
        "chronos",
        "Chronos-2 (zero-shot)",
        ChronosModel,
        "foundation",
        optional=True,
        availability=chronos_available,
    )
)
register(
    ModelEntry(
        "nhits", "NHITS", NHITSModel, "deep", optional=True, availability=neuralforecast_available
    )
)
register(
    ModelEntry(
        "nbeatsx",
        "NBEATSx",
        NBEATSxModel,
        "deep",
        optional=True,
        availability=neuralforecast_available,
    )
)

BASELINE_NAMES = {name for name, entry in REGISTRY.items() if entry.kind == "baseline"}


def get_entry(name: str) -> ModelEntry:
    if name not in REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available: {sorted(REGISTRY)}")
    return REGISTRY[name]


def is_baseline(name: str) -> bool:
    return name in BASELINE_NAMES


def build_models(
    names: list[str], profile: dict[str, Any], season: int = 7
) -> tuple[dict[str, ForecastModel], list[dict[str, str]]]:
    """Instantiate every requested model, collecting skips instead of failing."""
    models: dict[str, ForecastModel] = {}
    skipped: list[dict[str, str]] = []

    for name in names:
        entry = REGISTRY.get(name)
        if entry is None:
            skipped.append({"model": name, "reason": "not in the registry"})
            continue
        available, reason = entry.is_available()
        if not available:
            skipped.append({"model": name, "reason": reason})
            continue
        params = _params_for(name, profile, season)
        try:
            models[name] = entry.build(**params)
        except Exception as exc:  # noqa: BLE001 - construction must never be fatal
            skipped.append({"model": name, "reason": f"could not be constructed: {exc}"})
    return models, skipped


def _params_for(name: str, profile: dict[str, Any], season: int) -> dict[str, Any]:
    if name == "lightgbm":
        return dict(profile.get("lightgbm") or {})
    if name == "catboost":
        return dict(profile.get("catboost") or {})
    if name in {"nhits", "nbeatsx"}:
        return dict(profile.get("neural") or {})
    if name == "moving_average":
        return {"window": max(season * 4, 7)}
    if name == "seasonal_mean":
        return {"season": season}
    return {}


def describe_registry() -> list[dict[str, Any]]:
    """What the /api/v1/models/ endpoint reports about model availability."""
    out = []
    for name, entry in sorted(REGISTRY.items()):
        available, reason = entry.is_available()
        out.append(
            {
                "name": name,
                "label": entry.label,
                "kind": entry.kind,
                "optional": entry.optional,
                "available": available,
                "status": reason,
            }
        )
    return out
