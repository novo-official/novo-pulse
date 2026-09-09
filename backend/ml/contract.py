"""The data contract - the single source of truth for "what is this dataset?".

The whole pipeline reads column roles from here. Swapping the synthetic dataset
for the real competition dataset is therefore a configuration change, not a code
change: point the contract at the new file, remap the roles, done.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import yaml

from .paths import ACTIVE_CONTRACT_FILE, PROFILES_FILE, REPO_ROOT

# Canonical internal column names. The pipeline renames the user's columns to
# these once, at load time, and never looks at the original names again.
TS = "ds"
ENTITY = "entity_id"
TARGET = "y"
DESTINATION = "destination_id"
CATEGORY = "category_id"
MARKET = "market_id"
MARKET_VALUE = "market"

LEVELS = ("listing", "destination", "category", "market")
LEVEL_COLUMN = {
    "listing": ENTITY,
    "destination": DESTINATION,
    "category": CATEGORY,
    "market": MARKET,
}

FREQ_ALIASES = {"D": "D", "W": "W", "M": "MS", "MS": "MS", "H": "h", "h": "h"}
# Seasonal period (in steps) for each supported frequency.
FREQ_SEASONALITY = {"D": 7, "W": 52, "MS": 12, "h": 24}


@dataclass
class JoinSpec:
    """A side table merged onto the main frame."""

    path: str
    on: str | list[str]

    @property
    def keys(self) -> list[str]:
        return [self.on] if isinstance(self.on, str) else list(self.on)


@dataclass
class EvaluationSpec:
    primary_metric: str = "wape"
    secondary_metrics: list[str] = field(
        default_factory=lambda: ["mae", "rmse", "smape", "mape", "r2"]
    )
    horizons: list[int] = field(default_factory=lambda: [7, 14, 30, 60, 90])
    horizon_buckets: list[list[int]] = field(
        default_factory=lambda: [[1, 7], [8, 14], [15, 30], [31, 60], [61, 90]]
    )
    n_folds: int = 3
    step: int | None = None
    quantiles: list[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])

    @property
    def max_horizon(self) -> int:
        return max(self.horizons) if self.horizons else 30


@dataclass
class TargetOptions:
    non_negative: bool = True
    integer: bool = False
    log1p_transform: str | bool = "auto"
    censoring_column: str | None = None
    censoring_enabled: bool = False


@dataclass
class DataContract:
    """Column roles + evaluation setup for one dataset."""

    name: str = "dataset"
    path: str | None = None
    joins: list[JoinSpec] = field(default_factory=list)

    timestamp: str = "date"
    target: str = "y"
    entity_id: str | None = None
    frequency: str | None = "D"
    aggregation: str = "sum"
    # "auto" detects Jalali vs Gregorian from the values; force it when a
    # dataset mixes calendars or the year range is ambiguous.
    calendar: str = "auto"

    destination: str | None = None
    category: str | None = None

    # Display-only: maps a key column to a human-readable label column. These
    # columns are carried through the adapter but are never model features.
    labels: dict[str, str] = field(default_factory=dict)

    future_features: list[str] = field(default_factory=list)
    historical_features: list[str] = field(default_factory=list)
    static_features: list[str] = field(default_factory=list)
    ignored: list[str] = field(default_factory=list)

    target_options: TargetOptions = field(default_factory=TargetOptions)
    evaluation: EvaluationSpec = field(default_factory=EvaluationSpec)

    # ------------------------------------------------------------------ load
    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "DataContract":
        dataset = raw.get("dataset") or {}
        schema = raw.get("schema") or {}
        hierarchy = raw.get("hierarchy") or {}
        features = raw.get("features") or {}
        target_opts = raw.get("target_options") or {}
        evaluation = raw.get("evaluation") or {}
        cv = evaluation.get("cv") or {}

        return cls(
            name=dataset.get("name", "dataset"),
            path=dataset.get("path"),
            joins=[_join_spec(j) for j in (dataset.get("joins") or [])],
            timestamp=schema.get("timestamp", "date"),
            target=schema.get("target", "y"),
            entity_id=schema.get("entity_id"),
            frequency=schema.get("frequency"),
            aggregation=schema.get("aggregation", "sum"),
            calendar=schema.get("calendar", "auto"),
            destination=hierarchy.get("destination"),
            category=hierarchy.get("category"),
            labels=dict(raw.get("labels") or {}),
            future_features=list(features.get("future") or []),
            historical_features=list(features.get("historical") or []),
            static_features=list(features.get("static") or []),
            ignored=list(features.get("ignored") or []),
            target_options=TargetOptions(
                non_negative=target_opts.get("non_negative", True),
                integer=target_opts.get("integer", False),
                log1p_transform=target_opts.get("log1p_transform", "auto"),
                censoring_column=target_opts.get("censoring_column"),
                censoring_enabled=target_opts.get("censoring_enabled", False),
            ),
            evaluation=EvaluationSpec(
                primary_metric=evaluation.get("primary_metric", "wape"),
                secondary_metrics=list(
                    evaluation.get("secondary_metrics")
                    or ["mae", "rmse", "smape", "mape", "r2"]
                ),
                horizons=list(evaluation.get("horizons") or [7, 14, 30, 60, 90]),
                horizon_buckets=[
                    list(b)
                    for b in (
                        evaluation.get("horizon_buckets")
                        or [[1, 7], [8, 14], [15, 30], [31, 60], [61, 90]]
                    )
                ],
                n_folds=int(cv.get("n_folds", 3)),
                step=cv.get("step"),
                quantiles=list(evaluation.get("quantiles") or [0.1, 0.5, 0.9]),
            ),
        )

    @classmethod
    def load(cls, path: str | Path | None = None) -> "DataContract":
        """Load a contract - the active one written by the Data Lab by default.

        There is no shipped example contract: it described a synthetic dataset
        that no longer exists, and defaulting to it would have meant loading
        fabricated data whenever nothing real was configured.
        """
        if path is None:
            path = ACTIVE_CONTRACT_FILE
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"No data contract at {path}. Map a dataset first, or pass a path."
            )
        if not path.is_absolute():
            path = REPO_ROOT / path
        with path.open("r", encoding="utf-8") as handle:
            return cls.from_dict(yaml.safe_load(handle) or {})

    # ------------------------------------------------------------------ dump
    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": {
                "name": self.name,
                "path": self.path,
                "joins": [dataclasses.asdict(j) for j in self.joins],
            },
            "schema": {
                "timestamp": self.timestamp,
                "target": self.target,
                "entity_id": self.entity_id,
                "frequency": self.frequency,
                "aggregation": self.aggregation,
                "calendar": self.calendar,
            },
            "hierarchy": {"destination": self.destination, "category": self.category},
            "labels": self.labels,
            "features": {
                "future": self.future_features,
                "historical": self.historical_features,
                "static": self.static_features,
                "ignored": self.ignored,
            },
            "target_options": dataclasses.asdict(self.target_options),
            "evaluation": {
                "primary_metric": self.evaluation.primary_metric,
                "secondary_metrics": self.evaluation.secondary_metrics,
                "horizons": self.evaluation.horizons,
                "horizon_buckets": self.evaluation.horizon_buckets,
                "quantiles": self.evaluation.quantiles,
                "cv": {"n_folds": self.evaluation.n_folds, "step": self.evaluation.step},
            },
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        if not path.is_absolute():
            path = REPO_ROOT / path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(self.to_dict(), handle, sort_keys=False, allow_unicode=True)
        return path

    # ------------------------------------------------------------- helpers
    @property
    def pandas_freq(self) -> str:
        return FREQ_ALIASES.get(self.frequency or "D", "D")

    @property
    def seasonal_period(self) -> int:
        return FREQ_SEASONALITY.get(self.pandas_freq, 7)

    @property
    def available_levels(self) -> list[str]:
        """Hierarchy levels this dataset can actually serve."""
        levels = []
        if self.entity_id:
            levels.append("listing")
        if self.destination:
            levels.append("destination")
        if self.category:
            levels.append("category")
        levels.append("market")
        return levels

    def declared_columns(self) -> list[str]:
        """Every source column the contract references, de-duplicated."""
        cols: list[str] = [self.timestamp, self.target]
        for optional in (self.entity_id, self.destination, self.category):
            if optional:
                cols.append(optional)
        cols.extend(self.future_features)
        cols.extend(self.historical_features)
        cols.extend(self.static_features)
        cols.extend(self.labels.keys())
        cols.extend(self.labels.values())
        if self.target_options.censoring_column:
            cols.append(self.target_options.censoring_column)
        return _unique(cols)

    def with_overrides(self, **overrides: Any) -> "DataContract":
        """Return a copy with shallow field overrides (used by CLI flags)."""
        clone = dataclasses.replace(self)
        for key, value in overrides.items():
            if value is None:
                continue
            if key == "primary_metric":
                clone.evaluation.primary_metric = value
            elif key == "horizon":
                clone.evaluation.horizons = sorted(
                    {*clone.evaluation.horizons, int(value)}
                )
            elif hasattr(clone, key):
                setattr(clone, key, value)
        return clone


def _join_spec(raw: dict[Any, Any]) -> JoinSpec:
    """Build a JoinSpec, tolerating YAML 1.1 turning the bare key `on` into True."""
    normalised = {("on" if key is True else str(key)): value for key, value in raw.items()}
    return JoinSpec(path=normalised["path"], on=normalised["on"])


def _unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


# ---------------------------------------------------------------- profiles
def load_profile(name: str = "demo") -> dict[str, Any]:
    """Load a training profile (demo / competition / full)."""
    with PROFILES_FILE.open("r", encoding="utf-8") as handle:
        profiles = yaml.safe_load(handle) or {}
    if name not in profiles:
        available = ", ".join(sorted(profiles))
        raise KeyError(f"Unknown training profile '{name}'. Available: {available}")
    profile = dict(profiles[name])
    profile["name"] = name
    return profile


def list_profiles() -> dict[str, str]:
    with PROFILES_FILE.open("r", encoding="utf-8") as handle:
        profiles = yaml.safe_load(handle) or {}
    return {k: v.get("description", "") for k, v in profiles.items()}
