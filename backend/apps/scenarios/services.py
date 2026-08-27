"""What-if simulation.

A scenario re-runs the *same* fitted champion model against a modified copy of
the known-future covariates. Nothing is refitted and nothing is extrapolated by
hand, so the delta between baseline and scenario is exactly what the model
believes about the change.

Only covariates the model actually uses can be adjusted - the API refuses
anything else rather than silently returning an unchanged forecast.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml.contract import ENTITY, DataContract
from ml.data.adapter import DataAdapter
from ml.features.calendar import is_event_like, is_price_like
from ml.features.engineering import FeatureConfig, FeatureEngine
from ml.features.tensor import build_tensor
from ml.hierarchy import aggregate_bottom_up, entity_map
from ml.models.base import PredictContext

log = logging.getLogger(__name__)

_CACHE: dict[str, "ScenarioEngine"] = {}

ADJUSTMENT_KINDS = {
    "price": {"mode": "relative", "unit": "percent", "label_fa": "قیمت"},
    "capacity": {"mode": "relative", "unit": "percent", "label_fa": "ظرفیت"},
    "availability": {"mode": "relative", "unit": "percent", "label_fa": "موجودی"},
    "promotion": {"mode": "absolute", "unit": "flag", "label_fa": "کمپین تخفیف"},
    "holiday": {"mode": "absolute", "unit": "flag", "label_fa": "تعطیلات / رویداد"},
    "generic": {"mode": "relative", "unit": "percent", "label_fa": "متغیر"},
}


# Covariates that are facts about the calendar, not levers anyone can pull.
# Offering a "make it a weekend" switch would be a nonsense control.
NON_ACTIONABLE = ("is_weekend", "weekend", "day_of_week", "dayofweek", "month", "quarter", "year")


def is_actionable(column: str) -> bool:
    lowered = column.lower()
    return not any(token == lowered or token in lowered for token in NON_ACTIONABLE)


def classify(column: str) -> str:
    lowered = column.lower()
    if is_price_like(column):
        return "price"
    if is_event_like(column):
        return "holiday"
    if "promo" in lowered or "discount" in lowered or "campaign" in lowered:
        return "promotion"
    if "availab" in lowered:
        return "availability"
    if "capacity" in lowered:
        return "capacity"
    return "generic"


@dataclass
class ScenarioEngine:
    """Holds the artefacts a scenario needs: panel, engine and champion model."""

    run_id: str
    engine: FeatureEngine
    model: Any
    contract: DataContract
    mapping: pd.DataFrame
    labels: dict[str, str]
    horizon: int
    quantiles: tuple[float, ...]

    @property
    def adjustable(self) -> list[dict[str, Any]]:
        """Known-future covariates the fitted model genuinely consumes."""
        used = set(getattr(self.model, "feature_names", []) or [])
        if not used:
            used = set(self.engine.feature_names)
        out = []
        for column in self.engine.tensor.future.keys():
            if f"fut_{column}" not in used or not is_actionable(column):
                continue
            kind = classify(column)
            matrix = self.engine.tensor.future[column]
            window = matrix[:, self.engine.tensor.n_observed :]
            finite = window[np.isfinite(window)]
            out.append(
                {
                    "column": column,
                    "kind": kind,
                    **ADJUSTMENT_KINDS[kind],
                    "current_mean": round(float(finite.mean()), 4) if finite.size else None,
                    "is_binary": bool(finite.size and np.all(np.isin(finite, (0.0, 1.0)))),
                }
            )
        return out


def load_engine(run) -> ScenarioEngine:
    """Rebuild the forecasting context for a completed run (cached)."""
    cached = _CACHE.get(run.run_id)
    if cached is not None:
        return cached

    import joblib

    from ml.paths import REPO_ROOT

    run_dir = Path(run.run_dir)
    if not run_dir.is_absolute():
        run_dir = REPO_ROOT / run_dir

    contract = DataContract.load(run_dir / "config.yaml")
    metadata = _read_json(run_dir / "metadata.json")
    champion = metadata.get("champion")
    horizon = int(metadata.get("horizon") or 30)

    model_paths = metadata.get("models") or {}
    model = None
    if champion in model_paths:
        model = joblib.load(REPO_ROOT / model_paths[champion])
    elif model_paths:
        # The ensemble is not persisted as one file; fall back to the best member.
        name = next(iter(model_paths))
        model = joblib.load(REPO_ROOT / model_paths[name])
        champion = name
    if model is None:
        raise RuntimeError("No persisted model available for scenario simulation")

    panel = DataAdapter(contract).build()
    tensor = build_tensor(
        panel.frame,
        freq=panel.frequency,
        horizon=horizon,
        future_features=panel.future_features,
        past_features=panel.historical_features,
        static_features=panel.static_features,
    )
    engine = FeatureEngine(
        tensor, FeatureConfig.for_frequency(panel.frequency, max_horizon=horizon)
    ).prepare()

    result = ScenarioEngine(
        run_id=run.run_id,
        engine=engine,
        model=model,
        contract=contract,
        mapping=entity_map(panel.frame),
        labels=_read_json(run_dir / "labels.json"),
        horizon=horizon,
        quantiles=tuple(contract.evaluation.quantiles),
    )
    _CACHE.clear()
    _CACHE[run.run_id] = result
    return result


def _read_json(path: Path) -> dict:
    import json

    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def clear_cache() -> None:
    _CACHE.clear()


# --------------------------------------------------------------------------
def simulate(
    scenario: ScenarioEngine,
    adjustments: list[dict[str, Any]],
    level: str = "destination",
    entity_id: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    horizon: int | None = None,
) -> dict[str, Any]:
    """Run baseline vs. scenario and return both series plus the delta."""
    engine = scenario.engine
    tensor = engine.tensor
    horizon = min(int(horizon or scenario.horizon), scenario.horizon)
    origin = tensor.origin_index

    entity_positions = _entity_positions(scenario, level, entity_id)
    baseline_frame = engine.build_inference(origin, horizon)
    baseline_point = scenario.model.predict(
        PredictContext(engine=engine, origin_idx=origin, horizon=horizon, frame=baseline_frame)
    )

    # ---- apply the adjustments to a private copy of the future covariates ---
    scenario_tensor = tensor.copy_for_scenario()
    date_mask = _date_mask(scenario_tensor, origin, horizon, start_date, end_date)
    applied: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    adjustable = {item["column"]: item for item in scenario.adjustable}

    for adjustment in adjustments:
        column = adjustment.get("column")
        spec = adjustable.get(column)
        if spec is None:
            rejected.append(
                {
                    "column": column,
                    "reason": "the fitted model does not use this covariate, so changing it would not affect the forecast",
                }
            )
            continue
        matrix = scenario_tensor.future[column]
        rows = entity_positions[:, None]
        before = float(np.nanmean(matrix[np.ix_(entity_positions, np.where(date_mask)[0])]))

        if adjustment.get("mode", spec["mode"]) == "absolute" or "value" in adjustment:
            value = float(adjustment.get("value", 1.0))
            matrix[rows, date_mask] = value
            change_description = f"set to {value}"
        else:
            pct = float(adjustment.get("change_pct", 0.0))
            matrix[rows, date_mask] *= 1.0 + pct / 100.0
            change_description = f"{pct:+.1f}%"

        after = float(np.nanmean(matrix[np.ix_(entity_positions, np.where(date_mask)[0])]))
        applied.append(
            {
                "column": column,
                "kind": spec["kind"],
                "label_fa": spec["label_fa"],
                "change": change_description,
                "mean_before": round(before, 4),
                "mean_after": round(after, 4),
            }
        )

    scenario_engine = FeatureEngine(scenario_tensor, engine.config).prepare()
    scenario_frame = scenario_engine.build_inference(origin, horizon)
    scenario_point = scenario.model.predict(
        PredictContext(
            engine=scenario_engine, origin_idx=origin, horizon=horizon, frame=scenario_frame
        )
    )

    baseline = _to_frame(baseline_frame, baseline_point)
    modified = _to_frame(scenario_frame, scenario_point)

    if level != "listing":
        baseline = aggregate_bottom_up(baseline, scenario.mapping, level)
        modified = aggregate_bottom_up(modified, scenario.mapping, level)
    if entity_id:
        baseline = baseline[baseline[ENTITY].astype(str) == str(entity_id)]
        modified = modified[modified[ENTITY].astype(str) == str(entity_id)]

    baseline_series = baseline.groupby("ds", as_index=False)["forecast"].sum().sort_values("ds")
    scenario_series = modified.groupby("ds", as_index=False)["forecast"].sum().sort_values("ds")

    merged = baseline_series.merge(scenario_series, on="ds", suffixes=("_baseline", "_scenario"))
    merged["delta"] = merged["forecast_scenario"] - merged["forecast_baseline"]

    baseline_total = float(baseline_series["forecast"].sum())
    scenario_total = float(scenario_series["forecast"].sum())
    impact_pct = (
        round((scenario_total - baseline_total) / baseline_total * 100, 2)
        if baseline_total > 0
        else None
    )

    return {
        "level": level,
        "entity_id": entity_id,
        "label": scenario.labels.get(str(entity_id), entity_id) if entity_id else "کل بازار",
        "horizon": horizon,
        "window": {
            "start": str(scenario_tensor.dates[origin + 1].date()),
            "end": str(scenario_tensor.dates[origin + horizon].date()),
            "adjusted_start": start_date,
            "adjusted_end": end_date,
        },
        "baseline_total": round(baseline_total, 2),
        "scenario_total": round(scenario_total, 2),
        "delta_total": round(scenario_total - baseline_total, 2),
        "impact_pct": impact_pct,
        "applied": applied,
        "rejected": rejected,
        "model": getattr(scenario.model, "name", "unknown"),
        "series": [
            {
                "ds": pd.Timestamp(row["ds"]).strftime("%Y-%m-%d"),
                "baseline": round(float(row["forecast_baseline"]), 3),
                "scenario": round(float(row["forecast_scenario"]), 3),
                "delta": round(float(row["delta"]), 3),
            }
            for row in merged.to_dict(orient="records")
        ],
        "note_fa": (
            "این نتیجه بازتاب رفتار مدل نسبت به تغییر متغیرهاست و رابطه علّی قطعی را نشان نمی‌دهد."
        ),
    }


def _entity_positions(
    scenario: ScenarioEngine, level: str, entity_id: str | None
) -> np.ndarray:
    entities = scenario.engine.tensor.entities
    if not entity_id or level == "market":
        return np.arange(len(entities))
    if level == "listing":
        matches = np.where(entities.astype(str) == str(entity_id))[0]
        return matches if matches.size else np.arange(len(entities))

    column = {"destination": "destination_id", "category": "category_id"}.get(level)
    mapping = scenario.mapping
    if column is None or column not in mapping.columns:
        return np.arange(len(entities))
    members = set(
        mapping.loc[mapping[column].astype(str) == str(entity_id), ENTITY].astype(str)
    )
    matches = np.where(np.isin(entities.astype(str), list(members)))[0]
    return matches if matches.size else np.arange(len(entities))


def _date_mask(tensor, origin: int, horizon: int, start: str | None, end: str | None) -> np.ndarray:
    mask = np.zeros(tensor.n_periods, dtype=bool)
    mask[origin + 1 : origin + horizon + 1] = True
    dates = tensor.dates
    if start:
        mask &= dates >= pd.Timestamp(start)
    if end:
        mask &= dates <= pd.Timestamp(end)
    if not mask.any():
        # An out-of-range window would silently be a no-op; use the full horizon.
        mask[origin + 1 : origin + horizon + 1] = True
    return mask


def _to_frame(frame, point: np.ndarray) -> pd.DataFrame:
    out = frame.meta[[ENTITY, "ds", "horizon"]].copy()
    out["forecast"] = point
    return out
