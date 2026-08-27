"""Anomaly and peak detection.

Two complementary signals:

* **historical anomalies** - where the actual demand departed from what the
  model expected (backtest residual, scaled by a robust MAD estimate),
* **forecast peaks / troughs** - where the *future* forecast departs from the
  entity's own seasonal reference level.

Both use the median absolute deviation rather than the standard deviation, so a
single extreme day cannot mask the rest.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..contract import ENTITY

MAD_TO_SIGMA = 1.4826

# A series whose normal level is below this fraction of the panel median cannot
# produce a meaningful percentage swing (1 -> 2 is not "+100% demand").
MIN_BASELINE_MEDIAN_FRACTION = 0.05

SEVERITY_THRESHOLDS = ((5.0, "high"), (4.0, "medium"), (3.0, "low"))

STATUS_FA = {
    "normal": "عادی",
    "spike": "جهش تقاضا",
    "drop": "افت تقاضا",
}


@dataclass
class AnomalyConfig:
    # 3.0 robust sigma => ~0.3% false positives on clean data. Lower thresholds
    # bury the real anomalies in noise.
    min_score: float = 3.0
    min_periods: int = 21
    max_results: int = 100


def robust_scale(values: np.ndarray) -> float:
    """A spread estimate that a single extreme value cannot inflate.

    MAD first; when more than half the values are identical MAD collapses to
    zero, so we step down through the IQR and the mean absolute deviation
    before ever touching the standard deviation - which the outlier itself
    would dominate.
    """
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 1.0
    median = float(np.median(finite))

    mad = float(np.median(np.abs(finite - median))) * MAD_TO_SIGMA
    if mad > 1e-9:
        return mad

    iqr = float(np.subtract(*np.percentile(finite, [75, 25]))) / 1.349
    if abs(iqr) > 1e-9:
        return abs(iqr)

    mean_abs = float(np.mean(np.abs(finite - median)))
    if mean_abs > 1e-9:
        return mean_abs

    std = float(np.std(finite))
    return std if std > 1e-9 else 1.0


def robust_z(values: np.ndarray) -> np.ndarray:
    """Robust z-score around the median."""
    finite = values[np.isfinite(values)]
    if finite.size < 3:
        return np.zeros_like(values, dtype=float)
    median = float(np.median(finite))
    return (values - median) / robust_scale(finite)


def _severity(score: float) -> str:
    for threshold, label in SEVERITY_THRESHOLDS:
        if abs(score) >= threshold:
            return label
    return "low"


def detect_residual_anomalies(
    backtest: pd.DataFrame, config: AnomalyConfig | None = None
) -> pd.DataFrame:
    """Anomalies in observed history, measured against the model's expectation."""
    cfg = config or AnomalyConfig()
    if backtest.empty:
        return pd.DataFrame(
            columns=[ENTITY, "ds", "actual", "expected", "deviation", "score", "type", "severity"]
        )

    frame = backtest.copy()
    frame["residual"] = frame["actual"] - frame["prediction"]
    rows = []
    for entity, group in frame.groupby(ENTITY):
        if len(group) < cfg.min_periods:
            continue
        group = group.sort_values("ds")
        scores = robust_z(group["residual"].to_numpy(dtype=float))
        selected = np.abs(scores) >= cfg.min_score
        if not selected.any():
            continue
        window = group.loc[selected].copy()
        window["score"] = scores[selected]
        rows.append(window)

    if not rows:
        return pd.DataFrame(
            columns=[ENTITY, "ds", "actual", "expected", "deviation", "score", "type", "severity"]
        )

    result = pd.concat(rows, ignore_index=True)
    result = result.rename(columns={"prediction": "expected"})
    with np.errstate(divide="ignore", invalid="ignore"):
        result["deviation"] = np.where(
            np.abs(result["expected"]) > 1e-9,
            (result["actual"] - result["expected"]) / np.abs(result["expected"]),
            np.nan,
        )
    result["type"] = np.where(result["score"] > 0, "spike", "drop")
    result["severity"] = [_severity(s) for s in result["score"]]
    result["source"] = "residual"
    result = result.reindex(
        result["score"].abs().sort_values(ascending=False).index
    ).head(cfg.max_results)
    return result.reset_index(drop=True)


def detect_forecast_anomalies(
    forecast: pd.DataFrame,
    history: pd.DataFrame,
    season: int = 7,
    config: AnomalyConfig | None = None,
) -> pd.DataFrame:
    """Forecast points that break out of an entity's recent normal range."""
    cfg = config or AnomalyConfig()
    if forecast.empty or history.empty:
        return pd.DataFrame(
            columns=[ENTITY, "ds", "forecast", "expected", "deviation", "score", "type", "severity"]
        )

    reference = (
        history.sort_values("ds")
        .groupby(ENTITY)["y"]
        .agg(
            expected=lambda s: float(np.median(s.tail(season * 8))),
            spread=lambda s: float(
                np.median(np.abs(s.tail(season * 8) - np.median(s.tail(season * 8))))
            ),
            n=lambda s: int(len(s)),
        )
        .reset_index()
    )
    merged = forecast.merge(reference, on=ENTITY, how="left")
    merged = merged[merged["n"] >= cfg.min_periods]
    if merged.empty:
        return pd.DataFrame(
            columns=[ENTITY, "ds", "forecast", "expected", "deviation", "score", "type", "severity"]
        )

    scale = merged["spread"].to_numpy(dtype=float) * MAD_TO_SIGMA
    fallback = np.maximum(merged["expected"].to_numpy(dtype=float) * 0.25, 1.0)
    scale = np.where(scale < 1e-9, fallback, scale)
    merged["score"] = (merged["forecast"] - merged["expected"]) / scale
    with np.errstate(divide="ignore", invalid="ignore"):
        merged["deviation"] = np.where(
            np.abs(merged["expected"]) > 1e-9,
            (merged["forecast"] - merged["expected"]) / np.abs(merged["expected"]),
            np.nan,
        )
    selected = merged[np.abs(merged["score"]) >= cfg.min_score].copy()
    if selected.empty:
        return selected.assign(type=[], severity=[], source=[])
    selected["type"] = np.where(selected["score"] > 0, "spike", "drop")
    selected["severity"] = [_severity(s) for s in selected["score"]]
    selected["source"] = "forecast"
    selected = selected.reindex(
        selected["score"].abs().sort_values(ascending=False).index
    ).head(cfg.max_results)
    return selected.reset_index(drop=True)


def detect_peak_periods(
    forecast: pd.DataFrame,
    history: pd.DataFrame,
    season: int = 7,
    min_length: int = 2,
    threshold: float = 0.15,
    max_results: int = 20,
    min_baseline_quantile: float = 0.25,
) -> list[dict[str, Any]]:
    """Contiguous runs where the forecast sits well above/below the norm.

    A "peak" is a stretch of consecutive periods whose forecast exceeds the
    entity's recent median by more than `threshold`. This is what powers the
    "upcoming high demand: Kish, Dec 22-28, +31%" card.
    """
    if forecast.empty or history.empty:
        return []

    # The baseline is phase-aware: for daily data every Friday is compared with
    # recent Fridays. Without this, the regular weekend cycle would be reported
    # as a "peak" every single week and drown out the real ones.
    reference = _phase_baseline(history, season)
    levels = np.array(
        [v for entity in reference.values() for v in entity.values() if v and v > 0], dtype=float
    )
    # Two floors, because either alone can be defeated: the quantile handles a
    # long tail of small series, the median fraction handles a bimodal panel
    # where a quarter of the entities are near zero.
    min_baseline = 0.0
    if levels.size:
        min_baseline = max(
            float(np.quantile(levels, min_baseline_quantile)),
            float(np.median(levels)) * MIN_BASELINE_MEDIAN_FRACTION,
        )
    periods: list[dict[str, Any]] = []

    for entity, group in forecast.groupby(ENTITY):
        phases = reference.get(entity)
        if not phases:
            continue
        group = group.sort_values("ds").reset_index(drop=True)
        stamps = pd.to_datetime(group["ds"])
        phase_keys = _phase_of(stamps, season)
        baselines = np.array([phases.get(int(k), np.nan) for k in phase_keys], dtype=float)
        overall = float(np.nanmedian(list(phases.values())))
        baselines = np.where(np.isfinite(baselines) & (baselines > 0), baselines, overall)
        if not np.isfinite(overall) or overall <= 0 or overall < min_baseline:
            continue

        values = group["forecast"].to_numpy(dtype=float)
        relative = (values - baselines) / baselines
        direction = np.where(relative > threshold, 1, np.where(relative < -threshold, -1, 0))

        start = 0
        while start < len(direction):
            if direction[start] == 0:
                start += 1
                continue
            end = start
            while end + 1 < len(direction) and direction[end + 1] == direction[start]:
                end += 1
            length = end - start + 1
            if length >= min_length:
                window = group.iloc[start : end + 1]
                mean_change = float(np.mean(relative[start : end + 1]))
                width = None
                if {"lower", "upper"}.issubset(window.columns):
                    denominator = max(float(window["forecast"].mean()), 1e-6)
                    width = float((window["upper"] - window["lower"]).mean()) / denominator
                periods.append(
                    {
                        "entity_id": str(entity),
                        "type": "peak" if direction[start] > 0 else "trough",
                        "start": str(pd.Timestamp(window["ds"].iloc[0]).date()),
                        "end": str(pd.Timestamp(window["ds"].iloc[-1]).date()),
                        "days": int(length),
                        "expected_change_pct": round(mean_change * 100, 2),
                        "baseline_level": round(float(np.mean(baselines[start : end + 1])), 3),
                        "forecast_level": round(float(window["forecast"].mean()), 3),
                        "total_forecast": round(float(window["forecast"].sum()), 2),
                        "relative_interval_width": None if width is None else round(width, 4),
                        "confidence": _peak_confidence(length, abs(mean_change), width),
                    }
                )
            start = end + 1

    periods.sort(key=lambda item: -abs(item["expected_change_pct"]))
    return periods[:max_results]


def _phase_of(stamps: pd.Series, season: int) -> np.ndarray:
    """Position within the seasonal cycle (weekday for daily data)."""
    stamps = pd.to_datetime(stamps)
    if season == 7:
        return stamps.dt.dayofweek.to_numpy()
    if season == 12:
        return stamps.dt.month.to_numpy()
    if season == 24:
        return stamps.dt.hour.to_numpy()
    return (stamps.dt.dayofyear.to_numpy() % max(season, 1))


def _phase_baseline(history: pd.DataFrame, season: int, n_cycles: int = 8) -> dict:
    """Per-entity median demand for each phase of the seasonal cycle."""
    frame = history.sort_values("ds").copy()
    frame["__phase"] = _phase_of(frame["ds"], season)
    recent = frame.groupby(ENTITY, group_keys=False).tail(season * n_cycles)
    grouped = recent.groupby([ENTITY, "__phase"])["y"].median()
    out: dict = {}
    for (entity, phase), value in grouped.items():
        out.setdefault(entity, {})[int(phase)] = float(value)
    return out


def _peak_confidence(length: int, change: float, interval_width: float | None) -> str:
    """Longer, larger and tighter -> more confidence. Documented, not magic."""
    score = 0
    score += 1 if length >= 3 else 0
    score += 1 if change >= 0.25 else 0
    if interval_width is not None:
        score += 1 if interval_width <= 0.8 else 0
    else:
        score += 0
    return {3: "high", 2: "medium"}.get(score, "low")


def summarise_anomalies(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {"total": 0, "spikes": 0, "drops": 0, "high_severity": 0}
    return {
        "total": int(len(frame)),
        "spikes": int((frame["type"] == "spike").sum()),
        "drops": int((frame["type"] == "drop").sum()),
        "high_severity": int((frame["severity"] == "high").sum()),
    }
