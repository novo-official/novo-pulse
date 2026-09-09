"""Reads the Pol 4 artefacts and shapes them for the dashboard.

Everything here is a read of a file the offline pipeline already wrote. No model
is fitted, no 3.3-million-row CSV is opened, and nothing is invented: if an
artefact is missing the endpoint says so rather than substituting a number.

Artefacts are cached in memory and re-read when their mtime changes, so a fresh
pipeline run shows up without a restart.
"""
from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml.pol4.config import Pol4Config

log = logging.getLogger(__name__)

_LOCK = threading.Lock()
_CACHE: dict[str, tuple[float, Any]] = {}

CONFIG = Pol4Config()


class ArtifactMissing(FileNotFoundError):
    """An artefact the dashboard needs has not been generated yet."""


def artifacts_dir() -> Path:
    return CONFIG.artifacts_dir


def _load(name: str, reader) -> Any:
    """Read an artefact, re-reading only when the file has changed on disk."""
    path = artifacts_dir() / name
    if not path.exists():
        raise ArtifactMissing(
            f"{name} has not been generated. Run `make pol4` to build it."
        )
    stamp = path.stat().st_mtime
    with _LOCK:
        cached = _CACHE.get(name)
        if cached and cached[0] == stamp:
            return cached[1]
    value = reader(path)
    with _LOCK:
        _CACHE[name] = (stamp, value)
    return value


def clear_cache() -> None:
    with _LOCK:
        _CACHE.clear()


def _parquet(name: str) -> pd.DataFrame:
    return _load(name, pd.read_parquet)


def _json(name: str) -> Any:
    return _load(name, lambda p: json.loads(p.read_text(encoding="utf-8")))


def _csv(name: str) -> pd.DataFrame:
    return _load(name, pd.read_csv)


# ------------------------------------------------------------------ readers
def forecast() -> pd.DataFrame:
    """The submission, with names: one row per (city, Azar check-in)."""
    frame = _csv("results_named.csv").copy()
    frame["checkin"] = pd.to_datetime(frame["checkin"])
    return frame


def run_summary() -> dict[str, Any]:
    return _json("run_summary.json")


def backtest() -> dict[str, Any]:
    return _json("backtest_metrics_phase2.json")


def feature_importance() -> list[dict[str, Any]]:
    return _json("feature_importance.json")


def experiments() -> dict[str, Any]:
    return _json("experiment_summary.json")


def provinces() -> list[dict[str, Any]]:
    return _json("province_summary.json")


def momentum() -> pd.DataFrame:
    return _parquet("city_momentum.parquet")


def stability() -> pd.DataFrame:
    frame = _parquet("stability.parquet").copy()
    frame["checkin"] = pd.to_datetime(frame["checkin"])
    return frame


def target_pickup() -> pd.DataFrame:
    frame = _parquet("target_pickup.parquet").copy()
    frame["checkin"] = pd.to_datetime(frame["checkin"])
    return frame


def city_history() -> pd.DataFrame:
    frame = _parquet("city_history.parquet").copy()
    frame["checkin"] = pd.to_datetime(frame["checkin"])
    return frame


def pickup_curves() -> pd.DataFrame:
    return _parquet("pickup_curves.parquet")


# ------------------------------------------------------------------- shapes
def _iso(series: pd.Series) -> list[str]:
    return pd.DatetimeIndex(series).strftime("%Y-%m-%d").tolist()


def records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Rows as JSON-safe dicts.

    NaN and Infinity are valid Python floats and invalid JSON, so they have to
    become null here rather than at the encoder, which raises. They are not
    hypothetical: 82 cities have no observed Azar demand, so their pickup ratio
    is a division by zero - and "no signal" is exactly what null means.
    """
    cleaned = frame.replace([np.inf, -np.inf], np.nan)
    return [
        {key: (None if pd.isna(value) else value) for key, value in row.items()}
        for row in cleaned.to_dict(orient="records")
    ]


@dataclass
class CityRef:
    city_code: int
    city: str
    province: str


def city_index() -> pd.DataFrame:
    """Every city with its forecast totals - the selector's data source."""
    frame = forecast()
    grouped = frame.groupby(["city_code", "city", "province"], as_index=False).agg(
        predicted_demand=("predicted_demand", "sum"),
        observed_so_far=("observed_so_far", "sum"),
        predicted_remaining=("predicted_remaining", "sum"),
    )
    ratios = momentum()[["city_code", "pickup_ratio", "curve_source"]]
    grouped = grouped.merge(ratios, on="city_code", how="left")
    return grouped.sort_values("predicted_demand", ascending=False, ignore_index=True)


def resolve_city(identifier: str | int) -> CityRef:
    """Accept a city_code or a name, and return the canonical reference.

    The competition identifier is the code; the name is a display convenience,
    so both resolve here rather than the UI having to know which it holds.
    """
    index = city_index()
    text = str(identifier).strip()
    match = index[index["city_code"].astype(str) == text]
    if match.empty:
        match = index[index["city"].str.lower() == text.lower()]
    if match.empty:
        raise LookupError(f"Unknown city '{identifier}'")
    row = match.iloc[0]
    return CityRef(int(row["city_code"]), str(row["city"]), str(row["province"]))


def national_series() -> list[dict[str, Any]]:
    """Total demand per Azar check-in date, split observed vs remaining."""
    frame = forecast()
    daily = frame.groupby("checkin", as_index=False).agg(
        predicted_demand=("predicted_demand", "sum"),
        observed_so_far=("observed_so_far", "sum"),
        predicted_remaining=("predicted_remaining", "sum"),
    )
    daily["checkin"] = _iso(daily["checkin"])
    return records(daily)


def overview() -> dict[str, Any]:
    """The KPI block and the headline series, all from generated artefacts."""
    frame = forecast()
    series = national_series()
    cities = city_index()
    summary = run_summary()
    scores = backtest()["champion"]

    peak = max(series, key=lambda row: row["predicted_demand"])
    top_city = cities.iloc[0]
    hot = momentum().dropna(subset=["pickup_ratio"])
    # A ratio computed on a handful of searches is noise, so the headline only
    # considers cities with a meaningful base.
    hot = hot[hot["observed"] >= hot["observed"].quantile(0.75)]
    fastest = hot.sort_values("pickup_ratio", ascending=False).iloc[0] if len(hot) else None

    return {
        "cutoff": summary["cutoff"],
        "target_window": summary["target_window"],
        "kpis": {
            "total_predicted_demand": int(frame["predicted_demand"].sum()),
            "observed_so_far": int(frame["observed_so_far"].sum()),
            "predicted_remaining": int(frame["predicted_remaining"].sum()),
            "peak_date": peak["checkin"],
            "peak_demand": int(peak["predicted_demand"]),
            "top_city": str(top_city["city"]),
            "top_city_demand": int(top_city["predicted_demand"]),
            "fastest_pickup_city": None if fastest is None else str(fastest["city"]),
            "fastest_pickup_ratio": None if fastest is None else round(float(fastest["pickup_ratio"]), 3),
            "backtest_wape": scores["pooled"]["wape"],
            "baseline_wape": backtest()["baseline"]["pooled"]["wape"],
            "stability_score": summary.get("stability", {}).get("stability_score"),
            "cities": int(frame["city_code"].nunique()),
            "dates": int(frame["checkin"].nunique()),
        },
        "series": series,
        "top_cities": records(cities.head(20).round(3)),
        "provinces": provinces(),
        "momentum": _momentum_leaders(),
    }


def _momentum_leaders(limit: int = 10) -> list[dict[str, Any]]:
    frame = momentum().dropna(subset=["pickup_ratio"])
    if frame.empty:
        return []
    frame = frame[frame["observed"] >= frame["observed"].quantile(0.75)]
    return records(frame.sort_values("pickup_ratio", ascending=False).head(limit).round(3))


def heatmap(top_n: int = 20) -> dict[str, Any]:
    """City x date matrix for the top `top_n` cities by predicted demand."""
    frame = forecast()
    cities = city_index().head(top_n)["city_code"].tolist()
    subset = frame[frame["city_code"].isin(cities)]
    pivot = subset.pivot_table(
        index=["city_code", "city", "province"],
        columns="checkin",
        values="predicted_demand",
        aggfunc="sum",
        fill_value=0,
    )
    order = {code: position for position, code in enumerate(cities)}
    pivot = pivot.reset_index()
    pivot = pivot.sort_values("city_code", key=lambda s: s.map(order), ignore_index=True)
    dates = [c for c in pivot.columns if isinstance(c, pd.Timestamp)]
    return {
        "dates": [d.strftime("%Y-%m-%d") for d in dates],
        "rows": [
            {
                "city_code": int(row["city_code"]),
                "city": row["city"],
                "province": row["province"],
                "values": [int(row[d]) for d in dates],
            }
            for _, row in pivot.iterrows()
        ],
        "max": int(subset["predicted_demand"].max()) if len(subset) else 0,
    }


def city_detail(identifier: str | int, history_days: int = 90) -> dict[str, Any]:
    ref = resolve_city(identifier)
    frame = forecast()
    rows = frame[frame["city_code"] == ref.city_code].sort_values("checkin")

    series = rows[
        ["checkin", "observed_so_far", "predicted_remaining", "predicted_demand"]
    ].copy()
    series["checkin"] = _iso(series["checkin"])

    peaks = records(
        rows.sort_values("predicted_demand", ascending=False)
        .head(5)[["checkin", "predicted_demand", "observed_so_far"]]
        .assign(checkin=lambda f: _iso(f["checkin"]))
    )

    history = city_history()
    history = history[history["city_code"] == ref.city_code].sort_values("checkin").tail(
        history_days
    )
    history_rows = history[["checkin", "demand"]].copy()
    history_rows["checkin"] = _iso(history_rows["checkin"])

    signal = momentum()
    signal = signal[signal["city_code"] == ref.city_code]

    return {
        "city": {
            "city_code": ref.city_code,
            "city": ref.city,
            "province": ref.province,
        },
        "totals": {
            "predicted_demand": int(rows["predicted_demand"].sum()),
            "observed_so_far": int(rows["observed_so_far"].sum()),
            "predicted_remaining": int(rows["predicted_remaining"].sum()),
        },
        "momentum": None if signal.empty else records(signal.round(3))[0],
        "series": records(series),
        "peaks": peaks,
        "history": records(history_rows),
    }


def city_pickup(identifier: str | int, checkin: str | None = None) -> dict[str, Any]:
    """The pickup chart: what has arrived so far against what usually would.

    `expected_cumulative` is this city's own historical completion curve scaled
    to the predicted final demand - the shape history says the accumulation
    normally takes, not a second forecast.
    """
    ref = resolve_city(identifier)
    frame = forecast()
    rows = frame[frame["city_code"] == ref.city_code]
    if rows.empty:
        raise LookupError(f"No forecast for city {ref.city_code}")

    if checkin is None:
        target = rows.sort_values("predicted_demand", ascending=False).iloc[0]
    else:
        stamp = pd.Timestamp(checkin)
        match = rows[rows["checkin"] == stamp]
        if match.empty:
            raise LookupError(f"{checkin} is not in the target window")
        target = match.iloc[0]

    observed = target_pickup()
    observed = observed[
        (observed["city_code"] == ref.city_code) & (observed["checkin"] == target["checkin"])
    ].sort_values("days_to_checkin", ascending=False)

    curves = pickup_curves()
    city_curve = curves[(curves["level"] == "city") & (curves["key"] == ref.city_code)]
    if city_curve.empty:
        province_row = curves[(curves["level"] == "province")]
        province_row = province_row[province_row["name"] == ref.province]
        city_curve = province_row if not province_row.empty else curves[curves["level"] == "global"]
    curve_level = str(city_curve["level"].iloc[0]) if len(city_curve) else "global"

    predicted_final = float(target["predicted_demand"])
    expected = city_curve.sort_values("days_to_checkin", ascending=False)[
        ["days_to_checkin", "completion_fraction"]
    ].copy()
    expected["expected_cumulative"] = expected["completion_fraction"] * predicted_final

    horizon = int((target["checkin"] - pd.Timestamp(run_summary()["cutoff"])).days)
    return {
        "city": {"city_code": ref.city_code, "city": ref.city, "province": ref.province},
        "checkin": target["checkin"].strftime("%Y-%m-%d"),
        "horizon": horizon,
        "curve_level": curve_level,
        "observed_so_far": int(target["observed_so_far"]),
        "predicted_demand": int(predicted_final),
        "predicted_remaining": int(target["predicted_remaining"]),
        "observed": records(
            observed[["days_to_checkin", "searches", "observed_cumulative"]].astype(
                {"days_to_checkin": int, "searches": int, "observed_cumulative": int}
            )
        ),
        "expected": records(expected.round(3)),
        "available_checkins": _iso(rows.sort_values("checkin")["checkin"]),
    }


def stability_detail(
    identifier: str | int | None = None, checkin: str | None = None
) -> dict[str, Any]:
    """Snapshot ladder for one (city, check-in), plus the aggregate picture."""
    frame = stability()
    summary = run_summary().get("stability", {})

    if identifier is None:
        # Default to the pair carrying the most demand, so the page opens on a
        # case worth looking at rather than an arbitrary one.
        pick = frame.sort_values("actual", ascending=False).iloc[0]
        city_code, target = int(pick["city_code"]), pick["checkin"]
    else:
        ref = resolve_city(identifier)
        city_code = ref.city_code
        rows = frame[frame["city_code"] == city_code]
        if rows.empty:
            raise LookupError(f"No stability snapshots for city {city_code}")
        target = (
            pd.Timestamp(checkin)
            if checkin
            else rows.sort_values("actual", ascending=False).iloc[0]["checkin"]
        )

    selected = frame[
        (frame["city_code"] == city_code) & (frame["checkin"] == target)
    ].sort_values("horizon", ascending=False)
    if selected.empty:
        raise LookupError("No snapshots for that city and check-in")

    snapshots = selected[["horizon", "observed", "prediction", "actual"]].copy()
    actual = float(selected["actual"].iloc[0])
    snapshots["absolute_error"] = (snapshots["prediction"] - actual).abs()
    snapshots["relative_error"] = snapshots["absolute_error"] / max(actual, 1.0)

    choices = (
        frame[frame["city_code"] == city_code]
        .groupby("checkin", as_index=False)["actual"]
        .max()
        .sort_values("actual", ascending=False)
    )

    return {
        "city": {
            "city_code": city_code,
            "city": str(selected["city"].iloc[0]),
            "province": str(selected["province"].iloc[0]),
        },
        "checkin": pd.Timestamp(target).strftime("%Y-%m-%d"),
        "actual": int(actual),
        "snapshots": records(snapshots.round(3)),
        "available_checkins": _iso(choices["checkin"]),
        "window": {
            "target_start": summary.get("target_start"),
            "anchor_cutoff": summary.get("anchor_cutoff"),
        },
        "aggregate": {
            "stability_score": summary.get("stability_score"),
            "mean_absolute_revision": summary.get("mean_absolute_revision"),
            "mean_relative_revision": summary.get("mean_relative_revision"),
            "convergence_rate": summary.get("convergence_rate"),
            "by_step": summary.get("by_step", []),
            "by_horizon": summary.get("by_horizon", []),
        },
    }


def model_performance() -> dict[str, Any]:
    """Champion vs baseline, every fold and every breakdown. Nothing hidden."""
    scores = backtest()
    summary = run_summary()
    champion, baseline = scores["champion"], scores["baseline"]
    improvement = 1 - champion["pooled"]["wape"] / baseline["pooled"]["wape"]

    folds = [
        {
            "cutoff": cutoff,
            "champion_wape": champion["folds"][cutoff]["wape"],
            "baseline_wape": baseline["folds"][cutoff]["wape"],
            "actual_total": champion["folds"][cutoff]["actual_total"],
        }
        for cutoff in champion["folds"]
    ]
    hardest = max(folds, key=lambda row: row["champion_wape"])

    return {
        "champion": {
            **summary.get("champion", {}),
            "wape": champion["pooled"]["wape"],
            "normalised_bias": champion["pooled"]["normalised_bias"],
        },
        "baseline": {
            "name": "pickup baseline",
            "wape": baseline["pooled"]["wape"],
            "normalised_bias": baseline["pooled"]["normalised_bias"],
        },
        "improvement": round(improvement, 4),
        "folds": folds,
        "hardest_fold": hardest["cutoff"],
        "by_horizon_bucket": champion["by_horizon_bucket"],
        "baseline_by_horizon_bucket": baseline["by_horizon_bucket"],
        "by_province": champion["by_province"],
        "by_weekday": champion["by_weekday"],
        "by_demand_bucket": champion["by_demand_bucket"],
        "by_observation_state": champion["by_observation_state"],
        "high_demand": champion["high_demand"],
        "feature_importance": feature_importance()[:20],
        "experiments": experiments()["experiments"],
        "config": scores["config"],
    }
