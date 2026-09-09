"""A tiny two-clock dataset, shaped exactly like the competition's.

Small enough that the whole Pol 4 suite runs in under a second, and structured
so the interesting cases are all present: one city big enough for its own
pickup curve, small cities that must fall back to their province, a city that
falls through to the global curve, and (city, check-in) pairs that receive no
searches at all.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from ml.pol4.config import Pol4Config

# Cities 1-3 sit in province 10, cities 4-6 in province 20. City 1 carries the
# volume; city 6 is deliberately near-silent.
CITY_LEVELS = {1: 900.0, 2: 40.0, 3: 25.0, 4: 300.0, 5: 18.0, 6: 2.0}
PROVINCE_OF = {1: 10, 2: 10, 3: 10, 4: 20, 5: 20, 6: 20}

HISTORY_START = pd.Timestamp("2024-01-01")
CUTOFF = pd.Timestamp("2024-10-01")
TARGET_DAYS = 10
MAX_LEAD = 20


def _pickup_weights(max_lead: int) -> np.ndarray:
    """Demand share by days_to_checkin: most searches land close to the stay."""
    lead = np.arange(max_lead + 1)
    weights = np.exp(-lead / 5.0)
    return weights / weights.sum()


def build_events(seed: int = 11) -> pd.DataFrame:
    """Search rows for every check-in from HISTORY_START to CUTOFF + TARGET_DAYS."""
    rng = np.random.default_rng(seed)
    weights = _pickup_weights(MAX_LEAD)
    checkins = pd.date_range(HISTORY_START, CUTOFF + pd.Timedelta(days=TARGET_DAYS), freq="D")

    rows: list[dict] = []
    for city, level in CITY_LEVELS.items():
        for checkin in checkins:
            # A weekly cycle plus noise, so the target is not a constant.
            seasonal = 1.0 + 0.3 * np.sin(2 * np.pi * checkin.dayofyear / 7.0)
            total = rng.poisson(max(level * seasonal, 0.05))
            if total <= 0:
                continue
            counts = rng.multinomial(total, weights)
            for lead, count in enumerate(counts):
                if count <= 0:
                    continue
                rows.append(
                    {
                        "log_date": checkin - pd.Timedelta(days=lead),
                        "city_code": city,
                        "checkin": checkin,
                        "search_count": int(count),
                    }
                )
    return pd.DataFrame(rows)


def write_dataset(directory: Path, seed: int = 11) -> Pol4Config:
    """Write search_data.csv / evaluation.csv / cities.csv and return a config.

    The split mirrors the real one: `search_data.csv` holds check-ins up to the
    cutoff (complete), `evaluation.csv` holds partial observations of the target
    window recorded at or before the cutoff.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    events = build_events(seed)

    target_start = CUTOFF + pd.Timedelta(days=1)
    target_end = CUTOFF + pd.Timedelta(days=TARGET_DAYS)

    history = events[events["checkin"] <= CUTOFF]
    evaluation = events[
        (events["checkin"] >= target_start)
        & (events["checkin"] <= target_end)
        & (events["log_date"] <= CUTOFF)
    ]

    history.to_csv(directory / "search_data.csv", index=False, date_format="%Y-%m-%d")
    evaluation.to_csv(directory / "evaluation.csv", index=False, date_format="%Y-%m-%d")
    pd.DataFrame(
        {
            "city_code": list(CITY_LEVELS),
            "province_code": [PROVINCE_OF[c] for c in CITY_LEVELS],
            "lat": [35.0 + c for c in CITY_LEVELS],
            "long": [51.0 + c for c in CITY_LEVELS],
        }
    ).to_csv(directory / "cities.csv", index=False)

    return replace(
        Pol4Config(),
        raw_dir=directory,
        artifacts_dir=directory / "artifacts",
        cutoff=CUTOFF,
        target_start=target_start,
        target_days=TARGET_DAYS,
        # City 1 clears this comfortably, city 4 sits near it, the rest fall back.
        min_city_support=20_000,
        min_province_support=30_000,
        prior_window_days=90,
        backtest_cutoffs=["2024-08-01", "2024-09-01"],
        horizon_buckets=((1, 3), (4, 7), (8, 10)),
    )


def full_truth(seed: int = 11) -> pd.DataFrame:
    """Complete demand per (city, check-in), including the target window.

    The competition never gives us this for the target window; the fixture does,
    so tests can assert that a prediction is anchored to something real.
    """
    events = build_events(seed)
    return (
        events.groupby(["city_code", "checkin"], as_index=False)["search_count"]
        .sum()
        .rename(columns={"search_count": "final"})
    )
