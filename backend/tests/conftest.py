"""Shared fixtures.

Tests run against a small, fast synthetic panel rather than the full demo
dataset so the whole suite stays under a few seconds.
"""
from __future__ import annotations

import datetime as dt
import os

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
os.environ.setdefault("DEMO_MODE", "true")
os.environ.setdefault("SECRET_KEY", "test-key")

from ml.contract import DataContract  # noqa: E402
from ml.data.adapter import DataAdapter  # noqa: E402
from ml.features.engineering import FeatureConfig, FeatureEngine  # noqa: E402
from ml.features.tensor import build_tensor  # noqa: E402

N_ENTITIES = 8
N_DAYS = 260
HORIZON = 14


@pytest.fixture(scope="session")
def raw_frame() -> pd.DataFrame:
    """A tiny panel with a known weekly cycle, trend and price effect."""
    rng = np.random.default_rng(7)
    dates = pd.date_range(dt.date(2024, 1, 1), periods=N_DAYS, freq="D")
    rows = []
    for entity in range(N_ENTITIES):
        level = 10 + entity * 3
        weekly = 1 + 0.35 * np.sin(2 * np.pi * np.arange(N_DAYS) / 7)
        trend = 1 + np.arange(N_DAYS) * 0.001
        price = 100 * (1 + rng.normal(0, 0.05, N_DAYS))
        demand = level * weekly * trend * (price / 100) ** -0.8
        demand = np.maximum(rng.poisson(np.maximum(demand, 0.1)), 0)
        rows.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "listing_id": f"L{entity:02d}",
                    "city": f"city_{entity % 3}",
                    "kind": "hotel" if entity % 2 == 0 else "villa",
                    "bookings": demand,
                    "price": price.round(2),
                    "searches": rng.poisson(np.maximum(demand * 8, 1)),
                    "is_holiday": (np.arange(N_DAYS) % 60 < 2).astype(int),
                    "capacity": 20 + entity,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


@pytest.fixture(scope="session")
def contract() -> DataContract:
    return DataContract.from_dict(
        {
            "dataset": {"name": "test", "path": None},
            "schema": {
                "timestamp": "date",
                "target": "bookings",
                "entity_id": "listing_id",
                "frequency": "D",
                "aggregation": "sum",
            },
            "hierarchy": {"destination": "city", "category": "kind"},
            "features": {
                "future": ["price", "is_holiday"],
                "historical": ["searches"],
                "static": ["capacity", "kind"],
            },
            "target_options": {"non_negative": True, "integer": True, "log1p_transform": False},
            "evaluation": {
                "primary_metric": "wape",
                "horizons": [7, HORIZON],
                "horizon_buckets": [[1, 7], [8, 14]],
                "quantiles": [0.1, 0.5, 0.9],
                "cv": {"n_folds": 2, "step": 14},
            },
        }
    )


@pytest.fixture(scope="session")
def panel(contract, raw_frame):
    return DataAdapter(contract).build(raw_frame)


@pytest.fixture(scope="session")
def engine(panel):
    tensor = build_tensor(
        panel.frame,
        freq=panel.frequency,
        horizon=HORIZON,
        future_features=panel.future_features,
        past_features=panel.historical_features,
        static_features=panel.static_features,
    )
    return FeatureEngine(
        tensor, FeatureConfig.for_frequency("D", max_horizon=HORIZON, min_context=7)
    ).prepare()
