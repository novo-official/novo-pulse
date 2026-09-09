"""Pickup (completion) curves - how demand accumulates before a check-in.

For a completed check-in T, the share of its final demand that was already
visible h days beforehand is

    completion_fraction(h) = SUM(search_count where days_to_checkin >= h)
                             ---------------------------------------------
                             SUM(search_count)

Pooled as a ratio of sums rather than a mean of ratios, because the competition
metric (WAPE) is itself a ratio of sums: a city-day contributing 200,000
searches should move the curve 200,000 times more than one contributing one.

CUTOFF SAFETY IS THE POINT OF THIS MODULE. A curve fitted at cutoff C may only
read check-ins at or before C, because those are the only ones whose *final*
demand is knowable at C. `fit()` filters on that and nothing downstream can
widen it - which is what `tests/test_pol4_leakage.py` asserts by overwriting
every post-cutoff row with garbage and requiring an identical curve.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .config import MAX_LEAD_TIME, Pol4Config
from .loader import CITY, DTC, PROVINCE, SEARCHES, Pol4Data


def _curve_from_events(events: pd.DataFrame, n_horizons: int) -> tuple[np.ndarray, float]:
    """Completion fractions for one group, plus the demand they were fitted on."""
    by_dtc = events.groupby(DTC)[SEARCHES].sum()
    weights = np.zeros(n_horizons, dtype=np.float64)
    inside = by_dtc.index[(by_dtc.index >= 0) & (by_dtc.index < n_horizons)]
    weights[inside.to_numpy()] = by_dtc.loc[inside].to_numpy(dtype=np.float64)
    total = float(by_dtc.sum())
    if total <= 0:
        return np.full(n_horizons, np.nan), 0.0
    # fraction(h) = share of demand that arrives at or before h days out, i.e.
    # the reverse cumulative sum over days_to_checkin.
    reverse_cum = np.cumsum(weights[::-1])[::-1]
    return reverse_cum / total, total


def _grouped_curves(
    events: pd.DataFrame, key: str, n_horizons: int, min_support: float
) -> tuple[dict[Any, np.ndarray], dict[Any, float]]:
    curves: dict[Any, np.ndarray] = {}
    support: dict[Any, float] = {}
    for group, rows in events.groupby(key, sort=True):
        curve, total = _curve_from_events(rows, n_horizons)
        support[group] = total
        if total >= min_support and np.isfinite(curve).all():
            curves[group] = curve
    return curves, support


@dataclass
class PickupCurves:
    """Completion curves with a city -> province -> global fallback."""

    cutoff: pd.Timestamp
    n_horizons: int
    global_curve: np.ndarray
    province_curves: dict[int, np.ndarray]
    city_curves: dict[int, np.ndarray]
    city_support: dict[int, float]
    province_support: dict[int, float]
    province_of: dict[int, int]
    min_fraction: float = 0.005

    # ---------------------------------------------------------------- fit
    @classmethod
    def fit(
        cls,
        data: Pol4Data,
        cutoff: pd.Timestamp,
        config: Pol4Config | None = None,
    ) -> "PickupCurves":
        """Fit curves using ONLY check-ins completed at or before `cutoff`."""
        config = config or Pol4Config()
        cutoff = pd.Timestamp(cutoff)
        history = data.history_before(cutoff)
        if history.empty:
            raise ValueError(f"no completed check-ins at or before {cutoff.date()}")

        n_horizons = int(max(history[DTC].max(), MAX_LEAD_TIME)) + 1
        province_of = data.province_of

        global_curve, global_total = _curve_from_events(history, n_horizons)
        if global_total <= 0:
            raise ValueError("history carries no demand; cannot fit a pickup curve")

        city_curves, city_support = _grouped_curves(
            history, CITY, n_horizons, config.min_city_support
        )
        with_province = history.assign(**{PROVINCE: history[CITY].map(province_of)})
        province_curves, province_support = _grouped_curves(
            with_province.dropna(subset=[PROVINCE]),
            PROVINCE,
            n_horizons,
            config.min_province_support,
        )

        return cls(
            cutoff=cutoff,
            n_horizons=n_horizons,
            global_curve=global_curve,
            province_curves={int(k): v for k, v in province_curves.items()},
            city_curves={int(k): v for k, v in city_curves.items()},
            city_support={int(k): v for k, v in city_support.items()},
            province_support={int(k): v for k, v in province_support.items()},
            province_of={int(k): int(v) for k, v in province_of.items()},
            min_fraction=config.min_fraction,
        )

    # ------------------------------------------------------------- lookup
    def _stack(self, curves: dict[int, np.ndarray]) -> tuple[dict[int, int], np.ndarray]:
        keys = sorted(curves)
        index = {key: position for position, key in enumerate(keys)}
        matrix = (
            np.stack([curves[key] for key in keys])
            if keys
            else np.empty((0, self.n_horizons), dtype=np.float64)
        )
        return index, matrix

    def fraction(self, city_codes, horizons) -> np.ndarray:
        """Expected completion fraction per (city, horizon), with fallback.

        City curve when the city has enough history, else its province, else the
        global curve. Horizons beyond the fitted window clamp to the last
        column, where the curve is already at its floor - a search cannot arrive
        before the booking window opens, so nothing is visible that far out.
        """
        cities = pd.Series(np.asarray(city_codes)).astype("int64")
        h = np.clip(np.asarray(horizons, dtype=np.int64), 0, self.n_horizons - 1)

        out = self.global_curve[h].astype(np.float64)

        province = cities.map(self.province_of)
        prov_index, prov_matrix = self._stack(self.province_curves)
        prov_row = province.map(prov_index).to_numpy(dtype="float64")
        hit = np.isfinite(prov_row)
        if hit.any():
            out[hit] = prov_matrix[prov_row[hit].astype(np.int64), h[hit]]

        city_index, city_matrix = self._stack(self.city_curves)
        city_row = cities.map(city_index).to_numpy(dtype="float64")
        hit = np.isfinite(city_row)
        if hit.any():
            out[hit] = city_matrix[city_row[hit].astype(np.int64), h[hit]]

        return np.clip(out, self.min_fraction, 1.0)

    def source(self, city_codes) -> np.ndarray:
        """Which level of the hierarchy each city's curve came from."""
        return np.array(
            [
                "city"
                if int(c) in self.city_curves
                else "province"
                if self.province_of.get(int(c), -1) in self.province_curves
                else "global"
                for c in np.asarray(city_codes)
            ]
        )

    # ---------------------------------------------------------- artefacts
    def to_frame(self) -> pd.DataFrame:
        """Long-format curves, one row per (level, key, horizon)."""
        pieces = [
            pd.DataFrame(
                {
                    "level": "global",
                    "key": -1,
                    "days_to_checkin": np.arange(self.n_horizons),
                    "completion_fraction": self.global_curve,
                    "support": float(sum(self.city_support.values())),
                }
            )
        ]
        for level, curves, support in (
            ("province", self.province_curves, self.province_support),
            ("city", self.city_curves, self.city_support),
        ):
            for key, curve in sorted(curves.items()):
                pieces.append(
                    pd.DataFrame(
                        {
                            "level": level,
                            "key": key,
                            "days_to_checkin": np.arange(self.n_horizons),
                            "completion_fraction": curve,
                            "support": support.get(key, 0.0),
                        }
                    )
                )
        frame = pd.concat(pieces, ignore_index=True)
        frame["cutoff"] = self.cutoff.date().isoformat()
        return frame

    def summary(self) -> dict[str, Any]:
        marks = [1, 3, 7, 14, 21, 30, 45, 59]
        return {
            "cutoff": self.cutoff.date().isoformat(),
            "n_horizons": self.n_horizons,
            "cities_with_own_curve": len(self.city_curves),
            "provinces_with_own_curve": len(self.province_curves),
            "global_curve": {
                f"D-{m}": round(float(self.global_curve[m]), 4)
                for m in marks
                if m < self.n_horizons
            },
        }
