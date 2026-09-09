"""Training and inference frames for the remaining-demand model.

One training row is `(city c, check-in T, horizon h)`:

    features  read only from searches logged at or before  T - h
    target    remaining = final_demand(c, T) - observed(c, T, T - h)

which is the same shape as the real task: stand h days before a check-in,
knowing only what has been searched so far, and predict what is still to come.
Horizons are sampled rather than enumerated - 30 per pair is seven million rows
carrying no extra signal.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import Pol4Config
from .features import FeatureBuilder
from .loader import CHECKIN, CITY, Pol4Data


def _complete_grid(cities: np.ndarray, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Every (city, check-in) pair, including the ones that saw no searches.

    A pair absent from the log is a real observation - zero demand - and
    dropping it would teach the model that demand is never zero.
    """
    return pd.MultiIndex.from_product(
        [np.sort(cities), pd.DatetimeIndex(dates)], names=[CITY, CHECKIN]
    ).to_frame(index=False)


@dataclass
class SupervisedFrame:
    X: pd.DataFrame
    y: np.ndarray | None
    meta: pd.DataFrame          # city_code, checkin, horizon, observed, (final)

    def __len__(self) -> int:
        return len(self.X)


def build_training_frame(
    data: Pol4Data,
    cutoff: pd.Timestamp,
    groups: tuple[str, ...],
    baseline: Any,
    config: Pol4Config | None = None,
) -> SupervisedFrame:
    """Rows for every completed check-in in the training window."""
    config = config or Pol4Config()
    cutoff = pd.Timestamp(cutoff)

    history = data.history_before(cutoff)
    start = max(cutoff - pd.Timedelta(days=config.train_window_days - 1), history[CHECKIN].min())
    dates = pd.date_range(start, cutoff, freq="D")
    keys = _complete_grid(data.city_codes, dates)
    events = history[history[CHECKIN].isin(dates)]

    builder = FeatureBuilder.build(data, cutoff, keys, events, config)
    final = builder.tensor.final()

    blocks: list[pd.DataFrame] = []
    metas: list[pd.DataFrame] = []
    targets: list[np.ndarray] = []
    for horizon in config.train_horizons:
        frame = builder.at_horizon(horizon, groups, baseline)
        observed = frame["observed_total"].to_numpy()
        blocks.append(frame)
        metas.append(
            keys.assign(horizon=horizon, observed=observed, final=final)
        )
        targets.append(final - observed)

    X = pd.concat(blocks, ignore_index=True)
    meta = pd.concat(metas, ignore_index=True)
    y = np.concatenate(targets)

    if len(X) > config.max_train_rows:
        rng = np.random.default_rng(config.seed)
        picked = np.sort(rng.choice(len(X), size=config.max_train_rows, replace=False))
        X = X.iloc[picked].reset_index(drop=True)
        meta = meta.iloc[picked].reset_index(drop=True)
        y = y[picked]

    return SupervisedFrame(X=X, y=np.maximum(y, 0.0), meta=meta)


def export_training_frame(
    frame: SupervisedFrame, path: Path, sample: int | None = None, seed: int = 42
) -> Path:
    """Write the supervised frame - features, target and keys - to one file.

    The target column is named `remaining_demand` so the file is self-describing:
    it is what the model predicts, not the final demand, and adding
    `observed` back gives the forecast.

    `sample` writes a horizon-stratified subset instead of the whole frame. The
    full frame is ~77 MB at the competition cutoff, which is worth exporting on
    request but not worth committing; the sample is there so the shape is
    readable without regenerating anything.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    table = frame.X.copy()
    for column in frame.meta.columns:
        table[column] = frame.meta[column].to_numpy()
    table["remaining_demand"] = frame.y

    if sample:
        # Stratify by horizon: the bands are fitted separately, so a sample that
        # under-covers one of them would misrepresent the model's diet.
        per_horizon = max(1, sample // table["horizon"].nunique())
        rng = np.random.default_rng(seed)
        picked: list[np.ndarray] = []
        for _, positions in table.groupby("horizon").indices.items():
            take = min(len(positions), per_horizon)
            picked.append(rng.choice(positions, size=take, replace=False))
        table = table.iloc[np.sort(np.concatenate(picked))].sort_values(
            ["checkin", "city_code", "horizon"], ignore_index=True
        )

    keys = ["city_code", "checkin", "horizon", "observed", "final", "remaining_demand"]
    table = table[keys + [c for c in table.columns if c not in keys]]
    if path.suffix == ".csv":
        # Full float64 repr triples the file for precision no reader needs; the
        # parquet export keeps the exact values.
        table.to_csv(path, index=False, float_format="%.4g")
    else:
        table.to_parquet(path, index=False)
    return path


def build_inference_frame(
    data: Pol4Data,
    cutoff: pd.Timestamp,
    target_dates: pd.DatetimeIndex,
    groups: tuple[str, ...],
    baseline: Any,
    config: Pol4Config | None = None,
) -> SupervisedFrame:
    """Rows for the 30 target check-ins, each at its own horizon.

    Every city shares a check-in's horizon, so the features are built once per
    horizon and the matching pairs are selected out - 30 passes, not 9,630.
    """
    config = config or Pol4Config()
    cutoff = pd.Timestamp(cutoff)
    target_dates = pd.DatetimeIndex(target_dates)

    keys = _complete_grid(data.city_codes, target_dates)
    rows = pd.concat([data.search, data.evaluation], ignore_index=True)
    # Belt and braces: the tensor would ignore these columns anyway (a feature
    # at horizon h never reads below h), but filtering here makes the guarantee
    # visible and gives the leakage test something to bite on.
    events = rows[
        (rows["log_date"] <= cutoff)
        & (rows[CHECKIN] >= target_dates.min())
        & (rows[CHECKIN] <= target_dates.max())
    ]

    builder = FeatureBuilder.build(data, cutoff, keys, events, config)
    horizons = (pd.DatetimeIndex(keys[CHECKIN]) - cutoff).days.to_numpy()

    blocks: list[pd.DataFrame] = []
    metas: list[pd.DataFrame] = []
    for horizon in np.unique(horizons):
        mask = horizons == horizon
        frame = builder.at_horizon(int(horizon), groups, baseline)
        blocks.append(frame[mask])
        metas.append(
            keys[mask].assign(
                horizon=int(horizon), observed=frame.loc[mask, "observed_total"].to_numpy()
            )
        )

    X = pd.concat(blocks, ignore_index=True)
    meta = pd.concat(metas, ignore_index=True)
    order = np.lexsort((meta[CHECKIN].to_numpy(), meta[CITY].to_numpy()))
    return SupervisedFrame(
        X=X.iloc[order].reset_index(drop=True),
        y=None,
        meta=meta.iloc[order].reset_index(drop=True),
    )
