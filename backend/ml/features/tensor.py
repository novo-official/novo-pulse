"""Dense (entity x time) tensor view of a panel, extended into the future.

Every model in this project forecasts from a *forecast origin*. Holding the
panel as aligned matrices makes that explicit and cheap:

    origin o = index of the last observed period
    target  t = o + h                (h = 1 .. horizon)

A feature is leakage-safe if and only if it is read at index <= o (historical)
or comes from a covariate that is genuinely known for index t (calendar,
planned price, holiday flags). The tensor enforces that split structurally -
past covariates simply do not exist beyond `n_observed`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..contract import CATEGORY, DESTINATION, ENTITY, TARGET, TS
from .calendar import is_event_like, is_price_like


@dataclass
class PanelTensor:
    """Aligned matrices for one hierarchy level."""

    dates: pd.DatetimeIndex          # length T_ext (observed + future)
    entities: np.ndarray             # length E
    y: np.ndarray                    # (E, T_ext) float32, NaN outside observation
    observed: np.ndarray             # (E, T_ext) bool - True where y is real
    past: dict[str, np.ndarray] = field(default_factory=dict)    # (E, T_ext)
    future: dict[str, np.ndarray] = field(default_factory=dict)  # (E, T_ext)
    static: pd.DataFrame = field(default_factory=pd.DataFrame)
    freq: str = "D"
    n_observed: int = 0              # number of columns that hold real history

    @property
    def n_entities(self) -> int:
        return len(self.entities)

    @property
    def n_periods(self) -> int:
        return len(self.dates)

    @property
    def origin_index(self) -> int:
        """Index of the last observed period - the default forecast origin."""
        return self.n_observed - 1

    def entity_start_index(self) -> np.ndarray:
        """First observed column per entity (cold-start / late onboarding)."""
        has = self.observed
        first = np.argmax(has, axis=1)
        first[~has.any(axis=1)] = self.n_observed
        return first

    def copy_for_scenario(self) -> "PanelTensor":
        """Shallow copy with *independent* future covariate matrices.

        Used by the what-if simulator, which mutates future covariates only.
        """
        return PanelTensor(
            dates=self.dates,
            entities=self.entities,
            y=self.y,
            observed=self.observed,
            past=self.past,
            future={k: v.copy() for k, v in self.future.items()},
            static=self.static,
            freq=self.freq,
            n_observed=self.n_observed,
        )


def build_tensor(
    frame: pd.DataFrame,
    freq: str = "D",
    horizon: int = 30,
    future_features: list[str] | None = None,
    past_features: list[str] | None = None,
    static_features: list[str] | None = None,
    future_overrides: pd.DataFrame | None = None,
) -> PanelTensor:
    """Pivot a long panel into aligned matrices, extended `horizon` steps ahead.

    Future covariate values beyond the observed range are projected with a
    seasonal-naive rule (same period one season back) falling back to the last
    known value. `future_overrides` - a long frame with ds/entity_id plus
    covariate columns - takes precedence and is how planned prices or a known
    event calendar enter the system.
    """
    future_features = list(future_features or [])
    past_features = list(past_features or [])
    static_features = list(static_features or [])

    observed_dates = pd.DatetimeIndex(np.sort(frame[TS].unique()))
    step = _freq_step(observed_dates, freq)
    future_dates = pd.date_range(
        observed_dates[-1] + step, periods=horizon, freq=_pandas_freq(freq)
    )
    dates = observed_dates.append(future_dates)
    n_observed = len(observed_dates)

    entities = np.array(sorted(frame[ENTITY].astype(str).unique()))
    entity_pos = {e: i for i, e in enumerate(entities)}
    date_pos = {d: i for i, d in enumerate(dates)}

    rows = frame[ENTITY].astype(str).map(entity_pos).to_numpy()
    cols = frame[TS].map(date_pos).to_numpy()
    shape = (len(entities), len(dates))

    y = np.full(shape, np.nan, dtype=np.float32)
    y[rows, cols] = frame[TARGET].to_numpy(dtype=np.float32)
    observed = np.zeros(shape, dtype=bool)
    observed[rows, cols] = True

    past: dict[str, np.ndarray] = {}
    for column in past_features:
        if column not in frame.columns or not pd.api.types.is_numeric_dtype(frame[column]):
            continue
        matrix = np.full(shape, np.nan, dtype=np.float32)
        matrix[rows, cols] = frame[column].to_numpy(dtype=np.float32)
        past[column] = _forward_fill_rows(matrix, limit_col=n_observed)

    future: dict[str, np.ndarray] = {}
    for column in future_features:
        if column not in frame.columns or not pd.api.types.is_numeric_dtype(frame[column]):
            continue
        matrix = np.full(shape, np.nan, dtype=np.float32)
        matrix[rows, cols] = frame[column].to_numpy(dtype=np.float32)
        matrix = _forward_fill_rows(matrix, limit_col=n_observed)
        future[column] = _project_future(matrix, n_observed, freq, column)

    if future_overrides is not None and len(future_overrides):
        _apply_overrides(future, future_overrides, entity_pos, date_pos)

    static = _static_table(frame, static_features, entities)

    return PanelTensor(
        dates=dates,
        entities=entities,
        y=y,
        observed=observed,
        past=past,
        future=future,
        static=static,
        freq=freq,
        n_observed=n_observed,
    )


# --------------------------------------------------------------------------
def _pandas_freq(freq: str) -> str:
    return {"D": "D", "W": "W", "M": "MS", "MS": "MS", "H": "h", "h": "h"}.get(freq, freq)


def _freq_step(dates: pd.DatetimeIndex, freq: str) -> pd.Timedelta | pd.DateOffset:
    if len(dates) >= 2:
        return dates[-1] - dates[-2]
    return pd.Timedelta(days=1)


def _forward_fill_rows(matrix: np.ndarray, limit_col: int | None = None) -> np.ndarray:
    """Row-wise forward fill (then back fill) of NaNs, in place-safe fashion."""
    out = matrix.copy()
    window = out[:, :limit_col] if limit_col else out
    mask = np.isnan(window)
    if mask.any():
        idx = np.where(~mask, np.arange(window.shape[1])[None, :], 0)
        np.maximum.accumulate(idx, axis=1, out=idx)
        window[:] = window[np.arange(window.shape[0])[:, None], idx]
        # Anything still NaN precedes the first observation - back fill it.
        still = np.isnan(window)
        if still.any():
            reversed_window = window[:, ::-1]
            rmask = np.isnan(reversed_window)
            ridx = np.where(~rmask, np.arange(window.shape[1])[None, :], 0)
            np.maximum.accumulate(ridx, axis=1, out=ridx)
            filled = reversed_window[np.arange(window.shape[0])[:, None], ridx][:, ::-1]
            window[still] = filled[still]
    if limit_col:
        out[:, :limit_col] = window
    return out


def _project_future(
    matrix: np.ndarray, n_observed: int, freq: str, column: str
) -> np.ndarray:
    """Fill the forecast window of a known-future covariate.

    Real planned values should be supplied through `future_overrides`. When
    they are not, we project the historical pattern rather than inventing a
    number: a seasonal-naive copy for calendar-like flags, the trailing mean
    for continuous covariates.
    """
    out = matrix.copy()
    if n_observed >= out.shape[1]:
        return out

    season = {"D": 7, "W": 52, "MS": 12, "h": 24}.get(freq, 7)
    horizon = out.shape[1] - n_observed
    history = out[:, :n_observed]

    if is_event_like(column) or _looks_binary(history):
        # Calendar flags repeat on their seasonal cycle.
        for step in range(horizon):
            source = n_observed + step - season
            while source >= n_observed:
                source -= season
            out[:, n_observed + step] = out[:, max(source, 0)]
        return out

    if is_price_like(column):
        # Prices: seasonal-naive on the weekly cycle around a trailing level.
        trailing = np.nanmean(history[:, -min(28, n_observed):], axis=1)
        for step in range(horizon):
            source = n_observed + step - season
            while source >= n_observed:
                source -= season
            seasonal = out[:, max(source, 0)]
            base = np.nanmean(history[:, max(0, n_observed - season):n_observed], axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                ratio = np.where(np.abs(base) > 1e-9, seasonal / base, 1.0)
            out[:, n_observed + step] = trailing * np.clip(np.nan_to_num(ratio, nan=1.0), 0.5, 2.0)
        return out

    trailing = np.nanmean(history[:, -min(28, n_observed):], axis=1)
    out[:, n_observed:] = trailing[:, None]
    return out


def _looks_binary(history: np.ndarray) -> bool:
    finite = history[np.isfinite(history)]
    if finite.size == 0:
        return False
    sample = finite[:: max(1, finite.size // 20_000)]
    return bool(np.all(np.isin(sample, (0.0, 1.0))))


def _apply_overrides(
    future: dict[str, np.ndarray],
    overrides: pd.DataFrame,
    entity_pos: dict[str, int],
    date_pos: dict[pd.Timestamp, int],
) -> None:
    frame = overrides.copy()
    frame[TS] = pd.to_datetime(frame[TS])
    frame = frame[frame[TS].isin(date_pos.keys())]
    if frame.empty:
        return
    cols = frame[TS].map(date_pos).to_numpy()
    if ENTITY in frame.columns:
        keep = frame[ENTITY].astype(str).isin(entity_pos.keys())
        frame, cols = frame[keep], cols[keep.to_numpy()]
        rows = frame[ENTITY].astype(str).map(entity_pos).to_numpy()
        for column, matrix in future.items():
            if column in frame.columns:
                values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float32)
                mask = np.isfinite(values)
                matrix[rows[mask], cols[mask]] = values[mask]
    else:
        # Market-wide override (e.g. a holiday calendar) - applies to everyone.
        for column, matrix in future.items():
            if column in frame.columns:
                values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float32)
                mask = np.isfinite(values)
                matrix[:, cols[mask]] = values[mask][None, :]


def _static_table(
    frame: pd.DataFrame, static_features: list[str], entities: np.ndarray
) -> pd.DataFrame:
    # Only what the caller declared static becomes a feature; display-only
    # label columns are deliberately not passed in here.
    columns = [c for c in static_features if c in frame.columns]
    for hierarchy in (DESTINATION, CATEGORY):
        if hierarchy in frame.columns and hierarchy not in columns:
            columns.append(hierarchy)
    if not columns:
        return pd.DataFrame(index=pd.Index(entities, name=ENTITY))
    table = (
        frame.assign(**{ENTITY: frame[ENTITY].astype(str)})
        .groupby(ENTITY)[columns]
        .first()
        .reindex(entities)
    )
    table.index.name = ENTITY
    return table
