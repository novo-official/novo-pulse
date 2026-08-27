"""Leakage-safe supervised frame construction.

One sample is the triple `(entity e, forecast origin o, horizon h)` predicting
`y[e, o + h]`. Features are drawn from exactly two places:

  * index <= o  - lags, rolling statistics, past covariates  (historical)
  * index  = o + h - calendar and known-future covariates    (known-future)

Nothing else is reachable, which is what makes the direct multi-horizon setup
safe for 90-day forecasts: the model never needs a value it will not have.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..contract import CATEGORY, DESTINATION, ENTITY
from .calendar import calendar_frame, event_distance_features, is_event_like, is_price_like
from .rolling import gather, rolling_mean_std, rolling_min_max
from .tensor import PanelTensor

EPS = 1e-6

# Lags and windows per frequency, in native periods.
FREQ_LAGS = {
    "D": [1, 2, 3, 7, 14, 21, 28, 56],
    "W": [1, 2, 4, 8, 13, 26, 52],
    "MS": [1, 2, 3, 6, 12],
    "h": [1, 2, 3, 24, 48, 168],
}
FREQ_WINDOWS = {
    "D": [7, 14, 28, 56],
    "W": [4, 8, 26],
    "MS": [3, 6, 12],
    "h": [24, 168],
}
FREQ_SEASON = {"D": 7, "W": 52, "MS": 12, "h": 24}


@dataclass
class FeatureConfig:
    lags: list[int] = field(default_factory=list)
    windows: list[int] = field(default_factory=list)
    season: int = 7
    max_horizon: int = 30
    min_context: int = 14
    use_static: bool = True
    use_entity_id: bool = True

    @classmethod
    def for_frequency(cls, freq: str, max_horizon: int = 30, **kwargs) -> "FeatureConfig":
        return cls(
            lags=FREQ_LAGS.get(freq, FREQ_LAGS["D"]),
            windows=FREQ_WINDOWS.get(freq, FREQ_WINDOWS["D"]),
            season=FREQ_SEASON.get(freq, 7),
            max_horizon=max_horizon,
            **kwargs,
        )


@dataclass
class SupervisedFrame:
    X: pd.DataFrame
    y: np.ndarray | None
    meta: pd.DataFrame           # entity_id, ds, horizon, origin_ds
    categorical: list[str]

    def __len__(self) -> int:
        return len(self.X)


class FeatureEngine:
    """Builds supervised frames from a `PanelTensor`."""

    def __init__(self, tensor: PanelTensor, config: FeatureConfig | None = None):
        self.tensor = tensor
        self.config = config or FeatureConfig.for_frequency(tensor.freq)
        self._roll_mean: dict[int, np.ndarray] = {}
        self._roll_std: dict[int, np.ndarray] = {}
        self._roll_min: dict[int, np.ndarray] = {}
        self._roll_max: dict[int, np.ndarray] = {}
        self._past_roll: dict[tuple[str, int], np.ndarray] = {}
        self._future_roll: dict[tuple[str, int], np.ndarray] = {}
        self._calendar: pd.DataFrame | None = None
        self._event_distance: dict[str, np.ndarray] = {}
        self._static_encoded: pd.DataFrame | None = None
        self._categorical: list[str] = []
        self._feature_names: list[str] | None = None
        self._prepared = False

    # ------------------------------------------------------------- prepare
    def prepare(self) -> "FeatureEngine":
        if self._prepared:
            return self
        tensor = self.tensor
        y_hist = tensor.y.copy()
        # Only observed history may feed a rolling statistic.
        y_hist[~tensor.observed] = np.nan

        for window in self.config.windows:
            mean, std = rolling_mean_std(y_hist, window)
            self._roll_mean[window] = mean
            self._roll_std[window] = std
        minmax_windows = [w for w in self.config.windows if w <= 28][:2]
        for window in minmax_windows:
            minimum, maximum = rolling_min_max(y_hist, window)
            self._roll_min[window] = minimum
            self._roll_max[window] = maximum

        past_windows = [w for w in self.config.windows if w <= 28][:2] or [self.config.season]
        for name, matrix in tensor.past.items():
            for window in past_windows:
                mean, _ = rolling_mean_std(matrix, window)
                self._past_roll[(name, window)] = mean

        for name, matrix in tensor.future.items():
            if is_price_like(name):
                mean, _ = rolling_mean_std(matrix, max(self.config.windows[0], 7))
                self._future_roll[(name, 0)] = mean

        self._calendar = calendar_frame(tensor.dates, tensor.freq)
        for name, matrix in tensor.future.items():
            if is_event_like(name):
                # Flags are shared across entities; take the market-wide profile.
                flag = np.nanmax(matrix, axis=0)
                distances = event_distance_features(flag)
                self._event_distance[f"{name}_days_to"] = distances["days_to"]
                self._event_distance[f"{name}_days_from"] = distances["days_from"]

        self._static_encoded, self._categorical = self._encode_static()
        self._prepared = True
        return self

    def _encode_static(self) -> tuple[pd.DataFrame, list[str]]:
        tensor = self.tensor
        table = tensor.static.copy()
        categorical: list[str] = []

        if self.config.use_entity_id:
            table["ent_entity"] = pd.Categorical(tensor.entities)
            categorical.append("ent_entity")
        for column, alias in ((DESTINATION, "ent_destination"), (CATEGORY, "ent_category")):
            if column in table.columns:
                table[alias] = pd.Categorical(table[column].astype(str))
                categorical.append(alias)

        keep: list[str] = list(categorical)
        if self.config.use_static:
            for column in tensor.static.columns:
                if column in {DESTINATION, CATEGORY}:
                    continue
                series = tensor.static[column]
                name = f"stat_{column}"
                if pd.api.types.is_numeric_dtype(series):
                    table[name] = pd.to_numeric(series, errors="coerce").astype(np.float32).to_numpy()
                    keep.append(name)
                else:
                    unique = series.nunique(dropna=True)
                    if 1 < unique <= 200:
                        table[name] = pd.Categorical(series.astype(str))
                        keep.append(name)
                        categorical.append(name)
        encoded = table.loc[:, keep].reset_index(drop=True)
        return encoded, categorical

    # ------------------------------------------------------------- assemble
    def assemble(
        self, entity_idx: np.ndarray, target_idx: np.ndarray, horizons: np.ndarray
    ) -> SupervisedFrame:
        """Build the design matrix for arbitrary (entity, target, horizon) triples."""
        self.prepare()
        tensor = self.tensor
        rows = np.asarray(entity_idx, dtype=np.int64)
        targets = np.asarray(target_idx, dtype=np.int64)
        horizon = np.asarray(horizons, dtype=np.int64)
        origins = targets - horizon

        y_hist = tensor.y.copy()
        y_hist[~tensor.observed] = np.nan

        columns: dict[str, np.ndarray] = {}

        # -- horizon ---------------------------------------------------------
        columns["horizon"] = horizon.astype(np.float32)
        columns["horizon_log"] = np.log1p(horizon).astype(np.float32)

        # -- target history (read at or before the origin) --------------------
        columns["y_last"] = gather(y_hist, rows, origins)
        for lag in self.config.lags:
            columns[f"y_lag_{lag}"] = gather(y_hist, rows, origins - lag)

        for window in self.config.windows:
            columns[f"y_roll_mean_{window}"] = gather(self._roll_mean[window], rows, origins)
            columns[f"y_roll_std_{window}"] = gather(self._roll_std[window], rows, origins)
        for window in self._roll_min:
            columns[f"y_roll_min_{window}"] = gather(self._roll_min[window], rows, origins)
            columns[f"y_roll_max_{window}"] = gather(self._roll_max[window], rows, origins)

        short, long = self.config.windows[0], self.config.windows[-1]
        with np.errstate(invalid="ignore", divide="ignore"):
            columns["y_trend_ratio"] = (
                columns[f"y_roll_mean_{short}"] / (columns[f"y_roll_mean_{long}"] + EPS)
            ).astype(np.float32)
            columns["y_cv"] = (
                columns[f"y_roll_std_{short}"] / (columns[f"y_roll_mean_{short}"] + EPS)
            ).astype(np.float32)

        # -- seasonal reference aligned to the *target* date ------------------
        season = self.config.season
        cycles = np.ceil(horizon / season).astype(np.int64)
        columns["y_seasonal_naive"] = gather(y_hist, rows, targets - season * cycles)
        same_phase = [
            gather(y_hist, rows, targets - season * (cycles + offset)) for offset in range(4)
        ]
        stacked = np.vstack(same_phase)
        with np.errstate(invalid="ignore"):
            columns["y_same_phase_mean"] = np.nanmean(stacked, axis=0).astype(np.float32)
            columns["y_same_phase_std"] = np.nanstd(stacked, axis=0).astype(np.float32)

        # -- data sufficiency -------------------------------------------------
        starts = tensor.entity_start_index()[rows]
        columns["entity_history_len"] = (origins - starts + 1).astype(np.float32)

        # -- past covariates: strictly at the origin --------------------------
        for name, matrix in tensor.past.items():
            columns[f"past_{name}_last"] = gather(matrix, rows, origins)
            for (cov, window), rolled in self._past_roll.items():
                if cov != name:
                    continue
                mean_at_origin = gather(rolled, rows, origins)
                columns[f"past_{name}_mean_{window}"] = mean_at_origin
                # Funnel-style ratio: covariate level per unit of recent demand.
                base = columns.get(f"y_roll_mean_{window}")
                if base is not None:
                    with np.errstate(invalid="ignore", divide="ignore"):
                        columns[f"past_{name}_per_demand_{window}"] = (
                            mean_at_origin / (base + 1.0)
                        ).astype(np.float32)

        # -- known-future covariates: read at the target date -----------------
        for name, matrix in tensor.future.items():
            at_target = gather(matrix, rows, targets)
            columns[f"fut_{name}"] = at_target
            at_origin = gather(matrix, rows, origins)
            with np.errstate(invalid="ignore", divide="ignore"):
                columns[f"fut_{name}_vs_origin"] = (
                    at_target / (np.abs(at_origin) + EPS)
                ).astype(np.float32)
            if (name, 0) in self._future_roll:
                baseline = gather(self._future_roll[(name, 0)], rows, origins)
                with np.errstate(invalid="ignore", divide="ignore"):
                    columns[f"fut_{name}_vs_recent_avg"] = (
                        at_target / (np.abs(baseline) + EPS)
                    ).astype(np.float32)

        # -- calendar at the target date --------------------------------------
        calendar = self._calendar
        for column in calendar.columns:
            columns[column] = calendar[column].to_numpy()[targets].astype(np.float32)
        for name, values in self._event_distance.items():
            columns[f"cal_{name}"] = values[targets].astype(np.float32)

        X = pd.DataFrame({k: np.asarray(v, dtype=np.float32) for k, v in columns.items()})

        # -- static / identity block ------------------------------------------
        static = self._static_encoded.iloc[rows].reset_index(drop=True)
        X = pd.concat([X, static], axis=1)
        for column in self._categorical:
            if column in X.columns:
                X[column] = X[column].astype("category")

        if self._feature_names is None:
            self._feature_names = list(X.columns)
        else:
            X = X.reindex(columns=self._feature_names)

        target_values = None
        observed_target = tensor.observed[rows, np.clip(targets, 0, tensor.n_periods - 1)]
        if observed_target.any():
            target_values = gather(tensor.y, rows, targets)

        meta = pd.DataFrame(
            {
                ENTITY: tensor.entities[rows],
                "ds": tensor.dates[targets],
                "origin_ds": tensor.dates[np.clip(origins, 0, tensor.n_periods - 1)],
                "horizon": horizon,
                # Positional keys so series models (baselines, Chronos, NHITS)
                # can read the tensor directly and stay row-aligned with X.
                "entity_pos": rows,
                "target_idx": targets,
                "origin_idx": origins,
            }
        )
        return SupervisedFrame(X=X, y=target_values, meta=meta, categorical=list(self._categorical))

    # ------------------------------------------------------------- training
    def build_training(
        self,
        train_end_idx: int,
        samples_per_target: int = 3,
        max_rows: int | None = None,
        seed: int = 42,
        horizons: list[int] | None = None,
    ) -> SupervisedFrame:
        """Sample (entity, target, horizon) triples from the training window.

        Rather than materialising every (target, horizon) pair - which is
        `n_entities x n_periods x max_horizon` rows - we draw a stratified
        sample of horizons per target date. Every horizon bucket stays
        represented while the matrix stays laptop-sized.
        """
        self.prepare()
        tensor = self.tensor
        rng = np.random.default_rng(seed)
        max_h = self.config.max_horizon
        starts = tensor.entity_start_index()

        entity_rows: list[np.ndarray] = []
        target_rows: list[np.ndarray] = []
        horizon_rows: list[np.ndarray] = []

        for e in range(tensor.n_entities):
            first_target = starts[e] + self.config.min_context + 1
            if first_target > train_end_idx:
                continue
            candidates = np.arange(first_target, train_end_idx + 1)
            candidates = candidates[tensor.observed[e, candidates]]
            if candidates.size == 0:
                continue

            for _ in range(max(1, samples_per_target)):
                if horizons:
                    drawn = rng.choice(horizons, size=candidates.size)
                else:
                    # Log-uniform: short horizons matter most but the long tail
                    # still gets covered.
                    drawn = np.floor(
                        np.exp(rng.uniform(0, np.log(max_h + 1), candidates.size))
                    ).astype(np.int64)
                drawn = np.clip(drawn, 1, max_h)
                origins = candidates - drawn
                keep = origins >= starts[e] + self.config.min_context
                if not keep.any():
                    continue
                entity_rows.append(np.full(keep.sum(), e, dtype=np.int64))
                target_rows.append(candidates[keep])
                horizon_rows.append(drawn[keep])

        if not entity_rows:
            raise ValueError(
                "No trainable samples. The history is too short for the requested "
                f"horizon ({max_h}) and context ({self.config.min_context})."
            )

        entities = np.concatenate(entity_rows)
        targets = np.concatenate(target_rows)
        horizon = np.concatenate(horizon_rows)

        if max_rows and len(entities) > max_rows:
            pick = rng.choice(len(entities), size=max_rows, replace=False)
            entities, targets, horizon = entities[pick], targets[pick], horizon[pick]

        frame = self.assemble(entities, targets, horizon)
        mask = np.isfinite(frame.y)
        if not mask.all():
            frame = SupervisedFrame(
                X=frame.X.loc[mask].reset_index(drop=True),
                y=frame.y[mask],
                meta=frame.meta.loc[mask].reset_index(drop=True),
                categorical=frame.categorical,
            )
        return frame

    # ------------------------------------------------------------ inference
    def build_inference(
        self, origin_idx: int, horizon: int, entity_filter: np.ndarray | None = None
    ) -> SupervisedFrame:
        """Every (entity, h) pair for h = 1..horizon from a single origin."""
        self.prepare()
        tensor = self.tensor
        entity_positions = (
            np.arange(tensor.n_entities) if entity_filter is None else np.asarray(entity_filter)
        )
        # Entities with no history at all cannot be forecast from this origin.
        starts = tensor.entity_start_index()
        entity_positions = entity_positions[starts[entity_positions] <= origin_idx]

        steps = np.arange(1, horizon + 1)
        entities = np.repeat(entity_positions, horizon)
        horizons = np.tile(steps, len(entity_positions))
        targets = origin_idx + horizons
        valid = targets < tensor.n_periods
        return self.assemble(entities[valid], targets[valid], horizons[valid])

    # ------------------------------------------------------------- metadata
    @property
    def feature_names(self) -> list[str]:
        self.prepare()
        if self._feature_names is None:
            sample = self.assemble(np.array([0]), np.array([self.tensor.origin_index]), np.array([1]))
            self._feature_names = list(sample.X.columns)
        return list(self._feature_names)

    @property
    def categorical_features(self) -> list[str]:
        self.prepare()
        return list(self._categorical)

    def feature_groups(self) -> dict[str, str]:
        """Map each feature to a human-facing driver group (used by SHAP)."""
        groups: dict[str, str] = {}
        for name in self.feature_names:
            if name.startswith("cal_"):
                if "holiday" in name or "event" in name:
                    groups[name] = "holiday"
                elif "weekend" in name or "dow" in name or "day_of_week" in name:
                    groups[name] = "weekday"
                elif "month" in name or "doy" in name or "quarter" in name or "week_of_year" in name:
                    groups[name] = "season"
                else:
                    groups[name] = "calendar"
            elif name.startswith("fut_"):
                if is_price_like(name):
                    groups[name] = "price"
                elif is_event_like(name):
                    groups[name] = "holiday"
                elif "capacity" in name or "availab" in name:
                    groups[name] = "availability"
                elif "promo" in name or "discount" in name:
                    groups[name] = "promotion"
                else:
                    groups[name] = "planned_covariates"
            elif name.startswith("past_"):
                if "search" in name or "view" in name or "click" in name:
                    groups[name] = "search_activity"
                else:
                    groups[name] = "recent_behaviour"
            elif name.startswith("y_"):
                groups[name] = "demand_history"
            elif name.startswith("stat_") or name.startswith("ent_"):
                groups[name] = "entity_profile"
            elif name.startswith("horizon"):
                groups[name] = "horizon"
            else:
                groups[name] = "other"
        return groups
