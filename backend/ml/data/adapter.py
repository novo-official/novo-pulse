"""Load any dataset described by a `DataContract` into one canonical panel.

This is the seam between "whatever the organisers hand us" and the ML code.
Everything downstream sees only `ds` / `entity_id` / `y` plus typed covariates,
so nothing below this module knows a synthetic column name.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..contract import (
    CATEGORY,
    DESTINATION,
    ENTITY,
    LEVEL_COLUMN,
    MARKET,
    MARKET_VALUE,
    TARGET,
    TS,
    DataContract,
)
from ..paths import REPO_ROOT
from .dates import normalise_digit_series, parse_timestamps
from .frequency import detect_frequency

log = logging.getLogger(__name__)

_SINGLE_ENTITY = "__all__"


# --------------------------------------------------------------------------
def read_tabular(path: str | Path, nrows: int | None = None) -> pd.DataFrame:
    """Read CSV / Parquet / Excel by extension."""
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        frame = pd.read_parquet(path)
        return frame.head(nrows) if nrows else frame
    if suffix in {".xlsx", ".xls", ".xlsm"}:
        return pd.read_excel(path, nrows=nrows)
    if suffix in {".json"}:
        frame = pd.read_json(path)
        return frame.head(nrows) if nrows else frame
    sep = "\t" if suffix in {".tsv", ".tab"} else None
    return pd.read_csv(path, nrows=nrows, sep=sep, engine="python" if sep is None else "c")


@dataclass
class Panel:
    """A canonical, regularly-spaced, long-format panel."""

    frame: pd.DataFrame
    contract: DataContract
    static: pd.DataFrame
    frequency: str = "D"
    future_features: list[str] = field(default_factory=list)
    historical_features: list[str] = field(default_factory=list)
    static_features: list[str] = field(default_factory=list)
    label_columns: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    # ---------------------------------------------------------------- info
    @property
    def entities(self) -> np.ndarray:
        return self.frame[ENTITY].unique()

    @property
    def start(self) -> pd.Timestamp:
        return self.frame[TS].min()

    @property
    def end(self) -> pd.Timestamp:
        return self.frame[TS].max()

    @property
    def n_periods(self) -> int:
        return int(self.frame[TS].nunique())

    def summary(self) -> dict[str, Any]:
        target = self.frame[TARGET]
        return {
            "rows": int(len(self.frame)),
            "entities": int(self.frame[ENTITY].nunique()),
            "periods": self.n_periods,
            "start": self.start.date().isoformat(),
            "end": self.end.date().isoformat(),
            "frequency": self.frequency,
            "target_mean": float(target.mean()),
            "target_std": float(target.std()),
            "target_min": float(target.min()),
            "target_max": float(target.max()),
            "zero_ratio": float((target == 0).mean()),
            "levels": self.available_levels(),
            "future_features": self.future_features,
            "historical_features": self.historical_features,
            "static_features": self.static_features,
            "notes": self.notes,
        }

    def available_levels(self) -> list[str]:
        levels = ["listing"]
        if DESTINATION in self.frame.columns:
            levels.append("destination")
        if CATEGORY in self.frame.columns:
            levels.append("category")
        levels.append("market")
        return levels

    # ----------------------------------------------------------- reshaping
    def level_members(self, level: str) -> list[str]:
        column = LEVEL_COLUMN.get(level)
        if column is None or column not in self.frame.columns:
            return []
        return sorted(str(v) for v in self.frame[column].dropna().unique())

    def aggregate(self, level: str) -> "Panel":
        """Aggregate the panel up to `level` (bottom-up hierarchical view)."""
        if level == "listing":
            return self
        column = LEVEL_COLUMN.get(level)
        if column is None or column not in self.frame.columns:
            raise ValueError(f"Level '{level}' is not available in this dataset")

        sum_cols = [TARGET, *self._numeric_additive_features()]
        mean_cols = [c for c in self._numeric_features() if c not in sum_cols]

        agg: dict[str, str] = {c: "sum" for c in sum_cols}
        agg.update({c: "mean" for c in mean_cols})
        grouped = self.frame.groupby([TS, column], as_index=False).agg(agg)
        grouped = grouped.rename(columns={column: ENTITY})
        if column != ENTITY:
            grouped[ENTITY] = grouped[ENTITY].astype(str)

        static = pd.DataFrame({ENTITY: grouped[ENTITY].unique()}).set_index(ENTITY)
        return Panel(
            frame=grouped.sort_values([ENTITY, TS]).reset_index(drop=True),
            contract=self.contract,
            static=static,
            frequency=self.frequency,
            future_features=[c for c in self.future_features if c in grouped.columns],
            historical_features=[c for c in self.historical_features if c in grouped.columns],
            static_features=[],
            notes=[*self.notes, f"aggregated to level={level}"],
        )

    def _numeric_features(self) -> list[str]:
        cols = [*self.future_features, *self.historical_features]
        return [c for c in cols if pd.api.types.is_numeric_dtype(self.frame[c])]

    def _numeric_additive_features(self) -> list[str]:
        """Covariates that should be summed rather than averaged on roll-up."""
        additive_hints = (
            "count",
            "capacity",
            "nights",
            "revenue",
            "quantity",
            "volume",
            "views",
            "searches",
            "demand",
        )
        return [
            c
            for c in self._numeric_features()
            if any(hint in c.lower() for hint in additive_hints)
        ]


# --------------------------------------------------------------------------
def _looks_like_date(series: pd.Series) -> bool:
    """True when a join key holds dates rather than ids.

    Numeric ids are excluded outright: a 4-digit accommodation code would
    otherwise be read as a year.
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if not pd.api.types.is_object_dtype(series) and not pd.api.types.is_string_dtype(series):
        return False
    sample = series.dropna().astype(str).head(200)
    if sample.empty:
        return False
    return bool(sample.str.contains(r"\d{4}\s*[-/._]\s*\d{1,2}").mean() > 0.8)


class DataAdapter:
    """Turn raw files + a contract into a `Panel`."""

    def __init__(self, contract: DataContract):
        self.contract = contract
        self.join_notes: list[str] = []

    # ------------------------------------------------------------- loading
    def load_raw(self) -> pd.DataFrame:
        if not self.contract.path:
            raise ValueError("Contract has no dataset path")
        frame = read_tabular(self.contract.path)
        for join in self.contract.joins:
            frame = self._merge_side(frame, join)
        return frame

    def _merge_side(self, frame: pd.DataFrame, join) -> pd.DataFrame:
        """Left-merge one side table, reporting anything that went wrong.

        A join that silently does nothing is the worst outcome on competition
        day, so every skip and every unmatched row becomes a note the Data Lab
        shows rather than a log line nobody reads.
        """
        path = Path(join.path)
        if not path.is_absolute():
            path = REPO_ROOT / path
        if not path.exists():
            self.join_notes.append(f"join skipped - file not found: {path.name}")
            log.warning("Join source missing, skipping: %s", path)
            return frame

        side = read_tabular(path)
        keys = [k for k in join.keys if k in frame.columns and k in side.columns]
        if not keys:
            missing = [
                k for k in join.keys if k not in frame.columns or k not in side.columns
            ]
            self.join_notes.append(
                f"join with {path.name} skipped - key(s) {missing} are not in both files"
            )
            log.warning("Join key(s) %s absent, skipping %s", join.keys, path)
            return frame

        overlap = [c for c in side.columns if c in frame.columns and c not in keys]
        side = side.drop(columns=overlap)

        # Two files can spell the same day differently - Jalali on one side,
        # Gregorian on the other, or just a different separator. Matching those
        # as raw strings joins nothing, so date-like keys are parsed on both
        # sides first and compared as real dates.
        for key in keys:
            if not _looks_like_date(frame[key]) or not _looks_like_date(side[key]):
                continue
            left, left_cal = parse_timestamps(frame[key])
            right, right_cal = parse_timestamps(side[key])
            if left.notna().sum() == 0 or right.notna().sum() == 0:
                continue
            frame[key] = left.dt.normalize()
            side[key] = right.dt.normalize()
            if left_cal.calendar != right_cal.calendar:
                self.join_notes.append(
                    f"join key '{key}': {path.name} uses the "
                    f"{right_cal.calendar} calendar and the main file "
                    f"{left_cal.calendar}; both were converted before matching"
                )

        # A side table with duplicate keys would multiply the main frame's rows
        # and inflate every count that follows. Collapse it instead.
        duplicated = int(side.duplicated(subset=keys).sum())
        if duplicated:
            side = side.drop_duplicates(subset=keys, keep="first")
            self.join_notes.append(
                f"{path.name} had {duplicated} duplicate row(s) for {keys}; "
                "kept the first of each so the join cannot multiply rows"
            )

        before = len(frame)
        merged = frame.merge(side, on=keys, how="left", validate="m:1")
        added = [c for c in side.columns if c not in keys]
        if added:
            matched = merged[added[0]].notna().mean() if before else 0.0
            self.join_notes.append(
                f"joined {path.name} on {keys}: "
                f"{len(added)} column(s) added, {matched:.0%} of rows matched"
            )
            if matched == 0:
                self.join_notes.append(
                    f"WARNING: no row matched {path.name} - check the key values, "
                    "not just the key names"
                )
        return merged

    def build(self, frame: pd.DataFrame | None = None) -> Panel:
        raw = self.load_raw() if frame is None else frame.copy()
        return self.normalise(raw)

    # --------------------------------------------------------- normalising
    def normalise(self, raw: pd.DataFrame) -> Panel:
        contract = self.contract
        notes: list[str] = list(self.join_notes)

        counting = contract.aggregation == "count"
        if not counting and not contract.target:
            raise ValueError(
                "The contract declares no target column. Either name one, or set "
                'aggregation="count" if demand is the number of rows (a raw '
                "booking table)."
            )
        required = [contract.timestamp] if counting else [contract.timestamp, contract.target]
        missing = [c for c in required if c and c not in raw.columns]
        if missing:
            raise ValueError(
                f"Contract references column(s) {missing} which are not in the dataset. "
                f"Available: {list(raw.columns)[:25]}"
            )

        work = raw.copy()
        # Iranian exports are frequently Jalali (1403/05/12), sometimes with
        # Persian digits. Parsing those as Gregorian yields NaT for every row,
        # which would silently discard the entire dataset.
        calendar = None if contract.calendar in (None, "auto") else contract.calendar
        work[TS], detection = parse_timestamps(work[contract.timestamp], calendar)
        if detection.calendar == "jalali":
            notes.append(
                f"timestamps read as Jalali dates and converted to Gregorian "
                f"({detection.reason}); sample {detection.sample}"
            )
        elif detection.calendar == "unknown":
            notes.append(
                f"calendar could not be determined ({detection.reason}) - "
                "set schema.calendar explicitly if the dates look wrong"
            )
        bad_dates = int(work[TS].isna().sum())
        if bad_dates:
            notes.append(f"dropped {bad_dates} rows with unparseable timestamps")
            work = work.dropna(subset=[TS])

        if counting:
            # A booking table is one row per booking: demand is the row count,
            # not the value of any column. Each row contributes 1.
            work[TARGET] = 1.0
            notes.append(
                "aggregation='count': demand is the number of rows per "
                "(entity, period), which is how a transactional booking table "
                "becomes a demand series"
            )
        else:
            # Numeric columns can carry Persian digits too.
            work[TARGET] = pd.to_numeric(
                normalise_digit_series(work[contract.target]), errors="coerce"
            )
            bad_target = int(work[TARGET].isna().sum())
            if bad_target:
                notes.append(f"dropped {bad_target} rows with non-numeric target")
                work = work.dropna(subset=[TARGET])

        if contract.entity_id and contract.entity_id in work.columns:
            work[ENTITY] = work[contract.entity_id].astype(str)
        else:
            work[ENTITY] = _SINGLE_ENTITY
            notes.append("no entity column - treating the dataset as a single series")

        if contract.destination and contract.destination in work.columns:
            work[DESTINATION] = work[contract.destination].astype(str)
        if contract.category and contract.category in work.columns:
            work[CATEGORY] = work[contract.category].astype(str)
        work[MARKET] = MARKET_VALUE

        # -- frequency ----------------------------------------------------
        detected = detect_frequency(work[TS])
        frequency = contract.frequency or detected["frequency"]
        frequency = {"D": "D", "W": "W", "M": "MS", "MS": "MS", "H": "h", "h": "h"}.get(
            frequency, frequency
        )
        if not detected["regular"]:
            notes.append(
                f"irregular timestamps detected (median spacing "
                f"{detected['median_delta_days']}d) - resampled to {frequency}"
            )
        work[TS] = work[TS].dt.floor("D") if frequency in {"D", "W", "MS"} else work[TS]

        # -- feature roles ------------------------------------------------
        present = set(work.columns)
        future = [c for c in contract.future_features if c in present]
        historical = [c for c in contract.historical_features if c in present]
        static = [c for c in contract.static_features if c in present]
        dropped = [
            c
            for c in (
                contract.future_features + contract.historical_features + contract.static_features
            )
            if c not in present
        ]
        if dropped:
            notes.append(f"contract features absent from data, ignored: {dropped}")

        # The target must never leak in through a covariate of the same name.
        for group in (future, historical, static):
            if contract.target in group:
                group.remove(contract.target)
                notes.append("target column removed from the covariate list")

        # -- collapse duplicates + regular grid ---------------------------
        keep = [TS, ENTITY, TARGET, MARKET]
        for column in (DESTINATION, CATEGORY):
            if column in work.columns:
                keep.append(column)
        keep.extend([c for c in (*future, *historical) if c not in keep])
        keep.extend([c for c in static if c not in keep])
        # Label columns ride along for display; they never become features.
        label_columns = [c for c in contract.labels.values() if c in present]
        keep.extend([c for c in label_columns if c not in keep])
        work = work.loc[:, [c for c in dict.fromkeys(keep)]]

        panel_frame, dup_note = self._collapse(work, future + historical, static, frequency)
        if dup_note:
            notes.append(dup_note)

        static_frame = self._static_table(work, static + label_columns)
        panel_frame = self._regular_grid(panel_frame, frequency, future + historical, notes)

        # Static features live in one table keyed by entity; drop the per-row
        # copies first so the join cannot produce _x/_y duplicates.
        if not static_frame.empty:
            panel_frame = panel_frame.drop(
                columns=[c for c in static_frame.columns if c in panel_frame.columns]
            )
            panel_frame = panel_frame.merge(
                static_frame, left_on=ENTITY, right_index=True, how="left"
            )

        panel_frame = panel_frame.sort_values([ENTITY, TS]).reset_index(drop=True)

        return Panel(
            frame=panel_frame,
            contract=contract,
            static=static_frame,
            frequency=frequency,
            future_features=future,
            historical_features=historical,
            static_features=static,
            label_columns=label_columns,
            notes=notes,
        )

    # ------------------------------------------------------------ helpers
    def _collapse(
        self,
        work: pd.DataFrame,
        covariates: list[str],
        static: list[str],
        frequency: str,
    ) -> tuple[pd.DataFrame, str | None]:
        """Collapse duplicate (entity, timestamp) rows using contract rules."""
        group_cols = [ENTITY, TS]
        n_before = len(work)
        duplicated = int(work.duplicated(subset=group_cols).sum())
        if duplicated == 0 and self.contract.aggregation != "count":
            return work, None

        # "count" is expressed as summing the per-row 1s assigned above, which
        # keeps one code path for every aggregation.
        agg: dict[str, Any] = {
            TARGET: "sum" if self.contract.aggregation == "count" else self.contract.aggregation
        }
        for column in covariates:
            if column in static:
                agg[column] = "first"
            elif pd.api.types.is_numeric_dtype(work[column]):
                agg[column] = "sum" if _is_additive(column) else "mean"
            else:
                agg[column] = "first"
        for column in (DESTINATION, CATEGORY, MARKET):
            if column in work.columns:
                agg[column] = "first"
        for column in static:
            agg.setdefault(column, "first")

        collapsed = work.groupby(group_cols, as_index=False).agg(agg)
        return (
            collapsed,
            f"collapsed {duplicated} duplicate (entity, timestamp) rows "
            f"with aggregation='{self.contract.aggregation}' ({n_before} -> {len(collapsed)})",
        )

    def _static_table(self, work: pd.DataFrame, static: list[str]) -> pd.DataFrame:
        if not static:
            return pd.DataFrame(index=pd.Index([], name=ENTITY))
        table = work.groupby(ENTITY)[static].first()
        table.index.name = ENTITY
        return table

    def _regular_grid(
        self,
        frame: pd.DataFrame,
        frequency: str,
        covariates: list[str],
        notes: list[str],
    ) -> pd.DataFrame:
        """Reindex every entity onto a gap-free timeline.

        Each entity starts at its own first observation (so genuinely new
        listings are not back-filled with fake history) and ends at the global
        last timestamp.
        """
        global_end = frame[TS].max()
        pieces: list[pd.DataFrame] = []
        filled_total = 0
        fill_value = 0.0 if self.contract.target_options.non_negative else np.nan
        static_like = set(self.contract.static_features) | {DESTINATION, CATEGORY, MARKET}

        for entity, group in frame.groupby(ENTITY, sort=False):
            group = group.sort_values(TS)
            index = pd.date_range(group[TS].min(), global_end, freq=frequency)
            if len(index) == len(group) and (group[TS].to_numpy() == index.to_numpy()).all():
                pieces.append(group)
                continue
            reindexed = (
                group.set_index(TS).reindex(index).rename_axis(TS).reset_index()
            )
            filled_total += int(reindexed[TARGET].isna().sum())
            reindexed[ENTITY] = entity
            for column in reindexed.columns:
                if column in {TS, ENTITY, TARGET}:
                    continue
                if column in static_like or not pd.api.types.is_numeric_dtype(reindexed[column]):
                    reindexed[column] = reindexed[column].ffill().bfill()
                else:
                    reindexed[column] = reindexed[column].interpolate(limit_direction="both")
            reindexed[TARGET] = reindexed[TARGET].fillna(fill_value)
            pieces.append(reindexed)

        if filled_total:
            how = "0" if self.contract.target_options.non_negative else "NaN"
            notes.append(f"filled {filled_total} missing periods in the grid with {how}")
        return pd.concat(pieces, ignore_index=True)


def _is_additive(column: str) -> bool:
    hints = ("count", "nights", "revenue", "quantity", "volume", "views", "searches")
    return any(hint in column.lower() for hint in hints)


def load_panel(contract: DataContract | None = None) -> Panel:
    """Convenience: contract -> panel in one call."""
    contract = contract or DataContract.load()
    return DataAdapter(contract).build()
