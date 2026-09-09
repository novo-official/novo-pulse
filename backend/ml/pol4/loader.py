"""Load and validate the three Pol 4 files, preserving the event structure.

The raw search log is kept as (log_date, city_code, checkin, search_count)
rows. Nothing is pre-aggregated at load time, because every aggregation in this
package is *cutoff-dependent* and collapsing the log would throw away the log
axis that makes the cutoff meaningful.

A missing row means zero searches occurred for that (city, check-in, log date).
It does NOT mean the pair's final demand is zero - more searches can still
arrive before the check-in. That distinction is the whole competition.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import MAX_LEAD_TIME, N_CITIES, Pol4Config

log = logging.getLogger(__name__)

LOG_DATE = "log_date"
CITY = "city_code"
CHECKIN = "checkin"
SEARCHES = "search_count"
DTC = "days_to_checkin"
PROVINCE = "province_code"
CITY_NAME = "city"
PROVINCE_NAME = "province"

SEARCH_COLUMNS = (LOG_DATE, CITY, CHECKIN, SEARCHES)
CITY_COLUMNS = (CITY, PROVINCE, "lat", "long")
NAME_COLUMNS = (CITY_NAME, CITY, PROVINCE_NAME, PROVINCE)


class Pol4DataError(ValueError):
    """Raised when a dataset does not match the competition contract."""


@dataclass
class Pol4Data:
    """The competition datasets, validated and typed.

    `search` is history: every check-in in it is complete, because a search for
    check-in T can only occur in [T-59, T] and the file ends at the cutoff.

    `evaluation` is the opposite: partial observations of check-ins that have
    not happened yet.
    """

    search: pd.DataFrame           # log_date, city_code, checkin, search_count, days_to_checkin
    evaluation: pd.DataFrame       # same schema, Azar 1404 check-ins only
    cities: pd.DataFrame           # city_code, province_code, lat, long
    findings: list[dict[str, Any]] = field(default_factory=list)

    # ------------------------------------------------------------- lookups
    @property
    def city_codes(self) -> np.ndarray:
        return np.sort(self.cities[CITY].unique())

    @property
    def province_of(self) -> pd.Series:
        return self.cities.set_index(CITY)[PROVINCE]

    @property
    def has_names(self) -> bool:
        return CITY_NAME in self.cities.columns

    @property
    def city_name_of(self) -> pd.Series:
        """city_code -> readable name, falling back to the code as a string.

        Names are for humans. `results.csv` still carries the numeric
        `cluster_code` the organisers asked for - see `submission.py`.
        """
        if self.has_names:
            return self.cities.set_index(CITY)[CITY_NAME]
        return pd.Series(
            self.cities[CITY].astype(str).to_numpy(), index=self.cities[CITY], name=CITY_NAME
        )

    @property
    def province_name_of(self) -> pd.Series:
        if PROVINCE_NAME in self.cities.columns:
            return self.cities.set_index(CITY)[PROVINCE_NAME]
        return pd.Series(
            self.cities[PROVINCE].astype(str).to_numpy(),
            index=self.cities[CITY],
            name=PROVINCE_NAME,
        )

    def name(self, city_codes) -> np.ndarray:
        """Readable names for an array of city codes."""
        return self.city_name_of.reindex(pd.Index(np.asarray(city_codes))).to_numpy()

    def label(self, frame: pd.DataFrame, column: str = CITY) -> pd.DataFrame:
        """Insert `city` and `province` name columns next to a code column."""
        out = frame.copy()
        codes = pd.Index(out[column])
        out.insert(
            out.columns.get_loc(column) + 1, CITY_NAME, self.city_name_of.reindex(codes).to_numpy()
        )
        out.insert(
            out.columns.get_loc(CITY_NAME) + 1,
            PROVINCE_NAME,
            self.province_name_of.reindex(codes).to_numpy(),
        )
        return out

    def history_before(self, cutoff: pd.Timestamp) -> pd.DataFrame:
        """Search rows for check-ins that are already complete at `cutoff`.

        A check-in at or before the cutoff has received every search it will
        ever receive (the last possible one lands on the check-in date itself),
        so these are the only rows from which a *final* demand - and therefore a
        completion fraction - may be computed.
        """
        return self.search[self.search[CHECKIN] <= cutoff]

    def observed_at(
        self, cutoff: pd.Timestamp, start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.DataFrame:
        """Demand observed so far, for check-ins in [start, end], as of `cutoff`.

        This is the exact quantity `evaluation.csv` carries for the real
        competition window, reconstructed from history for a simulated one.
        """
        frames = [self.search, self.evaluation]
        rows = pd.concat([f for f in frames if len(f)], ignore_index=True)
        mask = (
            (rows[LOG_DATE] <= cutoff)
            & (rows[CHECKIN] >= start)
            & (rows[CHECKIN] <= end)
        )
        out = rows.loc[mask].groupby([CITY, CHECKIN], as_index=False)[SEARCHES].sum()
        return out.rename(columns={SEARCHES: "observed"})

    def final_demand(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """Complete demand for check-ins in [start, end]  (history only).

        `final_demand(city, checkin) = SUM(search_count)` over every log date,
        which is the competition's own definition of demand.
        """
        rows = self.search
        mask = (rows[CHECKIN] >= start) & (rows[CHECKIN] <= end)
        out = rows.loc[mask].groupby([CITY, CHECKIN], as_index=False)[SEARCHES].sum()
        return out.rename(columns={SEARCHES: "final"})

    def summary(self) -> dict[str, Any]:
        return {
            "search_rows": int(len(self.search)),
            "city_names": bool(self.has_names),
            "evaluation_rows": int(len(self.evaluation)),
            "cities": int(len(self.cities)),
            "provinces": int(self.cities[PROVINCE].nunique()),
            "log_date_min": str(self.search[LOG_DATE].min().date()),
            "log_date_max": str(self.search[LOG_DATE].max().date()),
            "checkin_min": str(self.search[CHECKIN].min().date()),
            "checkin_max": str(self.search[CHECKIN].max().date()),
            "max_lead_time": int(self.search[DTC].max()),
            "total_searches": int(self.search[SEARCHES].sum()),
            "evaluation_checkin_min": (
                str(self.evaluation[CHECKIN].min().date()) if len(self.evaluation) else None
            ),
            "evaluation_checkin_max": (
                str(self.evaluation[CHECKIN].max().date()) if len(self.evaluation) else None
            ),
            "evaluation_searches": int(self.evaluation[SEARCHES].sum()),
            "findings": self.findings,
        }


# --------------------------------------------------------------------- load
def _read_events(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise Pol4DataError(
            f"{label} not found at {path}. Place the competition CSVs in "
            f"{path.parent} (search_data.csv, evaluation.csv, cities.csv)."
        )
    frame = pd.read_csv(
        path,
        dtype={CITY: "int32", SEARCHES: "int64"},
        parse_dates=[LOG_DATE, CHECKIN],
    )
    missing = [c for c in SEARCH_COLUMNS if c not in frame.columns]
    if missing:
        raise Pol4DataError(f"{label} is missing column(s): {missing}")
    return frame.loc[:, list(SEARCH_COLUMNS)]


def _validate_events(
    frame: pd.DataFrame, label: str, findings: list[dict[str, Any]], strict: bool
) -> pd.DataFrame:
    """Check the invariants the brief guarantees, and record what we found."""
    problems: list[str] = []

    duplicates = int(frame.duplicated([LOG_DATE, CITY, CHECKIN]).sum())
    if duplicates:
        problems.append(f"{duplicates} duplicate (log_date, city_code, checkin) rows")

    if frame[SEARCHES].min() < 1:
        problems.append(
            f"{int((frame[SEARCHES] < 1).sum())} rows with search_count < 1 "
            "(a logged row should record at least one search)"
        )

    frame = frame.assign(**{DTC: (frame[CHECKIN] - frame[LOG_DATE]).dt.days.astype("int16")})

    impossible = int((frame[DTC] < 0).sum())
    if impossible:
        problems.append(f"{impossible} rows where checkin < log_date")

    over_window = int((frame[DTC] > MAX_LEAD_TIME).sum())
    if over_window:
        # Not fatal: it would simply mean the booking window is wider than the
        # 60 days this dataset shows. The curves adapt; the constant does not.
        findings.append(
            {
                "severity": "medium",
                "dataset": label,
                "title": "lead time exceeds the expected booking window",
                "detail": (
                    f"{over_window} rows have days_to_checkin > {MAX_LEAD_TIME}; "
                    "pickup curves will still cover them."
                ),
            }
        )

    nulls = int(frame.isna().any(axis=1).sum())
    if nulls:
        problems.append(f"{nulls} rows with a null in a required column")

    if problems:
        detail = "; ".join(problems)
        if strict:
            raise Pol4DataError(f"{label} failed validation: {detail}")
        findings.append(
            {"severity": "high", "dataset": label, "title": "validation failed", "detail": detail}
        )
    else:
        findings.append(
            {
                "severity": "info",
                "dataset": label,
                "title": "validated",
                "detail": (
                    f"{len(frame):,} rows, no duplicates, no impossible dates, "
                    f"lead time 0-{int(frame[DTC].max())} days"
                ),
            }
        )
    return frame


def _attach_names(
    cities: pd.DataFrame,
    config: Pol4Config,
    findings: list[dict[str, Any]],
    strict: bool,
) -> pd.DataFrame:
    """Join readable city and province names, when the mapping is present.

    The mapping is optional on purpose: the pipeline must still run on a clone
    that only has the three official files. Without it every report falls back
    to the numeric codes rather than failing.
    """
    path = config.city_names_path
    if not path.exists():
        findings.append(
            {
                "severity": "low",
                "dataset": path.name,
                "title": "no city name mapping",
                "detail": (
                    f"{path} not found; reports will show numeric city codes. "
                    "Add the mapping to label them."
                ),
            }
        )
        return cities

    names = pd.read_csv(path, dtype={CITY: "int32"}).dropna(how="all")
    missing = [c for c in NAME_COLUMNS if c not in names.columns]
    if missing:
        raise Pol4DataError(f"{path.name} is missing column(s): {missing}")
    names = names.loc[:, list(NAME_COLUMNS)].drop_duplicates(CITY)

    unnamed = sorted(set(cities[CITY]) - set(names[CITY]))
    if unnamed:
        detail = f"{len(unnamed)} city(ies) have no name in {path.name}, e.g. {unnamed[:5]}"
        if strict:
            raise Pol4DataError(detail)
        findings.append(
            {"severity": "medium", "dataset": path.name, "title": "unnamed cities", "detail": detail}
        )

    merged = cities.merge(names, on=CITY, how="left", suffixes=("", "_named"))
    # The mapping repeats province_code; it must agree with cities.csv or one of
    # the two files is wrong about which province a city is in.
    if f"{PROVINCE}_named" in merged.columns:
        clash = merged[PROVINCE] != merged[f"{PROVINCE}_named"]
        if clash.any():
            detail = (
                f"{int(clash.sum())} city(ies) have a different province_code in "
                f"{path.name} than in cities.csv"
            )
            if strict:
                raise Pol4DataError(detail)
            findings.append(
                {
                    "severity": "high",
                    "dataset": path.name,
                    "title": "province mismatch",
                    "detail": detail,
                }
            )
        merged = merged.drop(columns=[f"{PROVINCE}_named"])

    findings.append(
        {
            "severity": "info",
            "dataset": path.name,
            "title": "city names attached",
            "detail": f"{merged[CITY_NAME].notna().sum()} of {len(merged)} cities named",
        }
    )
    return merged


def load_pol4(config: Pol4Config | None = None, strict: bool = True) -> Pol4Data:
    """Load search_data.csv, evaluation.csv and cities.csv with validation.

    `strict=False` downgrades contract violations to findings, which is what a
    dashboard wants; the pipeline keeps the default and refuses to model a
    dataset that does not match the brief.
    """
    config = config or Pol4Config()
    findings: list[dict[str, Any]] = []

    search = _read_events(config.search_path, "search_data.csv")
    search = _validate_events(search, "search_data.csv", findings, strict)

    evaluation = _read_events(config.evaluation_path, "evaluation.csv")
    evaluation = _validate_events(evaluation, "evaluation.csv", findings, strict)

    if not config.cities_path.exists():
        raise Pol4DataError(f"cities.csv not found at {config.cities_path}")
    cities = pd.read_csv(config.cities_path, dtype={CITY: "int32", PROVINCE: "int32"})
    missing = [c for c in CITY_COLUMNS if c not in cities.columns]
    if missing:
        raise Pol4DataError(f"cities.csv is missing column(s): {missing}")
    cities = cities.loc[:, list(CITY_COLUMNS)].drop_duplicates(CITY).reset_index(drop=True)
    cities = _attach_names(cities, config, findings, strict)

    # -- city coverage, both directions ------------------------------------
    known = set(cities[CITY])
    seen = set(search[CITY]) | set(evaluation[CITY])
    orphans = sorted(seen - known)
    if orphans:
        detail = f"{len(orphans)} city_code(s) appear in the logs but not in cities.csv"
        if strict:
            raise Pol4DataError(detail)
        findings.append(
            {"severity": "high", "dataset": "cities.csv", "title": "unknown cities", "detail": detail}
        )
    silent = sorted(known - set(search[CITY]))
    if silent:
        findings.append(
            {
                "severity": "medium",
                "dataset": "cities.csv",
                "title": "cities with no search history",
                "detail": (
                    f"{len(silent)} city(ies) have no rows in search_data.csv; they still "
                    "receive a prediction, from the province and global curves."
                ),
            }
        )
    if len(cities) != N_CITIES:
        findings.append(
            {
                "severity": "medium",
                "dataset": "cities.csv",
                "title": "unexpected city count",
                "detail": f"expected {N_CITIES} cities, found {len(cities)}",
            }
        )

    # -- the split the brief promises: history ends where evaluation begins --
    if len(evaluation) and search[CHECKIN].max() >= evaluation[CHECKIN].min():
        findings.append(
            {
                "severity": "high",
                "dataset": "evaluation.csv",
                "title": "history and evaluation check-ins overlap",
                "detail": (
                    "search_data.csv contains check-ins at or after the first evaluation "
                    "check-in; historical final demand may be incomplete."
                ),
            }
        )

    return Pol4Data(search=search, evaluation=evaluation, cities=cities, findings=findings)
