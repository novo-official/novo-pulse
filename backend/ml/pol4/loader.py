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

SEARCH_COLUMNS = (LOG_DATE, CITY, CHECKIN, SEARCHES)
CITY_COLUMNS = (CITY, PROVINCE, "lat", "long")


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
