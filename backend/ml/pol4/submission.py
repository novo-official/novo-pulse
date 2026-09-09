"""The competition output contract: build it, then refuse to ship it if it is wrong.

`results.csv` must carry `cluster_code, checkin, predicted_demand` for every
city and every Azar 1404 check-in - 321 x 30 = 9,630 rows. Phase 1 uses no
clustering, so each city is its own cluster and `cluster_code = city_code`.

A malformed file scores zero regardless of how good the model is, so the
validator is deliberately loud, reports *every* problem it finds rather than the
first, and runs as part of the pipeline rather than as an afterthought.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import SUBMISSION_COLUMNS, Pol4Config
from .loader import CHECKIN, CITY, Pol4Data

DATE_FORMAT = "%Y-%m-%d"


class SubmissionError(ValueError):
    """Raised when a submission frame violates the competition contract."""


@dataclass
class SubmissionReport:
    rows: int
    cities: int
    dates: int
    checkin_min: str
    checkin_max: str
    total_demand: float
    zero_rows: int
    problems: list[str]

    @property
    def valid(self) -> bool:
        return not self.problems

    def as_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "rows": self.rows,
            "cities": self.cities,
            "dates": self.dates,
            "checkin_min": self.checkin_min,
            "checkin_max": self.checkin_max,
            "total_predicted_demand": round(self.total_demand, 2),
            "rows_predicting_zero": self.zero_rows,
            "problems": self.problems,
        }


# ------------------------------------------------------------------- build
def build_grid(
    data: Pol4Data,
    cutoff: pd.Timestamp,
    target_dates: pd.DatetimeIndex,
    city_codes: np.ndarray | None = None,
) -> pd.DataFrame:
    """The complete (city x check-in) grid with demand observed at `cutoff`.

    Every city appears for every date, including the ones with nothing observed
    yet - a missing row in the source data means "no searches so far", never
    "final demand is zero".
    """
    cutoff = pd.Timestamp(cutoff)
    cities = data.city_codes if city_codes is None else np.asarray(city_codes)
    grid = pd.MultiIndex.from_product(
        [np.sort(cities), pd.DatetimeIndex(target_dates)], names=[CITY, CHECKIN]
    ).to_frame(index=False)

    observed = data.observed_at(cutoff, target_dates.min(), target_dates.max())
    grid = grid.merge(observed, on=[CITY, CHECKIN], how="left")
    grid["observed"] = grid["observed"].fillna(0.0).astype(np.float64)
    return grid


def build_submission(predictions: pd.DataFrame, integer: bool = True) -> pd.DataFrame:
    """Reduce a prediction frame to the three submission columns.

    Demand is a count of searches, so predictions are rounded to integers by
    default. Rounding happens exactly once, here, so the artefact and the
    validated file are always the same numbers.
    """
    for column in (CITY, CHECKIN, "predicted_demand"):
        if column not in predictions.columns:
            raise SubmissionError(f"prediction frame is missing '{column}'")

    out = pd.DataFrame(
        {
            "cluster_code": predictions[CITY].to_numpy(),
            "checkin": pd.to_datetime(predictions[CHECKIN]).dt.strftime(DATE_FORMAT),
            "predicted_demand": predictions["predicted_demand"].to_numpy(dtype=np.float64),
        }
    )
    out["predicted_demand"] = np.maximum(out["predicted_demand"], 0.0)
    if integer:
        out["predicted_demand"] = np.rint(out["predicted_demand"]).astype(np.int64)
    return out.sort_values(["cluster_code", "checkin"], ignore_index=True)


def write_submission(frame: pd.DataFrame, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------- validate
def read_submission(path: Path) -> pd.DataFrame:
    return pd.read_csv(Path(path), dtype={"cluster_code": "int64"})


def validate_submission(
    frame: pd.DataFrame | Path | str,
    expected_cities: np.ndarray,
    config: Pol4Config | None = None,
    raise_on_error: bool = True,
) -> SubmissionReport:
    """Check every rule the competition imposes. Collects all failures."""
    config = config or Pol4Config()
    if isinstance(frame, (str, Path)):
        frame = read_submission(frame)

    problems: list[str] = []
    expected_dates = config.target_dates()
    expected_cities = np.sort(np.asarray(expected_cities))
    expected_rows = len(expected_cities) * len(expected_dates)

    # -- columns -----------------------------------------------------------
    if tuple(frame.columns) != SUBMISSION_COLUMNS:
        problems.append(
            f"columns are {tuple(frame.columns)}, expected {SUBMISSION_COLUMNS}"
        )
        # Without the right columns nothing else can be checked meaningfully.
        report = SubmissionReport(len(frame), 0, 0, "", "", 0.0, 0, problems)
        if raise_on_error:
            raise SubmissionError("; ".join(problems))
        return report

    # -- dates: format first, then range -----------------------------------
    raw_dates = frame["checkin"].astype(str)
    parsed = pd.to_datetime(raw_dates, format=DATE_FORMAT, errors="coerce")
    bad_format = int(parsed.isna().sum())
    if bad_format:
        sample = raw_dates[parsed.isna()].head(3).tolist()
        problems.append(
            f"{bad_format} checkin value(s) are not Gregorian {DATE_FORMAT} dates, e.g. {sample}"
        )

    # -- row count ---------------------------------------------------------
    if len(frame) != expected_rows:
        problems.append(f"{len(frame)} rows, expected {expected_rows}")

    # -- duplicates --------------------------------------------------------
    duplicates = int(frame.duplicated(["cluster_code", "checkin"]).sum())
    if duplicates:
        problems.append(f"{duplicates} duplicate (cluster_code, checkin) row(s)")

    # -- coverage ----------------------------------------------------------
    present_cities = set(frame["cluster_code"].astype("int64"))
    missing_cities = sorted(set(expected_cities.tolist()) - present_cities)
    extra_cities = sorted(present_cities - set(expected_cities.tolist()))
    if missing_cities:
        problems.append(
            f"{len(missing_cities)} city(ies) missing from the output, e.g. {missing_cities[:5]}"
        )
    if extra_cities:
        problems.append(
            f"{len(extra_cities)} unexpected cluster_code(s), e.g. {extra_cities[:5]}"
        )

    present_dates = set(parsed.dropna())
    missing_dates = sorted(set(expected_dates) - present_dates)
    extra_dates = sorted(present_dates - set(expected_dates))
    if missing_dates:
        problems.append(
            f"{len(missing_dates)} target date(s) missing, e.g. "
            f"{[d.date().isoformat() for d in missing_dates[:5]]}"
        )
    if extra_dates:
        problems.append(
            f"{len(extra_dates)} date(s) outside "
            f"{config.target_start.date()}..{config.target_end.date()}, e.g. "
            f"{[d.date().isoformat() for d in extra_dates[:5]]}"
        )

    if not (missing_cities or missing_dates or extra_cities or extra_dates or duplicates):
        expected_pairs = len(expected_cities) * len(expected_dates)
        if len(frame.drop_duplicates(["cluster_code", "checkin"])) != expected_pairs:
            problems.append("the output does not cover every (city, check-in) pair exactly once")

    # -- values ------------------------------------------------------------
    values = pd.to_numeric(frame["predicted_demand"], errors="coerce")
    non_numeric = int(values.isna().sum() - frame["predicted_demand"].isna().sum())
    if non_numeric > 0:
        problems.append(f"{non_numeric} non-numeric predicted_demand value(s)")
    nans = int(frame["predicted_demand"].isna().sum())
    if nans:
        problems.append(f"{nans} NaN predicted_demand value(s)")
    infinite = int(np.isinf(values.fillna(0.0).to_numpy()).sum())
    if infinite:
        problems.append(f"{infinite} infinite predicted_demand value(s)")
    negative = int((values.fillna(0.0) < 0).sum())
    if negative:
        problems.append(f"{negative} negative predicted_demand value(s)")

    finite = values.replace([np.inf, -np.inf], np.nan).dropna()
    report = SubmissionReport(
        rows=len(frame),
        cities=len(present_cities),
        dates=len(present_dates),
        checkin_min=str(parsed.min().date()) if parsed.notna().any() else "",
        checkin_max=str(parsed.max().date()) if parsed.notna().any() else "",
        total_demand=float(finite.sum()),
        zero_rows=int((finite <= 0).sum()),
        problems=problems,
    )
    if problems and raise_on_error:
        raise SubmissionError("results.csv is invalid: " + "; ".join(problems))
    return report
