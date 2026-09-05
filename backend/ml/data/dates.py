"""Timestamp parsing for Iranian datasets.

An Iranian marketplace exports dates in whatever its internal tooling used, and
that is very often the Jalali (Shamsi) calendar - `1403/05/12` - sometimes
written with Persian digits (`۱۴۰۳/۰۵/۱۲`). `pandas.to_datetime` returns NaT for
every one of those, which would silently drop the entire dataset at the first
step of the pipeline.

This module detects the calendar, normalises the digits, and converts to
standard Gregorian timestamps. Everything downstream continues to work in
Gregorian only - the calendar is an input format, never a modelling concern.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

# Digit maps: Persian (۰-۹) and Arabic-Indic (٠-٩) to ASCII.
_PERSIAN_DIGITS = "۰۱۲۳۴۵۶۷۸۹"
_ARABIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"
_DIGIT_TABLE = str.maketrans(
    {ord(c): str(i) for i, c in enumerate(_PERSIAN_DIGITS)}
    | {ord(c): str(i) for i, c in enumerate(_ARABIC_DIGITS)}
    # Arabic decimal/thousands separators occasionally appear inside numbers.
    | {ord("٫"): ".", ord("٬"): ",", ord("،"): ","}
)

# A Jalali year in any dataset worth forecasting sits in this range; 1300 is
# ~1921 and 1500 is ~2121. Gregorian years never land here, which is what makes
# the calendar detectable from the value alone.
JALALI_YEAR_MIN = 1300
JALALI_YEAR_MAX = 1500

_DATE_PATTERN = re.compile(r"^\s*(\d{4})\s*[-/._]\s*(\d{1,2})\s*[-/._]\s*(\d{1,2})")


def normalise_digits(value: Any) -> Any:
    """Convert Persian/Arabic-Indic digits to ASCII, leaving other text alone."""
    if isinstance(value, str):
        return value.translate(_DIGIT_TABLE)
    return value


def normalise_digit_series(series: pd.Series) -> pd.Series:
    if series.dtype == object or isinstance(series.dtype, pd.StringDtype):
        return series.map(normalise_digits)
    return series


@dataclass
class CalendarDetection:
    calendar: str          # "gregorian" | "jalali" | "unknown"
    confidence: float
    parsed_ratio: float
    sample: list[str]
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "calendar": self.calendar,
            "confidence": round(self.confidence, 3),
            "parsed_ratio": round(self.parsed_ratio, 3),
            "sample": self.sample[:3],
            "reason": self.reason,
        }


def jalali_available() -> bool:
    try:
        import jdatetime  # noqa: F401
    except ImportError:
        return False
    return True


def detect_calendar(series: pd.Series, sample_size: int = 500) -> CalendarDetection:
    """Decide whether a column holds Jalali or Gregorian dates.

    The year field decides it: a leading 13xx/14xx is Jalali, because no
    Gregorian dataset in this industry contains year 1403.
    """
    values = normalise_digit_series(pd.Series(series).dropna().astype(str))
    if values.empty:
        return CalendarDetection("unknown", 0.0, 0.0, [], "column is empty")

    sample = values.head(sample_size)
    jalali_hits = 0
    matched = 0
    for text in sample:
        match = _DATE_PATTERN.match(text)
        if not match:
            continue
        matched += 1
        year = int(match.group(1))
        if JALALI_YEAR_MIN <= year <= JALALI_YEAR_MAX:
            jalali_hits += 1

    if matched == 0:
        gregorian = pd.to_datetime(sample, errors="coerce", format="mixed")
        ratio = float(gregorian.notna().mean())
        return CalendarDetection(
            "gregorian" if ratio > 0.9 else "unknown",
            ratio,
            ratio,
            sample.head(3).tolist(),
            "no yyyy-mm-dd pattern; fell back to pandas parsing",
        )

    share = jalali_hits / matched
    if share > 0.9:
        return CalendarDetection(
            "jalali", share, matched / len(sample), sample.head(3).tolist(),
            f"year field is in the Jalali range ({JALALI_YEAR_MIN}-{JALALI_YEAR_MAX})",
        )
    if share < 0.1:
        return CalendarDetection(
            "gregorian", 1.0 - share, matched / len(sample), sample.head(3).tolist(),
            "year field is in the Gregorian range",
        )
    return CalendarDetection(
        "unknown", 0.5, matched / len(sample), sample.head(3).tolist(),
        f"mixed year ranges ({share:.0%} look Jalali) - confirm the calendar manually",
    )


def jalali_to_gregorian(series: pd.Series) -> pd.Series:
    """Convert a Jalali date column to Gregorian timestamps.

    Unparseable entries become NaT rather than raising, so one malformed row
    cannot take down an otherwise usable dataset.
    """
    values = normalise_digit_series(pd.Series(series).astype(str))

    try:
        import jdatetime
    except ImportError:  # pragma: no cover - exercised only without the extra
        return _jalali_to_gregorian_fallback(values)

    out: list[pd.Timestamp | Any] = []
    for text in values:
        match = _DATE_PATTERN.match(text or "")
        if not match:
            out.append(pd.NaT)
            continue
        year, month, day = (int(g) for g in match.groups())
        try:
            out.append(pd.Timestamp(jdatetime.date(year, month, day).togregorian()))
        except (ValueError, TypeError):
            out.append(pd.NaT)
    return pd.Series(out, index=values.index)


def _jalali_to_gregorian_fallback(values: pd.Series) -> pd.Series:
    """Pure-Python Jalali conversion, used when `jdatetime` is not installed.

    Implements the standard Birashk-style algorithm so the feature degrades to
    "slower" rather than "unavailable".
    """
    import datetime as dt

    def convert(text: str):
        match = _DATE_PATTERN.match(text or "")
        if not match:
            return pd.NaT
        jy, jm, jd = (int(g) for g in match.groups())
        if not (1 <= jm <= 12 and 1 <= jd <= 31):
            return pd.NaT
        jy += 1595
        days = -355668 + (365 * jy) + ((jy // 33) * 8) + (((jy % 33) + 3) // 4) + jd
        days += (jm - 1) * 31 if jm < 7 else ((jm - 7) * 30) + 186
        gy = 400 * (days // 146097)
        days %= 146097
        if days > 36524:
            days -= 1
            gy += 100 * (days // 36524)
            days %= 36524
            if days >= 365:
                days += 1
        gy += 4 * (days // 1461)
        days %= 1461
        if days > 365:
            gy += (days - 1) // 365
            days = (days - 1) % 365
        gd = days + 1
        leap = (gy % 4 == 0 and gy % 100 != 0) or (gy % 400 == 0)
        months = [0, 31, 29 if leap else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        gm = 0
        for index in range(1, 13):
            if gd <= months[index]:
                gm = index
                break
            gd -= months[index]
        try:
            return pd.Timestamp(dt.date(gy, gm, gd))
        except ValueError:
            return pd.NaT

    return pd.Series([convert(v) for v in values], index=values.index)


def parse_timestamps(
    series: pd.Series, calendar: str | None = None
) -> tuple[pd.Series, CalendarDetection]:
    """Parse a timestamp column in whatever calendar it happens to use.

    `calendar` forces the interpretation ("jalali" / "gregorian"); leaving it
    None auto-detects. Returns the Gregorian series plus what was detected, so
    the caller can report it rather than convert silently.
    """
    # Already parsed - by an upstream join, or by a Parquet/Excel reader that
    # types dates natively. Re-reading those digits as Jalali would shift every
    # date by ~621 years.
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.Series(series), CalendarDetection(
            "gregorian",
            1.0,
            float(pd.Series(series).notna().mean()),
            [str(v) for v in pd.Series(series).dropna().head(3)],
            "column is already a datetime, no conversion needed",
        )

    detection = detect_calendar(series)
    chosen = (calendar or detection.calendar).lower()

    if chosen == "jalali":
        parsed = jalali_to_gregorian(series)
    else:
        values = normalise_digit_series(pd.Series(series))
        parsed = pd.to_datetime(values, errors="coerce", format="mixed")
        # A Gregorian read that fails wholesale on a column that *looks* Jalali
        # is worth retrying rather than reporting an empty dataset.
        if parsed.notna().mean() < 0.5 and detection.calendar != "gregorian":
            retry = jalali_to_gregorian(series)
            if retry.notna().mean() > parsed.notna().mean():
                parsed = retry
                detection = CalendarDetection(
                    "jalali", 0.75, float(retry.notna().mean()), detection.sample,
                    "Gregorian parsing failed; Jalali conversion succeeded",
                )
    return parsed, detection


def to_jalali_string(value: Any) -> str:
    """Format a Gregorian date as Jalali, for display only."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    try:
        import jdatetime

        stamp = pd.Timestamp(value)
        jalali = jdatetime.date.fromgregorian(date=stamp.date())
        return f"{jalali.year:04d}-{jalali.month:02d}-{jalali.day:02d}"
    except Exception:  # noqa: BLE001 - display helper must never raise
        return str(value)
