"""Iranian holiday / event calendar used by the synthetic generator.

Dates are stored as Gregorian because the ML pipeline only ever works with
standard dates. The UI is free to render them in the Jalali calendar - the
model never sees a display format.

The table is deliberately approximate: it exists so that a *discoverable*
holiday effect is present in the synthetic data and so the holiday feature
plumbing is exercised end to end. On competition day a real `holidays.csv`
can be imported instead (see `load_holiday_table`).
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

# Nowruz starts on ~20/21 March every Gregorian year.
NOWRUZ_MONTH_DAY = (3, 20)

# (month, day, name, span_days) - recurring, approximately fixed Gregorian dates.
RECURRING: list[tuple[int, int, str, int]] = [
    (3, 20, "نوروز", 13),
    (4, 1, "سیزده‌بدر", 1),
    (6, 4, "رحلت امام خمینی", 2),
    (2, 1, "عید فطر", 2),
    (7, 16, "عید قربان", 1),
    (9, 26, "شب یلدا", 1),
    (2, 11, "روز کارگر", 1),
    (10, 5, "تعطیلات زمستانی", 3),
    (8, 22, "اربعین", 2),
    (11, 11, "دهه فجر", 3),
    (12, 15, "تعطیلات پایان سال میلادی", 4),
]

# Destination-flavoured events: (month, day, name, span, destinations)
EVENTS: list[tuple[int, int, str, int, tuple[str, ...]]] = [
    (1, 15, "جشنواره زمستانی کیش", 6, ("kish", "qeshm")),
    (2, 20, "نمایشگاه گردشگری تهران", 4, ("tehran",)),
    (4, 12, "جشنواره بهاره شمال", 4, ("ramsar", "rasht")),
    (5, 10, "جشنواره گل و گیاه", 4, ("isfahan", "shiraz")),
    (6, 18, "کنسرت تابستانی ساحلی", 3, ("bandar_abbas", "chabahar")),
    (7, 20, "فستیوال ساحلی شمال", 5, ("ramsar", "rasht")),
    (8, 14, "جشنواره کویر", 4, ("yazd", "kerman")),
    (9, 10, "کنفرانس گردشگری", 3, ("tehran", "mashhad")),
    (9, 5, "جشنواره شب‌های اصفهان", 5, ("isfahan", "shiraz")),
    (10, 8, "همایش زیارتی", 4, ("mashhad",)),
    (11, 25, "نمایشگاه صنایع دستی", 4, ("yazd", "tabriz")),
    (12, 10, "فستیوال زمستانی جزایر", 5, ("kish", "qeshm")),
]


def build_calendar(
    start: dt.date, end: dt.date, destinations: list[str] | None = None
) -> pd.DataFrame:
    """Build a daily calendar with weekend / holiday / season / event flags."""
    dates = pd.date_range(start, end, freq="D")
    frame = pd.DataFrame({"date": dates})

    # Iran's weekend is Thursday(3)/Friday(4) in pandas' Mon=0 numbering.
    dow = frame["date"].dt.dayofweek
    frame["day_of_week"] = dow
    frame["is_weekend"] = dow.isin([3, 4]).astype(int)

    holiday_name: dict[pd.Timestamp, str] = {}
    for month, day, name, span in RECURRING:
        for year in range(start.year - 1, end.year + 2):
            try:
                anchor = pd.Timestamp(year=year, month=month, day=day)
            except ValueError:  # e.g. Feb 30 in a shifted year
                continue
            for offset in range(span):
                holiday_name.setdefault(anchor + pd.Timedelta(days=offset), name)

    frame["holiday_name"] = frame["date"].map(holiday_name).fillna("")
    frame["is_holiday"] = (frame["holiday_name"] != "").astype(int)

    event_name: dict[pd.Timestamp, str] = {}
    event_dests: dict[pd.Timestamp, str] = {}
    for month, day, name, span, dests in EVENTS:
        for year in range(start.year - 1, end.year + 2):
            try:
                anchor = pd.Timestamp(year=year, month=month, day=day)
            except ValueError:
                continue
            for offset in range(span):
                stamp = anchor + pd.Timedelta(days=offset)
                event_name.setdefault(stamp, name)
                event_dests.setdefault(stamp, "|".join(dests))

    frame["event_name"] = frame["date"].map(event_name).fillna("")
    frame["event_destinations"] = frame["date"].map(event_dests).fillna("")
    frame["season"] = frame["date"].dt.month.map(_season)

    frame["days_to_holiday"] = _distance_to_flag(frame["is_holiday"].to_numpy(), forward=True)
    frame["days_from_holiday"] = _distance_to_flag(frame["is_holiday"].to_numpy(), forward=False)
    if destinations is not None:
        frame.attrs["destinations"] = destinations
    return frame


def _season(month: int) -> str:
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    if month in (9, 10, 11):
        return "autumn"
    return "winter"


def _distance_to_flag(flags, forward: bool = True, cap: int = 60):
    """Days until (or since) the nearest day where `flags` is 1."""
    import numpy as np

    n = len(flags)
    out = np.full(n, cap, dtype=float)
    indices = range(n - 1, -1, -1) if forward else range(n)
    running = cap
    for i in indices:
        running = 0 if flags[i] else min(running + 1, cap)
        out[i] = running
    return out


def load_holiday_table(path: str | Path) -> pd.DataFrame:
    """Import an external holidays.csv / events.csv.

    Expected columns: `date` plus at least one of `holiday_name` / `event_name`.
    Anything else is passed through untouched.
    """
    frame = pd.read_csv(path)
    date_col = next(
        (c for c in frame.columns if c.lower() in {"date", "day", "ds", "timestamp"}), None
    )
    if date_col is None:
        raise ValueError(f"{path}: no date-like column found")
    frame = frame.rename(columns={date_col: "date"})
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"])
    if "holiday_name" in frame.columns:
        frame["is_holiday"] = (frame["holiday_name"].fillna("") != "").astype(int)
    return frame
