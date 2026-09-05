"""Dataset profiling + automatic schema detection for the Data Lab.

Given an unknown file we report what is in it and guess which column plays
which role. Every guess is a *suggestion*: the user confirms or overrides it in
the UI before anything is trained.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .adapter import read_tabular
from .dates import detect_calendar, normalise_digit_series, parse_timestamps
from .frequency import detect_frequency

# Role keyword tables. Matched against normalised column names.
TIMESTAMP_HINTS = (
    "date", "datetime", "timestamp", "time", "day", "ds", "created_at",
    "period", "dt", "month", "week", "checkin", "check_in", "stay_date",
    # Persian
    "تاریخ", "روز", "زمان", "تاریخ_رزرو", "تاریخ_ورود", "تاریخ_اقامت", "ماه", "هفته",
)
TARGET_HINTS = (
    "booking_count", "bookings", "booking", "demand", "demand_count", "target",
    "reservation_count", "reservations", "reserved_nights", "y", "sales",
    "quantity", "orders", "occupancy", "revenue", "search_count", "conversion_rate",
    "nights", "units",
    # Persian
    "تعداد_رزرو", "رزرو", "تقاضا", "فروش", "شب_اقامت", "شب_رزرو", "اقامت",
    "درآمد", "تعداد_شب", "ضریب_اشغال", "اشغال",
)
ENTITY_HINTS = (
    "accommodation_id", "listing_id", "hotel_id", "property_id", "unit_id",
    "entity_id", "item_id", "product_id", "series_id", "host_id", "unit_code",
    "property_code", "listing_key", "sku", "id", "code", "key", "uid", "guid",
    # Persian
    "کد_اقامتگاه", "شناسه_اقامتگاه", "اقامتگاه", "کد_ملک", "شناسه", "کد",
    "کد_واحد", "واحد", "ملک", "هتل", "کد_هتل",
)
DESTINATION_HINTS = (
    "destination_id", "destination", "city", "city_id", "region", "province",
    "location", "area", "market", "geo", "zone", "district", "country",
    "state", "site", "branch", "store", "cluster", "territory",
    # Persian
    "شهر", "مقصد", "استان", "منطقه", "ناحیه", "محل", "کشور", "کد_شهر", "شهرستان",
)
CATEGORY_HINTS = (
    "category", "accommodation_type", "property_type", "type", "segment",
    "class", "group", "room_type", "tier", "brand", "family", "kind",
    # Persian
    "دسته", "دسته_بندی", "نوع", "نوع_اقامتگاه", "گروه", "طبقه_بندی", "سطح",
)
FUTURE_HINTS = (
    "price", "is_holiday", "holiday", "is_weekend", "weekend", "capacity",
    "available", "availability", "promotion", "promo", "discount", "event",
    "season", "day_of_week", "planned",
    # Persian
    "قیمت", "نرخ", "تعرفه", "مبلغ", "تعطیل", "تعطیلات", "مناسبت", "رویداد",
    "ظرفیت", "موجودی", "تخفیف", "کمپین", "فصل", "آخر_هفته",
)
HISTORICAL_HINTS = (
    "search", "view", "click", "impression", "visit", "cancel", "review",
    "rating_count", "actual", "conversion",
    # Persian
    "جستجو", "جست_وجو", "بازدید", "مشاهده", "کلیک", "لغو", "نظر", "امتیاز",
    "نرخ_تبدیل", "بازدیدکننده",
)

# Several columns can legitimately be "the target". When more than one matches
# we prefer the demand-side KPI over a funnel metric, but the user always
# decides in the end.
TARGET_PRIORITY: dict[str, float] = {
    "booking_count": 1.00, "bookings": 0.99, "booking": 0.98, "demand": 0.98,
    "demand_count": 0.98, "target": 0.97, "y": 0.96, "reservation_count": 0.96,
    "reservations": 0.95, "sales": 0.94, "orders": 0.94, "units": 0.92,
    "reserved_nights": 0.90, "nights": 0.88, "quantity": 0.88,
    "occupancy": 0.86, "revenue": 0.84, "conversion_rate": 0.80,
    "search_count": 0.78,
    # Persian
    "تعداد_رزرو": 1.00, "رزرو": 0.99, "تقاضا": 0.98, "شب_رزرو": 0.92,
    "شب_اقامت": 0.90, "اقامت": 0.88, "فروش": 0.88, "اشغال": 0.86,
    "ضریب_اشغال": 0.86, "درآمد": 0.84, "تعداد_شب": 0.90,
}


# Arabic letter forms and the zero-width non-joiner both appear in real Persian
# headers; unifying them is what makes "جست‌وجو" and "جستجو" the same word.
_PERSIAN_NORMALISE = str.maketrans({
    "ي": "ی", "ك": "ک", "ﻰ": "ی", "ة": "ه", "ۀ": "ه",
    "\u200c": "_", "\u200f": "", "\u200e": "",
    "أ": "ا", "إ": "ا", "آ": "ا",
})


def _normalise(name: str) -> str:
    """Lowercase and tokenise a column name, preserving Persian/Arabic letters.

    Stripping to `[a-z0-9]` would erase a Persian header entirely, so every
    role lookup would fail on exactly the datasets this product targets.
    """
    text = str(name).strip().lower().translate(_PERSIAN_NORMALISE)
    # Keep ASCII alphanumerics plus the Arabic/Persian letter block.
    text = re.sub(r"[^a-z0-9\u0620-\u064a\u0670-\u06d3]+", "_", text)
    return text.strip("_")


# Below this length a substring match is meaningless: the bare hint "id"
# matches "hol-id-ay", "val-id", "gu-id-e" and "w-id-th". Short hints are only
# ever matched as whole tokens.
MIN_SUBSTRING_HINT = 5


def _score(name: str, hints: tuple[str, ...]) -> float:
    """How strongly a column name matches a role's keyword table.

    Matching is token-aware. A naive substring test is far too eager on short
    keywords - it is what once made `public_holiday` the best "entity id"
    candidate in a dataset, purely because "holiday" contains "id".
    """
    norm = _normalise(name)
    tokens = [t for t in norm.split("_") if t]
    token_set = set(tokens)
    best = 0.0

    for hint in hints:
        hint_tokens = [t for t in hint.split("_") if t]

        if norm == hint:
            return 1.0
        # Every part of a multi-word hint present as a token: "unit_code"
        # matching a column literally named "unit_code" is handled above; this
        # catches "code_unit" and "unit code id".
        if len(hint_tokens) > 1 and set(hint_tokens).issubset(token_set):
            best = max(best, 0.92)
            continue
        # Single-word hint appearing as a whole token: "code" in "unit_code".
        if len(hint_tokens) == 1 and hint in token_set:
            best = max(best, 0.9 if tokens[-1] == hint else 0.82)
            continue
        # Affix match on a longer hint: "listing_identifier" for hint "listing".
        if len(hint) >= 4 and (norm.startswith(f"{hint}_") or norm.endswith(f"_{hint}")):
            best = max(best, 0.85)
            continue
        # Loose substring, only for hints long enough to be unambiguous.
        if len(hint) >= MIN_SUBSTRING_HINT and hint in norm:
            best = max(best, 0.6)
    return best


def detect_transactional(
    frame: pd.DataFrame, timestamp: str | None, entity: str | None
) -> dict[str, Any]:
    """Is this one row per event rather than one row per period?

    A booking export has many rows sharing the same (accommodation, day) and no
    column holding a count - the demand *is* the number of rows. Loading it
    without `aggregation="count"` produces a nonsense series, so this has to be
    detected rather than left for the user to notice.
    """
    if not timestamp or timestamp not in frame.columns:
        return {"transactional": False, "reason": "no timestamp column identified"}

    keys = [timestamp] + ([entity] if entity and entity in frame.columns else [])
    sample = frame.head(100_000)
    duplicated_share = float(sample.duplicated(subset=keys).mean())
    rows_per_key = len(sample) / max(sample.drop_duplicates(subset=keys).shape[0], 1)

    return {
        "transactional": bool(duplicated_share > 0.3 and rows_per_key > 1.5),
        "duplicate_share": round(duplicated_share, 3),
        "rows_per_period": round(rows_per_key, 2),
        "keys": keys,
        "reason": (
            f"{duplicated_share:.0%} of rows repeat a ({', '.join(keys)}) pair "
            f"({rows_per_key:.1f} rows per period)"
        ),
    }


def profile_dataset(
    path: str | Path, sample_rows: int = 200_000, preview_rows: int = 20
) -> dict[str, Any]:
    """Full profile of a tabular file: dtypes, nulls, candidates, preview."""
    frame = read_tabular(path)
    n_rows = len(frame)
    if n_rows > sample_rows:
        frame_sample = frame.sample(sample_rows, random_state=42)
    else:
        frame_sample = frame

    columns = [_profile_column(frame_sample, name) for name in frame.columns]
    candidates = _candidates(columns)
    suggestion = suggest_schema(columns, candidates)

    transactional = detect_transactional(
        frame_sample, suggestion.get("timestamp"), suggestion.get("entity_id")
    )
    if transactional["transactional"]:
        # Demand is the row count. Any "target" the keyword matcher picked
        # (guests, amount, nights) would be the wrong quantity entirely.
        suggestion["aggregation"] = "count"
        suggestion["target"] = None
        suggestion["non_negative"] = True
        suggestion["integer"] = True
    else:
        suggestion.setdefault("aggregation", "sum")

    freq_info = None
    calendar_info = None
    if suggestion.get("timestamp"):
        column = frame[suggestion["timestamp"]]
        calendar_info = detect_calendar(column).as_dict()
        parsed, _ = parse_timestamps(column)
        freq_info = detect_frequency(parsed)
        suggestion["calendar"] = calendar_info["calendar"]

    preview = frame.head(preview_rows).copy()
    for column in preview.columns:
        if pd.api.types.is_datetime64_any_dtype(preview[column]):
            preview[column] = preview[column].astype(str)
    preview = preview.replace({np.nan: None})

    return {
        "file": str(Path(path).name),
        "rows": int(n_rows),
        "columns": columns,
        "n_columns": len(columns),
        "candidates": candidates,
        "suggested_schema": suggestion,
        "frequency": freq_info,
        "calendar": calendar_info,
        "transactional": transactional,
        "sample_rows": preview.to_dict(orient="records"),
        "memory_mb": round(frame.memory_usage(deep=True).sum() / 1e6, 2),
    }


def _profile_column(frame: pd.DataFrame, name: str) -> dict[str, Any]:
    series = frame[name]
    non_null = series.dropna()
    n = len(series)

    kind = "unknown"
    if pd.api.types.is_datetime64_any_dtype(series):
        kind = "datetime"
    elif pd.api.types.is_bool_dtype(series):
        kind = "boolean"
    elif pd.api.types.is_numeric_dtype(series):
        kind = "numeric"
    else:
        kind = "categorical"

    parsed_dates = None
    calendar = None
    if kind == "categorical" and len(non_null) > 0:
        sample = non_null.astype(str).head(500)
        # Calendar-aware: a Jalali column parses to NaT under plain pandas and
        # would otherwise be classified as ordinary text.
        parsed, detection = parse_timestamps(sample)
        ratio = float(parsed.notna().mean())
        if ratio > 0.9:
            kind = "datetime_like"
            parsed_dates = ratio
            calendar = detection.calendar

    info: dict[str, Any] = {
        "name": str(name),
        "dtype": str(series.dtype),
        "kind": kind,
        "null_count": int(series.isna().sum()),
        "null_pct": round(float(series.isna().mean() * 100), 2) if n else 0.0,
        "unique_count": int(non_null.nunique()),
        "unique_pct": round(float(non_null.nunique() / n * 100), 2) if n else 0.0,
        "constant": bool(non_null.nunique() <= 1),
        "sample_values": [
            None if pd.isna(v) else (str(v) if not isinstance(v, (int, float)) else v)
            for v in non_null.head(5).tolist()
        ],
    }
    if parsed_dates is not None:
        info["date_parse_ratio"] = round(parsed_dates, 3)
    if calendar:
        info["calendar"] = calendar

    if kind == "numeric" and len(non_null):
        info.update(
            {
                "min": float(non_null.min()),
                "max": float(non_null.max()),
                "mean": round(float(non_null.mean()), 4),
                "std": round(float(non_null.std()), 4) if len(non_null) > 1 else 0.0,
                "zero_pct": round(float((non_null == 0).mean() * 100), 2),
                "negative_pct": round(float((non_null < 0).mean() * 100), 2),
                "integer_like": bool(np.allclose(non_null % 1, 0, atol=1e-9)),
            }
        )
    if kind in {"datetime", "datetime_like"} and len(non_null):
        parsed, detection = parse_timestamps(non_null)
        parsed = parsed.dropna()
        if len(parsed):
            info["min"] = str(parsed.min().date())
            info["max"] = str(parsed.max().date())
        info.setdefault("calendar", detection.calendar)
    return info


def _candidates(columns: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Rank every column for every role."""
    roles = {
        "timestamp": TIMESTAMP_HINTS,
        "target": TARGET_HINTS,
        "entity_id": ENTITY_HINTS,
        "destination": DESTINATION_HINTS,
        "category": CATEGORY_HINTS,
        "future": FUTURE_HINTS,
        "historical": HISTORICAL_HINTS,
    }
    out: dict[str, list[dict[str, Any]]] = {}
    for role, hints in roles.items():
        scored = []
        for column in columns:
            score = _score(column["name"], hints)
            score = _adjust_for_type(role, column, score)
            if score > 0:
                scored.append({"column": column["name"], "score": round(score, 3)})
        scored.sort(key=lambda item: -item["score"])
        out[role] = scored[:10]

    # Structural fallbacks: a datetime column is a timestamp candidate even if
    # it is called something unexpected.
    date_like = [c["name"] for c in columns if c["kind"] in {"datetime", "datetime_like"}]
    known = {c["column"] for c in out["timestamp"]}
    for name in date_like:
        if name not in known:
            out["timestamp"].append({"column": name, "score": 0.7})
    out["timestamp"].sort(key=lambda item: -item["score"])

    out["numeric"] = [
        {"column": c["name"], "score": 1.0} for c in columns if c["kind"] == "numeric"
    ]
    out["categorical"] = [
        {"column": c["name"], "score": 1.0}
        for c in columns
        if c["kind"] in {"categorical", "boolean"}
    ]
    return out


def _adjust_for_type(role: str, column: dict[str, Any], score: float) -> float:
    """A keyword match still has to be type-plausible."""
    kind = column["kind"]
    if role == "timestamp":
        return score * (1.0 if kind in {"datetime", "datetime_like"} else 0.25)
    if role == "target":
        if kind != "numeric":
            return 0.0
        if column.get("constant"):
            return 0.0
        return score * TARGET_PRIORITY.get(_normalise(column["name"]), 0.85)
    if role in {"entity_id", "destination", "category"}:
        if column.get("constant"):
            return score * 0.2
        # An identifier that is unique on every row cannot group a time series.
        if column.get("unique_pct", 0) > 95 and role == "entity_id":
            return score * 0.3
        return score
    if role in {"future", "historical"}:
        return score if kind in {"numeric", "boolean"} else score * 0.5
    return score


def suggest_schema(
    columns: list[dict[str, Any]], candidates: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    """Best-guess contract for the Data Lab form."""

    def top(role: str, minimum: float = 0.35) -> str | None:
        items = candidates.get(role) or []
        if items and items[0]["score"] >= minimum:
            return items[0]["column"]
        return None

    timestamp = top("timestamp", 0.2)
    target = top("target", 0.4)
    entity = top("entity_id", 0.5)
    destination = top("destination", 0.5)
    category = top("category", 0.5)

    taken = {timestamp, target, entity, destination, category} - {None}
    future = [
        c["column"]
        for c in (candidates.get("future") or [])
        if c["score"] >= 0.55 and c["column"] not in taken
    ]
    historical = [
        c["column"]
        for c in (candidates.get("historical") or [])
        if c["score"] >= 0.55 and c["column"] not in taken and c["column"] not in future
    ]
    by_name = {c["name"]: c for c in columns}
    target_info = by_name.get(target or "", {})

    return {
        "timestamp": timestamp,
        "target": target,
        "entity_id": entity,
        "destination": destination,
        "category": category,
        "future_features": future,
        "historical_features": historical,
        "static_features": [],
        "non_negative": bool(target_info.get("negative_pct", 0) == 0),
        "integer": bool(target_info.get("integer_like", False)),
        "calendar": "auto",
    }
